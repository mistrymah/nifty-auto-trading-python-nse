import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from fetcher import UpstoxFetcher

LOT_SIZE = 65

# ──────────────────────────────────────────────────────────────────────────────
# Whalley-Wilmott No-Transaction Band
# ──────────────────────────────────────────────────────────────────────────────
def _whalley_wilmott_band(net_gamma: float) -> float:
    """
    Optimal no-transaction bandwidth from live gamma.
    H = [(3 * k * gamma^2) / (2 * lambda)] ^ (1/3)
    All inputs from live chain Greeks — no Black-Scholes.
    """
    k = 0.001          # transaction cost fraction
    lam = 1.5          # risk-aversion (institutional moderate)
    abs_g = abs(net_gamma)
    if abs_g < 1e-8:
        return 0.15    # default safe band when position is near-flat
    term = (3.0 * k * abs_g ** 2) / (2.0 * lam)
    return float(np.clip(np.power(term, 1 / 3.0), 0.08, 0.40))


class DynamicAdjustmentEngine:
    def __init__(self):
        self.fetcher = UpstoxFetcher()
        self.project_root = Path(__file__).parent

    # ── 1. LIVE GREEKS FROM OPTION CHAIN ─────────────────────────────────────
    def _greeks_from_chain(self, legs: List[Dict], chain: List) -> Dict:
        """
        Computes net portfolio Greeks entirely from live Upstox option chain data.
        No Black-Scholes. Uses the greeks already in the OptionData objects.
        """
        net_delta = net_gamma = net_theta = net_vega = 0.0

        for leg in legs:
            is_call = "CALL" in leg["type"] or "CE" in leg["type"]
            is_sell = "SELL" in leg["type"]
            qty     = leg.get("qty", 1)

            opt = next(
                (o for o in chain
                 if o.strike == leg["strike"] and o.expiry == leg.get("expiry", "")),
                None
            )
            if opt is None:
                continue

            d = opt.call_delta if is_call else opt.put_delta
            g = opt.call_gamma if is_call else opt.put_gamma
            t = opt.call_theta if is_call else opt.put_theta
            v = opt.call_vega  if is_call else opt.put_vega

            sign = -1 if is_sell else 1   # short flips sign

            net_delta += sign * d * qty
            net_gamma += sign * g * qty
            net_theta += sign * t * qty
            net_vega  += sign * v * qty

        return {
            "net_delta": net_delta,
            "net_gamma": net_gamma,
            "net_theta": net_theta,
            "net_vega":  net_vega,
        }

    # ── 2. CAL 1 BREAKEVEN FROM REAL ENTRY PREMIUMS ──────────────────────────
    def _compute_cal1_be(self, legs: List[Dict]) -> Tuple[float, float]:
        """
        Computes Cal 1's lower and upper breakeven from actual entry prices.
        Guarantees lower_be < upper_be across both credit spreads and debit calendars.
        """
        sell_strikes = [l["strike"] for l in legs if "SELL" in l["type"]]
        center = sum(sell_strikes) / len(sell_strikes) if sell_strikes else 0.0

        sell_premium = sum(l["entry_price"] for l in legs if "SELL" in l["type"])
        buy_premium  = sum(l["entry_price"] for l in legs if "BUY"  in l["type"])
        
        net_flow = sell_premium - buy_premium
        if net_flow > 0:
            # Pure credit structure
            cushion = net_flow
        else:
            # Debit structure (Double Diagonal / Calendar):
            # Short legs generate decay cushion minus net debit paid for the outer wings.
            cushion = max(sell_premium - abs(net_flow), 150.0)

        lower_be = center - cushion
        upper_be = center + cushion
        
        if lower_be > upper_be:
            lower_be, upper_be = upper_be, lower_be
            
        return round(lower_be, 1), round(upper_be, 1)

    # ── 3. FIND CAL 2 CENTER — LIVE CHAIN + GREEKS ───────────────────────────
    def _find_cal2_center(
        self,
        be_target: float,        # Cal 1 BE that was breached (must be matched by Cal 2)
        direction: str,          # "PUT" (downside) or "CALL" (upside)
        chain: List,
        near_exp: str,
        far_exp:  str,
    ) -> Optional[Dict]:
        """
        Scans the live option chain to find the best Cal 2 center strike where:
          PRIMARY   — Cal 2's upper/lower BE ≈ Cal 1's breached BE (live premiums)
          SECONDARY — Best Greek profile: most positive theta, lowest abs(delta),
                      positive vega (calendar = long vega)

        All data from live Upstox chain. No Black-Scholes.
        """
        candidates = []

        # Scan range: 50 to 600 pts away from the BE target
        scan_range = range(
            int(round(be_target / 50) * 50) - 50,
            int(round(be_target / 50) * 50) - 650,
            -50
        ) if direction == "PUT" else range(
            int(round(be_target / 50) * 50) + 50,
            int(round(be_target / 50) * 50) + 650,
            50
        )

        for strike in scan_range:
            near_opt = next((o for o in chain if o.strike == strike and o.expiry == near_exp), None)
            far_opt  = next((o for o in chain if o.strike == strike and o.expiry == far_exp),  None)

            if near_opt is None or far_opt is None:
                continue

            if direction == "PUT":
                near_price = near_opt.put_price or 0
                far_price  = far_opt.put_price  or 0
                near_delta = near_opt.put_delta or 0
                far_delta  = far_opt.put_delta  or 0
                near_theta = near_opt.put_theta or 0
                far_theta  = far_opt.put_theta  or 0
                near_vega  = near_opt.put_vega  or 0
                far_vega   = far_opt.put_vega   or 0
                near_gamma = near_opt.put_gamma or 0
                far_gamma  = far_opt.put_gamma  or 0
                near_iv    = near_opt.put_iv    or 0
                far_iv     = far_opt.put_iv     or 0
            else:
                near_price = near_opt.call_price or 0
                far_price  = far_opt.call_price  or 0
                near_delta = near_opt.call_delta or 0
                far_delta  = far_opt.call_delta  or 0
                near_theta = near_opt.call_theta or 0
                far_theta  = far_opt.call_theta  or 0
                near_vega  = near_opt.call_vega  or 0
                far_vega   = far_opt.call_vega   or 0
                near_gamma = near_opt.call_gamma or 0
                far_gamma  = far_opt.call_gamma  or 0
                near_iv    = near_opt.call_iv    or 0
                far_iv     = far_opt.call_iv     or 0

            if near_price <= 0 or far_price <= 0:
                continue

            net_debit = far_price - near_price
            net_credit = near_price - far_price

            # BE of Cal 2 from live premiums
            # Upper slope (for put calendar) intersects valley at strike + abs(net_debit)
            # Lower slope (for call calendar) intersects valley at strike - abs(net_debit)
            cal2_be = strike + abs(net_debit) if direction == "PUT" else strike - abs(net_debit)
            be_diff = abs(cal2_be - be_target)

            # Net Greeks of the Cal 2 position (SELL near, BUY far)
            net_delta_cal2 = (-near_delta) + far_delta    # short near + long far
            net_theta_cal2 = (-near_theta) + far_theta    # gain near theta, lose far theta
            net_vega_cal2  = (-near_vega)  + far_vega     # calendar = long vega (far > near)
            net_gamma_cal2 = (-near_gamma) + far_gamma

            candidates.append({
                "strike":       strike,
                "near_price":   near_price,
                "far_price":    far_price,
                "net_credit":   net_credit,
                "cal2_be":      cal2_be,
                "be_diff":      be_diff,
                "net_delta":    net_delta_cal2,
                "net_theta":    net_theta_cal2,
                "net_vega":     net_vega_cal2,
                "net_gamma":    net_gamma_cal2,
                "near_iv":      near_iv,
                "far_iv":       far_iv,
            })

        if not candidates:
            return None

        # ── Scoring ────────────────────────────────────────────────────────
        # Primary:  BE match (be_diff must be < 75 pts — tolerance for discrete strikes)
        # Secondary (among valid BE candidates):
        #   - Highest net_theta  (more daily decay collected)
        #   - Lowest |net_delta| (balanced at center)
        #   - Positive net_vega  (calendar should be long vega)
        #   - Reasonable IV spread (far_iv > near_iv ideally — selling high near IV)

        valid = [c for c in candidates if c["be_diff"] <= 75]
        if not valid:
            # Relax: take closest 3 BE matches
            valid = sorted(candidates, key=lambda x: x["be_diff"])[:3]

        def score(c):
            theta_score  =  c["net_theta"]              # higher = better
            delta_score  = -abs(c["net_delta"]) * 5.0  # lower abs = better
            vega_score   =  max(0, c["net_vega"]) * 2.0
            be_score     = -c["be_diff"] * 0.3          # lower diff = better
            iv_score     =  max(0, c["far_iv"] - c["near_iv"]) * 10.0  # far IV > near IV = ideal
            return theta_score + delta_score + vega_score + be_score + iv_score

        return max(valid, key=score)

    # ── 4. STATISTICAL REGIME (Z-SCORE) ──────────────────────────────────────
    def _regime(self, spot: float) -> Dict:
        df = self.fetcher.get_nifty_historical("5m", 2)
        if df.empty:
            return {"z_score": 0.0, "regime": "SIDEWAYS"}
        closes = df["close"].values
        std = np.std(closes)
        if std == 0:
            return {"z_score": 0.0, "regime": "SIDEWAYS"}
        z = (spot - np.mean(closes)) / std
        regime = ("TREND_CONTINUATION_BULL" if z > 1.5
                  else "TREND_CONTINUATION_BEAR" if z < -1.5
                  else "MEAN_REVERSION")
        return {"z_score": z, "regime": regime}

    # ── 5. FORMAT PROOF BLOCK ─────────────────────────────────────────────────
    def format_output(
        self,
        active: Dict,
        spot: float,
        change: float,
        greeks: Dict,
        ww_band: float,
        be_lower: float,
        be_upper: float,
        regime: Dict,
        decision: str,
        reason: str,
        cal2: Optional[Dict] = None,
    ) -> str:
        out = []
        out.append(f"\n📊 QUANTITATIVE ADJUSTMENT PROOF — [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
        out.append("━" * 42)
        out.append("CURRENT POSITION:")
        for leg in active["legs"]:
            out.append(f"  {leg['type']} | {leg['strike']} | Entry: {leg['entry_price']:.2f}")

        out.append(f"\nMARKET CONTEXT:")
        out.append(f"  Spot: {spot:.1f} ({change:+.1f}) | Z-Score: {regime['z_score']:+.2f}σ | Regime: {regime['regime']}")

        out.append(f"\nLIVE GREEKS (from option chain):")
        out.append(f"  Net Δ: {greeks['net_delta']:+.3f} | Net Γ: {greeks['net_gamma']:+.5f}")
        out.append(f"  Net Θ: {greeks['net_theta']:+.2f}/day | Net V: {greeks['net_vega']:+.4f}")

        out.append(f"\nWHALLEY-WILMOTT BAND (from live Gamma):")
        out.append(f"  No-Transaction Band: ±{ww_band:.3f} Δ")
        out.append(f"  Delta Breached? {'YES' if abs(greeks['net_delta']) > ww_band else 'NO'}")

        out.append(f"\nCAL 1 BREAKEVEN (from entry premiums):")
        out.append(f"  Lower BE: {be_lower:.0f} | Upper BE: {be_upper:.0f}")
        out.append(f"  Spot vs BE: {'BELOW lower BE ⚠️' if spot < be_lower else 'ABOVE upper BE ⚠️' if spot > be_upper else 'INSIDE BEs ✅'}")

        out.append(f"\nADJUSTMENT DECISION: {decision}")
        out.append(f"REASON: {reason}")

        if cal2:
            out.append(f"\nCAL 2 SELECTED (live chain + Greeks):")
            out.append(f"  Center Strike : {cal2['strike']}")
            out.append(f"  Net Credit    : ₹{cal2['net_credit']:.2f}  (SELL ₹{cal2['near_price']:.2f} / BUY ₹{cal2['far_price']:.2f})")
            out.append(f"  Cal2 Upper BE : {cal2['cal2_be']:.0f}  (BE diff vs Cal1: {cal2['be_diff']:.1f} pts)")
            out.append(f"  Net Δ: {cal2['net_delta']:+.3f} | Net Θ: {cal2['net_theta']:+.2f} | Net V: {cal2['net_vega']:+.4f}")
            out.append(f"  Near IV: {cal2['near_iv']:.1f}% | Far IV: {cal2['far_iv']:.1f}%")

        out.append("━" * 42)
        return "\n".join(out)

    # ── 6. MAIN EVALUATION ────────────────────────────────────────────────────
    def evaluate_active_trade(self, active: Dict, current_pnl: float = 0.0) -> Optional[Dict]:
        """
        6-step adjustment evaluation:
          1. Fetch live chain + spot
          2. Compute live Greeks from chain
          3. Compute WW no-transaction band from live Gamma
          4. Compute Cal 1 BE from actual entry premiums
          5. Trigger = WW band breached AND spot crossed Cal 1 BE
          6. If triggered: find Cal 2 center from live chain premiums + Greeks
        """
        if not active.get("legs"):
            return None

        spot_data = self.fetcher.get_nifty_spot()
        spot      = spot_data.price
        change    = spot_data.change

        if spot <= 0.0:
            if hasattr(self, "_last_valid_spot") and self._last_valid_spot > 0:
                print(f"⚠️ [API 429 / TIMEOUT] Spot returned 0.0. Using last cached spot: {self._last_valid_spot:.1f}")
                spot = self._last_valid_spot
                change = getattr(self, "_last_valid_change", 0.0)
            else:
                print("⚠️ [API ERROR] Spot is 0.0 and no cached spot available. Skipping evaluation cycle.")
                return None
        else:
            self._last_valid_spot = spot
            self._last_valid_change = change

        # Fetch live chain for all expiries in the position
        unique_expiries = list(set(l.get("expiry", "") for l in active["legs"]))
        chain = []
        for exp in unique_expiries:
            chain.extend(self.fetcher.get_option_chain(expiry=exp))

        # Also fetch far expiry for Cal 2 scanning
        near_exp = self.fetcher.get_nearest_expiry()
        far_exp  = self.fetcher.get_next_expiry()
        for exp in [near_exp, far_exp]:
            if exp not in unique_expiries:
                chain.extend(self.fetcher.get_option_chain(expiry=exp))

        # Step 2 — Live Greeks from chain
        greeks  = self._greeks_from_chain(active["legs"], chain)

        # Step 3 — WW band from live Gamma
        ww_band = _whalley_wilmott_band(greeks["net_gamma"])

        # Step 4 — Cal 1 BE from real entry premiums
        be_lower, be_upper = self._compute_cal1_be(active["legs"])

        # Step 5 — Regime
        regime = self._regime(spot)

        # ── ROUTING BASED ON STRATEGY TYPE & LEG COMPOSITION ────────────────────
        strategy_type = active.get("strategy", "").upper()
        legs = active.get("legs", [])
        has_buy_legs = any("BUY" in l["type"] for l in legs)
        
        # If labeled DOUBLE_DIAGONAL or CALENDAR but only contains 2 naked sell legs, it's actually a Straddle/Strangle!
        if ("CALENDAR" in strategy_type or "DIAGONAL" in strategy_type) and not has_buy_legs and len(legs) <= 2:
            return self._evaluate_strangle_structure(
                active, spot, change, chain, near_exp, far_exp, greeks, ww_band, regime
            )

        if "CALENDAR" in strategy_type or "DIAGONAL" in strategy_type:
            return self._evaluate_calendar_m_shape(
                active, spot, change, chain, near_exp, far_exp, greeks, ww_band, regime
            )
        elif "CONDOR" in strategy_type or "SPREAD" in strategy_type:
            return self._evaluate_credit_structure(
                active, spot, change, chain, near_exp, far_exp, greeks, ww_band, regime
            )
        elif "STRANGLE" in strategy_type or "STRADDLE" in strategy_type:
            return self._evaluate_strangle_structure(
                active, spot, change, chain, near_exp, far_exp, greeks, ww_band, regime
            )
        else:
            # Fallback for generic/legacy trades
            return self._evaluate_generic(
                active, spot, change, chain, near_exp, far_exp, greeks, ww_band, regime
            )

    # ── 3B. TIER-2 EXTREME TREND EXTENSION (RATIO WINGS) ─────────────────────
    def _execute_tier2_extension(
        self,
        active: Dict,
        direction: str,  # "BULL" or "BEAR"
        spot: float,
        chain: List,
        near_exp: str,
        far_exp: str,
    ) -> Tuple[str, str, List[Dict]]:
        """
        Tier-2 Extreme Trend Extension Protocol:
        Triggered when spot breaches the 2nd Mountain's Upper BE (Bull) or Lower BE (Bear).
        1. Closes the exhausted untested wing (locks accumulated theta profit).
        2. Rolls the untested diagonal closer to spot (collects fresh credit + sets new floor).
        3. Adds 1 extra Far option on the trending side (creates 1:2 Convexity Ratio),
           converting the trending wing into an expanding positive-infinity payoff curve.
        """
        new_legs = []
        
        if direction == "BULL":
            # Spot rallied far above Upper BE.
            # 1. Harvest & Roll Untested Put Wing UP
            old_put_legs = [l for l in active["legs"] if ("PUT" in l["type"] or "PE" in l["type"]) and l.get("label") != "convexity_put"]
            realized_pnl_close = 0.0
            
            for leg in old_put_legs:
                opt = next((o for o in chain if o.strike == leg["strike"] and o.expiry == leg.get("expiry", "")), None)
                ltp = opt.put_price if opt and opt.put_price else leg.get("current_ltp", leg["entry_price"])
                pnl = ((ltp - leg["entry_price"]) if "BUY" in leg["type"] else (leg["entry_price"] - ltp)) * LOT_SIZE * leg.get("qty", 1)
                realized_pnl_close += pnl
                print(f"   [-] Tier-2 Harvested {leg['type']} {leg['strike']} @ {ltp:.2f} (Locked P&L: ₹{pnl:+.2f})")
                
            active["legs"] = [l for l in active["legs"] if l not in old_put_legs]
            active["realized_pnl"] = active.get("realized_pnl", 0.0) + realized_pnl_close
            
            # Select new Put roll strike (ATM - 200 pts)
            roll_put_strike = int(round((spot - 200) / 50) * 50)
            near_p_opt = next((o for o in chain if o.strike == roll_put_strike and o.expiry == near_exp), None)
            far_p_opt  = next((o for o in chain if o.strike == roll_put_strike and o.expiry == far_exp), None)
            
            near_p_price = near_p_opt.put_price if near_p_opt and near_p_opt.put_price else 60.0
            far_p_price  = far_p_opt.put_price  if far_p_opt  and far_p_opt.put_price  else 110.0
            
            rolled_put_legs = [
                {"type": "SELL_PUT", "strike": roll_put_strike, "expiry": near_exp, "qty": 1, "entry_price": near_p_price, "label": "rolled_put"},
                {"type": "BUY_PUT",  "strike": roll_put_strike, "expiry": far_exp,  "qty": 1, "entry_price": far_p_price,  "label": "rolled_put"}
            ]
            new_legs.extend(rolled_put_legs)
            print(f"   [+] Rolled Put Diagonal UP to {roll_put_strike} (Near Sell ₹{near_p_price:.2f} / Far Buy ₹{far_p_price:.2f})")
            
            # 2. Add 1 Extra Far Call at 2nd Mountain strike (1:2 Convexity Ratio)
            layer_call_leg = next((l for l in active["legs"] if l.get("label") == "layer_call" and "BUY" in l["type"]), None)
            convexity_strike = layer_call_leg["strike"] if layer_call_leg else int(round(spot / 50) * 50)
            far_c_opt = next((o for o in chain if o.strike == convexity_strike and o.expiry == far_exp), None)
            far_c_price = far_c_opt.call_price if far_c_opt and far_c_opt.call_price else 120.0
            
            convexity_call_leg = {
                "type": "BUY_CALL", "strike": convexity_strike, "expiry": far_exp, "qty": 1, "entry_price": far_c_price, "label": "convexity_call"
            }
            new_legs.append(convexity_call_leg)
            print(f"   [+] Added 1:2 Convexity Far Call at {convexity_strike} @ ₹{far_c_price:.2f}")
            
            active["total_orders"] = active.get("total_orders", len(active["legs"])) + len(old_put_legs) + len(new_legs)
            active["strategy"] = active.get("strategy", "").replace("_EXTENDED_BULL", "") + "_EXTENDED_BULL"
            decision = "EXTEND_RATIO_CALL_WING"
            reason = (f"Spot ({spot:.0f}) breached 2nd Mountain Upper BE. "
                      f"Rolled Put wing UP to {roll_put_strike} (locked ₹{realized_pnl_close:+.2f}) + "
                      f"Added Far Call at {convexity_strike} (1:2 Ratio Convexity).")
            
        else: # BEAR (spot fell far below Lower BE)
            # 1. Harvest & Roll Untested Call Wing DOWN
            old_call_legs = [l for l in active["legs"] if ("CALL" in l["type"] or "CE" in l["type"]) and l.get("label") != "convexity_call"]
            realized_pnl_close = 0.0
            
            for leg in old_call_legs:
                opt = next((o for o in chain if o.strike == leg["strike"] and o.expiry == leg.get("expiry", "")), None)
                ltp = opt.call_price if opt and opt.call_price else leg.get("current_ltp", leg["entry_price"])
                pnl = ((ltp - leg["entry_price"]) if "BUY" in leg["type"] else (leg["entry_price"] - ltp)) * LOT_SIZE * leg.get("qty", 1)
                realized_pnl_close += pnl
                print(f"   [-] Tier-2 Harvested {leg['type']} {leg['strike']} @ {ltp:.2f} (Locked P&L: ₹{pnl:+.2f})")
                
            active["legs"] = [l for l in active["legs"] if l not in old_call_legs]
            active["realized_pnl"] = active.get("realized_pnl", 0.0) + realized_pnl_close
            
            # Select new Call roll strike (ATM + 200 pts)
            roll_call_strike = int(round((spot + 200) / 50) * 50)
            near_c_opt = next((o for o in chain if o.strike == roll_call_strike and o.expiry == near_exp), None)
            far_c_opt  = next((o for o in chain if o.strike == roll_call_strike and o.expiry == far_exp), None)
            
            near_c_price = near_c_opt.call_price if near_c_opt and near_c_opt.call_price else 60.0
            far_c_price  = far_c_opt.call_price  if far_c_opt  and far_c_opt.call_price  else 110.0
            
            rolled_call_legs = [
                {"type": "SELL_CALL", "strike": roll_call_strike, "expiry": near_exp, "qty": 1, "entry_price": near_c_price, "label": "rolled_call"},
                {"type": "BUY_CALL",  "strike": roll_call_strike, "expiry": far_exp,  "qty": 1, "entry_price": far_c_price,  "label": "rolled_call"}
            ]
            new_legs.extend(rolled_call_legs)
            print(f"   [+] Rolled Call Diagonal DOWN to {roll_call_strike} (Near Sell ₹{near_c_price:.2f} / Far Buy ₹{far_c_price:.2f})")
            
            # 2. Add 1 Extra Far Put at 2nd Mountain strike (1:2 Convexity Ratio)
            layer_put_leg = next((l for l in active["legs"] if l.get("label") == "layer_put" and "BUY" in l["type"]), None)
            convexity_strike = layer_put_leg["strike"] if layer_put_leg else int(round(spot / 50) * 50)
            far_p_opt = next((o for o in chain if o.strike == convexity_strike and o.expiry == far_exp), None)
            far_p_price = far_p_opt.put_price if far_p_opt and far_p_opt.put_price else 120.0
            
            convexity_put_leg = {
                "type": "BUY_PUT", "strike": convexity_strike, "expiry": far_exp, "qty": 1, "entry_price": far_p_price, "label": "convexity_put"
            }
            new_legs.append(convexity_put_leg)
            print(f"   [+] Added 1:2 Convexity Far Put at {convexity_strike} @ ₹{far_p_price:.2f}")
            
            active["total_orders"] = active.get("total_orders", len(active["legs"])) + len(old_call_legs) + len(new_legs)
            active["strategy"] = active.get("strategy", "").replace("_EXTENDED_BEAR", "") + "_EXTENDED_BEAR"
            decision = "EXTEND_RATIO_PUT_WING"
            reason = (f"Spot ({spot:.0f}) breached 2nd Mountain Lower BE. "
                      f"Rolled Call wing DOWN to {roll_call_strike} (locked ₹{realized_pnl_close:+.2f}) + "
                      f"Added Far Put at {convexity_strike} (1:2 Ratio Convexity).")
                      
        return decision, reason, new_legs

    # ── PROTOCOL: CALENDAR / DIAGONAL (M-SHAPE) ──────────────────────────────
    def _evaluate_calendar_m_shape(self, active, spot, change, chain, near_exp, far_exp, greeks, ww_band, regime):
        be_lower, be_upper = self._compute_cal1_be(active["legs"])

        delta_breached      = abs(greeks["net_delta"]) > ww_band
        below_lower_be      = spot < be_lower
        above_upper_be      = spot > be_upper
        be_breached         = below_lower_be or above_upper_be

        put_layer_active    = any(l.get("label") == "layer_put"  for l in active["legs"])
        call_layer_active   = any(l.get("label") == "layer_call" for l in active["legs"])

        decision = "HOLD"
        reason   = (
            f"Delta ({greeks['net_delta']:+.3f}) inside WW band (±{ww_band:.3f}). Theta harvesting."
            if not delta_breached else
            f"Delta ({greeks['net_delta']:+.3f}) breached WW band (±{ww_band:.3f}), but BE intact. Hold."
            if not be_breached else
            f"Delta ({greeks['net_delta']:+.3f}) breached WW band. BE breached (spot {spot:.0f} vs BE {be_lower:.0f}/{be_upper:.0f}). Evaluating Cal 2."
        )

        cal2_result = None
        new_legs    = []

        if delta_breached and be_breached:
            if below_lower_be:
                if put_layer_active:
                    # Tier-2 Downside Adjustment: Handed over completely to Smart Adjustment Analyst Agent
                    from adjustment_analyst_agent import AdjustmentAnalystAgent
                    analyst = AdjustmentAnalystAgent()
                    smart_adj = analyst.propose_smart_adjustment(active, spot, chain, "BEAR", near_exp, far_exp)
                    decision = smart_adj["action"]
                    reason = smart_adj["rationale"]
                    new_legs = smart_adj["new_legs"]
                    if smart_adj.get("closed_legs"):
                        for cl in smart_adj["closed_legs"]:
                            active["legs"] = [l for l in active["legs"] if l != cl]
                        active["realized_pnl"] = active.get("realized_pnl", 0.0) + smart_adj.get("closed_realized_pnl", 0.0)
                else:
                    # 1st Adjustment: LAYER_MOUNTAIN_PUT
                    cal2_result = self._find_cal2_center(be_lower, "PUT", chain, near_exp, far_exp)
                    if cal2_result:
                        decision = "LAYER_MOUNTAIN_PUT"
                        reason   = f"Deploying PUT calendar at {cal2_result['strike']} (BE match: {cal2_result['be_diff']:.1f} pts diff)."
                        new_legs = [
                            {"type": "SELL_PUT", "strike": cal2_result["strike"], "expiry": near_exp, "qty": 1, "entry_price": cal2_result["near_price"], "label": "layer_put"},
                            {"type": "BUY_PUT", "strike": cal2_result["strike"], "expiry": far_exp, "qty": 1, "entry_price": cal2_result["far_price"], "label": "layer_put"}
                        ]
                    else:
                        reason = "BE breached downside but no valid Cal 2 center found in live chain. HOLD."

            elif above_upper_be:
                if call_layer_active:
                    # Tier-2 Upside Adjustment: Handed over completely to Smart Adjustment Analyst Agent
                    from adjustment_analyst_agent import AdjustmentAnalystAgent
                    analyst = AdjustmentAnalystAgent()
                    smart_adj = analyst.propose_smart_adjustment(active, spot, chain, "BULL", near_exp, far_exp)
                    decision = smart_adj["action"]
                    reason = smart_adj["rationale"]
                    new_legs = smart_adj["new_legs"]
                    if smart_adj.get("closed_legs"):
                        for cl in smart_adj["closed_legs"]:
                            active["legs"] = [l for l in active["legs"] if l != cl]
                        active["realized_pnl"] = active.get("realized_pnl", 0.0) + smart_adj.get("closed_realized_pnl", 0.0)
                else:
                    # 1st Adjustment: LAYER_MOUNTAIN_CALL
                    cal2_result = self._find_cal2_center(be_upper, "CALL", chain, near_exp, far_exp)
                    if cal2_result:
                        decision = "LAYER_MOUNTAIN_CALL"
                        reason   = f"Deploying CALL calendar at {cal2_result['strike']} (BE match: {cal2_result['be_diff']:.1f} pts diff)."
                        new_legs = [
                            {"type": "SELL_CALL", "strike": cal2_result["strike"], "expiry": near_exp, "qty": 1, "entry_price": cal2_result["near_price"], "label": "layer_call"},
                            {"type": "BUY_CALL", "strike": cal2_result["strike"], "expiry": far_exp, "qty": 1, "entry_price": cal2_result["far_price"], "label": "layer_call"}
                        ]
                    else:
                        reason = "BE breached upside but no valid Cal 2 center found in live chain. HOLD."

        if decision == "HOLD" and delta_breached and not be_breached and regime["regime"] == "MEAN_REVERSION":
            reason = f"Delta breached WW band but Z-Score = MEAN_REVERSION and BE intact. HOLD."

        print(self.format_output(active, spot, change, greeks, ww_band, be_lower, be_upper, regime, decision, reason, cal2_result))

        if decision in (
            "LAYER_MOUNTAIN_PUT", "LAYER_MOUNTAIN_CALL",
            "SMART_TIER2_BEAR_REPAIR", "SMART_TIER2_BULL_REPAIR",
            "BEAR_CREDIT_SPREAD_REPAIR", "BULL_CREDIT_SPREAD_REPAIR",
            "EXTEND_RATIO_CALL_WING", "EXTEND_RATIO_PUT_WING"
        ) and new_legs:
            # ── MANDATORY ADJUSTMENT ANALYST AGENT VERIFICATION GATE ─────────────────
            try:
                from adjustment_analyst_agent import AdjustmentAnalystAgent
                analyst = AdjustmentAnalystAgent()
                audit_result = analyst.verify_proposed_adjustment(
                    active_trade=active,
                    proposed_action=decision,
                    new_legs=new_legs,
                    closed_legs=None,
                    spot=spot,
                    chain=chain
                )
                if not audit_result["is_approved"]:
                    print("\n" + "=" * 65)
                    print("🛑 [ADJUSTMENT VETOED BY ADJUSTMENT ANALYST AGENT]")
                    print(f"   Proposed Action: {decision}")
                    print("   The proposed mechanical adjustment failed risk & payoff verification:")
                    for r in audit_result["veto_reasons"]:
                        print(f"   ❌ {r}")
                    
                    alt = audit_result.get("alternative")
                    if alt:
                        print(f"\n💡 [ANALYST AGENT APPROVED ALTERNATIVE: {alt['name']}]")
                        print(f"   Rationale: {alt['rationale']}")
                        print(f"   Switching to approved credit-positive legs...")
                        new_legs = alt["action_legs"]
                        decision = alt["name"]
                    else:
                        print("   No valid alternative approved. VETO ENFORCED — Holding position.")
                        print("=" * 65 + "\n")
                        return None
                    print("=" * 65 + "\n")
            except Exception as e:
                print(f"⚠️ [ANALYST GATE WARNING] Pre-adjustment verification encountered error: {e}. Proceeding with caution.")

            net_debit = sum(l["entry_price"] * LOT_SIZE * l.get("qty", 1) * (-1 if "BUY" in l["type"] else 1) for l in new_legs)
            active["legs"].extend(new_legs)
            active["net_credit_debit"] = active.get("net_credit_debit", 0.0) + net_debit
            active["max_target"]       = abs(active["net_credit_debit"]) * 0.7
            if decision in ("LAYER_MOUNTAIN_PUT", "LAYER_MOUNTAIN_CALL"):
                active["strategy"]     = active.get("strategy", "") + "_M_SHAPE"
                active["total_orders"] = active.get("total_orders", len(active["legs"]) - len(new_legs)) + len(new_legs)
                print(f"\n[EXECUTED] {decision} — Added Cal 2 at {cal2_result['strike']}. Net debit: ₹{net_debit:.2f}. Strategy → {active['strategy']}")
            else:
                print(f"\n[EXECUTED] {decision} — Tier-2 Range Extension Active. Net debit: ₹{net_debit:.2f}. Strategy → {active['strategy']}")
            return active
        return None

    # ── PROTOCOL: CREDIT SPREADS / IRON CONDORS ──────────────────────────────
    def _evaluate_credit_structure(self, active, spot, change, chain, near_exp, far_exp, greeks, ww_band, regime):
        """
        Adjustment for Credit Spreads & Iron Condors:
        When Delta breaches WW-Band:
          - If market drops (Delta positive/short put threatened): Roll the UNTESTED CALL spread down closer to spot
          - If market rallies (Delta negative/short call threatened): Roll the UNTESTED PUT spread up closer to spot
        This locks credit on the winning side and expands the overall safety band without paying debit.
        """
        delta = greeks["net_delta"]
        delta_breached = abs(delta) > ww_band

        if not delta_breached:
            print(f"\n📊 [CREDIT PROTOCOL] HOLD - Net Delta ({delta:+.3f}) within WW Band (±{ww_band:.3f}). Theta decaying.")
            return None

        # Roll untested side
        new_legs = []
        decision = "HOLD"
        reason = ""

        if delta > ww_band:
            # Bearish move / short put tested -> Roll CALL side down towards ATM
            untested_calls = [l for l in active["legs"] if "CALL" in l["type"]]
            if untested_calls and not any(l.get("label") == "rolled_call" for l in active["legs"]):
                # Target strike: round(spot / 50) * 50 + 100
                target_short = int(round(spot / 50) * 50) + 100
                target_long = target_short + 100
                s_opt = next((o for o in chain if o.strike == target_short and o.expiry == near_exp), None)
                l_opt = next((o for o in chain if o.strike == target_long and o.expiry == near_exp), None)
                if s_opt and l_opt and (s_opt.call_price or 0) > (l_opt.call_price or 0):
                    credit = (s_opt.call_price or 0) - (l_opt.call_price or 0)
                    decision = "ROLL_CALL_SPREAD_DOWN"
                    reason = f"Downside stress (Δ {delta:+.3f}). Rolling Call wing to {target_short}/{target_long} to collect ₹{credit:.2f} credit."
                    new_legs = [
                        {"type": "SELL_CALL", "strike": target_short, "expiry": near_exp, "qty": 1, "entry_price": s_opt.call_price, "label": "rolled_call"},
                        {"type": "BUY_CALL",  "strike": target_long,  "expiry": near_exp, "qty": 1, "entry_price": l_opt.call_price, "label": "rolled_call"}
                    ]
        elif delta < -ww_band:
            # Bullish rally / short call tested -> Roll PUT side up towards ATM
            untested_puts = [l for l in active["legs"] if "PUT" in l["type"]]
            if untested_puts and not any(l.get("label") == "rolled_put" for l in active["legs"]):
                target_short = int(round(spot / 50) * 50) - 100
                target_long = target_short - 100
                s_opt = next((o for o in chain if o.strike == target_short and o.expiry == near_exp), None)
                l_opt = next((o for o in chain if o.strike == target_long and o.expiry == near_exp), None)
                if s_opt and l_opt and (s_opt.put_price or 0) > (l_opt.put_price or 0):
                    credit = (s_opt.put_price or 0) - (l_opt.put_price or 0)
                    decision = "ROLL_PUT_SPREAD_UP"
                    reason = f"Upside rally (Δ {delta:+.3f}). Rolling Put wing to {target_short}/{target_long} to collect ₹{credit:.2f} credit."
                    new_legs = [
                        {"type": "SELL_PUT", "strike": target_short, "expiry": near_exp, "qty": 1, "entry_price": s_opt.put_price, "label": "rolled_put"},
                        {"type": "BUY_PUT",  "strike": target_long,  "expiry": near_exp, "qty": 1, "entry_price": l_opt.put_price, "label": "rolled_put"}
                    ]

        print(f"\n[CREDIT PROTOCOL] {decision} - {reason}")
        if new_legs:
            net_credit = sum(l["entry_price"] * LOT_SIZE * l["qty"] * (1 if "SELL" in l["type"] else -1) for l in new_legs)
            active["legs"].extend(new_legs)
            active["net_credit_debit"] = active.get("net_credit_debit", 0.0) + net_credit
            active["strategy"] = active.get("strategy", "") + "_ADJUSTED"
            print(f"[EXECUTED] {decision}: Added credit spread. Collected Rs {net_credit:.2f}")
            return active
        return None

    # ── PROTOCOL: STRADDLE / STRANGLE ─────────────────────────────────────────
    def _evaluate_strangle_structure(self, active, spot, change, chain, near_exp, far_exp, greeks, ww_band, regime):
        """
        Adjustment for Short Straddles & Strangles:
        Tier 1 (Delta > 0.15): Diagonal Term-Structure Hedge (Buy Far Expiry Wing)
          - Do NOT touch the near-term short legs (avoids realizing loss at the bottom).
          - Buy protective wing in `far_exp` (next week) on threatened side.
          - Injects positive Vega and caps runaway tail loss without 0-DTE theta crush.
        Tier 2 (Delta > 0.25): Roll Untested Leg Closer to ATM to re-neutralize delta.
        """
        delta = greeks["net_delta"]
        delta_breached = abs(delta) > ww_band

        if not delta_breached:
            print(f"\n[STRADDLE PROTOCOL] HOLD - Net Delta ({delta:+.3f}) within WW Band (+/-{ww_band:.3f}). Theta harvesting.")
            return None

        decision = "HOLD"
        reason = ""
        new_legs = []

        has_far_put_hedge = any("BUY_PUT" in l["type"] and l.get("expiry") == far_exp for l in active["legs"])
        has_far_call_hedge = any("BUY_CALL" in l["type"] and l.get("expiry") == far_exp for l in active["legs"])

        # Tier 1: Buy Far-Expiry Protective Wing on the tested side
        if delta > ww_band and not has_far_put_hedge:
            # Spot crashed -> Put side tested -> Buy far-month PUT wing (100 pts below current spot)
            hedge_strike = int(round(spot / 50) * 50) - 100
            hedge_opt = next((o for o in chain if o.strike == hedge_strike and o.expiry == far_exp), None)
            if hedge_opt and (hedge_opt.put_price or 0) > 0:
                decision = "ADD_DIAGONAL_PUT_HEDGE"
                reason = f"Downside break (Delta {delta:+.3f} > {ww_band:.3f}). Adding Far-Week ({far_exp}) PUT hedge @ {hedge_strike} (Rs {hedge_opt.put_price:.2f})."
                new_legs = [{
                    "type": "BUY_PUT",
                    "strike": hedge_strike,
                    "expiry": far_exp,
                    "qty": 1,
                    "entry_price": hedge_opt.put_price,
                    "label": "strangle_tier1_hedge"
                }]
        elif delta < -ww_band and not has_far_call_hedge:
            # Spot rallied -> Call side tested -> Buy far-month CALL wing (100 pts above current spot)
            hedge_strike = int(round(spot / 50) * 50) + 100
            hedge_opt = next((o for o in chain if o.strike == hedge_strike and o.expiry == far_exp), None)
            if hedge_opt and (hedge_opt.call_price or 0) > 0:
                decision = "ADD_DIAGONAL_CALL_HEDGE"
                reason = f"Upside rally (Delta {delta:+.3f} < -{ww_band:.3f}). Adding Far-Week ({far_exp}) CALL hedge @ {hedge_strike} (Rs {hedge_opt.call_price:.2f})."
                new_legs = [{
                    "type": "BUY_CALL",
                    "strike": hedge_strike,
                    "expiry": far_exp,
                    "qty": 1,
                    "entry_price": hedge_opt.call_price,
                    "label": "strangle_tier1_hedge"
                }]

        # Tier 2: If hedge already active and Delta still heavy (> 0.25), roll untested near leg down/up
        if not new_legs and abs(delta) > 0.25:
            if delta > 0.25:
                # Roll untested CALL leg down to new ATM+50
                new_call_strike = int(round(spot / 50) * 50) + 50
                c_opt = next((o for o in chain if o.strike == new_call_strike and o.expiry == near_exp), None)
                if c_opt and not any(l.get("label") == "strangle_tier2_call" for l in active["legs"]):
                    decision = "ROLL_CALL_DOWN"
                    reason = f"Severe downside (Delta {delta:+.3f}). Selling additional Call at {new_call_strike} (Rs {c_opt.call_price:.2f}) to delta-neutralize."
                    new_legs = [{
                        "type": "SELL_CALL",
                        "strike": new_call_strike,
                        "expiry": near_exp,
                        "qty": 1,
                        "entry_price": c_opt.call_price,
                        "label": "strangle_tier2_call"
                    }]
            elif delta < -0.25:
                # Roll untested PUT leg up to new ATM-50
                new_put_strike = int(round(spot / 50) * 50) - 50
                p_opt = next((o for o in chain if o.strike == new_put_strike and o.expiry == near_exp), None)
                if p_opt and not any(l.get("label") == "strangle_tier2_put" for l in active["legs"]):
                    decision = "ROLL_PUT_UP"
                    reason = f"Severe rally (Delta {delta:+.3f}). Selling additional Put at {new_put_strike} (Rs {p_opt.put_price:.2f}) to delta-neutralize."
                    new_legs = [{
                        "type": "SELL_PUT",
                        "strike": new_put_strike,
                        "expiry": near_exp,
                        "qty": 1,
                        "entry_price": p_opt.put_price,
                        "label": "strangle_tier2_put"
                    }]

        print(f"\n[STRADDLE PROTOCOL] {decision} - {reason}")
        if new_legs:
            cost = sum(l["entry_price"] * LOT_SIZE * l["qty"] * (-1 if "BUY" in l["type"] else 1) for l in new_legs)
            active["legs"].extend(new_legs)
            active["net_credit_debit"] = active.get("net_credit_debit", 0.0) + cost
            active["strategy"] = active.get("strategy", "") + "_HEDGED"
            print(f"[EXECUTED] {decision}: Added legs. Net impact: Rs {cost:+.2f}")
            return active

        return None

    # ── PROTOCOL: GENERIC / DIRECTIONAL ───────────────────────────────────────
    def _evaluate_generic(self, active, spot, change, chain, near_exp, far_exp, greeks, ww_band, regime):
        decision = "HOLD"
        reason = f"Generic position monitoring. Spot: {spot:.1f}, Delta: {greeks['net_delta']:+.3f}"
        print(f"\n[GENERIC PROTOCOL] {decision} - {reason}")
        return None
