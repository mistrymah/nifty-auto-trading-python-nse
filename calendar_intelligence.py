import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

# Ensure we can import fetcher
sys.path.insert(0, str(Path(__file__).parent))
from fetcher import UpstoxFetcher

LOT_SIZE = 65

class CalendarIntelligence:
    def __init__(self):
        self.fetcher = UpstoxFetcher()
        self.market_data_dir = Path(__file__).parent / "market_data"
        self.market_data_dir.mkdir(exist_ok=True)
        self.context_file = self.market_data_dir / "market_context.json"
        self.proposal_file = Path(__file__).parent / "proposed_strategy.json"

    def analyze_market(self) -> dict:
        """Pulls live option chain and calculates institutional GEX & Volatility metrics."""
        print("[INTELLIGENCE] Fetching live market data...")
        spot_data = self.fetcher.get_nifty_spot()
        spot = spot_data.price

        near_exp = self.fetcher.get_nearest_expiry()
        far_exp = self.fetcher.get_next_expiry()

        near_chain = self.fetcher.get_option_chain(expiry=near_exp)
        far_chain = self.fetcher.get_option_chain(expiry=far_exp)

        if not near_chain or not far_chain:
            print("[INTELLIGENCE] Failed to fetch option chains.")
            return {}

        # ── 1. Calculate Gamma Exposure (GEX) Profile ──────────────────────
        # Dealers hedge calls by buying spot (Positive GEX) and puts by selling spot (Negative GEX)
        total_gex = 0.0
        for opt in near_chain:
            c_oi = opt.call_oi or 0
            p_oi = opt.put_oi or 0
            c_gamma = opt.call_gamma or 0
            p_gamma = opt.put_gamma or 0
            
            # GEX approximation (Spot * Gamma * OI)
            total_gex += (c_oi * c_gamma - p_oi * p_gamma) * spot * LOT_SIZE

        # ── 2. Calculate IV Term Structure (Contango/Backwardation) ────────
        # Find ATM options
        atm_strike = round(spot / 50) * 50
        near_atm = next((o for o in near_chain if o.strike == atm_strike), None)
        far_atm = next((o for o in far_chain if o.strike == atm_strike), None)

        near_iv = ((near_atm.call_iv or 0) + (near_atm.put_iv or 0)) / 2 if near_atm else 0
        far_iv = ((far_atm.call_iv or 0) + (far_atm.put_iv or 0)) / 2 if far_atm else 0
        term_structure = "CONTANGO" if far_iv > near_iv else "BACKWARDATION"

        # ── 3. Calculate 25-Delta Risk Reversal (Volatility Skew) ──────────
        # Find strikes closest to 0.25 delta
        call_25d = min(near_chain, key=lambda o: abs((o.call_delta or 0) - 0.25), default=None)
        put_25d = min(near_chain, key=lambda o: abs((o.put_delta or 0) - (-0.25)), default=None)
        
        c_25_iv = call_25d.call_iv or 0 if call_25d else 0
        p_25_iv = put_25d.put_iv or 0 if put_25d else 0
        risk_reversal = p_25_iv - c_25_iv

        context = {
            "timestamp": datetime.now().isoformat(),
            "spot": spot,
            "near_expiry": near_exp,
            "far_expiry": far_exp,
            "metrics": {
                "total_gex_billions": total_gex / 1_000_000_000,
                "near_atm_iv": near_iv,
                "far_atm_iv": far_iv,
                "term_structure": term_structure,
                "risk_reversal_skew": risk_reversal
            },
            "interpretation": {
                "gex_regime": "PINNING (Mean Reversion)" if total_gex > 0 else "TRENDING (Explosive)",
                "sentiment": "BEARISH FEAR" if risk_reversal > 2.0 else "BULLISH GREED" if risk_reversal < -2.0 else "NEUTRAL"
            }
        }

        # Write context for AI to read
        with open(self.context_file, 'w') as f:
            json.dump(context, f, indent=4)
            
        print(f"[INTELLIGENCE] Context updated. GEX: {context['metrics']['total_gex_billions']:+.2f}B | Skew: {risk_reversal:+.1f}% | Structure: {term_structure}")
        return context

    def auto_decide(self, context: dict):
        """Hardcoded Quant rules fallback. Generates the setup without AI."""
        if not context:
            return

        spot = context["spot"]
        metrics = context["metrics"]
        interp = context["interpretation"]
        
        atm_strike = round(spot / 50) * 50

        # Hardcoded Matrix
        strategy_type = "DOUBLE_DIAGONAL"
        sell_call_strike = atm_strike
        sell_put_strike = atm_strike
        
        if interp["gex_regime"] == "PINNING (Mean Reversion)":
            # Market is sticky. Center the calendar.
            sell_call_strike = atm_strike
            sell_put_strike = atm_strike
            reason = "Hardcoded Rule: Positive GEX detected. Market likely to pin. Seeding Centered Calendar."
        else:
            # Negative GEX = Explosive. Check Skew to offset strikes.
            if interp["sentiment"] == "BEARISH FEAR":
                # Offset down
                sell_call_strike = atm_strike - 50
                sell_put_strike = atm_strike - 50
                reason = "Hardcoded Rule: Negative GEX + Bearish Skew. Deploying Bear-biased Diagonal."
            elif interp["sentiment"] == "BULLISH GREED":
                # Offset up
                sell_call_strike = atm_strike + 50
                sell_put_strike = atm_strike + 50
                reason = "Hardcoded Rule: Negative GEX + Bullish Skew. Deploying Bull-biased Diagonal."
            else:
                reason = "Hardcoded Rule: Negative GEX but Neutral Skew. Seeding Centered Calendar."

        # Diagonals buy protection 100pts away in far week
        setup = {
            "strategy": strategy_type,
            "target_multiplier": 0.7,
            "stoploss_multiplier": -0.5,
            "legs": [
                {"type": "SELL_CALL", "strike": sell_call_strike, "expiry": context["near_expiry"], "qty": 1},
                {"type": "SELL_PUT",  "strike": sell_put_strike,  "expiry": context["near_expiry"], "qty": 1},
                {"type": "BUY_CALL",  "strike": sell_call_strike + 100, "expiry": context["far_expiry"], "qty": 1},
                {"type": "BUY_PUT",   "strike": sell_put_strike - 100,  "expiry": context["far_expiry"], "qty": 1}
            ],
            "reasoning": reason
        }

        with open(self.proposal_file, 'w') as f:
            json.dump(setup, f, indent=4)
        print(f"[INTELLIGENCE] Auto-Decide triggered. Proposed strategy saved to {self.proposal_file.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-decide", action="store_true", help="Bypass AI and generate entry strictly via Python rules")
    args = parser.parse_args()

    engine = CalendarIntelligence()
    context = engine.analyze_market()

    if args.auto_decide:
        engine.auto_decide(context)
