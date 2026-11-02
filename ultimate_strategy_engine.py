import sys
sys.stdout.reconfigure(encoding='utf-8')
import time
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from fetcher import UpstoxFetcher
from pattern_recognition import PatternRecognizer, PatternResult
from multi_factor_strategy import MultiFactorStrategy, CPRLevels
from kronos_integration import NiftyKronosPredictor

class UltimateStrategyEngine:
    def __init__(self, data_dir: str = "market_data"):
        self.fetcher = UpstoxFetcher()
        self.multi_factor = MultiFactorStrategy()
        self.pattern_recognizer = PatternRecognizer()
        self.ml_predictor = NiftyKronosPredictor()
        
        self.project_root = Path(__file__).parent
        self.brain_dir = Path(r"C:\Users\admin\.gemini\antigravity\brain\f70b260d-7743-49b0-86a6-f817d402d052")
        self.brain_dir.mkdir(parents=True, exist_ok=True)

    def fetch_multi_timeframe_data(self) -> Dict[str, pd.DataFrame]:
        print("Fetching multi-timeframe data...")
        return {
            "1d": self.fetcher.get_nifty_historical("1day", 5),
            "1h": self.fetcher.get_nifty_historical("60minute", 45),
            "15m": self.fetcher.get_nifty_historical("15minute", 3),
            "5m": self.fetcher.get_nifty_historical("5minute", 3)
        }

    def analyze_technicals(self, mt_data: Dict[str, pd.DataFrame], spot: float) -> Dict:
        print("Analyzing technical patterns and S/R...")
        cpr, _ = self.multi_factor.get_cpr_from_yesterday()
        
        # Support/Resistance from H1 and 15m
        sr_h1 = self.pattern_recognizer.detect_support_resistance(mt_data["1h"], spot)
        sr_15m = self.pattern_recognizer.detect_support_resistance(mt_data["15m"], spot)
        
        # Patterns
        patterns_15m = self.pattern_recognizer.detect_candlestick(mt_data["15m"])
        patterns_1h = self.pattern_recognizer.detect_candlestick(mt_data["1h"])
        
        # Trend
        trend_5m = self.multi_factor.get_trend(mt_data["5m"])
        trend_15m = self.multi_factor.get_trend(mt_data["15m"])
        
        return {
            "cpr": cpr,
            "sr_zones": {"h1": sr_h1, "15m": sr_15m},
            "patterns": {"1h": [p.name for p in patterns_1h], "15m": [p.name for p in patterns_15m]},
            "trend": {"5m": trend_5m, "15m": trend_15m}
        }

    def get_market_intel(self, spot: float) -> Dict:
        print("Fetching Options Data & Market Intel...")
        try:
            from calendar_intelligence import CalendarIntelligence
            cal_intel = CalendarIntelligence()
            intel_context = cal_intel.analyze_market()
            
            chain = self.fetcher.get_option_chain()
            
            # PCR & Max Pain
            total_call_oi = sum(o.call_oi for o in chain if o.call_oi)
            total_put_oi = sum(o.put_oi for o in chain if o.put_oi)
            pcr = total_put_oi / total_call_oi if total_call_oi else 1.0
            
            strikes_oi = {}
            for o in chain:
                strikes_oi[o.strike] = (o.call_oi or 0) + (o.put_oi or 0)
            max_pain = max(strikes_oi.items(), key=lambda x: x[1])[0] if strikes_oi else int(spot)
            
            metrics = intel_context.get("metrics", {})
            interp = intel_context.get("interpretation", {})
            
            return {
                "pcr": pcr,
                "max_pain": max_pain,
                "vix": 13.5,
                "chain": chain,
                "total_gex": metrics.get("total_gex_billions", 0.0),
                "gex_regime": interp.get("gex_regime", "PINNING (Mean Reversion)"),
                "risk_reversal_skew": metrics.get("risk_reversal_skew", 0.0),
                "sentiment": interp.get("sentiment", "NEUTRAL"),
                "term_structure": metrics.get("term_structure", "CONTANGO")
            }
        except Exception as e:
            print(f"Error fetching market intel: {e}")
            return {
                "pcr": 1.0, "max_pain": int(spot), "vix": 13.5, "chain": [],
                "total_gex": 0.0, "gex_regime": "NEUTRAL", "risk_reversal_skew": 0.0,
                "sentiment": "NEUTRAL", "term_structure": "CONTANGO"
            }

    def get_ml_forecast(self, df_1h: pd.DataFrame) -> Dict:
        """
        Weekly Campaign ML Forecaster:
        Operates on 1-Hour candles across a 31-candle horizon (5 full trading days, ~31.25 hours).
        Directly matches the Tuesday 15:30 to next Tuesday 15:10 weekly expiry campaign.
        """
        print("Running Kronos 5-Day Weekly ML Forecast (1-Hour Candles)...")
        try:
            if df_1h.empty or len(df_1h) < 30:
                print("  -> Insufficient hourly data for Kronos. Falling back to neutral.")
                return {"status": "insufficient_data", "change_pts": 0.0, "pred_volatility": 0.0}
                
            # 31 hourly candles = exactly 5 full trading days (6.25 hrs/day * 5 days = 31.25 hours)
            pred_len = 31
            lookback = min(200, len(df_1h) - 1)
            pred = self.ml_predictor.forecast(df_1h, lookback=lookback, pred_len=pred_len)
            
            current_close = float(df_1h["close"].iloc[-1])
            predicted_expiry_close = float(pred["close"].iloc[-1])
            change_5d_pts = predicted_expiry_close - current_close
            change_5d_pct = (change_5d_pts / current_close) * 100.0
            
            pred_volatility = float(pred["close"].std())
            pred_high = float(pred["close"].max())
            pred_low = float(pred["close"].min())
            
            print(f"  -> Timeframe: 1-Hour (60m) candles | Context Lookback: {lookback} candles")
            print(f"  -> [5-DAY EXPIRY FORECAST]: {predicted_expiry_close:.2f} ({change_5d_pts:+.1f} pts | {change_5d_pct:+.2f}%)")
            print(f"  -> Projected 5-Day Range: Low {pred_low:.1f} <----> High {pred_high:.1f} (Span: {pred_high - pred_low:.1f} pts)")
            print(f"  -> Projected 5-Day Volatility (StdDev): {pred_volatility:.2f}")
            
            # Log the 5-day prediction for accuracy tracking
            try:
                log_dir = self.project_root / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = log_dir / "kronos_predictions.csv"
                
                last_timestamp = df_1h["timestamps"].iloc[-1]
                target_timestamp = last_timestamp + pd.Timedelta(hours=pred_len)
                
                new_row = pd.DataFrame([{
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "last_candle_time": str(last_timestamp),
                    "target_timestamp": str(target_timestamp),
                    "current_price": current_close,
                    "predicted_price": predicted_expiry_close,
                    "predicted_high": pred_high,
                    "predicted_low": pred_low,
                    "actual_price": "",
                    "accuracy_pct": ""
                }])
                
                if not log_file.exists():
                    new_row.to_csv(log_file, index=False)
                else:
                    new_row.to_csv(log_file, mode='a', header=False, index=False)
            except Exception as log_e:
                print(f"Error logging prediction: {log_e}")

            return {
                "status": "success",
                "current": current_close,
                "predicted": predicted_expiry_close,
                "change_pts": change_5d_pts,
                "change_pct": change_5d_pct,
                "pred_volatility": pred_volatility,
                "pred_high": pred_high,
                "pred_low": pred_low
            }
        except Exception as e:
            print(f"ML Forecast Error: {e}")
            return {"status": "error", "message": str(e), "change_pts": 0.0, "pred_volatility": 0.0}

    def strategy_selection_matrix(self, spot: float, tech: Dict, intel: Dict, ml: Dict, active_strategies: List[str]) -> List[Dict]:
        """Returns a prioritized list of candidate strategies to be mathematically tested."""
        volatility_env = "LOW" if intel.get("vix", 15) < 15 else "HIGH"
        trend_15m = tech["trend"]["15m"]
        
        ml_change = ml.get("change_pts", 0)
        ml_vol = ml.get("pred_volatility", 0)
        
        closest_sup_raw = tech["sr_zones"]["h1"]["support"][0] if tech["sr_zones"]["h1"]["support"] else spot - 100
        closest_res_raw = tech["sr_zones"]["h1"]["resistance"][0] if tech["sr_zones"]["h1"]["resistance"] else spot + 100
        closest_sup = int(round(closest_sup_raw / 50) * 50)
        closest_res = int(round(closest_res_raw / 50) * 50)
        
        candidates = []
        
        # We are Option Sellers. We ONLY harvest Theta. No debit spreads.
        # We use Kronos to determine where to safely center our Mountains (Diagonals/Calendars).
        
        spot_rounded = int(round(spot/50)*50)
        predicted_destination = int(round((spot + ml_change)/50)*50)
        
        gex_regime = intel.get("gex_regime", "PINNING (Mean Reversion)")
        skew = intel.get("risk_reversal_skew", 0.0)
        sentiment = intel.get("sentiment", "NEUTRAL")
        total_gex = intel.get("total_gex", 0.0)

        if not any(any(base_strat in s for base_strat in ["DOUBLE_DIAGONAL", "CALENDAR"]) for s in active_strategies):
            # ── INTEGRATED MULTI-FACTOR BIASING MATRIX ───────────────────────
            # Combines:
            #   1. Institutional Dealer GEX Profile (Pinning vs Explosive)
            #   2. 25-Delta Volatility Skew (Put IV - Call IV)
            #   3. Kronos ML Price Projection
            
            # Case 1: Positive GEX Pinning & Balanced Volatility Skew
            if "PINNING" in gex_regime and abs(skew) <= 2.0 and abs(ml_change) <= 60:
                candidates.append({
                    "strategy": "DOUBLE_DIAGONAL",
                    "strikes": [spot_rounded],
                    "reasoning": f"GEX Pinning ({total_gex:+.2f}B) + Neutral Vol Skew ({skew:+.1f}%) + ML ({ml_change:+.1f} pts). Centered ATM Double Diagonal.",
                    "ml_forecast": ml_change
                })
            # Case 2: Bullish Skew (Call Greed Skew < -2.0% OR ML Rally > +60 pts)
            elif (skew < -2.0 and ml_change >= -20) or ml_change > 60:
                candidates.append({
                    "strategy": "DOUBLE_DIAGONAL",
                    "strikes": [spot_rounded + 50],
                    "reasoning": f"Bullish Biased: Vol Skew ({skew:+.1f}% {sentiment}) + ML ({ml_change:+.1f} pts). Seeding Bull-skewed Double Diagonal.",
                    "ml_forecast": ml_change
                })
            # Case 3: Bearish Skew (Put Fear Skew > +2.0% OR ML Drop < -60 pts)
            elif (skew > 2.0 and ml_change <= 20) or ml_change < -60:
                candidates.append({
                    "strategy": "DOUBLE_DIAGONAL",
                    "strikes": [spot_rounded - 50],
                    "reasoning": f"Bearish Biased: Vol Skew ({skew:+.1f}% {sentiment}) + ML ({ml_change:+.1f} pts). Seeding Bear-skewed Double Diagonal.",
                    "ml_forecast": ml_change
                })
            # Case 4: Default Safe ATM
            else:
                candidates.append({
                    "strategy": "DOUBLE_DIAGONAL",
                    "strikes": [spot_rounded],
                    "reasoning": f"Balanced Dynamics: GEX ({total_gex:+.2f}B) | Skew ({skew:+.1f}%) | ML ({ml_change:+.1f} pts). Centered ATM Double Diagonal.",
                    "ml_forecast": ml_change
                })
            
        return candidates

    def generate_trade_setup(self, spot: float, candidate: Dict) -> Dict:
        strategy = candidate["strategy"]
        trade_id = f"AUTO-{strategy}-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:4]}"
        
        return {
            "trade_id": trade_id,
            "strategy": strategy,
            "entry_date": datetime.now().strftime("%Y-%m-%d"),
            "entry_time": datetime.now().strftime("%H:%M IST"),
            "spot": spot,
            "strikes": candidate["strikes"],
            "reasoning": candidate["reasoning"],
            "ml_forecast_pts": candidate["ml_forecast"],
            "status": "PENDING_EXECUTION"
        }

    def simulate_payoff(self, proposed_setup: Dict, active_portfolio: List[Dict], chain: List) -> float:
        import numpy as np
        
        # 1. Gather all existing legs and locked profit
        all_legs = []
        locked_pnl = 0.0
        for strat in active_portfolio:
            if strat.get("status") == "OPEN":
                all_legs.extend(strat.get("legs", []))
                locked_pnl += strat.get("realized_pnl", 0.0)
                
        # 2. Mock construct the proposed legs
        strategy = proposed_setup["strategy"]
        strikes = proposed_setup["strikes"]
        spot = proposed_setup["spot"]
        
        # Determine near and far expiry from chain (simple heuristic)
        expiries = sorted(list(set(o.expiry for o in chain)))
        near_exp = expiries[0] if len(expiries) > 0 else "2026-08-04"
        far_exp = expiries[1] if len(expiries) > 1 else near_exp
        
        proposed_legs = []
        if strategy == "DOUBLE_DIAGONAL" and strikes:
            proposed_legs = [
                {"type": "SELL_CALL", "strike": strikes[0], "qty": 1, "expiry": near_exp},
                {"type": "SELL_PUT", "strike": strikes[0], "qty": 1, "expiry": near_exp},
                {"type": "BUY_CALL", "strike": strikes[0] + 100, "qty": 1, "expiry": far_exp},
                {"type": "BUY_PUT", "strike": strikes[0] - 100, "qty": 1, "expiry": far_exp}
            ]
        elif strategy == "UPSIDE_CALL_CALENDAR" and len(strikes) == 2:
            proposed_legs = [
                {"type": "SELL_CALL", "strike": strikes[0], "qty": 1, "expiry": near_exp},
                {"type": "BUY_CALL", "strike": strikes[1], "qty": 2, "expiry": far_exp}
            ]
        elif strategy == "DOWNSIDE_PUT_CALENDAR" and len(strikes) == 2:
            proposed_legs = [
                {"type": "SELL_PUT", "strike": strikes[0], "qty": 1, "expiry": near_exp},
                {"type": "BUY_PUT", "strike": strikes[1], "qty": 2, "expiry": far_exp}
            ]
        elif strategy == "BULL_CALL_SPREAD" and len(strikes) == 2:
            proposed_legs = [
                {"type": "BUY_CALL", "strike": strikes[0], "qty": 1, "expiry": near_exp},
                {"type": "SELL_CALL", "strike": strikes[1], "qty": 1, "expiry": near_exp}
            ]
        elif strategy == "PUT_CREDIT_SPREAD" and len(strikes) == 2:
            proposed_legs = [
                {"type": "SELL_PUT", "strike": strikes[0], "qty": 1, "expiry": near_exp},
                {"type": "BUY_PUT", "strike": strikes[1], "qty": 1, "expiry": near_exp}
            ]
        elif strategy == "SHORT_IRON_CONDOR" and len(strikes) == 4:
            proposed_legs = [
                {"type": "BUY_PUT", "strike": strikes[0], "qty": 1, "expiry": near_exp},
                {"type": "SELL_PUT", "strike": strikes[1], "qty": 1, "expiry": near_exp},
                {"type": "SELL_CALL", "strike": strikes[2], "qty": 1, "expiry": near_exp},
                {"type": "BUY_CALL", "strike": strikes[3], "qty": 1, "expiry": near_exp}
            ]
            
        # Get prices for proposed legs
        for leg in proposed_legs:
            matching = next((o for o in chain if o.strike == leg['strike'] and o.expiry == leg['expiry']), None)
            if matching:
                leg['entry_price'] = matching.call_price if "CALL" in leg['type'] else matching.put_price
            else:
                leg['entry_price'] = 50.0 # fallback
                
        all_legs.extend(proposed_legs)
        
        # 3. Simulate Expiry P&L
        sim_spots = np.arange(spot - 300, spot + 300, 10)
        min_pnl = float('inf')
        
        for s in sim_spots:
            pnl = 0.0
            for leg in all_legs:
                price = leg.get('entry_price', 0)
                if price is None: price = 0
                
                intrinsic = 0.0
                if "CALL" in leg['type']:
                    intrinsic = max(0, s - leg['strike'])
                else:
                    intrinsic = max(0, leg['strike'] - s)
                    
                if "BUY" in leg['type']:
                    leg_pnl = (intrinsic - price) * 65 * leg.get('qty', 1)
                else:
                    leg_pnl = (price - intrinsic) * 65 * leg.get('qty', 1)
                    
                pnl += leg_pnl
            
            if pnl < min_pnl:
                min_pnl = pnl
                
        return min_pnl + locked_pnl


    def run_cycle(self):
        print(f"\n{'='*60}")
        print(f"[AI] ULTIMATE STRATEGY ENGINE | {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")
        
        # Load active portfolio to be Portfolio-Aware
        active_strategies = []
        portfolio = []
        active_file = self.project_root / "market_data" / "active_strategy_state.json"
        if active_file.exists():
            try:
                with open(active_file, 'r') as f:
                    data = json.load(f)
                    portfolio = [data] if isinstance(data, dict) else data
                    active_strategies = [t.get("strategy") for t in portfolio if t.get("status") == "OPEN"]
            except Exception:
                pass
                
        if active_strategies:
            print(f"[PORTFOLIO AWARENESS] Currently active strategies: {', '.join(active_strategies)}")
        
        spot_data = self.fetcher.get_nifty_spot()
        spot = spot_data.price
        print(f"Current Nifty Spot: {spot}")
        
        mt_data = self.fetch_multi_timeframe_data()
        tech_data = self.analyze_technicals(mt_data, spot)
        intel_data = self.get_market_intel(spot)
        ml_data = self.get_ml_forecast(mt_data["1h"])
        
        candidates = self.strategy_selection_matrix(spot, tech_data, intel_data, ml_data, active_strategies)
        
        if not candidates:
            print("\n[WAIT] Engine standing by. No directional or structural trigger found.")
            return

        print(f"\n[OPTIMIZATION MATRIX] Testing {len(candidates)} Candidate Structures...")
        
        best_setup = None
        best_pnl_floor = -float('inf')
        
        for candidate in candidates:
            # Skip if we already hold the core strategy to avoid duplicate layering
            base_strat = candidate["strategy"].replace("UPSIDE_", "").replace("DOWNSIDE_", "")
            if any(base_strat in s for s in active_strategies):
                print(f"  -> Skipping {candidate['strategy']} (Already active in portfolio)")
                continue
                
            setup = self.generate_trade_setup(spot, candidate)
            min_valley_pnl = self.simulate_payoff(setup, portfolio, intel_data["chain"])
            
            print(f"  -> {candidate['strategy']} | Projected P&L Floor: ₹{min_valley_pnl:.2f}")
            
            if min_valley_pnl > best_pnl_floor:
                best_pnl_floor = min_valley_pnl
                best_setup = setup
                
        # Determine safety floor: -100 for adjustments, -5000 for initial setups
        safety_floor = -100 if active_strategies else -5000
        
        if best_setup and best_pnl_floor >= safety_floor:
            print(f"\n✅ [SAFE] Matrix Selected: {best_setup['strategy']} (Winning Floor: ₹{best_pnl_floor:.2f})")
            target_file = self.brain_dir / "proposed_strategy.json"
            with open(target_file, "w") as f:
                json.dump(best_setup, f, indent=4)
            print(f"[OK] Trade Setup Generated and saved to {target_file.name}")
            print("Ready for autonomous execution.")
        else:
            print(f"\n🛑 [VETO] All candidates failed safety check. Best floor was ₹{best_pnl_floor:.2f}.")
            print("Strategy reverted to WAIT to protect capital.")


if __name__ == "__main__":
    engine = UltimateStrategyEngine()
    engine.run_cycle()
