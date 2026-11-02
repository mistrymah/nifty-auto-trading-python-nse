import sys
sys.stdout.reconfigure(encoding='utf-8')
import time
import json
import uuid
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from fetcher import UpstoxFetcher
from ultimate_adjustment_engine import DynamicAdjustmentEngine
# from trader import NiftyTrader # Uncomment if using actual Upstox execution

class UltimateExecutionManager:
    def __init__(self):
        self.fetcher = UpstoxFetcher()
        self.adj_engine = DynamicAdjustmentEngine()
        self.project_root = Path(__file__).parent
        self.brain_dir = Path(r"C:\Users\admin\.gemini\antigravity\brain\f70b260d-7743-49b0-86a6-f817d402d052")
        
        self.proposed_file = self.brain_dir / "proposed_strategy.json"
        self.active_file = self.project_root / "market_data" / "active_strategy_state.json"
        self.logs_dir = self.project_root / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        self.PAPER_TRADE_MODE = True  # Set to False to route real orders via trader.py
        
    def check_for_new_setup(self) -> Optional[Dict]:
        """Check if a new proposed strategy exists and hasn't been executed yet.
        Only the M‑Valley (DOUBLE_DIAGONAL) strategy is permitted.
        Any other strategy will be rejected at this stage to enforce rule compliance.
        """
        if not self.proposed_file.exists():
            return None
            
        try:
            with open(self.proposed_file, 'r') as f:
                setup = json.load(f)
                
            if setup.get("status") == "PENDING_EXECUTION":
                strat = setup.get("strategy", "").upper()
                if strat not in ["DOUBLE_DIAGONAL", "PORTFOLIO_SQUARE_OFF"]:
                    print(f"⚠️ [M-VALLEY STRICT] Rejecting non-M strategy '{strat}'. This engine is strictly locked to M-Valley (DOUBLE_DIAGONAL).")
                    setup["status"] = "REJECTED_NON_M_VALLEY"
                    try:
                        with open(self.proposed_file, 'w') as f_out:
                            json.dump(setup, f_out, indent=4)
                    except Exception:
                        pass
                    return None
                return setup
        except Exception as e:
            print(f"Error reading proposed setup: {e}")
            
        return None

    def construct_legs(self, strategy: str, strikes: List[int], spot: float) -> List[Dict]:
        """Convert a strategy name and key strikes into executable legs."""
        legs = []
        
        # Helper logic to get near and next expiry strings would go here.
        near_exp = self.fetcher.get_nearest_expiry()
        far_exp = self.fetcher.get_next_expiry()
        
        if strategy == "DOUBLE_DIAGONAL":
            # Strikes usually has 1 value: the center ATM strike
            center = strikes[0] if strikes else int(round(spot/50)*50)
            legs = [
                {"type": "SELL_CALL", "strike": center, "expiry": near_exp, "qty": 1},
                {"type": "SELL_PUT", "strike": center, "expiry": near_exp, "qty": 1},
                {"type": "BUY_CALL", "strike": center + 100, "expiry": far_exp, "qty": 1},
                {"type": "BUY_PUT", "strike": center - 100, "expiry": far_exp, "qty": 1}
            ]
            
        elif strategy == "SHORT_IRON_CONDOR":
            # Strikes expected: [Lower_Wing, Lower_Sell, Upper_Sell, Upper_Wing]
            if len(strikes) == 4:
                legs = [
                    {"type": "BUY_PUT", "strike": strikes[0], "expiry": near_exp, "qty": 1},
                    {"type": "SELL_PUT", "strike": strikes[1], "expiry": near_exp, "qty": 1},
                    {"type": "SELL_CALL", "strike": strikes[2], "expiry": near_exp, "qty": 1},
                    {"type": "BUY_CALL", "strike": strikes[3], "expiry": near_exp, "qty": 1}
                ]
                
        elif strategy == "BULL_CALL_SPREAD":
            # Strikes expected: [Long_Call, Short_Call]
            if len(strikes) == 2:
                legs = [
                    {"type": "BUY_CALL", "strike": strikes[0], "expiry": near_exp, "qty": 1},
                    {"type": "SELL_CALL", "strike": strikes[1], "expiry": near_exp, "qty": 1}
                ]
                
        elif strategy == "BEAR_PUT_SPREAD":
            # Strikes expected: [Long_Put, Short_Put]
            if len(strikes) == 2:
                legs = [
                    {"type": "BUY_PUT", "strike": strikes[0], "expiry": near_exp, "qty": 1},
                    {"type": "SELL_PUT", "strike": strikes[1], "expiry": near_exp, "qty": 1}
                ]
                
        elif strategy in ["PUT_CREDIT_SPREAD", "CALL_CREDIT_SPREAD"]:
            # Strikes expected: [Short_Strike, Long_Wing]
            opt_type = "PUT" if "PUT" in strategy else "CALL"
            if len(strikes) == 2:
                legs = [
                    {"type": f"SELL_{opt_type}", "strike": strikes[0], "expiry": near_exp, "qty": 1},
                    {"type": f"BUY_{opt_type}", "strike": strikes[1], "expiry": near_exp, "qty": 1}
                ]
        elif strategy == "UPSIDE_CALL_CALENDAR":
            # Strikes expected: [Near_Strike, Far_Strike]
            if len(strikes) == 2:
                legs = [
                    {"type": "SELL_CALL", "strike": strikes[0], "expiry": near_exp, "qty": 1},
                    {"type": "BUY_CALL", "strike": strikes[1], "expiry": far_exp, "qty": 2}  # Ratio 1:2 for W-shape convexity
                ]
                
        elif strategy == "DOWNSIDE_PUT_CALENDAR":
            # Strikes expected: [Near_Strike, Far_Strike]
            if len(strikes) == 2:
                legs = [
                    {"type": "SELL_PUT", "strike": strikes[0], "expiry": near_exp, "qty": 1},
                    {"type": "BUY_PUT", "strike": strikes[1], "expiry": far_exp, "qty": 2}  # Ratio 1:2 for W-shape convexity
                ]
                
        return legs

    def execute_setup(self, setup: Dict):
        print(f"\n[EXECUTION] Initiating {setup['strategy']}...")
        
        if setup['strategy'] == "PORTFOLIO_SQUARE_OFF":
            print("[SUCCESS] Portfolio Square-Off signal acknowledged. Active portfolios will be flushed.")
            if self.proposed_file.exists():
                os.remove(self.proposed_file)
            return
            
        legs = self.construct_legs(setup["strategy"], setup.get("strikes", []), setup.get("spot", 0.0))
        
        if not legs:
            print("[ERROR] Could not construct legs for strategy.")
            return
            
        # Fetch live chains for all unique expiries to resolve real prices
        unique_expiries = list(set(leg.get('expiry', '2026-09-15') for leg in legs))
        chain = []
        for exp in unique_expiries:
            chain.extend(self.fetcher.get_option_chain(expiry=exp))

        # Sort legs: BUY first, then SELL for margin benefit
        legs.sort(key=lambda x: 0 if "BUY" in x['type'] else 1)
        
        print(f"Legs to execute (Sorted for Margin Benefit):")
        for leg in legs:
            matching_opt = next((o for o in chain if o.strike == leg['strike'] and o.expiry == leg.get('expiry', '2026-09-15')), None)
            display_expiry = matching_opt.expiry if matching_opt else leg.get('expiry', '2026-09-15')
            print(f"  -> {leg['type']} | {leg['strike']} | {display_expiry} | Qty: {leg['qty']}")
            
        executed_legs = []
        total_entry_debit = 0.0
        
        if self.PAPER_TRADE_MODE:
            # Simulate execution using the fetched multi-expiry chain
            for leg in legs:
                # Find matching strike and expiry in combined chain
                matching_opt = next((o for o in chain if o.strike == leg['strike'] and o.expiry == leg.get('expiry', '2026-09-15')), None)
                if matching_opt:
                    price = matching_opt.call_price if "CALL" in leg['type'] else matching_opt.put_price
                    price = price or 50.0  # fallback
                    
                    if "BUY" in leg['type']:
                        total_entry_debit -= price * 65 * leg['qty'] # Assuming lot size 65
                    else:
                        total_entry_debit += price * 65 * leg['qty']
                        
                    executed_legs.append({
                        "type": leg['type'],
                        "strike": leg['strike'],
                        "entry_price": float(price),
                        "qty": leg['qty'],
                        "expiry": matching_opt.expiry
                    })

            if len(executed_legs) != len(legs):
                print(f"[REJECTED] Incomplete leg execution! Expected {len(legs)} legs, but only priced {len(executed_legs)}. Aborting entry to prevent unhedged exposure.")
                return
        else:
            # REAL EXECUTION LOGIC GOES HERE (Calling trader.enter_trade)
            pass

        # Save to active state portfolio
        active_portfolio = []
        if self.active_file.exists():
            try:
                with open(self.active_file, 'r') as f:
                    data = json.load(f)
                    active_portfolio = [data] if isinstance(data, dict) else data
            except json.JSONDecodeError:
                # File exists but is empty or corrupted, start fresh
                active_portfolio = []
                
        # CORE ARCHITECTURE SHIFT: 
        # Check if we already have an OPEN structural trade (like DOUBLE_DIAGONAL).
        # If we do, fold the new legs into that trade as an adjustment layer!
        open_strategy = next((s for s in active_portfolio if s.get("status") == "OPEN"), None)
        
        # Dynamic Target based on Days To Expiry (DTE)
        # Full cycle (DTE >= 4 days): 85% of debit, with minimum floor ₹3,000
        # Late cycle (DTE 2-3 days): 60%
        # Expiry approach (DTE <= 1 day): 35%
        near_expiry_str = min((leg.get('expiry', '2099-12-31') for leg in executed_legs), default='2099-12-31')
        try:
            dte = (datetime.strptime(near_expiry_str, '%Y-%m-%d').date() - datetime.now().date()).days
        except Exception:
            dte = 5
            
        if dte >= 4:
            target_mult = 0.85
            min_target_floor = 3000.0
        elif dte >= 2:
            target_mult = 0.60
            min_target_floor = 2200.0
        else:
            target_mult = 0.35
            min_target_floor = 1500.0
        
        if open_strategy:
            open_strategy["legs"].extend(executed_legs)
            open_strategy["net_credit_debit"] += total_entry_debit
            
            # Recalculate Max Target to scale dynamically with newly deployed capital
            base_target = abs(open_strategy["net_credit_debit"]) * target_mult
            open_strategy["max_target"] = max(base_target, min_target_floor * 1.5)
            
            # Label the core strategy with the new layers appended
            if setup["strategy"] not in open_strategy["strategy"]:
                open_strategy["strategy"] += f" + {setup['strategy']}"
                
            print(f"[SUCCESS] Adjustment Legs folded into {open_strategy['trade_id']}. Target increased to ₹{open_strategy['max_target']:.2f}")
        else:
            active_state = {
                "trade_id": setup["trade_id"],
                "strategy": setup["strategy"],
                "entry_time": datetime.now().isoformat(),
                "status": "OPEN",
                "net_credit_debit": total_entry_debit,
                "max_target": max(abs(total_entry_debit) * target_mult, min_target_floor),
                "max_loss": None, # Dynamic / Managed by Adjustment Protocol
                "legs": executed_legs,
                "reasoning": setup["reasoning"]
            }
            active_portfolio.append(active_state)
            print(f"[SUCCESS] Core Strategy executed. Net Entry Premium: ₹{total_entry_debit:.2f}")
            
        with open(self.active_file, 'w') as f:
            json.dump(active_portfolio, f, indent=4)
            
        # Delete proposed file so we don't execute it again
        if self.proposed_file.exists():
            try:
                self.proposed_file.unlink()
            except Exception:
                pass

    def monitor_active_trades(self):
        """Monitor all open trades in the portfolio for P&L, stop loss, targets, and adjustments."""
        if not self.active_file.exists():
            return
            
        try:
            with open(self.active_file, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    return # Empty or corrupted file, nothing to monitor
                    
            # Migration logic for old dict format
            if isinstance(data, dict):
                active_portfolio = [data]
            else:
                active_portfolio = data
                
            open_trades = [t for t in active_portfolio if t.get("status") == "OPEN"]
            if not open_trades:
                return
                
            updated_portfolio = []
            
            for active in open_trades:
                print(f"\n" + "="*65)
                print(f"[PORTFOLIO] Active Strategy: {active['strategy']}")
                print(f"ID: {active['trade_id']}")
                print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
                print("=" * 65)
                
                # ---> RUN ADJUSTMENT ENGINE <---
                adjusted_state = self.adj_engine.evaluate_active_trade(active, active.get("realized_pnl", 0.0))
                if adjusted_state:
                    active = adjusted_state
                    
                    if active.get("status") == "EXECUTE_FULL_ROLL":
                        print(f"\n🚀 [FULL STRUCTURE ROLL] {active['strategy']} is being rolled to new ATM!")
                        
                        # Realize P&L on current legs before rolling
                        unique_expiries_for_close = list(set(leg.get('expiry', '2026-09-15') for leg in active["legs"]))
                        temp_chain = []
                        for exp in unique_expiries_for_close:
                            temp_chain.extend(self.fetcher.get_option_chain(expiry=exp))
                            
                        realized_loss = 0.0
                        for leg in active["legs"]:
                            matching_opt = next((o for o in temp_chain if o.strike == leg['strike'] and o.expiry == leg.get('expiry', '2026-09-15')), None)
                            current_price = leg["entry_price"]
                            if matching_opt:
                                current_price = matching_opt.call_price if "CALL" in leg['type'] else matching_opt.put_price
                                current_price = current_price or leg["entry_price"]
                                
                            if "BUY" in leg['type']:
                                pnl = (current_price - leg["entry_price"]) * 65 * leg['qty']
                            else:
                                pnl = (leg["entry_price"] - current_price) * 65 * leg['qty']
                                
                            realized_loss += pnl
                            print(f"   [-] Closed {leg['type']} {leg['strike']} @ {current_price:.2f} (P&L: ₹{pnl:.2f})")
                            
                        active["realized_pnl"] = active.get("realized_pnl", 0.0) + realized_loss
                        
                        # Apply new legs
                        old_leg_count = len(active["legs"])
                        active["legs"] = active.pop("roll_legs")
                        active["status"] = "OPEN"
                        
                        # Update Brokerage counter (closing old legs + opening new ones)
                        active["total_orders"] = active.get("total_orders", old_leg_count) + old_leg_count + len(active["legs"])
                        
                        # Recalculate Net Debit
                        net_debit = sum([l['entry_price'] * 65 * l['qty'] * (-1 if 'BUY' in l['type'] else 1) for l in active["legs"]])
                        active["net_credit_debit"] = net_debit
                        
                        print(f"   [+] Opened New Legs. Total Realized P&L so far: ₹{active['realized_pnl']:.2f}")
                        continue
                        
                # Fetch live prices by combining chains for all required expiries
                unique_expiries = list(set(leg.get('expiry', '2026-09-15') for leg in active["legs"]))
                chain = []
                for exp in unique_expiries:
                    chain.extend(self.fetcher.get_option_chain(expiry=exp))
                    
                spot_data = self.fetcher.get_nifty_spot()
                if spot_data.price > 0:
                    self._last_valid_spot = spot_data.price
                elif hasattr(self, "_last_valid_spot") and self._last_valid_spot > 0:
                    spot_data.price = self._last_valid_spot
                # Auto-heal target if trade was seeded with stale 25% Monday multiplier
                near_expiry_str = min((leg.get('expiry', '2099-12-31') for leg in active["legs"]), default='2099-12-31')
                try:
                    dte_active = (datetime.strptime(near_expiry_str, '%Y-%m-%d').date() - datetime.now().date()).days
                except Exception:
                    dte_active = 5
                if dte_active >= 3 and active.get("max_target", 0.0) < 2500.0:
                    net_debit = abs(active.get("net_credit_debit", 3432.0))
                    active["max_target"] = max(net_debit * 0.85, 3000.0)
                    print(f"🎯 [TARGET RECALIBRATED] DTE={dte_active} days. Recalibrated target to ₹{active['max_target']:.2f}")

                current_pnl = active.get("realized_pnl", 0.0)
                
                # Per‑leg table removed – aggregated P&L shown in the weekly ledger
                # Per‑leg table header removed – aggregated view shown in weekly ledger
                print("-" * 65)
                
                surviving_legs = []
                
                for leg in active["legs"]:
                    matching_opt = next((o for o in chain if o.strike == leg['strike'] and o.expiry == leg.get('expiry', '2026-09-15')), None)
                    
                    if matching_opt:
                        is_call = "CALL" in leg['type'] or "CE" in leg['type']
                        current_price = matching_opt.call_price if is_call else matching_opt.put_price
                        current_price = current_price or leg["entry_price"]
                        
                        # ---> THETA PRUNING LOGIC (JUICE EXTRACTION & ROLLING) <---
                        if "SELL" in leg['type'] and current_price <= (leg['entry_price'] * 0.20):
                            harvested_profit = (leg["entry_price"] - current_price) * 65 * leg['qty']
                            active["realized_pnl"] = active.get("realized_pnl", 0.0) + harvested_profit
                            current_pnl += harvested_profit
                            print(f"🔪 [THETA PRUNING] Closed {leg['type']} {leg['strike']} @ {current_price:.2f} (Locked: ₹{harvested_profit:.2f})")
                            
                            # ROLL FORWARD LOGIC
                            target_premium = leg['entry_price'] * 0.50
                            best_opt = None
                            min_diff = float('inf')
                            
                            for opt in chain:
                                if opt.expiry == leg.get('expiry', '2026-09-15'):
                                    if is_call and opt.strike > spot_data.price and opt.strike < leg['strike']:
                                        price = opt.call_price or 0
                                        if price > 0 and abs(price - target_premium) < min_diff:
                                            min_diff = abs(price - target_premium)
                                            best_opt = opt
                                    elif not is_call and opt.strike < spot_data.price and opt.strike > leg['strike']:
                                        price = opt.put_price or 0
                                        if price > 0 and abs(price - target_premium) < min_diff:
                                            min_diff = abs(price - target_premium)
                                            best_opt = opt
                            
                            if best_opt:
                                new_price = best_opt.call_price if is_call else best_opt.put_price
                                new_leg = {
                                    "type": leg['type'],
                                    "strike": best_opt.strike,
                                    "expiry": leg.get('expiry', 'UNKNOWN'),
                                    "entry_price": new_price or target_premium,
                                    "qty": leg['qty']
                                }
                                surviving_legs.append(new_leg)
                                active["net_credit_debit"] += new_price
                                active["total_orders"] = active.get("total_orders", len(active["legs"])) + 2 # 1 close, 1 new open
                                print(f"🔄 [ROLL FORWARD] Sold new {new_leg['type']} at {new_leg['strike']} to collect ₹{new_price:.2f} more juice!")
                                
                            continue # Old leg is pruned, do not add it back
                            
                        if "BUY" in leg['type']:
                            pnl = (current_price - leg["entry_price"]) * 65 * leg['qty']
                        else:
                            pnl = (leg["entry_price"] - current_price) * 65 * leg['qty']
                            
                        current_pnl += pnl
                        surviving_legs.append(leg)
                        
                        pnl_str = f"₹{pnl:+.2f}"
                        # Use actual expiry date if we found it, or what is stored in the leg
                        expiry_str = matching_opt.expiry
                        leg['expiry'] = expiry_str # update memory
                        # Store live LTP & per-leg P&L so the grouped ledger can display them
                        leg['current_ltp'] = current_price
                        leg['current_pnl'] = pnl
                        
                active["legs"] = surviving_legs
                active["total_orders"] = active.get("total_orders", len(active["legs"]))
                net_pnl = self.print_weekly_campaign_ledger(active, current_pnl)
                
                # ---> TARGET SECURED LOGIC (Using NET P&L) <---
                if net_pnl >= active.get("max_target", float('inf')):
                    print(f"🎯 [PROFIT SECURED] Target of ₹{active.get('max_target'):.2f} reached! Squaring off entire portfolio. Final Net P&L: ₹{net_pnl:.2f}")
                    self.close_active_trade(active, net_pnl, "MAX_TARGET_REACHED")
                    continue
                
                # ---> STOP LOSS LOGIC (Using NET P&L) <---
                max_loss_limit = active.get("max_loss")
                if max_loss_limit is not None and net_pnl <= max_loss_limit:
                    print(f"🛑 [STOP LOSS TRIGGERED] Net Loss of ₹{net_pnl:.2f} hit Risk Limit (₹{max_loss_limit:.2f})! Closing campaign.")
                    self.close_active_trade(active, net_pnl, "MAX_LOSS_STOPPED_OUT")
                    continue
                    
                # ---> SYSTEM PURGE FOR NON-M (LEGACY / CREDIT / STRADDLE) TRADES <---
                is_m_shape = "DOUBLE_DIAGONAL" in active.get("strategy", "").upper() or "CALENDAR" in active.get("strategy", "").upper()
                if active.get("status") == "FORCE_CLOSE_LEGACY" or not is_m_shape:
                    print(f"⚠️ [SYSTEM PURGE] Non-M Strategy '{active.get('strategy')}' detected in M-Valley Engine. Purging to clear portfolio for Double Diagonal. Final P&L: ₹{current_pnl:.2f}")
                    self.close_active_trade(active, current_pnl, "PURGE_NON_M_STRATEGY")
                    continue
                
                # Expiry Day Closure Protocol (The Ultimate Motive)
                now = datetime.now()
                today_date = now.date()
                
                try:
                    if not active.get("legs"):
                        print("[EXIT] Empty strategy legs detected! Flushing from active portfolio.")
                        self.close_active_trade(active, current_pnl, "EMPTY_LEGS_FLUSH")
                        continue
                        
                    near_expiry_str = min(leg.get('expiry', '2099-12-31') for leg in active["legs"])
                    if near_expiry_str != '2099-12-31':
                        near_exp_date = datetime.strptime(near_expiry_str, '%Y-%m-%d').date()
                        
                        if today_date > near_exp_date:
                            print(f"[EXIT] Zombie Trade Detected! Near expiry ({near_expiry_str}) has passed. Flushing from active portfolio.")
                            self.close_active_trade(active, current_pnl, "POST_EXPIRY_FLUSH")
                            continue
                        elif today_date == near_exp_date:
                            if now.hour >= 15 and now.minute >= 10:
                                print(f"[EXIT] Mandatory Expiry Square-Off (15:10)! Closing campaign to prevent settlement risk. Final P&L: ₹{current_pnl:.2f}")
                                self.close_active_trade(active, current_pnl, "EXPIRY_SQUARE_OFF")
                                continue
                except Exception as e:
                    print(f"Expiry parse error: {e}")
                        
                updated_portfolio.append(active)
                    
            # Save the updated portfolio back to file
            with open(self.active_file, 'w') as f:
                json.dump(updated_portfolio, f, indent=4)
                
        except Exception as e:
            print(f"Error monitoring trades: {e}")

    def get_campaign_window(self):
        """
        Calculates the Tuesday-to-Tuesday weekly campaign cycle window.
        A weekly campaign runs from Tuesday EOD (15:30 PM, when the new trade is seeded)
        until the following Tuesday 15:10 PM (when it is squared off for expiry).
        
        Trades closed by Tuesday EOD belong to the previous week's expired cycle.
        From Wednesday onwards, the ledger ONLY includes the new trade opened on Tuesday EOD
        and any subsequent adjustments for next week's expiry.
        """
        now = datetime.now()
        # Weekday 1 is Tuesday
        if now.weekday() == 1:
            if now.hour < 15 or (now.hour == 15 and now.minute < 10):
                # Tuesday morning/intraday before 15:10: Expiring cycle that started last Tuesday at 15:30
                start_dt = (now - timedelta(days=7)).replace(hour=15, minute=30, second=0, microsecond=0)
                end_dt = now.replace(hour=15, minute=10, second=0, microsecond=0)
            else:
                # Tuesday post-expiry (after 15:10/15:30): New weekly cycle for NEXT Tuesday
                start_dt = now.replace(hour=15, minute=30, second=0, microsecond=0)
                end_dt = (start_dt + timedelta(days=7)).replace(hour=15, minute=10, second=0, microsecond=0)
        else:
            # Wednesday (2), Thursday (3), Friday (4), Sat (5), Sun (6), Monday (0)
            days_back = (now.weekday() - 1) % 7
            start_dt = (now - timedelta(days=days_back)).replace(hour=15, minute=30, second=0, microsecond=0)
            end_dt = (start_dt + timedelta(days=7)).replace(hour=15, minute=10, second=0, microsecond=0)
            
        return start_dt, end_dt

    def print_weekly_campaign_ledger(self, active: Dict, current_running_pnl: float) -> float:
        start_dt, end_dt = self.get_campaign_window()
        cycle_str = f"{start_dt.strftime('%Y-%m-%d %H:%M')} to {end_dt.strftime('%Y-%m-%d %H:%M')}"
        
        # Load all closed trades within this campaign cycle
        log_file = self.logs_dir / "ultimate_trades.json"
        closed_trades = []
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    all_history = json.load(f)
                    for t in all_history:
                        try:
                            t_entry = datetime.fromisoformat(t.get("entry_time", "2000-01-01"))
                            t_exit = datetime.fromisoformat(t.get("exit_time", "2000-01-01")) if t.get("exit_time") else None
                            
                            # Exclude any trades that ended by Tuesday EOD of the cycle start date
                            if t_exit and t_exit <= start_dt:
                                continue

                            # Exclude purged non-M intruder strategies (e.g. strangles from legacy scripts)
                            if t.get("exit_reason") == "PURGE_NON_M_STRATEGY" or t.get("strategy") not in ["DOUBLE_DIAGONAL", "DOUBLE_DIAGONAL_M_SHAPE", "CALENDAR_M_SHAPE"]:
                                continue
                                
                            if start_dt <= t_entry <= end_dt:
                                closed_trades.append(t)
                        except Exception:
                            continue
            except Exception as e:
                print(f"[CAMPAIGN] Ledger read error: {e}")

        # Compute totals
        total_closed_pnl = sum(t.get("final_pnl", 0.0) for t in closed_trades)
        total_closed_orders = sum(t.get("total_orders", len(t.get("legs", []))) for t in closed_trades)
        
        active_orders = max(active.get("total_orders", 0), len(active.get("legs", [])))
        total_cycle_orders = total_closed_orders + active_orders
        
        brokerage_per_order = 25.0  # ₹20 brokerage + ₹5 STT/Exchange charges
        total_cycle_brokerage = total_cycle_orders * brokerage_per_order
        
        active_brokerage = active_orders * brokerage_per_order
        active_net_pnl = current_running_pnl - active_brokerage
        
        campaign_gross_pnl = total_closed_pnl + current_running_pnl
        campaign_net_pnl = campaign_gross_pnl - total_cycle_brokerage
        
        W = 90
        LEG_COL = f"  {'TYPE':<12} | {'STRIKE':<7} | {'EXPIRY':<12} | {'ENTRY':>8} | {'EXIT/LTP':>8} | {'P&L':>10}"
        LEG_SEP = "  " + "-" * 72

        print("\n" + "=" * W)
        print(f"📅 WEEKLY CAMPAIGN LEDGER  (Tue→Tue Cycle: {cycle_str})")
        print("=" * W)

        # ── Closed trades – grouped view ──────────────────────────────────
        for idx, t in enumerate(closed_trades, 1):
            try:
                e_time = datetime.fromisoformat(t.get('entry_time')).strftime('%Y-%m-%d %H:%M')
            except Exception:
                e_time = str(t.get('entry_time', 'N/A'))[:16]
            try:
                x_time = datetime.fromisoformat(t.get('exit_time')).strftime('%Y-%m-%d %H:%M')
            except Exception:
                x_time = str(t.get('exit_time', 'N/A'))[:16]

            pnl_val = t.get('final_pnl', 0.0)
            pnl_str = f"₹{pnl_val:+.2f}"
            reason  = str(t.get('exit_reason', 'CLOSED'))
            strat   = t.get('strategy', 'STRATEGY')

            print(f"\nTrade {idx:<3}  {strat}  |  {e_time} → {x_time}  |  CLOSED  |  Total P&L: {pnl_str}  |  {reason}")
            print(LEG_COL)
            print(LEG_SEP)
            for leg in t.get('legs', []):
                print(f"  {leg['type']:<12} | {leg['strike']:<7} | {leg.get('expiry','N/A'):<12} | {leg['entry_price']:>8.2f} | {'  N/A':>8} | {'  N/A':>10}")

        # ── Active trade – grouped view with live LTP & per-leg P&L ───────
        total_idx = len(closed_trades) + 1
        try:
            act_etime = datetime.fromisoformat(active.get('entry_time')).strftime('%Y-%m-%d %H:%M')
        except Exception:
            act_etime = str(active.get('entry_time', 'N/A'))[:16]

        act_pnl_str = f"₹{current_running_pnl:+.2f}"
        act_strat   = active.get('strategy', 'STRATEGY')

        print(f"\nTrade {total_idx:<3}  {act_strat}  |  Entry: {act_etime}  |  ⚡ ACTIVE  |  Running P&L: {act_pnl_str}")
        print(LEG_COL)
        print(LEG_SEP)
        for leg in active.get('legs', []):
            ltp     = leg.get('current_ltp', leg['entry_price'])
            leg_pnl = leg.get('current_pnl', 0.0)
            print(f"  {leg['type']:<12} | {leg['strike']:<7} | {leg.get('expiry','N/A'):<12} | {leg['entry_price']:>8.2f} | {ltp:>8.2f} | {leg_pnl:>+10.2f}")

        # ── Campaign summary ───────────────────────────────────────────────
        print("\n" + "=" * W)
        print(f"📊 WEEKLY CAMPAIGN SUMMARY")
        print("=" * W)
        print(f"Total Trades           : {total_idx}  ({len(closed_trades)} Closed, 1 Active)")
        if closed_trades:
            print(f"Closed Trades P&L      : ₹{total_closed_pnl:+.2f}")
        if active.get("realized_pnl", 0.0) > 0:
            print(f"Active Locked Profit   : ₹{active['realized_pnl']:+.2f}  (theta locked)")
        print(f"Active Running P&L     : ₹{current_running_pnl:+.2f}  (Net after brokerage: ₹{active_net_pnl:+.2f})")
        print(f"Campaign Gross P&L     : ₹{campaign_gross_pnl:+.2f}")
        print(f"Cycle Brokerage/Taxes  : ₹{-total_cycle_brokerage:.2f}  ({total_cycle_orders} orders @ ₹25/order)")
        risk_str = f"₹{active['max_loss']:.2f}" if active.get("max_loss") is not None else "Dynamic (Adjustment Protected)"
        print(f"Target                 : ₹{active.get('max_target', 0):.2f}  or Expiry 15:10  |  Risk Limit: {risk_str}")
        print("=" * W + "\n")
        

        return active_net_pnl

    def close_active_trade(self, active_state: Dict, final_pnl: float, reason: str):
        active_state["status"] = "CLOSED"
        active_state["exit_time"] = datetime.now().isoformat()
        active_state["final_pnl"] = final_pnl
        active_state["exit_reason"] = reason
        
        # Log to historical ledger
        log_file = self.logs_dir / "ultimate_trades.json"
        history = []
        if log_file.exists():
            with open(log_file, 'r') as f:
                history = json.load(f)
        
        history.append(active_state)
        with open(log_file, 'w') as f:
            json.dump(history, f, indent=4)

    def has_open_trade(self) -> bool:
        if not self.active_file.exists():
            return False
        try:
            with open(self.active_file, 'r') as f:
                data = json.load(f)
                active_portfolio = [data] if isinstance(data, dict) else data
                return any(t.get("status") == "OPEN" for t in active_portfolio)
        except Exception:
            return False

    def run_daemon(self, interval_seconds: int = 60):
        print("\n" + "="*60)
        print("[EXEC] ULTIMATE EXECUTION MANAGER")
        print("Listening for setups & managing layered portfolio...")
        print("="*60 + "\n")
        
        # ── PRE-FLIGHT STARTUP AUDIT BY SMART ADJUSTMENT ANALYST AGENT ───────
        try:
            from adjustment_analyst_agent import (
                AdjustmentAnalystAgent, SituationClassifier, MarketStructureAnalyzer, compute_dte_days
            )
            analyst = AdjustmentAnalystAgent()
            trade = analyst.load_active_trade()
            if trade:
                spot_data = analyst.fetcher.get_nifty_spot()
                spot = spot_data.price if spot_data and spot_data.price > 0 else 22700.0
                chain = analyst.fetcher.get_option_chain() or []
                near_exp = analyst.fetcher.get_nearest_expiry()
                dte_days = compute_dte_days(near_exp)
                
                ms_analyzer = MarketStructureAnalyzer()
                cvd_data = ms_analyzer.compute_cvd(analyst.fetcher)
                oi_data = ms_analyzer.compute_max_pain_and_pcr(chain)
                sr_data = ms_analyzer.compute_cpr_and_sr(analyst.fetcher, spot)
                audit = analyst.audit_trade_structure(trade, spot)
                
                be_lower = audit["be_points"][0] if audit["be_points"] else spot - 200
                be_upper = audit["be_points"][-1] if audit["be_points"] else spot + 200
                put_layer_active = any(l.get("label") == "layer_put" for l in trade.get("legs", []))
                call_layer_active = any(l.get("label") == "layer_call" for l in trade.get("legs", []))
                
                situation = SituationClassifier.classify(
                    spot=spot, be_lower=be_lower, be_upper=be_upper,
                    net_delta=-0.05, ww_band=0.08,
                    put_layer_active=put_layer_active, call_layer_active=call_layer_active,
                    dte_days=dte_days, regime="NORMAL",
                    net_debit_inr=audit["net_debit_inr"], green_pct=audit["green_pct"]
                )
                
                print("="*70)
                print("🛡️  SMART ADJUSTMENT ANALYST — MARKET STRUCTURE & AUDIT SNAPSHOT")
                print(f"   Active Trade : {trade.get('trade_id')} | Strategy: {trade.get('strategy')}")
                print(f"   Nifty Spot   : {spot:.1f} | Near Expiry: {near_exp} (DTE: {dte_days:.1f} days)")
                print(f"   Order Flow   : CVD {cvd_data['cvd']:+,.0f} ({cvd_data['cvd_trend']}) | Volume: {cvd_data['intensity']}")
                print(f"   Option Intel : Max Pain: {oi_data['max_pain']} | PCR: {oi_data['pcr']:.2f} | Put Wall: {oi_data['put_wall']} | Call Wall: {oi_data['call_wall']}")
                print(f"   Daily CPR    : {sr_data['cpr_status']} (Pivot: {sr_data['pivot']:.1f})")
                print(f"   S/R Proximity: Support: {sr_data['nearest_support']:.1f} | Resistance: {sr_data['nearest_resistance']:.1f}")
                print(f"   Situation    : {situation['name']} ➔ Mandated Action: {situation['action']}")
                print(f"   Health State : Net Debit ₹{audit['net_debit_inr']:+.2f} | Positive Expiry Range: {audit['green_pct']}%")
                print("="*70 + "\n")
        except Exception as e:
            print(f"⚠️ [ANALYST STARTUP NOTICE] {e}")

        while True:
            try:
                # Market Hours Enforcement: Prevent pre-market false adjustments
                now = datetime.now()
                if now.hour < 9 or (now.hour == 9 and now.minute < 16):
                    print(f"[{now.strftime('%H:%M:%S')}] Market is PRE-OPEN. Waiting for 09:16 AM for option premiums to settle...")
                    time.sleep(interval_seconds)
                    continue
                elif now.hour >= 15 and now.minute > 30:
                    print(f"[{now.strftime('%H:%M:%S')}] Market is CLOSED. Execution Manager standing by...")
                    time.sleep(interval_seconds)
                    continue

                # 1. Manage ALL existing trades in portfolio
                self.monitor_active_trades()
                
                # 2. Check for new setups and layer them
                setup = self.check_for_new_setup()
                
                # Auto-seed: If portfolio is empty during market hours, seed the M-Valley DOUBLE_DIAGONAL
                if not setup and not self.has_open_trade():
                    print(f"[{now.strftime('%H:%M:%S')}] 🏔️ [EMPTY PORTFOLIO] No active trade found. Seeding M-Valley DOUBLE_DIAGONAL campaign...")
                    from ultimate_strategy_engine import UltimateStrategyEngine
                    engine = UltimateStrategyEngine()
                    engine.run_cycle()
                    setup = self.check_for_new_setup()

                if setup:
                    self.execute_setup(setup)
                        
                time.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                print("\n\n[EXEC_DAEMON] Shutting down.")
                break
            except Exception as e:
                print(f"[ERROR] Exec Manager: {e}")
                time.sleep(interval_seconds)

if __name__ == "__main__":
    manager = UltimateExecutionManager()
    manager.run_daemon()
