# Nifty Trading Agent

## Environment
- Python: `c:\Users\admin\Desktop\Trading_Workspace\Env antigravity\.venv\Scripts\python.exe`
- Config: `.env` file (see `.env.example` for template)
- Keys required: `UPSTOX_ACCESS_TOKEN`, `UPSTOX_API_KEY`, `GLM_API_KEY` (optional), `GROQ_API_KEY` (optional)

## MCP Server
- Entry: `python mcp_server.py` (JSON-RPC over stdin/stdout)
- Tools implemented in `trader.py` via `NiftyTrader` class

## Tools
| Tool | Description |
|------|-------------|
| `get_trades` | Returns all trades from `logs/trades.csv` |
| `get_performance` | Returns total trades, win rate, cumulative P&L |
| `generate_signal` | Generates trading signal with reasoning/confidence |
| `enter_trade` | Enters trade from signal JSON |
| `exit_trade` | Exits trade by ID with exit_price and reason |

## Dynamic Calendar M-Shape Strategy (Portfolio Layering)
- File: `ultimate_adjustment_engine.py` (and legacy `dynamic_calendar_adjuster.py`)
- Seeds a **Double Diagonal** (single mountain), monitors spot for breakeven (valley) triggers.
- Instead of warped ratio spreads, uses **Portfolio Layering** to create a continuous mountain range.
- When spot hits the **Put Trigger (Downside Crash)**, it calculates a `New Center Strike = Current Spot - (Original Center - Current Spot)` and deploys a secondary **PUT Diagonal** centered deep OTM. This ensures the new mountain's slope perfectly overlaps the current valley.
- When spot hits the **Call Trigger (Upside Rally)**, it calculates a `New Center Strike = Current Spot + (Current Spot - Original Center)` and deploys a secondary **CALL Diagonal**.
- This builds the M-Shape (Camel) dynamically while guaranteeing zero whipsaw risk on the opposite side.
- Position size remains 1:1 on the layers to prevent heavy debits.

### Layering Trigger Logic
| Stage | Trigger | Action | Why |
|---|---|---|---|
| **1st Adjustment** | `put_trigger` (Crash) | `LAYER_MOUNTAIN_PUT` | Builds a secondary Put diagonal centered deep OTM to catch the crash, interlocking at the breakeven valley (M-Shape). |
| **1st Adjustment** | `call_trigger` (Rally) | `LAYER_MOUNTAIN_CALL` | Builds a secondary Call diagonal centered deep OTM to catch the rally, interlocking at the breakeven valley (M-Shape). |
| **2nd Adjustment** | `2nd put_trigger` (Crash Continuation) | `EXTEND_RATIO_PUT_WING` | Closes exhausted Call wing (locks accumulated theta) + Rolls Call diagonal down closer to spot + Buys 1 Far Put (1:2 Ratio Convexity) for unlimited crash profit. |
| **2nd Adjustment** | `2nd call_trigger` (Rally Continuation) | `EXTEND_RATIO_CALL_WING` | Closes exhausted Put wing (locks accumulated theta) + Rolls Put diagonal up closer to spot + Buys 1 Far Call (1:2 Ratio Convexity) for unlimited rally profit. |

## Running Agents
```bash
# Adjustment Analyst Agent (Pre-execution audit, simulation, and surgical repair)
python adjustment_analyst_agent.py --audit
python adjustment_analyst_agent.py --simulate
python adjustment_analyst_agent.py --repair

# AI signal agent (60s interval)
python ai_signal_agent.py -i 60

# Pattern + CPR agent (1min candles)
python pattern_cpr_agent.py

# Option seller agent (5min interval)
python option_seller_agent.py

# Agent manager (combines Kronos + CPR + patterns)
python agent_manager.py

# Multi-factor strategy (improved version with direct Kronos + trend + patterns)
python multi_factor_strategy.py
```

## Architecture
- `adjustment_analyst_agent.py`: Autonomous quantitative analyst auditing and vetoing risky adjustments
- `fetcher.py`: Upstox API integration (historical, options, quotes)
- `trader.py`: Core trading logic, signal generation, trade management
- `agent_manager.py`: Multi-factor decision engine (Kronos + CPR + patterns)
- `multi_factor_strategy.py`: Enhanced multi-factor strategy (direct Kronos + CPR + patterns + trend)
- `kronos_integration.py`: Time series forecasting with Kronos AI
- `pattern_recognition.py`: Candlestick and chart pattern detection

## CPR Calculation
- Uses **YESTERDAY's daily OHLC data**
- Pivot = (High + Low + Close) / 3
- R1 = 2*P - Low, S1 = 2*P - High
- R2 = P + (High - Low), S2 = P - (High - Low)

## Trading Logic
- Option selling (SELL_PUT/SELL_CALL) targets price decay
- Target = entry_price × 0.5, Stop Loss = entry_price × 1.5
- Trailing stop: 10% from peak

## Data Files
- Trades: `logs/trades.csv`
- Signals: `logs/signals.csv`
- Market data: `market_data/`

## API Integration
- Upstox: `fetcher.get_nifty_historical()`, `fetcher.get_option_chain()`, `fetcher.get_nifty_spot()`
- Kronos: `NiftyKronosPredictor.forecast(df, lookback, pred_len)`

## Double-Expiry Camel Strategy Rules
- **Payoff Curve Shape**: Must always be a **W-shape** at expiry, with the outer wings rising to positive infinity (requiring a net long option ratio: $\text{Quantity}_{\text{Far}} > \text{Quantity}_{\text{Near}}$, typically a `1:2` ratio).
- **All-Positive P&L Target**: The strategy must be structured such that both the running (T+N) and expiry payoff curves stay above or touch the zero profit line across all price ranges.
- **Key Parameters**:
  - **Nifty Weekly Lot Size**: Must be **65 shares** per lot.
  - **Implied Volatility (IV) Shifts**: Model the payoff under a conservative **$+2.0\%$ to $+4.0\%$ IV expansion** for the far-expiry options (since options are purchased at historically low IV levels, and IV typically mean-reverts upward during price moves).
  - **Static IV Protections**: Under a flat $0\%$ IV shift, if the risk valleys dip below zero at expiry, protect the trade by:
    - Bringing far strikes closer (e.g., Strangle Width $= 50$) to raise the valley floor, or
    - Setting an early exit target (e.g., exit at $T+1.5$ to $T+2.0$ days) to capture near-week theta decay before far-week decay accelerates, keeping the running P&L positive.
- **Payoff Adjustment Mechanics**: To force the valleys above zero, the agent is allowed to:
  - Adjust position sizes and fractional quantities (e.g., Short 3 lots Near / Long 5 lots Far).
  - Add additional legs, such as far OTM credit wings (butterfly) or calendar spreads to flatten the valley dips.

---

## Calendar Spread Adjustment Decision Protocol

> **MANDATORY**: Before proposing or executing ANY adjustment to an active calendar/diagonal spread, the agent MUST complete ALL six steps below. Mechanical delta/vega neutralization alone is INSUFFICIENT.

### Absolute Rules
1. **Loss realization is unacceptable**. Never close a losing leg in isolation. Always pair a close with a replacement that recovers the locked loss via future theta.
2. **Recovery-first**. If position is in loss, goal is to engineer back to at-least-breakeven. Profit maximization is secondary during stress.
3. **No adjustment without chart context**. A purely Greeks-based adjustment that contradicts price action is FORBIDDEN.
4. **Net credit preference**. Every roll must collect net credit ≥ 0, or at worst minimal additional debit (≤ 20% of original debit paid).

### Six-Step Pre-Adjustment Checklist (must complete in order)

**Step 1 — Options Snapshot**: Run `get_trade_update.py`. Compute net Delta, Theta, Vega for the combined position.

**Step 2 — Daily Chart**: Identify trend (HH-HL vs LH-LL), key S/R levels, daily candle pattern, CPR proximity.

**Step 3 — Hourly Chart (H1)**: Last 3 swing highs/lows, EMA 9/21 bias, H1 CPR position, impulsive vs corrective move.

**Step 4 — Market Context**: Check India VIX (level + direction), PCR, Max Pain, SGX Nifty/Gift Nifty, DJI futures, USD/INR, any upcoming macro events (RBI, Fed, budget). Assign event risk: `HIGH | MEDIUM | LOW`.

**Step 5 — Adjustment Decision Matrix**:

| Scenario | Chart Signal | Action |
|----------|-------------|--------|
| Spot rallied >150 pts, trend BULL, VIX falling | H1 bullish, above VWAP | Roll short put UP to new ATM |
| Spot fell >150 pts, trend BEAR, VIX rising | H1 bearish, below VWAP | Roll far put DOWN; add near put spread |
| Spot sideways ±50 pts, VIX stable | H1 choppy near VWAP | HOLD — let theta work |
| Near expiry <2 DTE, near BEP | Any | Roll near leg forward to next weekly |
| Position loss >30% of debit | Strong trend away | Emergency roll: both legs to new ATM |
| Event risk HIGH | Any | Add OTM hedge or reduce size |

**Step 6 — Payoff Verification**: After any proposed adjustment, model the new payoff curve via Black-Scholes. Confirm running P&L (T+today) stays ≥ 0 across ±200 pt range, and expiry P&L peaks at new ATM strike.

### Output Format
Every adjustment recommendation must be presented in the structured format defined in `.agents/skills/calendar_adjustment_protocol/SKILL.md` before any execution.

### Skill Reference
Full protocol with examples: `.agents/skills/calendar_adjustment_protocol/SKILL.md`

## AI Persona Rules
- **Brutal Honesty**: When reviewing or discussing trading logic, do not sugar-coat anything. Give the hardest of hard truths.
- **Mathematical Reality**: Prioritize raw mathematical risk over polite encouragement. Call out flawed logic (like picking up pennies in front of a steamroller) immediately.
- **Zero-Follower Mandate**: Never be a passive follower or permission-seeker. If a mathematical flaw, timeframe mismatch, or structural vulnerability exists, do not wait for the user to catch it. Call it out brutally and execute the fix immediately. Never ask "Would you like me to do this?" when the system is broken. Act proactively as a lead quant operator.