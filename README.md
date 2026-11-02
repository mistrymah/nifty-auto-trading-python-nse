# Nifty Autonomous Options Trading & Adjustment Engine 🚀

An institutional-grade, fully autonomous algorithmic options trading system for Nifty 50 weekly cycles. Built around the **Dynamic Calendar M-Shape Strategy (Portfolio Layering)** with real-time volatility surface analysis, dealer gamma exposure (GEX), machine learning forecasts, and an autonomous **Smart Adjustment Analyst Agent**.

---

## 🏛️ Strategy Overview

The core strategy deploys weekly **Double Diagonals** (Tuesday-to-Tuesday cycle), actively defends breakevens using **Portfolio Layering**, and neutralizes asymmetric risk without buying toxic unfinanced options:

```
          M-Shape Double Mountain Payoff
             ▲                   ▲
            / \                 / \
           /   \               /   \
          /     \             /     \
  ───────/       \───────────/       \─────── (₹0 P&L)
                  \         /
                   \       /  <- Valley Overlap
                    ▼     ▼
```

### Key Pillars
1. **Dynamic Mountain Layering (M-Shape)**:
   - Initial entry: Centered Double Diagonal (1 Near Short Straddle/Strangle + 1 Far Long OTM Strangle).
   - On breakeven breach, layers a secondary Diagonal precisely at the valley to transform the payoff curve into an interlocking M-Shape (Camel) with zero whipsaw risk.
2. **Whalley-Wilmott Band Filtering**:
   - Calculates dynamic delta transaction bands from live portfolio Gamma to prevent churn from intraday market noise.
3. **Smart Adjustment Analyst Agent (`adjustment_analyst_agent.py`)**:
   - Enforces 5 strict risk gates before any trade or adjustment can execute.
   - Bans toxic unhedged far-month option purchases at peak IV.
   - Enforces net-credit repair preference when DTE $\le 1.5$ days.
   - Neutralizes directional cliffs via OTM vertical credit spreads.
4. **Institutional Intel**:
   - 25-Delta Volatility Skew & Risk Reversal sentiment.
   - Dealer Gamma Exposure (GEX) profiling (Pinning vs Volatility expansion regimes).
   - Kronos AI 5-day horizon ML forecasting on 1-hour candles.

---

## 📁 Repository Architecture

```
nifty_mcp_upstox/
├── adjustment_analyst_agent.py    # Autonomous quantitative analyst (audit, simulation, repair)
├── ultimate_execution_manager.py  # All-in-one console runner & live portfolio ledger
├── ultimate_strategy_engine.py    # Setup generator, strike biasing, ML integration
├── ultimate_adjustment_engine.py  # Real-time portfolio layering & WW band execution
├── calendar_intelligence.py       # Vol surface, dealer GEX, term structure analytics
├── fetcher.py                     # Upstox v2 API bridge (quotes, historical, option chains)
├── kronos_integration.py          # Time-series ML forecasting
├── config.py                      # System configuration & environment loader
├── run.bat                        # Single-window console launcher
├── run_all.bat                    # Full suite launcher
├── AGENTS.md                      # Strategy rules, adjustment protocols, guidelines
├── requirements.txt               # Python package dependencies
└── .env.example                   # Template for API keys
```

---

## ⚙️ Setup & Installation

### 1. Prerequisites
- Python 3.10+ (Recommended: Python 3.11 or 3.12)
- Upstox Developer Account with API Key and Access Token

### 2. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/nifty-options-m-shape-agent.git
cd nifty-options-m-shape-agent
```

### 3. Create a Virtual Environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / Mac
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure API Credentials
Copy the example environment file:
```bash
cp .env.example .env
```
Edit `.env` with your Upstox credentials:
```ini
UPSTOX_API_KEY=your_api_key_here
UPSTOX_ACCESS_TOKEN=your_access_token_here
UPSTOX_API_SECRET=your_api_secret_here
UPSTOX_REDIRECT_URI=http://localhost:8000/callback
```

---

## 🚀 Running the Strategy

### Single-Window Launcher (Recommended)
Double-click `run.bat` or run from PowerShell:
```cmd
run.bat
```
This automatically initiates:
1. Pre-flight market audit & portfolio health check via `AdjustmentAnalystAgent`.
2. Real-time 60-second monitoring daemon (`ultimate_execution_manager.py`).
3. Autonomous strike layering, delta management, and weekly campaign ledger tracking.

### Running Individual Agents
```bash
# Run Pre-flight Audit & Memory Inspection
python adjustment_analyst_agent.py --audit

# Run Self-Repair Diagnostics
python adjustment_analyst_agent.py --repair

# Run Strategy Generator Standalone
python ultimate_strategy_engine.py
```

---

## 🛡️ Risk Management Rules

- **Loss Realization Ban**: Never close a losing leg in isolation without pairing it with a recovery structure.
- **Credit Mandate**: Any adjustment under $\le 1.5$ DTE must collect net credit ($\ge 0$).
- **Cliff Protection**: A $\pm 400$ pt market reversal must never incur a loss exceeding ₹10,000.
- **Mandatory Expiry Flush**: Open campaigns are squared off at 15:10 IST on Tuesday expiry to eliminate settlement and physical delivery risks.

---

## 📄 License
MIT License. For educational and quantitative research purposes only.
