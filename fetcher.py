import requests
import os
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import importlib.util
import pathlib

# Load local config.py explicitly to avoid name clash with pip's 'config' package
config_path = pathlib.Path(__file__).parent / "config.py"
spec = importlib.util.spec_from_file_location("local_config", config_path)
local_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local_config)
config = local_config.config


@dataclass
class OptionData:
    strike: int
    call_price: Optional[float]
    call_oi: Optional[int]
    call_key: Optional[str]
    put_price: Optional[float]
    put_oi: Optional[int]
    put_key: Optional[str]
    expiry: str
    call_delta: float = 0.0
    call_gamma: float = 0.0
    call_theta: float = 0.0
    call_vega: float = 0.0
    call_iv: float = 0.0
    put_delta: float = 0.0
    put_gamma: float = 0.0
    put_theta: float = 0.0
    put_vega: float = 0.0
    put_iv: float = 0.0


@dataclass
class SpotData:
    price: float
    change: float
    change_pct: float
    timestamp: datetime
    high: float = 0.0
    low: float = 0.0


def _load_token() -> str:
    """Load Upstox access token from multiple sources (priority order):
    1. Desktop access_token.txt (same as working NiftyTrading MCP server)
    2. ~/.nifty_trading_token file
    3. Config / .env file
    4. Environment variable
    """
    # 1. Desktop token files (same paths as NiftyTrading MCP server - most reliable)
    _TOKEN_PATHS = [
        r"c:\Users\admin\Desktop\access_token.txt",
        r"c:\Users\admin\Desktop\access token.txt",
        r"c:\Users\admin\Desktop\Trading_Workspace\access_token.txt",
    ]
    for token_path in _TOKEN_PATHS:
        p = Path(token_path)
        if p.exists():
            try:
                content = p.read_bytes()
                # Handle UTF-16 BOM (same as MCP server)
                if content.startswith(b'\xff\xfe') or content.startswith(b'\xfe\xff'):
                    token = content.decode('utf-16').strip()
                else:
                    token = content.decode('utf-8', errors='ignore').strip()
                # Remove null bytes and non-printable characters
                token = "".join(c for c in token if c.isprintable() and not c.isspace() or c == '.')
                if token:
                    return token
            except Exception:
                continue
    
    # 2. Token file
    token_file = Path.home() / ".nifty_trading_token"
    if token_file.exists():
        token = token_file.read_text().strip().strip("'\"")
        if token:
            return token
    
    # 3. Config from .env
    if config.upstox_access_token:
        return config.upstox_access_token
    
    # 4. Environment variable
    return os.environ.get("UPSTOX_ACCESS_TOKEN", "")


import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class UpstoxFetcher:
    def __init__(self):
        self.base_url = "https://api.upstox.com/v2"
        self.historical_base_url = "https://api.upstox.com/v3"
        self.session = requests.Session()
        self.session.verify = False
        self._setup_headers()

    def _setup_headers(self):
        token = _load_token()
        if not token:
            print("WARNING: No Upstox access token found! Check ~/.nifty_trading_token or .env")
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self.session.headers.update(self.headers)

    def refresh_token(self):
        """Re-read token from file (useful if token was updated externally)"""
        self._setup_headers()

    def _convert_interval(self, interval: str) -> tuple:
        """Convert interval to Upstox V3 format (unit, interval_num)"""
        interval = interval.lower().strip()
        # If the interval is already in 'unit/num' format (e.g. 'minutes/5', 'days/1'),
        # split and return directly to avoid double-mapping to the default 1-minute.
        if "/" in interval:
            parts = interval.split("/", 1)
            return (parts[0], parts[1])
        if interval in ("1minute", "1min", "1m", "1"):
            return ("minutes", "1")
        elif interval in ("2minute", "2min", "2m", "2"):
            return ("minutes", "2")
        elif interval in ("3minute", "3min", "3m", "3"):
            return ("minutes", "3")
        elif interval in ("5minute", "5min", "5m", "5"):
            return ("minutes", "5")
        elif interval in ("10minute", "10min", "10m", "10"):
            return ("minutes", "10")
        elif interval in ("15minute", "15min", "15m", "15"):
            return ("minutes", "15")
        elif interval in ("30minute", "30min", "30m", "30"):
            return ("minutes", "30")
        elif interval in ("60minute", "60min", "1hour", "1hr", "1h", "60"):
            return ("hours", "1")
        elif interval in ("1day", "daily", "day", "1d"):
            return ("days", "1")
        elif interval in ("1week", "weekly", "week", "1w"):
            return ("weeks", "1")
        elif interval in ("1month", "monthly", "month"):
            return ("months", "1")
        else:
            return ("minutes", "1")

    def get_historical(
        self, instrument_key: str, interval: str = "1minute", days: int = 1
    ) -> pd.DataFrame:
        """Fetch historical OHLC data for any instrument key from Upstox API v3 with retry on 429"""
        import time
        from_date = (datetime.now() - timedelta(days=days+1)).strftime("%Y-%m-%d")
        to_date = datetime.now().strftime("%Y-%m-%d")
        
        unit, interval_num = self._convert_interval(interval)
        
        # URL encode the instrument key
        encoded_key = instrument_key.replace("|", "%7C").replace(" ", "%20")
        url = f"{self.historical_base_url}/historical-candle/{encoded_key}/{unit}/{interval_num}/{to_date}/{from_date}"
        
        max_retries = 5
        delay = 2
        data = []
        for attempt in range(max_retries):
            try:
                r = self.session.get(url, timeout=10)
                r.raise_for_status()
                data = r.json().get("data", {}).get("candles", [])
                break
            except Exception as e:
                print(f"Error fetching historical data for {instrument_key} (attempt {attempt+1}/{max_retries}): {e}")
                if ("429" in str(e) or "Too Many Requests" in str(e)) and attempt < max_retries - 1:
                    print(f"Rate limited. Waiting {delay}s before retry...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    break
        
        try:
            rows = []
            for candle in data:
                rows.append({
                    "timestamps": pd.to_datetime(candle[0]),
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "volume": int(candle[5]) if len(candle) > 5 else 0,
                })
            
            # Fetch today's intraday data to append to historical
            if unit == "minutes":
                resample_needed = False
                query_interval = f"{interval_num}minute"
                
                if query_interval not in ("1minute", "30minute"):
                    query_interval = "1minute"
                    resample_needed = True
                
                intraday_url = f"{self.base_url}/historical-candle/intraday/{encoded_key}/{query_interval}"
                delay_intra = 2
                intra_rows = []
                for attempt_intra in range(max_retries):
                    try:
                        r_intra = self.session.get(intraday_url, timeout=10)
                        if r_intra.status_code == 200:
                            intra_data = r_intra.json().get("data", {}).get("candles", [])
                            for candle in intra_data:
                                intra_rows.append({
                                    "timestamps": pd.to_datetime(candle[0]),
                                    "open": float(candle[1]),
                                    "high": float(candle[2]),
                                    "low": float(candle[3]),
                                    "close": float(candle[4]),
                                    "volume": int(candle[5]) if len(candle) > 5 else 0,
                                })
                            break
                        elif r_intra.status_code == 429:
                            if attempt_intra < max_retries - 1:
                                print(f"Intraday fetch rate limited. Waiting {delay_intra}s before retry...")
                                time.sleep(delay_intra)
                                delay_intra *= 2
                            else:
                                break
                        else:
                            r_intra.raise_for_status()
                    except Exception as e:
                        print(f"Error fetching intraday data for {instrument_key} (attempt {attempt_intra+1}/{max_retries}): {e}")
                        if "429" in str(e) and attempt_intra < max_retries - 1:
                            print(f"Rate limited. Waiting {delay_intra}s before retry...")
                            time.sleep(delay_intra)
                            delay_intra *= 2
                        else:
                            break
                
                if intra_rows:
                    if resample_needed:
                        # Convert to DataFrame to resample
                        temp_df = pd.DataFrame(intra_rows)
                        temp_df.set_index("timestamps", inplace=True)
                        temp_df.sort_index(inplace=True)
                        resampled = temp_df.resample(f"{interval_num}min", closed="left", label="left").agg({
                            "open": "first",
                            "high": "max",
                            "low": "min",
                            "close": "last",
                            "volume": "sum"
                        }).dropna().reset_index()
                        
                        for _, row in resampled.iterrows():
                            rows.append({
                                "timestamps": row["timestamps"],
                                "open": float(row["open"]),
                                "high": float(row["high"]),
                                "low": float(row["low"]),
                                "close": float(row["close"]),
                                "volume": int(row["volume"]),
                            })
                    else:
                        rows.extend(intra_rows)
            
            if not rows:
                return pd.DataFrame()
            
            df = pd.DataFrame(rows)
            # Sort by timestamp and drop any overlapping candles between historical and intraday
            df = df.sort_values("timestamps").drop_duplicates(subset=["timestamps"]).reset_index(drop=True)
            return df
        except Exception as e:
            print(f"Error parsing historical data for {instrument_key}: {e}")
            return pd.DataFrame()

    def get_nifty_historical(self, interval: str = "1minute", days: int = 1) -> pd.DataFrame:
        """Alias for get_historical with Nifty 50 Index"""
        # Handle string like "5d" for days
        if isinstance(days, str):
            if "d" in days:
                days = int(days.replace("d", ""))
            else:
                days = 1
        # Normalize interval: "5m" -> "5minute", "5d" -> "1day", etc.
        interval = interval.lower().strip()
        # Comprehensive interval map covering standard names and shorthand aliases
        interval_map = {
            # Standard names
            "1minute": "minutes/1",
            "2minute": "minutes/2",
            "3minute": "minutes/3",
            "5minute": "minutes/5",
            "10minute": "minutes/10",
            "15minute": "minutes/15",
            "30minute": "minutes/30",
            "1hour": "hours/1",
            "1day": "days/1",
            "1week": "weeks/1",
            "1month": "months/1",
            # Short aliases (used by agents and Kronos tool)
            "1m": "minutes/1",
            "2m": "minutes/2",
            "3m": "minutes/3",
            "5m": "minutes/5",
            "10m": "minutes/10",
            "15m": "minutes/15",
            "30m": "minutes/30",
            "1h": "hours/1",
            "1d": "days/1",
            "1w": "weeks/1",
            # Unambiguous text aliases
            "daily": "days/1",
            "day": "days/1",
            "weekly": "weeks/1",
            "week": "weeks/1",
            "monthly": "months/1",
            "month": "months/1",
        }
        normalized_interval = interval_map.get(interval, interval)
        return self.get_historical("NSE_INDEX|Nifty 50", normalized_interval, days)

    def get_nifty_options_instrument_list(self):
        """Alias for get_option_chain to support agent_manager.py calls"""
        return self.get_option_chain()

    def get_option_quote(self, instrument_key: str):
        """Helper to get a single quote for an option"""
        try:
            data = self.get_quote(instrument_key)
            # The structure for market/quotes might be complex, simplify it
            # Structure: {"status": "success", "data": {"NSE_FO:72325": {"last_price": 183.95, ...}}}
            all_data = data.get("data", {})
            for key, val in all_data.items():
                if instrument_key.replace("|", ":") in key:
                    return SpotData(
                        price=float(val.get("last_price", 0)),
                        change=0, change_pct=0, timestamp=datetime.now()
                    )
            return None
        except Exception as e:
            print(f"Error in get_option_quote: {e}")
            return None

    def get_nifty_spot(self) -> SpotData:
        """Fetch Nifty 50 spot price from Upstox API with retry on 429"""
        import time
        max_retries = 5
        delay = 2
        for attempt in range(max_retries):
            try:
                url = f"{self.base_url}/market-quote/quotes"
                params = {"instrument_key": "NSE_INDEX|Nifty 50"}
                r = self.session.get(url, params=params, timeout=10)
                r.raise_for_status()
                
                data = r.json().get("data", {}).get("NSE_INDEX:Nifty 50", {})
                if not data:
                    # Try alternate key format
                    all_data = r.json().get("data", {})
                    for key, val in all_data.items():
                        if "nifty" in key.lower() or "Nifty" in key:
                            data = val
                            break
                
                if not data:
                    raise Exception("Upstox API returned empty data for Nifty 50 spot")

                last_price = float(data.get("last_price", 0))
                if last_price == 0:
                    # Fallback to ohlc if LTP is 0 (pre-market)
                    last_price = float(data.get("ohlc", {}).get("close", 0))
                
                open_price = float(data.get("ohlc", {}).get("open", 0))
                high_price = float(data.get("ohlc", {}).get("high", 0))
                low_price = float(data.get("ohlc", {}).get("low", 0))
                change = last_price - open_price
                change_pct = (change / open_price * 100) if open_price > 0 else 0

                return SpotData(
                    price=last_price,
                    change=change,
                    change_pct=change_pct,
                    timestamp=datetime.now(),
                    high=high_price,
                    low=low_price
                )
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                else:
                    break

        # Fallback 1: Try /market-quote/ltp (lighter endpoint with separate rate limit)
        try:
            url_ltp = f"{self.base_url}/market-quote/ltp"
            params_ltp = {"instrument_key": "NSE_INDEX|Nifty 50"}
            r_ltp = self.session.get(url_ltp, params=params_ltp, timeout=5)
            if r_ltp.status_code == 200:
                data_ltp = r_ltp.json().get("data", {})
                for k, v in data_ltp.items():
                    if "nifty" in k.lower():
                        lp = float(v.get("last_price", 0))
                        if lp > 0:
                            return SpotData(price=lp, change=0, change_pct=0, timestamp=datetime.now())
        except Exception:
            pass

        # Fallback 2: Try latest 1-min intraday historical candle (separate v3 endpoint)
        try:
            url_candle = f"{self.base_url}/historical-candle/intraday/NSE_INDEX%7CNifty%2050/1minute"
            r_c = self.session.get(url_candle, timeout=5)
            if r_c.status_code == 200:
                candles = r_c.json().get("data", {}).get("candles", [])
                if candles:
                    latest_close = float(candles[0][4])
                    if latest_close > 0:
                        return SpotData(price=latest_close, change=0, change_pct=0, timestamp=datetime.now())
        except Exception:
            pass

        # Return a zero-state so the watcher knows data is unavailable
        return SpotData(price=0.0, change=0, change_pct=0, timestamp=datetime.now())

    def get_all_expiries(self) -> list[str]:
        """Fetch all sorted active expiry dates directly from Upstox contract API with retry"""
        import time
        max_retries = 5
        delay = 2
        for attempt in range(max_retries):
            try:
                url = f"{self.base_url}/option/contract"
                params = {"instrument_key": "NSE_INDEX|Nifty 50"}
                r = self.session.get(url, params=params, timeout=10)
                r.raise_for_status()
                data = r.json().get("data", [])
                if data:
                    today_str = datetime.now().strftime("%Y-%m-%d")
                    expiries = sorted(list(set(item.get("expiry") for item in data if item.get("expiry"))))
                    valid_expiries = [e for e in expiries if e >= today_str]
                    if valid_expiries:
                        return valid_expiries
            except Exception as e:
                print(f"Warning: Failed to fetch expiries from API (attempt {attempt+1}/{max_retries}): {e}")
                if "429" in str(e) and attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                else:
                    break
        # Fallback to estimated dates if API fails
        today = datetime.now()
        days_ahead = (0 - today.weekday() + 7) % 7
        target_date = today + timedelta(days=days_ahead if days_ahead != 0 else 7)
        return [target_date.strftime("%Y-%m-%d"), (target_date + timedelta(days=7)).strftime("%Y-%m-%d")]

    def get_nearest_expiry(self) -> str:
        """Get the nearest available expiry date from the API"""
        valid_expiries = self.get_all_expiries()
        if valid_expiries:
            now = datetime.now()
            today_exp = datetime.strptime(valid_expiries[0], "%Y-%m-%d").date()
            # EOD Expiry Roll: If today is expiry day and it's >= 15:15, skip today's expiry
            if len(valid_expiries) > 1 and now.date() == today_exp and (now.hour > 15 or (now.hour == 15 and now.minute >= 11)):
                print(f"[EOD ROLL] Rolling nearest expiry to {valid_expiries[1]} (Avoiding Expiry Day Settlement Risk)")
                return valid_expiries[1]
            return valid_expiries[0]
        return datetime.now().strftime("%Y-%m-%d")

    def get_next_expiry(self) -> str:
        """Get the next available exchange expiry date (not an estimated +7 days)"""
        valid_expiries = self.get_all_expiries()
        nearest = self.get_nearest_expiry()
        # Find next valid expiry after nearest
        subsequent = [e for e in valid_expiries if e > nearest]
        if subsequent:
            return subsequent[0]
        # Fallback
        dt = datetime.strptime(nearest, "%Y-%m-%d")
        return (dt + timedelta(days=7)).strftime("%Y-%m-%d")

    def get_expiry_for_premium(self, premium: float, threshold: float = 50) -> str:
        """Choose expiry based on premium - low premium = next week for more time value"""
        if premium < threshold:
            return self.get_next_expiry()
        return self.get_nearest_expiry()

    def get_option_chain(self, expiry: Optional[str] = None) -> list[OptionData]:
        import time
        if not expiry:
            expiry = self.get_nearest_expiry()

        url = f"{self.base_url}/option/chain"
        params = {
            "instrument_key": "NSE_INDEX|Nifty 50",
            "expiry_date": expiry,
        }
        
        max_retries = 5
        delay = 2
        data = {}
        for attempt in range(max_retries):
            try:
                r = self.session.get(url, params=params, timeout=10)
                r.raise_for_status()
                data = r.json()
                break
            except Exception as e:
                err_str = str(e)
                print(f"Error in get_option_chain (attempt {attempt+1}/{max_retries}): {e}")
                if ("429" in err_str or "500" in err_str or "502" in err_str or "503" in err_str or "504" in err_str) and attempt < max_retries - 1:
                    print(f"Retrying option chain in {delay}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(delay)
                    delay *= 1.5
                else:
                    raise e

        results = []
        for item in data.get("data", []):
            strike = int(item.get("strike_price", 0))
            
            call_opt = item.get("call_options") or {}
            put_opt = item.get("put_options") or {}
            
            call_market = call_opt.get("market_data") or {}
            put_market = put_opt.get("market_data") or {}
            
            call_greeks = call_opt.get("option_greeks") or {}
            put_greeks = put_opt.get("option_greeks") or {}
            
            def get_greek(g_dict, name):
                val = g_dict.get(name)
                if val is None:
                    return 0.0
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return 0.0

            results.append(
                OptionData(
                    strike=strike,
                    call_price=call_market.get("ltp", 0),
                    call_oi=call_market.get("oi", 0),
                    call_key=call_opt.get("instrument_key"),
                    put_price=put_market.get("ltp", 0),
                    put_oi=put_market.get("oi", 0),
                    put_key=put_opt.get("instrument_key"),
                    expiry=expiry,
                    call_delta=get_greek(call_greeks, "delta"),
                    call_gamma=get_greek(call_greeks, "gamma"),
                    call_theta=get_greek(call_greeks, "theta"),
                    call_vega=get_greek(call_greeks, "vega"),
                    call_iv=get_greek(call_greeks, "iv"),
                    put_delta=get_greek(put_greeks, "delta"),
                    put_gamma=get_greek(put_greeks, "gamma"),
                    put_theta=get_greek(put_greeks, "theta"),
                    put_vega=get_greek(put_greeks, "vega"),
                    put_iv=get_greek(put_greeks, "iv"),
                )
            )
        return results

    def get_quote(self, instrument_key: str) -> dict:
        """Fetch real-time quote for any instrument key"""
        url = f"{self.base_url}/market-quote/quotes"
        params = {
            "instrument_key": instrument_key,
            "format": "json",
        }
        r = self.session.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    def get_market_quote(self, instrument_key: str) -> dict:
        """Alias for get_quote for backward compatibility"""
        return self.get_quote(instrument_key)

    def get_account_balance(self) -> dict:
        """Fetch account balance/funds from Upstox API"""
        try:
            url = f"{self.base_url}/user/get-funds-and-margin"
            r = self.session.get(url, timeout=10)
            r.raise_for_status()
            data = r.json().get("data", {})
            return {
                "available_balance": data.get("available_balance", 0),
                "used_margin": data.get("used_margin", 0),
                "total_margin": data.get("total_margin", 0),
                "cash_available": data.get("cash_available", 0),
            }
        except Exception as e:
            print(f"Error fetching account balance: {e}")
            return {"error": str(e)}

    def get_profile(self) -> dict:
        """Fetch user profile from Upstox API"""
        try:
            url = f"{self.base_url}/user/profile"
            r = self.session.get(url, timeout=10)
            r.raise_for_status()
            return r.json().get("data", {})
        except Exception as e:
            print(f"Error fetching profile: {e}")
            return {"error": str(e)}

    def get_oi_analysis(self) -> dict:
        """Calculate OI-based analysis: PCR, Max Pain, total OI"""
        try:
            chain = self.get_option_chain()
            if not chain:
                return {"error": "No option chain data"}

            total_call_oi = sum(o.call_oi or 0 for o in chain)
            total_put_oi = sum(o.put_oi or 0 for o in chain)

            pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 1.0

            strikes = [o.strike for o in chain]
            if strikes:
                spot = self.get_nifty_spot().price
                atm = int(round(spot / 50) * 50)
                max_pain = atm
                max_pain_val = 0

                for s in strikes:
                    call_oi = next((o.call_oi or 0 for o in chain if o.strike == s), 0)
                    put_oi = next((o.put_oi or 0 for o in chain if o.strike == s), 0)
                    total_oi = call_oi + put_oi
                    if total_oi > max_pain_val:
                        max_pain_val = total_oi
                        max_pain = s

                return {
                    "pcr": round(pcr, 2),
                    "max_pain": max_pain,
                    "total_call_oi": int(total_call_oi),
                    "total_put_oi": int(total_put_oi),
                }
            return {"error": "No strikes found"}
        except Exception as e:
            print(f"Error in get_oi_analysis: {e}")
            return {"error": str(e)}


if __name__ == "__main__":
    fetcher = UpstoxFetcher()
    print("Testing Upstox API (no yfinance)...")

    spot = fetcher.get_nifty_spot()
    print(f"Spot: {spot.price} ({spot.change:+.2f} {spot.change_pct:+.2f}%)")

    expiry = fetcher.get_nearest_expiry()
    print(f"Expiry: {expiry}")

    chain = fetcher.get_option_chain(expiry)
    print(f"Option chain: {len(chain)} strikes")
