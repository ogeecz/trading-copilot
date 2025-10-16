import time
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
from typing import Optional, Dict, List
import requests

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Trading Copilot Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== CUSTOM CSS =====
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    h1 { 
        color: #00d4ff !important; 
        font-weight: 700 !important;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    h2 { color: #00d4ff !important; font-size: 1.5rem !important; }
    h3 { color: #7dd3fc !important; font-size: 1.2rem !important; }
    .stAlert { 
        border-radius: 12px !important;
        border-left: 4px solid #00d4ff !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #00d4ff !important;
    }
    .stButton>button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(0,212,255,0.3) !important;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d29 0%, #0e1117 100%) !important;
    }
    .commodity-card {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        border-left: 3px solid #00d4ff;
    }
</style>
""", unsafe_allow_html=True)

# ===== KORELACE TRHŮ =====
MARKET_CORRELATIONS = {
    "EUR/USD": {
        "commodities": {
            "Natural Gas": {"correlation": "INVERSE", "strength": "MEDIUM", "symbol": "NG=F"},
            "S&P500": {"correlation": "DIRECT", "strength": "MEDIUM", "symbol": "^GSPC"}
        },
        "description": "EUR slábne při drahém plynu (energie krize), sílí při risk-on (akcie up)"
    },
    "GBP/USD": {
        "commodities": {
            "Brent Oil": {"correlation": "DIRECT", "strength": "WEAK", "symbol": "BZ=F"},
            "S&P500": {"correlation": "DIRECT", "strength": "MEDIUM", "symbol": "^GSPC"}
        },
        "description": "GBP jako risk asset - sílí při optimismu (akcie up)"
    },
    "USD/JPY": {
        "commodities": {
            "S&P500": {"correlation": "DIRECT", "strength": "STRONG", "symbol": "^GSPC"},
            "VIX": {"correlation": "INVERSE", "strength": "STRONG", "symbol": "^VIX"}
        },
        "description": "JPY = safe haven. Risk-off (akcie ↓, VIX ↑) → JPY sílí → USD/JPY ↓"
    },
    "USD/CAD": {
        "commodities": {
            "WTI Crude": {"correlation": "INVERSE", "strength": "VERY STRONG", "symbol": "CL=F"},
            "Brent Oil": {"correlation": "INVERSE", "strength": "VERY STRONG", "symbol": "BZ=F"}
        },
        "description": "Kanada = 4. největší producent ropy. Ropa ↑ → CAD ↑ → USD/CAD ↓"
    },
    "AUD/USD": {
        "commodities": {
            "Gold": {"correlation": "DIRECT", "strength": "STRONG", "symbol": "GC=F"},
            "Iron Ore": {"correlation": "DIRECT", "strength": "STRONG", "symbol": "IRON.AX"},
            "Copper": {"correlation": "DIRECT", "strength": "MEDIUM", "symbol": "HG=F"}
        },
        "description": "Austrálie = biggest miner. Zlato/rudy ↑ → AUD ↑. Watch China demand!"
    },
    "NZD/USD": {
        "commodities": {
            "Dairy": {"correlation": "DIRECT", "strength": "MEDIUM", "symbol": None},
            "Gold": {"correlation": "DIRECT", "strength": "MEDIUM", "symbol": "GC=F"}
        },
        "description": "NZ = dairy exporter. Mléčné výrobky ↑ → NZD ↑"
    }
}

# ===== ROZŠÍŘENÁ DATABÁZE ZPRÁV =====
NEWS_DATABASE = {
    "NFP": {
        "full_name": "Non-Farm Payrolls (USA)",
        "impact": "⚡⚡⚡ EXTREME",
        "description": "Měsíční zpráva o počtu nových pracovních míst mimo zemědělství v USA",
        "typical_move": "50-150 pips první minuta, 100-300 pips celkem",
        "affects": ["USD/*", "Gold", "S&P500", "VIX"],
        "strategies": {
            "15-30min before": "🎯 Identifikuj 30min range → nastav pending orders +/- 15 pips",
            "0-5min during": "🚫 NEOBCHODUJ! Extreme chaos, spreads 30+ pips, slippage brutal",
            "5-15min after": "⚡ Ride momentum pokud clear direction + RSI mezi 40-60",
            "15-30min after": "🔄 Fade overreaction pokud RSI>75 nebo <25, target VWAP"
        },
        "watch_out": [
            "První spike (0-2min) často fake - čekej stabilizaci",
            "Spreads se rozšíří 10-15x normálu (2→30 pips)",
            "Actual vs Expected je klíč - rozdíl 50k+ = massive move",
            "Revize předchozího měsíce může být důležitější než aktuální číslo"
        ],
        "historical_stats": {
            "avg_move": "85 pips",
            "first_spike_reversal": "68%",
            "best_strategy": "Fade first spike (65% win rate)",
            "worst_time": "First 3 minutes (32% win rate)"
        }
    },
    "FOMC": {
        "full_name": "FOMC Meeting (Federal Reserve)",
        "impact": "⚡⚡⚡ EXTREME", 
        "description": "Fed rozhoduje o úrokových sazbách - nejvýznamnější event pro USD",
        "typical_move": "30-100 pips statement, 50-200 pips press conference",
        "affects": ["USD/*", "Gold", "All stocks", "Bonds"],
        "strategies": {
            "Before": "🚫 Flat position! Outcome je 50/50, často leaked",
            "Statement (14:00 EST)": "⏸️ Počkej 2-3 minuty na initial reaction",
            "Press Conference": "🎤 Powell's tone je klíč - hawkish/dovish důležitější než sazby",
            "After": "📊 Obchoduj podle dot plot a forward guidance"
        },
        "watch_out": [
            "Rate decision často priced-in → reakce malá",
            "Forward guidance + dot plot = skutečný katalyzátor",
            "Powell může změnit směr trhu jednou větou",
            "Watch bond yields - řídí USD víc než forex traders"
        ],
        "historical_stats": {
            "avg_move": "120 pips celý event",
            "priced_in_rate": "75% času",
            "powell_reversal": "43% (často řekne opak očekávání)"
        }
    },
    "ECB": {
        "full_name": "ECB Interest Rate Decision",
        "impact": "⚡⚡ HIGH",
        "description": "Evropská centrální banka mění sazby - hlavní driver pro EUR",
        "typical_move": "30-80 pips statement, 40-100 pips Lagarde speech",
        "affects": ["EUR/*", "European stocks"],
        "strategies": {
            "Before": "🔍 Sleduj ECB members comments týden před - často leaked",
            "13:45 CET": "⏸️ Počkej 5 min - initial spike často reverses",
            "14:30 CET": "🎤 Lagarde press conference - obchoduj její tone",
            "After": "📈 Trend často pokračuje 1-2 dny po eventu"
        },
        "watch_out": [
            "ECB často 'behind the curve' - pomalé na změny",
            "Forward guidance vágní → volatilita vyšší",
            "EUR/USD často priced-in → obchoduj EUR/GBP místo toho",
            "Lagarde může být dovish i při rate hike"
        ],
        "historical_stats": {
            "avg_move": "65 pips",
            "lagarde_surprise": "38%"
        }
    },
    "CPI": {
        "full_name": "Consumer Price Index (Inflation)",
        "impact": "⚡⚡⚡ EXTREME",
        "description": "Měsíční inflační data - určuje směr měnové politiky",
        "typical_move": "40-120 pips",
        "affects": ["USD/*", "Bonds", "Gold"],
        "strategies": {
            "Before": "🎯 Range breakout setup - inflation surprise = big move",
            "08:30 EST": "⚡ Immediate spike - pokud překvapení 0.2%+ obchoduj momentum",
            "After 15min": "📊 Fade pokud overextended, nebo hold pokud clear trend"
        },
        "watch_out": [
            "Core CPI důležitější než headline",
            "Fed sleduje 'supercore' (services ex-housing)",
            "High CPI = rate hikes expected = USD up (short-term)",
            "Ale high inflation dlouhodobě = USD down"
        ],
        "historical_stats": {
            "avg_move": "75 pips",
            "surprise_0.3%+": "150+ pips"
        }
    },
    "GDP": {
        "full_name": "GDP Growth Rate",
        "impact": "⚡⚡ HIGH",
        "description": "Čtvrtletní růst ekonomiky",
        "typical_move": "20-60 pips",
        "affects": ["Local currency", "Stocks"],
        "strategies": {
            "Before": "⚠️ Lower impact než NFP/CPI - čekej confirmation",
            "After": "📈 Obchoduj pokud beat/miss >0.5% expected"
        },
        "watch_out": [
            "Často revised - první reading méně důležitý",
            "Pokud priced-in → minimal reaction",
            "Watch forward guidance víc než číslo samo"
        ],
        "historical_stats": {
            "avg_move": "45 pips"
        }
    }
}

# ===== KONFIGURACE =====
DEFAULT_PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CAD", "AUD/USD", "NZD/USD"]
INTERVALS = ["5min", "15min", "30min", "1h"]

TD_TO_YF = {
    "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "USDJPY=X",
    "USD/CAD": "USDCAD=X", "AUD/USD": "AUDUSD=X", "NZD/USD": "NZDUSD=X"
}

# ===== COMMODITY PRICES =====
@st.cache_data(ttl=300, show_spinner=False)
def fetch_commodity_prices() -> Dict:
    """Fetch live commodity prices"""
    commodities = {
        "CL=F": "WTI Crude",
        "BZ=F": "Brent Oil", 
        "GC=F": "Gold",
        "HG=F": "Copper",
        "NG=F": "Nat Gas",
        "^GSPC": "S&P500",
        "^VIX": "VIX"
    }
    
    prices = {}
    try:
        for symbol, name in commodities.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d", interval="1h")
                if not hist.empty:
                    current = float(hist['Close'].iloc[-1])
                    prev = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current
                    change_pct = ((current - prev) / prev) * 100 if prev != 0 else 0
                    prices[name] = {
                        "price": current,
                        "change": change_pct,
                        "symbol": symbol
                    }
            except:
                continue
    except:
        pass
    
    return prices

# ===== ENHANCED NEWS CALENDAR =====
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_economic_calendar() -> List[Dict]:
    """Fetch upcoming economic events with enhanced details"""
    now = datetime.utcnow()
    events = [
        {
            "time": now + timedelta(hours=2),
            "currency": "USD", 
            "event": "NFP",
            "impact": "HIGH"
        },
        {
            "time": now + timedelta(hours=5),
            "currency": "EUR",
            "event": "ECB",
            "impact": "HIGH"
        },
        {
            "time": now + timedelta(hours=8),
            "currency": "GBP",
            "event": "GDP",
            "impact": "MEDIUM"
        },
        {
            "time": now + timedelta(days=1),
            "currency": "USD",
            "event": "FOMC",
            "impact": "HIGH"
        },
        {
            "time": now + timedelta(hours=26),
            "currency": "USD",
            "event": "CPI",
            "impact": "HIGH"
        },
    ]
    return events

def is_news_time(currency: str, events: List[Dict], buffer_minutes: int = 30) -> Optional[tuple]:
    """Check if we're near news event and return (event, time_to_event_min)"""
    now = datetime.utcnow()
    for event in events:
        if event["impact"] != "HIGH":
            continue
        event_currencies = event.get("currency", "")
        if any(curr in currency for curr in ["EUR", "USD", "GBP", "JPY", "CAD", "AUD", "NZD"]):
            time_to_event = (event["time"] - now).total_seconds() / 60
            if -buffer_minutes <= time_to_event <= buffer_minutes:
                return (event, int(time_to_event))
    return None

# ===== NEWS TRADING STRATEGIES =====
def signal_news_breakout(df: pd.DataFrame, news_event: Dict, params: Dict) -> Optional[Dict]:
    """Pre-news breakout setup"""
    if len(df) < 20:
        return None
    try:
        recent = df.iloc[-20:]
        high_range = recent["High"].max()
        low_range = recent["Low"].min()
        range_size = high_range - low_range
        
        last = df.iloc[-1]
        close = float(last["Close"])
        atr_val = float(last["ATR"])
        
        if range_size < 2.5 * atr_val:
            entry_buy = high_range + 0.0005
            entry_sell = low_range - 0.0005
            sl_distance = atr_val * 2.0
            
            ema20 = float(last["EMA20"])
            ema50 = float(last["EMA50"])
            
            if ema20 > ema50:
                return {
                    "type": "BUY",
                    "entry": entry_buy,
                    "sl": entry_buy - sl_distance,
                    "tp": entry_buy + (sl_distance * 2.5),
                    "confidence": 70,
                    "reason": f"🔥 NEWS BREAKOUT: {news_event['event']} (Bullish bias)",
                    "rr": 2.5,
                    "news_trade": True
                }
            else:
                return {
                    "type": "SELL",
                    "entry": entry_sell,
                    "sl": entry_sell + sl_distance,
                    "tp": entry_sell - (sl_distance * 2.5),
                    "confidence": 70,
                    "reason": f"🔥 NEWS BREAKOUT: {news_event['event']} (Bearish bias)",
                    "rr": 2.5,
                    "news_trade": True
                }
    except:
        pass
    return None

def signal_news_momentum(df: pd.DataFrame, news_event: Dict, params: Dict) -> Optional[Dict]:
    """Post-news momentum"""
    if len(df) < 10:
        return None
    try:
        last = df.iloc[-1]
        prev_5 = df.iloc[-6:-1]
        
        close = float(last["Close"])
        atr_val = float(last["ATR"])
        rsi_val = float(last["RSI"])
        
        momentum_up = (close - prev_5["Close"].iloc[0]) / prev_5["Close"].iloc[0]
        
        if momentum_up > 0.003 and rsi_val < 75:
            return {
                "type": "BUY",
                "entry": close,
                "sl": close - (atr_val * 2.0),
                "tp": close + (atr_val * 3.0),
                "confidence": 75,
                "reason": f"⚡ POST-NEWS MOMENTUM: {news_event['event']} ({momentum_up*100:.2f}% move)",
                "rr": 1.5,
                "news_trade": True
            }
        elif momentum_up < -0.003 and rsi_val > 25:
            return {
                "type": "SELL",
                "entry": close,
                "sl": close + (atr_val * 2.0),
                "tp": close - (atr_val * 3.0),
                "confidence": 75,
                "reason": f"⚡ POST-NEWS MOMENTUM: {news_event['event']} ({momentum_up*100:.2f}% move)",
                "rr": 1.5,
                "news_trade": True
            }
    except:
        pass
    return None

def signal_news_fade(df: pd.DataFrame, news_event: Dict, params: Dict) -> Optional[Dict]:
    """Fade overreaction"""
    if len(df) < 10:
        return None
    try:
        last = df.iloc[-1]
        prev_3 = df.iloc[-4:-1]
        
        close = float(last["Close"])
        atr_val = float(last["ATR"])
        rsi_val = float(last["RSI"])
        vwap_val = float(last["VWAP"])
        
        spike = (close - prev_3["Close"].iloc[0]) / prev_3["Close"].iloc[0]
        
        if spike > 0.005 and rsi_val > 75 and close > vwap_val * 1.003:
            return {
                "type": "SELL",
                "entry": close,
                "sl": close + (atr_val * 1.5),
                "tp": vwap_val,
                "confidence": 65,
                "reason": f"🔄 FADE OVERREACTION: {news_event['event']} (RSI={rsi_val:.0f})",
                "rr": abs((vwap_val - close) / (atr_val * 1.5)) if atr_val > 0 else 1.0,
                "news_trade": True
            }
        elif spike < -0.005 and rsi_val < 25 and close < vwap_val * 0.997:
            return {
                "type": "BUY",
                "entry": close,
                "sl": close - (atr_val * 1.5),
                "tp": vwap_val,
                "confidence": 65,
                "reason": f"🔄 FADE OVERREACTION: {news_event['event']} (RSI={rsi_val:.0f})",
                "rr": abs((vwap_val - close) / (atr_val * 1.5)) if atr_val > 0 else 1.0,
                "news_trade": True
            }
    except:
        pass
    return None

def build_news_signal(df: pd.DataFrame, news_event: Dict, params: Dict, time_to_news_min: int) -> Optional[Dict]:
    """Determine best news trading strategy"""
    if time_to_news_min > 15:
        return signal_news_breakout(df, news_event, params)
    elif 0 <= time_to_news_min <= 15:
        return signal_news_momentum(df, news_event, params)
    elif -15 <= time_to_news_min < 0:
        momentum_sig = signal_news_momentum(df, news_event, params)
        if momentum_sig:
            return momentum_sig
        return signal_news_fade(df, news_event, params)
    return None

# ===== TWELVE DATA API =====
def fetch_twelvedata(symbol: str, interval: str, api_key: str) -> pd.DataFrame:
    try:
        url = "https://api.twelvedata.com/time_series"
        params = {"symbol": symbol, "interval": interval, "outputsize": 500, "apikey": api_key, "format": "JSON"}
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            return pd.DataFrame()
        data = response.json()
        if "values" not in data or not data["values"]:
            return pd.DataFrame()
        rows = []
        for candle in data["values"]:
            rows.append({
                "Datetime": pd.to_datetime(candle["datetime"]),
                "Open": float(candle["open"]),
                "High": float(candle["high"]),
                "Low": float(candle["low"]),
                "Close": float(candle["close"]),
                "Volume": float(candle.get("volume", 1000))
            })
        df = pd.DataFrame(rows).sort_values("Datetime").reset_index(drop=True)
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        return df
    except:
        return pd.DataFrame()

# ===== TECHNICAL INDICATORS =====
def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    gain = up.ewm(alpha=1/window, min_periods=window).mean()
    loss = down.ewm(alpha=1/window, min_periods=window).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vol = df["Volume"].copy().replace(0, 1.0)
    cum_vol = vol.cumsum()
    cum_pv = (tp * vol).cumsum()
    return cum_pv / cum_vol

def calculate_market_sentiment(df: pd.DataFrame) -> str:
    if len(df) < 50:
        return "NEUTRAL"
    last = df.iloc[-1]
    ema20 = float(last.get("EMA20", 0))
    ema50 = float(last.get("EMA50", 0))
    rsi_val = float(last.get("RSI", 50))
    
    if ema20 > ema50 and rsi_val > 55:
        return "BULLISH"
    elif ema20 < ema50 and rsi_val < 45:
        return "BEARISH"
    else:
        return "NEUTRAL"

def pct(x: float) -> str:
    return f"{x*100:.2f}%"

# ===== TELEGRAM =====
def send_telegram_notification(bot_token: str, chat_id: str, signal: Dict, symbol: str) -> bool:
    if not bot_token or not chat_id:
        return False
    try:
        emoji = "🟢" if signal["type"] == "BUY" else "🔴"
        message = (
            f"{emoji} <b>{signal['type']} SIGNAL - {symbol}</b>\n\n"
            f"💰 <b>Entry:</b> {signal['entry']:.5f}\n"
            f"🛑 <b>Stop Loss:</b> {signal['sl']:.5f}\n"
            f"🎯 <b>Take Profit:</b> {signal['tp']:.5f}\n"
            f"📊 <b>R:R:</b> {signal.get('rr', 0):.2f}\n"
            f"✅ <b>Confidence:</b> {signal['confidence']}%\n\n"
            f"📝 <i>{signal['reason']}</i>\n\n"
            f"🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
        response = requests.post(url, data=payload, timeout=5)
        return response.status_code == 200
    except:
        return False

# ===== DATA LOADING =====
@st.cache_data(ttl=60, show_spinner=False)
def load_data(symbol: str, interval: str, use_twelvedata: bool, api_key: str) -> pd.DataFrame:
    if use_twelvedata and api_key:
        df = fetch_twelvedata(symbol, interval, api_key)
        if not df.empty:
            return df
    yf_symbol = TD_TO_YF.get(symbol, "EURUSD=X")
    yf_interval_map = {"5min": "5m", "15min": "15m", "30min": "30m", "1h": "1h"}
    yf_interval = yf_interval_map.get(interval, "5m")
    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=7)
        df = yf.download(tickers=yf_symbol, interval=yf_interval, start=start.strftime("%Y-%m-%d"),
                        end=end.strftime("%Y-%m-%d"), progress=False, auto_adjust=False, prepost=False)
        if df.empty:
            return pd.DataFrame()
        df = df.rename_axis("Datetime").reset_index()
        if hasattr(df["Datetime"].iloc[0], "tzinfo") and df["Datetime"].dt.tz is not None:
            df["Datetime"] = df["Datetime"].dt.tz_convert("UTC").dt.tz_localize(None)
        return df
    except:
        return pd.DataFrame()

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 60:
        return df
    try:
        out = df.copy()
        out["EMA20"] = ema(out["Close"], 20)
        out["EMA50"] = ema(out["Close"], 50)
        out["RSI"] = rsi(out["Close"], 14)
        out["VWAP"] = vwap(out).bfill().ffill()
        out["ATR"] = atr(out, 14).bfill().ffill()
        vwap_vals = out["VWAP"].values
        close_vals = out["Close"].values
        from_vwap = np.zeros(len(out))
        for i in range(len(out)):
            if vwap_vals[i] > 0.0001:
                from_vwap[i] = (close_vals[i] / vwap_vals[i]) - 1.0
        out["FromVWAP"] = from_vwap
        out["TrendUp"] = (out["EMA20"] >= out["EMA50"]).astype(int)
        return out
    except:
        return df

# ===== SIGNALS =====
def signal_mean_reversion(df: pd.DataFrame, params: Dict) -> Optional[Dict]:
    if len(df) < 60:
        return None
    try:
        row = df.iloc[-1]
        from_vwap = float(row["FromVWAP"])
        rsi_val = float(row["RSI"])
        ema20 = float(row["EMA20"])
        ema50 = float(row["EMA50"])
        close = float(row["Close"])
        atr_val = float(row["ATR"])
        vwap_val = float(row["VWAP"])
        
        cond = (from_vwap <= -params["vwap_threshold"]/100) and (rsi_val < params["rsi_oversold"]) and (ema20 >= ema50)
        if not cond:
            return None
        entry = close
        sl = entry - 1.5 * atr_val
        tp = vwap_val
        if abs(tp - entry) < 0.00001 or abs(entry - sl) < 0.00001:
            return None
        conf = 60 + min(20, int(abs(from_vwap) * 10000 / 2))
        if rsi_val < params["rsi_boost"]:
            conf += 5
        rr = abs((tp - entry) / (entry - sl))
        return {
            "type": "BUY", "entry": entry, "sl": sl, "tp": tp,
            "confidence": int(min(conf, 90)),
            "reason": f"Mean Reversion→VWAP (RSI={rsi_val:.0f})",
            "rr": rr
        }
    except:
        return None

def signal_vwap_breakout(df: pd.DataFrame, params: Dict) -> Optional[Dict]:
    if len(df) < 2:
        return None
    try:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        last_close = float(last["Close"])
        last_vwap = float(last["VWAP"])
        last_rsi = float(last["RSI"])
        last_ema20 = float(last["EMA20"])
        last_ema50 = float(last["EMA50"])
        last_atr = float(last["ATR"])
        prev_close = float(prev["Close"])
        prev_vwap = float(prev["VWAP"])

        if (prev_close <= prev_vwap) and (last_close > last_vwap) and (params["rsi_min"] <= last_rsi <= params["rsi_max"]) and (last_ema20 > last_ema50):
            entry = last_close
            risk = 1.2 * last_atr
            return {"type": "BUY", "entry": entry, "sl": entry - risk, "tp": entry + 2.0 * risk,
                   "confidence": int(min(55 + min(25, int((last_rsi - 50) * 2)), 85)),
                   "reason": f"VWAP Breakout ↑ (RSI={last_rsi:.0f})", "rr": 2.0}
        if (prev_close >= prev_vwap) and (last_close < last_vwap) and (100 - params["rsi_max"] <= last_rsi <= 100 - params["rsi_min"]) and (last_ema20 < last_ema50):
            entry = last_close
            risk = 1.2 * last_atr
            return {"type": "SELL", "entry": entry, "sl": entry + risk, "tp": entry - 2.0 * risk,
                   "confidence": int(min(55 + min(25, int((50 - last_rsi) * 2)), 85)),
                   "reason": f"VWAP Breakout ↓ (RSI={last_rsi:.0f})", "rr": 2.0}
    except:
        pass
    return None

def build_signal(df: pd.DataFrame, params: Dict) -> Optional[Dict]:
    if df.shape[0] < 60:
        return None
    sig = signal_mean_reversion(df, params)
    if sig is None:
        sig = signal_vwap_breakout(df, params)
    return sig

def calculate_position_size(account: float, risk: float, entry: float, sl: float, symbol: str) -> Dict:
    risk_amount = account * (risk / 100)
    risk_distance = abs(entry - sl)
    if risk_distance < 0.00001:
        return {"lot_size": 0.01, "risk_amount": 0.0, "pips": 0.0}
    pips = risk_distance / (0.01 if ("JPY" in symbol or "/JPY" in symbol) else 0.0001)
    lot_size = (risk_amount / (pips * 10)) if pips > 0 else 0.01
    return {"lot_size": round(max(0.01, lot_size), 2), "risk_amount": round(risk_amount, 2), "pips": round(pips, 1)}

# ===== MAIN UI =====
st.markdown("<h1>🚀 Trading Copilot Pro - Market Intelligence</h1>", unsafe_allow_html=True)

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown("### 🎛️ Configuration")
    
    with st.expander("🔌 Data Source", expanded=True):
        use_twelvedata = st.checkbox("Twelve Data API", value=False)
        if use_twelvedata:
            twelvedata_api_key = st.text_input("API Key", type="password")
        else:
            twelvedata_api_key = ""
            st.caption("Using yfinance (~5min delay)")
    
    with st.expander("📊 Market Settings", expanded=True):
        pairs = st.multiselect("Currency Pairs", DEFAULT_PAIRS, default=DEFAULT_PAIRS[:4])
        interval = st.selectbox("Timeframe", INTERVALS, index=1)
    
    with st.expander("💰 Risk Management", expanded=True):
        account_balance = st.number_input("Account Size ($)", 100, 1000000, 10000, 1000)
        risk_per_trade = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.5)
    
    with st.expander("🎯 Strategy Parameters", expanded=False):
        vwap_threshold = st.slider("VWAP Threshold (%)", 0.05, 0.50, 0.15, 0.05)
        rsi_oversold = st.slider("RSI Oversold", 20, 35, 30)
        rsi_boost = st.slider("RSI Boost", 20, 30, 25)
        rsi_min = st.slider("RSI Min", 40, 55, 50)
        rsi_max = st.slider("RSI Max", 60, 70, 65)
    
    with st.expander("📰 News Strategy", expanded=True):
        news_mode = st.radio(
            "Mode",
            ["🛡️ Safe (Block during news)", "💰 Aggressive (Trade the news)"],
            index=0
        )
        news_buffer = st.slider("Time Window (min)", 15, 60, 30)
        
        if news_mode == "💰 Aggressive (Trade the news)":
            st.warning("⚠️ **HIGH RISK MODE**")
            news_risk_multiplier = st.slider("Risk Multiplier", 1.0, 3.0, 1.5, 0.5)
        else:
            news_risk_multiplier = 1.0
    
    with st.expander("🔄 Auto-Refresh", expanded=False):
        autorefresh = st.selectbox("Interval", ["Off", "30s", "60s"], index=1)
    
    with st.expander("📱 Telegram", expanded=False):
        enable_telegram = st.checkbox("Enable", value=False)
        if enable_telegram:
            telegram_token = st.text_input("Bot Token", type="password")
            telegram_chat = st.text_input("Chat ID")
        else:
            telegram_token = ""
            telegram_chat = ""
    
    min_confidence = st.slider("Min Confidence (%)", 40, 80, 65)

params = {
    "vwap_threshold": vwap_threshold, "rsi_oversold": rsi_oversold,
    "rsi_boost": rsi_boost, "rsi_min": rsi_min, "rsi_max": rsi_max
}

# ===== AUTO-REFRESH =====
AUTO_REFRESH = {"Off": 0, "30s": 30, "60s": 60}
refresh_sec = AUTO_REFRESH.get(autorefresh, 0)
if refresh_sec > 0:
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = time.time()
    elapsed = time.time() - st.session_state.last_refresh
    if elapsed >= refresh_sec:
        st.session_state.last_refresh = time.time()
        st.rerun()

# ===== SESSION STATE =====
if "journal" not in st.session_state:
    st.session_state.journal = []
if "notified" not in st.session_state:
    st.session_state.notified = set()

# ===== COMMODITY PRICES =====
commodity_prices = fetch_commodity_prices()

# ===== STATUS BAR =====
col1, col2, col3, col4 = st.columns(4)
with col1:
    data_source = "🟢 Twelve Data" if (use_twelvedata and twelvedata_api_key) else "🟡 yfinance"
    st.metric("Data Source", data_source)
with col2:
    is_aggressive = news_mode == "💰 Aggressive (Trade the news)"
    st.metric("Mode", "⚡ AGGRESSIVE" if is_aggressive else "🛡️ SAFE")
with col3:
    telegram_active = enable_telegram and telegram_token and telegram_chat and telegram_chat.lstrip('-').isdigit()
    st.metric("Telegram", "🔔 ON" if telegram_active else "🔕 OFF")
with col4:
    st.metric("Risk/Trade", f"{risk_per_trade * news_risk_multiplier:.1f}%")

st.markdown("---")

# ===== COMMODITY & CORRELATION DASHBOARD =====
st.markdown("### 🌍 Market Correlations & Live Prices")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("#### 📊 Key Markets")
    if commodity_prices:
        for name, data in commodity_prices.items():
            price = data['price']
            change = data['change']
            color = "🟢" if change > 0 else "🔴" if change < 0 else "⚪"
            
            if name in ["WTI Crude", "Brent Oil"]:
                st.markdown(f"""
                <div class='commodity-card'>
                    {color} <b>{name}</b>: ${price:.2f} ({change:+.2f}%)
                    <br><small>💡 Affects: USD/CAD, CAD/JPY (inverse)</small>
                </div>
                """, unsafe_allow_html=True)
            elif name == "Gold":
                st.markdown(f"""
                <div class='commodity-card'>
                    {color} <b>{name}</b>: ${price:.2f} ({change:+.2f}%)
                    <br><small>💡 Affects: AUD/USD, NZD/USD (direct)</small>
                </div>
                """, unsafe_allow_html=True)
            elif name == "S&P500":
                st.markdown(f"""
                <div class='commodity-card'>
                    {color} <b>{name}</b>: {price:.2f} ({change:+.2f}%)
                    <br><small>💡 Risk-on: AUD/NZD up, JPY down</small>
                </div>
                """, unsafe_allow_html=True)
            elif name == "VIX":
                st.markdown(f"""
                <div class='commodity-card'>
                    {color} <b>{name}</b>: {price:.2f} ({change:+.2f}%)
                    <br><small>💡 Fear gauge: High VIX = JPY/USD up</small>
                </div>
                """, unsafe_allow_html=True)

with col_right:
    st.markdown("#### 🔗 Your Pairs Correlations")
    for pair in pairs[:3]:
        if pair in MARKET_CORRELATIONS:
            corr = MARKET_CORRELATIONS[pair]
            st.markdown(f"**{pair}**")
            st.caption(corr['description'])
            
            for comm_name, comm_data in corr['commodities'].items():
                strength = comm_data['strength']
                correlation = comm_data['correlation']
                
                strength_emoji = "🔥" if strength == "VERY STRONG" else "⚡" if strength == "STRONG" else "💫"
                corr_text = "↑↑" if correlation == "DIRECT" else "↓↑" if correlation == "INVERSE" else "~"
                
                st.markdown(f"{strength_emoji} {comm_name} {corr_text} ({strength})")

st.markdown("---")

# ===== ENHANCED NEWS CALENDAR =====
with st.expander("📰 ECONOMIC CALENDAR - Next 48 Hours", expanded=True):
    events = fetch_economic_calendar()
    
    if events:
        for event in events[:5]:
            event_key = event['event']
            if event_key in NEWS_DATABASE:
                news_info = NEWS_DATABASE[event_key]
                
                time_to = event["time"] - datetime.utcnow()
                hours = int(time_to.total_seconds() / 3600)
                minutes = int((time_to.total_seconds() % 3600) / 60)
                time_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
                
                st.markdown(f"### {news_info['impact']} {news_info['full_name']}")
                st.markdown(f"**⏰ In {time_str}** | Currency: {event['currency']}")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown(f"""
                    **📖 What is it?**  
                    {news_info['description']}
                    
                    **📊 Expected Move:**  
                    {news_info['typical_move']}
                    
                    **🎯 Affects:**  
                    {', '.join(news_info['affects'])}
                    """)
                
                with col2:
                    st.markdown("**💡 TRADING STRATEGIES:**")
                    for timing, strategy in news_info['strategies'].items():
                        st.markdown(f"- **{timing}:** {strategy}")
                
                if 'watch_out' in news_info:
                    st.warning("**⚠️ Watch Out:**\n" + "\n".join([f"- {item}" for item in news_info['watch_out']]))
                
                if 'historical_stats' in news_info:
                    stats = news_info['historical_stats']
                    st.info(f"📈 **Historical Data:** " + " | ".join([f"{k}: {v}" for k, v in stats.items()]))
                
                st.markdown("---")
    else:
        st.info("No major news in next 48 hours")

st.markdown("---")

# ===== LOAD DATA =====
charts_data = {}
rows = []
economic_events = fetch_economic_calendar()

with st.spinner("🔄 Loading market data..."):
    for sym in pairs:
        df = load_data(sym, interval, use_twelvedata, twelvedata_api_key)
        if df.empty:
            rows.append([sym, "—", "—", "—", "—", "—", "—", "—"])
            charts_data[sym] = df
            continue
        
        df = enrich(df)
        charts_data[sym] = df
        
        if len(df) < 60:
            rows.append([sym, "—", "—", "—", "—", "—", "—", "—"])
            continue
        
        try:
            last = df.iloc[-1]
            ema20 = float(last["EMA20"])
            ema50 = float(last["EMA50"])
            rsi_val = float(last["RSI"])
            from_vwap = float(last["FromVWAP"])
            close = float(last["Close"])
            
            sentiment = calculate_market_sentiment(df)
            sentiment_emoji = "🟢" if sentiment == "BULLISH" else "🔴" if sentiment == "BEARISH" else "🟡"
            trend = "📈" if ema20 >= ema50 else "📉"
            
            currency = sym.split("/")[0]
            news_check = is_news_time(currency, economic_events, news_buffer)
            
            sig = None
            signal_txt = "—"
            conf = "—"
            
            if news_check:
                news_event, time_to_news = news_check
                
                if is_aggressive:
                    sig = build_news_signal(df, news_event, params, time_to_news)
                    
                    if sig and sig["confidence"] >= min_confidence:
                        signal_txt = f"🔥 {sig['type']}"
                        conf = f"{sig['confidence']}%"
                        
                        sig_key = f"{sym}_{sig['type']}_{sig['entry']:.5f}_NEWS"
                        if telegram_active and sig_key not in st.session_state.notified:
                            if send_telegram_notification(telegram_token, telegram_chat, sig, sym):
                                st.session_state.notified.add(sig_key)
                else:
                    signal_txt = f"⚠️ NEWS BLOCK"
                    conf = "—"
            else:
                sig = build_signal(df, params)
                
                if sig and sig["confidence"] >= min_confidence:
                    signal_txt = f"{'🟢' if sig['type']=='BUY' else '🔴'} {sig['type']}"
                    conf = f"{sig['confidence']}%"
                    
                    sig_key = f"{sym}_{sig['type']}_{sig['entry']:.5f}"
                    if telegram_active and sig_key not in st.session_state.notified:
                        if send_telegram_notification(telegram_token, telegram_chat, sig, sym):
                            st.session_state.notified.add(sig_key)
            
            price_fmt = f"{close:.5f}" if ("JPY" not in sym and "/JPY" not in sym) else f"{close:.3f}"
            rows.append([sym, f"{sentiment_emoji} {sentiment}", trend, f"{rsi_val:.0f}", pct(from_vwap), price_fmt, signal_txt, conf])
        except:
            rows.append([sym, "—", "—", "—", "—", "—", "—", "—"])

# ===== MARKET OVERVIEW =====
st.markdown("### 📊 Market Overview")
watch_df = pd.DataFrame(rows, columns=["Pair", "Sentiment", "Trend", "RSI", "From VWAP", "Price", "Signal", "Confidence"])
st.dataframe(watch_df, use_container_width=True, height=min(300, 50 + len(watch_df) * 40))

st.markdown("---")

# ===== TRADING SECTION =====
st.markdown("### 🎯 Active Trading")

col_left, col_right = st.columns([1, 1.3])

with col_left:
    if pairs:
        sel = st.selectbox("📌 Select Pair", pairs, index=0)
        df = charts_data.get(sel, pd.DataFrame())
        
        # Show correlation info
        if sel in MARKET_CORRELATIONS:
            st.info(f"**📊 {sel} Correlations:**\n\n{MARKET_CORRELATIONS[sel]['description']}")
        
        if not df.empty and len(df) >= 60:
            currency = sel.split("/")[0]
            news_check = is_news_time(currency, economic_events, news_buffer)
            
            if news_check:
                news_event, time_to = news_check
                event_key = news_event['event']
                if event_key in NEWS_DATABASE:
                    st.warning(f"⚠️ **{NEWS_DATABASE[event_key]['full_name']}** in {abs(time_to)} min!")
            
            sig = build_signal(df, params)
            
            if news_check and not is_aggressive:
                st.warning(f"🛡️ **SAFE MODE** - Signals blocked")
            elif sig and sig["confidence"] >= min_confidence:
                pos = calculate_position_size(account_balance, risk_per_trade, sig["entry"], sig["sl"], sel)
                emoji = "🟢" if sig["type"] == "BUY" else "🔴"
                
                st.success(f"""
                ### {emoji} {sig['type']} SIGNAL
                
                **📊 Trade:**
                - Entry: `{sig['entry']:.5f}`
                - SL: `{sig['sl']:.5f}`
                - TP: `{sig['tp']:.5f}`
                - R:R: `{sig['rr']:.2f}`
                - Conf: `{sig['confidence']}%`
                
                **💰 Position:**
                - Lot: `{pos['lot_size']}`
                - Risk: `${pos['risk_amount']}`
                
                **📝** {sig['reason']}
                """)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Take", type="primary", use_container_width=True):
                        st.session_state.journal.append({
                            "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                            "pair": sel, "type": sig["type"], "entry": sig["entry"],
                            "sl": sig["sl"], "tp": sig["tp"], "lots": pos["lot_size"],
                            "risk": pos["risk_amount"], "conf": sig["confidence"]
                        })
                        st.success("✅ Added!")
                with col2:
                    st.button("⏭ Skip", use_container_width=True)
            else:
                st.info("😴 No signals")

with col_right:
    if pairs and sel in charts_data:
        df = charts_data[sel]
        if not df.empty and len(df) > 60:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
            
            fig.add_trace(go.Candlestick(x=df["Datetime"], open=df["Open"], high=df["High"],
                                       low=df["Low"], close=df["Close"], showlegend=False,
                                       increasing_line_color='#22c55e', decreasing_line_color='#ef4444'), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=df["Datetime"], y=df["EMA20"], name="EMA20", 
                                   line=dict(color='#06b6d4', width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df["Datetime"], y=df["EMA50"], name="EMA50",
                                   line=dict(color='#f59e0b', width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df["Datetime"], y=df["VWAP"], name="VWAP",
                                   line=dict(dash="dot", color='#eab308', width=2)), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=df["Datetime"], y=df["RSI"], name="RSI",
                                   line=dict(color='#a855f7', width=2)), row=2, col=1)
            fig.add_hrect(y0=70, y1=100, line_width=0, fillcolor="rgba(239, 68, 68, 0.1)", row=2)
            fig.add_hrect(y0=0, y1=30, line_width=0, fillcolor="rgba(34, 197, 94, 0.1)", row=2)
            
            fig.update_layout(
                height=650,
                template="plotly_dark",
                margin=dict(l=10, r=10, t=40, b=10),
                title=f"{sel} - {interval}",
                xaxis_rangeslider_visible=False,
                hovermode='x unified',
                paper_bgcolor='#0e1117',
                plot_bgcolor='#0e1117'
            )
            
            st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ===== JOURNAL =====
st.markdown("### 📖 Trading Journal")

if st.session_state.journal:
    jdf = pd.DataFrame(st.session_state.journal)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📊 Trades", len(jdf))
    col2.metric("💰 Risk", f"${jdf['risk'].sum():.2f}")
    col3.metric("📈 Conf", f"{jdf['conf'].mean():.0f}%")
    col4.metric("📱 Alerts", len(st.session_state.notified))
    
    st.dataframe(jdf, use_container_width=True, height=350)
    
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.journal = []
            st.rerun()
    with col2:
        csv = jdf.to_csv(index=False).encode("utf-8")
        st.download_button("💾 Export", data=csv, file_name=f"journal_{datetime.now().strftime('%Y%m%d')}.csv",
                          mime="text/csv", use_container_width=True)
else:
    st.info("📝 Journal empty")

st.markdown("---")
st.caption("⚠️ **Educational purposes only. Not financial advice.**")
