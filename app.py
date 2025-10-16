import time
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
from typing import Optional, Dict
import requests
import json

# -------------------------
# Configuration
# -------------------------
DEFAULT_PAIRS = ["USDCAD=X", "EURUSD=X", "GBPUSD=X", "USDJPY=X"]
INTERVALS = ["5m", "15m", "30m", "1h"]

# -------------------------
# Helpers (Indicators)
# -------------------------
def ema(series: pd.Series, window: int) -> pd.Series:
    """Exponential Moving Average"""
    return series.ewm(span=window, adjust=False).mean()

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index"""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    gain = up.ewm(alpha=1/window, min_periods=window).mean()
    loss = down.ewm(alpha=1/window, min_periods=window).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range"""
    prev_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def vwap(df: pd.DataFrame) -> pd.Series:
    """Volume Weighted Average Price"""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vol = df["Volume"].copy()
    vol = vol.replace(0, 1.0)
    cum_vol = vol.cumsum()
    cum_pv = (tp * vol).cumsum()
    return cum_pv / cum_vol

def pct(x: float) -> str:
    """Format percentage"""
    return f"{x*100:.2f}%"

# -------------------------
# Notifications
# -------------------------
def send_telegram_notification(bot_token: str, chat_id: str, signal: Dict, symbol: str) -> bool:
    """Send signal notification via Telegram"""
    # Clean inputs
    bot_token = bot_token.strip() if bot_token else ""
    chat_id = chat_id.strip() if chat_id else ""
    
    if not bot_token or not chat_id or len(bot_token) < 10:
        return False
    
    try:
        emoji = "🟢" if signal["type"] == "BUY" else "🔴"
        
        message = (
            f"{emoji} <b>{signal['type']} SIGNAL - {symbol}</b>\n\n"
            f"💰 <b>Entry:</b> {signal['entry']:.5f}\n"
            f"🛑 <b>Stop Loss:</b> {signal['sl']:.5f}\n"
            f"🎯 <b>Take Profit:</b> {signal['tp']:.5f}\n"
            f"📊 <b>R:R:</b> {signal.get('rr', 0):.2f}\n"
            f"✅ <b>Důvěra:</b> {signal['confidence']}%\n\n"
            f"📝 <i>{signal['reason']}</i>\n\n"
            f"🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )
        
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML"
        }
        
        response = requests.post(url, data=payload, timeout=10)
        
        if response.status_code != 200:
            error_msg = response.json().get('description', 'Unknown error')
            st.error(f"❌ Telegram chyba: {error_msg}")
            return False
        
        return True
        
    except requests.exceptions.Timeout:
        st.error("❌ Telegram timeout - zkus to znovu")
        return False
    except Exception as e:
        st.error(f"❌ Telegram chyba: {str(e)}")
        return False

def send_discord_notification(webhook_url: str, signal: Dict, symbol: str) -> bool:
    """Send signal notification via Discord webhook"""
    if not webhook_url or webhook_url == "YOUR_WEBHOOK_URL":
        return False
    
    try:
        emoji = "🟢" if signal["type"] == "BUY" else "🔴"
        color = 0x00ff00 if signal["type"] == "BUY" else 0xff0000
        
        embed = {
            "title": f"{emoji} {signal['type']} SIGNAL - {symbol}",
            "color": color,
            "fields": [
                {"name": "💰 Entry", "value": f"{signal['entry']:.5f}", "inline": True},
                {"name": "🛑 Stop Loss", "value": f"{signal['sl']:.5f}", "inline": True},
                {"name": "🎯 Take Profit", "value": f"{signal['tp']:.5f}", "inline": True},
                {"name": "📊 R:R", "value": f"{signal.get('rr', 0):.2f}", "inline": True},
                {"name": "✅ Důvěra", "value": f"{signal['confidence']}%", "inline": True},
                {"name": "📝 Důvod", "value": signal['reason'], "inline": False}
            ],
            "footer": {
                "text": f"Trading Copilot Pro • {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
            }
        }
        
        data = {"embeds": [embed]}
        
        response = requests.post(webhook_url, json=data, timeout=10)
        return response.status_code == 204
        
    except Exception as e:
        st.warning(f"Chyba při odesílání Discord notifikace: {str(e)}")
        return False

# -------------------------
# Data Loading
# -------------------------
@st.cache_data(ttl=300, show_spinner=False)
def load_intraday(symbol: str, interval: str = "5m", lookback_days: int = 7) -> pd.DataFrame:
    """Load intraday data from yfinance with error handling"""
    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=lookback_days)
        df = yf.download(
            tickers=symbol,
            interval=interval,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=False,
            prepost=False,
            threads=True,
        )
        if df.empty:
            return pd.DataFrame()
        
        df = df.rename_axis("Datetime").reset_index()
        
        if hasattr(df["Datetime"].iloc[0], "tzinfo") and df["Datetime"].dt.tz is not None:
            df["Datetime"] = df["Datetime"].dt.tz_convert("UTC").dt.tz_localize(None)
        
        return df
    except Exception as e:
        st.error(f"⚠️ Chyba při načítání {symbol}: {str(e)}")
        return pd.DataFrame()

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to dataframe"""
    if df.empty or len(df) < 60:
        return df
    
    try:
        out = df.copy()
        
        # Moving averages
        out["EMA20"] = ema(out["Close"], 20)
        out["EMA50"] = ema(out["Close"], 50)
        
        # RSI
        out["RSI"] = rsi(out["Close"], 14)
        
        # VWAP
        out["VWAP"] = vwap(out)
        out["VWAP"] = out["VWAP"].bfill().ffill()
        
        # ATR
        out["ATR"] = atr(out, 14)
        out["ATR"] = out["ATR"].bfill().ffill()
        
        # FromVWAP - calculated safely
        vwap_vals = out["VWAP"].values
        close_vals = out["Close"].values
        from_vwap = np.zeros(len(out))
        
        for i in range(len(out)):
            if vwap_vals[i] > 0.0001:
                from_vwap[i] = (close_vals[i] / vwap_vals[i]) - 1.0
        
        out["FromVWAP"] = from_vwap
        out["TrendUp"] = (out["EMA20"] >= out["EMA50"]).astype(int)
        
        return out
        
    except Exception as e:
        st.warning(f"⚠️ Chyba při výpočtu indikátorů: {str(e)}")
        return df

# -------------------------
# Signal Generation
# -------------------------
def signal_mean_reversion(df: pd.DataFrame, vwap_threshold: float, rsi_oversold: int, rsi_boost: int) -> Optional[Dict]:
    """Mean reversion strategy to VWAP"""
    if len(df) < 60:
        return None
    
    try:
        row = df.iloc[-1]
        
        if not all(k in row.index for k in ["FromVWAP", "RSI", "EMA20", "EMA50", "Close", "ATR", "VWAP"]):
            return None
        
        # Get scalar values
        from_vwap = float(row["FromVWAP"])
        rsi_val = float(row["RSI"])
        ema20 = float(row["EMA20"])
        ema50 = float(row["EMA50"])
        close = float(row["Close"])
        atr_val = float(row["ATR"])
        vwap_val = float(row["VWAP"])
        
        cond = (
            (from_vwap <= -vwap_threshold/100) and 
            (rsi_val < rsi_oversold) and 
            (ema20 >= ema50)
        )
        
        if not cond:
            return None
        
        entry = close
        sl = entry - 1.5 * atr_val
        tp = vwap_val
        
        if abs(tp - entry) < 0.00001 or abs(entry - sl) < 0.00001:
            return None
        
        conf = 60 + min(20, int(abs(from_vwap) * 10000 / 2))
        if rsi_val < rsi_boost:
            conf += 5
        
        rr = abs((tp - entry) / (entry - sl))
        
        return {
            "type": "BUY",
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "confidence": int(min(conf, 90)),
            "reason": f"Mean Reversion→VWAP (RSI={rsi_val:.0f}, {pct(from_vwap)} od VWAP)",
            "rr": rr
        }
    except Exception:
        return None

def signal_vwap_breakout(df: pd.DataFrame, rsi_min: int, rsi_max: int) -> Optional[Dict]:
    """VWAP breakout strategy"""
    if len(df) < 2:
        return None
    
    try:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        if not all(k in last.index for k in ["Close", "VWAP", "RSI", "EMA20", "EMA50", "ATR"]):
            return None
        
        # Get scalar values
        last_close = float(last["Close"])
        last_vwap = float(last["VWAP"])
        last_rsi = float(last["RSI"])
        last_ema20 = float(last["EMA20"])
        last_ema50 = float(last["EMA50"])
        last_atr = float(last["ATR"])
        
        prev_close = float(prev["Close"])
        prev_vwap = float(prev["VWAP"])

        # Bullish breakout
        if (prev_close <= prev_vwap) and (last_close > last_vwap) and \
           (rsi_min <= last_rsi <= rsi_max) and (last_ema20 > last_ema50):
            entry = last_close
            risk = 1.2 * last_atr
            sl = entry - risk
            tp = entry + 2.0 * risk
            conf = 55 + min(25, int((last_rsi - 50) * 2))
            return {
                "type": "BUY",
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "confidence": int(min(conf, 85)),
                "reason": f"VWAP Breakout ↑ (RSI={last_rsi:.0f})",
                "rr": 2.0
            }

        # Bearish breakout
        if (prev_close >= prev_vwap) and (last_close < last_vwap) and \
           (100 - rsi_max <= last_rsi <= 100 - rsi_min) and (last_ema20 < last_ema50):
            entry = last_close
            risk = 1.2 * last_atr
            sl = entry + risk
            tp = entry - 2.0 * risk
            conf = 55 + min(25, int((50 - last_rsi) * 2))
            return {
                "type": "SELL",
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "confidence": int(min(conf, 85)),
                "reason": f"VWAP Breakout ↓ (RSI={last_rsi:.0f})",
                "rr": 2.0
            }
    except Exception:
        pass
    
    return None

def build_signal(df: pd.DataFrame, params: Dict) -> Optional[Dict]:
    """Build signal from available strategies"""
    if df.shape[0] < 60:
        return None
    
    sig = signal_mean_reversion(df, params["vwap_threshold"], params["rsi_oversold"], params["rsi_boost"])
    if sig is None:
        sig = signal_vwap_breakout(df, params["rsi_min"], params["rsi_max"])
    
    return sig

# -------------------------
# Risk Management
# -------------------------
def calculate_position_size(account_balance: float, risk_pct: float, entry: float, sl: float, symbol: str) -> Dict:
    """Calculate position size based on risk percentage"""
    risk_amount = account_balance * (risk_pct / 100)
    risk_distance = abs(entry - sl)
    
    if risk_distance < 0.00001:
        return {"lot_size": 0.01, "risk_amount": 0.0, "pips": 0.0}
    
    if "JPY" in symbol:
        pip_value = 0.01
        pips = risk_distance / pip_value
    else:
        pip_value = 0.0001
        pips = risk_distance / pip_value
    
    lot_size = risk_amount / (pips * 10) if pips > 0 else 0.01
    
    return {
        "lot_size": round(max(0.01, lot_size), 2),
        "risk_amount": round(risk_amount, 2),
        "pips": round(pips, 1)
    }

# -------------------------
# UI Setup
# -------------------------
st.set_page_config(
    page_title="Trading Copilot Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("<h1>🚀 Trading Copilot Pro — FX Signály</h1>", unsafe_allow_html=True)

# -------------------------
# Sidebar Configuration
# -------------------------
with st.sidebar:
    st.markdown("### ⚙️ Nastavení")
    
    pairs = st.multiselect(
        "Měnové páry",
        DEFAULT_PAIRS,
        default=DEFAULT_PAIRS
    )
    
    interval = st.selectbox("Časový interval", INTERVALS, index=0)
    lookback = st.slider("Historie (dní)", 3, 30, 7)
    
    st.markdown("---")
    st.markdown("### 💰 Risk Management")
    
    account_balance = st.number_input(
        "Velikost účtu ($)",
        min_value=100,
        max_value=1000000,
        value=10000,
        step=1000
    )
    
    risk_per_trade = st.slider(
        "Riziko na obchod (%)",
        min_value=0.5,
        max_value=5.0,
        value=1.0,
        step=0.5
    )
    
    st.markdown("---")
    st.markdown("### 🎯 Parametry strategií")
    
    with st.expander("Mean Reversion", expanded=True):
        vwap_threshold = st.slider("VWAP práh (%)", 0.05, 0.50, 0.15, 0.05)
        rsi_oversold = st.slider("RSI přeprodáno", 20, 35, 30)
        rsi_boost = st.slider("RSI extra boost", 20, 30, 25)
    
    with st.expander("VWAP Breakout", expanded=True):
        rsi_min = st.slider("RSI min", 40, 55, 50)
        rsi_max = st.slider("RSI max", 60, 70, 65)
    
    st.markdown("---")
    st.markdown("### 🔄 Auto-refresh")
    
    autorefresh = st.selectbox(
        "Obnovovací interval",
        ["Vypnuto", "30s", "60s", "120s"],
        index=2
    )
    
    st.markdown("---")
    st.markdown("### 🔔 Notifikace")
    
    notification_type = st.selectbox(
        "Typ notifikace",
        ["Vypnuto", "Telegram", "Discord"],
        index=0
    )
    
    telegram_bot_token = ""
    telegram_chat_id = ""
    discord_webhook = ""
    
    if notification_type == "Telegram":
        st.info("📱 **Jak nastavit Telegram:**\n1. Otevři [@BotFather](https://t.me/botfather)\n2. Napiš `/newbot` a vytvoř bota\n3. Zkopíruj Bot Token\n4. Najdi svůj Chat ID přes [@userinfobot](https://t.me/userinfobot)")
        telegram_bot_token = st.text_input("Bot Token", type="password", placeholder="123456:ABC-DEF...")
        telegram_chat_id = st.text_input("Chat ID", placeholder="123456789")
        
        if st.button("🧪 Testovat Telegram"):
            test_signal = {
                "type": "BUY",
                "entry": 1.2345,
                "sl": 1.2300,
                "tp": 1.2400,
                "confidence": 75,
                "reason": "Test notifikace",
                "rr": 1.5
            }
            if send_telegram_notification(telegram_bot_token, telegram_chat_id, test_signal, "TEST"):
                st.success("✅ Telegram funguje!")
            else:
                st.error("❌ Zkontroluj token a chat ID")
    
    elif notification_type == "Discord":
        st.info("💬 **Jak nastavit Discord:**\n1. Otevři Server Settings\n2. Integrations → Webhooks\n3. New Webhook\n4. Zkopíruj Webhook URL")
        discord_webhook = st.text_input("Webhook URL", type="password", placeholder="https://discord.com/api/webhooks/...")
        
        if st.button("🧪 Testovat Discord"):
            test_signal = {
                "type": "BUY",
                "entry": 1.2345,
                "sl": 1.2300,
                "tp": 1.2400,
                "confidence": 75,
                "reason": "Test notifikace",
                "rr": 1.5
            }
            if send_discord_notification(discord_webhook, test_signal, "TEST"):
                st.success("✅ Discord funguje!")
            else:
                st.error("❌ Zkontroluj webhook URL")
    
    st.markdown("---")
    min_confidence = st.slider("Min. důvěra signálu (%)", 40, 80, 60)

# Strategy parameters
params = {
    "vwap_threshold": vwap_threshold,
    "rsi_oversold": rsi_oversold,
    "rsi_boost": rsi_boost,
    "rsi_min": rsi_min,
    "rsi_max": rsi_max
}

# -------------------------
# Auto-refresh Logic
# -------------------------
AUTO_REFRESH_SEC = {
    "Vypnuto": 0,
    "30s": 30,
    "60s": 60,
    "120s": 120
}
refresh_sec = AUTO_REFRESH_SEC.get(autorefresh, 0)

if refresh_sec > 0:
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = time.time()
    
    time_since_refresh = time.time() - st.session_state.last_refresh
    
    if time_since_refresh >= refresh_sec:
        st.session_state.last_refresh = time.time()
        st.rerun()
    
    remaining = int(refresh_sec - time_since_refresh)
    st.sidebar.info(f"⏱️ Další refresh za {remaining}s")

# -------------------------
# Initialize Session State
# -------------------------
if "journal" not in st.session_state:
    st.session_state.journal = []

if "notified_signals" not in st.session_state:
    st.session_state.notified_signals = set()  # Track which signals were already sent

# -------------------------
# Main Dashboard
# -------------------------
st.caption(
    f"📊 Data: yfinance • Interval: {interval} • "
    f"Aktualizace: {datetime.utcnow().strftime('%H:%M:%S')} UTC"
)

# Load and process data
charts_data = {}
rows = []

with st.spinner("Načítám data..."):
    for sym in pairs:
        df = load_intraday(sym, interval=interval, lookback_days=lookback)
        
        if df.empty:
            rows.append([sym, "—", "—", "—", "—", "—", "—"])
            charts_data[sym] = df
            continue
        
        df = enrich(df)
        charts_data[sym] = df
        
        if df.empty or len(df) < 60:
            rows.append([sym, "—", "—", "—", "—", "—", "—"])
            continue
        
        try:
            last = df.iloc[-1]
            
            # Check if required columns exist
            required_cols = ["EMA20", "EMA50", "RSI", "FromVWAP", "Close"]
            if not all(col in df.columns for col in required_cols):
                rows.append([sym, "—", "—", "—", "—", "—", "—"])
                continue
            
            # Get scalar values, not Series
            ema20_val = float(last["EMA20"])
            ema50_val = float(last["EMA50"])
            rsi_val = float(last["RSI"])
            from_vwap_val = float(last["FromVWAP"])
            close_val = float(last["Close"])
            
            trend = "📈 Up" if ema20_val >= ema50_val else "📉 Down"
            sig = build_signal(df, params)
            
            signal_txt = "—"
            conf = "—"
            rr = "—"
            
            if sig and sig["confidence"] >= min_confidence:
                signal_txt = f"{'🟢' if sig['type']=='BUY' else '🔴'} {sig['type']}"
                conf = f"{sig['confidence']}%"
                rr = f"{sig.get('rr', 0):.2f}"
                
                # Send notification if enabled and not already sent
                signal_key = f"{sym}_{sig['type']}_{sig['entry']:.5f}"
                
                if signal_key not in st.session_state.notified_signals:
                    notification_sent = False
                    
                    if notification_type == "Telegram" and telegram_bot_token and telegram_chat_id:
                        if send_telegram_notification(telegram_bot_token, telegram_chat_id, sig, sym):
                            notification_sent = True
                    
                    elif notification_type == "Discord" and discord_webhook:
                        if send_discord_notification(discord_webhook, sig, sym):
                            notification_sent = True
                    
                    if notification_sent:
                        st.session_state.notified_signals.add(signal_key)
                        # Keep only last 50 signals in memory
                        if len(st.session_state.notified_signals) > 50:
                            st.session_state.notified_signals = set(list(st.session_state.notified_signals)[-50:])
            
            price_fmt = f"{close_val:.5f}" if 'JPY' not in sym else f"{close_val:.3f}"
            
            rows.append([
                sym,
                trend,
                f"{rsi_val:.0f}",
                pct(from_vwap_val),
                price_fmt,
                signal_txt,
                conf
            ])
        except Exception as e:
            st.warning(f"Chyba u {sym}: {str(e)}")
            rows.append([sym, "—", "—", "—", "—", "—", "—"])

# Watchlist table
watch_df = pd.DataFrame(
    rows,
    columns=["Symbol", "Trend", "RSI", "Od VWAP", "Cena", "Signál", "Důvěra"]
)

st.markdown("### 📊 Přehled trhů")

# Show notification status
if notification_type != "Vypnuto":
    notif_emoji = "📱" if notification_type == "Telegram" else "💬"
    st.caption(f"{notif_emoji} Notifikace: **{notification_type}** aktivní")

st.dataframe(
    watch_df,
    use_container_width=True,
    height=min(250, 50 + len(watch_df) * 35)
)

# -------------------------
# Detail View
# -------------------------
st.markdown("### 🎯 Detail & Trading")

col_left, col_right = st.columns([1.0, 1.3])

with col_left:
    if pairs:
        selected_sym = st.selectbox("Vyber pár", pairs, index=0)
        df = charts_data.get(selected_sym, pd.DataFrame())
        
        if not df.empty and len(df) >= 60:
            sig = build_signal(df, params)
            
            if sig and sig["confidence"] >= min_confidence:
                pos_info = calculate_position_size(
                    account_balance,
                    risk_per_trade,
                    sig["entry"],
                    sig["sl"],
                    selected_sym
                )
                
                signal_emoji = "🟢" if sig["type"] == "BUY" else "🔴"
                st.success(
                    f"### {signal_emoji} {sig['type']} Signal\n\n"
                    f"**Entry:** {sig['entry']:.5f} | **SL:** {sig['sl']:.5f} | **TP:** {sig['tp']:.5f}\n\n"
                    f"**R:R:** {sig.get('rr', 0):.2f} | **Důvěra:** {sig['confidence']}%\n\n"
                    f"_{sig['reason']}_"
                )
                
                st.info(
                    f"💰 **Position Size:** {pos_info['lot_size']} lots\n\n"
                    f"**Riziko:** ${pos_info['risk_amount']} ({risk_per_trade}%)\n\n"
                    f"**SL Distance:** {pos_info['pips']} pips"
                )
                
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("✅ Vzít obchod", use_container_width=True, type="primary"):
                        st.session_state.journal.append({
                            "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                            "symbol": selected_sym,
                            "type": sig["type"],
                            "entry": sig["entry"],
                            "sl": sig["sl"],
                            "tp": sig["tp"],
                            "lot_size": pos_info["lot_size"],
                            "risk": pos_info["risk_amount"],
                            "confidence": sig["confidence"],
                            "reason": sig["reason"]
                        })
                        st.success("✅ Přidáno do deníku!")
                
                with c2:
                    st.button("⏭ Přeskočit", use_container_width=True)
            else:
                st.info("😴 Momentálně žádný silný signál")
                st.caption(f"Min. důvěra: {min_confidence}%")
        else:
            st.info("📊 Načítám data nebo nedostatek historie...")

with col_right:
    if pairs and selected_sym in charts_data:
        df = charts_data[selected_sym]
        
        if not df.empty and len(df) > 60:
            required_chart_cols = ["Datetime", "Open", "High", "Low", "Close", "EMA20", "EMA50", "VWAP", "RSI"]
            
            if all(col in df.columns for col in required_chart_cols):
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.7, 0.3]
                )
                
                fig.add_trace(
                    go.Candlestick(
                        x=df["Datetime"],
                        open=df["Open"],
                        high=df["High"],
                        low=df["Low"],
                        close=df["Close"],
                        name="Cena",
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(go.Scatter(x=df["Datetime"], y=df["EMA20"], name="EMA20", line=dict(color="cyan")), row=1, col=1)
                fig.add_trace(go.Scatter(x=df["Datetime"], y=df["EMA50"], name="EMA50", line=dict(color="orange")), row=1, col=1)
                fig.add_trace(go.Scatter(x=df["Datetime"], y=df["VWAP"], name="VWAP", line=dict(dash="dot", color="yellow")), row=1, col=1)
                
                fig.add_trace(go.Scatter(x=df["Datetime"], y=df["RSI"], name="RSI", line=dict(color="purple")), row=2, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
                
                fig.update_layout(
                    height=600,
                    template="plotly_dark",
                    margin=dict(l=10, r=10, t=30, b=10),
                    title=f"{selected_sym} - {interval}",
                    xaxis_rangeslider_visible=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Chybí data pro zobrazení grafu")

# -------------------------
# Trading Journal
# -------------------------
st.markdown("### 📖 Obchodní deník")

if st.session_state.journal:
    journal_df = pd.DataFrame(st.session_state.journal)
    
    if len(journal_df) > 0:
        col1, col2, col3 = st.columns(3)
        total_trades = len(journal_df)
        total_risk = journal_df["risk"].sum()
        avg_confidence = journal_df["confidence"].mean()
        
        col1.metric("Celkem obchodů", total_trades)
        col2.metric("Celkové riziko", f"${total_risk:.2f}")
        col3.metric("Průměrná důvěra", f"{avg_confidence:.1f}%")
    
    st.dataframe(
        journal_df,
        use_container_width=True,
        height=min(400, 50 + len(journal_df) * 35)
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🗑️ Vymazat deník"):
            st.session_state.journal = []
            st.rerun()
    with col2:
        if st.session_state.journal:
            journal_csv = pd.DataFrame(st.session_state.journal).to_csv(index=False)
            st.download_button(
                "💾 Export CSV",
                data=journal_csv.encode("utf-8"),
                file_name=f"trading_journal_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
else:
    st.info("📝 Deník je prázdný. Začni tím, že přidáš obchod z aktivního signálu.")

# -------------------------
# Footer
# -------------------------
st.markdown("---")

if notification_type != "Vypnuto":
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption(
            "⚠️ **Upozornění:** Tento nástroj slouží pouze pro vzdělávací účely. "
            "Není to investiční poradenství. Trading je rizikový. "
            "Data z yfinance mohou mít zpoždění ~5 minut."
        )
    with col2:
        total_notified = len(st.session_state.notified_signals)
        st.metric("🔔 Odesláno notifikací", total_notified)
else:
    st.caption(
        "⚠️ **Upozornění:** Tento nástroj slouží pouze pro vzdělávací účely. "
        "Není to investiční poradenství. Trading je rizikový. "
        "Data z yfinance mohou mít zpoždění ~5 minut."
    )
