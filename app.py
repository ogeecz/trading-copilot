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

# ===== KONFIGURACE =====
DEFAULT_PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CAD", "AUD/USD", "NZD/USD"]
INTERVALS = ["5min", "15min", "30min", "1h"]

# Twelve Data to yfinance mapping
TD_TO_YF = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X", 
    "USD/JPY": "USDJPY=X",
    "USD/CAD": "USDCAD=X",
    "AUD/USD": "AUDUSD=X",
    "NZD/USD": "NZDUSD=X"
}

# ===== TWELVE DATA API =====
def fetch_twelvedata(symbol: str, interval: str, api_key: str) -> pd.DataFrame:
    """Fetch real-time forex data from Twelve Data API"""
    try:
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": 500,
            "apikey": api_key,
            "format": "JSON"
        }
        
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
        
        df = pd.DataFrame(rows)
        df = df.sort_values("Datetime").reset_index(drop=True)
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        return df
        
    except Exception as e:
        st.warning(f"Twelve Data chyba pro {symbol}: {str(e)}")
        return pd.DataFrame()

# ===== TECHNICKÉ INDIKÁTORY =====
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
    vol = df["Volume"].copy()
    vol = vol.replace(0, 1.0)
    cum_vol = vol.cumsum()
    cum_pv = (tp * vol).cumsum()
    return cum_pv / cum_vol

def pct(x: float) -> str:
    return f"{x*100:.2f}%"

# ===== TELEGRAM NOTIFIKACE =====
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
            f"✅ <b>Důvěra:</b> {signal['confidence']}%\n\n"
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
    """Load data from Twelve Data or yfinance fallback"""
    
    if use_twelvedata and api_key:
        # Twelve Data API
        df = fetch_twelvedata(symbol, interval, api_key)
        
        if not df.empty:
            return df
        else:
            st.warning(f"⚠️ Twelve Data selhala, fallback na yfinance pro {symbol}")
    
    # Fallback na yfinance
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

# ===== SIGNÁLY =====
def signal_mean_reversion(df: pd.DataFrame, vwap_threshold: float, rsi_oversold: int, rsi_boost: int) -> Optional[Dict]:
    if len(df) < 60:
        return None
    try:
        row = df.iloc[-1]
        if not all(k in row.index for k in ["FromVWAP", "RSI", "EMA20", "EMA50", "Close", "ATR", "VWAP"]):
            return None
        from_vwap = float(row["FromVWAP"])
        rsi_val = float(row["RSI"])
        ema20 = float(row["EMA20"])
        ema50 = float(row["EMA50"])
        close = float(row["Close"])
        atr_val = float(row["ATR"])
        vwap_val = float(row["VWAP"])
        
        cond = (from_vwap <= -vwap_threshold/100) and (rsi_val < rsi_oversold) and (ema20 >= ema50)
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
            "type": "BUY", "entry": entry, "sl": sl, "tp": tp,
            "confidence": int(min(conf, 90)),
            "reason": f"Mean Reversion→VWAP (RSI={rsi_val:.0f}, {pct(from_vwap)} od VWAP)",
            "rr": rr
        }
    except:
        return None

def signal_vwap_breakout(df: pd.DataFrame, rsi_min: int, rsi_max: int) -> Optional[Dict]:
    if len(df) < 2:
        return None
    try:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        if not all(k in last.index for k in ["Close", "VWAP", "RSI", "EMA20", "EMA50", "ATR"]):
            return None
        last_close = float(last["Close"])
        last_vwap = float(last["VWAP"])
        last_rsi = float(last["RSI"])
        last_ema20 = float(last["EMA20"])
        last_ema50 = float(last["EMA50"])
        last_atr = float(last["ATR"])
        prev_close = float(prev["Close"])
        prev_vwap = float(prev["VWAP"])

        if (prev_close <= prev_vwap) and (last_close > last_vwap) and (rsi_min <= last_rsi <= rsi_max) and (last_ema20 > last_ema50):
            entry = last_close
            risk = 1.2 * last_atr
            return {"type": "BUY", "entry": entry, "sl": entry - risk, "tp": entry + 2.0 * risk,
                   "confidence": int(min(55 + min(25, int((last_rsi - 50) * 2)), 85)),
                   "reason": f"VWAP Breakout ↑ (RSI={last_rsi:.0f})", "rr": 2.0}
        if (prev_close >= prev_vwap) and (last_close < last_vwap) and (100 - rsi_max <= last_rsi <= 100 - rsi_min) and (last_ema20 < last_ema50):
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
    sig = signal_mean_reversion(df, params["vwap_threshold"], params["rsi_oversold"], params["rsi_boost"])
    if sig is None:
        sig = signal_vwap_breakout(df, params["rsi_min"], params["rsi_max"])
    return sig

def calculate_position_size(account_balance: float, risk_pct: float, entry: float, sl: float, symbol: str) -> Dict:
    risk_amount = account_balance * (risk_pct / 100)
    risk_distance = abs(entry - sl)
    if risk_distance < 0.00001:
        return {"lot_size": 0.01, "risk_amount": 0.0, "pips": 0.0}
    if "JPY" in symbol or "/JPY" in symbol:
        pips = risk_distance / 0.01
    else:
        pips = risk_distance / 0.0001
    lot_size = risk_amount / (pips * 10) if pips > 0 else 0.01
    return {"lot_size": round(max(0.01, lot_size), 2), "risk_amount": round(risk_amount, 2), "pips": round(pips, 1)}

# ===== STREAMLIT UI =====
st.set_page_config(page_title="Trading Copilot Pro", layout="wide", initial_sidebar_state="expanded")
st.markdown("<h1>🚀 Trading Copilot Pro — FX Signály</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🔌 Data Source")
    
    use_twelvedata = st.checkbox("Použít Twelve Data API (real-time)", value=False)
    
    if use_twelvedata:
        st.info("📊 **Jak získat Twelve Data API:**\n\n1. Jdi na [twelvedata.com](https://twelvedata.com)\n2. Vytvoř FREE účet\n3. Dashboard → API Key\n4. Zkopíruj klíč\n\n**Free tier:** 800 requests/den")
        twelvedata_api_key = st.text_input("Twelve Data API Key", type="password", placeholder="abc123def456...")
        
        if twelvedata_api_key:
            st.success("✅ API Key vyplněn")
        else:
            st.warning("⚠️ Vyplň API Key")
    else:
        st.info("📊 Použije se yfinance (5min zpoždění)")
        twelvedata_api_key = ""
    
    st.markdown("---")
    st.markdown("### ⚙️ Nastavení")
    
    pairs = st.multiselect("Měnové páry", DEFAULT_PAIRS, default=DEFAULT_PAIRS[:4])
    interval = st.selectbox("Časový interval", INTERVALS, index=0)
    
    st.markdown("---")
    st.markdown("### 💰 Risk Management")
    account_balance = st.number_input("Velikost účtu ($)", 100, 1000000, 10000, 1000)
    risk_per_trade = st.slider("Riziko na obchod (%)", 0.5, 5.0, 1.0, 0.5)
    
    st.markdown("---")
    st.markdown("### 🎯 Parametry")
    with st.expander("Mean Reversion"):
        vwap_threshold = st.slider("VWAP práh (%)", 0.05, 0.50, 0.15, 0.05)
        rsi_oversold = st.slider("RSI přeprodáno", 20, 35, 30)
        rsi_boost = st.slider("RSI boost", 20, 30, 25)
    with st.expander("VWAP Breakout"):
        rsi_min = st.slider("RSI min", 40, 55, 50)
        rsi_max = st.slider("RSI max", 60, 70, 65)
    
    st.markdown("---")
    st.markdown("### 🔄 Auto-refresh")
    autorefresh = st.selectbox("Interval", ["Vypnuto", "30s", "60s"], index=1)
    
    st.markdown("---")
    min_confidence = st.slider("Min. důvěra (%)", 40, 80, 60)
    
    st.markdown("---")
    st.markdown("### 📱 Telegram")
    enable_telegram = st.checkbox("Zapnout notifikace", value=False)
    if enable_telegram:
        st.info("💡 Vyplň údaje a otestuj před použitím!")
        telegram_token = st.text_input("Bot Token", type="password", placeholder="123:ABC...", 
                                       help="Z @BotFather")
        telegram_chat = st.text_input("Chat ID", placeholder="123456789",
                                      help="Z @userinfobot - musí to být ČÍSLO!")
        
        # Validate and test
        if telegram_token and telegram_chat:
            if not telegram_chat.lstrip('-').isdigit():
                st.error("❌ Chat ID musí být číslo (ne @username)!")
            else:
                if st.button("🧪 Test zprávu", use_container_width=True, type="primary"):
                    with st.spinner("Odesílám..."):
                        test_signal = {
                            "type": "BUY",
                            "entry": 1.0850,
                            "sl": 1.0820,
                            "tp": 1.0900,
                            "confidence": 75,
                            "reason": "🎉 Test z Trading Copilot - FUNGUJE!",
                            "rr": 1.67
                        }
                        if send_telegram_notification(telegram_token, telegram_chat, test_signal, "EUR_USD"):
                            st.success("✅ Telegram FUNGUJE! Zkontroluj zprávu v Telegramu 📱")
                        else:
                            st.error("❌ Nepodařilo se odeslat. Zkontroluj:\n- Bot Token je správný\n- Chat ID je správné číslo\n- Klikl jsi START ve svém botovi")
        else:
            st.warning("⚠️ Vyplň Bot Token a Chat ID")
    else:
        telegram_token = ""
        telegram_chat = ""
        st.caption("Notifikace vypnuty")

params = {
    "vwap_threshold": vwap_threshold, "rsi_oversold": rsi_oversold,
    "rsi_boost": rsi_boost, "rsi_min": rsi_min, "rsi_max": rsi_max
}

# Auto-refresh
AUTO_REFRESH = {"Vypnuto": 0, "30s": 30, "60s": 60}
refresh_sec = AUTO_REFRESH.get(autorefresh, 0)

if refresh_sec > 0:
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = time.time()
    elapsed = time.time() - st.session_state.last_refresh
    if elapsed >= refresh_sec:
        st.session_state.last_refresh = time.time()
        st.rerun()
    st.sidebar.info(f"⏱️ Refresh za {int(refresh_sec - elapsed)}s")

if "journal" not in st.session_state:
    st.session_state.journal = []
if "notified" not in st.session_state:
    st.session_state.notified = set()

# Info bar
data_source = "🟢 Twelve Data (Real-time)" if (use_twelvedata and twelvedata_api_key) else "🟡 yfinance (~5min delay)"
telegram_active = enable_telegram and telegram_token and telegram_chat and telegram_chat.lstrip('-').isdigit()
notif_status = "🔔 Telegram ON" if telegram_active else "🔕 Telegram OFF"
st.caption(f"{data_source} • Interval: {interval} • {notif_status} • {datetime.utcnow().strftime('%H:%M:%S')} UTC")

# Load data
charts_data = {}
rows = []

with st.spinner("Načítám data..."):
    for sym in pairs:
        df = load_data(sym, interval, use_twelvedata, twelvedata_api_key)
        if df.empty:
            rows.append([sym, "—", "—", "—", "—", "—", "—"])
            charts_data[sym] = df
            continue
        
        df = enrich(df)
        charts_data[sym] = df
        
        if len(df) < 60:
            rows.append([sym, "—", "—", "—", "—", "—", "—"])
            continue
        
        try:
            last = df.iloc[-1]
            ema20 = float(last["EMA20"])
            ema50 = float(last["EMA50"])
            rsi_val = float(last["RSI"])
            from_vwap = float(last["FromVWAP"])
            close = float(last["Close"])
            
            trend = "📈" if ema20 >= ema50 else "📉"
            sig = build_signal(df, params)
            
            signal_txt = "—"
            conf = "—"
            
            if sig and sig["confidence"] >= min_confidence:
                signal_txt = f"{'🟢' if sig['type']=='BUY' else '🔴'} {sig['type']}"
                conf = f"{sig['confidence']}%"
                
                # Send Telegram (only if properly configured)
                sig_key = f"{sym}_{sig['type']}_{sig['entry']:.5f}"
                if telegram_active and sig_key not in st.session_state.notified:
                    if send_telegram_notification(telegram_token, telegram_chat, sig, sym):
                        st.session_state.notified.add(sig_key)
            
            price_fmt = f"{close:.5f}" if ("JPY" not in sym and "/JPY" not in sym) else f"{close:.3f}"
            rows.append([sym, trend, f"{rsi_val:.0f}", pct(from_vwap), price_fmt, signal_txt, conf])
        except:
            rows.append([sym, "—", "—", "—", "—", "—", "—"])

watch_df = pd.DataFrame(rows, columns=["Pár", "Trend", "RSI", "Od VWAP", "Cena", "Signál", "Důvěra"])

st.markdown("### 📊 Přehled")
st.dataframe(watch_df, use_container_width=True, height=240)

# Detail
st.markdown("### 🎯 Trading")
col1, col2 = st.columns([1, 1.3])

with col1:
    if pairs:
        sel = st.selectbox("Pár", pairs, index=0)
        df = charts_data.get(sel, pd.DataFrame())
        
        if not df.empty and len(df) >= 60:
            sig = build_signal(df, params)
            if sig and sig["confidence"] >= min_confidence:
                pos = calculate_position_size(account_balance, risk_per_trade, sig["entry"], sig["sl"], sel)
                emoji = "🟢" if sig["type"] == "BUY" else "🔴"
                st.success(f"### {emoji} {sig['type']}\n**Entry:** {sig['entry']:.5f} | **SL:** {sig['sl']:.5f} | **TP:** {sig['tp']:.5f}\n**R:R:** {sig['rr']:.2f} | **Důvěra:** {sig['confidence']}%\n_{sig['reason']}_")
                st.info(f"💰 {pos['lot_size']} lots • Riziko: ${pos['risk_amount']} • {pos['pips']} pips")
                
                if st.button("✅ Vzít obchod", use_container_width=True, type="primary"):
                    st.session_state.journal.append({
                        "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                        "pair": sel, "type": sig["type"], "entry": sig["entry"],
                        "sl": sig["sl"], "tp": sig["tp"], "lots": pos["lot_size"],
                        "risk": pos["risk_amount"], "conf": sig["confidence"]
                    })
                    st.success("✅ V deníku!")
            else:
                st.info("😴 Žádný signál")

with col2:
    if pairs and sel in charts_data:
        df = charts_data[sel]
        if not df.empty and len(df) > 60:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=df["Datetime"], open=df["Open"], high=df["High"],
                                       low=df["Low"], close=df["Close"], showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=df["Datetime"], y=df["EMA20"], name="EMA20", line=dict(color="cyan")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df["Datetime"], y=df["EMA50"], name="EMA50", line=dict(color="orange")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df["Datetime"], y=df["VWAP"], name="VWAP", line=dict(dash="dot", color="yellow")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df["Datetime"], y=df["RSI"], name="RSI", line=dict(color="purple")), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2)
            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2)
            fig.update_layout(height=600, template="plotly_dark", margin=dict(l=10,r=10,t=30,b=10),
                            title=f"{sel} - {interval}", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

# Journal
st.markdown("### 📖 Deník")
if st.session_state.journal:
    jdf = pd.DataFrame(st.session_state.journal)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Obchodů", len(jdf))
    col2.metric("Riziko", f"${jdf['risk'].sum():.2f}")
    col3.metric("Avg důvěra", f"{jdf['conf'].mean():.0f}%")
    col4.metric("📱 Telegram", len(st.session_state.notified))
    st.dataframe(jdf, use_container_width=True, height=300)
    if st.button("🗑️ Vymazat"):
        st.session_state.journal = []
        st.rerun()
else:
    st.info("📝 Prázdný deník")
    if telegram_active:
        st.caption(f"📱 Odesláno Telegram zpráv: {len(st.session_state.notified)}")

st.caption("⚠️ Jen pro vzdělávání • Trading je rizikový • Nejde o investiční radu")
