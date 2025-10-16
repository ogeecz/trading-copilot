import time
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
from typing import Optional, Dict, List

# -------------------------
# Configuration
# -------------------------
DEFAULT_PAIRS = ["USDCAD=X", "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "NZDUSD=X"]
INTERVALS = ["1m", "5m", "15m", "30m", "1h"]

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

def true_range(df: pd.DataFrame) -> pd.Series:
    """True Range for ATR calculation"""
    prev_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range"""
    tr = true_range(df)
    return tr.rolling(window).mean()

def vwap(df: pd.DataFrame) -> pd.Series:
    """Volume Weighted Average Price"""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vol = df["Volume"].replace(0, np.nan).fillna(1.0)
    cum_vol = vol.cumsum()
    cum_pv = (tp * vol).cumsum()
    return cum_pv / cum_vol

def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0):
    """Bollinger Bands"""
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, sma, lower

def pct(x: float) -> str:
    """Format percentage"""
    return f"{x*100:.2f}%"

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
            return df
        df = df.rename_axis("Datetime").reset_index()
        if hasattr(df["Datetime"].iloc[0], "tzinfo") and df["Datetime"].dt.tz is not None:
            df["Datetime"] = df["Datetime"].dt.tz_convert("UTC").dt.tz_localize(None)
        return df
    except Exception as e:
        st.error(f"⚠️ Chyba při načítání {symbol}: {str(e)}")
        return pd.DataFrame()

def enrich(df: pd.DataFrame, ema_fast: int = 20, ema_slow: int = 50) -> pd.DataFrame:
    """Add technical indicators to dataframe"""
    if df.empty:
        return df
    out = df.copy()
    
    # Moving averages
    out["EMA20"] = ema(out["Close"], ema_fast)
    out["EMA50"] = ema(out["Close"], ema_slow)
    out["SMA200"] = out["Close"].rolling(200).mean()
    
    # Momentum
    out["RSI"] = rsi(out["Close"], 14)
    
    # Volatility
    out["VWAP"] = vwap(out).bfill().ffill()
    out["ATR"] = atr(out, 14).bfill().ffill()
    
    # Bollinger Bands
    bb_upper, bb_mid, bb_lower = bollinger_bands(out["Close"], 20, 2.0)
    out["BB_Upper"] = bb_upper
    out["BB_Mid"] = bb_mid
    out["BB_Lower"] = bb_lower
    
    # Derived metrics (safe calculation to avoid division by zero)
    # FromVWAP calculation
    out["FromVWAP"] = 0.0
    mask = out["VWAP"] != 0
    out.loc[mask, "FromVWAP"] = (out.loc[mask, "Close"] / out.loc[mask, "VWAP"]) - 1.0
    
    out["TrendUp"] = (out["EMA20"] >= out["EMA50"]).astype(int)
    
    # BB_Width calculation
    out["BB_Width"] = 0.0
    mask_bb = out["BB_Mid"] != 0
    out.loc[mask_bb, "BB_Width"] = (out.loc[mask_bb, "BB_Upper"] - out.loc[mask_bb, "BB_Lower"]) / out.loc[mask_bb, "BB_Mid"]
    
    return out

# -------------------------
# Signal Generation
# -------------------------
def signal_mean_reversion(df: pd.DataFrame, vwap_threshold: float, rsi_oversold: int, rsi_boost: int) -> Optional[Dict]:
    """Mean reversion strategy to VWAP"""
    row = df.iloc[-1]
    cond = (
        (row["FromVWAP"] <= -vwap_threshold/100) and 
        (row["RSI"] < rsi_oversold) and 
        (row["EMA20"] >= row["EMA50"])
    )
    if not cond:
        return None
    
    entry = float(row["Close"])
    sl = entry - 1.5 * float(row["ATR"])
    tp = float(row["VWAP"])
    
    conf = 60 + min(20, int(abs(row["FromVWAP"]) * 10000 / 2))
    if row["RSI"] < rsi_boost:
        conf += 5
    
    return {
        "type": "BUY",
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "confidence": int(min(conf, 90)),
        "reason": f"Mean Reversion→VWAP (RSI={row['RSI']:.0f}, {pct(row['FromVWAP'])} od VWAP)",
        "rr": abs((tp - entry) / (entry - sl))
    }

def signal_vwap_breakout(df: pd.DataFrame, rsi_min: int, rsi_max: int) -> Optional[Dict]:
    """VWAP breakout strategy"""
    if len(df) < 2:
        return None
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    sig = None

    # Bullish breakout
    if (prev["Close"] <= prev["VWAP"]) and (last["Close"] > last["VWAP"]) and \
       (rsi_min <= last["RSI"] <= rsi_max) and (last["EMA20"] > last["EMA50"]):
        entry = float(last["Close"])
        risk = 1.2 * float(last["ATR"])
        sl = entry - risk
        tp = entry + 2.0 * risk
        conf = 55 + min(25, int((last["RSI"] - 50) * 2))
        sig = {
            "type": "BUY",
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "confidence": int(min(conf, 85)),
            "reason": f"VWAP Breakout ↑ (RSI={last['RSI']:.0f})",
            "rr": 2.0
        }

    # Bearish breakout
    if (prev["Close"] >= prev["VWAP"]) and (last["Close"] < last["VWAP"]) and \
       (100 - rsi_max <= last["RSI"] <= 100 - rsi_min) and (last["EMA20"] < last["EMA50"]):
        entry = float(last["Close"])
        risk = 1.2 * float(last["ATR"])
        sl = entry + risk
        tp = entry - 2.0 * risk
        conf = 55 + min(25, int((50 - last["RSI"]) * 2))
        sig = {
            "type": "SELL",
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "confidence": int(min(conf, 85)),
            "reason": f"VWAP Breakout ↓ (RSI={last['RSI']:.0f})",
            "rr": 2.0
        }

    return sig

def signal_bollinger_bounce(df: pd.DataFrame) -> Optional[Dict]:
    """Bollinger Bands bounce strategy"""
    if len(df) < 2:
        return None
    
    last = df.iloc[-1]
    
    # Bounce from lower band
    if last["Close"] <= last["BB_Lower"] and last["RSI"] < 30:
        entry = float(last["Close"])
        sl = entry - 1.5 * float(last["ATR"])
        tp = float(last["BB_Mid"])
        return {
            "type": "BUY",
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "confidence": 65,
            "reason": f"BB Bounce ↑ (RSI={last['RSI']:.0f})",
            "rr": abs((tp - entry) / (entry - sl))
        }
    
    # Bounce from upper band
    if last["Close"] >= last["BB_Upper"] and last["RSI"] > 70:
        entry = float(last["Close"])
        sl = entry + 1.5 * float(last["ATR"])
        tp = float(last["BB_Mid"])
        return {
            "type": "SELL",
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "confidence": 65,
            "reason": f"BB Bounce ↓ (RSI={last['RSI']:.0f})",
            "rr": abs((tp - entry) / (entry - sl))
        }
    
    return None

def build_signal(df: pd.DataFrame, params: Dict) -> Optional[Dict]:
    """Build signal from available strategies"""
    if df.shape[0] < 60:
        return None
    
    # Try strategies in order of preference
    sig = signal_mean_reversion(df, params["vwap_threshold"], params["rsi_oversold"], params["rsi_boost"])
    if sig is None:
        sig = signal_vwap_breakout(df, params["rsi_min"], params["rsi_max"])
    if sig is None and params["enable_bb"]:
        sig = signal_bollinger_bounce(df)
    
    return sig

# -------------------------
# Risk Management
# -------------------------
def calculate_position_size(account_balance: float, risk_pct: float, entry: float, sl: float, symbol: str) -> Dict:
    """Calculate position size based on risk percentage"""
    risk_amount = account_balance * (risk_pct / 100)
    risk_distance = abs(entry - sl)
    
    # Pip value calculation (simplified)
    if "JPY" in symbol:
        pip_value = 0.01
        pips = risk_distance / pip_value
    else:
        pip_value = 0.0001
        pips = risk_distance / pip_value
    
    # Standard lot = 100,000 units, pip value ≈ $10
    lot_size = risk_amount / (pips * 10)
    
    return {
        "lot_size": round(lot_size, 2),
        "risk_amount": round(risk_amount, 2),
        "pips": round(pips, 1)
    }

# -------------------------
# Backtesting
# -------------------------
def run_backtest(df: pd.DataFrame, params: Dict) -> Dict:
    """Simple backtest of strategy"""
    if df.shape[0] < 100:
        return {"error": "Nedostatek dat pro backtest"}
    
    trades = []
    equity = [10000]  # Starting capital
    
    for i in range(60, len(df)):
        hist = df.iloc[:i+1].copy()
        sig = build_signal(hist, params)
        
        if sig:
            # Simulate trade
            entry = sig["entry"]
            sl = sig["sl"]
            tp = sig["tp"]
            
            # Look ahead to see what happened
            future = df.iloc[i+1:min(i+50, len(df))]
            if future.empty:
                continue
            
            if sig["type"] == "BUY":
                hit_sl = (future["Low"] <= sl).any()
                hit_tp = (future["High"] >= tp).any()
            else:
                hit_sl = (future["High"] >= sl).any()
                hit_tp = (future["Low"] <= tp).any()
            
            if hit_tp:
                pnl = abs(tp - entry)
                result = "WIN"
            elif hit_sl:
                pnl = -abs(entry - sl)
                result = "LOSS"
            else:
                continue
            
            equity.append(equity[-1] + pnl * 10000)  # Scale for visualization
            trades.append({
                "entry": entry,
                "exit": tp if result == "WIN" else sl,
                "pnl": pnl,
                "result": result,
                "type": sig["type"]
            })
    
    if not trades:
        return {"error": "Žádné obchody v backtestingu"}
    
    wins = [t for t in trades if t["result"] == "WIN"]
    losses = [t for t in trades if t["result"] == "LOSS"]
    
    return {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(trades) * 100 if trades else 0,
        "avg_win": np.mean([t["pnl"] for t in wins]) if wins else 0,
        "avg_loss": np.mean([abs(t["pnl"]) for t in losses]) if losses else 0,
        "profit_factor": abs(sum([t["pnl"] for t in wins]) / sum([t["pnl"] for t in losses])) if losses else 0,
        "equity_curve": equity,
        "trades": trades
    }

# -------------------------
# Correlation Analysis
# -------------------------
def calculate_correlations(charts_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Calculate price correlations between pairs"""
    closes = {}
    for sym, df in charts_data.items():
        if not df.empty and len(df) > 50:
            closes[sym] = df.set_index("Datetime")["Close"]
    
    if len(closes) < 2:
        return pd.DataFrame()
    
    price_df = pd.DataFrame(closes)
    returns = price_df.pct_change().dropna()
    corr = returns.corr()
    
    return corr

# -------------------------
# UI Setup
# -------------------------
st.set_page_config(
    page_title="Trading Copilot Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stAlert > div { padding: 0.5rem 1rem; }
    .metric-card { 
        background: #1e1e1e; 
        padding: 1rem; 
        border-radius: 0.5rem; 
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🚀 Trading Copilot Pro — FX Signály</h1>", unsafe_allow_html=True)

# -------------------------
# Sidebar Configuration
# -------------------------
with st.sidebar:
    st.markdown("### ⚙️ Nastavení")
    
    # Market selection
    pairs = st.multiselect(
        "Měnové páry",
        DEFAULT_PAIRS,
        default=DEFAULT_PAIRS[:4]
    )
    
    interval = st.selectbox("Časový interval", INTERVALS, index=1)
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
    
    with st.expander("Mean Reversion", expanded=False):
        vwap_threshold = st.slider("VWAP práh (%)", 0.05, 0.50, 0.15, 0.05)
        rsi_oversold = st.slider("RSI přeprodáno", 20, 35, 30)
        rsi_boost = st.slider("RSI extra boost", 20, 30, 25)
    
    with st.expander("VWAP Breakout", expanded=False):
        rsi_min = st.slider("RSI min", 40, 55, 50)
        rsi_max = st.slider("RSI max", 60, 70, 65)
    
    with st.expander("Další strategie", expanded=False):
        enable_bb = st.checkbox("Bollinger Bands", value=True)
    
    st.markdown("---")
    st.markdown("### 🔄 Auto-refresh")
    
    autorefresh = st.selectbox(
        "Obnovovací interval",
        ["Vypnuto", "30s", "60s", "120s", "300s"],
        index=2
    )
    
    st.markdown("---")
    st.markdown("### 🔔 Notifikace")
    
    enable_sound = st.checkbox("Zvukové upozornění", value=False)
    min_confidence = st.slider("Min. důvěra signálu (%)", 40, 80, 60)

# Strategy parameters
params = {
    "vwap_threshold": vwap_threshold,
    "rsi_oversold": rsi_oversold,
    "rsi_boost": rsi_boost,
    "rsi_min": rsi_min,
    "rsi_max": rsi_max,
    "enable_bb": enable_bb
}

# -------------------------
# Auto-refresh Logic
# -------------------------
AUTO_REFRESH_SEC = {
    "Vypnuto": 0,
    "30s": 30,
    "60s": 60,
    "120s": 120,
    "300s": 300
}
refresh_sec = AUTO_REFRESH_SEC.get(autorefresh, 0)

if refresh_sec > 0:
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = time.time()
    
    time_since_refresh = time.time() - st.session_state.last_refresh
    
    if time_since_refresh >= refresh_sec:
        st.session_state.last_refresh = time.time()
        st.rerun()
    
    # Show countdown
    remaining = int(refresh_sec - time_since_refresh)
    st.sidebar.info(f"⏱️ Další refresh za {remaining}s")

# -------------------------
# Initialize Session State
# -------------------------
if "journal" not in st.session_state:
    st.session_state.journal = []

# -------------------------
# Main Dashboard
# -------------------------
st.caption(
    f"📊 Data: yfinance • Interval: {interval} • "
    f"Aktualizace: {datetime.utcnow().strftime('%H:%M:%S')} UTC"
)

# Load and process data
with st.spinner("Načítám data..."):
    charts_data = {}
    rows = []
    
    for sym in pairs:
        df = enrich(load_intraday(sym, interval=interval, lookback_days=lookback), ema_fast=20, ema_slow=50)
        charts_data[sym] = df
        
        if df.empty:
            rows.append([sym, "—", "—", "—", "—", "—", "—", "—"])
            continue
        
        last = df.iloc[-1]
        trend = "📈 Up" if last["EMA20"] >= last["EMA50"] else "📉 Down"
        sig = build_signal(df, params)
        
        signal_txt = "—"
        conf = "—"
        rr = "—"
        
        if sig and sig["confidence"] >= min_confidence:
            signal_txt = f"{'🟢' if sig['type']=='BUY' else '🔴'} {sig['type']}"
            conf = f"{sig['confidence']}%"
            rr = f"{sig.get('rr', 0):.2f}"
        
        price_fmt = f"{last['Close']:.5f}" if 'JPY' not in sym else f"{last['Close']:.3f}"
        
        rows.append([
            sym,
            trend,
            f"{last['RSI']:.0f}",
            pct(last["FromVWAP"]),
            price_fmt,
            signal_txt,
            conf,
            rr
        ])

# Watchlist table
watch_df = pd.DataFrame(
    rows,
    columns=["Symbol", "Trend", "RSI", "Od VWAP", "Cena", "Signál", "Důvěra", "R:R"]
)

st.markdown("### 📊 Přehled trhů")
st.dataframe(
    watch_df,
    use_container_width=True,
    height=min(250, 50 + len(watch_df) * 35)
)

# -------------------------
# Correlation Matrix
# -------------------------
with st.expander("📈 Korelační matice párů", expanded=False):
    corr_matrix = calculate_correlations(charts_data)
    if not corr_matrix.empty:
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale="RdYlGn",
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10}
        ))
        fig_corr.update_layout(
            height=400,
            template="plotly_dark",
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Nedostatek dat pro korelační analýzu")

# -------------------------
# Detail View
# -------------------------
st.markdown("### 🎯 Detail & Trading")

col_left, col_right = st.columns([1.0, 1.3])

with col_left:
    if pairs:
        selected_sym = st.selectbox("Vyber pár", pairs, index=0, key="selected_pair")
        df = charts_data.get(selected_sym, pd.DataFrame())
        
        if not df.empty:
            sig = build_signal(df, params)
            
            if sig and sig["confidence"] >= min_confidence:
                # Calculate position size
                pos_info = calculate_position_size(
                    account_balance,
                    risk_per_trade,
                    sig["entry"],
                    sig["sl"],
                    selected_sym
                )
                
                # Signal card
                signal_emoji = "🟢" if sig["type"] == "BUY" else "🔴"
                st.success(
                    f"### {signal_emoji} {sig['type']} Signal\n\n"
                    f"**Entry:** {sig['entry']:.5f} | **SL:** {sig['sl']:.5f} | **TP:** {sig['tp']:.5f}\n\n"
                    f"**R:R:** {sig.get('rr', 0):.2f} | **Důvěra:** {sig['confidence']}%\n\n"
                    f"_{sig['reason']}_"
                )
                
                # Position sizing info
                st.info(
                    f"💰 **Position Size:** {pos_info['lot_size']} lots\n\n"
                    f"**Riziko:** ${pos_info['risk_amount']} ({risk_per_trade}%)\n\n"
                    f"**SL Distance:** {pos_info['pips']} pips"
                )
                
                # Action buttons
                c1, c2, c3 = st.columns(3)
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
                            "reason": sig["reason"],
                            "status": "OPEN"
                        })
                        st.success("✅ Přidáno do deníku!")
                        if enable_sound:
                            st.markdown(
                                '<audio autoplay><source src="https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3"></audio>',
                                unsafe_allow_html=True
                            )
                
                with c2:
                    st.button("⏭ Přeskočit", use_container_width=True)
                
                with c3:
                    if st.session_state.journal:
                        journal_csv = pd.DataFrame(st.session_state.journal).to_csv(index=False)
                        st.download_button(
                            "💾 Export",
                            data=journal_csv.encode("utf-8"),
                            file_name=f"trading_journal_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
            else:
                st.info("😴 Momentálně žádný silný signál")
                st.caption(f"Min. důvěra: {min_confidence}% (nastav v sidebaru)")

with col_right:
    if selected_sym and selected_sym in charts_data:
        df = charts_data[selected_sym]
        
        if not df.empty and len(df) > 60:
            # Main price chart
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3]
            )
            
            # Candlestick
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
            
            # Indicators
            fig.add_trace(go.Scatter(x=df["Datetime"], y=df["EMA20"], name="EMA20", line=dict(color="cyan")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df["Datetime"], y=df["EMA50"], name="EMA50", line=dict(color="orange")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df["Datetime"], y=df["VWAP"], name="VWAP", line=dict(dash="dot", color="yellow")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df["Datetime"], y=df["BB_Upper"], name="BB Upper", line=dict(dash="dot", color="gray"), opacity=0.5), row=1, col=1)
            fig.add_trace(go.Scatter(x=df["Datetime"], y=df["BB_Lower"], name="BB Lower", line=dict(dash="dot", color="gray"), opacity=0.5), row=1, col=1)
            
            # RSI
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

# -------------------------
# Backtest Section
# -------------------------
with st.expander("📊 Backtest strategií", expanded=False):
    if selected_sym and selected_sym in charts_data:
        df_bt = charts_data[selected_sym]
        
        if not df_bt.empty:
            if st.button("▶️ Spustit backtest", type="primary"):
                with st.spinner("Probíhá backtest..."):
                    results = run_backtest(df_bt, params)
                    
                    if "error" in results:
                        st.warning(results["error"])
                    else:
                        # Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Počet obchodů", results["total_trades"])
                        col2.metric("Win Rate", f"{results['win_rate']:.1f}%")
                        col3.metric("Profit Factor", f"{results['profit_factor']:.2f}")
                        col4.metric("Avg Win/Loss", f"{results['avg_win']:.5f} / {results['avg_loss']:.5f}")
                        
                        # Equity curve
                        if results["equity_curve"]:
                            fig_eq = go.Figure()
                            fig_eq.add_trace(go.Scatter(
                                y=results["equity_curve"],
                                mode='lines',
                                name='Equity',
                                line=dict(color='cyan')
                            ))
                            fig_eq.update_layout(
                                title="Equity Curve",
                                height=300,
                                template="plotly_dark",
                                margin=dict(l=10, r=10, t=40, b=10)
                            )
                            st.plotly_chart(fig_eq, use_container_width=True)
                        
                        # Trade list
                        if results["trades"]:
                            st.markdown("**Poslední obchody:**")
                            trades_df = pd.DataFrame(results["trades"][-10:])
                            st.dataframe(trades_df, use_container_width=True)

# -------------------------
# Trading Journal
# -------------------------
st.markdown("### 📖 Obchodní deník")

if st.session_state.journal:
    journal_df = pd.DataFrame(st.session_state.journal)
    
    # Summary metrics
    if len(journal_df) > 0:
        col1, col2, col3, col4 = st.columns(4)
        total_trades = len(journal_df)
        total_risk = journal_df["risk"].sum()
        avg_confidence = journal_df["confidence"].mean()
        
        col1.metric("Celkem obchodů", total_trades)
        col2.metric("Celkové riziko", f"${total_risk:.2f}")
        col3.metric("Průměrná důvěra", f"{avg_confidence:.1f}%")
        col4.metric("Otevřené pozice", len(journal_df[journal_df["status"] == "OPEN"]))
    
    # Journal table
    st.dataframe(
        journal_df,
        use_container_width=True,
        height=min(400, 50 + len(journal_df) * 35)
    )
    
    # Actions
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🗑️ Vymazat deník", use_container_width=True):
            st.session_state.journal = []
            st.rerun()
else:
    st.info("📝 Deník je prázdný. Začni tím, že přidáš obchod z aktivního signálu.")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption(
    "⚠️ **Upozornění:** Tento nástroj slouží pouze pro vzdělávací účely. "
    "Není to investiční poradenství. Trading je rizikový. "
    "Data z yfinance mohou mít zpoždění ~5 minut."
)
st.caption(f"💻 Trading Copilot Pro v1.0 | Made with ❤️ by ogeecz")
