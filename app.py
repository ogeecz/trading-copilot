import time
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# -------------------------
# Helpers (indikátory)
# -------------------------
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

def true_range(df: pd.DataFrame):
    prev_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr(df: pd.DataFrame, window: int = 14):
    tr = true_range(df)
    return tr.rolling(window).mean()

def vwap(df: pd.DataFrame):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vol = df["Volume"].replace(0, np.nan).fillna(1.0)
    cum_vol = vol.cumsum()
    cum_pv = (tp * vol).cumsum()
    return cum_pv / cum_vol

def pct(x):
    return f"{x*100:.2f}%"

# -------------------------
# Data
# -------------------------
DEFAULT_PAIRS = ["USDCAD=X", "EURUSD=X", "GBPUSD=X", "USDJPY=X"]

@st.cache_data(ttl=60, show_spinner=False)
def load_intraday(symbol: str, interval: str = "5m", lookback_days: int = 7) -> pd.DataFrame:
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

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["EMA20"] = ema(out["Close"], 20)
    out["EMA50"] = ema(out["Close"], 50)
    out["RSI"] = rsi(out["Close"], 14)
    out["VWAP"] = vwap(out).fillna(method="bfill").fillna(method="ffill")
    out["ATR"] = atr(out, 14).fillna(method="bfill").fillna(method="ffill")

    # Bezpečný výpočet FromVWAP
    out["FromVWAP"] = np.where(out["VWAP"] != 0, (out["Close"] / out["VWAP"]) - 1.0, 0.0)
    out["TrendUp"] = (out["EMA20"] >= out["EMA50"]).astype(int)
    return out

# -------------------------
# Signály
# -------------------------
def signal_mean_reversion(df: pd.DataFrame):
    row = df.iloc[-1]
    cond = (row["FromVWAP"] <= -0.0015) and (row["RSI"] < 30) and (row["EMA20"] >= row["EMA50"])
    if not cond:
        return None
    entry = float(row["Close"])
    sl = entry - 1.5 * float(row["ATR"])
    tp = float(row["VWAP"])
    conf = 60 + min(20, int(abs(row["FromVWAP"]) * 10000 / 2)) + (5 if row["RSI"] < 25 else 0)
    return {"type": "BUY", "entry": entry, "sl": sl, "tp": tp, "confidence": int(min(conf, 90)),
            "reason": "MR→VWAP (RSI<30, pod VWAP, EMA20≥EMA50)"}

def signal_vwap_breakout(df: pd.DataFrame):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    sig = None

    if (prev["Close"] <= prev["VWAP"]) and (last["Close"] > last["VWAP"]) and (50 <= last["RSI"] <= 65) and (last["EMA20"] > last["EMA50"]):
        entry = float(last["Close"])
        risk = 1.2 * float(last["ATR"])
        sl = entry - risk
        tp = entry + 2.0 * risk
        sig = {"type": "BUY", "entry": entry, "sl": sl, "tp": tp,
               "confidence": 55 + min(25, int((last["RSI"] - 50) * 2)),
               "reason": "VWAP breakout ↑ (RSI 50–65, EMA20>EMA50)"}

    if (prev["Close"] >= prev["VWAP"]) and (last["Close"] < last["VWAP"]) and (35 <= last["RSI"] <= 50) and (last["EMA20"] < last["EMA50"]):
        entry = float(last["Close"])
        risk = 1.2 * float(last["ATR"])
        sl = entry + risk
        tp = entry - 2.0 * risk
        sig = {"type": "SELL", "entry": entry, "sl": sl, "tp": tp,
               "confidence": 55 + min(25, int((50 - last["RSI"]) * 2)),
               "reason": "VWAP breakout ↓ (RSI 35–50, EMA20<EMA50)"}

    return sig

def build_signal(df: pd.DataFrame):
    if df.shape[0] < 60:
        return None
    sig = signal_mean_reversion(df)
    if sig is None:
        sig = signal_vwap_breakout(df)
    return sig

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Trading Copilot (FX)", layout="wide", initial_sidebar_state="collapsed")

st.markdown("<h2 style='margin-top:0'>Trading Copilot — FX signály (paper)</h2>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Nastavení")
    pairs = st.multiselect("Páry (yfinance symboly)", DEFAULT_PAIRS, default=DEFAULT_PAIRS)
    interval = st.selectbox("Interval", ["5m", "15m"], index=0)
    lookback = st.slider("Lookback (dní)", 3, 30, 7)
    autorefresh = st.selectbox("Auto-refresh", ["Vypnuto", "30 s", "60 s", "120 s"], index=2)

# --- Auto-refresh (nová verze) ---
AUTO_REFRESH_MS = {"Vypnuto": 0, "30 s": 30000, "60 s": 60000, "120 s": 120000}
refresh_ms = AUTO_REFRESH_MS.get(autorefresh, 0)

if refresh_ms > 0:
    try:
        st.query_params.update({"_": str(int(time.time()))})
    except Exception:
        pass
    import threading
    def _ref():
        time.sleep(refresh_ms / 1000.0)
        st.rerun()
    threading.Thread(target=_ref, daemon=True).start()

# -------------------------
# Data & hlavní logika
# -------------------------
if "journal" not in st.session_state:
    st.session_state.journal = []

st.caption(f"Data: yfinance • Interval: {interval} • Poslední aktualizace: {datetime.utcnow().strftime('%H:%M:%S')} UTC")

rows = []
charts_data = {}
for sym in pairs:
    df = enrich(load_intraday(sym, interval=interval, lookback_days=lookback))
    charts_data[sym] = df
    if df.empty:
        rows.append([sym, "—", "—", "—", "—", "—", "—"])
        continue
    last = df.iloc[-1]
    trend = "Up" if last["EMA20"] >= last["EMA50"] else "Down"
    sig = build_signal(df)
    signal_txt = sig["type"] if sig else "Neutral"
    conf = f"{sig['confidence']}%" if sig else "—"
    rows.append([sym, trend, f"{last['RSI']:.0f}", pct(last["FromVWAP"]),
                 f"{last['Close']:.5f}" if 'JPY' not in sym else f"{last['Close']:.3f}",
                 signal_txt, conf])

watch_df = pd.DataFrame(rows, columns=["Symbol", "Trend", "RSI", "Od VWAP", "Cena", "Signál", "Důvěra"])
st.dataframe(watch_df, use_container_width=True, height=220)

left, right = st.columns([1.05, 1.2])
with left:
    sym = st.selectbox("Detail páru", pairs, index=0)
    df = charts_data[sym]
    if not df.empty:
        sig = build_signal(df)
        if sig:
            st.success(f"{sym} • {sig['type']} • Entry {sig['entry']:.5f} • SL {sig['sl']:.5f} • TP {sig['tp']:.5f} "
                       f"• Důvěra {sig['confidence']}%  \n_{sig['reason']}_")
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("✅ Vzal bych obchod", use_container_width=True):
                    st.session_state.journal.append({
                        "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                        "symbol": sym,
                        "type": sig["type"],
                        "entry": sig["entry"],
                        "sl": sig["sl"],
                        "tp": sig["tp"],
                        "note": sig["reason"],
                    })
            with c2:
                st.button("⏭ Přeskočit", use_container_width=True)
            with c3:
                st.download_button("⬇ Export deníku (CSV)",
                                   data=pd.DataFrame(st.session_state.journal).to_csv(index=False).encode("utf-8"),
                                   file_name="journal.csv",
                                   mime="text/csv",
                                   use_container_width=True)
        else:
            st.info("Žádný silný signál pro vybraný pár.")

with right:
    df = charts_data.get(sym, pd.DataFrame())
    if not df.empty and df.shape[0] > 60:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df["Datetime"], open=df["Open"], high=df["High"],
                                     low=df["Low"], close=df["Close"], name="Cena", showlegend=False))
        fig.add_trace(go.Scatter(x=df["Datetime"], y=df["EMA20"], name="EMA20"))
        fig.add_trace(go.Scatter(x=df["Datetime"], y=df["EMA50"], name="EMA50"))
        fig.add_trace(go.Scatter(x=df["Datetime"], y=df["VWAP"], name="VWAP", line=dict(dash="dot")))
        fig.update_layout(height=420, template="plotly_dark", margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df["Datetime"], y=df["RSI"], name="RSI"))
        fig2.add_hrect(y0=30, y1=70, line_width=0, fillcolor="gray", opacity=0.1)
        fig2.update_layout(height=160, template="plotly_dark", margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig2, use_container_width=True)

st.markdown("### Deník (paper)")
journal_df = pd.DataFrame(st.session_state.journal)
if journal_df.empty:
    st.write("Zatím prázdné. Klikni na „Vzal bych obchod“ u signálu.")
else:
    st.dataframe(journal_df, use_container_width=True, height=220)

st.caption("Upozornění: Vzdělávací účely. Nejde o investiční doporučení. Data: yfinance (přibližně 5min).")
