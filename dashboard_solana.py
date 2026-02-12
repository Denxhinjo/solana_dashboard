# solana_dashboard_news_right_demo.py
# Streamlit dashboard for Solana (SOL-USD) using Coinbase (real OHLCV candles, no API key)
# Shows ONLY the last 1 hour + a small (demo) news panel on the right.

import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import feedparser
from datetime import datetime, timezone, timedelta
from urllib.parse import quote

st.set_page_config(page_title="Solana Dashboard (1H + News)", layout="wide")
st.title("Solana Trend Dashboard — Last 1 Hour + News (Demo)")

# ----------------------------
# Coinbase candles (public, no key)
# ----------------------------
@st.cache_data(ttl=60 * 2, show_spinner=False)
def fetch_coinbase_candles(product: str, start_dt: datetime, end_dt: datetime, granularity_sec: int) -> pd.DataFrame:
    """
    Coinbase Exchange candles endpoint:
    returns [time, low, high, open, close, volume] (max ~300 candles per request),
    so we paginate in windows.
    """
    url = f"https://api.exchange.coinbase.com/products/{product}/candles"
    headers = {"accept": "application/json", "user-agent": "sol-dashboard/1.0"}

    window_seconds = granularity_sec * 300
    cur = start_dt.replace(tzinfo=timezone.utc)
    end = end_dt.replace(tzinfo=timezone.utc)

    rows = []
    while cur < end:
        nxt = min(cur + timedelta(seconds=window_seconds), end)
        params = {"start": cur.isoformat(), "end": nxt.isoformat(), "granularity": granularity_sec}
        r = requests.get(url, params=params, headers=headers, timeout=30)

        if r.status_code >= 400:
            raise RuntimeError(f"Coinbase API error {r.status_code}: {r.text}\nRequest: {r.url}")

        rows.extend(r.json())
        cur = nxt
        time.sleep(0.2)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["ts", "low", "high", "open", "close", "volume"])
    df["date"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(None)
    df = df.drop(columns=["ts"]).set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def get_last_hour_candles(product: str, granularity_sec: int) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=2)  # fetch a bit more then filter (safer)
    end = now
    df = fetch_coinbase_candles(product, start, end, granularity_sec)
    if df.empty:
        return df

    # normalize to UTC-aware index for filtering
    df = df.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    cutoff = now - timedelta(hours=1)
    out = df[df.index >= cutoff]
    return out


# ----------------------------
# Indicators
# ----------------------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))


def add_indicators(df: pd.DataFrame, granularity_sec: int) -> pd.DataFrame:
    out = df.copy()
    out["price"] = out["close"]
    out["ret"] = out["price"].pct_change()

    out["ema_20"] = out["price"].ewm(span=20, adjust=False).mean()

    # RSI + MACD
    out["rsi_14"] = rsi(out["price"], 14)
    ema12 = out["price"].ewm(span=12, adjust=False).mean()
    ema26 = out["price"].ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    # Annualized vol (rough): crypto trades 24/7
    periods_per_year = (365 * 24 * 3600) / granularity_sec
    out["vol_30_ann"] = out["ret"].rolling(30).std() * np.sqrt(periods_per_year)

    # Drawdown
    peak = out["price"].cummax()
    out["drawdown"] = out["price"] / peak - 1.0
    return out


# ----------------------------
# News (demo: small list)
# ----------------------------
def google_news_rss_url(query: str) -> str:
    return f"https://news.google.com/rss/search?q={quote(query)}&hl=en-US&gl=US&ceid=US:en"

def parse_published(entry):
    for key in ("published_parsed", "updated_parsed"):
        t = getattr(entry, key, None)
        if t is not None:
            return datetime(*t[:6], tzinfo=timezone.utc)
    return None

@st.cache_data(ttl=60 * 10, show_spinner=False)
def fetch_news_demo(max_items: int = 10) -> pd.DataFrame:
    # Use both "Solana" and "SOL" terms to catch more headlines, still limited for demo
    q = 'Solana OR SOL cryptocurrency OR "Solana network"'
    parsed = feedparser.parse(google_news_rss_url(q))

    rows = []
    for e in parsed.entries:
        title = getattr(e, "title", "").strip()
        link = getattr(e, "link", "").strip()
        if not title or not link:
            continue
        rows.append({
            "title": title,
            "link": link,
            "published": parse_published(e),
            "source": "Google News"
        })

    if not rows:
        return pd.DataFrame(columns=["title", "link", "published", "source"])

    df = pd.DataFrame(rows)
    df["title_key"] = df["title"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    df = df.drop_duplicates(subset=["link"], keep="first").drop_duplicates(subset=["title_key"], keep="first")
    df["_t"] = df["published"].fillna(pd.Timestamp("1970-01-01", tz="UTC"))
    df = df.sort_values("_t", ascending=False).drop(columns=["_t", "title_key"])
    return df.head(max_items)


# ----------------------------
# UI controls
# ----------------------------
with st.sidebar:
    st.header("Controls")

    product = st.selectbox("Market (Coinbase)", ["SOL-USD", "SOL-EUR"], index=0)

    granularity_label = st.selectbox("Candle size", ["1 min", "5 min", "15 min", "1 hour"], index=1)
    granularity_sec = {"1 min": 60, "5 min": 300, "15 min": 900, "1 hour": 3600}[granularity_label]

    st.subheader("News (Demo)")
    max_news = st.slider("Number of headlines", 3, 20, 10, 1)

    show_rsi = st.checkbox("Show RSI", value=True)
    show_macd = st.checkbox("Show MACD", value=True)
    show_drawdown = st.checkbox("Show Drawdown", value=True)

    refresh = st.button("Refresh")

if refresh:
    st.cache_data.clear()

# Layout: main + right panel (news)
main_col, news_col = st.columns([3.2, 1.2], gap="large")

with main_col:
    with st.spinner("Loading Solana candles..."):
        df_1h = get_last_hour_candles(product, granularity_sec)

    if df_1h.empty:
        st.error("No data returned from Coinbase. Try a different candle size or market (SOL-USD vs SOL-EUR).")
        st.stop()

    dfi = add_indicators(df_1h, granularity_sec)

    # Metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    last_close = float(dfi["close"].iloc[-1])
    c1.metric("Last close", f"{last_close:,.4f}")

    if len(dfi) >= 2:
        prev = float(dfi["close"].iloc[-2])
        c2.metric("Δ last bar", f"{(last_close - prev):.4f}", f"{((last_close/prev)-1)*100:.2f}%")
    else:
        c2.metric("Δ last bar", "—")

    v = dfi["vol_30_ann"].iloc[-1]
    c3.metric("Ann. vol (30)", f"{v*100:.2f}%" if pd.notna(v) else "—")

    dd = dfi["drawdown"].iloc[-1]
    c4.metric("Drawdown", f"{dd*100:.2f}%" if pd.notna(dd) else "—")

    # For crypto in a 1-hour window, "regime" via SMA50/200 isn’t meaningful.
    # So we show a simple trend hint using EMA20 slope.
    ema_slope = dfi["ema_20"].iloc[-1] - dfi["ema_20"].iloc[max(0, len(dfi)-5)]
    trend_hint = "up" if ema_slope > 0 else "down"
    c5.metric("Short trend (EMA20)", trend_hint)

    st.caption(f"Showing ONLY the last 1 hour. Latest bar (UTC): {dfi.index[-1]}")

    # Price chart
    st.subheader("Price (last 1 hour)")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=dfi.index,
        open=dfi["open"], high=dfi["high"], low=dfi["low"], close=dfi["close"],
        name="Candles"
    ))
    fig.add_trace(go.Scatter(x=dfi.index, y=dfi["ema_20"], name="EMA 20"))
    fig.update_layout(height=520, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Volume
    st.subheader("Volume")
    vfig = go.Figure()
    vfig.add_trace(go.Bar(x=dfi.index, y=dfi["volume"], name="Volume"))
    vfig.update_layout(height=240, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(vfig, use_container_width=True)

    # RSI / MACD / Drawdown
    colA, colB = st.columns(2)

    if show_rsi:
        with colA:
            st.subheader("RSI (14)")
            rfig = go.Figure()
            rfig.add_trace(go.Scatter(x=dfi.index, y=dfi["rsi_14"], name="RSI(14)"))
            rfig.add_hline(y=70, line_dash="dash")
            rfig.add_hline(y=30, line_dash="dash")
            rfig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(rfig, use_container_width=True)
    else:
        with colA:
            st.info("RSI hidden.")

    if show_macd:
        with colB:
            st.subheader("MACD")
            mfig = go.Figure()
            mfig.add_trace(go.Scatter(x=dfi.index, y=dfi["macd"], name="MACD"))
            mfig.add_trace(go.Scatter(x=dfi.index, y=dfi["macd_signal"], name="Signal"))
            mfig.add_trace(go.Bar(x=dfi.index, y=dfi["macd_hist"], name="Hist"))
            mfig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(mfig, use_container_width=True)
    else:
        with colB:
            st.info("MACD hidden.")

    if show_drawdown:
        st.subheader("Drawdown")
        dfig = go.Figure()
        dfig.add_trace(go.Scatter(x=dfi.index, y=dfi["drawdown"], name="Drawdown"))
        dfig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(dfig, use_container_width=True)

    # Table + download
    st.subheader("Data")
    st.dataframe(dfi.tail(500), use_container_width=True)

    csv = dfi.to_csv().encode("utf-8")
    st.download_button(
        "Download CSV (last 1 hour)",
        data=csv,
        file_name=f"{product}_last_1h_{granularity_label.replace(' ','_')}.csv",
        mime="text/csv"
    )

with news_col:
    st.subheader("News (demo)")
    st.caption("Small list for demo purposes.")

    with st.spinner("Loading headlines..."):
        news_df = fetch_news_demo(max_items=max_news)

    st.markdown(
        """
        <style>
        .news-box { height: 78vh; overflow-y: auto; padding-right: 6px;
                    border-left: 1px solid rgba(255,255,255,0.08); }
        .news-item { margin-bottom: 12px; }
        .news-meta { font-size: 0.8rem; opacity: 0.75; }
        </style>
        """,
        unsafe_allow_html=True
    )

    if news_df.empty:
        st.info("No headlines found right now.")
    else:
        st.markdown('<div class="news-box">', unsafe_allow_html=True)
        for _, row in news_df.iterrows():
            when = ""
            if pd.notna(row["published"]):
                when = pd.to_datetime(row["published"]).strftime("%Y-%m-%d %H:%M UTC")

            st.markdown(
                f"""
                <div class="news-item">
                  <div><a href="{row['link']}" target="_blank" rel="noopener noreferrer"><b>{row['title']}</b></a></div>
                  <div class="news-meta">{row['source']} • {when}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)
