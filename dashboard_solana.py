# solana_dashboard_news_right_demo.py
# Streamlit dashboard for Solana (SOL) with:
# - LIVE mode (last 1 hour + indicators + news)
# - RESEARCH mode (session stats, event study at US market open, heatmaps, rolling correlations)
# - ET-aware session tagging (DST-safe)
# - News tagging + simple post-headline reaction metrics
# - Raw candle caching and local resampling for efficiency

import time
import math
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import feedparser

from datetime import datetime, timezone, timedelta
from urllib.parse import quote
from zoneinfo import ZoneInfo

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="Solana Dashboard (Live + Research)", layout="wide")
st.title("Solana Dashboard — Live + Research (US Open Analysis)")

ET_TZ = ZoneInfo("America/New_York")
UTC = timezone.utc

# ----------------------------
# Coinbase candles (public, no key)
# ----------------------------
@st.cache_data(ttl=60 * 5, show_spinner=False)
def fetch_coinbase_candles(product: str, start_dt: datetime, end_dt: datetime, granularity_sec: int) -> pd.DataFrame:
    """
    Coinbase Exchange candles endpoint:
    returns [time, low, high, open, close, volume]
    endpoint returns max ~300 candles per request => paginate by time windows.
    """
    url = f"https://api.exchange.coinbase.com/products/{product}/candles"
    headers = {"accept": "application/json", "user-agent": "sol-dashboard/2.0"}

    # ensure aware UTC
    start_dt = start_dt.astimezone(UTC)
    end_dt = end_dt.astimezone(UTC)

    window_seconds = granularity_sec * 300
    cur = start_dt
    rows = []

    while cur < end_dt:
        nxt = min(cur + timedelta(seconds=window_seconds), end_dt)
        params = {"start": cur.isoformat(), "end": nxt.isoformat(), "granularity": granularity_sec}
        r = requests.get(url, params=params, headers=headers, timeout=30)

        if r.status_code >= 400:
            raise RuntimeError(f"Coinbase API error {r.status_code}: {r.text}\nRequest: {r.url}")

        payload = r.json()
        if isinstance(payload, list):
            rows.extend(payload)
        else:
            raise RuntimeError(f"Unexpected Coinbase response: {payload}")

        cur = nxt
        time.sleep(0.15)  # polite pacing

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["ts", "low", "high", "open", "close", "volume"])
    df["date"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df = df.drop(columns=["ts"]).set_index("date").sort_index()

    # de-dup
    df = df[~df.index.duplicated(keep="last")]

    # numeric safety
    for c in ["low", "high", "open", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])

    return df


@st.cache_data(ttl=60 * 5, show_spinner=False)
def get_raw_candles(product: str, lookback_days: int, fetch_granularity_sec: int) -> pd.DataFrame:
    """
    Fetch raw candles for the lookback window at a chosen base granularity.
    Cached by (product, lookback_days, fetch_granularity_sec).
    """
    now = datetime.now(UTC)
    start = now - timedelta(days=lookback_days)
    return fetch_coinbase_candles(product, start, now, fetch_granularity_sec)


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample OHLCV from a finer base series.
    """
    if df.empty:
        return df

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    out = df.resample(rule).agg(agg).dropna(subset=["open", "high", "low", "close"])
    return out


def get_last_hour_from_raw(df_raw: pd.DataFrame, target_rule: str) -> pd.DataFrame:
    if df_raw.empty:
        return df_raw
    df = resample_ohlcv(df_raw, target_rule)
    now = datetime.now(UTC)
    cutoff = now - timedelta(hours=1)
    return df[df.index >= cutoff]


# ----------------------------
# Indicators (conditional)
# ----------------------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_features(
    df: pd.DataFrame,
    granularity_sec: int,
    compute_rsi: bool = False,
    compute_macd: bool = False,
    compute_drawdown: bool = False,
    compute_vol: bool = True,
) -> pd.DataFrame:
    out = df.copy()
    out["price"] = out["close"]
    out["ret"] = out["price"].pct_change()
    out["ema_20"] = out["price"].ewm(span=20, adjust=False).mean()

    if compute_rsi:
        out["rsi_14"] = rsi(out["price"], 14)

    if compute_macd:
        ema12 = out["price"].ewm(span=12, adjust=False).mean()
        ema26 = out["price"].ewm(span=26, adjust=False).mean()
        out["macd"] = ema12 - ema26
        out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
        out["macd_hist"] = out["macd"] - out["macd_signal"]

    if compute_vol:
        periods_per_year = (365 * 24 * 3600) / max(granularity_sec, 1)
        out["vol_30_ann"] = out["ret"].rolling(30).std() * np.sqrt(periods_per_year)

    if compute_drawdown:
        peak = out["price"].cummax()
        out["drawdown"] = out["price"] / peak - 1.0

    return out


# ----------------------------
# ET session tagging (DST-aware)
# ----------------------------
def add_time_features_et(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    et = out.index.tz_convert(ET_TZ)
    out["ts_et"] = et
    out["date_et"] = et.date
    out["hour_et"] = et.hour
    out["minute_et"] = et.minute
    out["weekday_et"] = et.dayofweek  # Mon=0
    out["weekday_name"] = et.day_name()
    out["time_et"] = et.time
    out["minutes_since_midnight_et"] = et.hour * 60 + et.minute
    return out


def tag_us_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Session buckets in ET:
    - pre_open: 07:00–09:29
    - us_open_burst: 09:30–10:29
    - us_regular: 09:30–16:00
    - after_hours: 16:01–19:59
    - overnight: rest
    """
    out = add_time_features_et(df)

    m = out["minutes_since_midnight_et"]
    # ET minute boundaries
    pre_open = (m >= 7 * 60) & (m < 9 * 60 + 30)
    open_burst = (m >= 9 * 60 + 30) & (m < 10 * 60 + 30)
    us_regular = (m >= 9 * 60 + 30) & (m <= 16 * 60)
    after_hours = (m > 16 * 60) & (m < 20 * 60)

    out["session_bucket"] = "overnight"
    out.loc[pre_open, "session_bucket"] = "pre_open"
    out.loc[open_burst, "session_bucket"] = "us_open_burst"
    # keep open_burst separate but also track regular
    out["is_us_regular"] = us_regular
    out.loc[after_hours, "session_bucket"] = "after_hours"

    return out


# ----------------------------
# News (Google News RSS demo)
# ----------------------------
def google_news_rss_url(query: str) -> str:
    return f"https://news.google.com/rss/search?q={quote(query)}&hl=en-US&gl=US&ceid=US:en"


def parse_published(entry):
    for key in ("published_parsed", "updated_parsed"):
        t = getattr(entry, key, None)
        if t is not None:
            return datetime(*t[:6], tzinfo=UTC)
    return None


def tag_news_title(title: str) -> str:
    t = (title or "").lower()

    if any(k in t for k in ["etf", "sec", "regulator", "regulation", "approval"]):
        return "ETF / Regulatory"
    if any(k in t for k in ["outage", "downtime", "congestion", "validator", "network"]):
        return "Network / Infra"
    if any(k in t for k in ["defi", "dex", "staking", "yield", "protocol", "solana ecosystem"]):
        return "DeFi / Ecosystem"
    if any(k in t for k in ["fed", "cpi", "inflation", "rates", "treasury", "macro"]):
        return "Macro"
    if any(k in t for k in ["memecoin", "meme coin", "token launch", "airdrop"]):
        return "Memecoin / Flows"
    return "General"


@st.cache_data(ttl=60 * 10, show_spinner=False)
def fetch_news_demo(max_items: int = 10) -> pd.DataFrame:
    q = 'Solana OR SOL cryptocurrency OR "Solana network"'
    parsed = feedparser.parse(google_news_rss_url(q))

    rows = []
    for e in getattr(parsed, "entries", []):
        title = getattr(e, "title", "").strip()
        link = getattr(e, "link", "").strip()
        if not title or not link:
            continue
        rows.append(
            {
                "title": title,
                "link": link,
                "published": parse_published(e),
                "source": "Google News",
                "tag": tag_news_title(title),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["title", "link", "published", "source", "tag"])

    df = pd.DataFrame(rows)
    df["title_key"] = df["title"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    df = df.drop_duplicates(subset=["link"], keep="first").drop_duplicates(subset=["title_key"], keep="first")
    df["_t"] = df["published"].fillna(pd.Timestamp("1970-01-01", tz="UTC"))
    df = df.sort_values("_t", ascending=False).drop(columns=["_t", "title_key"])
    return df.head(max_items).reset_index(drop=True)


def attach_news_reaction(news_df: pd.DataFrame, price_df: pd.DataFrame, horizons=("15min", "1h", "4h")) -> pd.DataFrame:
    """
    For each news item, compute post-headline price returns at specified horizons
    using nearest-forward candle in price_df (must be UTC-indexed and sorted).
    """
    if news_df.empty or price_df.empty:
        return news_df.copy()

    out = news_df.copy()
    p = price_df[["close"]].sort_index().copy()

    # ensure UTC and sorted
    if p.index.tz is None:
        p.index = p.index.tz_localize("UTC")
    else:
        p.index = p.index.tz_convert("UTC")

    out["headline_price"] = np.nan
    for h in horizons:
        out[f"ret_{h}"] = np.nan

    idx = p.index

    for i, row in out.iterrows():
        t = row["published"]
        if pd.isna(t):
            continue

        t = pd.Timestamp(t).tz_convert("UTC")
        # first candle at or after news time
        pos = idx.searchsorted(t, side="left")
        if pos >= len(idx):
            continue

        t0 = idx[pos]
        p0 = float(p.iloc[pos]["close"])
        out.at[i, "headline_price"] = p0

        for h in horizons:
            target = t0 + pd.Timedelta(h)
            pos2 = idx.searchsorted(target, side="left")
            if pos2 < len(idx):
                p1 = float(p.iloc[pos2]["close"])
                out.at[i, f"ret_{h}"] = (p1 / p0) - 1.0

    return out


# ----------------------------
# Session stats / Event study / Heatmaps
# ----------------------------
def session_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary stats by session bucket and also a separate 'us_regular' aggregate.
    Expects columns: ret, volume, session_bucket, is_us_regular
    """
    if df.empty:
        return pd.DataFrame()

    tmp = df.copy()
    tmp["abs_ret"] = tmp["ret"].abs()

    groups = []
    # bucket stats
    g1 = (
        tmp.groupby("session_bucket")
        .agg(
            bars=("ret", "count"),
            mean_ret=("ret", "mean"),
            median_ret=("ret", "median"),
            win_rate=("ret", lambda x: (x > 0).mean()),
            avg_abs_ret=("abs_ret", "mean"),
            avg_volume=("volume", "mean"),
            ret_std=("ret", "std"),
        )
        .reset_index()
        .rename(columns={"session_bucket": "segment"})
    )
    groups.append(g1)

    # us_regular aggregate
    reg = tmp[tmp["is_us_regular"]].copy()
    if not reg.empty:
        reg["segment"] = "us_regular"
        g2 = pd.DataFrame(
            {
                "segment": ["us_regular"],
                "bars": [reg["ret"].count()],
                "mean_ret": [reg["ret"].mean()],
                "median_ret": [reg["ret"].median()],
                "win_rate": [(reg["ret"] > 0).mean()],
                "avg_abs_ret": [reg["abs_ret"].mean()],
                "avg_volume": [reg["volume"].mean()],
                "ret_std": [reg["ret"].std()],
            }
        )
        groups.append(g2)

    out = pd.concat(groups, ignore_index=True)
    out["ann_vol_proxy"] = out["ret_std"] * np.sqrt(24 * 365)  # rough for hourly-like stats (proxy only)
    return out.sort_values("segment").reset_index(drop=True)


def build_open_event_study(df: pd.DataFrame, bar_minutes: int) -> pd.DataFrame:
    """
    Event study around US market open (9:30 ET), per ET day.
    Windows:
      -30m_to_+30m, 9:30_to_10:00, 9:30_to_11:00
    Uses nearest bars on/after target timestamps.
    """
    if df.empty:
        return pd.DataFrame()

    tmp = add_time_features_et(df)
    tmp = tmp.sort_index().copy()

    # One target event per ET calendar day at 9:30 ET
    days = pd.Series(tmp["date_et"].unique()).sort_values().tolist()
    rows = []

    idx = tmp.index
    closes = tmp["close"]

    for d in days:
        event_et = pd.Timestamp(datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET_TZ))
        event_utc = event_et.tz_convert("UTC")

        # helper to get close at or after timestamp
        def price_at_or_after(ts: pd.Timestamp):
            pos = idx.searchsorted(ts, side="left")
            if pos >= len(idx):
                return np.nan, None
            return float(closes.iloc[pos]), idx[pos]

        p_m30, t_m30 = price_at_or_after(event_utc - pd.Timedelta(minutes=30))
        p_0, t_0 = price_at_or_after(event_utc)
        p_p30, t_p30 = price_at_or_after(event_utc + pd.Timedelta(minutes=30))
        p_p60, t_p60 = price_at_or_after(event_utc + pd.Timedelta(minutes=60))
        p_p90, t_p90 = price_at_or_after(event_utc + pd.Timedelta(minutes=90))

        # require at least event and one later point
        row = {"date_et": d, "event_ts_utc": event_utc}
        row["ret_-30m_to_+30m"] = (p_p30 / p_m30 - 1) if pd.notna(p_m30) and pd.notna(p_p30) else np.nan
        row["ret_9_30_to_10_00"] = (p_p30 / p_0 - 1) if pd.notna(p_0) and pd.notna(p_p30) else np.nan
        row["ret_9_30_to_10_30"] = (p_p60 / p_0 - 1) if pd.notna(p_0) and pd.notna(p_p60) else np.nan
        row["ret_9_30_to_11_00"] = (p_p90 / p_0 - 1) if pd.notna(p_0) and pd.notna(p_p90) else np.nan
        rows.append(row)

    es = pd.DataFrame(rows)
    if es.empty:
        return es

    # compare against random windows of same length (simple baseline on same dataset)
    # Using 60-minute windows as baseline (for 9:30->10:30)
    # Approximate by bar count window
    approx_bars = max(1, int(round(60 / max(bar_minutes, 1))))
    ret_series = tmp["close"].pct_change(approx_bars).shift(-approx_bars)
    es["random_60m_mean_benchmark"] = ret_series.mean(skipna=True)

    return es.sort_values("date_et").reset_index(drop=True)


def summarize_event_study(es: pd.DataFrame) -> pd.DataFrame:
    if es.empty:
        return pd.DataFrame()

    cols = [c for c in es.columns if c.startswith("ret_")]
    rows = []
    for c in cols:
        x = pd.to_numeric(es[c], errors="coerce").dropna()
        if x.empty:
            continue
        rows.append(
            {
                "window": c,
                "n_days": int(x.shape[0]),
                "mean": x.mean(),
                "median": x.median(),
                "std": x.std(),
                "win_rate": (x > 0).mean(),
                "p25": x.quantile(0.25),
                "p75": x.quantile(0.75),
            }
        )
    return pd.DataFrame(rows).sort_values("window").reset_index(drop=True)


def hour_week_heatmaps(df: pd.DataFrame):
    """
    Returns pivot tables for avg return, realized vol proxy (std of returns), avg volume
    rows=weekday, cols=hour ET
    """
    if df.empty:
        return None, None, None

    tmp = add_time_features_et(df).copy()
    tmp["ret"] = tmp["close"].pct_change()
    tmp = tmp.dropna(subset=["ret"])

    # order weekdays
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    ret_piv = tmp.pivot_table(index="weekday_name", columns="hour_et", values="ret", aggfunc="mean")
    vol_piv = tmp.pivot_table(index="weekday_name", columns="hour_et", values="ret", aggfunc="std")
    volm_piv = tmp.pivot_table(index="weekday_name", columns="hour_et", values="volume", aggfunc="mean")

    ret_piv = ret_piv.reindex([d for d in weekday_order if d in ret_piv.index])
    vol_piv = vol_piv.reindex([d for d in weekday_order if d in vol_piv.index])
    volm_piv = volm_piv.reindex([d for d in weekday_order if d in volm_piv.index])

    return ret_piv, vol_piv, volm_piv


# ----------------------------
# Correlation panel helpers
# ----------------------------
def build_returns_series(df: pd.DataFrame, name: str) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float, name=name)
    s = df["close"].pct_change().rename(name)
    return s


@st.cache_data(ttl=60 * 10, show_spinner=False)
def fetch_coinbase_multi(products, lookback_days: int, granularity_sec: int) -> dict:
    out = {}
    now = datetime.now(UTC)
    start = now - timedelta(days=lookback_days)
    for p in products:
        try:
            out[p] = fetch_coinbase_candles(p, start, now, granularity_sec)
        except Exception:
            out[p] = pd.DataFrame()
    return out


@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_optional_yfinance(symbols, period="6mo", interval="1h"):
    """
    Optional market proxies (SPY/QQQ/DXY) if yfinance is installed and network allows.
    """
    try:
        import yfinance as yf
    except Exception:
        return {}, "yfinance not installed"

    out = {}
    for sym in symbols:
        try:
            df = yf.download(sym, period=period, interval=interval, progress=False, auto_adjust=False, threads=False)
            if df is None or df.empty:
                out[sym] = pd.DataFrame()
                continue
            # normalize columns
            cols = {c.lower(): c for c in df.columns.astype(str)}
            close_col = "Close" if "Close" in df.columns else list(df.columns)[0]
            dfx = pd.DataFrame(index=pd.to_datetime(df.index, utc=True))
            dfx["close"] = pd.to_numeric(df[close_col], errors="coerce")
            dfx = dfx.dropna()
            out[sym] = dfx
        except Exception:
            out[sym] = pd.DataFrame()
    return out, None


def rolling_corr_panel(sol_df: pd.DataFrame, lookback_days: int, base_granularity_sec: int, target_rule: str):
    st.subheader("Rolling Correlations (SOL vs BTC/ETH + optional SPY/QQQ/DXY)")

    # Coinbase crypto peers
    peers = fetch_coinbase_multi(["BTC-USD", "ETH-USD"], lookback_days=lookback_days, granularity_sec=base_granularity_sec)

    target_map = {"1min": "1T", "5min": "5T", "15min": "15T", "1h": "1H"}
    rule = target_rule

    sol_r = resample_ohlcv(sol_df, rule) if {"open", "high", "low", "close", "volume"}.issubset(sol_df.columns) else sol_df.copy()
    if "close" not in sol_r.columns or sol_r.empty:
        st.info("Not enough SOL data for rolling correlation.")
        return

    base = pd.DataFrame(index=sol_r.index)
    base["SOL"] = build_returns_series(sol_r, "SOL")

    for k, v in peers.items():
        if v.empty:
            continue
        rv = resample_ohlcv(v, rule)
        if not rv.empty:
            base[k.replace("-USD", "")] = build_returns_series(rv, k.replace("-USD", ""))

    # optional stocks/macro
    yf_data, yf_err = fetch_optional_yfinance(["SPY", "QQQ", "DX-Y.NYB"], period="6mo", interval="60m")
    if yf_err:
        st.caption("Optional SPY/QQQ/DXY correlations unavailable (yfinance not installed).")
    else:
        for sym, dfy in yf_data.items():
            if dfy.empty:
                continue
            # align to UTC hourly-like bars
            dfx = dfy.copy()
            dfx = dfx[~dfx.index.duplicated(keep="last")]
            base[sym] = dfx["close"].pct_change()

    base = base.sort_index().dropna(how="all")
    if base.shape[1] < 2 or base.empty:
        st.info("Not enough peer series available for correlation.")
        return

    # rolling window by bars
    if rule in ["1T", "5T", "15T"]:
        # approx 1 day of bars
        bar_minutes = {"1T": 1, "5T": 5, "15T": 15}[rule]
        window = max(24, int((24 * 60) / bar_minutes))
    else:
        window = 24 * 7  # 1 week on hourly bars

    corr_cols = [c for c in base.columns if c != "SOL"]
    fig = go.Figure()
    for c in corr_cols:
        x = base[["SOL", c]].dropna()
        if len(x) < window + 5:
            continue
        rc = x["SOL"].rolling(window).corr(x[c])
        fig.add_trace(go.Scatter(x=rc.index, y=rc, name=f"SOL vs {c}"))

    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Rolling corr")
    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"Rolling window: {window} bars (depends on chosen analysis resolution).")


# ----------------------------
# Chart helpers
# ----------------------------
def add_news_markers_to_price(fig: go.Figure, news_df: pd.DataFrame, price_df: pd.DataFrame, max_markers: int = 8):
    if news_df.empty or price_df.empty:
        return fig

    p = price_df[["close"]].sort_index()
    if p.index.tz is None:
        p.index = p.index.tz_localize("UTC")
    else:
        p.index = p.index.tz_convert("UTC")
    idx = p.index

    markers = news_df.dropna(subset=["published"]).copy().head(max_markers)
    xs, ys, txt = [], [], []
    for _, row in markers.iterrows():
        t = pd.Timestamp(row["published"]).tz_convert("UTC")
        pos = idx.searchsorted(t, side="left")
        if pos >= len(idx):
            continue
        ts = idx[pos]
        xs.append(ts)
        ys.append(float(p.loc[ts, "close"]))
        txt.append(f"{row['tag']}: {row['title'][:120]}")

    if xs:
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys,
                mode="markers",
                name="News",
                text=txt,
                hovertemplate="%{text}<br>%{x}<br>Price=%{y:.4f}<extra></extra>",
            )
        )
    return fig


# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.header("Controls")

    mode = st.radio("Mode", ["Live (1h)", "Research"], index=0)

    product = st.selectbox("Market (Coinbase)", ["SOL-USD", "SOL-EUR"], index=0)

    st.subheader("Data Resolution")
    fetch_granularity_label = st.selectbox(
        "Base fetch granularity (for caching / resampling)",
        ["1 min", "5 min", "15 min", "1 hour"],
        index=1,
        help="Fetch at this granularity, then resample locally to other views."
    )
    fetch_granularity_sec = {"1 min": 60, "5 min": 300, "15 min": 900, "1 hour": 3600}[fetch_granularity_label]

    view_granularity_label = st.selectbox(
        "Display / analysis resolution",
        ["1 min", "5 min", "15 min", "1 hour"],
        index=1
    )
    view_granularity_sec = {"1 min": 60, "5 min": 300, "15 min": 900, "1 hour": 3600}[view_granularity_label]
    rule_map = {"1 min": "1T", "5 min": "5T", "15 min": "15T", "1 hour": "1H"}
    target_rule = rule_map[view_granularity_label]

    if mode == "Research":
        lookback_days = st.slider("Research lookback (days)", 30, 365, 120, 10)
        st.caption("Longer lookback = better statistics, but slower.")
    else:
        lookback_days = 7  # enough cache for news-reaction on live + some flexibility

    st.subheader("Indicators / Charts")
    show_rsi = st.checkbox("Show RSI", value=True)
    show_macd = st.checkbox("Show MACD", value=True)
    show_drawdown = st.checkbox("Show Drawdown", value=True)

    st.subheader("News (Demo)")
    max_news = st.slider("Number of headlines", 3, 25, 10, 1)
    show_news_markers = st.checkbox("Show news markers on price chart", value=True)

    st.subheader("Research Panels")
    show_session_stats = st.checkbox("Session analyzer", value=True)
    show_event_study = st.checkbox("9:30 ET event study", value=True)
    show_heatmaps = st.checkbox("Hour × weekday heatmaps", value=True)
    show_corr_panel = st.checkbox("Rolling correlation panel", value=True)

    refresh = st.button("Refresh (clear cache)")

if refresh:
    st.cache_data.clear()

# ----------------------------
# Load core data once (cached raw)
# ----------------------------
with st.spinner("Loading SOL candles..."):
    raw_sol = get_raw_candles(product, lookback_days=lookback_days, fetch_granularity_sec=fetch_granularity_sec)

if raw_sol.empty:
    st.error("No data returned from Coinbase. Try a different market or granularity.")
    st.stop()

sol_view = resample_ohlcv(raw_sol, target_rule)
if sol_view.empty:
    st.error("Resampling resulted in empty data. Try a finer base granularity.")
    st.stop()

# Compute bar minutes for event study windows
bar_minutes_map = {"1T": 1, "5T": 5, "15T": 15, "1H": 60}
bar_minutes = bar_minutes_map[target_rule]

# News (fetched once, reused in both modes)
with st.spinner("Loading headlines..."):
    news_df = fetch_news_demo(max_items=max_news)

# Attach news reactions using broader dataset (prefer research window if available)
news_df_react = attach_news_reaction(news_df, sol_view, horizons=("15min", "1h", "4h"))

# ----------------------------
# Mode: Live
# ----------------------------
if mode == "Live (1h)":
    # Layout main + news right panel
    main_col, news_col = st.columns([3.2, 1.2], gap="large")

    with main_col:
        df_1h = sol_view[sol_view.index >= (datetime.now(UTC) - timedelta(hours=1))].copy()
        if df_1h.empty:
            st.error("No data in the last 1 hour. Try a different display resolution.")
            st.stop()

        dfi = add_features(
            df_1h,
            granularity_sec=view_granularity_sec,
            compute_rsi=show_rsi,
            compute_macd=show_macd,
            compute_drawdown=show_drawdown,
            compute_vol=True,
        )

        # Metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        last_close = float(dfi["close"].iloc[-1])
        c1.metric("Last close", f"{last_close:,.4f}")

        if len(dfi) >= 2 and pd.notna(dfi["close"].iloc[-2]):
            prev = float(dfi["close"].iloc[-2])
            delta_abs = last_close - prev
            delta_pct = ((last_close / prev) - 1) * 100 if prev != 0 else np.nan
            c2.metric("Δ last bar", f"{delta_abs:.4f}", f"{delta_pct:.2f}%")
        else:
            c2.metric("Δ last bar", "—")

        v = dfi["vol_30_ann"].iloc[-1] if "vol_30_ann" in dfi.columns else np.nan
        c3.metric("Ann. vol (30)", f"{v*100:.2f}%" if pd.notna(v) else "—")

        if "drawdown" in dfi.columns:
            dd = dfi["drawdown"].iloc[-1]
            c4.metric("Drawdown", f"{dd*100:.2f}%" if pd.notna(dd) else "—")
        else:
            c4.metric("Drawdown", "hidden")

        # short trend hint via EMA20 slope over last ~5 bars
        if len(dfi) >= 6:
            ema_slope = dfi["ema_20"].iloc[-1] - dfi["ema_20"].iloc[-6]
            trend_hint = "up" if ema_slope > 0 else "down"
        else:
            trend_hint = "—"
        c5.metric("Short trend (EMA20)", trend_hint)

        # ET note about US open
        latest_et = dfi.index[-1].tz_convert(ET_TZ)
        us_open_today_et = pd.Timestamp(datetime(latest_et.year, latest_et.month, latest_et.day, 9, 30, tzinfo=ET_TZ))
        mins_from_open = (latest_et - us_open_today_et).total_seconds() / 60
        st.caption(
            f"Showing only the last 1 hour ({view_granularity_label} bars). "
            f"Latest bar UTC: {dfi.index[-1]} | ET: {latest_et.strftime('%Y-%m-%d %H:%M %Z')} "
            f"| Minutes from US open (9:30 ET): {mins_from_open:+.0f}"
        )

        # Price chart
        st.subheader("Price (last 1 hour)")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=dfi.index,
            open=dfi["open"], high=dfi["high"], low=dfi["low"], close=dfi["close"],
            name="Candles"
        ))
        fig.add_trace(go.Scatter(x=dfi.index, y=dfi["ema_20"], name="EMA 20"))

        if show_news_markers and not news_df_react.empty:
            recent_news = news_df_react.dropna(subset=["published"]).copy()
            recent_news = recent_news[pd.to_datetime(recent_news["published"], utc=True) >= (dfi.index.min() - pd.Timedelta(hours=1))]
            fig = add_news_markers_to_price(fig, recent_news, dfi, max_markers=8)

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

        if show_rsi and "rsi_14" in dfi.columns:
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

        if show_macd and {"macd", "macd_signal", "macd_hist"}.issubset(dfi.columns):
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

        if show_drawdown and "drawdown" in dfi.columns:
            st.subheader("Drawdown")
            dfig = go.Figure()
            dfig.add_trace(go.Scatter(x=dfi.index, y=dfi["drawdown"], name="Drawdown"))
            dfig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(dfig, use_container_width=True)

        st.subheader("Live Data (last 1 hour)")
        st.dataframe(dfi.tail(500), use_container_width=True)

        csv = dfi.to_csv().encode("utf-8")
        st.download_button(
            "Download CSV (last 1 hour)",
            data=csv,
            file_name=f"{product}_last_1h_{view_granularity_label.replace(' ','_')}.csv",
            mime="text/csv"
        )

    with news_col:
        st.subheader("News (demo)")
        st.caption("Tagged headlines + simple post-news price reaction estimates.")

        st.markdown(
            """
            <style>
            .news-box { height: 78vh; overflow-y: auto; padding-right: 6px;
                        border-left: 1px solid rgba(255,255,255,0.08); }
            .news-item { margin-bottom: 12px; }
            .news-meta { font-size: 0.8rem; opacity: 0.75; }
            .news-tag { font-size: 0.78rem; opacity: 0.90; padding: 2px 6px; border-radius: 6px;
                        border: 1px solid rgba(255,255,255,0.12); display:inline-block; margin-top:4px;}
            </style>
            """,
            unsafe_allow_html=True
        )

        if news_df_react.empty:
            st.info("No headlines found right now.")
        else:
            st.markdown('<div class="news-box">', unsafe_allow_html=True)
            for _, row in news_df_react.iterrows():
                when = ""
                if pd.notna(row["published"]):
                    when = pd.to_datetime(row["published"], utc=True).strftime("%Y-%m-%d %H:%M UTC")

                def fmt_ret(x):
                    return "—" if pd.isna(x) else f"{x*100:+.2f}%"

                st.markdown(
                    f"""
                    <div class="news-item">
                      <div><a href="{row['link']}" target="_blank" rel="noopener noreferrer"><b>{row['title']}</b></a></div>
                      <div class="news-meta">{row['source']} • {when}</div>
                      <div class="news-tag">{row.get('tag','General')}</div>
                      <div class="news-meta">Post-news: 15m {fmt_ret(row.get('ret_15min', np.nan))} • 1h {fmt_ret(row.get('ret_1h', np.nan))} • 4h {fmt_ret(row.get('ret_4h', np.nan))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Mode: Research
# ----------------------------
else:
    st.subheader("Research Mode")
    st.caption(
        "Use this mode to test hypotheses like: 'Does SOL behave differently around the U.S. market open (9:30 ET)?' "
        "This is more useful for your question than RSI/MACD on the last hour."
    )

    # Build research dataset
    dfr = sol_view.copy()
    dfr = add_features(
        dfr,
        granularity_sec=view_granularity_sec,
        compute_rsi=False,
        compute_macd=False,
        compute_drawdown=False,
        compute_vol=True,
    )
    dfr = tag_us_sessions(dfr)

    # Top metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(dfr):,}")
    c2.metric("Lookback days", f"{lookback_days}")
    c3.metric("Resolution", view_granularity_label)
    c4.metric("Latest ET", dfr.index[-1].tz_convert(ET_TZ).strftime("%Y-%m-%d %H:%M %Z"))

    # Price overview with optional news markers
    st.subheader("SOL Price (research window)")
    pfig = go.Figure()
    pfig.add_trace(go.Scatter(x=dfr.index, y=dfr["close"], name="SOL close"))
    pfig.add_trace(go.Scatter(x=dfr.index, y=dfr["ema_20"], name="EMA20"))
    if show_news_markers and not news_df_react.empty:
        pfig = add_news_markers_to_price(pfig, news_df_react, dfr, max_markers=12)
    pfig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(pfig, use_container_width=True)

    # A) Session Analyzer
    if show_session_stats:
        st.subheader("A) Session Analyzer (ET-based)")
        ss = session_summary(dfr)
        if ss.empty:
            st.info("Session summary unavailable.")
        else:
            ss_display = ss.copy()
            for c in ["mean_ret", "median_ret", "win_rate", "avg_abs_ret", "ret_std", "ann_vol_proxy"]:
                if c in ss_display.columns:
                    ss_display[c] = ss_display[c] * 100
            st.dataframe(ss_display, use_container_width=True)

            # Bar chart: avg return and avg abs return
            s_col1, s_col2 = st.columns(2)
            with s_col1:
                fig1 = px.bar(ss, x="segment", y="mean_ret", title="Mean return by session segment")
                fig1.update_yaxes(tickformat=".2%")
                fig1.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig1, use_container_width=True)

            with s_col2:
                fig2 = px.bar(ss, x="segment", y="avg_abs_ret", title="Average absolute return by session segment")
                fig2.update_yaxes(tickformat=".2%")
                fig2.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig2, use_container_width=True)

    # B) Event Study (9:30 ET)
    if show_event_study:
        st.subheader("B) 9:30 ET Event Study (U.S. Market Open)")
        es = build_open_event_study(dfr, bar_minutes=bar_minutes)
        es_sum = summarize_event_study(es)

        if es.empty or es_sum.empty:
            st.info("Event study unavailable for selected resolution/window.")
        else:
            # Display summary
            es_sum_display = es_sum.copy()
            for c in ["mean", "median", "std", "win_rate", "p25", "p75"]:
                if c in es_sum_display.columns:
                    es_sum_display[c] = es_sum_display[c] * 100
            st.dataframe(es_sum_display, use_container_width=True)

            # Distribution chart for one key window
            key_window = "ret_9_30_to_10_30" if "ret_9_30_to_10_30" in es.columns else es.filter(like="ret_").columns[0]
            x = pd.to_numeric(es[key_window], errors="coerce").dropna()

            e1, e2 = st.columns([1.3, 1.7])
            with e1:
                st.markdown(f"**Selected window:** `{key_window}`")
                if not x.empty:
                    st.metric("Mean", f"{x.mean()*100:+.3f}%")
                    st.metric("Median", f"{x.median()*100:+.3f}%")
                    st.metric("Win rate", f"{(x>0).mean()*100:.1f}%")
                    st.metric("Days", f"{len(x)}")
                    if "random_60m_mean_benchmark" in es.columns:
                        bmk = pd.to_numeric(es["random_60m_mean_benchmark"], errors="coerce").dropna()
                        if not bmk.empty:
                            st.metric("Random 60m mean benchmark", f"{bmk.iloc[0]*100:+.3f}%")
                else:
                    st.info("No valid event rows for selected window.")

            with e2:
                if not x.empty:
                    hfig = px.histogram(x * 100, nbins=30, title=f"Distribution of {key_window} returns (%)")
                    hfig.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(hfig, use_container_width=True)

            # Daily time series of open-window return
            if key_window in es.columns:
                tsfig = go.Figure()
                tsfig.add_trace(go.Scatter(x=pd.to_datetime(es["date_et"]), y=es[key_window], mode="lines+markers", name=key_window))
                tsfig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
                tsfig.update_yaxes(tickformat=".2%")
                st.plotly_chart(tsfig, use_container_width=True)

    # C) Hour-of-day × Weekday Heatmaps
    if show_heatmaps:
        st.subheader("C) Hour-of-Day × Weekday Heatmaps (ET)")
        ret_piv, vol_piv, volm_piv = hour_week_heatmaps(dfr)

        if ret_piv is None:
            st.info("Heatmaps unavailable.")
        else:
            h1, h2, h3 = st.columns(3)

            with h1:
                fig_ret = px.imshow(
                    ret_piv,
                    aspect="auto",
                    title="Avg Return by ET Hour / Weekday",
                    labels=dict(x="Hour (ET)", y="Weekday", color="Avg ret"),
                )
                fig_ret.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_ret, use_container_width=True)

            with h2:
                fig_vol = px.imshow(
                    vol_piv,
                    aspect="auto",
                    title="Return Std (Vol Proxy) by ET Hour / Weekday",
                    labels=dict(x="Hour (ET)", y="Weekday", color="Std ret"),
                )
                fig_vol.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_vol, use_container_width=True)

            with h3:
                fig_volm = px.imshow(
                    volm_piv,
                    aspect="auto",
                    title="Avg Volume by ET Hour / Weekday",
                    labels=dict(x="Hour (ET)", y="Weekday", color="Volume"),
                )
                fig_volm.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_volm, use_container_width=True)

    # D) Rolling correlations panel
    if show_corr_panel:
        rolling_corr_panel(raw_sol, lookback_days=lookback_days, base_granularity_sec=fetch_granularity_sec, target_rule=target_rule)

    # E) News reactions table (research window)
    st.subheader("D/E) News Tags + Post-Headline Reaction (Demo)")
    if news_df_react.empty:
        st.info("No headlines found.")
    else:
        nd = news_df_react.copy()
        for c in ["ret_15min", "ret_1h", "ret_4h"]:
            if c in nd.columns:
                nd[c] = nd[c] * 100
        st.dataframe(
            nd[["published", "tag", "title", "ret_15min", "ret_1h", "ret_4h", "link"]]
            if set(["published", "tag", "title", "ret_15min", "ret_1h", "ret_4h", "link"]).issubset(nd.columns)
            else nd,
            use_container_width=True
        )

    # Export research data
    st.subheader("Export")
    export_df = dfr.copy()
    st.download_button(
        "Download research dataset CSV",
        data=export_df.to_csv().encode("utf-8"),
        file_name=f"{product}_research_{lookback_days}d_{view_granularity_label.replace(' ','_')}.csv",
        mime="text/csv"
    )