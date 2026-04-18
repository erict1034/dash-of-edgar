import time
import sqlite3
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
import logging
from central_logging import get_error_logger
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import dash
from dash import get_app, Input, Output, State, dcc, html
from dotenv import load_dotenv
from edgar import Company, set_identity
from urllib.parse import parse_qs
from storage_paths import CENTRAL_SQLITE_PATH, parquet_path

dash.register_page(
    __name__, path="/pull", name="Public Company Insider Purchases", order=1
)

load_dotenv()

# Important: Set your email here to comply with SEC EDGAR access policies.
# This is required to use the edgar package for fetching Form 4 filings.
SEC_CONTACT_EMAIL = os.getenv("SEC_CONTACT_EMAIL", "your_email@example.com")
set_identity(SEC_CONTACT_EMAIL)

# Cache settings
CACHE_TTL_SECONDS = 900
DEFAULT_FILING_LIMIT = 200
FORM4_CACHE = {}
PRICE_CACHE = {}
ALPHAVANTAGE_BASE_URL = "https://www.alphavantage.co/query"
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")
FORM4_PARQUET_PATH = parquet_path("ef4_form4")
PRICE_PARQUET_PATH = parquet_path("ef4_monthly_prices")
SQLITE_DB_PATH = CENTRAL_SQLITE_PATH
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("ef4_pull_dash")
error_logger = get_error_logger("ef4_pull_dash")

# Visual theme colors for chart + layout
COLORS = {
    "bg": "#f8f9fa",
    "card_bg": "#ffffff",
    "card_border": "#dee2e6",
    "text": "#212529",
    "muted": "#6c757d",
    "sales": "#f97316",
    "acq": "#10b981",
    "price": "#2563eb",
}

CARD_STYLE = {
    "backgroundColor": COLORS["card_bg"],
    "border": f"1px solid {COLORS['card_border']}",
    "borderRadius": "0.5rem",
    "color": COLORS["text"],
}


def _is_light_color(hex_color):
    value = (hex_color or "").lstrip("#")
    if len(value) != 6:
        return False
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return luminance > 0.62


def build_container_style(bg_color, text_color=None):
    text_color = text_color or ("#111111" if _is_light_color(bg_color) else "#ffffff")
    return {
        "background": bg_color,
        "minHeight": "100vh",
        "color": text_color,
    }


def build_card_style(card_bg, text_color=None):
    is_light = _is_light_color(card_bg)
    border = (
        "1px solid rgba(0, 0, 0, 0.2)"
        if is_light
        else "1px solid rgba(255, 255, 255, 0.2)"
    )
    text_color = text_color or ("#111111" if is_light else "#ffffff")
    return {
        "backgroundColor": card_bg,
        "border": border,
        "borderRadius": "0.75rem",
        "color": text_color,
    }


def hex_to_rgba(hex_color, alpha):
    value = (hex_color or "").lstrip("#")
    if len(value) != 6:
        return f"rgba(250, 183, 0, {alpha})"
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# Custom exception for data source errors to provide clearer error handling and messaging
class DataSourceError(Exception):
    pass


# Helper functions for caching, CIK resolution, data fetching,
# figure building, and error classification are defined below to keep the main callback logic
# clean and focused on orchestrating the dashboard updates.
def _get_cached(cache_store, key):
    cached = cache_store.get(key)
    if not cached:
        return None
    age = time.time() - cached["ts"]
    if age > CACHE_TTL_SECONDS:
        cache_store.pop(key, None)
        return None
    return cached["value"]


# Helper function to set cache with current timestamp
def _set_cached(cache_store, key, value):
    cache_store[key] = {"ts": time.time(), "value": value}


# Helper function to resolve CIK from ticker using
# SEC's company_tickers.json endpoint
def _resolve_cik_from_ticker(ticker):
    ticker = (ticker or "").strip().upper()
    if not ticker:
        raise DataSourceError("Ticker is empty.")

    headers = {
        "User-Agent": f"ef4-dashboard ({SEC_CONTACT_EMAIL})",
        "Accept": "application/json, text/plain, */*",
        "Accept-Encoding": "gzip, deflate",
    }

    try:
        response = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=headers,
            timeout=15,
        )
        response.raise_for_status()
        text = (response.text or "").strip()
        if not text:
            raise DataSourceError(
                "SEC company_tickers.json returned an empty response body."
            )
        cik_lookup = response.json()
    except Exception as exc:
        try:
            fallback = requests.get(
                "https://www.sec.gov/include/ticker.txt",
                headers=headers,
                timeout=15,
            )
            fallback.raise_for_status()
            for line in fallback.text.splitlines():
                parts = line.strip().split("\t")
                if len(parts) != 2:
                    continue
                if parts[0].strip().upper() == ticker:
                    return str(parts[1].strip()).zfill(10)
        except Exception:
            pass
        raise DataSourceError(f"Failed to download SEC CIK lookup: {exc}") from exc

    companies = cik_lookup.values() if isinstance(cik_lookup, dict) else cik_lookup
    for company in companies:
        if not isinstance(company, dict):
            continue
        if str(company.get("ticker", "")).upper() == ticker:
            cik = company.get("cik_str")
            if cik is None:
                continue
            return str(cik).zfill(10)

    raise DataSourceError(f"Ticker not found in SEC company_tickers.json: {ticker}.")


# Main function to fetch Form 4 data for a given ticker and filing
# limit, with caching support
def fetch_form4_dataframe(ticker, filing_limit=DEFAULT_FILING_LIMIT):
    cache_key = (ticker, filing_limit)
    cached = _get_cached(FORM4_CACHE, cache_key)
    if cached is not None:
        return cached.copy(deep=True), True

    try:
        cik = _resolve_cik_from_ticker(ticker)
        company = Company(cik)
        filings = company.get_filings(form="4").head(filing_limit)
    except Exception as exc:
        raise DataSourceError(f"SEC lookup failed for {ticker}: {exc}") from exc

    frames = []
    for filing in filings:
        try:
            frames.append(filing.obj().to_dataframe().fillna(""))
        except Exception:
            continue

    if not frames:
        raise DataSourceError(f"No Form 4 filings were returned for {ticker}.")

    df = pd.concat(frames, ignore_index=True)
    _set_cached(FORM4_CACHE, cache_key, df)
    return df.copy(deep=True), False


# Function to build a monthly summary of sales (S) and purchases (P) from the Form 4 data,
def build_monthly(df, metric_mode="count"):
    if df.empty or "Code" not in df.columns or "Date" not in df.columns:
        return pd.DataFrame(columns=["S", "P"])

    working = df[["Code", "Date"]].copy()
    use_shares = metric_mode == "shares" and "Shares" in df.columns
    if use_shares:
        working["Shares"] = pd.to_numeric(
            df["Shares"].astype(str).str.replace(",", "", regex=False), errors="coerce"
        ).fillna(0.0)
    working["Date"] = pd.to_datetime(working["Date"], errors="coerce")
    working = working.dropna(subset=["Date"])
    working = working[working["Code"].isin(["S", "P"])]

    if working.empty:
        return pd.DataFrame(columns=["S", "P"])

    working["Month"] = working["Date"].dt.to_period("M").astype(str)
    if use_shares:
        monthly = (
            working.groupby(["Month", "Code"])["Shares"].sum().unstack().sort_index()
        )
        monthly = monthly.fillna(0.0).astype(float)
    else:
        monthly = working.groupby(["Month", "Code"]).size().unstack().sort_index()
        monthly = monthly.fillna(0).astype(int)

    if "S" not in monthly.columns:
        monthly["S"] = 0.0 if use_shares else 0
    if "P" not in monthly.columns:
        monthly["P"] = 0.0 if use_shares else 0

    return monthly[["S", "P"]]


# Function to fetch monthly prices from Alpha Vantage's free endpoint.
def _fetch_alphavantage_monthly(ticker, api_key):
    params = {
        "function": "TIME_SERIES_MONTHLY",
        "symbol": ticker,
        "apikey": api_key,
    }
    last_error = None
    for attempt in range(1, 4):
        try:
            response = requests.get(ALPHAVANTAGE_BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
            break
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(attempt)
    else:
        raise DataSourceError(
            f"Alpha Vantage price lookup failed for {ticker}: {last_error}"
        ) from last_error

    if "Error Message" in payload:
        raise DataSourceError(payload["Error Message"])
    if "Note" in payload:
        raise DataSourceError(payload["Note"])
    if "Information" in payload:
        raise DataSourceError(payload["Information"])

    series = payload.get("Monthly Time Series")
    if not series:
        raise DataSourceError(
            f"No Alpha Vantage monthly time series returned for {ticker}."
        )

    price_df = pd.DataFrame.from_dict(series, orient="index")
    if "4. close" not in price_df.columns:
        raise DataSourceError(
            f"Alpha Vantage response missing close price field for {ticker}."
        )
    close_col = "4. close"

    return (
        price_df[[close_col]]
        .rename(columns={close_col: "AlphaVantageClose"})
        .assign(Date=lambda d: pd.to_datetime(d.index, errors="coerce"))
        .dropna(subset=["Date"])
        .assign(
            AlphaVantageClose=lambda d: pd.to_numeric(
                d["AlphaVantageClose"], errors="coerce"
            )
        )
        .dropna(subset=["AlphaVantageClose"])
        .set_index("Date")
        .sort_index()
    )


# Function to fetch monthly closing prices from Alpha Vantage for the
# date range covered by the Form 4 data, with caching support
def fetch_monthly_prices(ticker, monthly):
    if monthly.empty:
        return pd.DataFrame(columns=["Month", "AlphaVantageClose"]), False

    month_idx = pd.to_datetime(monthly.index, format="%Y-%m")
    start_date = month_idx.min().strftime("%Y-%m-%d")
    end_date = (month_idx.max() + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")

    cache_key = (ticker, start_date, end_date)
    cached = _get_cached(PRICE_CACHE, cache_key)
    if cached is not None:
        return cached.copy(deep=True), True

    api_key = (ALPHAVANTAGE_API_KEY or "").strip()
    if not api_key:
        raise DataSourceError(
            "Missing Alpha Vantage API key. Set ALPHAVANTAGE_API_KEY in your environment and retry."
        )

    try:
        price_df = _fetch_alphavantage_monthly(ticker, api_key)
    except DataSourceError as exc:
        cached_prices = _fetch_cached_monthly_prices(ticker)
        if not cached_prices.empty:
            _set_cached(PRICE_CACHE, cache_key, cached_prices)
            return cached_prices.copy(deep=True), True
        raise exc
    price_df = price_df.loc[start_date:end_date]
    if price_df.empty:
        raise DataSourceError(
            f"Alpha Vantage returned no price data for {ticker} in range {start_date} to {end_date}."
        )

    monthly_price_df = (
        price_df[["AlphaVantageClose"]]
        .assign(Month=lambda d: d.index.to_period("M").astype(str))
        .reset_index(drop=True)
    )

    _set_cached(PRICE_CACHE, cache_key, monthly_price_df)
    return monthly_price_df.copy(deep=True), False


def _normalize_live_form4(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "record_id",
                "ticker",
                "company",
                "filing_date",
                "code",
                "shares",
                "pulled_at_utc",
            ]
        )

    working = pd.DataFrame(index=df.index)
    date_source = (
        df["Date"]
        if "Date" in df.columns
        else pd.Series([None] * len(df), index=df.index)
    )
    code_source = (
        df["Code"]
        if "Code" in df.columns
        else pd.Series([""] * len(df), index=df.index)
    )
    company_source = (
        df["Company"]
        if "Company" in df.columns
        else pd.Series([""] * len(df), index=df.index)
    )

    working["filing_date"] = pd.to_datetime(
        date_source, errors="coerce"
    ).dt.date.astype(str)
    working["code"] = code_source.astype(str).str.strip()
    if "Shares" in df.columns:
        working["shares"] = pd.to_numeric(
            df["Shares"].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        ).fillna(0.0)
    else:
        working["shares"] = 0.0
    working["company"] = company_source.astype(str).str.strip()
    working["ticker"] = (ticker or "").strip().upper()
    working["pulled_at_utc"] = _utc_now_iso()

    id_parts = (
        working["ticker"]
        + "|"
        + working["filing_date"]
        + "|"
        + working["code"]
        + "|"
        + working["shares"].astype(str)
        + "|"
        + working["company"]
    )
    working["record_id"] = id_parts.map(
        lambda value: hashlib.sha1(value.encode("utf-8")).hexdigest()
    )

    return working[
        [
            "record_id",
            "ticker",
            "company",
            "filing_date",
            "code",
            "shares",
            "pulled_at_utc",
        ]
    ].drop_duplicates(subset=["record_id"])


def _save_parquet(data: pd.DataFrame, parquet_path: Path) -> Path:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data.to_parquet(parquet_path, index=False)
    except Exception as exc:
        raise RuntimeError(
            "Failed to write Parquet. Install a parquet engine: pip install pyarrow"
        ) from exc
    return parquet_path


def _upsert_sqlite_form4(
    data: pd.DataFrame, db_path: Path = SQLITE_DB_PATH
) -> tuple[int, int]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS form4_transactions (
                record_id TEXT PRIMARY KEY,
                ticker TEXT NOT NULL,
                company TEXT,
                filing_date TEXT NOT NULL,
                code TEXT,
                shares REAL,
                pulled_at_utc TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_form4_ticker ON form4_transactions(ticker)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_form4_filing_date ON form4_transactions(filing_date)"
        )

        rows = [
            (
                row.record_id,
                row.ticker,
                row.company,
                row.filing_date,
                row.code,
                float(row.shares),
                row.pulled_at_utc,
            )
            for row in data.itertuples(index=False)
        ]
        before = conn.execute("SELECT COUNT(*) FROM form4_transactions").fetchone()[0]
        conn.executemany(
            """
            INSERT OR IGNORE INTO form4_transactions
            (record_id, ticker, company, filing_date, code, shares, pulled_at_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        after = conn.execute("SELECT COUNT(*) FROM form4_transactions").fetchone()[0]
    return len(rows), after - before


def _upsert_sqlite_prices(
    data: pd.DataFrame, ticker: str, db_path: Path = SQLITE_DB_PATH
) -> tuple[int, int]:
    if data is None or data.empty:
        return 0, 0

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS monthly_prices (
                ticker TEXT NOT NULL,
                month TEXT NOT NULL,
                close REAL,
                pulled_at_utc TEXT NOT NULL,
                PRIMARY KEY (ticker, month)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_prices_ticker ON monthly_prices(ticker)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_prices_month ON monthly_prices(month)"
        )

        rows = [
            (
                (ticker or "").strip().upper(),
                row.Month,
                float(row.AlphaVantageClose),
                _utc_now_iso(),
            )
            for row in data.itertuples(index=False)
        ]
        before = conn.execute("SELECT COUNT(*) FROM monthly_prices").fetchone()[0]
        conn.executemany(
            """
            INSERT OR REPLACE INTO monthly_prices (ticker, month, close, pulled_at_utc)
            VALUES (?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        after = conn.execute("SELECT COUNT(*) FROM monthly_prices").fetchone()[0]
    return len(rows), after - before


def _persist_live_pull(
    ticker: str, raw_df: pd.DataFrame, monthly_price_df: pd.DataFrame
):
    normalized_df = _normalize_live_form4(raw_df, ticker=ticker)
    form4_parquet_path = _save_parquet(normalized_df, FORM4_PARQUET_PATH)
    rows_attempted, rows_inserted = _upsert_sqlite_form4(normalized_df, SQLITE_DB_PATH)

    price_parquet_path = ""
    price_rows_attempted = 0
    price_rows_inserted = 0
    if monthly_price_df is not None and not monthly_price_df.empty:
        price_parquet_path = _save_parquet(monthly_price_df, PRICE_PARQUET_PATH)
        price_rows_attempted, price_rows_inserted = _upsert_sqlite_prices(
            monthly_price_df, ticker=ticker, db_path=SQLITE_DB_PATH
        )

    return {
        "form4_parquet_path": str(form4_parquet_path),
        "price_parquet_path": str(price_parquet_path),
        "rows_attempted": int(rows_attempted),
        "rows_inserted": int(rows_inserted),
        "price_rows_attempted": int(price_rows_attempted),
        "price_rows_inserted": int(price_rows_inserted),
    }


def _fetch_cached_monthly_summary(ticker, metric_mode="count", db_path=SQLITE_DB_PATH):
    metric_mode = "shares" if metric_mode == "shares" else "count"
    metric_expr = "SUM(shares)" if metric_mode == "shares" else "COUNT(*)"
    sql = f"""
    SELECT
        substr(filing_date, 1, 7) AS month,
        code,
        {metric_expr} AS value
    FROM form4_transactions
    WHERE ticker = ?
      AND code IN ('S', 'P')
    GROUP BY month, code
    ORDER BY month
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(sql, conn, params=(ticker.upper(),))
    if df.empty:
        return pd.DataFrame(columns=["Month", "S", "P"])
    pivot = df.pivot(index="month", columns="code", values="value").fillna(0)
    if "S" not in pivot.columns:
        pivot["S"] = 0
    if "P" not in pivot.columns:
        pivot["P"] = 0
    pivot = pivot.reset_index().rename(columns={"month": "Month"})
    return pivot[["Month", "S", "P"]]


def _fetch_cached_monthly_prices(ticker, db_path=SQLITE_DB_PATH):
    sql = """
    SELECT month, close
    FROM monthly_prices
    WHERE ticker = ?
    ORDER BY month
    """
    with sqlite3.connect(db_path) as conn:
        try:
            df = pd.read_sql_query(sql, conn, params=(ticker.upper(),))
        except Exception:
            return pd.DataFrame(columns=["Month", "AlphaVantageClose"])
    if df.empty:
        return pd.DataFrame(columns=["Month", "AlphaVantageClose"])
    return df.rename(columns={"month": "Month", "close": "AlphaVantageClose"})


def load_ticker_dashboard_cached(
    ticker,
    metric_mode="count",
    s_color=None,
    a_color=None,
    price_color=None,
    text_color=None,
    card_bg=None,
):
    if not SQLITE_DB_PATH.exists():
        raise DataSourceError(
            f"SQLite cache not found at {SQLITE_DB_PATH}. Run the pipeline first."
        )

    monthly_table = _fetch_cached_monthly_summary(ticker, metric_mode=metric_mode)
    if monthly_table.empty:
        raise DataSourceError(
            f"No cached Form 4 data found for {ticker}. Run the pipeline first."
        )

    monthly = monthly_table.set_index("Month")
    monthly_price_df = _fetch_cached_monthly_prices(ticker)

    fig = build_figure(
        monthly,
        monthly_price_df,
        ticker,
        metric_mode=metric_mode,
        s_color=s_color,
        a_color=a_color,
        price_color=price_color,
        text_color=text_color,
        card_bg=card_bg,
    )
    if metric_mode == "shares":
        total_s = f"{int(round(monthly['S'].sum())):,}" if not monthly.empty else "0"
        total_a = f"{int(round(monthly['P'].sum())):,}" if not monthly.empty else "0"
    else:
        total_s = f"{int(monthly['S'].sum()):,}" if not monthly.empty else "0"
        total_a = f"{int(monthly['P'].sum()):,}" if not monthly.empty else "0"
    latest_close = (
        f"${monthly_price_df['AlphaVantageClose'].iloc[-1]:,.2f}"
        if not monthly_price_df.empty
        else "N/A"
    )

    return fig, total_s, total_a, latest_close


# Function to build the Plotly figure showing monthly sales and acquisitions,
# along with optional trend lines and price overlay, based on the provided data
# and metric mode (count vs shares)
def build_figure(
    monthly,
    monthly_price_df,
    ticker,
    metric_mode="count",
    s_color=None,
    a_color=None,
    price_color=None,
    text_color=None,
    card_bg=None,
):
    metric_label = "Shares" if metric_mode == "shares" else "Count"
    s_color = s_color or COLORS["sales"]
    a_color = a_color or COLORS["acq"]
    price_color = price_color or COLORS["price"]
    text_color = text_color or COLORS["text"]
    card_bg = card_bg or COLORS["card_bg"]
    s_bar_color = hex_to_rgba(s_color, 0.45)
    a_bar_color = hex_to_rgba(a_color, 0.45)
    if monthly.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"No Form 4 S/P {metric_label.lower()} data found for {ticker}",
            xaxis_title="Month",
            yaxis_title=metric_label,
            template="plotly_white",
        )
        return fig

    monthly_df = monthly.reset_index().melt(
        id_vars="Month", value_vars=["S", "P"], var_name="Code", value_name="Count"
    )

    fig = px.bar(
        monthly_df,
        x="Month",
        y="Count",
        color="Code",
        barmode="group",
        title=f"Insider Monthly Sales (S) vs Purchases (P) {metric_label} - {ticker}",
        color_discrete_map={"S": s_bar_color, "P": a_bar_color},
    )

    monthly_s = monthly["S"].reset_index(name="SCount")
    if len(monthly_s) >= 2:
        x_idx = np.arange(len(monthly_s))
        s_coeffs = np.polyfit(x_idx, monthly_s["SCount"], 1)
        fig.add_trace(
            go.Scatter(
                x=monthly_s["Month"],
                y=np.polyval(s_coeffs, x_idx),
                mode="lines",
                name=f"Trend Sales ({metric_label})",
                line=dict(color=s_color, width=3, dash="dash"),
            )
        )

    monthly_p = monthly["P"].reset_index(name="PCount")
    if len(monthly_p) >= 2:
        x_idx = np.arange(len(monthly_p))
        a_coeffs = np.polyfit(x_idx, monthly_p["PCount"], 1)
        fig.add_trace(
            go.Scatter(
                x=monthly_p["Month"],
                y=np.polyval(a_coeffs, x_idx),
                mode="lines",
                name=f"Trend Purchases ({metric_label})",
                line=dict(color=a_color, width=3, dash="dash"),
            )
        )

    if not monthly_price_df.empty:
        fig.add_trace(
            go.Scatter(
                x=monthly_price_df["Month"],
                y=monthly_price_df["AlphaVantageClose"],
                mode="lines",
                name=f"{ticker} Month Closing Price",
                line=dict(color=price_color, width=3),
                yaxis="y2",
            )
        )
        fig.update_layout(
            yaxis2=dict(
                title=f"{ticker} Monthly Closing Price",
                overlaying="y",
                side="right",
                showgrid=False,
            )
        )

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        yaxis_title=metric_label,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb")
    fig.update_yaxes(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb")
    return fig


# Function to classify error messages into user-friendly categories for display in the
# dashboard status message area
def classify_error_message(exc):
    text = str(exc)
    lowered = text.lower()
    is_edgar_error = any(
        key in lowered for key in ["sec", "edgar", "form 4", "cik", "company_tickers"]
    )
    is_alpha_error = "alpha vantage" in lowered or "alphavantage" in lowered

    if "429" in text or "rate" in lowered or "too many" in lowered:
        if is_edgar_error:
            return "EDGAR rate limit hit. Wait a minute and retry."
        if is_alpha_error:
            return "Alpha Vantage rate limit hit. Wait a minute and retry."
        return "Rate limit hit. Wait a minute and retry."
    if "no form 4 filings" in lowered:
        return f"EDGAR: {text}"
    if "missing alpha vantage api key" in lowered:
        return "Alpha Vantage: Missing ALPHAVANTAGE_API_KEY in environment. Add your key and retry."

    if "no price data" in lowered or "possibly delisted" in lowered:
        return "Alpha Vantage: No price data found for this ticker/date range."
    if "not found" in lowered or "invalid" in lowered:
        if is_edgar_error:
            return "EDGAR: Ticker not recognized. Check symbol and retry."
        if is_alpha_error:
            return "Alpha Vantage: Ticker not recognized. Check symbol and retry."
        return "Ticker not recognized. Check symbol and retry."
    if is_edgar_error:
        return f"EDGAR: {text}"
    if is_alpha_error:
        return f"Alpha Vantage: {text}"
    return text


# Main function to load Form 4 data, build the dashboard figure, and calculate summary metrics
# for a given ticker and filing limit, with error handling and caching support
def load_ticker_dashboard(
    ticker,
    filing_limit,
    metric_mode="count",
    s_color=None,
    a_color=None,
    price_color=None,
    text_color=None,
    card_bg=None,
    persist_live=False,
):
    df, filings_from_cache = fetch_form4_dataframe(ticker, filing_limit)
    monthly = build_monthly(df, metric_mode=metric_mode)
    if monthly.empty:
        raise DataSourceError(
            f"No S/P transactions found in Form 4 filings for {ticker}."
        )

    monthly_price_df, prices_from_cache = fetch_monthly_prices(ticker, monthly)

    persist_note = ""
    if persist_live:
        try:
            persist_result = _persist_live_pull(
                ticker, raw_df=df, monthly_price_df=monthly_price_df
            )
            persist_note = (
                " (Saved to SQLite/Parquet)"
                if persist_result.get("rows_attempted", 0) > 0
                else " (No rows persisted)"
            )
            logger.info(
                "Persist live pull OK | ticker=%s rows_attempted=%s rows_inserted=%s price_rows_attempted=%s price_rows_inserted=%s",
                ticker,
                persist_result.get("rows_attempted", 0),
                persist_result.get("rows_inserted", 0),
                persist_result.get("price_rows_attempted", 0),
                persist_result.get("price_rows_inserted", 0),
            )
        except Exception as exc:
            persist_note = f" (Persist error: {exc})"
            logger.exception("Persist live pull failed | ticker=%s", ticker)
            error_logger.exception("Persist live pull failed | ticker=%s", ticker)

    fig = build_figure(
        monthly,
        monthly_price_df,
        ticker,
        metric_mode=metric_mode,
        s_color=s_color,
        a_color=a_color,
        price_color=price_color,
        text_color=text_color,
        card_bg=card_bg,
    )
    if metric_mode == "shares":
        total_s = f"{int(round(monthly['S'].sum())):,}" if not monthly.empty else "0"
        total_a = f"{int(round(monthly['P'].sum())):,}" if not monthly.empty else "0"
    else:
        total_s = f"{int(monthly['S'].sum()):,}" if not monthly.empty else "0"
        total_a = f"{int(monthly['P'].sum()):,}" if not monthly.empty else "0"
    latest_close = (
        f"${monthly_price_df['AlphaVantageClose'].iloc[-1]:,.2f}"
        if not monthly_price_df.empty
        else "N/A"
    )

    return (
        fig,
        total_s,
        total_a,
        latest_close,
        filings_from_cache,
        prices_from_cache,
        persist_note,
    )


# Helper function to create an empty Plotly figure with a given title, used for initial
# state and error cases
def empty_figure(title, text_color=None, card_bg=None):
    fig = go.Figure()
    fig.update_layout(
        title=title,
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
    )
    fig.update_xaxes(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb")
    fig.update_yaxes(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb")
    return fig


def build_layout():
    # Define the app layout with input fields, summary cards, and graph area
    return dbc.Container(
        [
            dcc.Location(id="ef4-url", refresh=False),
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5(
                                    "Recent Counts of Insider Stock Sales and Purchases from SEC Form 4 Filings",
                                    className="card-title mb-0",
                                ),
                                html.P(
                                    "Enter a ticker to pull live Form 4 (EDGAR) and price (Alpha Vantage) data",
                                    id="ef4-subtitle-text",
                                    className="mb-0 text-muted",
                                ),
                            ]
                        ),
                        id="ef4-header-card",
                        className="shadow-sm",
                    ),
                    width=12,
                ),
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Input(
                            id="ef4-ticker-input",
                            type="text",
                            value="",
                            placeholder="Enter ticker (e.g., MSFT)",
                        ),
                        md=4,
                    ),
                    dbc.Col(
                        dcc.Dropdown(
                            id="ef4-limit-input",
                            options=[
                                {"label": "50 filings", "value": 50},
                                {"label": "100 filings", "value": 100},
                                {"label": "200 filings", "value": 200},
                            ],
                            value=None,
                            placeholder="Select number of recent filings",
                            clearable=False,
                            style={"color": "black"},
                        ),
                        md=3,
                    ),
                    dbc.Col(
                        dbc.RadioItems(
                            id="ef4-metric-mode",
                            options=[
                                {"label": "Count", "value": "count"},
                                {"label": "Shares", "value": "shares"},
                            ],
                            value="count",
                            inline=True,
                        ),
                        md=3,
                    ),
                    dbc.Col(
                        dbc.RadioItems(
                            id="ef4-data-source",
                            options=[
                                {"label": "Live Pull", "value": "live"},
                                {"label": "From Database", "value": "cached"},
                            ],
                            value="live",
                            inline=True,
                        ),
                        md=2,
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Pull Data",
                            id="ef4-pull-button",
                            color="primary",
                            n_clicks=0,
                            disabled=True,
                        ),
                        md="auto",
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                id="ef4-updating-msg",
                                className="fw-semibold",
                            ),
                            html.Div(id="ef4-status-msg", className="text-muted small"),
                        ],
                        md=4,
                    ),
                ],
                className="mb-3 align-items-center",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [html.H6("Total Sales"), html.H4("-", id="ef4-total-s")]
                            ),
                            id="ef4-sales-card",
                            className="shadow-sm",
                        ),
                        md=4,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H6("Total Purchases"),
                                    html.H4("-", id="ef4-total-a"),
                                ]
                            ),
                            id="ef4-acq-card",
                            className="shadow-sm",
                        ),
                        md=4,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H6(
                                        "Latest Close", id="ef4-latest-close-label"
                                    ),
                                    html.H4("-", id="ef4-latest-close"),
                                ]
                            ),
                            id="ef4-close-card",
                            className="shadow-sm",
                        ),
                        md=4,
                    ),
                ],
                className="mb-3",
            ),
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            dcc.Graph(
                                id="ef4-filings-chart",
                                figure=empty_figure(
                                    "Enter a ticker and click Pull Data"
                                ),
                            )
                        ),
                        id="ef4-chart-card",
                        className="shadow-sm",
                    ),
                    width=12,
                )
            ),
        ],
        id="ef4-app-container",
        fluid=True,
        className="py-3",
    )


layout = build_layout()

_callbacks_registered = False


def register_callbacks(app):
    global _callbacks_registered
    if _callbacks_registered:
        return
    _callbacks_registered = True

    # Define the callback function that orchestrates the data fetching, processing, and figure building
    @app.callback(
        Output("ef4-filings-chart", "figure"),
        Output("ef4-total-s", "children"),
        Output("ef4-total-a", "children"),
        Output("ef4-latest-close", "children"),
        Output("ef4-latest-close-label", "children"),
        Output("ef4-status-msg", "children"),
        Input("ef4-pull-button", "n_clicks"),
        Input("ef4-metric-mode", "value"),
        Input("ef4-data-source", "value"),
        State("ef4-ticker-input", "value"),
        State("ef4-limit-input", "value"),
        running=[
            (
                Output("ef4-updating-msg", "children"),
                html.Span(
                    [
                        dbc.Spinner(size="sm", color="primary", type="border"),
                        html.Span(
                            " Updating dashboard... pulling data and building chart. This is a Government data source, so this could take a few minutes. Be patient ",
                            className="ms-2",
                        ),
                    ],
                    className="d-inline-flex align-items-center",
                ),
                "",
            ),
        ],
    )
    def pull_and_render(
        _n_clicks,
        metric_mode,
        data_source,
        ticker_value,
        limit_value,
    ):
        ticker = (ticker_value or "").strip().upper()
        filing_limit = int(limit_value or DEFAULT_FILING_LIMIT)
        metric_mode = "shares" if metric_mode == "shares" else "count"
        s_color = COLORS["sales"]
        a_color = COLORS["acq"]
        price_color = COLORS["price"]
        text_color = COLORS["text"]
        card_bg = COLORS["card_bg"]

        if not ticker:
            return (
                empty_figure(
                    "No ticker provided", text_color=text_color, card_bg=card_bg
                ),
                "-",
                "-",
                "-",
                "Latest Close",
                "",
            )

        try:
            start = time.perf_counter()
            data_source = (data_source or "live").strip().lower()
            if data_source == "cached":
                fig, total_s, total_a, latest_close = load_ticker_dashboard_cached(
                    ticker,
                    metric_mode=metric_mode,
                    s_color=s_color,
                    a_color=a_color,
                    price_color=price_color,
                    text_color=text_color,
                    card_bg=card_bg,
                )
                f_cached = True
                p_cached = True
                persist_note = ""
            else:
                (
                    fig,
                    total_s,
                    total_a,
                    latest_close,
                    f_cached,
                    p_cached,
                    persist_note,
                ) = load_ticker_dashboard(
                    ticker,
                    filing_limit,
                    metric_mode=metric_mode,
                    s_color=s_color,
                    a_color=a_color,
                    price_color=price_color,
                    text_color=text_color,
                    card_bg=card_bg,
                    persist_live=True,
                )
            elapsed = time.perf_counter() - start

            cache_parts = []
            if data_source == "cached":
                cache_parts.append("SQLite cache")
            else:
                if f_cached:
                    cache_parts.append("EDGAR cache")
                if p_cached:
                    cache_parts.append("Alpha Vantage cache")
            cache_suffix = f" ({', '.join(cache_parts)})" if cache_parts else ""

            status_msg = f"Loaded {ticker} with {metric_mode} view in {elapsed:.2f}s{cache_suffix}"
            if data_source != "cached" and persist_note:
                status_msg += persist_note
            logger.info(
                "Status | source=%s ticker=%s metric=%s limit=%s elapsed=%.2fs msg=%s",
                data_source,
                ticker,
                metric_mode,
                filing_limit,
                elapsed,
                status_msg,
            )

            return (
                fig,
                total_s,
                total_a,
                latest_close,
                f"{ticker} Latest Close",
                status_msg,
            )
        except Exception as exc:
            logger.exception("Pull failed | source=%s ticker=%s", data_source, ticker)
            error_logger.exception(
                "Pull failed | source=%s ticker=%s", data_source, ticker
            )
            return (
                empty_figure(
                    f"Unable to load data for {ticker}",
                    text_color=text_color,
                    card_bg=card_bg,
                ),
                "-",
                "-",
                "-",
                "Latest Close",
                f"Error: {classify_error_message(exc)}",
            )

    @app.callback(
        Output("ef4-pull-button", "disabled"),
        Input("ef4-limit-input", "value"),
        Input("ef4-data-source", "value"),
    )
    def toggle_pull_button(limit_value, data_source):  # noqa: F811
        if (data_source or "live") == "cached":
            return False
        return limit_value is None

    @app.callback(
        Output("ef4-ticker-input", "value"),
        Input("ef4-url", "search"),
        State("ef4-ticker-input", "value"),
    )
    def seed_ticker_from_url(search, current_value):
        if not search:
            return current_value or ""
        params = parse_qs(search.lstrip("?"))
        ticker = params.get("ticker", [None])[0]
        if not ticker:
            return current_value or ""
        return str(ticker).strip().upper()


register_callbacks(get_app())
