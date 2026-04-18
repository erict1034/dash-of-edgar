import dash_ag_grid as dag
import sqlite3
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
import numpy as np
import requests
import os
import pandas as pd
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import dash
from dash import get_app, dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc
from storage_paths import DATA_DIR, CENTRAL_SQLITE_PATH


dash.register_page(__name__, path="/earnings-quality", name="Earnings Quality", order=2)


# Display name mapping for ag-grid columns
DISPLAY_NAME_MAP = {
    "end": "Period End",
    "ticker": "Ticker",
    "Fiscal Year": "Fiscal Year",
    "Fiscal Quarter": "Fiscal Quarter",
    "NetIncomeLoss": "Net Income",
    "NetCashProvidedByUsedInOperatingActivities": "Operating Cash Flow",
    "Assets": "Assets",
    "AverageTotalAssets": "Avg Total Assets",
    "AccrualRatio": "Accrual Ratio",
    "Revenues": "Revenue",
    "EarningsGrowthPct": "Earnings Growth %",
    "YoYGrowthPct": "YoY Growth %",
    "AccelerationPct": "Growth Acceleration %",
}

# Columns for formatting
CURRENCY_COLUMNS = [
    "NetIncomeLoss",
    "NetCashProvidedByUsedInOperatingActivities",
    "Assets",
    "AverageTotalAssets",
    "Revenues",
]
PERCENT_COLUMNS = [
    "EarningsGrowthPct",
    "YoYGrowthPct",
    "AccelerationPct",
]

# SEC API headers (SEC requires a descriptive User-Agent with contact info)
SEC_USER_AGENT = os.getenv(
    "SEC_EDGAR_USER_AGENT",
    f"EdgarDash/1.0 ({os.getenv('SEC_CONTACT_EMAIL', 'support@example.com')})",
)
HEADERS = {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}

# Parquet paths
EARNINGS_PARQUET_PATH = Path(DATA_DIR) / "earnings_quality.parquet"
EARNINGS_10K_GROWTH_PARQUET_PATH = Path(DATA_DIR) / "earnings_10k_growth.parquet"
EARNINGS_10Q_PARQUET_PATH = Path(DATA_DIR) / "earnings_10q.parquet"


# -----------------------------
# Logger utility
# -----------------------------
def _get_logger() -> "logging.Logger":
    import logging

    logger = logging.getLogger("edgar_earnings_quality")
    if logger.handlers:
        return logger
    logger.setLevel(logging.WARNING)
    if hasattr(DATA_DIR, "mkdir"):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(
        DATA_DIR / "edgar_earnings_quality.log", encoding="utf-8"
    )
    formatter = logging.Formatter(
        "%(asctime)sZ %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


# Ensure this is defined before any function uses it
EARNINGS_SQLITE_PATH = CENTRAL_SQLITE_PATH


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalize_ticker(ticker: str) -> str:
    return (ticker or "").strip().upper()


def _format_refreshed(ticker: str) -> str:
    et_time = datetime.now(ZoneInfo("America/New_York")).strftime(
        "%Y-%m-%d %H:%M:%S %Z"
    )
    return f"Last refreshed {_normalize_ticker(ticker)} at {et_time}"


def _save_parquet_snapshot(data: pd.DataFrame, parquet_path: Path) -> Path:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(parquet_path, index=False)
    return parquet_path


def _order_display_columns(columns: list[str]) -> list[str]:
    priority = [
        "end",
        "ticker",
        "Fiscal Year",
        "Fiscal Quarter",
        "NetIncomeLoss",
        "NetCashProvidedByUsedInOperatingActivities",
        "Assets",
        "AverageTotalAssets",
        "AccrualRatio",
        "Revenues",
        "EarningsGrowthPct",
        "YoYGrowthPct",
        "AccelerationPct",
    ]
    ordered = [c for c in priority if c in columns]
    ordered.extend([c for c in columns if c not in ordered])
    return ordered


def _build_column_defs(columns: list[str]) -> list[dict]:
    defs: list[dict] = []
    for col in columns:
        col_def: dict = {
            "headerName": DISPLAY_NAME_MAP.get(col, col),
            "field": col,
            "sortable": True,
            "filter": True,
            "resizable": True,
        }
        if col in CURRENCY_COLUMNS:
            col_def["type"] = "rightAligned"
            col_def["valueFormatter"] = {
                "function": "params.value == null ? '' : '$' + Number(params.value).toLocaleString(undefined, {maximumFractionDigits: 0})"
            }
        elif col in PERCENT_COLUMNS or col == "AccrualRatio":
            col_def["type"] = "rightAligned"
            col_def["valueFormatter"] = {
                "function": "params.value == null ? '' : (Number(params.value) * 100).toFixed(2) + '%'"
            }
        defs.append(col_def)
    return defs


def _upsert_sqlite_earnings(
    data: pd.DataFrame, db_path: Path = EARNINGS_SQLITE_PATH
) -> tuple[int, int]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS earnings_quality (
                ticker TEXT NOT NULL,
                period_end TEXT NOT NULL,
                net_income REAL,
                operating_cash_flow REAL,
                assets REAL,
                avg_assets REAL,
                accrual_ratio REAL,
                pulled_at_utc TEXT NOT NULL,
                PRIMARY KEY (ticker, period_end)
            )
            """
        )
        rows = [
            (
                row.ticker,
                str(row.end),
                float(row.NetIncomeLoss),
                float(row.NetCashProvidedByUsedInOperatingActivities),
                float(row.Assets),
                float(row.AverageTotalAssets),
                float(row.AccrualRatio),
                row.pulled_at_utc,
            )
            for row in data.itertuples(index=False)
        ]
        before = conn.execute("SELECT COUNT(*) FROM earnings_quality").fetchone()[0]
        conn.executemany(
            """
            INSERT OR REPLACE INTO earnings_quality
            (ticker, period_end, net_income, operating_cash_flow, assets, avg_assets, accrual_ratio, pulled_at_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        after = conn.execute("SELECT COUNT(*) FROM earnings_quality").fetchone()[0]
    return len(rows), after - before


def _upsert_sqlite_10q_earnings(
    data: pd.DataFrame, db_path: Path = EARNINGS_SQLITE_PATH
) -> tuple[int, int]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS earnings_10q (
                ticker TEXT NOT NULL,
                period_end TEXT NOT NULL,
                net_income REAL,
                pulled_at_utc TEXT NOT NULL,
                PRIMARY KEY (ticker, period_end)
            )
            """
        )
        rows = [
            (
                row.ticker,
                str(row.end),
                float(row.NetIncomeLoss),
                row.pulled_at_utc,
            )
            for row in data.itertuples(index=False)
        ]
        before = conn.execute("SELECT COUNT(*) FROM earnings_10q").fetchone()[0]
        conn.executemany(
            """
            INSERT OR REPLACE INTO earnings_10q
            (ticker, period_end, net_income, pulled_at_utc)
            VALUES (?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        after = conn.execute("SELECT COUNT(*) FROM earnings_10q").fetchone()[0]
    return len(rows), after - before


def _upsert_sqlite_annual_earnings_growth(
    data: pd.DataFrame, db_path: Path = EARNINGS_SQLITE_PATH
) -> tuple[int, int]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS earnings_growth_annual (
                ticker TEXT NOT NULL,
                period_end TEXT NOT NULL,
                net_income REAL,
                earnings_growth_pct REAL,
                pulled_at_utc TEXT NOT NULL,
                PRIMARY KEY (ticker, period_end)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_eq_growth_ticker ON earnings_growth_annual(ticker)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_eq_growth_period_end ON earnings_growth_annual(period_end)"
        )

        rows = [
            (
                row.ticker,
                str(row.end),
                float(row.NetIncomeLoss),
                None
                if pd.isna(row.EarningsGrowthPct)
                else float(row.EarningsGrowthPct),
                row.pulled_at_utc,
            )
            for row in data.itertuples(index=False)
        ]
        before = conn.execute("SELECT COUNT(*) FROM earnings_growth_annual").fetchone()[
            0
        ]
        conn.executemany(
            """
            INSERT OR REPLACE INTO earnings_growth_annual
            (ticker, period_end, net_income, earnings_growth_pct, pulled_at_utc)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        after = conn.execute("SELECT COUNT(*) FROM earnings_growth_annual").fetchone()[
            0
        ]
    return len(rows), after - before


def _fetch_cached_earnings(
    ticker: str, db_path: Path = EARNINGS_SQLITE_PATH
) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    sql = """
    SELECT
        period_end AS end,
        net_income AS NetIncomeLoss,
        operating_cash_flow AS NetCashProvidedByUsedInOperatingActivities,
        assets AS Assets,
        avg_assets AS AverageTotalAssets,
        accrual_ratio AS AccrualRatio
    FROM earnings_quality
    WHERE ticker = ?
    ORDER BY period_end
    """
    with sqlite3.connect(db_path) as conn:
        try:
            return pd.read_sql_query(sql, conn, params=(ticker,))
        except Exception:
            # Centralized DB may exist before this table is initialized.
            return pd.DataFrame()


def _fetch_cached_10q_earnings(
    ticker: str, db_path: Path = EARNINGS_SQLITE_PATH
) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    sql = """
    SELECT
        period_end AS end,
        net_income AS NetIncomeLoss
    FROM earnings_10q
    WHERE ticker = ?
    ORDER BY period_end
    """
    with sqlite3.connect(db_path) as conn:
        try:
            return pd.read_sql_query(sql, conn, params=(ticker,))
        except Exception:
            return pd.DataFrame()


def _fetch_cached_annual_earnings_growth(
    ticker: str, db_path: Path = EARNINGS_SQLITE_PATH
) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    sql = """
    SELECT
        period_end AS end,
        net_income AS NetIncomeLoss,
        earnings_growth_pct AS EarningsGrowthPct
    FROM earnings_growth_annual
    WHERE ticker = ?
    ORDER BY period_end
    """
    with sqlite3.connect(db_path) as conn:
        try:
            cached = pd.read_sql_query(sql, conn, params=(ticker,))
            return _normalize_annual_earnings_growth(cached)
        except Exception:
            return pd.DataFrame()


def _safe_get_json(url: str) -> dict | None:
    logger = _get_logger()
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
    except Exception:
        logger.exception("SEC request failed: %s", url)
        return None

    if resp.status_code != 200:
        logger.warning("SEC request non-200: %s status=%s", url, resp.status_code)
        return None

    content_type = resp.headers.get("Content-Type", "")
    if "application/json" not in content_type.lower():
        logger.warning(
            "SEC request unexpected content-type: %s content-type=%s",
            url,
            content_type,
        )
        return None

    try:
        return resp.json()
    except Exception:
        logger.exception("SEC response JSON decode failed: %s", url)
        return None


# -----------------------------
# Get company CIK from ticker
# -----------------------------
def get_cik(ticker):
    url = "https://www.sec.gov/files/company_tickers.json"
    data = _safe_get_json(url)
    if not data:
        return None

    for item in data.values():
        if item.get("ticker", "").lower() == (ticker or "").lower():
            return str(item.get("cik_str", "")).zfill(10)

    return None


# -----------------------------
# Pull XBRL company facts
# -----------------------------
def get_company_facts(cik):
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    return _safe_get_json(url) or {}


# -----------------------------
# Extract quarterly data
# -----------------------------
def extract_series(facts, tag, forms: tuple[str, ...] = ("10-Q", "10-K")):
    try:
        usgaap = facts["facts"]["us-gaap"][tag]["units"]["USD"]
        df = pd.DataFrame(usgaap)

        df = df[df["form"].isin(forms)]
        if "filed" in df.columns:
            df = df.sort_values(["end", "filed"])
            df = df.drop_duplicates(subset=["end"], keep="last")
        else:
            df = df.sort_values("end").drop_duplicates(subset=["end"], keep="last")

        return df[["end", "val"]].rename(columns={"val": tag})
    except Exception:
        return pd.DataFrame()


def _normalize_annual_earnings_growth(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    annual_df = df.copy()
    annual_df["end"] = pd.to_datetime(annual_df["end"], errors="coerce")
    annual_df = annual_df.dropna(subset=["end"]).sort_values("end")
    annual_df["fiscal_year"] = annual_df["end"].dt.year
    annual_df = annual_df.drop_duplicates(subset=["fiscal_year"], keep="last")
    annual_df = annual_df.drop(columns=["fiscal_year"])
    annual_df["end"] = annual_df["end"].dt.strftime("%Y-%m-%d")
    annual_df["EarningsGrowthPct"] = annual_df["NetIncomeLoss"].pct_change()
    annual_df["AccelerationPct"] = annual_df["EarningsGrowthPct"].diff()
    return annual_df.tail(10).reset_index(drop=True)


def _add_fiscal_columns(
    df_display: pd.DataFrame, fy_end_month: int | None
) -> pd.DataFrame:
    if df_display is None or df_display.empty:
        return df_display

    end_dt = pd.to_datetime(df_display["end"], errors="coerce")
    df_display["Fiscal Year"] = end_dt.dt.year
    df_display["Fiscal Quarter"] = end_dt.dt.quarter.map(lambda q: f"Q{q}")

    if fy_end_month is not None:
        adj_end = end_dt - pd.Timedelta(days=7)
        fy_start_month = (fy_end_month % 12) + 1
        end_month = adj_end.dt.month
        offset = (end_month - fy_start_month) % 12
        df_display["Fiscal Quarter"] = (offset // 3 + 1).map(lambda q: f"Q{q}")
        df_display["Fiscal Year"] = adj_end.dt.year + (end_month > fy_end_month)

    df_display["Fiscal Year"] = pd.to_numeric(
        df_display["Fiscal Year"], errors="coerce"
    ).astype("Int64")

    if "fp" in df_display.columns:
        fp_series = df_display["fp"].astype(str).str.upper()
        valid_fp = fp_series.isin(["Q1", "Q2", "Q3", "Q4"])
        df_display.loc[valid_fp, "Fiscal Quarter"] = fp_series[valid_fp]

    return df_display


def _compute_quarterly_yoy(df: pd.DataFrame, fy_end_month: int | None) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype="float")

    series = df.copy()
    series["end"] = pd.to_datetime(series["end"], errors="coerce")
    series = series.dropna(subset=["end"]).sort_values("end").reset_index(drop=True)
    series = _add_fiscal_columns(series, fy_end_month)

    yoy_vals: list[float | None] = [None] * len(series)
    lookup = {}
    for _, row in series.iterrows():
        lookup[(row["Fiscal Year"], row["Fiscal Quarter"])] = row["NetIncomeLoss"]

    for idx, row in series.iterrows():
        key = (row["Fiscal Year"], row["Fiscal Quarter"])
        if key[0] is None or key[1] is None:
            continue
        prev_key = (key[0] - 1, key[1])
        prev_val = lookup.get(prev_key)
        if prev_val is None or pd.isna(prev_val) or pd.isna(row["NetIncomeLoss"]):
            continue
        if prev_val == 0:
            continue
        yoy_vals[idx] = row["NetIncomeLoss"] / prev_val - 1

    return pd.Series(yoy_vals)


def _normalize_quarterly_earnings(
    df: pd.DataFrame, fy_end_month: int | None = None
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    quarterly_df = df.copy()
    quarterly_df["end"] = pd.to_datetime(quarterly_df["end"], errors="coerce")
    quarterly_df = quarterly_df.dropna(subset=["end"]).sort_values("end")
    quarterly_df = quarterly_df.drop_duplicates(
        subset=["end"], keep="last"
    ).reset_index(drop=True)

    quarterly_df["YoYGrowthPct"] = _compute_quarterly_yoy(quarterly_df, fy_end_month)

    # Add fiscal quarter label before formatting dates
    quarterly_df = _add_fiscal_columns(quarterly_df, fy_end_month)

    quarterly_df["end"] = quarterly_df["end"].dt.strftime("%Y-%m-%d")
    display_cols = [
        "end",
        "Fiscal Year",
        "Fiscal Quarter",
        "NetIncomeLoss",
        "YoYGrowthPct",
    ]
    # Only keep columns that actually exist (Fiscal Year/Quarter need _add_fiscal_columns)
    display_cols = [c for c in display_cols if c in quarterly_df.columns]
    return quarterly_df[display_cols].tail(48).reset_index(drop=True)


def _infer_fiscal_year_end_month_from_facts(facts: dict) -> int | None:
    usgaap = facts.get("facts", {}).get("us-gaap", {}).get("NetIncomeLoss", {})
    annual_units = pd.DataFrame(usgaap.get("units", {}).get("USD", []))
    if annual_units.empty:
        return None

    annual_units = annual_units[annual_units["form"].isin(["10-K", "10-K/A"])]
    if annual_units.empty or "end" not in annual_units.columns:
        return None

    annual_ends = pd.to_datetime(annual_units["end"], errors="coerce").dropna()
    if annual_ends.empty:
        return None
    return int(annual_ends.max().month)


def _derive_q4_earnings_from_annual(
    quarterly_df: pd.DataFrame,
    annual_ni_df: pd.DataFrame,
    fy_end_month: int | None,
) -> pd.DataFrame:
    """Derive Q4 net income = Annual NI − (Q1 + Q2 + Q3) for each fiscal year."""
    if annual_ni_df is None or annual_ni_df.empty:
        return quarterly_df
    if quarterly_df is None or quarterly_df.empty:
        return quarterly_df

    q = quarterly_df.copy()
    q["end"] = pd.to_datetime(q["end"], errors="coerce")
    q = _add_fiscal_columns(q, fy_end_month)

    a = annual_ni_df.copy()
    a["end"] = pd.to_datetime(a["end"], errors="coerce")
    a = _add_fiscal_columns(a, fy_end_month)

    derived = []
    for _, ann_row in a.iterrows():
        fy = ann_row.get("Fiscal Year")
        annual_val = ann_row.get("NetIncomeLoss")
        ann_end = ann_row.get("end")
        if fy is None or pd.isna(fy) or annual_val is None or pd.isna(annual_val):
            continue

        q_mask = (q["Fiscal Year"] == fy) & (
            q["Fiscal Quarter"].isin(["Q1", "Q2", "Q3"])
        )
        prior_qs = q[q_mask]
        if len(prior_qs) != 3:
            continue

        q4_val = int(annual_val) - int(prior_qs["NetIncomeLoss"].sum())
        row: dict = {"end": ann_end, "NetIncomeLoss": q4_val}
        if "fp" in quarterly_df.columns:
            row["fp"] = "Q4"
        if "fy" in quarterly_df.columns:
            row["fy"] = int(fy)
        derived.append(row)

    if not derived:
        return quarterly_df

    result = pd.concat([quarterly_df, pd.DataFrame(derived)], ignore_index=True)
    return result.sort_values("end").reset_index(drop=True)


def _build_quarterly_10q_earnings_frame(facts) -> pd.DataFrame:
    usgaap = facts.get("facts", {}).get("us-gaap", {}).get("NetIncomeLoss", {})
    quarterly_units = pd.DataFrame(usgaap.get("units", {}).get("USD", []))
    if quarterly_units.empty:
        return pd.DataFrame()

    quarterly_units = quarterly_units[quarterly_units["form"].isin(["10-Q", "10-Q/A"])]
    if quarterly_units.empty:
        return pd.DataFrame()

    if "start" in quarterly_units.columns and "end" in quarterly_units.columns:
        quarterly_units["start"] = pd.to_datetime(
            quarterly_units["start"], errors="coerce"
        )
        quarterly_units["end"] = pd.to_datetime(quarterly_units["end"], errors="coerce")
        quarterly_units = quarterly_units.dropna(subset=["start", "end"])
        quarterly_units["duration_days"] = (
            quarterly_units["end"] - quarterly_units["start"]
        ).dt.days
        quarterly_units = quarterly_units[
            quarterly_units["duration_days"].between(60, 120)
        ]
    if quarterly_units.empty:
        return pd.DataFrame()

    if "filed" in quarterly_units.columns:
        quarterly_units["filed"] = pd.to_datetime(
            quarterly_units["filed"], errors="coerce"
        )
        quarterly_units = quarterly_units.sort_values(["end", "filed"])
    else:
        quarterly_units = quarterly_units.sort_values("end")

    cols = ["end", "val"]
    if "fp" in quarterly_units.columns:
        cols.append("fp")
    if "fy" in quarterly_units.columns:
        cols.append("fy")

    quarterly_df = (
        quarterly_units[cols]
        .rename(columns={"val": "NetIncomeLoss"})
        .drop_duplicates(subset=["end"], keep="last")
        .reset_index(drop=True)
    )

    fy_end_month = _infer_fiscal_year_end_month_from_facts(facts)

    # Extract annual 10-K NI to derive Q4 (companies file 10-K for Q4, not 10-Q)
    all_units = pd.DataFrame(
        facts.get("facts", {})
        .get("us-gaap", {})
        .get("NetIncomeLoss", {})
        .get("units", {})
        .get("USD", [])
    )
    annual_ni_df: pd.DataFrame | None = None
    if not all_units.empty and "form" in all_units.columns:
        ann = all_units[all_units["form"].isin(["10-K", "10-K/A"])].copy()
        if not ann.empty and "start" in ann.columns and "end" in ann.columns:
            ann["start"] = pd.to_datetime(ann["start"], errors="coerce")
            ann["end"] = pd.to_datetime(ann["end"], errors="coerce")
            ann = ann.dropna(subset=["start", "end"])
            ann["duration_days"] = (ann["end"] - ann["start"]).dt.days
            ann = ann[ann["duration_days"].between(300, 380)]
        if not ann.empty:
            if "filed" in ann.columns:
                ann["filed"] = pd.to_datetime(ann["filed"], errors="coerce")
                ann = ann.sort_values(["end", "filed"])
            annual_ni_df = (
                ann[["end", "val"]]
                .rename(columns={"val": "NetIncomeLoss"})
                .drop_duplicates(subset=["end"], keep="last")
                .reset_index(drop=True)
            )

    quarterly_df = _derive_q4_earnings_from_annual(
        quarterly_df, annual_ni_df, fy_end_month
    )
    return _normalize_quarterly_earnings(quarterly_df, fy_end_month)


_REVENUE_TAGS = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
]


def _extract_annual_revenue(facts) -> pd.DataFrame:
    """Return DataFrame[end, Revenues] from 10-K filings with fallback across revenue tags."""
    usgaap = facts.get("facts", {}).get("us-gaap", {})
    candidates: list[pd.DataFrame] = []

    for idx, tag in enumerate(_REVENUE_TAGS):
        units = pd.DataFrame(usgaap.get(tag, {}).get("units", {}).get("USD", []))
        if units.empty:
            continue
        if "form" in units.columns:
            units = units[units["form"].isin(["10-K", "10-K/A"])]
        if units.empty or "end" not in units.columns or "val" not in units.columns:
            continue

        units["end"] = pd.to_datetime(units["end"], errors="coerce")
        units = units.dropna(subset=["end"])

        # Prefer annual-duration facts when period bounds are present.
        if "start" in units.columns:
            units["start"] = pd.to_datetime(units["start"], errors="coerce")
            duration_ready = units["start"].notna() & units["end"].notna()
            if duration_ready.any():
                units = units[
                    ~duration_ready
                    | ((units["end"] - units["start"]).dt.days.between(300, 380))
                ]
        if units.empty:
            continue

        if "filed" in units.columns:
            units["filed"] = pd.to_datetime(units["filed"], errors="coerce")
        else:
            units["filed"] = pd.NaT

        compact = units[["end", "val", "filed"]].copy()
        compact["tag_priority"] = idx
        candidates.append(compact)

    if not candidates:
        return pd.DataFrame()

    merged = pd.concat(candidates, ignore_index=True)
    merged["val"] = pd.to_numeric(merged["val"], errors="coerce")
    merged = merged.dropna(subset=["end", "val"])
    if merged.empty:
        return pd.DataFrame()

    # Per fiscal-year end: prefer primary tag first, otherwise fallback tags; within a tag use latest filing.
    merged = merged.sort_values(
        ["end", "tag_priority", "filed"],
        ascending=[True, True, False],
        na_position="last",
    )
    merged = merged.drop_duplicates(subset=["end"], keep="first")
    return merged[["end", "val"]].rename(columns={"val": "Revenues"})


def _build_annual_earnings_growth_frame(facts) -> pd.DataFrame:
    annual_net_income = extract_series(facts, "NetIncomeLoss", forms=("10-K", "10-K/A"))
    if annual_net_income.empty:
        return pd.DataFrame()

    usgaap = facts.get("facts", {}).get("us-gaap", {}).get("NetIncomeLoss", {})
    annual_units = pd.DataFrame(usgaap.get("units", {}).get("USD", []))
    if annual_units.empty:
        return pd.DataFrame()

    annual_units = annual_units[annual_units["form"].isin(["10-K", "10-K/A"])]
    if "start" not in annual_units.columns or "end" not in annual_units.columns:
        return pd.DataFrame()

    annual_units["start"] = pd.to_datetime(annual_units["start"], errors="coerce")
    annual_units["end"] = pd.to_datetime(annual_units["end"], errors="coerce")
    annual_units = annual_units.dropna(subset=["start", "end"])
    annual_units["duration_days"] = (
        annual_units["end"] - annual_units["start"]
    ).dt.days
    annual_units = annual_units[annual_units["duration_days"].between(300, 380)]

    if annual_units.empty:
        return pd.DataFrame()

    if "filed" in annual_units.columns:
        annual_units["filed"] = pd.to_datetime(annual_units["filed"], errors="coerce")
        annual_units = annual_units.sort_values(["end", "filed", "duration_days"])
    else:
        annual_units = annual_units.sort_values(["end", "duration_days"])

    annual_net_income = annual_units[["end", "val"]].rename(
        columns={"val": "NetIncomeLoss"}
    )
    revenue_df = _extract_annual_revenue(facts)
    if not revenue_df.empty:
        annual_net_income = annual_net_income.merge(revenue_df, on="end", how="left")
    return _normalize_annual_earnings_growth(annual_net_income)


# -----------------------------
# Build financial dataframe
# -----------------------------
def build_financials(ticker, use_cache: bool = True, cache_only: bool = False):
    logger = _get_logger()

    ticker_norm = _normalize_ticker(ticker)
    if not ticker_norm:
        return None

    if use_cache:
        cached = _fetch_cached_earnings(ticker_norm)
        if cached is not None and not cached.empty:
            return cached.tail(12)
        if cache_only:
            return pd.DataFrame()

    cik = get_cik(ticker_norm)
    if not cik:
        return None

    facts = get_company_facts(cik)

    ni = extract_series(facts, "NetIncomeLoss")
    ocf = extract_series(facts, "NetCashProvidedByUsedInOperatingActivities")
    assets = extract_series(facts, "Assets")

    df = ni.merge(ocf, on="end", how="inner")
    df = df.merge(assets, on="end", how="inner")

    df["AverageTotalAssets"] = df["Assets"].rolling(2).mean()

    # Accrual Ratio
    df["AccrualRatio"] = (
        df["NetIncomeLoss"] - df["NetCashProvidedByUsedInOperatingActivities"]
    ) / df["AverageTotalAssets"]

    df.dropna(inplace=True)

    df = df.tail(12)

    persist_df = df.copy()
    persist_df["ticker"] = ticker_norm
    persist_df["pulled_at_utc"] = _utc_now_iso()

    try:
        _save_parquet_snapshot(persist_df, EARNINGS_PARQUET_PATH)
        _upsert_sqlite_earnings(persist_df, EARNINGS_SQLITE_PATH)
    except Exception:
        logger.exception("persistence failed for ticker=%s", ticker_norm)

    return df


def build_annual_earnings_growth(
    ticker, use_cache: bool = True, cache_only: bool = False
):
    logger = _get_logger()

    ticker_norm = _normalize_ticker(ticker)
    if not ticker_norm:
        return None

    if use_cache:
        cached = _fetch_cached_annual_earnings_growth(ticker_norm)
        if cached is not None and not cached.empty:
            return cached.tail(10)
        if cache_only:
            return pd.DataFrame()

    cik = get_cik(ticker_norm)
    if not cik:
        return None

    facts = get_company_facts(cik)
    annual_df = _build_annual_earnings_growth_frame(facts)
    if annual_df.empty:
        return annual_df

    persist_df = annual_df.copy()
    persist_df["ticker"] = ticker_norm
    persist_df["pulled_at_utc"] = _utc_now_iso()

    try:
        _save_parquet_snapshot(persist_df, EARNINGS_10K_GROWTH_PARQUET_PATH)
        _upsert_sqlite_annual_earnings_growth(persist_df, EARNINGS_SQLITE_PATH)
    except Exception:
        logger.exception("10-K growth persistence failed for ticker=%s", ticker_norm)

    return annual_df


def build_quarterly_10q_earnings(
    ticker, use_cache: bool = True, cache_only: bool = False
):
    logger = _get_logger()

    ticker_norm = _normalize_ticker(ticker)
    if not ticker_norm:
        return None

    if use_cache:
        cached = _fetch_cached_10q_earnings(ticker_norm)
        if cached is not None and not cached.empty:
            return _normalize_quarterly_earnings(cached)
        if cache_only:
            return pd.DataFrame()

    cik = get_cik(ticker_norm)
    if not cik:
        return None

    facts = get_company_facts(cik)
    quarterly_df = _build_quarterly_10q_earnings_frame(facts)
    if quarterly_df.empty:
        return quarterly_df

    persist_df = quarterly_df.copy()
    persist_df["ticker"] = ticker_norm
    persist_df["pulled_at_utc"] = _utc_now_iso()

    try:
        _save_parquet_snapshot(persist_df, EARNINGS_10Q_PARQUET_PATH)
        _upsert_sqlite_10q_earnings(persist_df, EARNINGS_SQLITE_PATH)
    except Exception:
        logger.exception("10-Q earnings persistence failed for ticker=%s", ticker_norm)

    return quarterly_df


# -----------------------------
# Create Plotly Figure
# -----------------------------
def create_figure(df, ticker, refreshed_text: str = ""):

    fig = go.Figure()

    # Net Income
    fig.add_trace(
        go.Scatter(
            x=df["end"],
            y=df["NetIncomeLoss"],
            name="Net Income",
            mode="lines+markers",
            yaxis="y1",
        )
    )

    # Operating Cash Flow
    fig.add_trace(
        go.Scatter(
            x=df["end"],
            y=df["NetCashProvidedByUsedInOperatingActivities"],
            name="Operating Cash Flow",
            mode="lines+markers",
            yaxis="y1",
        )
    )

    # Accrual Ratio (secondary axis)
    fig.add_trace(
        go.Bar(
            x=df["end"],
            y=df["AccrualRatio"],
            name="Accrual Ratio",
            opacity=0.4,
            yaxis="y2",
        )
    )

    # Danger zone shading
    fig.add_hrect(
        y0=0.15,
        y1=1,
        line_width=0,
        fillcolor="rgba(255,255,255,0)",
        opacity=0.0,
        yref="y2",
    )

    title = f"{ticker.upper()} Earnings Quality"
    if refreshed_text:
        title = f"{title} · {refreshed_text}"

    fig.update_layout(
        title=title,
        xaxis_title="Quarter",
        yaxis=dict(title="USD"),
        yaxis2=dict(
            title="Accrual Ratio",
            overlaying="y",
            side="right",
        ),
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        height=650,
    )
    fig.update_xaxes(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb")
    fig.update_yaxes(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb")

    return fig


def create_10k_growth_figure(df: pd.DataFrame, ticker: str, refreshed_text: str = ""):
    title = f"{_normalize_ticker(ticker)} 10-K Earnings Growth"
    if refreshed_text:
        title = f"{title} · {refreshed_text}"

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.3, 0.2],
        vertical_spacing=0.06,
        subplot_titles=(
            "A) Net Income (USD)  |  Stock Price",
            "B) Growth %",
            "C) Growth Acceleration / Deceleration",
        ),
        specs=[
            [{"secondary_y": True}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
        ],
    )

    if df is None or df.empty:
        fig.update_layout(
            title=title,
            template="plotly_white",
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="black"),
            height=780,
        )
        fig.add_annotation(
            text="No 10-K earnings growth data available.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14),
        )
        return fig

    series = df.copy()
    series["end"] = pd.to_datetime(series["end"], errors="coerce")
    series = series.dropna(subset=["end"]).sort_values("end")
    if series.empty:
        return fig

    fig.add_trace(
        go.Bar(
            x=series["end"],
            y=series["NetIncomeLoss"],
            name="Net Income",
            marker_color="#2563eb",
            opacity=0.8,
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    if ticker:
        end_dates = pd.to_datetime(series["end"], errors="coerce").dropna()
        if not end_dates.empty:
            price_start = str((end_dates.min() - pd.DateOffset(years=1)).date())
            price_end = str((end_dates.max() + pd.DateOffset(years=1)).date())
            price_df = _fetch_yearly_prices(ticker, price_start, price_end)
            if not price_df.empty:
                fig.add_trace(
                    go.Scatter(
                        x=price_df["date"],
                        y=price_df["close"],
                        name="Stock Price",
                        mode="lines+markers",
                        line=dict(color="#6366f1", width=2),
                        marker=dict(size=6),
                    ),
                    row=1,
                    col=1,
                    secondary_y=True,
                )
                fig.update_yaxes(
                    title_text="Stock Price (USD)",
                    secondary_y=True,
                    row=1,
                    col=1,
                    tickprefix="$",
                    gridcolor="rgba(0,0,0,0)",
                )

    if "EarningsGrowthPct" in series.columns:
        growth = series["EarningsGrowthPct"]
        growth_colors = [
            "#22c55e" if (v is not None and not pd.isna(v) and v >= 0) else "#ef4444"
            for v in growth
        ]
        fig.add_trace(
            go.Bar(
                x=series["end"],
                y=growth,
                name="Growth %",
                marker_color=growth_colors,
                opacity=0.75,
            ),
            row=2,
            col=1,
        )
        growth_ma = growth.rolling(window=3, min_periods=2).mean()
        fig.add_trace(
            go.Scatter(
                x=series["end"],
                y=growth_ma,
                name="3-Yr Avg Growth",
                mode="lines",
                line=dict(color="#f59e0b", width=2.5, dash="dot"),
            ),
            row=2,
            col=1,
        )

        accel = (
            series["AccelerationPct"]
            if "AccelerationPct" in series.columns
            else growth.diff()
        )
        accel_colors = [
            "#16a34a" if (v is not None and not pd.isna(v) and v >= 0) else "#dc2626"
            for v in accel
        ]
        fig.add_trace(
            go.Bar(
                x=series["end"],
                y=accel,
                name="Growth Acceleration",
                marker_color=accel_colors,
                opacity=0.75,
            ),
            row=3,
            col=1,
        )
        accel_ma = (
            pd.to_numeric(accel, errors="coerce")
            .rolling(window=3, min_periods=2)
            .mean()
        )
        fig.add_trace(
            go.Scatter(
                x=series["end"],
                y=accel_ma,
                name="3-Yr Avg Acceleration",
                mode="lines",
                line=dict(color="#7c3aed", width=2.5),
            ),
            row=3,
            col=1,
        )
        fig.add_hline(y=0, line_color="#6b7280", line_width=1, row=3, col=1)

    fig.update_layout(
        title=title,
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        height=780,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        bargap=0.2,
    )
    fig.update_xaxes(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb")
    fig.update_yaxes(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb")
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.update_yaxes(tickformat=".0%", row=3, col=1)
    return fig


def _fetch_yearly_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Return DataFrame[date, close] with yearly (12-month resampled) close prices."""
    ticker_norm = _normalize_ticker(ticker)
    if not ticker_norm:
        return pd.DataFrame(columns=["date", "close"])
    try:
        raw = yf.Ticker(ticker_norm).history(start=start, end=end, auto_adjust=True)
    except Exception:
        return pd.DataFrame(columns=["date", "close"])
    if raw is None or raw.empty or "Close" not in raw.columns:
        return pd.DataFrame(columns=["date", "close"])

    raw = raw.reset_index()
    date_col = "Date" if "Date" in raw.columns else raw.columns[0]
    raw[date_col] = pd.to_datetime(
        raw[date_col], errors="coerce", utc=True
    ).dt.tz_localize(None)
    raw = raw[[date_col, "Close"]].rename(columns={date_col: "date", "Close": "close"})
    raw = raw.dropna()
    raw = raw.set_index("date")["close"].resample("YE").last().dropna().reset_index()
    return raw


def _fetch_quarterly_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Return DataFrame[date, close] with quarterly (3-month resampled) close prices."""
    ticker_norm = _normalize_ticker(ticker)
    if not ticker_norm:
        return pd.DataFrame(columns=["date", "close"])
    try:
        raw = yf.Ticker(ticker_norm).history(start=start, end=end, auto_adjust=True)
    except Exception:
        return pd.DataFrame(columns=["date", "close"])
    if raw is None or raw.empty or "Close" not in raw.columns:
        return pd.DataFrame(columns=["date", "close"])
    raw = raw.reset_index()
    date_col = "Date" if "Date" in raw.columns else raw.columns[0]
    raw[date_col] = pd.to_datetime(
        raw[date_col], errors="coerce", utc=True
    ).dt.tz_localize(None)
    raw = raw[[date_col, "Close"]].rename(columns={date_col: "date", "Close": "close"})
    raw = raw.dropna()
    # Resample to end-of-quarter so the line aligns with quarterly bars
    raw = raw.set_index("date")["close"].resample("QE").last().dropna().reset_index()
    return raw


def _predict_next_quarter_earnings(series: pd.DataFrame):
    """Predict next-quarter net income using a blend of recent trend and YoY signal."""
    if series is None or series.empty or "NetIncomeLoss" not in series.columns:
        return None

    work = series.copy()
    work["end"] = pd.to_datetime(work["end"], errors="coerce")
    work["NetIncomeLoss"] = pd.to_numeric(work["NetIncomeLoss"], errors="coerce")
    work = work.dropna(subset=["end", "NetIncomeLoss"]).sort_values("end")
    if work.empty:
        return None

    ni = work["NetIncomeLoss"]
    last_val = float(ni.iloc[-1])
    last_end = work["end"].iloc[-1]
    next_end = last_end + pd.DateOffset(months=3)
    # If the immediate next quarter has already passed, roll forward to upcoming quarter.
    today = pd.Timestamp.today().normalize()
    while next_end <= today:
        next_end = next_end + pd.DateOffset(months=3)

    # Short-term trend from recent quarter-over-quarter deltas.
    recent_ni = ni.tail(6)
    delta_mean = recent_ni.diff().dropna().mean()
    trend_pred = last_val + (float(delta_mean) if pd.notna(delta_mean) else 0.0)

    # Growth-based component from recent YoY growth values when available.
    growth_pred = trend_pred
    if "YoYGrowthPct" in work.columns:
        yoy = pd.to_numeric(work["YoYGrowthPct"], errors="coerce").dropna().tail(4)
        if not yoy.empty:
            growth_pred = last_val * (1 + float(yoy.mean()))

    pred_val = 0.6 * trend_pred + 0.4 * growth_pred
    if not pd.notna(pred_val):
        return None

    return {
        "next_end": next_end,
        "predicted_net_income": float(pred_val),
        "last_end": last_end,
        "last_net_income": last_val,
    }


def _predict_next_quarter_earnings_diagnostics(series: pd.DataFrame):
    """Predict next-quarter earnings using diagnostics from same-quarter trends."""
    if series is None or series.empty or "NetIncomeLoss" not in series.columns:
        return None

    work = series.copy()
    work["end"] = pd.to_datetime(work["end"], errors="coerce")
    work["NetIncomeLoss"] = pd.to_numeric(work["NetIncomeLoss"], errors="coerce")
    work = work.dropna(subset=["end", "NetIncomeLoss"]).sort_values("end")
    if work.empty:
        return None

    ni = work["NetIncomeLoss"].astype(float)
    last_val = float(ni.iloc[-1])
    last_end = work["end"].iloc[-1]

    if "Fiscal Quarter" in work.columns:
        fq = work["Fiscal Quarter"].astype(str).str.extract(r"([1-4])", expand=False)
        work["__fq"] = pd.to_numeric(fq, errors="coerce")
    else:
        work["__fq"] = ((work["end"].dt.month - 1) // 3) + 1

    if "Fiscal Year" in work.columns:
        work["__fy"] = pd.to_numeric(work["Fiscal Year"], errors="coerce")
    else:
        work["__fy"] = work["end"].dt.year

    work = work.dropna(subset=["__fy", "__fq"]).sort_values(["__fy", "end"])
    if work.empty:
        return _predict_next_quarter_earnings(work)

    next_end = last_end + pd.DateOffset(months=3)
    today = pd.Timestamp.today().normalize()
    while next_end <= today:
        next_end = next_end + pd.DateOffset(months=3)

    last_fy = int(work["__fy"].iloc[-1])
    last_fq = int(work["__fq"].iloc[-1])
    target_fy, target_fq = last_fy, last_fq
    step_end = last_end
    while step_end < next_end:
        step_end = step_end + pd.DateOffset(months=3)
        target_fq += 1
        if target_fq > 4:
            target_fq = 1
            target_fy += 1

    same_q = work[work["__fq"] == target_fq].sort_values("__fy")
    ni_same_q = pd.to_numeric(same_q["NetIncomeLoss"], errors="coerce").dropna()

    # Use same-quarter diagnostics for the target forecast quarter.
    yoy_series = ni_same_q.pct_change()
    accel_series = yoy_series.diff()

    latest_growth = (
        float(yoy_series.dropna().iloc[-1]) if not yoy_series.dropna().empty else None
    )
    latest_accel = (
        float(accel_series.dropna().iloc[-1])
        if not accel_series.dropna().empty
        else None
    )

    def _cagr(vals: pd.Series, years: int) -> float | None:
        if len(vals) < years + 1:
            return None
        start = float(vals.iloc[-(years + 1)])
        end = float(vals.iloc[-1])
        if start <= 0 or end <= 0:
            return None
        return (end / start) ** (1 / years) - 1

    cagr_1 = _cagr(ni_same_q, 1)
    cagr_3 = _cagr(ni_same_q, 3)
    cagr_full = (
        _cagr(ni_same_q, max(len(ni_same_q) - 1, 1)) if len(ni_same_q) > 1 else None
    )

    yoy_recent = pd.to_numeric(yoy_series, errors="coerce").dropna().tail(4)
    yoy_signal = float(yoy_recent.mean()) if not yoy_recent.empty else None

    weighted = []
    if cagr_1 is not None:
        weighted.append((0.5, cagr_1))
    if cagr_3 is not None:
        weighted.append((0.3, cagr_3))
    if cagr_full is not None:
        weighted.append((0.2, cagr_full))
    cagr_signal = (
        sum(w * v for w, v in weighted) / sum(w for w, _ in weighted)
        if weighted
        else None
    )

    if yoy_signal is not None and cagr_signal is not None:
        blended_growth = 0.6 * yoy_signal + 0.4 * cagr_signal
    elif yoy_signal is not None:
        blended_growth = yoy_signal
    elif cagr_signal is not None:
        blended_growth = cagr_signal
    else:
        # Fallback to the existing model when diagnostics are insufficient.
        return _predict_next_quarter_earnings(work)

    regime_adj = 0.0
    if latest_growth is not None and latest_accel is not None:
        if latest_growth >= 0 and latest_accel >= 0:
            regime_adj = 0.02
            regime = "Expanding"
        elif latest_growth >= 0 and latest_accel < 0:
            regime_adj = -0.01
            regime = "Slowing Growth"
        else:
            regime_adj = -0.03
            regime = "Contracting"
    else:
        regime = "n/a"

    accel_adj = float(
        np.clip(latest_accel if latest_accel is not None else 0.0, -0.25, 0.25)
    )
    effective_growth = float(
        np.clip(blended_growth + 0.5 * accel_adj + regime_adj, -0.6, 1.5)
    )

    # Base the forecast on the prior-year same quarter when available.
    prev_same_q = same_q[
        pd.to_numeric(same_q["__fy"], errors="coerce") == (target_fy - 1)
    ]
    if not prev_same_q.empty:
        base_val = float(
            pd.to_numeric(prev_same_q["NetIncomeLoss"], errors="coerce")
            .dropna()
            .iloc[-1]
        )
    elif not ni_same_q.empty:
        base_val = float(ni_same_q.iloc[-1])
    else:
        base_val = last_val

    if base_val <= 0:
        return _predict_next_quarter_earnings(work)

    pred_val = base_val * (1 + effective_growth)
    if not pd.notna(pred_val):
        return None

    return {
        "next_end": next_end,
        "predicted_net_income": float(pred_val),
        "last_end": last_end,
        "last_net_income": last_val,
        "blended_growth": float(blended_growth),
        "regime": regime,
        "target_fq": int(target_fq),
        "target_fy": int(target_fy),
        "base_same_quarter": float(base_val),
    }


def create_10q_earnings_figure(df: pd.DataFrame, ticker: str, refreshed_text: str = ""):
    title = f"{_normalize_ticker(ticker)} 10-Q Earnings"
    if refreshed_text:
        title = f"{title} · {refreshed_text}"

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.42, 0.24, 0.18, 0.16],
        vertical_spacing=0.09,
        subplot_titles=(
            "A) Net Income (USD)  |  Stock Price",
            "B) YoY / 2Y / 3Y Same-Quarter Growth % (incl. forecasted quarter)",
            "C) Growth Acceleration / Deceleration (incl. forecasted quarter)",
            "D) Next Quarter Earnings Prediction",
        ),
        specs=[
            [{"secondary_y": True}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
        ],
    )

    if df is None or df.empty:
        fig.update_layout(
            title=title,
            template="plotly_white",
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="black"),
            height=1080,
        )
        fig.add_annotation(
            text="No 10-Q earnings data available.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14),
        )
        return fig

    series = df.copy()
    series["end"] = pd.to_datetime(series["end"], errors="coerce")
    series = series.dropna(subset=["end"]).sort_values("end")
    if series.empty:
        return fig

    if {"Fiscal Year", "Fiscal Quarter"}.issubset(series.columns):
        qtr_labels = (
            series["Fiscal Year"].astype("Int64").astype(str)
            + " "
            + series["Fiscal Quarter"].astype(str)
        )
    else:
        qtr_labels = (
            series["end"].dt.year.astype(str)
            + " Q"
            + series["end"].dt.quarter.astype(str)
        )
    quarter_tickvals = list(series["end"])
    quarter_ticktext = list(qtr_labels)

    # Top panel — Net Income bars
    fig.add_trace(
        go.Bar(
            x=series["end"],
            y=series["NetIncomeLoss"],
            name="Net Income",
            marker_color="#0f766e",
            opacity=0.85,
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    # Top panel — Stock price line (secondary y)
    if ticker:
        end_dates = pd.to_datetime(series["end"], errors="coerce").dropna()
        if not end_dates.empty:
            price_start = str((end_dates.min() - pd.DateOffset(months=3)).date())
            price_end = str((end_dates.max() + pd.DateOffset(months=3)).date())
            price_df = _fetch_quarterly_prices(ticker, price_start, price_end)
            if not price_df.empty:
                fig.add_trace(
                    go.Scatter(
                        x=price_df["date"],
                        y=price_df["close"],
                        name="Stock Price",
                        mode="lines+markers",
                        line=dict(color="#6366f1", width=2),
                        marker=dict(size=5),
                    ),
                    row=1,
                    col=1,
                    secondary_y=True,
                )
                fig.update_yaxes(
                    title_text="Stock Price (USD)",
                    secondary_y=True,
                    row=1,
                    col=1,
                    tickprefix="$",
                    gridcolor="rgba(0,0,0,0)",
                )

    # Bottom panel — YoY growth bars + MA line
    if "YoYGrowthPct" in series.columns:
        yoy = series["YoYGrowthPct"]
        ni_series = pd.to_numeric(series["NetIncomeLoss"], errors="coerce")
        if {"Fiscal Year", "Fiscal Quarter"}.issubset(series.columns):
            q_year = pd.to_numeric(series["Fiscal Year"], errors="coerce")
            q_fq = pd.to_numeric(
                series["Fiscal Quarter"]
                .astype(str)
                .str.extract(r"([1-4])", expand=False),
                errors="coerce",
            )
        else:
            q_year = series["end"].dt.year
            q_fq = ((series["end"].dt.month - 1) // 3) + 1
        ni_lookup = {
            (int(yr), int(fq)): ni_val
            for yr, fq, ni_val in zip(q_year, q_fq, ni_series)
            if pd.notna(ni_val) and pd.notna(yr) and pd.notna(fq)
        }

        s2y = []
        s3y = []
        for yr, fq, ni_val in zip(q_year, q_fq, ni_series):
            if pd.isna(ni_val) or pd.isna(yr) or pd.isna(fq):
                s2y.append(None)
                s3y.append(None)
                continue

            p2 = ni_lookup.get((int(yr) - 2, int(fq)))
            p3 = ni_lookup.get((int(yr) - 3, int(fq)))

            s2y.append(
                (ni_val - p2) / abs(p2)
                if p2 is not None and pd.notna(p2) and abs(p2) > 0
                else None
            )
            s3y.append(
                (ni_val - p3) / abs(p3)
                if p3 is not None and pd.notna(p3) and abs(p3) > 0
                else None
            )

        bar_colors = [
            "#22c55e" if (v is not None and not pd.isna(v) and v >= 0) else "#ef4444"
            for v in yoy
        ]
        fig.add_trace(
            go.Bar(
                x=series["end"],
                y=yoy,
                name="YoY Growth %",
                marker_color=bar_colors,
                opacity=0.75,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=series["end"],
                y=s2y,
                name="2Y Same-Quarter",
                mode="lines+markers",
                line=dict(color="#6366f1", width=2.5),
                marker=dict(size=5),
                connectgaps=True,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=series["end"],
                y=s3y,
                name="3Y Same-Quarter",
                mode="lines+markers",
                line=dict(color="#14b8a6", width=2.5, dash="dot"),  # teal
                marker=dict(size=5),
                connectgaps=True,
            ),
            row=2,
            col=1,
        )
        ma = yoy.rolling(window=4, min_periods=2).mean()
        fig.add_trace(
            go.Scatter(
                x=series["end"],
                y=ma,
                name="4-Qtr Avg Growth",
                mode="lines",
                line=dict(color="#7c3aed", width=2.5, dash="dot"),  # purple
            ),
            row=2,
            col=1,
        )

        # Third panel — acceleration/deceleration of growth (change in YoY)
        accel_raw = yoy.diff()
        accel_colors = [
            "#16a34a" if (v is not None and not pd.isna(v) and v >= 0) else "#dc2626"
            for v in accel_raw
        ]
        fig.add_trace(
            go.Bar(
                x=series["end"],
                y=accel_raw,
                name="Growth Acceleration",
                marker_color=accel_colors,
                opacity=0.75,
            ),
            row=3,
            col=1,
        )
        accel_ma = accel_raw.rolling(window=4, min_periods=2).mean()
        fig.add_trace(
            go.Scatter(
                x=series["end"],
                y=accel_ma,
                name="4-Qtr Avg Acceleration",
                mode="lines",
                line=dict(color="#7c3aed", width=2.5),
            ),
            row=3,
            col=1,
        )
        fig.add_hline(y=0, line_color="#6b7280", line_width=1, row=3, col=1)

        accel_non_null = accel_raw.dropna()
        avg_accel = float(accel_non_null.mean()) if not accel_non_null.empty else 0.0
        avg_accel_color = "#16a34a" if avg_accel >= 0 else "#dc2626"

    # Always compute bar width for forecast bars (used in 1-A, 1-B, 1-C)
    if len(series) > 1:
        bar_width_days = series["end"].diff().dt.days.median() or 90
    else:
        bar_width_days = 90
    bar_width_ms = int(bar_width_days * 24 * 3600 * 1000)

    diagnostics_prediction = _predict_next_quarter_earnings_diagnostics(series)
    if diagnostics_prediction is not None:
        recent = (
            series.copy()
            .assign(end=pd.to_datetime(series["end"], errors="coerce"))
            .dropna(subset=["end"])
            .sort_values("end")
            .tail(8)
        )
        fig.add_trace(
            go.Scatter(
                x=recent["end"],
                y=recent["NetIncomeLoss"],
                name="Recent Net Income",
                mode="lines+markers",
                line=dict(color="#64748b", width=2),
                marker=dict(size=5),
            ),
            row=4,
            col=1,
        )
        # Gold diagnostics forecast path and dot for 1-D
        fig.add_trace(
            go.Scatter(
                x=[
                    diagnostics_prediction["last_end"],
                    diagnostics_prediction["next_end"],
                ],
                y=[
                    diagnostics_prediction["last_net_income"],
                    diagnostics_prediction["predicted_net_income"],
                ],
                name="Forecast Path",
                mode="lines",
                line=dict(color="#f59e42", width=3, dash="dot"),
                showlegend=False,
                hovertemplate="%{x|%Y Q%q}<br>Forecast Path: %{y:,.0f}<extra></extra>",
            ),
            row=4,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[diagnostics_prediction["next_end"]],
                y=[diagnostics_prediction["predicted_net_income"]],
                name="Forecast",
                mode="markers",
                marker=dict(color="#f59e42", size=13, symbol="circle"),
                hovertemplate="%{x|%Y Q%q}<br>Forecast: %{y:,.0f}<extra></extra>",
            ),
            row=4,
            col=1,
        )

        next_label = (
            f"{diagnostics_prediction['next_end'].year} "
            f"Q{diagnostics_prediction['next_end'].quarter}"
        )
        if diagnostics_prediction["next_end"] not in quarter_tickvals:
            quarter_tickvals.append(diagnostics_prediction["next_end"])
            quarter_ticktext.append(next_label)

        def _fmt_b(v: float) -> str:
            av = abs(v)
            if av >= 1e9:
                return f"${v / 1e9:.2f}B"
            if av >= 1e6:
                return f"${v / 1e6:.2f}M"
            return f"${v:,.0f}"

        q_lookup = series.copy()
        q_lookup["end"] = pd.to_datetime(q_lookup["end"], errors="coerce")
        if "Fiscal Quarter" in q_lookup.columns:
            q_lookup["_qtr"] = pd.to_numeric(
                q_lookup["Fiscal Quarter"]
                .astype(str)
                .str.extract(r"([1-4])", expand=False),
                errors="coerce",
            )
        else:
            q_lookup["_qtr"] = ((q_lookup["end"].dt.month - 1) // 3) + 1

        if "Fiscal Year" in q_lookup.columns:
            q_lookup["_year"] = pd.to_numeric(q_lookup["Fiscal Year"], errors="coerce")
        else:
            q_lookup["_year"] = q_lookup["end"].dt.year

        q_lookup = q_lookup.dropna(subset=["_year", "_qtr", "NetIncomeLoss"])
        ni_lookup = {
            (int(row["_year"]), int(row["_qtr"])): float(row["NetIncomeLoss"])
            for _, row in q_lookup.iterrows()
        }

        forecast_end = pd.to_datetime(
            diagnostics_prediction["next_end"], errors="coerce"
        )
        forecast_val = float(diagnostics_prediction["predicted_net_income"])
        f_year = int(forecast_end.year)
        f_qtr = int(((forecast_end.month - 1) // 3) + 1)
        prev_val = ni_lookup.get((f_year - 1, f_qtr))
        diag_yoy = None
        if prev_val is not None and abs(prev_val) > 1e-9:
            diag_yoy = (forecast_val - prev_val) / abs(prev_val)

        # diag_label removed (no longer used)
        diag_yoy_text = "n/a" if diag_yoy is None else f"{diag_yoy * 100:+.1f}%"

        # Add gold forecast bars to panels B (YoY) and C (acceleration)
        if diag_yoy is not None and pd.notna(diag_yoy):
            # Panel 1-B (YoY) gold bar
            fig.add_trace(
                go.Bar(
                    x=[forecast_end],
                    y=[diag_yoy],
                    name="Forecast YoY",
                    marker_color="#f59e42",  # gold
                    opacity=0.95,
                    width=[bar_width_ms],
                    hovertemplate="%{x|%Y Q%q}<br>Forecast YoY: %{y:.2%}<extra></extra>",
                ),
                row=2,
                col=1,
            )

            # Panel 1-B (YoY) gold extension for 3Y Same-Quarter
            # Compute the last value of s3y and extend to forecast
            s3y_last = None
            if "s3y" in locals() and len(s3y) > 0:
                for val in reversed(s3y):
                    if val is not None and not pd.isna(val):
                        s3y_last = val
                        break
            if s3y_last is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[series["end"].iloc[-1], forecast_end],
                        y=[s3y_last, diag_yoy],
                        name="3Y Same-Quarter Forecast",
                        mode="lines+markers",
                        line=dict(color="#f59e42", width=2.5, dash="dot"),
                        marker=dict(size=7, color="#f59e42"),
                        showlegend=True,
                        hovertemplate="%{x|%Y Q%q}<br>3Y Forecast: %{y:.2%}<extra></extra>",
                    ),
                    row=2,
                    col=1,
                )

            last_yoy = pd.to_numeric(series["YoYGrowthPct"], errors="coerce").dropna()
            latest_yoy = float(last_yoy.iloc[-1]) if not last_yoy.empty else None
            if latest_yoy is not None and pd.notna(latest_yoy):
                forecast_accel = float(diag_yoy - latest_yoy)
                # Panel 1-C (acceleration) gold bar
                fig.add_trace(
                    go.Bar(
                        x=[forecast_end],
                        y=[forecast_accel],
                        name="Forecast Acceleration",
                        marker_color="#f59e42",  # gold
                        opacity=0.95,
                        width=[bar_width_ms],
                        hovertemplate="%{x|%Y Q%q}<br>Forecast Accel: %{y:.2%}<extra></extra>",
                    ),
                    row=3,
                    col=1,
                )

                # Panel 1-C (acceleration) gold extension for 4-Qtr Avg Acceleration line
                accel_ma = (
                    pd.to_numeric(series["YoYGrowthPct"], errors="coerce")
                    .diff()
                    .rolling(window=4, min_periods=2)
                    .mean()
                )
                accel_ma_last = None
                if len(accel_ma) > 0:
                    for val in reversed(accel_ma):
                        if val is not None and not pd.isna(val):
                            accel_ma_last = val
                            break
                if accel_ma_last is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=[series["end"].iloc[-1], forecast_end],
                            y=[accel_ma_last, forecast_accel],
                            name="4-Qtr Avg Accel Forecast",
                            mode="lines+markers",
                            line=dict(color="#f59e42", width=2.5, dash="dot"),
                            marker=dict(size=7, color="#f59e42"),
                            showlegend=True,
                            hovertemplate="%{x|%Y Q%q}<br>4Q Avg Accel Forecast: %{y:.2%}<extra></extra>",
                        ),
                        row=3,
                        col=1,
                    )

        # Panel A (Net Income): add a forecast bar with a distinct color (no annotation)
        # Match the forecast bar width to the median width of the actual bars
        if len(series) > 1:
            bar_width_days = series["end"].diff().dt.days.median() or 90
        else:
            bar_width_days = 90
        bar_width_ms = int(bar_width_days * 24 * 3600 * 1000)
        fig.add_trace(
            go.Bar(
                x=[forecast_end],
                y=[forecast_val],
                name="Forecast Net Income",
                marker_color="#f59e42",  # orange
                opacity=0.95,
                width=[bar_width_ms],
                hovertemplate="%{x|%Y Q%q}<br>Forecast Net Income: %{y:,.0f}<extra></extra>",
            ),
            row=1,
            col=1,
            secondary_y=False,
        )

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        height=1080,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        bargap=0.2,
        margin=dict(r=130, t=160, b=90),
        title=dict(
            text=title,
            y=0.98,
            yanchor="top",
            pad=dict(b=20),
        ),
    )
    fig.update_xaxes(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb")
    fig.update_yaxes(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb")
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.update_yaxes(tickformat=".0%", row=3, col=1)
    fig.update_xaxes(
        tickmode="array",
        tickvals=quarter_tickvals,
        ticktext=quarter_ticktext,
        tickangle=-35,
        tickfont=dict(size=10),
        showticklabels=True,
        row=1,
        col=1,
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=quarter_tickvals,
        ticktext=quarter_ticktext,
        tickangle=-35,
        tickfont=dict(size=10),
        showticklabels=True,
        row=2,
        col=1,
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=quarter_tickvals,
        ticktext=quarter_ticktext,
        tickangle=-35,
        tickfont=dict(size=10),
        showticklabels=True,
        row=3,
        col=1,
    )
    fig.update_xaxes(
        title_text="Fiscal Quarter",
        tickmode="array",
        tickvals=quarter_tickvals,
        ticktext=quarter_ticktext,
        tickangle=-35,
        tickfont=dict(size=10),
        showticklabels=True,
        row=4,
        col=1,
    )
    return fig


def create_earnings_analysis_figure(
    annual_df: pd.DataFrame,
    quarterly_df: pd.DataFrame,
    ticker: str,
    refreshed_text: str = "",
):
    ticker_norm = _normalize_ticker(ticker)
    no_annual = annual_df is None or annual_df.empty
    no_quarterly = quarterly_df is None or quarterly_df.empty

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "A) Are earnings growing?",
            "B) How fast? (Annual YoY Growth %)",
            "C) Is growth improving? (Acceleration)",
            "D) Is it consistent? (Quarterly YoY Distribution)",
        ],
        vertical_spacing=0.16,
        horizontal_spacing=0.10,
    )

    if not no_annual:
        ann = annual_df.copy().sort_values("end")

        # Panel 1: Earnings trend line
        fig.add_trace(
            go.Scatter(
                x=ann["end"],
                y=ann["NetIncomeLoss"],
                name="Net Income",
                mode="lines+markers",
                line=dict(color="#2563eb", width=2),
                marker=dict(size=6),
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        # Panel 2: YoY growth bars (green=positive, red=negative)
        if "EarningsGrowthPct" in ann.columns:
            yoy = ann["EarningsGrowthPct"].fillna(0)
            bar_colors = ["#16a34a" if v >= 0 else "#dc2626" for v in yoy]
            fig.add_trace(
                go.Bar(
                    x=ann["end"],
                    y=yoy,
                    name="YoY Growth %",
                    marker_color=bar_colors,
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

        # Panel 3: Acceleration (change in YoY growth rate, pp)
        if "AccelerationPct" in ann.columns:
            accel = ann["AccelerationPct"].fillna(0)
            accel_colors = ["#16a34a" if v >= 0 else "#dc2626" for v in accel]
            fig.add_trace(
                go.Bar(
                    x=ann["end"],
                    y=accel,
                    name="Acceleration",
                    marker_color=accel_colors,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

    # Panel 4: Quarterly YoY distribution histogram
    if not no_quarterly and "YoYGrowthPct" in quarterly_df.columns:
        yoy_q = quarterly_df["YoYGrowthPct"].dropna()
        if not yoy_q.empty:
            fig.add_trace(
                go.Histogram(
                    x=yoy_q,
                    nbinsx=12,
                    name="YoY % Dist.",
                    marker_color="#7c3aed",
                    showlegend=False,
                ),
                row=2,
                col=2,
            )

    title = f"{ticker_norm} Earnings Growth Analysis"
    if refreshed_text:
        title = f"{title} · {refreshed_text}"

    fig.update_layout(
        title=title,
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        height=650,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb")
    fig.update_yaxes(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb")
    return fig


def create_revenue_vs_earnings_figure(
    annual_df: pd.DataFrame, ticker: str, refreshed_text: str = ""
):
    ticker_norm = _normalize_ticker(ticker)
    fig = go.Figure()

    if annual_df is None or annual_df.empty:
        fig.update_layout(
            title=f"{ticker_norm} Revenue vs Earnings",
            template="plotly_white",
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="black"),
            height=400,
        )
        fig.add_annotation(
            text="No revenue / earnings data available.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14),
        )
        return fig

    ann = annual_df.copy().sort_values("end")

    if "Revenues" in ann.columns:
        fig.add_trace(
            go.Bar(
                x=ann["end"],
                y=ann["Revenues"],
                name="Revenue",
                marker_color="#0369a1",
                opacity=0.85,
            )
        )

    fig.add_trace(
        go.Bar(
            x=ann["end"],
            y=ann["NetIncomeLoss"],
            name="Net Income",
            marker_color="#2563eb",
            opacity=0.85,
        )
    )

    title = f"{ticker_norm} Revenue vs Earnings — Why?"
    if refreshed_text:
        title = f"{title} · {refreshed_text}"

    fig.update_layout(
        title=title,
        barmode="group",
        xaxis_title="Fiscal Year",
        yaxis_title="USD",
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb")
    fig.update_yaxes(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb")
    return fig


def create_10k_trend_diagnostics(annual_df: pd.DataFrame, ticker: str):
    ticker_norm = _normalize_ticker(ticker)
    if annual_df is None or annual_df.empty:
        return html.Div("No 10-K diagnostics available.", className="text-muted")

    ann = annual_df.copy()
    ann["end"] = pd.to_datetime(ann["end"], errors="coerce")
    ann["NetIncomeLoss"] = pd.to_numeric(ann["NetIncomeLoss"], errors="coerce")
    ann = ann.dropna(subset=["end", "NetIncomeLoss"]).sort_values("end")
    if ann.empty:
        return html.Div("No 10-K diagnostics available.", className="text-muted")

    growth = pd.to_numeric(ann.get("EarningsGrowthPct"), errors="coerce")
    accel = pd.to_numeric(ann.get("AccelerationPct", growth.diff()), errors="coerce")

    def _cagr(vals: pd.Series, periods: int) -> float | None:
        clean = pd.to_numeric(vals, errors="coerce").dropna()
        if len(clean) < periods + 1:
            return None
        start = float(clean.iloc[-(periods + 1)])
        end = float(clean.iloc[-1])
        if start <= 0 or end <= 0:
            return None
        return (end / start) ** (1 / periods) - 1

    ni = ann["NetIncomeLoss"]
    cagr_3 = _cagr(ni, 3)
    cagr_5 = _cagr(ni, 5)
    cagr_full = _cagr(ni, max(len(ni) - 1, 1)) if len(ni) > 1 else None

    growth_non_null = growth.dropna()
    accel_non_null = accel.dropna()
    growth_hit = (
        float((growth_non_null > 0).mean()) if not growth_non_null.empty else None
    )
    accel_hit = float((accel_non_null > 0).mean()) if not accel_non_null.empty else None
    growth_vol = float(growth_non_null.std()) if len(growth_non_null) > 1 else None
    avg_yoy_4q = float(growth.tail(4).mean()) if not growth.empty else None
    avg_yoy_all = float(growth.mean()) if not growth.empty else None
    yoy_hit = float((growth > 0).mean()) if not growth.empty else None
    yoy_vol = float(growth.std()) if len(growth) > 1 else None
    avg_accel = float(accel.mean()) if not accel.empty else None
    avg_accel_4q = float(accel.tail(4).mean()) if not accel.empty else None
    latest_2y = None
    avg_2y = None
    latest_3y = None
    avg_3y = None

    running_peak = ni.cummax()
    drawdown = ni / running_peak - 1
    max_drawdown = float(drawdown.min()) if not drawdown.empty else None

    latest_growth = growth_non_null.iloc[-1] if not growth_non_null.empty else None
    latest_accel = accel_non_null.iloc[-1] if not accel_non_null.empty else None
    regime = "n/a"
    if latest_growth is not None and pd.notna(latest_growth):
        if latest_growth >= 0 and latest_accel is not None and pd.notna(latest_accel):
            regime = "Expanding" if latest_accel >= 0 else "Slowing Growth"
        elif latest_growth < 0:
            regime = "Contracting"

    def _fmt_pct(v: float | None) -> str:
        return "n/a" if v is None or pd.isna(v) else f"{v * 100:.1f}%"

    def _regime_with_definition(regime: str) -> str:
        regime_defs = {
            "Expanding": "positive growth with non-negative acceleration",
            "Slowing Growth": "positive growth with negative acceleration",
            "Contracting": "negative growth",
            "n/a": "insufficient data",
        }
        return f"{regime} ({regime_defs.get(regime, 'insufficient data')})"

    return html.Div(
        [
            html.Div(
                f"{ticker_norm} 10-K Trend Diagnostics", className="fw-semibold mb-2"
            ),
            html.Div(
                "All multi-year rates compare only the same fiscal quarter (e.g. Q1 vs Q1). "
                "Acceleration is the quarter-to-quarter change in YoY growth.",
                className="text-muted small mb-2",
            ),
            # 1-year (YoY)
            html.Div(
                "1-Year (YoY) Same-Quarter Growth",
                className="fw-semibold mt-2 mb-1 small text-secondary",
            ),
            html.Ul(
                [
                    html.Li(f"Latest YoY growth: {_fmt_pct(latest_growth)}"),
                    html.Li(f"4-quarter average YoY: {_fmt_pct(avg_yoy_4q)}"),
                    html.Li(f"Full-history average YoY: {_fmt_pct(avg_yoy_all)}"),
                    html.Li(f"Positive growth hit rate: {_fmt_pct(yoy_hit)}"),
                    html.Li(f"YoY volatility (std dev): {_fmt_pct(yoy_vol)}"),
                ],
                className="mb-0",
            ),
            # 2-year same-quarter
            html.Div(
                "2-Year Same-Quarter Growth",
                className="fw-semibold mt-2 mb-1 small text-secondary",
            ),
            html.Ul(
                [
                    html.Li(f"Latest 2Y: {_fmt_pct(latest_2y)}"),
                    html.Li(f"Average 2Y (all quarters): {_fmt_pct(avg_2y)}"),
                ],
                className="mb-0",
            ),
            # 3-year same-quarter
            html.Div(
                "3-Year Same-Quarter Growth",
                className="fw-semibold mt-2 mb-1 small text-secondary",
            ),
            html.Ul(
                [
                    html.Li(f"Latest 3Y: {_fmt_pct(latest_3y)}"),
                    html.Li(f"Average 3Y (all quarters): {_fmt_pct(avg_3y)}"),
                ],
                className="mb-0",
            ),
            # Momentum & risk
            html.Div(
                "Momentum & Risk",
                className="fw-semibold mt-2 mb-1 small text-secondary",
            ),
            html.Ul(
                [
                    html.Li(
                        f"Average acceleration (pp/qtr): {'n/a' if avg_accel is None or pd.isna(avg_accel) else f'{avg_accel * 100:+.1f}'}"
                    ),
                    html.Li(
                        f"4Q average acceleration (pp/qtr): {'n/a' if avg_accel_4q is None or pd.isna(avg_accel_4q) else f'{avg_accel_4q * 100:+.1f}'}"
                    ),
                    html.Li(
                        f"Max earnings drawdown from peak: {_fmt_pct(max_drawdown)}"
                    ),
                    html.Li(f"Latest regime: {regime}"),
                ],
                className="mb-0",
            ),
        ]
    )


def create_10q_diagnostics_figure(
    quarterly_df: pd.DataFrame, ticker: str, refreshed_text: str = ""
) -> go.Figure:
    ticker_norm = _normalize_ticker(ticker)
    title = f"{ticker_norm} 10-Q Growth Rate Comparison (Same-Quarter)"
    if refreshed_text:
        title = f"{title} · {refreshed_text}"

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.68, 0.32],
        vertical_spacing=0.16,
        subplot_titles=(
            "A) YoY / 2Y / 3Y Same-Quarter Growth %",
            "B) Growth Acceleration (pp/qtr)",
        ),
    )

    if quarterly_df is None or quarterly_df.empty:
        fig.update_layout(
            title=dict(text=title, y=0.99, yanchor="top", pad=dict(b=32)),
            template="plotly_white",
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="black"),
            height=520,
            margin=dict(r=20, t=180, b=60),
        )
        fig.add_annotation(
            text="No 10-Q data available.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14),
        )
        return fig

    q = quarterly_df.copy()
    q["end"] = pd.to_datetime(q["end"], errors="coerce")
    q["NetIncomeLoss"] = pd.to_numeric(q["NetIncomeLoss"], errors="coerce")
    q["YoYGrowthPct"] = pd.to_numeric(q.get("YoYGrowthPct"), errors="coerce")
    q = (
        q.dropna(subset=["end", "NetIncomeLoss"])
        .sort_values("end")
        .reset_index(drop=True)
    )

    if q.empty:
        fig.update_layout(title=dict(text=title), height=520)
        return fig

    # Quarter labels
    if {"Fiscal Year", "Fiscal Quarter"}.issubset(q.columns):
        q_labels = (
            q["Fiscal Year"].astype("Int64").astype(str)
            + " "
            + q["Fiscal Quarter"].astype(str)
        )
    else:
        q_labels = q["end"].dt.year.astype(str) + " Q" + q["end"].dt.quarter.astype(str)

    # Build same-quarter 2Y / 3Y series aligned to q index
    q["_year"] = q["end"].dt.year
    q["_fq"] = ((q["end"].dt.month - 1) // 3) + 1
    ni_lookup: dict = {
        (int(r["_year"]), int(r["_fq"])): r["NetIncomeLoss"] for _, r in q.iterrows()
    }

    s2y, s3y = [], []
    for _, row in q.iterrows():
        yr, fq, ni_val = int(row["_year"]), int(row["_fq"]), row["NetIncomeLoss"]
        p2 = ni_lookup.get((yr - 2, fq))
        p3 = ni_lookup.get((yr - 3, fq))
        s2y.append(
            (ni_val - p2) / abs(p2)
            if p2 is not None and pd.notna(p2) and abs(p2) > 0
            else None
        )
        s3y.append(
            (ni_val - p3) / abs(p3)
            if p3 is not None and pd.notna(p3) and abs(p3) > 0
            else None
        )

    yoy = q["YoYGrowthPct"]
    accel = yoy.diff()

    # ── Row 1: YoY bars + 2Y / 3Y lines ───────────────────────────────────
    bar_colors = [
        "#22c55e" if (v is not None and pd.notna(v) and v >= 0) else "#ef4444"
        for v in yoy
    ]
    fig.add_trace(
        go.Bar(
            x=q["end"],
            y=yoy,
            name="YoY (1Y)",
            marker_color=bar_colors,
            opacity=0.75,
            customdata=list(q_labels),
            hovertemplate="%{customdata}<br>YoY: %{y:.1%}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=q["end"],
            y=s2y,
            name="2Y Same-Quarter",
            mode="lines+markers",
            line=dict(color="#6366f1", width=2.5),
            marker=dict(size=5),
            customdata=list(q_labels),
            hovertemplate="%{customdata}<br>2Y: %{y:.1%}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=q["end"],
            y=s3y,
            name="3Y Same-Quarter",
            mode="lines+markers",
            line=dict(color="#f59e0b", width=2.5, dash="dot"),
            marker=dict(size=5),
            customdata=list(q_labels),
            hovertemplate="%{customdata}<br>3Y: %{y:.1%}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=0, line_color="#6b7280", line_width=1, row=1, col=1)

    # ── Row 2: Acceleration ────────────────────────────────────────────────
    accel_colors = [
        "#16a34a" if (v is not None and pd.notna(v) and v >= 0) else "#dc2626"
        for v in accel
    ]
    fig.add_trace(
        go.Bar(
            x=q["end"],
            y=accel,
            name="Growth Acceleration",
            marker_color=accel_colors,
            opacity=0.75,
            customdata=list(q_labels),
            hovertemplate="%{customdata}<br>Accel: %{y:.1%}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=q["end"],
            y=accel.rolling(window=4, min_periods=2).mean(),
            name="4-Qtr Avg Accel",
            mode="lines",
            line=dict(color="#7c3aed", width=2),
            showlegend=True,
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0, line_color="#6b7280", line_width=1, row=2, col=1)

    tick_vals = list(q["end"])
    tick_text = list(q_labels)

    fig.update_layout(
        title=dict(text=title, y=0.99, yanchor="top", pad=dict(b=32)),
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        bargap=0.2,
        margin=dict(r=20, t=180, b=60),
    )
    fig.update_xaxes(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb")
    fig.update_yaxes(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb")
    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    for r in (1, 2):
        fig.update_xaxes(
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_text,
            tickangle=-35,
            tickfont=dict(size=10),
            showticklabels=(r == 2),
            row=r,
            col=1,
        )
    return fig


def create_10q_per_quarter_cagr(quarterly_df: pd.DataFrame, ticker: str):
    ticker_norm = _normalize_ticker(ticker)
    if quarterly_df is None or quarterly_df.empty:
        return html.Div(
            "No 10-Q per-quarter CAGR data available.", className="text-muted"
        )

    q = quarterly_df.copy()
    q["end"] = pd.to_datetime(q["end"], errors="coerce")
    q["NetIncomeLoss"] = pd.to_numeric(q["NetIncomeLoss"], errors="coerce")
    q = (
        q.dropna(subset=["end", "NetIncomeLoss"])
        .sort_values("end")
        .reset_index(drop=True)
    )
    if q.empty:
        return html.Div(
            "No 10-Q per-quarter CAGR data available.", className="text-muted"
        )

    if "Fiscal Quarter" in q.columns:
        q["_fq"] = (
            q["Fiscal Quarter"]
            .astype(str)
            .str.extract(r"([1-4])", expand=False)
            .astype(float)
        )
    else:
        q["_fq"] = ((q["end"].dt.month - 1) // 3) + 1

    if "Fiscal Year" in q.columns:
        q["_fy"] = pd.to_numeric(q["Fiscal Year"], errors="coerce")
    else:
        q["_fy"] = q["end"].dt.year

    def _cagr(ni_series: pd.Series, periods: int) -> float | None:
        clean = ni_series.dropna()
        if len(clean) < periods + 1:
            return None
        start = float(clean.iloc[-(periods + 1)])
        end = float(clean.iloc[-1])
        if start <= 0 or end <= 0:
            return None
        return (end / start) ** (1 / periods) - 1

    def _fmt_pct(v: float | None) -> str:
        return "n/a" if v is None or pd.isna(v) else f"{v * 100:.1f}%"

    def _regime_with_definition(regime: str) -> str:
        regime_defs = {
            "Expanding": "positive growth with non-negative acceleration",
            "Slowing Growth": "positive growth with negative acceleration",
            "Contracting": "negative growth",
            "n/a": "insufficient data",
        }
        return f"{regime} ({regime_defs.get(regime, 'insufficient data')})"

    def _quarter_metrics(fq_num: int) -> dict:
        label = f"Q{fq_num}"
        sub = q[q["_fq"] == fq_num].sort_values("_fy").reset_index(drop=True)
        if sub.empty or len(sub) < 2:
            return {"label": label, "has_data": False}

        ni = sub["NetIncomeLoss"]
        cagr_1 = _cagr(ni, 1)
        cagr_3 = _cagr(ni, 3)
        cagr_5 = _cagr(ni, 5)
        cagr_full = _cagr(ni, max(len(ni) - 1, 1)) if len(ni) > 1 else None

        yoy = ni.pct_change()
        accel = yoy.diff()
        yoy_nn = yoy.dropna()
        accel_nn = accel.dropna()
        growth_hit = float((yoy_nn > 0).mean()) if not yoy_nn.empty else None
        accel_hit = float((accel_nn > 0).mean()) if not accel_nn.empty else None
        growth_vol = float(yoy_nn.std()) if len(yoy_nn) > 1 else None

        running_peak = ni.cummax()
        drawdown = ni / running_peak - 1
        max_drawdown = float(drawdown.min()) if not drawdown.empty else None

        latest_growth = float(yoy_nn.iloc[-1]) if not yoy_nn.empty else None
        latest_accel = float(accel_nn.iloc[-1]) if not accel_nn.empty else None
        prev_accel = float(accel_nn.iloc[-2]) if len(accel_nn) >= 2 else None
        regime = "n/a"
        if latest_growth is not None and pd.notna(latest_growth):
            if (
                latest_growth >= 0
                and latest_accel is not None
                and pd.notna(latest_accel)
            ):
                regime = "Expanding" if latest_accel >= 0 else "Slowing Growth"
            elif latest_growth < 0:
                regime = "Contracting"

        current_accel_pp = None
        delta_accel_pp = None
        if latest_accel is None or pd.isna(latest_accel):
            accel_status = "n/a"
            accel_color = "black"
        elif abs(latest_accel) < 0.005:  # within ±0.5 pp — stable
            accel_status = "Stable"
            accel_color = "#6b7280"
            current_accel_pp = latest_accel * 100
        elif latest_accel > 0:
            accel_status = "Accelerating"
            accel_color = "#16a34a"
            current_accel_pp = latest_accel * 100
        else:
            accel_status = "Decelerating"
            accel_color = "#dc2626"
            current_accel_pp = latest_accel * 100

        if latest_accel is not None and prev_accel is not None and pd.notna(prev_accel):
            delta_accel_pp = (latest_accel - prev_accel) * 100

        return {
            "label": label,
            "has_data": True,
            "cagr_1": cagr_1,
            "cagr_3": cagr_3,
            "cagr_5": cagr_5,
            "cagr_full": cagr_full,
            "growth_hit": growth_hit,
            "accel_hit": accel_hit,
            "growth_vol": growth_vol,
            "max_drawdown": max_drawdown,
            "regime": regime,
            "accel_status": accel_status,
            "accel_color": accel_color,
            "current_accel_pp": current_accel_pp,
            "delta_accel_pp": delta_accel_pp,
        }

    def _quarter_block(metrics: dict) -> html.Div:
        label = metrics["label"]
        if not metrics.get("has_data"):
            return html.Div(
                [
                    html.Div(label, className="fw-semibold mt-3 mb-1"),
                    html.Span("Not enough data.", className="text-muted small"),
                ]
            )

        return html.Div(
            [
                html.Div(label, className="fw-semibold mt-3 mb-1 text-primary"),
                html.Ul(
                    [
                        html.Li(
                            f"CAGR: 1Y {_fmt_pct(metrics.get('cagr_1'))} | 3Y {_fmt_pct(metrics.get('cagr_3'))} | 5Y {_fmt_pct(metrics.get('cagr_5'))} | Full {_fmt_pct(metrics.get('cagr_full'))}"
                        ),
                        html.Li(
                            [
                                html.Span("Current same Quarter YOY acceleration: "),
                                html.Span(
                                    (
                                        f"{metrics.get('accel_status')} "
                                        f"({metrics.get('current_accel_pp'):+.2f} pp)"
                                        if metrics.get("current_accel_pp") is not None
                                        else "n/a"
                                    ),
                                    style={
                                        "color": metrics.get("accel_color"),
                                        "fontWeight": "600",
                                    },
                                ),
                            ]
                        ),
                        html.Li(
                            (
                                f"Change vs previous same quarter YOY acceleration pp: {metrics.get('delta_accel_pp'):+.2f}"
                                if metrics.get("delta_accel_pp") is not None
                                else "Change vs previous same quarter YOY acceleration pp: n/a"
                            )
                        ),
                        html.Li(
                            f"Consistency: Positive growth {_fmt_pct(metrics.get('growth_hit'))} | Positive acceleration {_fmt_pct(metrics.get('accel_hit'))}"
                        ),
                        html.Li(
                            f"Growth volatility (std dev): {_fmt_pct(metrics.get('growth_vol'))}"
                        ),
                        html.Li(
                            f"Max earnings drawdown from peak: {_fmt_pct(metrics.get('max_drawdown'))}"
                        ),
                        html.Li(
                            f"Latest regime: {_regime_with_definition(metrics.get('regime', 'n/a'))}"
                        ),
                    ],
                    className="mb-0 small",
                ),
            ]
        )

    def _quarter_findings_block(metrics: dict) -> html.Div:
        label = metrics["label"]
        if not metrics.get("has_data"):
            return html.Div(
                [
                    html.Div(f"{label} CAGR Analysis", className="fw-semibold mb-1"),
                    html.Div(
                        "Not enough filings yet to compute reliable trend diagnostics.",
                        className="small text-muted",
                    ),
                ],
                className="border rounded p-3 h-100 bg-light",
            )

        accel_status = metrics.get("accel_status", "n/a")
        accel_color = metrics.get("accel_color", "black")
        current_accel_pp = metrics.get("current_accel_pp")
        delta_accel_pp = metrics.get("delta_accel_pp")

        if current_accel_pp is not None:
            accel_explain = (
                f"The YoY growth rate for {label} changed by {current_accel_pp:+.2f} pp "
                f"compared to the prior {label}. This means the growth rate itself is "
                f"{accel_status.lower()} — earnings are growing {'faster' if current_accel_pp > 0 else 'slower'} "
                f"than they were one year ago."
            )
        else:
            accel_explain = f"Not enough data to compute acceleration for {label}."

        if delta_accel_pp is not None:
            delta_explain = (
                f"The acceleration itself changed by {delta_accel_pp:+.2f} pp vs the prior period, "
                f"meaning momentum is {'strengthening' if delta_accel_pp > 0 else 'fading'}."
            )
        else:
            delta_explain = None

        cagr_1 = metrics.get("cagr_1")
        cagr_3 = metrics.get("cagr_3")
        cagr_5 = metrics.get("cagr_5")
        cagr_full = metrics.get("cagr_full")

        # 1Y vs 3Y: near-term momentum
        if cagr_1 is not None and cagr_3 is not None:
            if cagr_1 > cagr_3 + 0.005:
                recent_dir = "increasing"
                recent_color = "#16a34a"
                recent_note = f"The 1-year CAGR ({_fmt_pct(cagr_1)}) is above the 3-year CAGR ({_fmt_pct(cagr_3)}), meaning growth has picked up in the most recent year."
            elif cagr_1 < cagr_3 - 0.005:
                recent_dir = "decreasing"
                recent_color = "#dc2626"
                recent_note = f"The 1-year CAGR ({_fmt_pct(cagr_1)}) is below the 3-year CAGR ({_fmt_pct(cagr_3)}), meaning growth has cooled in the most recent year."
            else:
                recent_dir = "stable"
                recent_color = "#6b7280"
                recent_note = f"The 1-year CAGR ({_fmt_pct(cagr_1)}) is roughly in line with the 3-year CAGR ({_fmt_pct(cagr_3)}), indicating a stable near-term pace."
        else:
            recent_dir = "n/a"
            recent_color = "black"
            recent_note = "Not enough data to compare the 1-year and 3-year CAGR."

        # 3Y vs Full: medium-term trend
        if cagr_3 is not None and cagr_full is not None:
            if cagr_3 > cagr_full + 0.005:
                trend_dir = "increasing"
                trend_color = "#16a34a"
                trend_note = f"The 3-year CAGR ({_fmt_pct(cagr_3)}) is above the full-history CAGR ({_fmt_pct(cagr_full)}), meaning recent growth is outpacing the long-run average."
            elif cagr_3 < cagr_full - 0.005:
                trend_dir = "decreasing"
                trend_color = "#dc2626"
                trend_note = f"The 3-year CAGR ({_fmt_pct(cagr_3)}) is below the full-history CAGR ({_fmt_pct(cagr_full)}), meaning recent growth is lagging the long-run average."
            else:
                trend_dir = "stable"
                trend_color = "#6b7280"
                trend_note = f"The 3-year CAGR ({_fmt_pct(cagr_3)}) is roughly in line with the full-history CAGR ({_fmt_pct(cagr_full)}), indicating a stable long-run pace."
        else:
            trend_dir = "n/a"
            trend_color = "black"
            trend_note = (
                "Not enough data to compare short-term and long-run CAGR trends."
            )

        if cagr_3 is not None and cagr_5 is not None:
            if cagr_3 > cagr_5 + 0.005:
                mid_note = f"Growth has picked up in the last 3 years vs the 5-year window ({_fmt_pct(cagr_3)} vs {_fmt_pct(cagr_5)})."
            elif cagr_3 < cagr_5 - 0.005:
                mid_note = f"Growth has slowed in the last 3 years vs the 5-year window ({_fmt_pct(cagr_3)} vs {_fmt_pct(cagr_5)})."
            else:
                mid_note = f"Growth has been consistent between the 3-year and 5-year windows ({_fmt_pct(cagr_3)} vs {_fmt_pct(cagr_5)})."
        else:
            mid_note = "Insufficient data to compare 3-year and 5-year windows."

        return html.Div(
            [
                html.Div(f"{label} CAGR Analysis", className="fw-semibold mb-1"),
                html.Ul(
                    [
                        html.Li(
                            [
                                html.Span("Near-term (1Y vs 3Y): "),
                                html.Span(
                                    recent_dir.capitalize(),
                                    style={
                                        "color": recent_color,
                                        "fontWeight": "600",
                                    },
                                ),
                            ]
                        ),
                        html.Li(recent_note),
                        html.Li(
                            [
                                html.Span("Medium-term (3Y vs Full): "),
                                html.Span(
                                    trend_dir.capitalize(),
                                    style={
                                        "color": trend_color,
                                        "fontWeight": "600",
                                    },
                                ),
                            ]
                        ),
                        html.Li(trend_note),
                        html.Li(mid_note),
                        html.Li(
                            f"Full-history CAGR of {_fmt_pct(cagr_full)} reflects the compound annual growth rate across all available quarters for {label}."
                        ),
                    ],
                    className="small mb-0",
                ),
                html.Hr(className="my-2"),
                html.Div(f"{label} Acceleration", className="fw-semibold mb-1"),
                html.Ul(
                    [
                        html.Li(
                            [
                                html.Span("Current same Quarter YOY acceleration: "),
                                html.Span(
                                    accel_status,
                                    style={
                                        "color": accel_color,
                                        "fontWeight": "600",
                                    },
                                ),
                                html.Span(
                                    f" ({current_accel_pp:+.2f} pp)"
                                    if current_accel_pp is not None
                                    else ""
                                ),
                            ]
                        ),
                        html.Li(accel_explain),
                    ]
                    + ([html.Li(delta_explain)] if delta_explain is not None else []),
                    className="small mb-0",
                ),
            ],
            className="border rounded p-3 h-100 bg-light",
        )

    quarter_metrics = [_quarter_metrics(i) for i in range(1, 5)]

    return html.Div(
        [
            html.Div(
                f"{ticker_norm} 10-Q Per-Quarter CAGR Diagnostics",
                className="fw-semibold mb-1",
            ),
            html.Div(
                "CAGR (Compound Annual Growth Rate) is the smoothed annual growth from start to end period: "
                "CAGR = (Ending / Beginning)^(1/Years) \u2212 1. "
                "Each fiscal quarter is analysed independently against the same quarter in prior years.",
                className="text-muted small mb-2",
            ),
            html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(_quarter_block(metrics), xs=12, lg=7),
                            dbc.Col(_quarter_findings_block(metrics), xs=12, lg=5),
                        ],
                        className="g-3 align-items-stretch",
                    )
                    for metrics in quarter_metrics
                ],
                className="d-grid gap-2",
            ),
        ]
    )


def create_10q_four_quarter_avg_cagr(quarterly_df: pd.DataFrame, ticker: str):
    ticker_norm = _normalize_ticker(ticker)
    if quarterly_df is None or quarterly_df.empty:
        return html.Div(
            "No 10-Q 4-quarter average diagnostics available.",
            className="text-muted",
        )

    q = quarterly_df.copy()
    q["end"] = pd.to_datetime(q["end"], errors="coerce")
    q["NetIncomeLoss"] = pd.to_numeric(q["NetIncomeLoss"], errors="coerce")
    q = (
        q.dropna(subset=["end", "NetIncomeLoss"])
        .sort_values("end")
        .reset_index(drop=True)
    )
    if q.empty or len(q) < 4:
        return html.Div(
            "No 10-Q 4-quarter average diagnostics available.",
            className="text-muted",
        )

    q["FourQuarterAvgNI"] = q["NetIncomeLoss"].rolling(window=4, min_periods=4).mean()
    avg4 = q.dropna(subset=["FourQuarterAvgNI"]).reset_index(drop=True)
    if avg4.empty or len(avg4) < 2:
        return html.Div(
            "No 10-Q 4-quarter average diagnostics available.",
            className="text-muted",
        )

    def _cagr(vals: pd.Series, years: int) -> float | None:
        clean = vals.dropna()
        steps = years * 4
        if len(clean) < steps + 1:
            return None
        start = float(clean.iloc[-(steps + 1)])
        end = float(clean.iloc[-1])
        if start <= 0 or end <= 0:
            return None
        return (end / start) ** (1 / years) - 1

    def _full_cagr(vals: pd.Series) -> float | None:
        clean = vals.dropna()
        if len(clean) < 2:
            return None
        start = float(clean.iloc[0])
        end = float(clean.iloc[-1])
        years = (len(clean) - 1) / 4
        if start <= 0 or end <= 0 or years <= 0:
            return None
        return (end / start) ** (1 / years) - 1

    def _fmt_pct(v: float | None) -> str:
        return "n/a" if v is None or pd.isna(v) else f"{v * 100:.1f}%"

    def _fmt_b(v: float | None) -> str:
        if v is None or pd.isna(v):
            return "n/a"
        av = abs(v)
        if av >= 1e9:
            return f"${v / 1e9:.2f}B"
        if av >= 1e6:
            return f"${v / 1e6:.2f}M"
        return f"${v:,.0f}"

    def _regime_with_definition(regime: str) -> str:
        regime_defs = {
            "Expanding": "positive growth with non-negative acceleration",
            "Slowing Growth": "positive growth with negative acceleration",
            "Contracting": "negative growth",
            "n/a": "insufficient data",
        }
        return f"{regime} ({regime_defs.get(regime, 'insufficient data')})"

    ni = avg4["FourQuarterAvgNI"]
    cagr_1 = _cagr(ni, 1)
    cagr_3 = _cagr(ni, 3)
    cagr_5 = _cagr(ni, 5)
    cagr_full = _full_cagr(ni)

    yoy = ni.pct_change(4)
    accel = yoy.diff()
    yoy_nn = yoy.dropna()
    accel_nn = accel.dropna()
    growth_hit = float((yoy_nn > 0).mean()) if not yoy_nn.empty else None
    accel_hit = float((accel_nn > 0).mean()) if not accel_nn.empty else None
    growth_vol = float(yoy_nn.std()) if len(yoy_nn) > 1 else None
    running_peak = ni.cummax()
    drawdown = ni / running_peak - 1
    max_drawdown = float(drawdown.min()) if not drawdown.empty else None

    latest_growth = float(yoy_nn.iloc[-1]) if not yoy_nn.empty else None
    latest_accel = float(accel_nn.iloc[-1]) if not accel_nn.empty else None
    prev_accel = float(accel_nn.iloc[-2]) if len(accel_nn) >= 2 else None

    regime = "n/a"
    if latest_growth is not None and pd.notna(latest_growth):
        if latest_growth >= 0 and latest_accel is not None and pd.notna(latest_accel):
            regime = "Expanding" if latest_accel >= 0 else "Slowing Growth"
        elif latest_growth < 0:
            regime = "Contracting"

    current_accel_pp = None
    delta_accel_pp = None
    if latest_accel is None or pd.isna(latest_accel):
        accel_status = "n/a"
        accel_color = "black"
    elif abs(latest_accel) < 0.005:
        accel_status = "Stable"
        accel_color = "#6b7280"
        current_accel_pp = latest_accel * 100
    elif latest_accel > 0:
        accel_status = "Accelerating"
        accel_color = "#16a34a"
        current_accel_pp = latest_accel * 100
    else:
        accel_status = "Decelerating"
        accel_color = "#dc2626"
        current_accel_pp = latest_accel * 100

    if latest_accel is not None and prev_accel is not None and pd.notna(prev_accel):
        delta_accel_pp = (latest_accel - prev_accel) * 100

    if cagr_1 is not None and cagr_3 is not None:
        if cagr_1 > cagr_3 + 0.005:
            recent_dir = "increasing"
            recent_color = "#16a34a"
            recent_note = f"The 1-year CAGR ({_fmt_pct(cagr_1)}) is above the 3-year CAGR ({_fmt_pct(cagr_3)}), meaning the trailing 4-quarter average has picked up recently."
        elif cagr_1 < cagr_3 - 0.005:
            recent_dir = "decreasing"
            recent_color = "#dc2626"
            recent_note = f"The 1-year CAGR ({_fmt_pct(cagr_1)}) is below the 3-year CAGR ({_fmt_pct(cagr_3)}), meaning the trailing 4-quarter average has cooled recently."
        else:
            recent_dir = "stable"
            recent_color = "#6b7280"
            recent_note = f"The 1-year CAGR ({_fmt_pct(cagr_1)}) is roughly in line with the 3-year CAGR ({_fmt_pct(cagr_3)}), indicating a stable trailing 4-quarter average pace."
    else:
        recent_dir = "n/a"
        recent_color = "black"
        recent_note = "Not enough data to compare the 1-year and 3-year CAGR on the trailing 4-quarter average."

    if cagr_3 is not None and cagr_full is not None:
        if cagr_3 > cagr_full + 0.005:
            trend_dir = "increasing"
            trend_color = "#16a34a"
            trend_note = f"The 3-year CAGR ({_fmt_pct(cagr_3)}) is above the full-history CAGR ({_fmt_pct(cagr_full)}), meaning recent trailing 4-quarter average growth is outpacing the long-run average."
        elif cagr_3 < cagr_full - 0.005:
            trend_dir = "decreasing"
            trend_color = "#dc2626"
            trend_note = f"The 3-year CAGR ({_fmt_pct(cagr_3)}) is below the full-history CAGR ({_fmt_pct(cagr_full)}), meaning recent trailing 4-quarter average growth is lagging the long-run average."
        else:
            trend_dir = "stable"
            trend_color = "#6b7280"
            trend_note = f"The 3-year CAGR ({_fmt_pct(cagr_3)}) is roughly in line with the full-history CAGR ({_fmt_pct(cagr_full)}), indicating a stable long-run trailing-average pace."
    else:
        trend_dir = "n/a"
        trend_color = "black"
        trend_note = "Not enough data to compare medium-term and long-run trailing 4-quarter average trends."

    if cagr_3 is not None and cagr_5 is not None:
        if cagr_3 > cagr_5 + 0.005:
            mid_note = f"The trailing 4-quarter average has picked up in the last 3 years vs the 5-year window ({_fmt_pct(cagr_3)} vs {_fmt_pct(cagr_5)})."
        elif cagr_3 < cagr_5 - 0.005:
            mid_note = f"The trailing 4-quarter average has slowed in the last 3 years vs the 5-year window ({_fmt_pct(cagr_3)} vs {_fmt_pct(cagr_5)})."
        else:
            mid_note = f"The trailing 4-quarter average has been consistent between the 3-year and 5-year windows ({_fmt_pct(cagr_3)} vs {_fmt_pct(cagr_5)})."
    else:
        mid_note = (
            "Insufficient data to compare 3-year and 5-year trailing-average windows."
        )

    if current_accel_pp is not None:
        accel_explain = f"The 4-quarter average YoY growth rate changed by {current_accel_pp:+.2f} pp versus the prior quarter, so the smoothed growth rate is {accel_status.lower()}."
    else:
        accel_explain = "Not enough data to compute 4-quarter average acceleration."

    if delta_accel_pp is not None:
        delta_explain = f"The acceleration itself changed by {delta_accel_pp:+.2f} pp vs the prior period, meaning smoothed momentum is {'strengthening' if delta_accel_pp > 0 else 'fading'}."
    else:
        delta_explain = None

    return html.Div(
        [
            html.Div(
                f"{ticker_norm} 10-Q 4-Quarter Average CAGR Diagnostics",
                className="fw-semibold mb-1",
            ),
            html.Div(
                "This view smooths quarterly seasonality by using a rolling average of the last 4 quarters of net income, then applies the same CAGR and acceleration diagnostics.",
                className="text-muted small mb-2",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.Div(
                                    "4-Quarter Average Diagnostics",
                                    className="fw-semibold mt-1 mb-1 text-primary",
                                ),
                                html.Ul(
                                    [
                                        html.Li(
                                            f"Current 4-quarter average earnings: {_fmt_b(float(ni.iloc[-1]))}"
                                        ),
                                        html.Li(
                                            f"CAGR: 1Y {_fmt_pct(cagr_1)} | 3Y {_fmt_pct(cagr_3)} | 5Y {_fmt_pct(cagr_5)} | Full {_fmt_pct(cagr_full)}"
                                        ),
                                        html.Li(
                                            [
                                                html.Span(
                                                    "Current 4-quarter average YOY acceleration: "
                                                ),
                                                html.Span(
                                                    (
                                                        f"{accel_status} ({current_accel_pp:+.2f} pp)"
                                                        if current_accel_pp is not None
                                                        else "n/a"
                                                    ),
                                                    style={
                                                        "color": accel_color,
                                                        "fontWeight": "600",
                                                    },
                                                ),
                                            ]
                                        ),
                                        html.Li(
                                            (
                                                f"Change vs previous 4-quarter average YOY acceleration pp: {delta_accel_pp:+.2f}"
                                                if delta_accel_pp is not None
                                                else "Change vs previous 4-quarter average YOY acceleration pp: n/a"
                                            )
                                        ),
                                        html.Li(
                                            f"Consistency: Positive growth {_fmt_pct(growth_hit)} | Positive acceleration {_fmt_pct(accel_hit)}"
                                        ),
                                        html.Li(
                                            f"Growth volatility (std dev): {_fmt_pct(growth_vol)}"
                                        ),
                                        html.Li(
                                            f"Max earnings drawdown from peak: {_fmt_pct(max_drawdown)}"
                                        ),
                                        html.Li(
                                            f"Latest regime: {_regime_with_definition(regime)}"
                                        ),
                                    ],
                                    className="mb-0 small",
                                ),
                            ]
                        ),
                        xs=12,
                        lg=7,
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.Div(
                                    "4-Quarter Average CAGR Analysis",
                                    className="fw-semibold mb-1",
                                ),
                                html.Ul(
                                    [
                                        html.Li(
                                            [
                                                html.Span("Near-term (1Y vs 3Y): "),
                                                html.Span(
                                                    recent_dir.capitalize(),
                                                    style={
                                                        "color": recent_color,
                                                        "fontWeight": "600",
                                                    },
                                                ),
                                            ]
                                        ),
                                        html.Li(recent_note),
                                        html.Li(
                                            [
                                                html.Span("Medium-term (3Y vs Full): "),
                                                html.Span(
                                                    trend_dir.capitalize(),
                                                    style={
                                                        "color": trend_color,
                                                        "fontWeight": "600",
                                                    },
                                                ),
                                            ]
                                        ),
                                        html.Li(trend_note),
                                        html.Li(mid_note),
                                        html.Li(
                                            f"Full-history CAGR of {_fmt_pct(cagr_full)} reflects the compound annual growth rate across the smoothed trailing 4-quarter average series."
                                        ),
                                    ],
                                    className="small mb-0",
                                ),
                                html.Hr(className="my-2"),
                                html.Div(
                                    "4-Quarter Average Acceleration",
                                    className="fw-semibold mb-1",
                                ),
                                html.Ul(
                                    [
                                        html.Li(
                                            [
                                                html.Span(
                                                    "Current 4-quarter average YOY acceleration: "
                                                ),
                                                html.Span(
                                                    accel_status,
                                                    style={
                                                        "color": accel_color,
                                                        "fontWeight": "600",
                                                    },
                                                ),
                                                html.Span(
                                                    f" ({current_accel_pp:+.2f} pp)"
                                                    if current_accel_pp is not None
                                                    else ""
                                                ),
                                            ]
                                        ),
                                        html.Li(accel_explain),
                                    ]
                                    + (
                                        [html.Li(delta_explain)]
                                        if delta_explain is not None
                                        else []
                                    ),
                                    className="small mb-0",
                                ),
                            ],
                            className="border rounded p-3 h-100 bg-light",
                        ),
                        xs=12,
                        lg=5,
                    ),
                ],
                className="g-3 align-items-stretch",
            ),
        ]
    )


def create_10q_trend_diagnostics(quarterly_df: pd.DataFrame, ticker: str):
    ticker_norm = _normalize_ticker(ticker)
    if quarterly_df is None or quarterly_df.empty:
        return html.Div("No 10-Q diagnostics available.", className="text-muted")

    q = quarterly_df.copy()
    q["end"] = pd.to_datetime(q["end"], errors="coerce")
    q["NetIncomeLoss"] = pd.to_numeric(q["NetIncomeLoss"], errors="coerce")
    q["YoYGrowthPct"] = pd.to_numeric(q.get("YoYGrowthPct"), errors="coerce")
    q = (
        q.dropna(subset=["end", "NetIncomeLoss"])
        .sort_values("end")
        .reset_index(drop=True)
    )
    if q.empty:
        return html.Div("No 10-Q diagnostics available.", className="text-muted")

    # ── YoY (1-year same-quarter) ──────────────────────────────────────────
    yoy = q["YoYGrowthPct"].dropna()
    accel = q["YoYGrowthPct"].diff().dropna()

    latest_yoy = float(yoy.iloc[-1]) if not yoy.empty else None
    avg_yoy_4q = float(yoy.tail(4).mean()) if not yoy.empty else None
    avg_yoy_all = float(yoy.mean()) if not yoy.empty else None
    yoy_hit = float((yoy > 0).mean()) if not yoy.empty else None
    yoy_vol = float(yoy.std()) if len(yoy) > 1 else None
    avg_accel = float(accel.mean()) if not accel.empty else None
    avg_accel_4q = float(accel.tail(4).mean()) if not accel.empty else None
    latest_accel = float(accel.iloc[-1]) if not accel.empty else None

    # ── Multi-year same-quarter growth ─────────────────────────────────────
    # Build lookup: (year, fiscal_quarter) -> NetIncomeLoss
    q["_year"] = q["end"].dt.year
    q["_fq"] = ((q["end"].dt.month - 1) // 3) + 1  # 1=Q1 … 4=Q4
    ni_lookup: dict = {}
    for _, row in q.iterrows():
        ni_lookup[(int(row["_year"]), int(row["_fq"]))] = row["NetIncomeLoss"]

    rates_2y: list[float] = []
    rates_3y: list[float] = []
    for _, row in q.iterrows():
        yr, fq, ni_val = int(row["_year"]), int(row["_fq"]), row["NetIncomeLoss"]
        if pd.isna(ni_val):
            continue
        p2 = ni_lookup.get((yr - 2, fq))
        p3 = ni_lookup.get((yr - 3, fq))
        if p2 is not None and pd.notna(p2) and abs(p2) > 0:
            rates_2y.append((ni_val - p2) / abs(p2))
        if p3 is not None and pd.notna(p3) and abs(p3) > 0:
            rates_3y.append((ni_val - p3) / abs(p3))

    latest_2y = float(rates_2y[-1]) if rates_2y else None
    avg_2y = float(np.mean(rates_2y)) if rates_2y else None
    latest_3y = float(rates_3y[-1]) if rates_3y else None
    avg_3y = float(np.mean(rates_3y)) if rates_3y else None

    # ── Drawdown & regime ──────────────────────────────────────────────────
    ni = q["NetIncomeLoss"]
    running_peak = ni.cummax()
    drawdown = ni / running_peak - 1
    max_drawdown = float(drawdown.min()) if not drawdown.empty else None

    regime = "n/a"
    if latest_yoy is not None and pd.notna(latest_yoy):
        if latest_yoy >= 0 and latest_accel is not None and pd.notna(latest_accel):
            regime = "Accelerating Growth" if latest_accel >= 0 else "Slowing Growth"
        elif latest_yoy < 0:
            regime = "Contracting"

    def _fmt_pct(v: float | None) -> str:
        return "n/a" if v is None or pd.isna(v) else f"{v * 100:.1f}%"

    def _fmt_pct_signed(v: float | None) -> str:
        return "n/a" if v is None or pd.isna(v) else f"{v * 100:+.1f}%"

    return html.Div(
        [
            html.Div(
                f"{ticker_norm} 10-Q Trend Diagnostics", className="fw-semibold mb-2"
            ),
            html.Div(
                "All multi-year rates compare only the same fiscal quarter (e.g. Q1 vs Q1). "
                "Acceleration is the quarter-to-quarter change in YoY growth.",
                className="text-muted small mb-2",
            ),
            # 1-year (YoY)
            html.Div(
                "1-Year (YoY) Same-Quarter Growth",
                className="fw-semibold mt-2 mb-1 small text-secondary",
            ),
            html.Ul(
                [
                    html.Li(f"Latest YoY growth: {_fmt_pct(latest_yoy)}"),
                    html.Li(f"4-quarter average YoY: {_fmt_pct(avg_yoy_4q)}"),
                    html.Li(f"Full-history average YoY: {_fmt_pct(avg_yoy_all)}"),
                    html.Li(f"Positive growth hit rate: {_fmt_pct(yoy_hit)}"),
                    html.Li(f"YoY volatility (std dev): {_fmt_pct(yoy_vol)}"),
                ],
                className="mb-0",
            ),
            # 2-year same-quarter
            html.Div(
                "2-Year Same-Quarter Growth",
                className="fw-semibold mt-2 mb-1 small text-secondary",
            ),
            html.Ul(
                [
                    html.Li(f"Latest 2Y: {_fmt_pct(latest_2y)}"),
                    html.Li(f"Average 2Y (all quarters): {_fmt_pct(avg_2y)}"),
                ],
                className="mb-0",
            ),
            # 3-year same-quarter
            html.Div(
                "3-Year Same-Quarter Growth",
                className="fw-semibold mt-2 mb-1 small text-secondary",
            ),
            html.Ul(
                [
                    html.Li(f"Latest 3Y: {_fmt_pct(latest_3y)}"),
                    html.Li(f"Average 3Y (all quarters): {_fmt_pct(avg_3y)}"),
                ],
                className="mb-0",
            ),
            # Momentum & risk
            html.Div(
                "Momentum & Risk",
                className="fw-semibold mt-2 mb-1 small text-secondary",
            ),
            html.Ul(
                [
                    html.Li(
                        f"Average acceleration (pp/qtr): {'n/a' if avg_accel is None or pd.isna(avg_accel) else f'{avg_accel * 100:+.1f}'}"
                    ),
                    html.Li(
                        f"4Q average acceleration (pp/qtr): {'n/a' if avg_accel_4q is None or pd.isna(avg_accel_4q) else f'{avg_accel_4q * 100:+.1f}'}"
                    ),
                    html.Li(
                        f"Max earnings drawdown from peak: {_fmt_pct(max_drawdown)}"
                    ),
                    html.Li(f"Latest regime: {regime}"),
                ],
                className="mb-0",
            ),
        ]
    )


def create_10q_ml_signals(quarterly_df: pd.DataFrame, ticker: str):
    ticker_norm = _normalize_ticker(ticker)
    if quarterly_df is None or quarterly_df.empty:
        return html.Div("No 10-Q ML signals available.", className="text-muted")

    q = quarterly_df.copy()
    q["end"] = pd.to_datetime(q["end"], errors="coerce")
    q["NetIncomeLoss"] = pd.to_numeric(q["NetIncomeLoss"], errors="coerce")
    q = (
        q.dropna(subset=["end", "NetIncomeLoss"])
        .sort_values("end")
        .reset_index(drop=True)
    )

    if len(q) < 8:
        return html.Div(
            "Insufficient data for 10-Q ML signals (need >=8 quarters).",
            className="text-muted",
        )

    ni = q["NetIncomeLoss"].astype(float).values
    all_positive = bool((ni > 0).all())
    y_fit = np.log1p(ni) if all_positive else ni
    x_idx = np.arange(len(y_fit), dtype=float)

    coeffs = np.polyfit(x_idx, y_fit, 1)
    y_pred = np.polyval(coeffs, x_idx)
    residual_std = float(np.std(y_fit - y_pred))
    slope = float(coeffs[0])

    next_idx = float(len(y_fit))
    next_y = float(np.polyval(coeffs, next_idx))
    if all_positive:
        next_val = float(np.expm1(next_y))
        next_lo = float(np.expm1(next_y - residual_std))
        next_hi = float(np.expm1(next_y + residual_std))
        slope_pct = float(np.expm1(abs(slope))) - 1.0
        direction = "Up" if slope > 0 else "Down"
        trend_label = f"{direction} (~{slope_pct * 100:.1f}%/q implied)"
    else:
        next_val = next_y
        next_lo = next_y - residual_std
        next_hi = next_y + residual_std
        direction = "Up" if slope > 1e4 else ("Down" if slope < -1e4 else "Flat")
        trend_label = direction

    qoq = np.diff(ni) / (np.abs(ni[:-1]) + 1e-9)
    qoq_std = float(np.std(qoq)) if len(qoq) else None
    if qoq_std is None:
        volatility_regime = "n/a"
    elif qoq_std >= 0.35:
        volatility_regime = "High"
    elif qoq_std >= 0.20:
        volatility_regime = "Moderate"
    else:
        volatility_regime = "Low"

    # Keep the model forecast aligned with panel D, and add diagnostics forecast.
    shared_prediction = _predict_next_quarter_earnings(q)
    diagnostics_prediction = _predict_next_quarter_earnings_diagnostics(q)

    if shared_prediction is not None:
        model_next_end = pd.to_datetime(shared_prediction["next_end"], errors="coerce")
        model_next_val = float(shared_prediction["predicted_net_income"])
        band = max(abs(next_hi - next_lo) / 2.0, abs(model_next_val) * 0.05)
        model_next_lo = model_next_val - band
        model_next_hi = model_next_val + band
    else:
        model_next_end = q["end"].iloc[-1] + pd.DateOffset(months=3)
        model_next_val = float(next_val)
        model_next_lo = float(next_lo)
        model_next_hi = float(next_hi)

    if diagnostics_prediction is not None:
        diag_next_end = pd.to_datetime(
            diagnostics_prediction["next_end"], errors="coerce"
        )
        diag_next_val = float(diagnostics_prediction["predicted_net_income"])
    else:
        diag_next_end = model_next_end
        diag_next_val = None

    # Same-quarter prior-year lookup for forecast YoY comparisons.
    q_lookup = q.copy()
    if "Fiscal Quarter" in q_lookup.columns:
        q_lookup["_qtr"] = pd.to_numeric(
            q_lookup["Fiscal Quarter"]
            .astype(str)
            .str.extract(r"([1-4])", expand=False),
            errors="coerce",
        )
    else:
        q_lookup["_qtr"] = ((q_lookup["end"].dt.month - 1) // 3) + 1

    if "Fiscal Year" in q_lookup.columns:
        q_lookup["_year"] = pd.to_numeric(q_lookup["Fiscal Year"], errors="coerce")
    else:
        q_lookup["_year"] = q_lookup["end"].dt.year

    q_lookup = q_lookup.dropna(subset=["_year", "_qtr"])
    ni_lookup = {
        (int(row["_year"]), int(row["_qtr"])): float(row["NetIncomeLoss"])
        for _, row in q_lookup.iterrows()
        if pd.notna(row["NetIncomeLoss"])
    }

    # Use fiscal stepping when fiscal labels exist to target the right quarter key.
    last_row = q_lookup.iloc[-1] if not q_lookup.empty else None
    last_fy = int(last_row["_year"]) if last_row is not None else None
    last_fq = int(last_row["_qtr"]) if last_row is not None else None
    last_end_lookup = (
        pd.to_datetime(last_row["end"], errors="coerce")
        if last_row is not None
        else None
    )

    def _same_quarter_yoy_from_forecast(
        forecast_end: pd.Timestamp, forecast_val: float | None
    ) -> float | None:
        if forecast_val is None or forecast_end is None or pd.isna(forecast_end):
            return None

        if (
            last_fy is not None
            and last_fq is not None
            and last_end_lookup is not None
            and pd.notna(last_end_lookup)
        ):
            months_diff = (forecast_end.year - last_end_lookup.year) * 12 + (
                forecast_end.month - last_end_lookup.month
            )
            steps = max(0, int(round(months_diff / 3)))
            fy, fq = int(last_fy), int(last_fq)
            for _ in range(steps):
                fq += 1
                if fq > 4:
                    fq = 1
                    fy += 1
        else:
            fy = int(forecast_end.year)
            fq = int(((forecast_end.month - 1) // 3) + 1)

        prev_val = ni_lookup.get((fy - 1, fq))
        if prev_val is None or not pd.notna(prev_val) or abs(prev_val) < 1e-9:
            return None
        return float((forecast_val - prev_val) / abs(prev_val))

    model_same_q_yoy = _same_quarter_yoy_from_forecast(model_next_end, model_next_val)
    diag_same_q_yoy = _same_quarter_yoy_from_forecast(diag_next_end, diag_next_val)

    anomaly_items = []
    if len(qoq) >= 4:
        median_q = float(np.median(qoq))
        mad = float(np.median(np.abs(qoq - median_q))) or 1e-9
        z_scores = (qoq - median_q) / (1.4826 * mad)
        for i, z in enumerate(z_scores):
            if abs(z) > 2.5:
                qtr_end = q["end"].iloc[i + 1]
                q_lbl = f"{qtr_end.year} Q{((qtr_end.month - 1) // 3) + 1}"
                flag_dir = "surge" if z > 0 else "drop"
                anomaly_items.append(
                    html.Li(f"{q_lbl}: anomalous {flag_dir} (z={z:.1f})")
                )
    if not anomaly_items:
        anomaly_items = [html.Li("No anomalous quarters detected")]

    model_next_label = f"{model_next_end.year} Q{((model_next_end.month - 1) // 3) + 1}"
    diag_next_label = f"{diag_next_end.year} Q{((diag_next_end.month - 1) // 3) + 1}"

    def _fmt_b(v: float) -> str:
        av = abs(v)
        if av >= 1e9:
            return f"${v / 1e9:.2f}B"
        if av >= 1e6:
            return f"${v / 1e6:.2f}M"
        return f"${v:,.0f}"

    def _fmt_pct_signed(v: float | None) -> str:
        return "n/a" if v is None or pd.isna(v) else f"{v * 100:+.1f}%"

    model_forecast_text = (
        f"{model_next_label} model forecast: {_fmt_b(model_next_val)}"
        f" (range {_fmt_b(model_next_lo)} - {_fmt_b(model_next_hi)})"
    )
    diagnostics_forecast_text = (
        f"{diag_next_label} diagnostics forecast: {_fmt_b(diag_next_val)}"
        if diag_next_val is not None
        else "Diagnostics forecast: n/a"
    )
    spread_text = (
        f"Forecast spread (diagnostics - model): {_fmt_b(diag_next_val - model_next_val)}"
        if diag_next_val is not None
        else "Forecast spread (diagnostics - model): n/a"
    )

    return html.Div(
        [
            html.Div(
                f"{ticker_norm} 10-Q ML Trend Signals", className="fw-semibold mb-2"
            ),
            html.Div(
                "Signals use linear trend fit on quarterly earnings and robust anomaly detection on QoQ growth.",
                className="text-muted small mb-2",
            ),
            html.Ul(
                [
                    html.Li(f"Trend direction: {trend_label}"),
                    html.Li(model_forecast_text),
                    html.Li(
                        f"{model_next_label} model same-quarter YoY change: {_fmt_pct_signed(model_same_q_yoy)}"
                    ),
                    html.Li(diagnostics_forecast_text),
                    html.Li(
                        f"{diag_next_label} diagnostics same-quarter YoY change: {_fmt_pct_signed(diag_same_q_yoy)}"
                    ),
                    html.Li(spread_text),
                    html.Li(
                        "QoQ growth volatility: "
                        f"{'n/a' if qoq_std is None else f'{qoq_std * 100:.1f}%'} ({volatility_regime})"
                    ),
                    html.Li(
                        [
                            html.Span("Anomaly flags:"),
                            html.Ul(anomaly_items, className="mb-0 mt-1"),
                        ]
                    ),
                ],
                className="mb-0",
            ),
        ]
    )


def create_10k_ml_signals(annual_df: pd.DataFrame, ticker: str):
    ticker_norm = _normalize_ticker(ticker)
    if annual_df is None or annual_df.empty:
        return html.Div("No ML signals available.", className="text-muted")

    df = annual_df.copy()
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    df["NetIncomeLoss"] = pd.to_numeric(df["NetIncomeLoss"], errors="coerce")
    df = df.dropna(subset=["end", "NetIncomeLoss"]).sort_values("end")
    if len(df) < 3:
        return html.Div(
            "Insufficient data for ML signals (need \u22653 years).",
            className="text-muted",
        )

    ni = df["NetIncomeLoss"].values
    years = np.array([d.year for d in df["end"]])

    # ── Trend forecast (linear regression on log1p or raw) ─────────────────
    all_positive = bool((ni > 0).all())
    y_fit = np.log1p(ni) if all_positive else ni.astype(float)
    x_idx = np.arange(len(y_fit), dtype=float)
    coeffs = np.polyfit(x_idx, y_fit, 1)
    y_pred = np.polyval(coeffs, x_idx)
    residual_std = float(np.std(y_fit - y_pred))
    slope = float(coeffs[0])

    next_idx = float(len(y_fit))
    next_log = float(np.polyval(coeffs, next_idx))
    if all_positive:
        next_val = float(np.expm1(next_log))
        next_lo = float(np.expm1(next_log - residual_std))
        next_hi = float(np.expm1(next_log + residual_std))
        slope_pct = float(np.expm1(abs(slope))) - 1.0
        direction = "Up" if slope > 0 else "Down"
        trend_label = f"{direction} (~{slope_pct * 100:.1f}%/yr implied)"
    else:
        next_val, next_lo, next_hi = (
            next_log,
            next_log - residual_std,
            next_log + residual_std,
        )
        direction = "Up" if slope > 1e4 else ("Down" if slope < -1e4 else "Flat")
        trend_label = direction

    next_year = int(years[-1]) + 1

    def _fmt_b(v: float) -> str:
        av = abs(v)
        if av >= 1e9:
            return f"${v / 1e9:.2f}B"
        if av >= 1e6:
            return f"${v / 1e6:.2f}M"
        return f"${v:,.0f}"

    forecast_text = (
        f"{next_year} forecast: {_fmt_b(next_val)}"
        f" (range {_fmt_b(next_lo)} \u2013 {_fmt_b(next_hi)})"
    )

    # ── Change-point detection (variance + mean shift across halves) ────────
    mid = len(ni) // 2
    cp_signal = "n/a"
    if mid >= 2:
        first_half, second_half = ni[:mid].astype(float), ni[mid:].astype(float)
        var1, var2 = float(np.var(first_half)), float(np.var(second_half))
        mean1, mean2 = float(np.mean(first_half)), float(np.mean(second_half))
        var_ratio = var2 / var1 if var1 > 0 else float("inf")
        mean_shift_pct = abs(mean2 - mean1) / (abs(mean1) + 1e-9)
        if var_ratio > 2.0 or mean_shift_pct > 0.5:
            cp_signal = f"Structural shift detected around {int(years[mid])}"
        else:
            cp_signal = "No major structural break detected"

    # ── Anomaly flags (MAD-based z-score on YoY growth) ────────────────────
    growth = np.diff(ni.astype(float)) / (np.abs(ni[:-1].astype(float)) + 1e-9)
    anomaly_items = []
    if len(growth) >= 3:
        median_g = float(np.median(growth))
        mad = float(np.median(np.abs(growth - median_g))) or 1e-9
        z_scores = (growth - median_g) / (1.4826 * mad)
        for i, z in enumerate(z_scores):
            if abs(z) > 2.5:
                flag_dir = "surge" if z > 0 else "drop"
                anomaly_items.append(
                    html.Li(f"{int(years[i + 1])}: anomalous {flag_dir} (z={z:.1f})")
                )
    if not anomaly_items:
        anomaly_items = [html.Li("No anomalous years detected")]

    return html.Div(
        [
            html.Div(
                f"{ticker_norm} 10-K ML Trend Signals", className="fw-semibold mb-2"
            ),
            html.Div(
                "Signals use linear regression on earnings, variance-based change-point "
                "detection, and MAD robust z-scores \u2014 no external ML libraries.",
                className="text-muted small mb-2",
            ),
            html.Ul(
                [
                    html.Li(f"Trend direction: {trend_label}"),
                    html.Li(forecast_text),
                    html.Li(f"Change-point: {cp_signal}"),
                    html.Li(
                        [
                            html.Span("Anomaly flags:"),
                            html.Ul(anomaly_items, className="mb-0 mt-1"),
                        ]
                    ),
                ],
                className="mb-0",
            ),
        ]
    )


# -----------------------------
# Dash App
# -----------------------------
def build_layout():
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H2(
                                "EDGAR Earnings Quality Dashboard", className="mb-1"
                            ),
                            html.Div(
                                "Earnings quality from EDGAR XBRL filings.",
                                className="text-muted",
                            ),
                        ],
                        width=12,
                    )
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Inputs"),
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Label(
                                                            "Ticker",
                                                            className="form-label",
                                                        ),
                                                        dcc.Input(
                                                            id="eq-ticker-input",
                                                            placeholder="Enter ticker (AAPL)",
                                                            value="AAPL",
                                                            className="form-control",
                                                        ),
                                                    ],
                                                    width=12,
                                                    lg=4,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Button(
                                                            "Load (Live)",
                                                            id="eq-load-btn",
                                                            className="btn btn-primary w-100",
                                                        ),
                                                        html.Div(
                                                            id="eq-live-refreshed",
                                                            className="text-muted small mt-1",
                                                        ),
                                                        html.Button(
                                                            "Load (DB)",
                                                            id="eq-load-db-btn",
                                                            className="btn btn-outline-secondary w-100 mt-2",
                                                        ),
                                                    ],
                                                    width=12,
                                                    lg=3,
                                                ),
                                            ],
                                            className="g-2 align-items-end",
                                        )
                                    ]
                                ),
                            ],
                            className="shadow-sm",
                        ),
                        width=12,
                    )
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("1. 10-Q Earnings"),
                                dbc.CardBody(
                                    dcc.Graph(id="eq-10q-earnings-chart"),
                                    className="p-0",
                                ),
                            ],
                            className="shadow-sm",
                        ),
                        width=12,
                    )
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("2. 10-Q Per-Quarter CAGR Diagnostics"),
                                dbc.CardBody(
                                    html.Div(id="eq-10q-per-quarter-cagr"),
                                ),
                            ],
                            className="shadow-sm",
                        ),
                        width=12,
                    )
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    "3. 10-Q 4-Quarter Average CAGR Diagnostics"
                                ),
                                dbc.CardBody(
                                    html.Div(id="eq-10q-4qavg-cagr"),
                                ),
                            ],
                            className="shadow-sm",
                        ),
                        width=12,
                    )
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("4. 10-Q Earnings Detail"),
                                dbc.Col(
                                    dbc.Button(
                                        "Show",
                                        id="eq-10q-earnings-detail-toggle",
                                        color="link",
                                        size="sm",
                                        className="text-decoration-none p-0",
                                    ),
                                    width="auto",
                                    className="ms-auto",
                                ),
                            ],
                            className="g-0",
                        ),
                        width=12,
                    )
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Collapse(
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        dag.AgGrid(
                                            id="eq-10q-earnings-table",
                                            rowData=[],
                                            columnDefs=[],
                                            defaultColDef={
                                                "sortable": True,
                                                "filter": True,
                                                "resizable": True,
                                                "floatingFilter": True,
                                            },
                                            columnSize="sizeToFit",
                                            dashGridOptions={
                                                "pagination": False,
                                                "domLayout": "autoHeight",
                                            },
                                            className="ag-theme-alpine",
                                            style={"width": "100%"},
                                        ),
                                        className="p-0",
                                    ),
                                ],
                                className="shadow-sm",
                            ),
                            id="eq-10q-earnings-detail-collapse",
                            is_open=False,
                        ),
                        width=12,
                    )
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("5. 10-K Earnings Growth"),
                                dbc.CardBody(
                                    dcc.Graph(id="eq-10k-growth-chart"),
                                    className="p-0",
                                ),
                            ],
                            className="shadow-sm",
                        ),
                        width=12,
                    )
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("6. 10-K Trend Diagnostics"),
                                dbc.CardBody(
                                    html.Div(id="eq-10k-trend-summary"),
                                ),
                            ],
                            className="shadow-sm",
                        ),
                        width=12,
                    )
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("7. 10-K ML Trend Signals"),
                                dbc.CardBody(
                                    html.Div(id="eq-10k-ml-signals"),
                                ),
                            ],
                            className="shadow-sm",
                        ),
                        width=12,
                    )
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("8. 10-K Earnings Growth Detail"),
                                dbc.CardBody(
                                    dag.AgGrid(
                                        id="eq-10k-growth-table",
                                        rowData=[],
                                        columnDefs=[],
                                        defaultColDef={
                                            "sortable": True,
                                            "filter": True,
                                            "resizable": True,
                                            "floatingFilter": True,
                                        },
                                        columnSize="sizeToFit",
                                        dashGridOptions={
                                            "pagination": False,
                                            "domLayout": "autoHeight",
                                        },
                                        className="ag-theme-alpine",
                                        style={"width": "100%"},
                                    ),
                                    className="p-0",
                                ),
                            ],
                            className="shadow-sm",
                        ),
                        width=12,
                    )
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("9. Revenue vs Earnings — Why?"),
                                dbc.CardBody(
                                    dcc.Graph(id="eq-revenue-earnings-chart"),
                                    className="p-0",
                                ),
                            ],
                            className="shadow-sm",
                        ),
                        width=12,
                    )
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("10. Earnings Quality"),
                                dbc.CardBody(
                                    dcc.Graph(id="eq-earnings-chart"),
                                    className="p-0",
                                ),
                            ],
                            className="shadow-sm",
                        ),
                        width=12,
                    )
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("11. How Earnings Quality Works"),
                                dbc.CardBody(
                                    [
                                        html.P(
                                            "The chart compares reported net income to operating cash flow and shows the accrual ratio for each quarter."
                                        ),
                                        html.P(
                                            "Formulas used in both chart and table:"
                                        ),
                                        html.Ul(
                                            [
                                                html.Li(
                                                    "Average Total Assets = rolling 2-quarter mean of Assets"
                                                ),
                                                html.Li(
                                                    "Accrual Ratio = (Net Income - Operating Cash Flow) / Average Total Assets"
                                                ),
                                            ]
                                        ),
                                        html.P(
                                            "Interpretation: lower or negative accrual ratio generally indicates earnings are backed by cash flow, while persistently high positive accruals can indicate weaker earnings quality."
                                        ),
                                        html.P(
                                            "The Quarterly Detail table lists the exact values used to render the chart, including Net Income, Operating Cash Flow, Assets, Average Total Assets, and Accrual Ratio."
                                        ),
                                    ]
                                ),
                            ],
                            className="shadow-sm",
                        ),
                        width=12,
                    )
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Quarterly Detail"),
                                dbc.CardBody(
                                    dag.AgGrid(
                                        id="eq-earnings-table",
                                        rowData=[],
                                        columnDefs=[],
                                        defaultColDef={
                                            "sortable": True,
                                            "filter": True,
                                            "resizable": True,
                                            "floatingFilter": True,
                                        },
                                        columnSize="sizeToFit",
                                        dashGridOptions={
                                            "pagination": False,
                                            "domLayout": "autoHeight",
                                        },
                                        className="ag-theme-alpine",
                                        style={"width": "100%"},
                                    ),
                                    className="p-0",
                                ),
                            ],
                            className="shadow-sm",
                        ),
                        width=12,
                    )
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(id="eq-status-label", className="text-muted"),
                        width=12,
                    )
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(id="eq-last-refreshed", className="text-muted"),
                        width=12,
                    )
                ],
                className="mb-3",
            ),
        ],
        fluid=True,
        className="py-4",
    )


layout = build_layout()


def register_callbacks(app):
    @app.callback(
        Output("eq-10q-earnings-chart", "figure"),
        Output("eq-10q-per-quarter-cagr", "children"),
        Output("eq-10q-4qavg-cagr", "children"),
        Output("eq-10q-earnings-table", "rowData"),
        Output("eq-10q-earnings-table", "columnDefs"),
        Output("eq-earnings-chart", "figure"),
        Output("eq-earnings-table", "rowData"),
        Output("eq-earnings-table", "columnDefs"),
        Output("eq-10k-growth-chart", "figure"),
        Output("eq-10k-growth-table", "rowData"),
        Output("eq-10k-growth-table", "columnDefs"),
        Output("eq-10k-trend-summary", "children"),
        Output("eq-10k-ml-signals", "children"),
        Output("eq-revenue-earnings-chart", "figure"),
        Output("eq-status-label", "children"),
        Output("eq-last-refreshed", "children"),
        Output("eq-live-refreshed", "children"),
        Input("eq-load-btn", "n_clicks"),
        Input("eq-load-db-btn", "n_clicks"),
        Input("eq-ticker-input", "value"),
    )
    def update_chart(_n, _db_n, ticker):
        if _n is None and _db_n is None:
            quarterly_10q_df = build_quarterly_10q_earnings(
                ticker, use_cache=True, cache_only=True
            )
            df = build_financials(ticker, use_cache=True, cache_only=True)
            annual_df = build_annual_earnings_growth(
                ticker, use_cache=True, cache_only=True
            )
            if df is None or df.empty:
                return (
                    create_10q_earnings_figure(pd.DataFrame(), ticker),
                    create_10q_per_quarter_cagr(pd.DataFrame(), ticker),
                    create_10q_four_quarter_avg_cagr(pd.DataFrame(), ticker),
                    [],
                    [],
                    go.Figure(),
                    [],
                    [],
                    create_10k_growth_figure(pd.DataFrame(), ticker),
                    [],
                    [],
                    create_10k_trend_diagnostics(pd.DataFrame(), ticker),
                    create_10k_ml_signals(pd.DataFrame(), ticker),
                    create_revenue_vs_earnings_figure(pd.DataFrame(), ticker),
                    "No cached data found. Load live or run pipeline.",
                    "",
                    "",
                )
            df_display = df.copy()
            df_display["end"] = pd.to_datetime(df_display["end"]).dt.date.astype(str)
            df_display["ticker"] = _normalize_ticker(ticker)
            ordered_cols = _order_display_columns(df_display.columns.tolist())
            df_display = df_display[ordered_cols]
            columns = _build_column_defs(ordered_cols)
            data = df_display.to_dict("records")
            quarterly_10q_columns = []
            quarterly_10q_data = []
            if quarterly_10q_df is not None and not quarterly_10q_df.empty:
                quarterly_10q_display = quarterly_10q_df.copy()
                quarterly_10q_display["end"] = pd.to_datetime(
                    quarterly_10q_display["end"], errors="coerce"
                )
                quarterly_10q_display = quarterly_10q_display.sort_values(
                    "end", ascending=False
                )
                quarterly_10q_display["end"] = quarterly_10q_display[
                    "end"
                ].dt.date.astype(str)
                quarterly_10q_display["ticker"] = _normalize_ticker(ticker)
                quarterly_10q_ordered_cols = _order_display_columns(
                    quarterly_10q_display.columns.tolist()
                )
                quarterly_10q_display = quarterly_10q_display[
                    quarterly_10q_ordered_cols
                ]
                quarterly_10q_columns = _build_column_defs(quarterly_10q_ordered_cols)
                quarterly_10q_data = quarterly_10q_display.to_dict("records")
            annual_display = pd.DataFrame()
            annual_columns = []
            annual_data = []
            if annual_df is not None and not annual_df.empty:
                annual_display = annual_df.copy()
                annual_display["end"] = pd.to_datetime(
                    annual_display["end"], errors="coerce"
                )
                annual_display = annual_display.sort_values("end", ascending=False)
                annual_display["end"] = annual_display["end"].dt.date.astype(str)
                annual_display["ticker"] = _normalize_ticker(ticker)
                annual_ordered_cols = _order_display_columns(
                    annual_display.columns.tolist()
                )
                annual_display = annual_display[annual_ordered_cols]
                annual_columns = _build_column_defs(annual_ordered_cols)
                annual_data = annual_display.to_dict("records")
            refreshed = _format_refreshed(ticker)
            return (
                create_10q_earnings_figure(quarterly_10q_df, ticker, refreshed),
                create_10q_per_quarter_cagr(quarterly_10q_df, ticker),
                create_10q_four_quarter_avg_cagr(quarterly_10q_df, ticker),
                quarterly_10q_data,
                quarterly_10q_columns,
                create_figure(df, ticker, refreshed),
                data,
                columns,
                create_10k_growth_figure(annual_df, ticker, refreshed),
                annual_data,
                annual_columns,
                create_10k_trend_diagnostics(annual_df, ticker),
                create_10k_ml_signals(annual_df, ticker),
                create_revenue_vs_earnings_figure(annual_df, ticker, refreshed),
                "Loaded from SQLite cache.",
                refreshed,
                "",
            )

        triggered = ctx.triggered_id
        cache_only = triggered == "eq-load-db-btn"
        use_cache = triggered != "eq-load-btn"
        quarterly_10q_df = build_quarterly_10q_earnings(
            ticker, use_cache=use_cache, cache_only=cache_only
        )
        df = build_financials(ticker, use_cache=use_cache, cache_only=cache_only)
        annual_df = build_annual_earnings_growth(
            ticker, use_cache=use_cache, cache_only=cache_only
        )
        if df is None or df.empty:
            message = (
                "No cached data found. Run live load or pipeline."
                if cache_only
                else "No data returned for this ticker."
            )
            return (
                create_10q_earnings_figure(pd.DataFrame(), ticker),
                create_10q_per_quarter_cagr(pd.DataFrame(), ticker),
                create_10q_four_quarter_avg_cagr(pd.DataFrame(), ticker),
                [],
                [],
                go.Figure(),
                [],
                [],
                create_10k_growth_figure(pd.DataFrame(), ticker),
                [],
                [],
                create_10k_trend_diagnostics(pd.DataFrame(), ticker),
                create_10k_ml_signals(pd.DataFrame(), ticker),
                create_revenue_vs_earnings_figure(pd.DataFrame(), ticker),
                message,
                "",
                "",
            )

        df_display = df.copy()
        df_display["end"] = pd.to_datetime(df_display["end"]).dt.date.astype(str)
        df_display["ticker"] = _normalize_ticker(ticker)
        ordered_cols = _order_display_columns(df_display.columns.tolist())
        df_display = df_display[ordered_cols]
        columns = _build_column_defs(ordered_cols)
        data = df_display.to_dict("records")

        quarterly_10q_columns = []
        quarterly_10q_data = []
        if quarterly_10q_df is not None and not quarterly_10q_df.empty:
            quarterly_10q_display = quarterly_10q_df.copy()
            quarterly_10q_display["end"] = pd.to_datetime(
                quarterly_10q_display["end"], errors="coerce"
            )
            quarterly_10q_display = quarterly_10q_display.sort_values(
                "end", ascending=False
            )
            quarterly_10q_display["end"] = quarterly_10q_display["end"].dt.date.astype(
                str
            )
            quarterly_10q_display["ticker"] = _normalize_ticker(ticker)
            quarterly_10q_ordered_cols = _order_display_columns(
                quarterly_10q_display.columns.tolist()
            )
            quarterly_10q_display = quarterly_10q_display[quarterly_10q_ordered_cols]
            quarterly_10q_columns = _build_column_defs(quarterly_10q_ordered_cols)
            quarterly_10q_data = quarterly_10q_display.to_dict("records")

        annual_columns = []
        annual_data = []
        if annual_df is not None and not annual_df.empty:
            annual_display = annual_df.copy()
            annual_display["end"] = pd.to_datetime(
                annual_display["end"], errors="coerce"
            )
            annual_display = annual_display.sort_values("end", ascending=False)
            annual_display["end"] = annual_display["end"].dt.date.astype(str)
            annual_display["ticker"] = _normalize_ticker(ticker)
            annual_ordered_cols = _order_display_columns(
                annual_display.columns.tolist()
            )
            annual_display = annual_display[annual_ordered_cols]
            annual_columns = _build_column_defs(annual_ordered_cols)
            annual_data = annual_display.to_dict("records")

        status = "Loaded from SQLite cache." if cache_only else "Loaded live from SEC."
        refreshed = _format_refreshed(ticker)
        live_refreshed = refreshed if not cache_only else ""
        return (
            create_10q_earnings_figure(quarterly_10q_df, ticker, refreshed),
            create_10q_per_quarter_cagr(quarterly_10q_df, ticker),
            create_10q_four_quarter_avg_cagr(quarterly_10q_df, ticker),
            quarterly_10q_data,
            quarterly_10q_columns,
            create_figure(df, ticker, refreshed),
            data,
            columns,
            create_10k_growth_figure(annual_df, ticker, refreshed),
            annual_data,
            annual_columns,
            create_10k_trend_diagnostics(annual_df, ticker),
            create_10k_ml_signals(annual_df, ticker),
            create_revenue_vs_earnings_figure(annual_df, ticker, refreshed),
            status,
            refreshed,
            live_refreshed,
        )

    @app.callback(
        Output("eq-10q-earnings-detail-collapse", "is_open"),
        Output("eq-10q-earnings-detail-toggle", "children"),
        Input("eq-10q-earnings-detail-toggle", "n_clicks"),
        State("eq-10q-earnings-detail-collapse", "is_open"),
    )
    def toggle_10q_earnings_detail(n_clicks, is_open):
        if not n_clicks:
            return is_open, "Show"
        next_state = not is_open
        return next_state, ("Hide" if next_state else "Show")


register_callbacks(get_app())
