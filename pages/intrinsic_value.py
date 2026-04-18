import os
import re
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import yfinance as yf
import dash
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
from dash import get_app, Input, Output, State, ctx, dcc, html
from dotenv import load_dotenv

from storage_paths import DATA_DIR, parquet_path

dash.register_page(__name__, path="/intrinsic-value", name="Intrinsic Value", order=7)
# =========================
# CONFIG
# =========================
load_dotenv()

SEC_CONTACT_EMAIL = os.getenv("SEC_CONTACT_EMAIL", "your_email@example.com")
HEADERS = {"User-Agent": f"dash-of-edgar intrinsic-value {SEC_CONTACT_EMAIL}"}
REQUEST_TIMEOUT = 20

INTRINSIC_PARQUET_PATH = parquet_path("edgar_intrinsic_value")
INTRINSIC_PAYLOAD_PARQUET_PATH = parquet_path("edgar_intrinsic_value_payload")
INTRINSIC_SQLITE_PATH = DATA_DIR / "edgar_intrinsic_value.sqlite"

_CIK_CACHE = {}
_BETA_CACHE = {}

TAGS = {
    "revenue": [
        "Revenues",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
    ],
    "ebit": ["OperatingIncomeLoss"],
    "taxes": ["IncomeTaxExpenseBenefit"],
    "ocf": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "depreciation": ["DepreciationDepletionAndAmortization"],
    "working_capital_change": ["IncreaseDecreaseInOperatingCapital"],
    "ppe": ["PropertyPlantAndEquipmentNet"],
    "accounts_payable": ["AccountsPayableCurrent"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "debt": [
        "LongTermDebt",
        "LongTermDebtNoncurrent",
        "LongTermDebtAndFinanceLeaseObligations",
    ],
    "shares_common": [
        "CommonStockSharesOutstanding",
        "EntityCommonStockSharesOutstanding",
    ],
    "shares_diluted": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
}

DISPLAY_NAME_MAP = {
    "row_type": "Row Type",
    "calc_label": "Calculation",
    "calc_value": "Value",
    "calc_value_text": "Value (Text)",
    "calc_formula": "Formula",
    "calc_math": "Math",
    "period_end": "Period End",
    "form": "Filing Form",
    "period_type": "Period Type",
    "fp": "Fiscal Period",
    "fy": "Fiscal Year",
    "year": "Forecast Year",
    "shares_outstanding": "Shares Outstanding",
    "projected_revenue": "Projected Revenue",
    "projected_fcf": "Projected FCF",
    "fcf_margin": "FCF Margin",
    "nopat": "NOPAT",
    "tax_rate": "Tax Rate",
    "roic": "ROIC",
    "discounted_fcf": "Discounted FCF",
    "terminal_value": "Terminal Value",
    "terminal_pv": "Terminal PV",
    "source": "Source",
}

CURRENCY_FIELDS = {
    "projected_revenue",
    "projected_fcf",
    "nopat",
    "discounted_fcf",
    "terminal_value",
    "terminal_pv",
}
PERCENT_FIELDS = {"fcf_margin", "tax_rate", "roic"}
NUMBER_FIELDS = {"shares_outstanding", "year"}


def _ag_header_name(field: str) -> str:
    return DISPLAY_NAME_MAP.get(field, field.replace("_", " ").title())


def _build_projection_col_def(field: str) -> dict:
    col_def = {"headerName": _ag_header_name(field), "field": field}

    if field in CURRENCY_FIELDS:
        col_def["valueFormatter"] = {
            "function": "params.value == null ? '' : Number(params.value).toLocaleString('en-US', {style: 'currency', currency: 'USD', minimumFractionDigits: 2, maximumFractionDigits: 2})"
        }
        col_def["type"] = "rightAligned"
    elif field in PERCENT_FIELDS:
        col_def["valueFormatter"] = {
            "function": "params.value == null ? '' : `${(Number(params.value) * 100).toFixed(2)}%`"
        }
        col_def["type"] = "rightAligned"
    elif field in NUMBER_FIELDS:
        col_def["valueFormatter"] = {
            "function": "params.value == null ? '' : Number(params.value).toLocaleString('en-US', {maximumFractionDigits: 2})"
        }
        col_def["type"] = "rightAligned"

    return col_def


# =========================
# EDGAR FETCH
# =========================
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalize_ticker(ticker: str) -> str:
    return str(ticker or "").strip().upper()


def get_cik(ticker: str):
    ticker_norm = _normalize_ticker(ticker)
    if not ticker_norm:
        return None

    cached = _CIK_CACHE.get(ticker_norm)
    if cached:
        return cached

    url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()

    for item in data.values():
        if str(item.get("ticker", "")).upper() == ticker_norm:
            cik = str(item.get("cik_str", "")).zfill(10)
            if cik:
                _CIK_CACHE[ticker_norm] = cik
                return cik

    return None


def get_company_facts(cik: str):
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def extract_series(facts, taxonomy, tag, units):
    try:
        unit_buckets = facts["facts"][taxonomy][tag]["units"]
        data = None
        for unit_name in units:
            if unit_name in unit_buckets:
                data = unit_buckets[unit_name]
                break
        if data is None:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        if df.empty:
            return pd.DataFrame()

        # Annual intrinsic model: use 10-K filings only.
        if "form" in df.columns:
            df = df[df["form"] == "10-K"]

        df["end"] = pd.to_datetime(df["end"], errors="coerce")
        df = df[df["end"].notna()]

        if "filed" in df.columns:
            df = df.sort_values(["end", "filed"]).drop_duplicates("end", keep="last")
        else:
            df = df.sort_values("end").drop_duplicates("end", keep="last")

        cols = ["end", "val"]
        if "form" in df.columns:
            cols.append("form")
        if "form" in df.columns:
            df["period_type"] = "FY"
            cols.append("period_type")
        if "fp" in df.columns:
            cols.append("fp")
        if "fy" in df.columns:
            cols.append("fy")

        return df[cols].rename(columns={"val": tag})
    except (KeyError, ValueError, TypeError):
        return pd.DataFrame()


def extract_first_tag_series(
    facts,
    taxonomy: str,
    tag_candidates: list[str],
    units: tuple[str, ...],
) -> pd.DataFrame:
    best_series = pd.DataFrame()
    best_latest = pd.Timestamp.min
    best_count = -1

    for tag in tag_candidates:
        series = extract_series(facts, taxonomy, tag, units)
        if series.empty:
            continue

        candidate = series.copy()
        candidate_end = pd.to_datetime(candidate["end"], errors="coerce")
        latest = candidate_end.max()
        count = int(candidate_end.notna().sum())

        if pd.isna(latest):
            continue

        # Prefer the most recent series; if tied, keep the one with more history.
        if latest > best_latest or (latest == best_latest and count > best_count):
            best_latest = latest
            best_count = count
            best_series = candidate

    if best_series.empty:
        return pd.DataFrame()

    # Normalize to a stable canonical column name used downstream.
    value_cols = [
        c
        for c in best_series.columns
        if c not in {"end", "form", "period_type", "fp", "fy"}
    ]
    if not value_cols:
        return pd.DataFrame()
    source_val_col = value_cols[0]
    return best_series.rename(columns={source_val_col: tag_candidates[0]})


# =========================
# FINANCIAL MODEL
# =========================
def build_financials(facts):
    revenue = extract_first_tag_series(
        facts,
        "us-gaap",
        TAGS["revenue"],
        units=("USD",),
    )
    ocf = extract_first_tag_series(
        facts,
        "us-gaap",
        TAGS["ocf"],
        units=("USD",),
    )
    capex = extract_first_tag_series(
        facts,
        "us-gaap",
        TAGS["capex"],
        units=("USD",),
    )
    ebit = extract_first_tag_series(
        facts,
        "us-gaap",
        TAGS["ebit"],
        units=("USD",),
    )
    taxes = extract_first_tag_series(
        facts,
        "us-gaap",
        TAGS["taxes"],
        units=("USD",),
    )
    depreciation = extract_first_tag_series(
        facts,
        "us-gaap",
        TAGS["depreciation"],
        units=("USD",),
    )
    wc_change = extract_first_tag_series(
        facts,
        "us-gaap",
        TAGS["working_capital_change"],
        units=("USD",),
    )
    ppe = extract_first_tag_series(
        facts,
        "us-gaap",
        TAGS["ppe"],
        units=("USD",),
    )
    accounts_payable = extract_first_tag_series(
        facts,
        "us-gaap",
        TAGS["accounts_payable"],
        units=("USD",),
    )
    cash = extract_first_tag_series(
        facts,
        "us-gaap",
        TAGS["cash"],
        units=("USD",),
    )
    debt = extract_first_tag_series(
        facts,
        "us-gaap",
        TAGS["debt"],
        units=("USD",),
    )
    shares_common = extract_first_tag_series(
        facts,
        "us-gaap",
        TAGS["shares_common"],
        units=("shares", "pure"),
    )
    shares_diluted = extract_first_tag_series(
        facts,
        "us-gaap",
        TAGS["shares_diluted"],
        units=("shares", "pure"),
    )

    if revenue.empty or ocf.empty or capex.empty:
        return None

    def _value_only(frame: pd.DataFrame, value_col: str) -> pd.DataFrame:
        if frame is None or frame.empty or value_col not in frame.columns:
            return pd.DataFrame(columns=["end", value_col])
        return frame[["end", value_col]].copy()

    df = revenue.copy()
    df = df.merge(_value_only(ocf, TAGS["ocf"][0]), on="end", how="inner")
    df = df.merge(_value_only(capex, TAGS["capex"][0]), on="end", how="inner")
    if not ebit.empty:
        df = df.merge(_value_only(ebit, TAGS["ebit"][0]), on="end", how="left")
    if not taxes.empty:
        df = df.merge(_value_only(taxes, TAGS["taxes"][0]), on="end", how="left")
    if not depreciation.empty:
        df = df.merge(
            _value_only(depreciation, TAGS["depreciation"][0]), on="end", how="left"
        )
    if not wc_change.empty:
        df = df.merge(
            _value_only(wc_change, TAGS["working_capital_change"][0]),
            on="end",
            how="left",
        )
    if not ppe.empty:
        df = df.merge(_value_only(ppe, TAGS["ppe"][0]), on="end", how="left")
    if not accounts_payable.empty:
        df = df.merge(
            _value_only(accounts_payable, TAGS["accounts_payable"][0]),
            on="end",
            how="left",
        )
    if not cash.empty:
        df = df.merge(_value_only(cash, TAGS["cash"][0]), on="end", how="left")
    if not debt.empty:
        df = df.merge(_value_only(debt, TAGS["debt"][0]), on="end", how="left")

    if df.empty:
        return None

    df = df.sort_values("end").tail(10).copy()
    df["revenue"] = pd.to_numeric(df[TAGS["revenue"][0]], errors="coerce")
    df["ocf"] = pd.to_numeric(df[TAGS["ocf"][0]], errors="coerce")
    df["capex"] = pd.to_numeric(df[TAGS["capex"][0]], errors="coerce")
    if TAGS["ebit"][0] in df.columns:
        df["ebit"] = pd.to_numeric(df[TAGS["ebit"][0]], errors="coerce")
    else:
        df["ebit"] = np.nan
    if TAGS["taxes"][0] in df.columns:
        df["taxes"] = pd.to_numeric(df[TAGS["taxes"][0]], errors="coerce")
    else:
        df["taxes"] = np.nan
    if TAGS["depreciation"][0] in df.columns:
        df["depreciation"] = pd.to_numeric(df[TAGS["depreciation"][0]], errors="coerce")
    else:
        df["depreciation"] = 0.0
    if TAGS["working_capital_change"][0] in df.columns:
        df["working_capital_change"] = pd.to_numeric(
            df[TAGS["working_capital_change"][0]], errors="coerce"
        )
    else:
        df["working_capital_change"] = 0.0
    if TAGS["ppe"][0] in df.columns:
        df["ppe"] = pd.to_numeric(df[TAGS["ppe"][0]], errors="coerce")
    else:
        df["ppe"] = np.nan
    if TAGS["accounts_payable"][0] in df.columns:
        df["accounts_payable"] = pd.to_numeric(
            df[TAGS["accounts_payable"][0]], errors="coerce"
        )
    else:
        df["accounts_payable"] = np.nan
    if TAGS["cash"][0] in df.columns:
        df["cash"] = pd.to_numeric(df[TAGS["cash"][0]], errors="coerce")
    else:
        df["cash"] = np.nan
    if TAGS["debt"][0] in df.columns:
        df["debt"] = pd.to_numeric(df[TAGS["debt"][0]], errors="coerce")
    else:
        df["debt"] = np.nan

    df = df.dropna(subset=["revenue", "ocf", "capex"])
    if df.empty:
        return None

    # Detailed profitability path when EBIT/tax is present; fallback remains OCF - CapEx.
    df["tax_rate"] = np.where(
        (df["ebit"].notna()) & (df["ebit"] != 0),
        df["taxes"] / df["ebit"],
        0.21,
    )
    df["tax_rate"] = df["tax_rate"].clip(0, 0.35).fillna(0.21)
    df["nopat"] = np.where(
        df["ebit"].notna(), df["ebit"] * (1 - df["tax_rate"]), np.nan
    )
    df["fcf_detailed"] = (
        df["nopat"]
        + df["depreciation"].fillna(0)
        - np.abs(df["capex"])
        - df["working_capital_change"].fillna(0)
    )
    df["fcf_base"] = df["ocf"] - np.abs(df["capex"])
    df["fcf"] = df["fcf_detailed"].where(df["fcf_detailed"].notna(), df["fcf_base"])

    df["invested_capital"] = df["ppe"] - df["accounts_payable"]
    df["roic"] = np.where(
        (df["nopat"].notna())
        & (df["invested_capital"].notna())
        & (df["invested_capital"] > 0),
        df["nopat"] / df["invested_capital"],
        np.nan,
    )

    meta_cols = [c for c in ["form", "period_type", "fp", "fy"] if c in df.columns]
    df = df[
        meta_cols
        + ["end", "revenue", "fcf", "nopat", "tax_rate", "roic", "cash", "debt"]
    ].copy()

    shares_outstanding = None
    if not shares_common.empty:
        shares_common = shares_common.sort_values("end")
        last_common = pd.to_numeric(
            shares_common.iloc[-1][TAGS["shares_common"][0]], errors="coerce"
        )
        if pd.notna(last_common) and last_common > 0:
            shares_outstanding = float(last_common)

    if shares_outstanding is None and not shares_diluted.empty:
        shares_diluted = shares_diluted.sort_values("end")
        last_diluted = pd.to_numeric(
            shares_diluted.iloc[-1][TAGS["shares_diluted"][0]], errors="coerce"
        )
        if pd.notna(last_diluted) and last_diluted > 0:
            shares_outstanding = float(last_diluted)

    if shares_outstanding is None:
        shares_outstanding = 1_000_000_000.0

    return df, shares_outstanding


def _save_parquet_snapshot(data: pd.DataFrame) -> None:
    INTRINSIC_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if INTRINSIC_PARQUET_PATH.exists():
        try:
            existing = pd.read_parquet(INTRINSIC_PARQUET_PATH)
        except Exception:
            existing = pd.DataFrame()
        if not existing.empty:
            data = pd.concat([existing, data], ignore_index=True)
    data = data.drop_duplicates(
        subset=["ticker", "period_end", "discount_rate", "terminal_growth"],
        keep="last",
    )
    data.to_parquet(INTRINSIC_PARQUET_PATH, index=False)


def _save_payload_parquet_snapshot(
    ticker: str, payload_df: pd.DataFrame, shares: float, pulled_at_utc: str
) -> None:
    if payload_df is None or payload_df.empty:
        return

    INTRINSIC_PAYLOAD_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = payload_df.copy()
    payload["ticker"] = str(ticker).upper()
    payload["shares_outstanding"] = float(shares)
    payload["pulled_at_utc"] = str(pulled_at_utc)
    payload["period_end"] = pd.to_datetime(
        payload["end"], errors="coerce"
    ).dt.date.astype(str)

    keep_cols = [
        "ticker",
        "period_end",
        "revenue",
        "fcf",
        "nopat",
        "tax_rate",
        "roic",
        "cash",
        "debt",
        "form",
        "period_type",
        "fp",
        "fy",
        "shares_outstanding",
        "pulled_at_utc",
    ]
    for col in keep_cols:
        if col not in payload.columns:
            payload[col] = None
    payload = payload[keep_cols]

    if INTRINSIC_PAYLOAD_PARQUET_PATH.exists():
        try:
            existing = pd.read_parquet(INTRINSIC_PAYLOAD_PARQUET_PATH)
        except Exception:
            existing = pd.DataFrame()
        if not existing.empty:
            payload = pd.concat([existing, payload], ignore_index=True)

    payload = payload.drop_duplicates(
        subset=["ticker", "period_end", "pulled_at_utc"], keep="last"
    )
    payload.to_parquet(INTRINSIC_PAYLOAD_PARQUET_PATH, index=False)


def _upsert_sqlite_intrinsic(data: pd.DataFrame) -> tuple[int, int]:
    INTRINSIC_SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(INTRINSIC_SQLITE_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS intrinsic_value (
                ticker TEXT NOT NULL,
                period_end TEXT NOT NULL,
                discount_rate REAL NOT NULL,
                terminal_growth REAL NOT NULL,
                revenue REAL,
                fcf REAL,
                shares_outstanding REAL,
                enterprise_value REAL,
                intrinsic_price REAL,
                pulled_at_utc TEXT NOT NULL,
                PRIMARY KEY (ticker, period_end, discount_rate, terminal_growth)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_intrinsic_ticker ON intrinsic_value(ticker)"
        )
        rows = [
            tuple(row)
            for row in data[
                [
                    "ticker",
                    "period_end",
                    "discount_rate",
                    "terminal_growth",
                    "revenue",
                    "fcf",
                    "shares_outstanding",
                    "enterprise_value",
                    "intrinsic_price",
                    "pulled_at_utc",
                ]
            ].itertuples(index=False, name=None)
        ]

        before = conn.execute("SELECT COUNT(*) FROM intrinsic_value").fetchone()[0]
        conn.executemany(
            """
            INSERT OR REPLACE INTO intrinsic_value (
                ticker,
                period_end,
                discount_rate,
                terminal_growth,
                revenue,
                fcf,
                shares_outstanding,
                enterprise_value,
                intrinsic_price,
                pulled_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        after = conn.execute("SELECT COUNT(*) FROM intrinsic_value").fetchone()[0]

    return len(rows), after - before


def _fetch_cached_intrinsic_input(ticker: str) -> dict | None:
    if not INTRINSIC_SQLITE_PATH.exists():
        return None

    sql = """
    SELECT
        ticker,
        period_end,
        revenue,
        fcf,
        shares_outstanding,
        pulled_at_utc
    FROM intrinsic_value
    WHERE ticker = ?
    ORDER BY pulled_at_utc DESC
    LIMIT 1
    """

    try:
        with sqlite3.connect(INTRINSIC_SQLITE_PATH) as conn:
            df = pd.read_sql_query(sql, conn, params=(ticker.upper(),))
    except Exception:
        return None

    if df.empty:
        return None

    return df.iloc[0].to_dict()


def _upsert_sqlite_payload_cache(
    ticker: str, payload_df: pd.DataFrame, shares: float, pulled_at_utc: str
) -> int:
    if payload_df is None or payload_df.empty:
        return 0

    INTRINSIC_SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = payload_df.copy()
    payload["period_end"] = pd.to_datetime(
        payload["end"], errors="coerce"
    ).dt.date.astype(str)

    rows = []
    for row in payload.itertuples(index=False):
        rows.append(
            (
                str(ticker).upper(),
                str(getattr(row, "period_end", "")),
                float(getattr(row, "revenue", np.nan))
                if pd.notna(getattr(row, "revenue", np.nan))
                else None,
                float(getattr(row, "fcf", np.nan))
                if pd.notna(getattr(row, "fcf", np.nan))
                else None,
                float(getattr(row, "nopat", np.nan))
                if pd.notna(getattr(row, "nopat", np.nan))
                else None,
                float(getattr(row, "tax_rate", np.nan))
                if pd.notna(getattr(row, "tax_rate", np.nan))
                else None,
                float(getattr(row, "roic", np.nan))
                if pd.notna(getattr(row, "roic", np.nan))
                else None,
                float(getattr(row, "cash", np.nan))
                if pd.notna(getattr(row, "cash", np.nan))
                else None,
                float(getattr(row, "debt", np.nan))
                if pd.notna(getattr(row, "debt", np.nan))
                else None,
                str(getattr(row, "form", "")) or None,
                str(getattr(row, "period_type", "")) or None,
                str(getattr(row, "fp", "")) or None,
                str(getattr(row, "fy", "")) or None,
                float(shares),
                str(pulled_at_utc),
            )
        )

    with sqlite3.connect(INTRINSIC_SQLITE_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS intrinsic_payload (
                ticker TEXT NOT NULL,
                period_end TEXT NOT NULL,
                revenue REAL,
                fcf REAL,
                nopat REAL,
                tax_rate REAL,
                roic REAL,
                cash REAL,
                debt REAL,
                form TEXT,
                period_type TEXT,
                fp TEXT,
                fy TEXT,
                shares_outstanding REAL,
                pulled_at_utc TEXT NOT NULL,
                PRIMARY KEY (ticker, period_end, pulled_at_utc)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_intrinsic_payload_ticker_ts ON intrinsic_payload(ticker, pulled_at_utc)"
        )
        existing_cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(intrinsic_payload)").fetchall()
        }
        if "cash" not in existing_cols:
            conn.execute("ALTER TABLE intrinsic_payload ADD COLUMN cash REAL")
        if "debt" not in existing_cols:
            conn.execute("ALTER TABLE intrinsic_payload ADD COLUMN debt REAL")
        conn.executemany(
            """
            INSERT OR REPLACE INTO intrinsic_payload (
                ticker,
                period_end,
                revenue,
                fcf,
                nopat,
                tax_rate,
                roic,
                cash,
                debt,
                form,
                period_type,
                fp,
                fy,
                shares_outstanding,
                pulled_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()

    return len(rows)


def _fetch_cached_intrinsic_payload(
    ticker: str,
) -> tuple[pd.DataFrame, float | None, str | None] | tuple[None, None, None]:
    if not INTRINSIC_SQLITE_PATH.exists():
        return None, None, None

    ticker_norm = str(ticker or "").upper().strip()
    if not ticker_norm:
        return None, None, None

    latest_ts_sql = """
    SELECT MAX(pulled_at_utc) AS pulled_at_utc
    FROM intrinsic_payload
    WHERE ticker = ?
    """

    payload_sql = """
    SELECT
        period_end AS end,
        revenue,
        fcf,
        nopat,
        tax_rate,
        roic,
        cash,
        debt,
        form,
        period_type,
        fp,
        fy,
        shares_outstanding,
        pulled_at_utc
    FROM intrinsic_payload
    WHERE ticker = ? AND pulled_at_utc = ?
    ORDER BY period_end ASC
    """

    try:
        with sqlite3.connect(INTRINSIC_SQLITE_PATH) as conn:
            ts_df = pd.read_sql_query(latest_ts_sql, conn, params=(ticker_norm,))
            if ts_df.empty or pd.isna(ts_df.loc[0, "pulled_at_utc"]):
                return None, None, None
            pulled_at_utc = str(ts_df.loc[0, "pulled_at_utc"])
            payload_df = pd.read_sql_query(
                payload_sql,
                conn,
                params=(ticker_norm, pulled_at_utc),
            )
    except Exception:
        return None, None, None

    if payload_df.empty:
        return None, None, None

    shares = pd.to_numeric(payload_df["shares_outstanding"], errors="coerce").dropna()
    shares_outstanding = float(shares.iloc[-1]) if not shares.empty else None
    payload_df = payload_df.drop(columns=["shares_outstanding", "pulled_at_utc"])

    return payload_df, shares_outstanding, pulled_at_utc


def _build_projection_table(
    payload_df: pd.DataFrame,
    revs: np.ndarray,
    fcfs: np.ndarray,
    discounted: np.ndarray,
    shares: float,
    discount: float,
    terminal: float,
    source: str,
    terminal_value_override: float | None = None,
    terminal_pv_override: float | None = None,
    enterprise_value: float | None = None,
    equity_value: float | None = None,
    intrinsic_price: float | None = None,
    net_debt: float | None = None,
    terminal_fcff_next: float | None = None,
    reinvestment_rate: float | None = None,
    discount_mode: str | None = None,
    cash_value: float | None = None,
    debt_value: float | None = None,
) -> tuple[list[dict], list[dict]]:
    def _calc_margin(fcf_value, revenue_value):
        if pd.isna(fcf_value) or pd.isna(revenue_value) or float(revenue_value) == 0:
            return None
        return round(float(fcf_value) / float(revenue_value), 4)

    years = np.arange(1, len(fcfs) + 1)
    terminal_value = (
        float(terminal_value_override)
        if terminal_value_override is not None
        else (fcfs[-1] * (1 + terminal) / (discount - terminal))
    )
    terminal_pv = (
        float(terminal_pv_override)
        if terminal_pv_override is not None
        else (terminal_value / ((1 + discount) ** len(fcfs)))
    )

    rows = []
    base_year = datetime.now(timezone.utc).year
    baseline_roic = None
    if payload_df is not None and not payload_df.empty:
        payload_display = payload_df.copy()
        payload_display = payload_display.sort_values("end", ascending=True)
        payload_display["end"] = pd.to_datetime(payload_display["end"]).dt.date.astype(
            str
        )
        payload_display["revenue"] = pd.to_numeric(
            payload_display["revenue"], errors="coerce"
        )
        payload_display["fcf"] = pd.to_numeric(payload_display["fcf"], errors="coerce")
        if "roic" in payload_display.columns:
            roic_vals = pd.to_numeric(payload_display["roic"], errors="coerce")
            roic_vals = roic_vals[np.isfinite(roic_vals)]
            if not roic_vals.empty:
                baseline_roic = float(roic_vals.median())

        try:
            payload_years = pd.to_datetime(
                payload_display["end"], errors="coerce"
            ).dt.year
            if payload_years.notna().any():
                base_year = int(payload_years.max())
        except Exception:
            pass

        for row in payload_display.itertuples(index=False):
            rows.append(
                {
                    "row_type": "Payload",
                    "period_end": str(row.end),
                    "form": getattr(row, "form", None),
                    "period_type": getattr(row, "period_type", None),
                    "fp": getattr(row, "fp", None),
                    "fy": getattr(row, "fy", None),
                    "year": None,
                    "shares_outstanding": round(float(shares), 2),
                    "projected_revenue": round(float(row.revenue), 2)
                    if pd.notna(row.revenue)
                    else None,
                    "projected_fcf": round(float(row.fcf), 2)
                    if pd.notna(row.fcf)
                    else None,
                    "fcf_margin": _calc_margin(row.fcf, row.revenue),
                    "nopat": round(float(getattr(row, "nopat", np.nan)), 2)
                    if pd.notna(getattr(row, "nopat", np.nan))
                    else None,
                    "tax_rate": round(float(getattr(row, "tax_rate", np.nan)), 4)
                    if pd.notna(getattr(row, "tax_rate", np.nan))
                    else None,
                    "roic": round(float(getattr(row, "roic", np.nan)), 4)
                    if pd.notna(getattr(row, "roic", np.nan))
                    else None,
                    "discounted_fcf": None,
                    "terminal_value": None,
                    "terminal_pv": None,
                    "source": source,
                }
            )

    for idx, year in enumerate(years):
        calendar_year = base_year + int(year)
        rows.append(
            {
                "row_type": "Forecast",
                "period_end": None,
                "form": None,
                "period_type": None,
                "fp": None,
                "fy": None,
                "year": calendar_year,
                "shares_outstanding": round(float(shares), 2),
                "projected_revenue": round(float(revs[idx]), 2),
                "projected_fcf": round(float(fcfs[idx]), 2),
                "fcf_margin": _calc_margin(fcfs[idx], revs[idx]),
                "nopat": None,
                "tax_rate": None,
                "roic": round(float(baseline_roic), 4)
                if baseline_roic is not None
                else None,
                "discounted_fcf": round(float(discounted[idx]), 2),
                "terminal_value": None,
                "terminal_pv": None,
                "source": source,
            }
        )

    rows.append(
        {
            "row_type": "Terminal",
            "period_end": None,
            "form": None,
            "period_type": None,
            "fp": None,
            "fy": None,
            "year": base_year + int(years[-1]),
            "shares_outstanding": round(float(shares), 2),
            "projected_revenue": None,
            "projected_fcf": round(float(fcfs[-1]), 2),
            "fcf_margin": None,
            "nopat": None,
            "tax_rate": None,
            "roic": round(float(baseline_roic), 4)
            if baseline_roic is not None
            else None,
            "discounted_fcf": None,
            "terminal_value": round(float(terminal_value), 2),
            "terminal_pv": round(float(terminal_pv), 2),
            "source": source,
        }
    )

    def _fmt_usd(value: float | None) -> str:
        if value is None or not np.isfinite(value):
            return "N/A"
        return f"${float(value):,.2f}"

    def _fmt_num(value: float | None, decimals: int = 2) -> str:
        if value is None or not np.isfinite(value):
            return "N/A"
        return f"{float(value):,.{decimals}f}"

    def _fmt_pct(value: float | None, decimals: int = 2) -> str:
        if value is None or not np.isfinite(value):
            return "N/A"
        return f"{float(value) * 100:.{decimals}f}%"

    calc_rows = [
        {
            "row_type": "Calc",
            "calc_label": "PV of Forecast FCFs",
            "calc_value": round(float(discounted.sum()), 2),
            "calc_value_text": None,
            "calc_formula": "Sum of discounted_fcf (Years 1-20)",
            "calc_math": f"{_fmt_usd(float(discounted.sum()))} = PV_FCF_1 + PV_FCF_2 + ... + PV_FCF_20",
            "source": source,
        },
        {
            "row_type": "Calc",
            "calc_label": "Terminal FCFF (Year 21)",
            "calc_value": round(float(terminal_fcff_next), 2)
            if terminal_fcff_next is not None
            else None,
            "calc_value_text": None,
            "calc_formula": (
                "NOPAT * (1 - g/ROIC) * (1 + g)"
                if reinvestment_rate is not None
                else "FCF_20 * (1 + g)"
            ),
            "calc_math": (
                f"{_fmt_usd(terminal_fcff_next)} = NOPAT * (1 - {_fmt_pct(reinvestment_rate, 2)}) * (1 + {_fmt_pct(terminal, 2)})"
                if reinvestment_rate is not None and terminal_fcff_next is not None
                else (
                    f"{_fmt_usd(terminal_fcff_next)} = FCF_20 * (1 + {_fmt_pct(terminal, 2)})"
                    if terminal_fcff_next is not None
                    else None
                )
            ),
            "source": source,
        },
        {
            "row_type": "Calc",
            "calc_label": "Terminal Value",
            "calc_value": round(float(terminal_value), 2),
            "calc_value_text": None,
            "calc_formula": "Terminal FCFF / (discount - terminal_growth)",
            "calc_math": (
                f"{_fmt_usd(terminal_value)} = {_fmt_usd(terminal_fcff_next)} / ({_fmt_num(discount, 4)} - {_fmt_num(terminal, 4)})"
                if terminal_fcff_next is not None
                else None
            ),
            "source": source,
        },
        {
            "row_type": "Calc",
            "calc_label": "PV of Terminal Value",
            "calc_value": round(float(terminal_pv), 2),
            "calc_value_text": None,
            "calc_formula": "Terminal Value / (1 + discount)^20",
            "calc_math": f"{_fmt_usd(terminal_pv)} = {_fmt_usd(terminal_value)} / (1 + {_fmt_num(discount, 4)})^20",
            "source": source,
        },
        {
            "row_type": "Calc",
            "calc_label": "Enterprise Value",
            "calc_value": round(float(enterprise_value), 2)
            if enterprise_value is not None
            else None,
            "calc_value_text": None,
            "calc_formula": "PV Forecast FCFs + PV Terminal Value",
            "calc_math": (
                f"{_fmt_usd(enterprise_value)} = {_fmt_usd(float(discounted.sum()))} + {_fmt_usd(terminal_pv)}"
                if enterprise_value is not None
                else None
            ),
            "source": source,
        },
        {
            "row_type": "Calc",
            "calc_label": "Net Debt (Debt - Cash)",
            "calc_value": -abs(round(float(net_debt), 2))
            if net_debt is not None
            else None,
            "calc_value_text": None,
            "calc_formula": "Used when debt and cash are available",
            "calc_math": (
                f"{_fmt_usd(net_debt)} = {_fmt_usd(debt_value)} - {_fmt_usd(cash_value)}"
                if net_debt is not None
                and debt_value is not None
                and cash_value is not None
                else "N/A"
            ),
            "source": source,
        },
        {
            "row_type": "Calc",
            "calc_label": "Equity Value",
            "calc_value": round(float(equity_value), 2)
            if equity_value is not None
            else None,
            "calc_value_text": None,
            "calc_formula": "Enterprise Value - Net Debt",
            "calc_math": (
                f"{_fmt_usd(equity_value)} = {_fmt_usd(enterprise_value)} - {_fmt_usd(net_debt)}"
                if equity_value is not None
                and enterprise_value is not None
                and net_debt is not None
                else (
                    f"{_fmt_usd(equity_value)} = {_fmt_usd(enterprise_value)}"
                    if equity_value is not None and enterprise_value is not None
                    else None
                )
            ),
            "source": source,
        },
        {
            "row_type": "Calc",
            "calc_label": "Shares Outstanding",
            "calc_value": None,
            "calc_value_text": _fmt_num(shares, 2),
            "calc_formula": "Most recent common shares (or diluted fallback)",
            "calc_math": f"Shares = {_fmt_num(shares, 2)}",
            "source": source,
        },
        {
            "row_type": "Calc",
            "calc_label": "Intrinsic Price Per Share",
            "calc_value": round(float(intrinsic_price), 4)
            if intrinsic_price is not None
            else None,
            "calc_value_text": None,
            "calc_formula": "Equity Value / Shares Outstanding",
            "calc_math": (
                f"${float(intrinsic_price):,.4f} = {_fmt_usd(equity_value)} / {_fmt_num(shares, 2)}"
                if intrinsic_price is not None and equity_value is not None
                else None
            ),
            "source": source,
        },
        {
            "row_type": "Calc",
            "calc_label": "Active Discount",
            "calc_value": None,
            "calc_value_text": f"{_fmt_num(discount, 6)} ({_fmt_pct(discount, 2)})",
            "calc_formula": str(discount_mode or "Unknown"),
            "calc_math": f"discount = {_fmt_num(discount, 6)} ({_fmt_pct(discount, 2)})",
            "source": source,
        },
    ]
    rows.extend(calc_rows)

    ordered_cols = [
        "row_type",
        "calc_label",
        "calc_value",
        "calc_value_text",
        "calc_formula",
        "calc_math",
        "period_end",
        "form",
        "period_type",
        "fp",
        "fy",
        "year",
        "shares_outstanding",
        "projected_revenue",
        "projected_fcf",
        "fcf_margin",
        "nopat",
        "tax_rate",
        "roic",
        "discounted_fcf",
        "terminal_value",
        "terminal_pv",
        "source",
    ]
    columns = [_build_projection_col_def(col) for col in ordered_cols]
    return rows, columns


def _baseline_fcf_margin_from_payload(payload_df: pd.DataFrame) -> float | None:
    if payload_df is None or payload_df.empty:
        return None
    tmp = payload_df.copy()
    tmp["revenue"] = pd.to_numeric(tmp.get("revenue"), errors="coerce")
    tmp["fcf"] = pd.to_numeric(tmp.get("fcf"), errors="coerce")
    tmp = tmp.dropna(subset=["revenue", "fcf"])
    if tmp.empty:
        return None
    revenue_avg = float(tmp["revenue"].mean())
    if revenue_avg == 0:
        return None
    return float(tmp["fcf"].mean()) / revenue_avg


def _clamp_fcf_margin_for_slider(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(np.clip(value, 0.05, 0.35))


def _latest_numeric_from_payload(payload_df: pd.DataFrame, column: str) -> float | None:
    if payload_df is None or payload_df.empty or column not in payload_df.columns:
        return None
    values = pd.to_numeric(payload_df[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.iloc[-1])


def _estimate_terminal_assumptions(payload_df: pd.DataFrame) -> dict:
    assumptions = {
        "tax_rate": 0.21,
        "roic": None,
        "nopat_margin": None,
        "cash": None,
        "debt": None,
    }

    if payload_df is None or payload_df.empty:
        return assumptions

    tax_rate = _latest_numeric_from_payload(payload_df, "tax_rate")
    if tax_rate is not None and np.isfinite(tax_rate):
        assumptions["tax_rate"] = float(np.clip(tax_rate, 0.0, 0.35))

    roic_series = pd.to_numeric(payload_df.get("roic"), errors="coerce")
    roic_series = roic_series[(roic_series > 0) & np.isfinite(roic_series)]
    if not roic_series.empty:
        assumptions["roic"] = float(roic_series.median())

    if "nopat" in payload_df.columns and "revenue" in payload_df.columns:
        tmp = payload_df[["nopat", "revenue"]].copy()
        tmp["nopat"] = pd.to_numeric(tmp["nopat"], errors="coerce")
        tmp["revenue"] = pd.to_numeric(tmp["revenue"], errors="coerce")
        tmp = tmp[(tmp["revenue"] > 0) & tmp["nopat"].notna()]
        if not tmp.empty:
            margins = tmp["nopat"] / tmp["revenue"]
            margins = margins[np.isfinite(margins)]
            if not margins.empty:
                assumptions["nopat_margin"] = float(
                    np.clip(margins.median(), 0.02, 0.60)
                )

    assumptions["cash"] = _latest_numeric_from_payload(payload_df, "cash")
    assumptions["debt"] = _latest_numeric_from_payload(payload_df, "debt")
    return assumptions


def _get_equity_beta(ticker: str) -> float:
    ticker_norm = _normalize_ticker(ticker)
    if not ticker_norm:
        return 1.0

    cached = _BETA_CACHE.get(ticker_norm)
    if cached is not None:
        return float(cached)

    beta = 1.0
    try:
        info = yf.Ticker(ticker_norm).info
        candidate = pd.to_numeric(info.get("beta"), errors="coerce")
        if pd.notna(candidate) and np.isfinite(candidate):
            beta = float(np.clip(candidate, 0.3, 3.0))
    except Exception:
        beta = 1.0

    _BETA_CACHE[ticker_norm] = beta
    return beta


def _estimate_wacc(
    ticker: str,
    shares: float,
    tax_rate: float,
    debt_value: float | None,
) -> float | None:
    if shares is None or not np.isfinite(shares) or shares <= 0:
        return None

    market_price = _fetch_current_market_price(ticker)
    if market_price is None or not np.isfinite(market_price) or market_price <= 0:
        return None

    market_cap = float(market_price) * float(shares)
    if market_cap <= 0:
        return None

    beta = _get_equity_beta(ticker)
    risk_free = 0.042
    equity_risk_premium = 0.055
    cost_of_equity = risk_free + beta * equity_risk_premium

    debt_for_weights = 0.0
    if debt_value is not None and np.isfinite(debt_value) and debt_value > 0:
        debt_for_weights = float(debt_value)

    pre_tax_cost_of_debt = risk_free + 0.018
    after_tax_cost_of_debt = pre_tax_cost_of_debt * (
        1 - float(np.clip(tax_rate, 0, 0.35))
    )

    if debt_for_weights <= 0:
        return float(np.clip(cost_of_equity, 0.05, 0.20))

    total_capital = market_cap + debt_for_weights
    if total_capital <= 0:
        return None

    w_e = market_cap / total_capital
    w_d = debt_for_weights / total_capital
    wacc = w_e * cost_of_equity + w_d * after_tax_cost_of_debt
    return float(np.clip(wacc, 0.05, 0.20))


def _resolve_discount_rate(
    ticker: str,
    shares: float,
    slider_discount: float,
    discount_mode: str,
    terminal_tax_rate: float,
    debt_value: float | None,
) -> tuple[float, str]:
    base_discount = float(slider_discount)
    mode = str(discount_mode or "auto").strip().lower()
    estimated_wacc = _estimate_wacc(ticker, shares, terminal_tax_rate, debt_value)

    if mode == "auto":
        if estimated_wacc is not None:
            return float(estimated_wacc), "WACC auto"
        return base_discount, "Auto fallback"

    return base_discount, "Manual"


def _equity_value_from_enterprise(
    enterprise_value: float, cash_value: float | None, debt_value: float | None
) -> tuple[float, float | None]:
    cash = (
        float(cash_value)
        if cash_value is not None and np.isfinite(cash_value)
        else None
    )
    debt = (
        float(debt_value)
        if debt_value is not None and np.isfinite(debt_value)
        else None
    )
    if cash is None or debt is None:
        return float(enterprise_value), None

    net_debt = debt - cash
    return float(enterprise_value - net_debt), float(net_debt)


# =========================
# FORECAST
# =========================
def forecast_fcf(last_rev, fcf_margin, growth_start=0.10, growth_end=0.03, years=20):

    growth_rates = np.linspace(growth_start, growth_end, years)

    revs = []
    fcfs = []

    r = last_rev

    for g in growth_rates:
        r *= 1 + g
        revs.append(r)
        fcfs.append(r * fcf_margin)

    return np.array(revs), np.array(fcfs)


# =========================
# DCF
# =========================
def dcf_value(fcfs, discount=0.10, terminal_growth=0.025):

    enterprise_value, discounted, _tv, _tv_pv, _fcff_next, _reinvest = (
        dcf_value_with_terminal_adjustments(
            fcfs,
            discount=discount,
            terminal_growth=terminal_growth,
            terminal_nopat=None,
            terminal_roic=None,
        )
    )

    return enterprise_value, discounted


def dcf_value_with_terminal_adjustments(
    fcfs,
    discount=0.10,
    terminal_growth=0.025,
    terminal_nopat: float | None = None,
    terminal_roic: float | None = None,
):

    if discount <= terminal_growth:
        raise ValueError("Discount rate must be greater than terminal growth")

    years = np.arange(1, len(fcfs) + 1)

    discounted = fcfs / ((1 + discount) ** years)

    reinvestment_rate = None
    if (
        terminal_nopat is not None
        and np.isfinite(terminal_nopat)
        and terminal_nopat > 0
        and terminal_roic is not None
        and np.isfinite(terminal_roic)
        and terminal_roic > terminal_growth
    ):
        reinvestment_rate = float(np.clip(terminal_growth / terminal_roic, 0.0, 0.90))
        terminal_fcff_next = (
            float(terminal_nopat) * (1 - reinvestment_rate) * (1 + terminal_growth)
        )
    else:
        terminal_fcff_next = float(fcfs[-1]) * (1 + terminal_growth)

    terminal = terminal_fcff_next / (discount - terminal_growth)

    terminal_pv = terminal / ((1 + discount) ** len(fcfs))

    enterprise_value = discounted.sum() + terminal_pv

    return (
        float(enterprise_value),
        discounted,
        float(terminal),
        float(terminal_pv),
        float(terminal_fcff_next),
        reinvestment_rate,
    )


def _empty_intrinsic_vs_actual_figure(title: str = "Intrinsic vs Actual Share Price"):
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="Price (USD)",
        template="plotly_white",
    )
    fig.update_xaxes(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb")
    fig.update_yaxes(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb")
    return fig


def _fetch_actual_yearly_prices(ticker: str) -> pd.DataFrame:
    try:
        raw = yf.Ticker(ticker).history(period="max", auto_adjust=True)
    except Exception:
        return pd.DataFrame(columns=["year", "actual_price"])

    if raw is None or raw.empty:
        return pd.DataFrame(columns=["year", "actual_price"])

    prices = raw.reset_index()
    date_col = "Date" if "Date" in prices.columns else prices.columns[0]
    prices[date_col] = pd.to_datetime(prices[date_col], errors="coerce")
    prices = prices[prices[date_col].notna()].copy()
    if prices.empty:
        return pd.DataFrame(columns=["year", "actual_price"])

    prices["year"] = prices[date_col].dt.year
    yearly = prices.sort_values(date_col).groupby("year", as_index=False).tail(1)
    yearly["actual_price"] = pd.to_numeric(yearly["Close"], errors="coerce")
    yearly = yearly[["year", "actual_price"]].dropna()
    return yearly


def _fetch_current_market_price(ticker: str) -> float | None:
    ticker_norm = _normalize_ticker(ticker)
    if not ticker_norm:
        return None
    try:
        hist = yf.Ticker(ticker_norm).history(period="5d", auto_adjust=True)
    except Exception:
        return None
    if hist is None or hist.empty or "Close" not in hist.columns:
        return None
    close = pd.to_numeric(hist["Close"], errors="coerce").dropna()
    if close.empty:
        return None
    return float(close.iloc[-1])


def _build_intrinsic_vs_actual_figure(
    ticker: str,
    payload_df: pd.DataFrame,
    shares: float,
    discount: float,
    terminal: float,
    growth_start: float,
    latest_intrinsic_price: float,
) -> go.Figure:
    if payload_df is None or payload_df.empty or shares <= 0:
        return _empty_intrinsic_vs_actual_figure(
            f"{ticker.upper()} Intrinsic vs Actual Share Price"
        )

    payload = payload_df.copy()
    payload["end"] = pd.to_datetime(payload["end"], errors="coerce")
    payload["year"] = payload["end"].dt.year
    payload["revenue"] = pd.to_numeric(payload["revenue"], errors="coerce")
    payload["fcf"] = pd.to_numeric(payload["fcf"], errors="coerce")
    payload = payload.dropna(subset=["year", "revenue", "fcf"])
    if payload.empty:
        return _empty_intrinsic_vs_actual_figure(
            f"{ticker.upper()} Intrinsic vs Actual Share Price"
        )

    intrinsic_rows = []
    for row in payload.itertuples(index=False):
        if row.revenue <= 0:
            continue
        fcf_margin = float(row.fcf) / float(row.revenue)
        revs, fcfs = forecast_fcf(
            float(row.revenue), fcf_margin, growth_start=growth_start
        )
        try:
            ev, _disc, _tv, _tv_pv, _next_fcff, _reinvest = (
                dcf_value_with_terminal_adjustments(
                    fcfs,
                    discount,
                    terminal,
                    terminal_nopat=None,
                    terminal_roic=None,
                )
            )
        except ValueError:
            continue
        cash_value = pd.to_numeric(getattr(row, "cash", np.nan), errors="coerce")
        debt_value = pd.to_numeric(getattr(row, "debt", np.nan), errors="coerce")
        equity_value, _net_debt = _equity_value_from_enterprise(
            float(ev),
            float(cash_value) if pd.notna(cash_value) else None,
            float(debt_value) if pd.notna(debt_value) else None,
        )
        intrinsic_rows.append(
            {
                "year": int(row.year),
                "intrinsic_price": float(equity_value) / float(shares),
            }
        )

    intrinsic_df = pd.DataFrame(intrinsic_rows)
    if intrinsic_df.empty:
        latest_year = int(payload["year"].max())
        intrinsic_df = pd.DataFrame(
            [{"year": latest_year, "intrinsic_price": float(latest_intrinsic_price)}]
        )
    else:
        intrinsic_df = intrinsic_df.sort_values("year").drop_duplicates(
            "year", keep="last"
        )

    actual_df = _fetch_actual_yearly_prices(ticker)
    if not actual_df.empty:
        actual_df = actual_df[actual_df["year"].isin(intrinsic_df["year"])].copy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=intrinsic_df["year"],
            y=intrinsic_df["intrinsic_price"],
            name="Intrinsic Price",
            mode="lines+markers",
            line=dict(color="#2563eb", width=2),
        )
    )

    if not actual_df.empty:
        fig.add_trace(
            go.Scatter(
                x=actual_df["year"],
                y=actual_df["actual_price"],
                name="Actual Share Price",
                mode="lines+markers",
                line=dict(color="#16a34a", width=2),
            )
        )

    fig.update_layout(
        title=f"{ticker.upper()} Intrinsic vs Actual Share Price",
        xaxis_title="Year",
        yaxis_title="Price (USD)",
        template="plotly_white",
    )
    fig.update_xaxes(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb")
    fig.update_yaxes(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb")
    return fig


# =========================
# DASH APP
# =========================
def build_layout():
    return dbc.Container(
        [
            dcc.Store(id="intrinsic-last-data", storage_type="memory"),
            dcc.Graph(id="projection_chart", style={"display": "none"}),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H2(
                                "EDGAR Intrinsic Value Dashboard", className="mb-1"
                            ),
                            html.Div(
                                "Discounted cash flow valuation from SEC 10-K company facts.",
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
                                dbc.CardHeader("Intrinsic Price"),
                                dbc.CardBody(
                                    html.H3(
                                        "$--",
                                        id="intrinsic-kpi-value",
                                        className="mb-0",
                                    )
                                ),
                            ],
                            className="shadow-sm",
                        ),
                        width=12,
                        lg=4,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Current Market Price"),
                                dbc.CardBody(
                                    html.H3(
                                        "$--",
                                        id="market-kpi-value",
                                        className="mb-0",
                                    )
                                ),
                            ],
                            className="shadow-sm",
                        ),
                        width=12,
                        lg=4,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Shares Outstanding"),
                                dbc.CardBody(
                                    html.H3(
                                        "--",
                                        id="shares-kpi-value",
                                        className="mb-0",
                                    )
                                ),
                            ],
                            className="shadow-sm",
                        ),
                        width=12,
                        lg=4,
                    ),
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Ticker & Actions"),
                                dbc.CardBody(
                                    [
                                        html.Label(
                                            "Ticker",
                                            className="form-label",
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dcc.Input(
                                                        id="ticker",
                                                        value="AAPL",
                                                        type="text",
                                                        debounce=True,
                                                        placeholder="Enter ticker (AAPL)",
                                                        className="form-control",
                                                    ),
                                                    width=12,
                                                    lg=6,
                                                ),
                                                dbc.Col(
                                                    html.Button(
                                                        "DB Pull",
                                                        id="db-pull-btn",
                                                        n_clicks=0,
                                                        className="btn btn-outline-secondary w-100",
                                                    ),
                                                    width=6,
                                                    lg=3,
                                                ),
                                                dbc.Col(
                                                    html.Button(
                                                        "EDGAR Pull",
                                                        id="edgar-pull-btn",
                                                        n_clicks=0,
                                                        className="btn btn-primary w-100",
                                                    ),
                                                    width=6,
                                                    lg=3,
                                                ),
                                            ],
                                            className="g-2",
                                        ),
                                    ]
                                ),
                            ],
                            className="shadow-sm h-100",
                        ),
                        width=12,
                        lg=4,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Sliders"),
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Div(
                                                            [
                                                                html.Label(
                                                                    "Discount Rate",
                                                                    className="form-label mb-0",
                                                                ),
                                                                html.Span(
                                                                    id="discount-value",
                                                                    className="small text-muted",
                                                                ),
                                                            ],
                                                            className="d-flex justify-content-between align-items-center",
                                                        ),
                                                        dcc.RadioItems(
                                                            id="discount-mode",
                                                            options=[
                                                                {
                                                                    "label": "Auto (WACC)",
                                                                    "value": "auto",
                                                                },
                                                                {
                                                                    "label": "Manual",
                                                                    "value": "manual",
                                                                },
                                                            ],
                                                            value="auto",
                                                            inline=True,
                                                            className="mb-2",
                                                        ),
                                                        dcc.Slider(
                                                            0.05,
                                                            0.15,
                                                            0.005,
                                                            value=0.10,
                                                            id="discount",
                                                            marks={
                                                                0.05: "5%",
                                                                0.10: "10%",
                                                                0.15: "15%",
                                                            },
                                                        ),
                                                    ],
                                                    width=12,
                                                    lg=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Div(
                                                            [
                                                                html.Label(
                                                                    "Terminal Growth",
                                                                    className="form-label mb-0",
                                                                ),
                                                                html.Span(
                                                                    id="terminal-value",
                                                                    className="small text-muted",
                                                                ),
                                                            ],
                                                            className="d-flex justify-content-between align-items-center",
                                                        ),
                                                        dcc.Slider(
                                                            0.01,
                                                            0.04,
                                                            0.002,
                                                            value=0.025,
                                                            id="terminal",
                                                            marks={
                                                                0.01: "1%",
                                                                0.025: "2.5%",
                                                                0.04: "4%",
                                                            },
                                                        ),
                                                    ],
                                                    width=12,
                                                    lg=6,
                                                ),
                                            ],
                                            className="g-3",
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Div(
                                                            [
                                                                html.Label(
                                                                    "FCF Margin",
                                                                    className="form-label mb-0",
                                                                ),
                                                                html.Span(
                                                                    id="fcf-margin-value",
                                                                    className="small text-muted",
                                                                ),
                                                            ],
                                                            className="d-flex justify-content-between align-items-center",
                                                        ),
                                                        dcc.Slider(
                                                            0.05,
                                                            0.35,
                                                            0.01,
                                                            value=0.15,
                                                            id="fcf_margin",
                                                            marks={
                                                                0.05: "5%",
                                                                0.15: "15%",
                                                                0.25: "25%",
                                                                0.35: "35%",
                                                            },
                                                        ),
                                                    ],
                                                    width=12,
                                                    lg=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Div(
                                                            [
                                                                html.Label(
                                                                    "Revenue Growth Start",
                                                                    className="form-label mb-0",
                                                                ),
                                                                html.Span(
                                                                    id="growth-start-value",
                                                                    className="small text-muted",
                                                                ),
                                                            ],
                                                            className="d-flex justify-content-between align-items-center",
                                                        ),
                                                        dcc.Slider(
                                                            0.03,
                                                            0.20,
                                                            0.005,
                                                            value=0.10,
                                                            id="growth_start",
                                                            marks={
                                                                0.03: "3%",
                                                                0.10: "10%",
                                                                0.20: "20%",
                                                            },
                                                        ),
                                                    ],
                                                    width=12,
                                                    lg=6,
                                                ),
                                            ],
                                            className="g-3",
                                        ),
                                    ]
                                ),
                            ],
                            className="shadow-sm h-100",
                        ),
                        width=12,
                        lg=8,
                    ),
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Historical Data (Payload)"),
                                dbc.CardBody(
                                    dag.AgGrid(
                                        id="intrinsic-payload-table",
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
                                        },
                                        className="ag-theme-alpine",
                                        style={"height": "600px", "width": "100%"},
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
                                dbc.CardHeader("20-Year Forecast"),
                                dbc.CardBody(
                                    dag.AgGrid(
                                        id="intrinsic-forecast-table",
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
                                        },
                                        className="ag-theme-alpine",
                                        style={"height": "600px", "width": "100%"},
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
                                dbc.CardHeader("Valuation Calculation Steps"),
                                dbc.CardBody(
                                    dag.AgGrid(
                                        id="intrinsic-calc-table",
                                        rowData=[],
                                        columnDefs=[],
                                        defaultColDef={
                                            "sortable": False,
                                            "filter": False,
                                            "resizable": True,
                                            "cellStyle": {
                                                "borderRight": "1px solid #d1d5db",
                                                "borderBottom": "1px solid #d1d5db",
                                            },
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
            html.Div(id="price_output", style={"display": "none"}),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Intrinsic vs Actual Share Price"),
                                dbc.CardBody(
                                    dcc.Graph(id="intrinsic-vs-price-chart"),
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
        ],
        fluid=True,
        className="py-4",
    )


# app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
# app.layout = build_layout()
layout = build_layout()


def register_callbacks(app):

    @app.callback(
        Output("intrinsic-kpi-value", "children"),
        Output("market-kpi-value", "children"),
        Output("shares-kpi-value", "children"),
        Input("price_output", "children"),
        Input("ticker", "value"),
        Input("intrinsic-last-data", "data"),
    )
    def update_kpi_cards(price_text, ticker, last_data):
        intrinsic_display = "$--"
        if isinstance(price_text, str):
            match = re.search(r"\$([0-9,]+(?:\.\d+)?)", price_text)
            if match:
                intrinsic_display = f"${match.group(1)}"

        market = _fetch_current_market_price(str(ticker or ""))
        market_display = "$--" if market is None else f"${market:,.2f}"

        shares_display = "--"
        try:
            if isinstance(last_data, dict) and last_data.get("shares"):
                shares = float(last_data["shares"])
                if shares >= 1_000_000_000:
                    shares_display = f"{shares / 1_000_000_000:.2f}B"
                elif shares >= 1_000_000:
                    shares_display = f"{shares / 1_000_000:.2f}M"
                else:
                    shares_display = f"{shares:,.0f}"
        except (TypeError, ValueError):
            shares_display = "--"

        return intrinsic_display, market_display, shares_display

    @app.callback(
        Output("discount-value", "children"),
        Output("terminal-value", "children"),
        Output("fcf-margin-value", "children"),
        Output("growth-start-value", "children"),
        Input("discount", "value"),
        Input("terminal", "value"),
        Input("fcf_margin", "value"),
        Input("growth_start", "value"),
    )
    def update_slider_labels(discount, terminal, fcf_margin, growth_start):
        discount_text = f"{float(discount or 0) * 100:.1f}%"
        terminal_text = f"{float(terminal or 0) * 100:.1f}%"
        fcf_margin_text = f"{float(fcf_margin or 0) * 100:.1f}%"
        growth_text = f"{float(growth_start or 0) * 100:.1f}%"
        return discount_text, terminal_text, fcf_margin_text, growth_text

    @app.callback(
        Output("discount", "disabled"),
        Input("discount-mode", "value"),
    )
    def toggle_discount_slider(discount_mode):
        return str(discount_mode or "auto").lower() == "auto"

    @app.callback(
        Output("fcf_margin", "value"),
        Input("db-pull-btn", "n_clicks"),
        Input("edgar-pull-btn", "n_clicks"),
        State("ticker", "value"),
        prevent_initial_call=True,
    )
    def sync_fcf_margin_with_pull(_db_clicks, _edgar_clicks, ticker):
        ticker_norm = _normalize_ticker(ticker)
        if not ticker_norm:
            return 0.15

        triggered = ctx.triggered_id

        if triggered == "db-pull-btn":
            payload_df, _shares, _ts = _fetch_cached_intrinsic_payload(ticker_norm)
            baseline = _baseline_fcf_margin_from_payload(payload_df)
            if baseline is None:
                cached = _fetch_cached_intrinsic_input(ticker_norm)
                if cached:
                    revenue = pd.to_numeric(cached.get("revenue"), errors="coerce")
                    fcf = pd.to_numeric(cached.get("fcf"), errors="coerce")
                    if pd.notna(revenue) and pd.notna(fcf) and float(revenue) != 0:
                        baseline = float(fcf) / float(revenue)
            slider_value = _clamp_fcf_margin_for_slider(baseline)
            return 0.15 if slider_value is None else slider_value

        if triggered == "edgar-pull-btn":
            try:
                cik = get_cik(ticker_norm)
                if not cik:
                    return 0.15
                facts = get_company_facts(cik)
                result = build_financials(facts)
                if result is None:
                    return 0.15
                df, _shares = result
                baseline = _baseline_fcf_margin_from_payload(df)
                slider_value = _clamp_fcf_margin_for_slider(baseline)
                return 0.15 if slider_value is None else slider_value
            except Exception:
                return 0.15

        return 0.15

    # =========================
    # CALLBACK
    # =========================
    @app.callback(
        Output("projection_chart", "figure"),
        Output("price_output", "children"),
        Output("intrinsic-payload-table", "rowData"),
        Output("intrinsic-payload-table", "columnDefs"),
        Output("intrinsic-forecast-table", "rowData"),
        Output("intrinsic-forecast-table", "columnDefs"),
        Output("intrinsic-calc-table", "rowData"),
        Output("intrinsic-calc-table", "columnDefs"),
        Output("intrinsic-vs-price-chart", "figure"),
        Output("intrinsic-last-data", "data"),
        Input("db-pull-btn", "n_clicks"),
        Input("edgar-pull-btn", "n_clicks"),
        Input("ticker", "value"),
        Input("discount", "value"),
        Input("discount-mode", "value"),
        Input("terminal", "value"),
        Input("fcf_margin", "value"),
        Input("growth_start", "value"),
        State("intrinsic-last-data", "data"),
        prevent_initial_call=True,
    )
    def update(
        _db_clicks,
        _edgar_clicks,
        ticker,
        discount,
        discount_mode_choice,
        terminal,
        fcf_margin,
        growth_start,
        last_data,
    ):

        def _split_projection_and_calc_tables(rows, columns):
            calc_col_ids = {
                "calc_label",
                "calc_value",
                "calc_value_text",
                "calc_formula",
                "calc_math",
            }

            primary_cols = [c for c in columns if c.get("field") not in calc_col_ids]

            # Forecast grid: remove columns 2-6 (indices 1-5), keep column 1 and columns 7+
            forecast_cols = (
                [primary_cols[0]] + primary_cols[6:]
                if len(primary_cols) > 6
                else [primary_cols[0]]
                if primary_cols
                else []
            )

            only_payload_rows = [r for r in rows if r.get("row_type") == "Payload"]
            forecast_terminal_rows = [
                r for r in rows if r.get("row_type") in {"Forecast", "Terminal"}
            ]
            all_non_calc_rows = [r for r in rows if r.get("row_type") != "Calc"]
            calc_rows = [r for r in rows if r.get("row_type") == "Calc"]

            calc_cols = [
                {
                    "headerName": _ag_header_name("calc_label"),
                    "field": "calc_label",
                    "cellStyle": {"fontWeight": 600},
                },
                {
                    "headerName": _ag_header_name("calc_value"),
                    "field": "calc_value",
                    "valueFormatter": {
                        "function": "params.value == null ? ((params.data && params.data.calc_label === 'Shares Outstanding') ? '÷' : '') : Number(params.value).toLocaleString('en-US', {style: 'currency', currency: 'USD', minimumFractionDigits: 2, maximumFractionDigits: 2})"
                    },
                    "cellClassRules": {
                        "custom-bottom-border": "params.data && ['Net Debt (Debt - Cash)', 'PV of Terminal Value'].includes(params.data.calc_label)",
                        "equity-value-bold": "params.data && ['Equity Value', 'Intrinsic Price Per Share'].includes(params.data.calc_label)",
                        "shares-division-sign": "params.data && params.data.calc_label === 'Shares Outstanding'",
                    },
                    "type": "rightAligned",
                },
                {
                    "headerName": _ag_header_name("calc_value_text"),
                    "field": "calc_value_text",
                    "type": "rightAligned",
                },
                {
                    "headerName": _ag_header_name("calc_formula"),
                    "field": "calc_formula",
                },
                {"headerName": _ag_header_name("calc_math"), "field": "calc_math"},
                {"headerName": _ag_header_name("source"), "field": "source"},
            ]

            return (
                only_payload_rows,
                primary_cols,
                forecast_terminal_rows,
                forecast_cols,
                all_non_calc_rows,
                primary_cols,
                calc_rows,
                calc_cols,
            )

        def _build_outputs_from_payload(
            payload_df: pd.DataFrame, shares: float, source: str
        ):
            payload = payload_df.copy()
            payload["end"] = pd.to_datetime(payload["end"], errors="coerce")
            payload["revenue"] = pd.to_numeric(payload["revenue"], errors="coerce")
            payload["fcf"] = pd.to_numeric(payload["fcf"], errors="coerce")
            payload = payload.dropna(subset=["end", "revenue", "fcf"]).sort_values(
                "end"
            )
            if payload.empty:
                return (
                    go.Figure(),
                    "Stored payload is empty. Click DB Pull or EDGAR Pull.",
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    _empty_intrinsic_vs_actual_figure(),
                    last_data,
                )

            revenue_avg = payload["revenue"].mean()
            if revenue_avg == 0 or pd.isna(revenue_avg):
                return (
                    go.Figure(),
                    "Cannot calculate FCF margin from zero revenue.",
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    _empty_intrinsic_vs_actual_figure(),
                    last_data,
                )

            baseline_fcf_margin = payload["fcf"].mean() / revenue_avg
            effective_fcf_margin = (
                float(fcf_margin)
                if fcf_margin is not None and np.isfinite(fcf_margin) and fcf_margin > 0
                else float(baseline_fcf_margin)
            )
            last_rev = float(payload["revenue"].iloc[-1])
            revs, fcfs = forecast_fcf(
                last_rev, effective_fcf_margin, growth_start=growth_start
            )
            terminal_assumptions = _estimate_terminal_assumptions(payload)
            effective_discount, discount_mode = _resolve_discount_rate(
                ticker_norm,
                float(shares),
                float(discount),
                str(discount_mode_choice or "auto"),
                float(terminal_assumptions["tax_rate"]),
                terminal_assumptions["debt"],
            )
            terminal_nopat = None
            if terminal_assumptions["nopat_margin"] is not None:
                terminal_nopat = float(revs[-1]) * float(
                    terminal_assumptions["nopat_margin"]
                )

            (
                enterprise_value,
                discounted,
                terminal_value,
                terminal_pv,
                _terminal_fcff_next,
                _reinvestment_rate,
            ) = dcf_value_with_terminal_adjustments(
                fcfs,
                effective_discount,
                terminal,
                terminal_nopat=terminal_nopat,
                terminal_roic=terminal_assumptions["roic"],
            )
            equity_value, net_debt = _equity_value_from_enterprise(
                float(enterprise_value),
                terminal_assumptions["cash"],
                terminal_assumptions["debt"],
            )
            price = float(equity_value) / float(shares)

            fig = go.Figure()
            years = np.arange(1, len(revs) + 1)
            fig.add_trace(go.Scatter(x=years, y=revs, name="Projected Revenue"))
            fig.add_trace(go.Bar(x=years, y=discounted, name="Discounted FCF"))
            fig.update_layout(
                title=f"{ticker_norm} 20-Year Projection ({source})",
                xaxis_title="Forecast Year",
                yaxis_title="USD",
                template="plotly_white",
            )
            fig.update_xaxes(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb")
            fig.update_yaxes(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb")

            table_data, table_cols = _build_projection_table(
                payload,
                revs,
                fcfs,
                discounted,
                float(shares),
                float(effective_discount),
                float(terminal),
                source,
                terminal_value_override=float(terminal_value),
                terminal_pv_override=float(terminal_pv),
                enterprise_value=float(enterprise_value),
                equity_value=float(equity_value),
                intrinsic_price=float(price),
                net_debt=float(net_debt) if net_debt is not None else None,
                terminal_fcff_next=float(_terminal_fcff_next),
                reinvestment_rate=_reinvestment_rate,
                discount_mode=discount_mode,
                cash_value=terminal_assumptions["cash"],
                debt_value=terminal_assumptions["debt"],
            )
            (
                payload_rows,
                payload_cols,
                forecast_rows,
                forecast_cols,
                _,
                _,
                calc_data,
                calc_cols,
            ) = _split_projection_and_calc_tables(table_data, table_cols)
            price_fig = _build_intrinsic_vs_actual_figure(
                ticker_norm,
                payload,
                float(shares),
                float(effective_discount),
                float(terminal),
                float(growth_start),
                float(price),
            )

            latest = payload.sort_values("end").iloc[-1]
            detail_parts = []
            if pd.notna(latest.get("tax_rate")):
                detail_parts.append(f"Tax rate: {float(latest['tax_rate']) * 100:.1f}%")
            if pd.notna(latest.get("roic")):
                detail_parts.append(f"ROIC: {float(latest['roic']) * 100:.1f}%")
            if pd.notna(latest.get("nopat")):
                detail_parts.append(f"NOPAT: ${float(latest['nopat']):,.0f}")
            detail_parts.append(
                f"Discount: {float(effective_discount) * 100:.1f}% ({discount_mode})"
            )
            if net_debt is not None:
                detail_parts.append(f"Net debt: ${float(net_debt):,.0f}")
            detail_note = f" | {' | '.join(detail_parts)}" if detail_parts else ""
            new_last_data = {
                "ticker": ticker_norm,
                "source": source,
                "shares": float(shares),
                "payload_rows": payload.to_dict("records"),
            }

            return (
                fig,
                f"Estimated Intrinsic Price for {ticker_norm}: ${price:,.2f} | FCF Margin: {effective_fcf_margin * 100:.1f}%{detail_note}",
                payload_rows,
                payload_cols,
                forecast_rows,
                forecast_cols,
                calc_data,
                calc_cols,
                price_fig,
                new_last_data,
            )

        def _empty_outputs(message: str):
            return (
                go.Figure(),
                message,
                [],
                [],
                [],
                [],
                [],
                [],
                _empty_intrinsic_vs_actual_figure(),
                last_data,
            )

        ticker_norm = _normalize_ticker(ticker)
        if not ticker_norm:
            return (
                go.Figure(),
                "Enter a ticker (for example: AAPL, MSFT, NVDA).",
                [],
                [],
                [],
                [],
                [],
                [],
                _empty_intrinsic_vs_actual_figure(),
                last_data,
            )

        if growth_start is None:
            growth_start = 0.10

        triggered = ctx.triggered_id

        if triggered in {
            "discount",
            "discount-mode",
            "terminal",
            "fcf_margin",
            "growth_start",
        }:
            if (
                isinstance(last_data, dict)
                and last_data.get("ticker") == ticker_norm
                and isinstance(last_data.get("payload_rows"), list)
                and last_data.get("shares")
            ):
                try:
                    payload_df = pd.DataFrame(last_data["payload_rows"])
                    shares = float(last_data["shares"])
                    source = str(last_data.get("source", "Loaded"))
                    (
                        fig,
                        msg,
                        payload_rows,
                        payload_cols,
                        forecast_rows,
                        forecast_cols,
                        calc_data,
                        calc_cols,
                        price_fig,
                        _new_last_data,
                    ) = _build_outputs_from_payload(payload_df, shares, source)
                    return (
                        fig,
                        msg,
                        payload_rows,
                        payload_cols,
                        forecast_rows,
                        forecast_cols,
                        calc_data,
                        calc_cols,
                        price_fig,
                        last_data,
                    )
                except Exception:
                    return _empty_outputs(
                        "Could not recompute from stored payload. Click DB Pull or EDGAR Pull."
                    )
            return _empty_outputs("Click DB Pull or EDGAR Pull to load data.")

        if triggered not in {"db-pull-btn", "edgar-pull-btn"}:
            return _empty_outputs("Click DB Pull or EDGAR Pull to load data.")

        if triggered == "db-pull-btn":
            cached_payload_df, cached_payload_shares, cached_payload_ts = (
                _fetch_cached_intrinsic_payload(ticker_norm)
            )
            if (
                cached_payload_df is not None
                and not cached_payload_df.empty
                and cached_payload_shares is not None
                and cached_payload_shares > 0
            ):
                try:
                    (
                        fig,
                        msg,
                        payload_rows,
                        payload_cols,
                        forecast_rows,
                        forecast_cols,
                        calc_data,
                        calc_cols,
                        price_fig,
                        new_last_data,
                    ) = _build_outputs_from_payload(
                        cached_payload_df,
                        float(cached_payload_shares),
                        "DB",
                    )
                    ts_note = (
                        f" Last cached payload pull: {cached_payload_ts}."
                        if cached_payload_ts
                        else ""
                    )
                    return (
                        fig,
                        f"{msg} (loaded from SQLite payload cache).{ts_note}",
                        payload_rows,
                        payload_cols,
                        forecast_rows,
                        forecast_cols,
                        calc_data,
                        calc_cols,
                        price_fig,
                        new_last_data,
                    )
                except Exception:
                    pass

            cached = _fetch_cached_intrinsic_input(ticker_norm)
            if not cached:
                return _empty_outputs(
                    f"No cached intrinsic row found for {ticker_norm}. Click EDGAR Pull first."
                )

            revenue = pd.to_numeric(cached.get("revenue"), errors="coerce")
            fcf = pd.to_numeric(cached.get("fcf"), errors="coerce")
            shares = pd.to_numeric(cached.get("shares_outstanding"), errors="coerce")
            if (
                pd.isna(revenue)
                or pd.isna(fcf)
                or pd.isna(shares)
                or revenue <= 0
                or shares <= 0
            ):
                return _empty_outputs(
                    f"Cached row for {ticker_norm} is incomplete. Click EDGAR Pull to refresh."
                )

            baseline_fcf_margin = float(fcf) / float(revenue)
            effective_fcf_margin = (
                float(fcf_margin)
                if fcf_margin is not None and np.isfinite(fcf_margin) and fcf_margin > 0
                else baseline_fcf_margin
            )
            last_rev = float(revenue)

            revs, fcfs = forecast_fcf(
                last_rev, effective_fcf_margin, growth_start=growth_start
            )
            try:
                effective_discount, discount_mode = _resolve_discount_rate(
                    ticker_norm,
                    float(shares),
                    float(discount),
                    str(discount_mode_choice or "auto"),
                    0.21,
                    None,
                )
                (
                    enterprise_value,
                    discounted,
                    terminal_value,
                    terminal_pv,
                    _terminal_fcff_next,
                    _reinvestment_rate,
                ) = dcf_value_with_terminal_adjustments(
                    fcfs,
                    effective_discount,
                    terminal,
                    terminal_nopat=None,
                    terminal_roic=None,
                )
            except ValueError as exc:
                return _empty_outputs(str(exc))

            equity_value, net_debt = _equity_value_from_enterprise(
                float(enterprise_value), None, None
            )
            price = equity_value / float(shares)

            fig = go.Figure()
            years = np.arange(1, len(revs) + 1)
            fig.add_trace(go.Scatter(x=years, y=revs, name="Projected Revenue"))
            fig.add_trace(go.Bar(x=years, y=discounted, name="Discounted FCF"))
            fig.update_layout(
                title=f"{ticker_norm} 20-Year Projection (DB)",
                xaxis_title="Forecast Year",
                yaxis_title="USD",
                template="plotly_white",
            )
            fig.update_xaxes(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb")
            fig.update_yaxes(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb")

            db_payload_df = pd.DataFrame(
                [
                    {
                        "end": cached.get("period_end"),
                        "revenue": revenue,
                        "fcf": fcf,
                    }
                ]
            )

            table_data, table_cols = _build_projection_table(
                db_payload_df,
                revs,
                fcfs,
                discounted,
                float(shares),
                float(effective_discount),
                float(terminal),
                "DB",
                terminal_value_override=float(terminal_value),
                terminal_pv_override=float(terminal_pv),
                enterprise_value=float(enterprise_value),
                equity_value=float(equity_value),
                intrinsic_price=float(price),
                net_debt=float(net_debt) if net_debt is not None else None,
                terminal_fcff_next=float(_terminal_fcff_next),
                reinvestment_rate=_reinvestment_rate,
                discount_mode=discount_mode,
                cash_value=None,
                debt_value=None,
            )
            (
                payload_rows,
                payload_cols,
                forecast_rows,
                forecast_cols,
                _,
                _,
                calc_data,
                calc_cols,
            ) = _split_projection_and_calc_tables(table_data, table_cols)
            price_fig = _build_intrinsic_vs_actual_figure(
                ticker_norm,
                db_payload_df,
                float(shares),
                float(effective_discount),
                float(terminal),
                float(growth_start),
                float(price),
            )

            pulled_at = str(cached.get("pulled_at_utc", "")).strip()
            ts_note = f" Last cached pull: {pulled_at}." if pulled_at else ""
            new_last_data = {
                "ticker": ticker_norm,
                "source": "DB",
                "shares": float(shares),
                "payload_rows": db_payload_df.to_dict("records"),
            }
            return (
                fig,
                f"Estimated Intrinsic Price for {ticker_norm}: ${price:,.2f} | FCF Margin: {effective_fcf_margin * 100:.1f}% | Discount: {effective_discount * 100:.1f}% ({discount_mode}) (loaded from SQLite cache).{ts_note}",
                payload_rows,
                payload_cols,
                forecast_rows,
                forecast_cols,
                calc_data,
                calc_cols,
                price_fig,
                new_last_data,
            )

        try:
            cik = get_cik(ticker_norm)
        except requests.RequestException as exc:
            return _empty_outputs(f"Failed to resolve ticker from SEC: {exc}")

        if not cik:
            return _empty_outputs(f"Ticker '{ticker_norm}' was not found in SEC lookup")

        try:
            facts = get_company_facts(cik)
        except requests.RequestException as exc:
            return _empty_outputs(f"Failed to load company facts from SEC: {exc}")

        result = build_financials(facts)

        if result is None:
            return _empty_outputs("Financial data unavailable")

        df, shares = result

        revenue_avg = df["revenue"].mean()
        if revenue_avg == 0 or pd.isna(revenue_avg):
            return _empty_outputs("Cannot calculate FCF margin from zero revenue")

        baseline_fcf_margin = df["fcf"].mean() / revenue_avg
        effective_fcf_margin = (
            float(fcf_margin)
            if fcf_margin is not None and np.isfinite(fcf_margin) and fcf_margin > 0
            else float(baseline_fcf_margin)
        )

        last_rev = df["revenue"].iloc[-1]

        revs, fcfs = forecast_fcf(
            last_rev, effective_fcf_margin, growth_start=growth_start
        )
        terminal_assumptions = _estimate_terminal_assumptions(df)
        effective_discount, discount_mode = _resolve_discount_rate(
            ticker_norm,
            float(shares),
            float(discount),
            str(discount_mode_choice or "auto"),
            float(terminal_assumptions["tax_rate"]),
            terminal_assumptions["debt"],
        )
        terminal_nopat = None
        if terminal_assumptions["nopat_margin"] is not None:
            terminal_nopat = float(revs[-1]) * float(
                terminal_assumptions["nopat_margin"]
            )

        try:
            (
                enterprise_value,
                discounted,
                terminal_value,
                terminal_pv,
                _terminal_fcff_next,
                _reinvestment_rate,
            ) = dcf_value_with_terminal_adjustments(
                fcfs,
                effective_discount,
                terminal,
                terminal_nopat=terminal_nopat,
                terminal_roic=terminal_assumptions["roic"],
            )
        except ValueError as exc:
            return _empty_outputs(str(exc))

        equity_value, net_debt = _equity_value_from_enterprise(
            float(enterprise_value),
            terminal_assumptions["cash"],
            terminal_assumptions["debt"],
        )
        price = equity_value / float(shares)

        # Plot
        fig = go.Figure()

        years = np.arange(1, len(revs) + 1)

        fig.add_trace(go.Scatter(x=years, y=revs, name="Projected Revenue"))

        fig.add_trace(go.Bar(x=years, y=discounted, name="Discounted FCF"))

        fig.update_layout(
            title=f"{ticker_norm} 20-Year Projection",
            xaxis_title="Forecast Year",
            yaxis_title="USD",
            template="plotly_white",
        )
        fig.update_xaxes(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb")
        fig.update_yaxes(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb")

        valuation_row = pd.DataFrame(
            [
                {
                    "ticker": ticker_norm,
                    "period_end": str(df["end"].iloc[-1]),
                    "discount_rate": float(effective_discount),
                    "terminal_growth": float(terminal),
                    "revenue": float(df["revenue"].iloc[-1]),
                    "fcf": float(df["fcf"].iloc[-1]),
                    "shares_outstanding": float(shares),
                    "enterprise_value": float(enterprise_value),
                    "intrinsic_price": float(price),
                    "pulled_at_utc": _utc_now_iso(),
                }
            ]
        )

        persistence_note = ""
        try:
            pull_ts = str(valuation_row["pulled_at_utc"].iloc[0])
            _save_parquet_snapshot(valuation_row)
            _save_payload_parquet_snapshot(ticker_norm, df, float(shares), pull_ts)
            _upsert_sqlite_intrinsic(valuation_row)
            _upsert_sqlite_payload_cache(ticker_norm, df, float(shares), pull_ts)
            persistence_note = " (saved to SQLite/Parquet)"
        except Exception as exc:
            persistence_note = f" (persistence warning: {exc})"

        table_data, table_cols = _build_projection_table(
            df,
            revs,
            fcfs,
            discounted,
            float(shares),
            float(effective_discount),
            float(terminal),
            "EDGAR",
            terminal_value_override=float(terminal_value),
            terminal_pv_override=float(terminal_pv),
            enterprise_value=float(enterprise_value),
            equity_value=float(equity_value),
            intrinsic_price=float(price),
            net_debt=float(net_debt) if net_debt is not None else None,
            terminal_fcff_next=float(_terminal_fcff_next),
            reinvestment_rate=_reinvestment_rate,
            discount_mode=discount_mode,
            cash_value=terminal_assumptions["cash"],
            debt_value=terminal_assumptions["debt"],
        )
        (
            payload_rows,
            payload_cols,
            forecast_rows,
            forecast_cols,
            _,
            _,
            calc_data,
            calc_cols,
        ) = _split_projection_and_calc_tables(table_data, table_cols)
        price_fig = _build_intrinsic_vs_actual_figure(
            ticker_norm,
            df,
            float(shares),
            float(effective_discount),
            float(terminal),
            float(growth_start),
            float(price),
        )

        latest = df.sort_values("end").iloc[-1]
        detail_parts = []
        if pd.notna(latest.get("tax_rate")):
            detail_parts.append(f"Tax rate: {float(latest['tax_rate']) * 100:.1f}%")
        if pd.notna(latest.get("roic")):
            detail_parts.append(f"ROIC: {float(latest['roic']) * 100:.1f}%")
        if pd.notna(latest.get("nopat")):
            detail_parts.append(f"NOPAT: ${float(latest['nopat']):,.0f}")
        detail_parts.append(
            f"Discount: {float(effective_discount) * 100:.1f}% ({discount_mode})"
        )
        if net_debt is not None:
            detail_parts.append(f"Net debt: ${float(net_debt):,.0f}")
        detail_note = f" | {' | '.join(detail_parts)}" if detail_parts else ""
        new_last_data = {
            "ticker": ticker_norm,
            "source": "EDGAR",
            "shares": float(shares),
            "payload_rows": df.to_dict("records"),
        }

        return (
            fig,
            f"Estimated Intrinsic Price for {ticker_norm}: ${price:,.2f} | FCF Margin: {effective_fcf_margin * 100:.1f}%{persistence_note}{detail_note}",
            payload_rows,
            payload_cols,
            forecast_rows,
            forecast_cols,
            calc_data,
            calc_cols,
            price_fig,
            new_last_data,
        )


register_callbacks(get_app())
