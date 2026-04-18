import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
import logging
import re
import xml.etree.ElementTree as ET

import requests
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import dash
import dash_bootstrap_components as dbc
from dash import get_app, dcc, html, Input, Output, dash_table, ctx
from storage_paths import DATA_DIR, CENTRAL_SQLITE_PATH, parquet_path


dash.register_page(__name__, path="/revenue", name="Revenue", order=4)

# -----------------------------
# SEC requires identity header
# -----------------------------
HEADERS = {"User-Agent": "EarlyWarningDashboard your_email@example.com"}
REVENUE_PARQUET_PATH = parquet_path("edgar_revenue")
REVENUE_SQLITE_PATH = CENTRAL_SQLITE_PATH
REVENUE_LOG_PATH = DATA_DIR / "edgar_revenue.log"
REVENUE_ANNUAL_PARQUET_PATH = parquet_path("edgar_revenue_annual")
REVENUE_ANNUAL_SQLITE_PATH = CENTRAL_SQLITE_PATH
PRICE_PARQUET_PATH = parquet_path("ticker_prices")
PRICE_SQLITE_PATH = CENTRAL_SQLITE_PATH

REVENUE_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
]


def _get_logger() -> logging.Logger:
    logger = logging.getLogger("edgar_revenue")
    if logger.handlers:
        return logger

    logger.setLevel(logging.WARNING)
    REVENUE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(REVENUE_LOG_PATH, encoding="utf-8")
    formatter = logging.Formatter(
        "%(asctime)sZ %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalize_price_frame(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    frame = df.copy()
    frame["ticker"] = ticker.upper()
    frame["pulled_at_utc"] = _utc_now_iso()
    frame["date"] = pd.to_datetime(frame["date"])
    for col in ["open", "high", "low", "close"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["volume"] = (
        pd.to_numeric(frame["volume"], errors="coerce").fillna(0).astype(int)
    )
    return frame[
        ["ticker", "date", "open", "high", "low", "close", "volume", "pulled_at_utc"]
    ]


def _fetch_yahoo_prices(ticker: str, period: str = "max") -> pd.DataFrame:
    raw = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if raw.empty:
        raise RuntimeError(f"Yahoo Finance returned no data for {ticker}.")
    raw = raw.reset_index()
    raw.columns = [c.lower() for c in raw.columns]
    raw = raw.rename(
        columns={"date": "date", "stock splits": "splits", "capital gains": "capgains"}
    )
    raw["date"] = pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d")
    return _normalize_price_frame(
        raw[["date", "open", "high", "low", "close", "volume"]], ticker
    )


def _save_price_parquet(df: pd.DataFrame, path: Path = PRICE_PARQUET_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_parquet(path)
        existing["date"] = pd.to_datetime(existing["date"])
        df = pd.concat([existing, df]).drop_duplicates(
            subset=["ticker", "date"], keep="last"
        )
    df.to_parquet(path, index=False)


def _upsert_price_sqlite(
    df: pd.DataFrame, path: Path = PRICE_SQLITE_PATH
) -> tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ticker_prices (
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                pulled_at_utc TEXT NOT NULL,
                PRIMARY KEY (ticker, date)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tp_ticker ON ticker_prices(ticker)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tp_date ON ticker_prices(date)")

        rows = [
            (
                r.ticker,
                r.date.strftime("%Y-%m-%d") if hasattr(r.date, "strftime") else r.date,
                r.open,
                r.high,
                r.low,
                r.close,
                r.volume,
                r.pulled_at_utc,
            )
            for r in df.itertuples(index=False)
        ]
        before = conn.execute("SELECT COUNT(*) FROM ticker_prices").fetchone()[0]
        conn.executemany(
            """
            INSERT OR REPLACE INTO ticker_prices
            (ticker, date, open, high, low, close, volume, pulled_at_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        after = conn.execute("SELECT COUNT(*) FROM ticker_prices").fetchone()[0]
    return len(rows), after - before


def _load_price_series(ticker: str, path: Path = PRICE_SQLITE_PATH) -> pd.DataFrame:
    if path.exists():
        with sqlite3.connect(path) as conn:
            df = pd.read_sql_query(
                "SELECT * FROM ticker_prices WHERE ticker = ? ORDER BY date",
                conn,
                params=(ticker.upper(),),
            )
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                return df

    if PRICE_PARQUET_PATH.exists():
        df = pd.read_parquet(PRICE_PARQUET_PATH)
        df = df[df["ticker"] == ticker.upper()].copy()
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date")

    return pd.DataFrame()


def _normalize_ticker(ticker: str) -> str:
    return str(ticker or "").strip().upper()


def _format_refreshed(ticker: str) -> str:
    est = timezone(timedelta(hours=-5), name="EST")
    local_dt = datetime.now(est)
    date_str = local_dt.strftime("%m-%d-%Y %I:%M %p")
    return f"{_normalize_ticker(ticker)} last refreshed (EST): {date_str}"


def _save_parquet_snapshot(data: pd.DataFrame, parquet_path: Path) -> Path:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    if parquet_path.exists():
        try:
            existing = pd.read_parquet(parquet_path)
        except Exception:
            existing = pd.DataFrame()
        if existing is not None and not existing.empty:
            data = pd.concat([existing, data], ignore_index=True)
            data = data.drop_duplicates(subset=["ticker", "end"])
    try:
        data.to_parquet(parquet_path, index=False)
    except Exception as exc:
        raise RuntimeError(
            "Failed to write Parquet. Install a parquet engine: pip install pyarrow"
        ) from exc
    return parquet_path


def _upsert_sqlite_revenue(
    data: pd.DataFrame, db_path: Path = REVENUE_SQLITE_PATH
) -> tuple[int, int]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS revenue_series (
                ticker TEXT NOT NULL,
                period_end TEXT NOT NULL,
                revenue REAL,
                form TEXT,
                period_type TEXT,
                fp TEXT,
                fy INTEGER,
                pulled_at_utc TEXT NOT NULL,
                PRIMARY KEY (ticker, period_end)
            )
            """
        )
        existing_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(revenue_series)")
        }
        if "form" not in existing_cols:
            conn.execute("ALTER TABLE revenue_series ADD COLUMN form TEXT")
        if "period_type" not in existing_cols:
            conn.execute("ALTER TABLE revenue_series ADD COLUMN period_type TEXT")
        if "fp" not in existing_cols:
            conn.execute("ALTER TABLE revenue_series ADD COLUMN fp TEXT")
        if "fy" not in existing_cols:
            conn.execute("ALTER TABLE revenue_series ADD COLUMN fy INTEGER")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rev_ticker ON revenue_series(ticker)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rev_period_end ON revenue_series(period_end)"
        )

        rows = [
            (
                row.ticker,
                str(row.end),
                float(row.Revenue),
                getattr(row, "form", None),
                getattr(row, "period_type", None),
                getattr(row, "fp", None),
                getattr(row, "fy", None),
                row.pulled_at_utc,
            )
            for row in data.itertuples(index=False)
        ]
        before = conn.execute("SELECT COUNT(*) FROM revenue_series").fetchone()[0]
        conn.executemany(
            """
            INSERT OR REPLACE INTO revenue_series
            (ticker, period_end, revenue, form, period_type, fp, fy, pulled_at_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        after = conn.execute("SELECT COUNT(*) FROM revenue_series").fetchone()[0]
    return len(rows), after - before


def _upsert_sqlite_revenue_annual(
    data: pd.DataFrame, db_path: Path = REVENUE_ANNUAL_SQLITE_PATH
) -> tuple[int, int]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS revenue_annual (
                ticker TEXT NOT NULL,
                period_end TEXT NOT NULL,
                revenue REAL,
                form TEXT,
                period_type TEXT,
                fp TEXT,
                fy INTEGER,
                pulled_at_utc TEXT NOT NULL,
                PRIMARY KEY (ticker, period_end)
            )
            """
        )
        existing_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(revenue_annual)")
        }
        if "form" not in existing_cols:
            conn.execute("ALTER TABLE revenue_annual ADD COLUMN form TEXT")
        if "period_type" not in existing_cols:
            conn.execute("ALTER TABLE revenue_annual ADD COLUMN period_type TEXT")
        if "fp" not in existing_cols:
            conn.execute("ALTER TABLE revenue_annual ADD COLUMN fp TEXT")
        if "fy" not in existing_cols:
            conn.execute("ALTER TABLE revenue_annual ADD COLUMN fy INTEGER")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rev_annual_ticker ON revenue_annual(ticker)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rev_annual_period_end ON revenue_annual(period_end)"
        )

        rows = [
            (
                row.ticker,
                str(row.end),
                float(row.Revenue),
                getattr(row, "form", None),
                getattr(row, "period_type", None),
                getattr(row, "fp", None),
                getattr(row, "fy", None),
                row.pulled_at_utc,
            )
            for row in data.itertuples(index=False)
        ]
        before = conn.execute("SELECT COUNT(*) FROM revenue_annual").fetchone()[0]
        conn.executemany(
            """
            INSERT OR REPLACE INTO revenue_annual
            (ticker, period_end, revenue, form, period_type, fp, fy, pulled_at_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        after = conn.execute("SELECT COUNT(*) FROM revenue_annual").fetchone()[0]
    return len(rows), after - before


def _fetch_cached_revenue(
    ticker: str, db_path: Path = REVENUE_SQLITE_PATH
) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    sql = """
    SELECT
        period_end AS end,
        revenue AS Revenue,
        form,
        period_type,
        fp,
        fy
    FROM revenue_series
    WHERE ticker = ?
    ORDER BY period_end
    """
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=(ticker,))


def _fetch_cached_revenue_annual(
    ticker: str, db_path: Path = REVENUE_ANNUAL_SQLITE_PATH
) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    sql = """
    SELECT
        period_end AS end,
        revenue AS Revenue,
        form,
        period_type,
        fp,
        fy
    FROM revenue_annual
    WHERE ticker = ?
    ORDER BY period_end
    """
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=(ticker,))


def _normalize_annual_series(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    clean = df.copy()
    clean["end"] = pd.to_datetime(clean["end"], errors="coerce")
    clean = clean.dropna(subset=["end"])
    clean = clean.sort_values("end")
    clean["year"] = clean["end"].dt.year
    # Keep the latest period end per year to avoid quarterly bleed-in
    clean = clean.groupby("year", as_index=False).tail(1)
    clean = clean.drop(columns=["year"])
    return clean


def _derive_q4_from_annual(
    quarterly_df: pd.DataFrame, annual_df: pd.DataFrame
) -> pd.DataFrame:
    if (
        quarterly_df is None
        or quarterly_df.empty
        or annual_df is None
        or annual_df.empty
    ):
        return quarterly_df if quarterly_df is not None else pd.DataFrame()

    q = quarterly_df.copy()
    a = annual_df.copy()
    q["end"] = pd.to_datetime(q["end"], errors="coerce")
    a["end"] = pd.to_datetime(a["end"], errors="coerce")
    q = q.dropna(subset=["end"])
    a = a.dropna(subset=["end"])

    derived_rows = []
    for row in a.itertuples(index=False):
        end = getattr(row, "end", None)
        revenue = getattr(row, "Revenue", None)
        if end is None or pd.isna(end) or revenue is None or pd.isna(revenue):
            continue

        window_start = end - pd.Timedelta(days=370)
        prior_quarters = (
            q[(q["end"] > window_start) & (q["end"] < end)].sort_values("end").tail(3)
        )
        if len(prior_quarters) != 3:
            continue

        q4_value = revenue - prior_quarters["Revenue"].sum()
        if pd.isna(q4_value):
            continue

        derived_rows.append(
            {
                "end": end,
                "Revenue": q4_value,
                "form": "10-K (derived Q4)",
                "period_type": "Q",
                "fp": "Q4",
                "fy": pd.to_datetime(end, errors="coerce").year,
            }
        )

    if derived_rows:
        q = pd.concat([q, pd.DataFrame(derived_rows)], ignore_index=True)
        q = q.drop_duplicates(subset=["end"], keep="last")

    return q.sort_values("end")


def _normalize_quarterly_series(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    series = df.copy()
    if "form" in series.columns:
        quarterly = series[series["form"] == "10-Q"].copy()
        annual = series[series["form"] == "10-K"].copy()
    else:
        quarterly = series.copy()
        annual = pd.DataFrame()

    if not annual.empty:
        quarterly = _derive_q4_from_annual(quarterly, annual)

    return quarterly.sort_values("end")


def _clean_cached_quarterly(
    cached_df: pd.DataFrame, annual_df: pd.DataFrame
) -> pd.DataFrame:
    if cached_df is None or cached_df.empty:
        return pd.DataFrame()

    clean = cached_df.copy()
    clean["end"] = pd.to_datetime(clean["end"], errors="coerce")
    clean = clean.dropna(subset=["end"])

    if annual_df is not None and not annual_df.empty:
        annual_ends = pd.to_datetime(annual_df["end"], errors="coerce").dropna()
        if not annual_ends.empty:
            clean = clean[~clean["end"].isin(annual_ends)]
        clean = _derive_q4_from_annual(clean, annual_df)

    return clean.sort_values("end")


def _infer_fiscal_year_end_month(annual_df: pd.DataFrame) -> int | None:
    if annual_df is None or annual_df.empty:
        return None
    end_dt = pd.to_datetime(annual_df["end"], errors="coerce").dropna()
    if end_dt.empty:
        return None
    latest_end = end_dt.max()
    return int(latest_end.month)


def _add_fiscal_columns(
    df_display: pd.DataFrame, fy_end_month: int | None
) -> pd.DataFrame:
    if df_display is None or df_display.empty:
        return df_display

    end_dt = pd.to_datetime(df_display["end"], errors="coerce")
    # Default to calendar mapping first.
    df_display["Fiscal Year"] = end_dt.dt.year
    df_display["Fiscal Quarter"] = end_dt.dt.quarter.map(lambda q: f"Q{q}")

    if fy_end_month is not None:
        # Shift a few days earlier so quarter-ends that land in the first days
        # of the next month still map to the intended fiscal quarter.
        adj_end = end_dt - pd.Timedelta(days=7)
        fy_start_month = (fy_end_month % 12) + 1
        end_month = adj_end.dt.month
        offset = (end_month - fy_start_month) % 12
        df_display["Fiscal Quarter"] = (offset // 3 + 1).map(lambda q: f"Q{q}")
        df_display["Fiscal Year"] = adj_end.dt.year + (end_month > fy_end_month)

    df_display["Fiscal Year"] = pd.to_numeric(
        df_display["Fiscal Year"], errors="coerce"
    ).astype("Int64")

    # Prefer SEC metadata when available.
    if "fy" in df_display.columns:
        fy_series = pd.to_numeric(df_display["fy"], errors="coerce").astype("Int64")
        end_year = end_dt.dt.year.astype("Int64")
        mask = fy_series.notna() & end_year.notna()
        # Only accept FY values that are plausible for the period end year.
        # (Typically FY is end_year or end_year + 1.)
        mask = mask & ((fy_series == end_year) | (fy_series == (end_year + 1)))
        if mask.any():
            df_display.loc[mask, "Fiscal Year"] = fy_series[mask].to_numpy()
    if "fp" in df_display.columns:
        fp_series = df_display["fp"].astype(str).str.upper()
        valid_fp = fp_series.isin(["Q1", "Q2", "Q3", "Q4"])
        df_display.loc[valid_fp, "Fiscal Quarter"] = fp_series[valid_fp]
        df_display.loc[fp_series == "FY", "Fiscal Quarter"] = "FY"
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
        lookup[(row["Fiscal Year"], row["Fiscal Quarter"])] = row["Revenue"]

    for idx, row in series.iterrows():
        key = (row["Fiscal Year"], row["Fiscal Quarter"])
        if key[0] is None or key[1] is None:
            continue
        prev_key = (key[0] - 1, key[1])
        prev_val = lookup.get(prev_key)
        if prev_val is None or pd.isna(prev_val) or pd.isna(row["Revenue"]):
            continue
        if prev_val == 0:
            continue
        yoy_vals[idx] = (row["Revenue"] / prev_val - 1) * 100

    return pd.Series(yoy_vals)


def _load_or_fetch_prices(ticker: str, allow_fetch: bool = True) -> pd.DataFrame:
    if not ticker:
        return pd.DataFrame()
    try:
        df = _load_price_series(ticker)
    except Exception:
        df = pd.DataFrame()
    if df is not None and not df.empty:
        return df
    if not allow_fetch:
        return pd.DataFrame()
    try:
        df = _fetch_yahoo_prices(ticker)
    except Exception:
        return pd.DataFrame()
    try:
        _save_price_parquet(df)
        _upsert_price_sqlite(df)
    except Exception:
        pass
    try:
        return _load_price_series(ticker)
    except Exception:
        return df


# -----------------------------
# Get company CIK from ticker
# -----------------------------
def get_cik(ticker):
    url = "https://www.sec.gov/files/company_tickers.json"
    data = requests.get(url, headers=HEADERS).json()

    for item in data.values():
        if item["ticker"].lower() == ticker.lower():
            return str(item["cik_str"]).zfill(10)

    return None


# -----------------------------
# Pull XBRL company facts
# -----------------------------
def get_company_facts(cik):
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    return requests.get(url, headers=HEADERS).json()


def get_company_submissions(cik: str) -> dict:
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    return requests.get(url, headers=HEADERS).json()


def _latest_filing_accession(
    submissions: dict, forms: tuple[str, ...]
) -> tuple[str | None, str | None]:
    recent = submissions.get("filings", {}).get("recent", {})
    forms_list = recent.get("form", [])
    accession_list = recent.get("accessionNumber", [])
    for form, accession in zip(forms_list, accession_list):
        if form in forms:
            return accession, form
    return None, None


def _fetch_filing_index(cik: str, accession: str) -> dict | None:
    acc_no = accession.replace("-", "")
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no}"
    url = f"{base}/index.json"
    try:
        return requests.get(url, headers=HEADERS).json()
    except Exception:
        return None


def _select_instance_filename(index_json: dict | None) -> str | None:
    if not index_json:
        return None
    items = index_json.get("directory", {}).get("item", [])
    if not items:
        return None

    skip_suffixes = ("_cal.xml", "_def.xml", "_lab.xml", "_pre.xml")
    skip_names = {"filesummary.xml", "metalinks.json"}

    candidates = []
    for item in items:
        name = item.get("name", "")
        if not name.lower().endswith(".xml"):
            continue
        if name.lower() in skip_names:
            continue
        if any(name.lower().endswith(s) for s in skip_suffixes):
            continue
        candidates.append(name)

    if not candidates:
        return None

    # Prefer _htm.xml inline instance, then plain ticker-date.xml
    for name in candidates:
        if name.lower().endswith("_htm.xml"):
            return name
    for name in candidates:
        if re.match(r".+-\d{8}\.xml$", name, re.IGNORECASE):
            return name
    return candidates[0]


def _fetch_instance_xml(cik: str, accession: str, filename: str) -> str | None:
    acc_no = accession.replace("-", "")
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no}"
    url = f"{base}/{filename}"
    try:
        return requests.get(url, headers=HEADERS).text
    except Exception:
        return None


# -----------------------------
# Extract quarterly data
# -----------------------------
def is_quarter(start, end) -> bool:
    if start is None or end is None:
        return False
    days = (end - start).days
    return 80 <= days <= 100


def extract_series(facts, tag):
    try:
        usgaap = facts["facts"]["us-gaap"][tag]["units"]["USD"]
        df = pd.DataFrame(usgaap)

        df = df[df["form"].isin(["10-Q", "10-K"])]
        if "start" in df.columns:
            df["start"] = pd.to_datetime(df["start"], errors="coerce")
            df["end"] = pd.to_datetime(df["end"], errors="coerce")
            df = df[df["start"].notna() & df["end"].notna()]
            # Only enforce quarter-length windows on 10-Q rows.
            is_q = df["form"] == "10-Q"
            if "fp" in df.columns:
                fp_q = df["fp"].astype(str).str.upper().isin(["Q1", "Q2", "Q3", "Q4"])
                df = pd.concat([df[~is_q], df[is_q & fp_q]], ignore_index=True)
            else:
                df = pd.concat(
                    [
                        df[~is_q],
                        df[
                            is_q
                            & df.apply(
                                lambda row: is_quarter(row["start"], row["end"]),
                                axis=1,
                            )
                        ],
                    ],
                    ignore_index=True,
                )
        if "filed" in df.columns:
            df = df.sort_values(["end", "filed"])
            df = df.drop_duplicates(subset=["end"], keep="last")
        else:
            df = df.sort_values("end").drop_duplicates(subset=["end"], keep="last")

        df["period_type"] = df["form"].map({"10-Q": "Q", "10-K": "FY"}).fillna("")
        cols = ["end", "val", "form", "period_type"]
        if "fp" in df.columns:
            cols.append("fp")
        if "fy" in df.columns:
            cols.append("fy")
        return df[cols].rename(columns={"val": tag})
    except Exception:
        return pd.DataFrame()


def _extract_revenue(facts) -> tuple[pd.DataFrame, str | None]:
    for tag in REVENUE_TAGS:
        df = extract_series(facts, tag)
        if df is not None and not df.empty:
            df = df.rename(columns={tag: "Revenue"})
            return df, tag
    return pd.DataFrame(), None


# -----------------------------
# Build revenue dataframe
# -----------------------------
def build_revenue(ticker, use_cache: bool = True, cache_only: bool = False):
    logger = _get_logger()

    ticker_norm = _normalize_ticker(ticker)
    if not ticker_norm:
        return None, None, None

    if use_cache:
        cached = _fetch_cached_revenue(ticker_norm)
        cached_annual = _normalize_annual_series(
            _fetch_cached_revenue_annual(ticker_norm)
        )
        if cached is not None and not cached.empty:
            cleaned = _clean_cached_quarterly(cached, cached_annual)
            return cleaned.tail(60), "cache", cached_annual
        if cache_only:
            return pd.DataFrame(), "cache", cached_annual

    cik = get_cik(ticker_norm)
    if not cik:
        return None, None, None

    facts = get_company_facts(cik)
    df, tag_used = _extract_revenue(facts)
    if df is None or df.empty:
        logger.warning("no revenue tag found for ticker=%s", ticker_norm)
        return pd.DataFrame(), tag_used, pd.DataFrame()

    df = df.sort_values("end")
    annual_df = df[df["form"] == "10-K"].copy()
    annual_df = _normalize_annual_series(annual_df)
    df = _normalize_quarterly_series(df)
    df = df.tail(60)

    persist_df = df.copy()
    persist_df["ticker"] = ticker_norm
    persist_df["pulled_at_utc"] = _utc_now_iso()
    persist_df["end"] = pd.to_datetime(
        persist_df["end"], errors="coerce"
    ).dt.date.astype(str)
    persist_df = persist_df.copy()

    annual_persist = annual_df.copy()
    annual_persist["ticker"] = ticker_norm
    annual_persist["pulled_at_utc"] = _utc_now_iso()
    annual_persist["end"] = pd.to_datetime(
        annual_persist["end"], errors="coerce"
    ).dt.date.astype(str)
    annual_persist = annual_persist.copy()

    try:
        _save_parquet_snapshot(persist_df, REVENUE_PARQUET_PATH)
        _upsert_sqlite_revenue(persist_df, REVENUE_SQLITE_PATH)
        if not annual_persist.empty:
            _save_parquet_snapshot(annual_persist, REVENUE_ANNUAL_PARQUET_PATH)
            _upsert_sqlite_revenue_annual(annual_persist, REVENUE_ANNUAL_SQLITE_PATH)
    except Exception:
        logger.exception("persistence failed for ticker=%s", ticker_norm)

    return df, tag_used, annual_df


# -----------------------------
# Create Plotly Figure
# -----------------------------
def create_figure(df, ticker, refreshed_text: str = "", tag_used: str | None = None):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["end"],
            y=df["Revenue"],
            name="Revenue",
            mode="lines+markers",
        )
    )

    title = f"{ticker.upper()} Revenue (Quarterly)"
    if tag_used:
        title = f"{title} - {tag_used}"
    if refreshed_text:
        title = f"{title} - {refreshed_text}"

    fig.update_layout(
        title=title,
        xaxis_title="Quarter",
        yaxis=dict(title="USD"),
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        height=650,
    )
    fig.update_xaxes(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb")
    fig.update_yaxes(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb")

    return fig


def _build_growth_momentum_figure(
    df: pd.DataFrame, fy_end_month: int | None
) -> go.Figure:
    fig = go.Figure()
    if df is None or df.empty:
        fig.update_layout(
            title="Growth Momentum",
            template="plotly_white",
        )
        return fig

    series = df.copy()
    series = series.sort_values("end").reset_index(drop=True)
    series["YoY %"] = _compute_quarterly_yoy(series, fy_end_month)
    series["YoY MA (4Q) %"] = series["YoY %"].rolling(4).mean()

    fig.add_trace(
        go.Scatter(
            x=series["end"],
            y=series["YoY %"],
            name="Same Quarter Last Year (YoY) %",
            mode="lines+markers",
            line=dict(color="#2563eb"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=series["end"],
            y=series["YoY MA (4Q) %"],
            name="YoY Moving Avg (4Q) %",
            mode="lines",
            line=dict(color="#10b981", dash="dash"),
        )
    )
    fig.update_layout(
        title="Growth Momentum (YoY vs YoY Moving Avg)",
        xaxis_title="Quarter",
        yaxis=dict(title="Growth %"),
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        height=420,
    )
    fig.update_xaxes(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb")
    fig.update_yaxes(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb")
    return fig


def _build_annual_figure(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    if df is None or df.empty:
        fig.update_layout(
            title="Annual Revenue (10-K)",
            template="plotly_white",
        )
        return fig

    series = df.copy().sort_values("end")
    fig.add_trace(
        go.Bar(
            x=series["end"],
            y=series["Revenue"],
            name="Annual Revenue",
            marker_color="#f97316",
        )
    )
    fig.update_layout(
        title=f"{ticker.upper()} Annual Revenue (10-K)",
        xaxis_title="Fiscal Year End",
        yaxis=dict(title="USD"),
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        height=420,
    )
    fig.update_xaxes(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb")
    fig.update_yaxes(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb")
    return fig


def _build_annual_growth_figure(
    df: pd.DataFrame,
    window_years: int | None,
    price_df: pd.DataFrame,
) -> go.Figure:
    fig = go.Figure()
    if df is None or df.empty:
        fig.update_layout(
            title="Annual Growth Momentum",
            template="plotly_white",
        )
        return fig

    series = df.copy()
    series["end"] = pd.to_datetime(series["end"], errors="coerce")
    series = series.dropna(subset=["end"]).sort_values("end")
    series["YoY %"] = series["Revenue"].pct_change() * 100
    rolling_years = window_years if window_years and window_years > 0 else 3
    series[f"YoY MA ({rolling_years}Y) %"] = (
        series["YoY %"].rolling(rolling_years).mean()
    )
    if window_years and window_years > 0:
        # Keep one extra year to compute YoY for the first visible point
        series = series.tail(window_years + 1)

    fig.add_trace(
        go.Scatter(
            x=series["end"],
            y=series["YoY %"],
            name="Annual YoY %",
            mode="lines+markers",
            line=dict(color="#f97316"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=series["end"],
            y=series[f"YoY MA ({rolling_years}Y) %"],
            name=f"YoY Moving Avg ({rolling_years}Y) %",
            mode="lines",
            line=dict(color="#ea580c", dash="dash"),
        )
    )
    if price_df is not None and not price_df.empty:
        price_series = price_df.copy()
        price_series["date"] = pd.to_datetime(price_series["date"], errors="coerce")
        price_series = price_series.dropna(subset=["date"]).sort_values("date")
        price_series["year"] = price_series["date"].dt.year
        year_close = price_series.groupby("year", as_index=False).tail(1)
        series["year"] = series["end"].dt.year
        series["Price"] = series["year"].map(
            year_close.set_index("year")["close"].to_dict()
        )
        fig.add_trace(
            go.Scatter(
                x=series["end"],
                y=series["Price"],
                name="Year-End Price",
                mode="lines+markers",
                line=dict(color="#0f172a"),
                yaxis="y2",
            )
        )
    fig.update_layout(
        title="Annual Growth Momentum (YoY vs Moving Avg)",
        xaxis_title="Fiscal Year End",
        yaxis=dict(title="Growth %"),
        yaxis2=dict(
            title="Price",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        height=420,
    )
    fig.update_xaxes(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb")
    fig.update_yaxes(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb")
    return fig


OCF_TAGS = [
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
]


def fetch_ocf_annual(cik: str) -> pd.DataFrame:
    try:
        facts = get_company_facts(cik)
    except Exception:
        return pd.DataFrame()
    for tag in OCF_TAGS:
        try:
            rows = facts["facts"]["us-gaap"][tag]["units"]["USD"]
            df = pd.DataFrame(rows)
            df = df[df["form"] == "10-K"]
            if df.empty:
                continue
            if "filed" in df.columns:
                df = df.sort_values(["end", "filed"]).drop_duplicates(
                    subset=["end"], keep="last"
                )
            else:
                df = df.sort_values("end").drop_duplicates(subset=["end"], keep="last")
            df["end"] = pd.to_datetime(df["end"], errors="coerce")
            df = df.dropna(subset=["end"])
            return df[["end", "val"]].rename(columns={"val": "OCF"})
        except Exception:
            continue
    return pd.DataFrame()


def _build_ocf_margin_figure(
    annual_df: pd.DataFrame, ocf_df: pd.DataFrame, ticker: str
) -> go.Figure:
    fig = go.Figure()
    if annual_df is None or annual_df.empty:
        fig.update_layout(title="Revenue vs OCF (Margin)", template="plotly_white")
        return fig

    rev = annual_df.copy()
    rev["end"] = pd.to_datetime(rev["end"], errors="coerce")
    rev = rev.dropna(subset=["end"]).sort_values("end")

    if ocf_df is not None and not ocf_df.empty:
        ocf = ocf_df.copy()
        ocf["end"] = pd.to_datetime(ocf["end"], errors="coerce")
        merged = pd.merge_asof(
            rev.sort_values("end"),
            ocf.sort_values("end"),
            on="end",
            tolerance=pd.Timedelta(days=35),
            direction="nearest",
        )
    else:
        merged = rev.copy()
        merged["OCF"] = None

    merged["OCF Margin %"] = merged["OCF"] / merged["Revenue"] * 100

    fig.add_trace(
        go.Bar(
            x=merged["end"],
            y=merged["Revenue"],
            name="Revenue",
            marker_color="#f97316",
            opacity=0.85,
        )
    )
    if merged["OCF"].notna().any():
        fig.add_trace(
            go.Bar(
                x=merged["end"],
                y=merged["OCF"],
                name="Operating Cash Flow",
                marker_color="#10b981",
                opacity=0.85,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=merged["end"],
                y=merged["OCF Margin %"],
                name="OCF Margin % (OCF / Revenue)",
                mode="lines+markers",
                line=dict(color="#7c3aed", dash="dot"),
                yaxis="y2",
            )
        )
    fig.update_layout(
        title=f"{ticker.upper()} Revenue vs OCF with Margin (Annual)",
        xaxis_title="Fiscal Year End",
        yaxis=dict(title="USD"),
        yaxis2=dict(title="OCF Margin %", overlaying="y", side="right", showgrid=False),
        barmode="group",
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        height=420,
    )
    fig.update_xaxes(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb")
    fig.update_yaxes(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb")
    return fig


def _build_quality_figure(
    annual_df: pd.DataFrame, ocf_df: pd.DataFrame, ticker: str
) -> go.Figure:
    fig = go.Figure()
    if annual_df is None or annual_df.empty:
        fig.update_layout(
            title="Revenue vs Operating Cash Flow", template="plotly_white"
        )
        return fig

    rev = annual_df.copy()
    rev["end"] = pd.to_datetime(rev["end"], errors="coerce")
    rev = rev.dropna(subset=["end"]).sort_values("end")

    if ocf_df is not None and not ocf_df.empty:
        ocf = ocf_df.copy()
        ocf["end"] = pd.to_datetime(ocf["end"], errors="coerce")
        merged = pd.merge_asof(
            rev.sort_values("end"),
            ocf.sort_values("end"),
            on="end",
            tolerance=pd.Timedelta(days=35),
            direction="nearest",
        )
    else:
        merged = rev.copy()
        merged["OCF"] = None

    merged["Accrual Ratio"] = (
        (merged["Revenue"] - merged["OCF"]) / merged["Revenue"] * 100
    )

    fig.add_trace(
        go.Bar(
            x=merged["end"],
            y=merged["Revenue"],
            name="Revenue",
            marker_color="#f97316",
            opacity=0.85,
        )
    )
    if merged["OCF"].notna().any():
        fig.add_trace(
            go.Bar(
                x=merged["end"],
                y=merged["OCF"],
                name="Operating Cash Flow",
                marker_color="#10b981",
                opacity=0.85,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=merged["end"],
                y=merged["Accrual Ratio"],
                name="Accrual Ratio % (Rev - OCF) / Rev",
                mode="lines+markers",
                line=dict(color="#ef4444", dash="dot"),
                yaxis="y2",
            )
        )
    fig.update_layout(
        title=f"{ticker.upper()} Revenue vs Operating Cash Flow (Annual)",
        xaxis_title="Fiscal Year End",
        yaxis=dict(title="USD"),
        yaxis2=dict(
            title="Accrual Ratio %", overlaying="y", side="right", showgrid=False
        ),
        barmode="group",
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        height=420,
    )
    fig.update_xaxes(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb")
    fig.update_yaxes(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb")
    return fig


def _extract_segment_revenue(facts) -> pd.DataFrame:
    def _clean_segment_label(raw: str) -> str | None:
        if raw is None:
            return None
        label = str(raw).strip()
        if not label:
            return None
        # Pull the member part when dimension strings include '='.
        if "=" in label:
            label = label.split("=")[-1]
        # Drop namespace prefix.
        if ":" in label:
            label = label.split(":")[-1]
        # Remove common suffix and prettify camel case.
        if label.endswith("Member"):
            label = label[: -len("Member")]
        label = label.replace("_", " ").replace("-", " ")
        label = re.sub(r"(?<!^)(?=[A-Z])", " ", label)
        label = " ".join(label.split())
        return label or None

    def _segment_to_label_and_axis(value) -> tuple[str | None, str | None]:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None, None
        if isinstance(value, str):
            return _clean_segment_label(value), None
        if isinstance(value, dict):
            # Common shape: {"dimension": {"AxisName": "ns:Member"}}
            if "dimension" in value and isinstance(value.get("dimension"), dict):
                dim_map = value.get("dimension", {})
                for axis, member in dim_map.items():
                    label = _clean_segment_label(member)
                    axis_label = _clean_segment_label(axis)
                    if label:
                        return label, axis_label
            if "dimension" in value and "member" in value:
                label = _clean_segment_label(value.get("member"))
                axis_label = _clean_segment_label(value.get("dimension"))
                return label, axis_label
            if "value" in value:
                return _clean_segment_label(value.get("value")), None
            if "segment" in value:
                return _clean_segment_label(value.get("segment")), None
            parts = []
            for key in ("dimension", "member", "axis"):
                if key in value:
                    label = _clean_segment_label(value.get(key))
                    if label:
                        parts.append(label)
            if parts:
                return " | ".join(parts), None
            return _clean_segment_label(" ".join(str(v) for v in value.values())), None
        if isinstance(value, (list, tuple)):
            parts = []
            axes = []
            for item in value:
                label, axis_label = _segment_to_label_and_axis(item)
                if label:
                    parts.append(label)
                if axis_label:
                    axes.append(axis_label)
            if parts:
                axis = axes[0] if axes else None
                return " | ".join(parts), axis
            return None, None
        return _clean_segment_label(value), None

    def _extract_segment_from_payload(payload) -> pd.DataFrame:
        if payload is None or not isinstance(payload, dict):
            return pd.DataFrame()
        units = payload.get("units", {})
        if not units:
            return pd.DataFrame()

        unit_keys = list(units.keys())
        preferred_units = ["USD"] + [k for k in unit_keys if "USD" in k and k != "USD"]
        for unit in preferred_units + [
            k for k in unit_keys if k not in preferred_units
        ]:
            try:
                df = pd.DataFrame(units.get(unit, []))
            except Exception:
                continue
            if df.empty or "segment" not in df.columns:
                continue

            df = df[df["form"].isin(["10-Q", "10-K"])]
            seg_pairs = df["segment"].apply(_segment_to_label_and_axis)
            df["segment"] = seg_pairs.apply(lambda x: x[0])
            df["segment_axis"] = seg_pairs.apply(lambda x: x[1])
            df = df[df["segment"].notna()]
            if df.empty:
                continue
            axis_mask = (
                df["segment_axis"]
                .fillna("")
                .str.contains(
                    "BusinessSegmentsAxis|StatementBusinessSegmentsAxis",
                    case=False,
                    regex=True,
                )
            )
            if axis_mask.any():
                df = df[axis_mask]
            if df.empty:
                continue
            return df[["end", "segment", "val"]].rename(columns={"val": "Revenue"})

        return pd.DataFrame()

    def _extract_for_tag(tag: str) -> pd.DataFrame:
        try:
            payload = facts["facts"]["us-gaap"][tag]
        except Exception:
            return pd.DataFrame()

        return _extract_segment_from_payload(payload)

    for tag in REVENUE_TAGS:
        df = _extract_for_tag(tag)
        if df is not None and not df.empty:
            return df

    # Fallback: search other US-GAAP tags that include segment data for revenue-like labels.
    try:
        all_taxonomies = facts["facts"]
    except Exception:
        return pd.DataFrame()

    # Look for revenue-like tags across all taxonomies (including custom filers).
    revenue_like = ("revenue", "sales", "netsales", "netsales", "net sales")
    for tax_name, taxonomy in all_taxonomies.items():
        if not isinstance(taxonomy, dict):
            continue
        candidate_tags = [
            tag
            for tag in taxonomy.keys()
            if any(key in str(tag).lower() for key in revenue_like)
        ]
        for tag in candidate_tags:
            df = (
                _extract_for_tag(tag)
                if tax_name == "us-gaap"
                else _extract_segment_from_payload(taxonomy.get(tag))
            )
            if df is not None and not df.empty:
                return df

    # Last-ditch: try any tag with segment data in any taxonomy/units.
    for tax_name, taxonomy in all_taxonomies.items():
        if not isinstance(taxonomy, dict):
            continue
        for tag, payload in taxonomy.items():
            df = _extract_segment_from_payload(payload)
            if df is not None and not df.empty:
                return df

    return pd.DataFrame()


def _extract_segment_revenue_from_instance(xml_text: str) -> pd.DataFrame:
    if not xml_text:
        return pd.DataFrame()
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return pd.DataFrame()

    def _local(tag: str) -> str:
        return tag.split("}")[-1] if "}" in tag else tag

    def _clean_segment_label(raw: str | None) -> str | None:
        if raw is None:
            return None
        label = str(raw).strip()
        if not label:
            return None
        if ":" in label:
            label = label.split(":")[-1]
        if label.endswith("Member"):
            label = label[: -len("Member")]
        label = label.replace("_", " ").replace("-", " ")
        label = re.sub(r"(?<!^)(?=[A-Z])", " ", label)
        return " ".join(label.split()) or None

    ctx_map: dict[str, dict] = {}
    for ctx in root.findall(".//{http://www.xbrl.org/2003/instance}context"):
        ctx_id = ctx.get("id")
        if not ctx_id:
            continue
        end_date = None
        start_date = None
        period = ctx.find("{http://www.xbrl.org/2003/instance}period")
        if period is not None:
            start_el = period.find("{http://www.xbrl.org/2003/instance}startDate")
            end_el = period.find("{http://www.xbrl.org/2003/instance}endDate")
            inst_el = period.find("{http://www.xbrl.org/2003/instance}instant")
            if start_el is not None and start_el.text:
                start_date = start_el.text.strip()
            if end_el is not None and end_el.text:
                end_date = end_el.text.strip()
            elif inst_el is not None and inst_el.text:
                end_date = inst_el.text.strip()
        segment_labels = []
        segment_axis = None
        seg = ctx.find(
            "{http://www.xbrl.org/2003/instance}entity/{http://www.xbrl.org/2003/instance}segment"
        )
        if seg is not None:
            for mem in seg.findall(".//{http://xbrl.org/2006/xbrldi}explicitMember"):
                axis = mem.get("dimension")
                member = mem.text
                label = _clean_segment_label(member)
                axis_label = _clean_segment_label(axis) if axis else None
                if label:
                    segment_labels.append(label)
                if axis_label and segment_axis is None:
                    segment_axis = axis_label
        if segment_labels:
            ctx_map[ctx_id] = {
                "end": end_date,
                "start": start_date,
                "segment": " | ".join(segment_labels),
                "axis": segment_axis,
            }

    if not ctx_map:
        return pd.DataFrame()

    rows = []
    revenue_like = ("revenue", "sales")
    for elem in root.iter():
        ctx_ref = elem.get("contextRef")
        if not ctx_ref or ctx_ref not in ctx_map:
            continue
        tag = _local(elem.tag)
        tag_lower = tag.lower()
        if tag not in REVENUE_TAGS and not any(
            key in tag_lower for key in revenue_like
        ):
            continue
        text = elem.text.strip() if elem.text else ""
        try:
            val = float(text)
        except Exception:
            continue
        meta = ctx_map[ctx_ref]
        rows.append(
            {
                "end": meta.get("end"),
                "start": meta.get("start"),
                "segment": meta.get("segment"),
                "Revenue": val,
                "segment_axis": meta.get("axis"),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df[["end", "start", "segment", "Revenue", "segment_axis"]]


def fetch_segment_revenue_from_instance(cik: str, form: str = "10-Q") -> pd.DataFrame:
    """Fetch segment data from the latest filing of the given form (10-K or 10-Q).
    Filters to only the most recent period end date in that filing.
    """
    try:
        submissions = get_company_submissions(cik)
    except Exception:
        return pd.DataFrame()
    accession, _form = _latest_filing_accession(submissions, (form,))
    if not accession:
        return pd.DataFrame()
    index_json = _fetch_filing_index(cik, accession)
    filename = _select_instance_filename(index_json)
    if not filename:
        return pd.DataFrame()
    xml_text = _fetch_instance_xml(cik, accession, filename)
    df = _extract_segment_revenue_from_instance(xml_text or "")
    if df.empty:
        return df
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    df["start"] = pd.to_datetime(df.get("start"), errors="coerce")
    df = df[df["end"] == df["end"].max()].copy()
    # For 10-Q: keep only single-quarter rows (60-105 days), drop YTD rows
    if form == "10-Q" and "start" in df.columns:
        has_start = df["start"].notna()
        days = (df["end"] - df["start"]).dt.days
        quarter_mask = has_start & (days >= 60) & (days <= 105)
        if quarter_mask.any():
            df = df[quarter_mask | ~has_start]
    # Keep only revenue-relevant axes, drop financial instrument / fair value noise
    _REVENUE_AXES = (
        "Product Or Service Axis",
        "Business Segments Axis",
        "Geographical Axis",
        "Consolidation Items Axis",
    )
    if "segment_axis" in df.columns:
        axis_mask = (
            df["segment_axis"]
            .fillna("")
            .str.contains("|".join(_REVENUE_AXES), case=False, regex=True)
        )
        if axis_mask.any():
            df = df[axis_mask]
    # Drop high-level rollup rows ("Product", "Service Other") when detailed rows exist under the same axis
    _GENERIC_ROLLUPS = {"product", "service", "service other", "services", "products"}
    if "segment_axis" in df.columns:
        for axis_val in df["segment_axis"].dropna().unique():
            axis_rows = df[df["segment_axis"] == axis_val]
            if len(axis_rows) > 3:
                rollup_mask = axis_rows["segment"].str.lower().isin(_GENERIC_ROLLUPS)
                if rollup_mask.any() and (~rollup_mask).any():
                    df = df.drop(axis_rows[rollup_mask].index)
    df["end"] = df["end"].dt.date.astype(str)
    df["form"] = _form
    return df


def _find_segment_tags(facts) -> list[str]:
    try:
        all_taxonomies = facts["facts"]
    except Exception:
        return []

    tags: list[str] = []
    for tax_name, taxonomy in all_taxonomies.items():
        if not isinstance(taxonomy, dict):
            continue
        for tag, payload in taxonomy.items():
            try:
                units = payload.get("units", {})
                if not units:
                    continue
                for unit, rows in units.items():
                    df = pd.DataFrame(rows)
                    if df.empty or "segment" not in df.columns:
                        continue
                    if df["segment"].isna().all():
                        continue
                    tags.append(f"{tax_name}:{tag} [{unit}]")
                    break
            except Exception:
                continue
    return sorted(tags)


def _summarize_segment_facts(facts, limit: int = 12) -> str:
    try:
        all_taxonomies = facts["facts"]
    except Exception:
        return "No facts payload available."

    summaries: list[tuple[str, int]] = []
    total = 0
    for tax_name, taxonomy in all_taxonomies.items():
        if not isinstance(taxonomy, dict):
            continue
        count = 0
        for _, payload in taxonomy.items():
            try:
                units = payload.get("units", {})
                if not units:
                    continue
                for _, rows in units.items():
                    df = pd.DataFrame(rows)
                    if df.empty or "segment" not in df.columns:
                        continue
                    if df["segment"].isna().all():
                        continue
                    count += 1
                    break
            except Exception:
                continue
        if count:
            summaries.append((tax_name, count))
            total += count

    if not summaries:
        return "No segment-tagged facts found in any taxonomy."

    summaries = sorted(summaries, key=lambda x: x[1], reverse=True)
    head = ", ".join(f"{name}:{cnt}" for name, cnt in summaries[:limit])
    more = "" if len(summaries) <= limit else f" (+{len(summaries) - limit} more)"
    return f"Segment-tagged fact counts by taxonomy (total {total}): {head}{more}"


_PRODUCT_MAP = {
    "I Phone": "iPhone",
    "I Pad": "iPad",
    "Mac": "Mac",
    "Wearables Homeand Accessories": "Wearables",
    "Product": "Products (total)",
    "Service": "Services (total)",
}
_GEO_MAP = {
    "Operating Segments | Americas Segment": "Americas",
    "Operating Segments | Europe Segment": "Europe",
    "Operating Segments | Greater China Segment": "Greater China",
    "Operating Segments | Japan Segment": "Japan",
    "Operating Segments | Rest Of Asia Pacific Segment": "Rest of Asia Pacific",
}
_PRODUCT_AXIS_KEY = "Product Or Service Axis"
_GEO_AXIS_KEY = "Geographical Axis|Consolidation Items Axis"


def _build_segment_chart(
    segment_df: pd.DataFrame,
    annual_df: pd.DataFrame | None = None,
    quarterly_df: pd.DataFrame | None = None,
    segment_period: str = "10-Q",
) -> tuple[go.Figure, str, list[dict], list[dict]]:
    empty_fig = go.Figure()
    empty_fig.update_layout(title="Segment Analysis", template="plotly_white")
    if segment_df is None or segment_df.empty:
        return (
            empty_fig,
            "No segment data found (requires segment-tagged XBRL).",
            [],
            [],
        )

    df = segment_df.copy()
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    df = df.dropna(subset=["end"])
    latest_end = df["end"].max()
    latest = df[df["end"] == latest_end].copy()
    axis_col = "segment_axis" if "segment_axis" in latest.columns else None

    # Product / Service sub-segments
    if axis_col:
        prod_mask = (
            latest[axis_col].fillna("").str.contains(_PRODUCT_AXIS_KEY, case=False)
        )
        prod_df = latest[prod_mask].copy()
    else:
        prod_df = latest.copy()
    prod_df["label"] = prod_df["segment"].map(_PRODUCT_MAP).fillna(prod_df["segment"])
    prod_df = prod_df[prod_df["label"].notna()]
    prod_df = prod_df.drop_duplicates(subset=["label", "Revenue"])
    prod_df = prod_df.groupby("label", as_index=False)["Revenue"].max()
    sub_products = ["iPhone", "Mac", "iPad", "Wearables"]
    totals = ["Products (total)", "Services (total)"]
    mapped_labels = set(_PRODUCT_MAP.values())
    is_apple = prod_df["label"].isin(mapped_labels).any()
    if is_apple:
        prod_sub = prod_df[prod_df["label"].isin(sub_products)].sort_values(
            "Revenue", ascending=False
        )
        prod_tot = prod_df[prod_df["label"].isin(totals)].sort_values(
            "Revenue", ascending=False
        )
    else:
        prod_sub = prod_df.sort_values("Revenue", ascending=False)
        prod_tot = pd.DataFrame(columns=prod_df.columns)

    colors = {
        "iPhone": "#2563eb",
        "Mac": "#10b981",
        "iPad": "#f59e0b",
        "Wearables": "#8b5cf6",
    }
    palette = [
        "#2563eb",
        "#10b981",
        "#f59e0b",
        "#8b5cf6",
        "#0ea5e9",
        "#f97316",
        "#06b6d4",
        "#84cc16",
        "#ec4899",
        "#6b7280",
    ]
    fig = go.Figure()
    for i, (_, row) in enumerate(prod_sub.iterrows()):
        fig.add_trace(
            go.Bar(
                x=[row["label"]],
                y=[row["Revenue"]],
                name=row["label"],
                marker_color=colors.get(row["label"], palette[i % len(palette)]),
            )
        )
    for _, row in prod_tot.iterrows():
        fig.add_trace(
            go.Bar(
                x=[row["label"]],
                y=[row["Revenue"]],
                name=row["label"],
                marker_color="#d1d5db",
                marker_line=dict(color="#6b7280", width=1),
            )
        )
    fig.update_layout(
        title=f"Product & Service Revenue (Latest: {latest_end.date()})",
        xaxis_title="Segment",
        yaxis=dict(title="USD"),
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        height=420,
        showlegend=False,
    )
    fig.update_xaxes(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb")
    fig.update_yaxes(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb")

    # Datatable — match total revenue to the same period as segment data
    total_rev = prod_sub["Revenue"].sum()
    total_form = ""
    # Pick reference series based on which form the segment data came from
    ref_pairs = (
        [(quarterly_df, "10-Q"), (annual_df, "10-K")]
        if segment_period == "10-Q"
        else [(annual_df, "10-K"), (quarterly_df, "10-Q")]
    )
    for ref_df, label in ref_pairs:
        if ref_df is not None and not ref_df.empty:
            ref = ref_df.copy()
            ref["end"] = pd.to_datetime(ref["end"], errors="coerce")
            ref = ref.dropna(subset=["end"]).sort_values("end")
            closest_idx = (ref["end"] - latest_end).abs().argsort()
            closest = ref.iloc[closest_idx].iloc[0]
            if abs((closest["end"] - latest_end).days) <= 35:
                total_rev = closest["Revenue"]
                total_form = closest.get("form", label)
                break
    date_str = str(latest_end.date())
    total_label = f"Total Revenue ({total_form or segment_period})"
    rows_out = (
        [
            {
                "segment": total_label,
                "Revenue": total_rev,
                "end": date_str,
                "axis": "annual",
                "form": total_form,
            }
        ]
        + [
            {
                "segment": r["label"],
                "Revenue": r["Revenue"],
                "end": date_str,
                "axis": "product",
                "form": "",
            }
            for _, r in prod_sub.iterrows()
        ]
        + [
            {
                "segment": r["label"],
                "Revenue": r["Revenue"],
                "end": date_str,
                "axis": "product-total",
                "form": "",
            }
            for _, r in prod_tot.iterrows()
        ]
    )
    if axis_col:
        geo_mask = latest[axis_col].fillna("").str.contains(_GEO_AXIS_KEY, case=False)
        geo_sub = latest[geo_mask].copy()
        if not geo_sub.empty:
            geo_sub["label"] = (
                geo_sub["segment"].map(_GEO_MAP).fillna(geo_sub["segment"])
            )
            geo_sub = geo_sub[geo_sub["label"].notna()]
            geo_sub = geo_sub.groupby("label", as_index=False)["Revenue"].sum()
            rows_out += [
                {
                    "segment": r["label"],
                    "Revenue": r["Revenue"],
                    "end": date_str,
                    "axis": "geo",
                    "form": "",
                }
                for _, r in geo_sub.iterrows()
            ]

    display = pd.DataFrame(rows_out)
    display["Revenue"] = display["Revenue"].map(lambda x: f"${x:,.0f}")
    cols = ["segment", "Revenue", "axis", "end", "form"]
    columns = [{"name": c, "id": c} for c in cols]
    data = display[cols].to_dict("records")
    return fig, f"Latest segment data: {latest_end.date()}", data, columns


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
                            html.H2("EDGAR Revenue Dashboard", className="mb-1"),
                            html.Div(
                                "Quarterly revenue from EDGAR XBRL filings.",
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
                                                            id="rev-ticker-input",
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
                                                            id="rev-load-btn",
                                                            className="btn btn-primary w-100",
                                                        ),
                                                        html.Div(
                                                            id="rev-live-refreshed",
                                                            className="text-muted small mt-1",
                                                        ),
                                                        html.Button(
                                                            "Load (DB)",
                                                            id="rev-load-db-btn",
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
                                dbc.CardHeader("Annual Revenue (10-K)"),
                                dbc.CardBody(
                                    dcc.Graph(id="rev-annual-chart"),
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
                                dbc.CardHeader("Annual Growth Momentum"),
                                dbc.CardBody(
                                    [
                                        dcc.RadioItems(
                                            id="rev-annual-growth-window",
                                            options=[
                                                {"label": "1Y", "value": 1},
                                                {"label": "2Y", "value": 2},
                                                {"label": "3Y", "value": 3},
                                                {"label": "4Y", "value": 4},
                                                {"label": "5Y", "value": 5},
                                                {"label": "ALL", "value": 0},
                                            ],
                                            value=3,
                                            inline=True,
                                            inputStyle={"marginRight": "4px"},
                                            labelStyle={
                                                "marginRight": "12px",
                                                "fontWeight": 500,
                                            },
                                        ),
                                        dcc.Graph(
                                            id="rev-annual-growth-chart",
                                            style={"marginTop": "8px"},
                                        ),
                                    ],
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
                                dbc.CardHeader("Annual Revenue Detail"),
                                dbc.CardBody(
                                    dash_table.DataTable(
                                        id="rev-annual-table",
                                        page_size=12,
                                        sort_action="native",
                                        filter_action="native",
                                        style_table={"overflowX": "auto"},
                                        style_cell={
                                            "fontFamily": "Segoe UI, Arial, sans-serif",
                                            "fontSize": 12,
                                            "padding": "6px",
                                            "textAlign": "left",
                                        },
                                        style_header={
                                            "backgroundColor": "#e9ecef",
                                            "fontWeight": "bold",
                                            "color": "#1f2937",
                                        },
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
                                dbc.CardHeader("Revenue Trend"),
                                dbc.CardBody(
                                    dcc.Graph(id="rev-chart"),
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
                                dbc.CardHeader("Growth Momentum"),
                                dbc.CardBody(
                                    dcc.Graph(id="rev-growth-chart"),
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
                                dbc.CardHeader("Quarterly Detail"),
                                dbc.CardBody(
                                    dash_table.DataTable(
                                        id="rev-table",
                                        page_size=12,
                                        sort_action="native",
                                        filter_action="native",
                                        css=[
                                            {
                                                "selector": ".dash-filter--case",
                                                "rule": "background-color: #dbeafe; border: 1px solid #93c5fd; color: #000;",
                                            },
                                            {
                                                "selector": ".dash-filter--case--sensitive",
                                                "rule": "background-color: #bfdbfe; border: 1px solid #60a5fa; color: #000;",
                                            },
                                            {
                                                "selector": ".dash-filter--case--insensitive",
                                                "rule": "background-color: #dbeafe; border: 1px solid #93c5fd; color: #000;",
                                            },
                                            {
                                                "selector": ".dash-spreadsheet-container th .column-header--sort",
                                                "rule": "color: #60a5fa;",
                                            },
                                            {
                                                "selector": ".dash-spreadsheet-container th .column-header--sort-icon",
                                                "rule": "color: #60a5fa; fill: #60a5fa;",
                                            },
                                            {
                                                "selector": ".dash-spreadsheet-container th .sort-icon",
                                                "rule": "color: #60a5fa; fill: #60a5fa;",
                                            },
                                        ],
                                        style_data_conditional=[
                                            {
                                                "if": {"state": "active"},
                                                "backgroundColor": "#dbeafe",
                                                "border": "1px solid #93c5fd",
                                            },
                                            {
                                                "if": {"state": "selected"},
                                                "backgroundColor": "#bfdbfe",
                                                "border": "1px solid #60a5fa",
                                            },
                                        ],
                                        style_table={"overflowX": "auto"},
                                        style_cell={
                                            "fontFamily": "Segoe UI, Arial, sans-serif",
                                            "fontSize": 12,
                                            "padding": "6px",
                                            "textAlign": "left",
                                        },
                                        style_header={
                                            "backgroundColor": "#e9ecef",
                                            "fontWeight": "bold",
                                            "color": "#1f2937",
                                        },
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
                                dbc.CardHeader("Revenue Quality Comparison"),
                                dbc.CardBody(
                                    dcc.Graph(id="rev-quality-chart"),
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
                                dbc.CardHeader("OCF Margin"),
                                dbc.CardBody(
                                    dcc.Graph(id="rev-ocf-margin-chart"),
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
                                dbc.CardHeader("Segment Analysis"),
                                dbc.CardBody(
                                    [
                                        dcc.RadioItems(
                                            id="rev-segment-period",
                                            options=[
                                                {
                                                    "label": "Latest Quarter (10-Q)",
                                                    "value": "10-Q",
                                                },
                                                {
                                                    "label": "Latest Annual (10-K)",
                                                    "value": "10-K",
                                                },
                                            ],
                                            value="10-Q",
                                            inline=True,
                                            inputStyle={"marginRight": "4px"},
                                            labelStyle={
                                                "marginRight": "16px",
                                                "fontWeight": 500,
                                            },
                                            className="mb-2",
                                        ),
                                        dcc.Graph(id="rev-segment-chart"),
                                        dcc.Graph(
                                            id="rev-segment-geo-chart",
                                            style={"marginTop": "16px"},
                                        ),
                                        html.Div(
                                            id="rev-segment-status",
                                            className="text-muted small mt-2",
                                        ),
                                    ],
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
                        html.Div(id="rev-status-label", className="text-muted"),
                        width=12,
                    )
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(id="rev-last-refreshed", className="text-muted"),
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
                                dbc.CardHeader("Segment Detail"),
                                dbc.CardBody(
                                    dash_table.DataTable(
                                        id="rev-segment-table",
                                        page_size=12,
                                        sort_action="native",
                                        filter_action="native",
                                        style_table={"overflowX": "auto"},
                                        style_cell={
                                            "fontFamily": "Segoe UI, Arial, sans-serif",
                                            "fontSize": 12,
                                            "padding": "6px",
                                            "textAlign": "left",
                                        },
                                        style_header={
                                            "backgroundColor": "#e9ecef",
                                            "fontWeight": "bold",
                                            "color": "#1f2937",
                                        },
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
        ],
        fluid=True,
        className="py-4",
    )


layout = build_layout()


def register_callbacks(app):
    @app.callback(
        Output("rev-chart", "figure"),
        Output("rev-annual-chart", "figure"),
        Output("rev-annual-growth-chart", "figure"),
        Output("rev-annual-table", "data"),
        Output("rev-annual-table", "columns"),
        Output("rev-growth-chart", "figure"),
        Output("rev-quality-chart", "figure"),
        Output("rev-ocf-margin-chart", "figure"),
        Output("rev-segment-chart", "figure"),
        Output("rev-segment-geo-chart", "figure"),
        Output("rev-segment-status", "children"),
        Output("rev-segment-table", "data"),
        Output("rev-segment-table", "columns"),
        Output("rev-table", "data"),
        Output("rev-table", "columns"),
        Output("rev-status-label", "children"),
        Output("rev-last-refreshed", "children"),
        Output("rev-live-refreshed", "children"),
        Input("rev-load-btn", "n_clicks"),
        Input("rev-load-db-btn", "n_clicks"),
        Input("rev-annual-growth-window", "value"),
        Input("rev-ticker-input", "value"),
        Input("rev-segment-period", "value"),
    )
    def update_chart(_n, _db_n, annual_window, ticker, segment_period):
        if _n is None and _db_n is None:
            df, tag_used, annual_df = build_revenue(
                ticker, use_cache=True, cache_only=True
            )
            price_df = _load_or_fetch_prices(
                _normalize_ticker(ticker), allow_fetch=False
            )
            if df is None or df.empty:
                return (
                    go.Figure(),
                    go.Figure(),
                    go.Figure(),
                    [],
                    [],
                    go.Figure(),
                    go.Figure(),
                    go.Figure(),
                    go.Figure(),
                    go.Figure(),
                    "No cached data found. Load live or run pipeline.",
                    [],
                    [],
                    [],
                    [],
                    "No cached data found. Load live or run pipeline.",
                    "",
                    "",
                )
            df_display = df.copy()
            df_display = df_display.sort_values("end").reset_index(drop=True)
            df_display["end"] = pd.to_datetime(df_display["end"]).dt.date.astype(str)
            fy_end_month = _infer_fiscal_year_end_month(annual_df)
            df_display = _add_fiscal_columns(df_display, fy_end_month)
            df_display["YoY %"] = _compute_quarterly_yoy(df_display, fy_end_month)
            if "fp" in df_display.columns:
                df_display["fp"] = df_display["Fiscal Quarter"]
            if "fy" in df_display.columns:
                df_display["fy"] = df_display["Fiscal Year"]
            df_display["ticker"] = _normalize_ticker(ticker)
            df_display["YoY %"] = df_display["YoY %"].map(
                lambda x: "" if pd.isna(x) else f"{x:+.1f}%"
            )
            df_display = df_display.sort_values("end", ascending=False)
            columns = [
                {"name": col, "id": col}
                for col in df_display.columns
                if col not in {"fp", "fy", "Fiscal Year", "period_type"}
            ]
            data = df_display.to_dict("records")
            refreshed = _format_refreshed(ticker)
            return (
                create_figure(df, ticker, refreshed, tag_used),
                _build_annual_figure(annual_df, ticker),
                _build_annual_growth_figure(annual_df, annual_window, price_df),
                [],
                [],
                _build_growth_momentum_figure(df, fy_end_month),
                _build_quality_figure(
                    annual_df, pd.DataFrame(), _normalize_ticker(ticker)
                ),
                _build_ocf_margin_figure(
                    annual_df, pd.DataFrame(), _normalize_ticker(ticker)
                ),
                go.Figure(),
                go.Figure(),
                "Segment data requires live load.",
                [],
                [],
                data,
                columns,
                "Loaded from SQLite cache.",
                refreshed,
                "",
            )

        triggered = ctx.triggered_id
        cache_only = triggered == "rev-load-db-btn"
        use_cache = triggered != "rev-load-btn"
        df, tag_used, annual_df = build_revenue(
            ticker, use_cache=use_cache, cache_only=cache_only
        )
        price_df = _load_or_fetch_prices(
            _normalize_ticker(ticker),
            allow_fetch=not cache_only,
        )
        if df is None or df.empty:
            message = (
                "No cached data found. Run live load or pipeline."
                if cache_only
                else "No data returned for this ticker."
            )
            return (
                go.Figure(),
                go.Figure(),
                go.Figure(),
                [],
                [],
                go.Figure(),
                go.Figure(),
                go.Figure(),
                go.Figure(),
                go.Figure(),
                message,
                [],
                [],
                [],
                [],
                message,
                "",
                "",
            )

        df_display = df.copy()
        df_display = df_display.sort_values("end").reset_index(drop=True)
        df_display["end"] = pd.to_datetime(df_display["end"]).dt.date.astype(str)
        fy_end_month = _infer_fiscal_year_end_month(annual_df)
        df_display = _add_fiscal_columns(df_display, fy_end_month)
        df_display["YoY %"] = _compute_quarterly_yoy(df_display, fy_end_month)
        if "fp" in df_display.columns:
            df_display["fp"] = df_display["Fiscal Quarter"]
        if "fy" in df_display.columns:
            df_display["fy"] = df_display["Fiscal Year"]
        df_display["ticker"] = _normalize_ticker(ticker)
        df_display["YoY %"] = df_display["YoY %"].map(
            lambda x: "" if pd.isna(x) else f"{x:+.1f}%"
        )
        df_display = df_display.sort_values("end", ascending=False)
        columns = [
            {"name": col, "id": col}
            for col in df_display.columns
            if col not in {"fp", "fy", "Fiscal Year", "period_type"}
        ]
        data = df_display.to_dict("records")

        status = "Loaded from SQLite cache." if cache_only else "Loaded live from SEC."
        refreshed = _format_refreshed(ticker)
        live_refreshed = refreshed if not cache_only else ""

        ocf_df = pd.DataFrame()
        segment_fig = go.Figure()
        segment_geo_fig = go.Figure()
        segment_status = "Segment data requires live load."
        segment_data: list[dict] = []
        segment_columns: list[dict] = []
        if not cache_only:
            try:
                cik = get_cik(_normalize_ticker(ticker))
                if cik:
                    ocf_df = fetch_ocf_annual(cik)
                    facts = get_company_facts(cik)
                    segment_df = _extract_segment_revenue(facts)
                    if segment_df is None or segment_df.empty:
                        instance_df = fetch_segment_revenue_from_instance(
                            cik, form=segment_period
                        )
                        if instance_df is not None and not instance_df.empty:
                            segment_df = instance_df
                    segment_fig, segment_status, segment_data, segment_columns = (
                        _build_segment_chart(segment_df, annual_df, df, segment_period)
                    )
                    if (
                        segment_df is not None
                        and not segment_df.empty
                        and "segment_axis" in segment_df.columns
                    ):
                        geo_mask = (
                            segment_df["segment_axis"]
                            .fillna("")
                            .str.contains(_GEO_AXIS_KEY, case=False)
                        )
                        geo_sub = segment_df[geo_mask].copy()
                        if not geo_sub.empty:
                            geo_sub["end"] = pd.to_datetime(
                                geo_sub["end"], errors="coerce"
                            )
                            geo_sub = geo_sub[
                                geo_sub["end"] == geo_sub["end"].max()
                            ].copy()
                            geo_sub["label"] = (
                                geo_sub["segment"]
                                .map(_GEO_MAP)
                                .fillna(geo_sub["segment"])
                            )
                            geo_sub = geo_sub[geo_sub["label"].notna()]
                            geo_sub = (
                                geo_sub.groupby("label", as_index=False)["Revenue"]
                                .sum()
                                .sort_values("Revenue", ascending=False)
                            )
                            geo_colors = [
                                "#0ea5e9",
                                "#06b6d4",
                                "#14b8a6",
                                "#84cc16",
                                "#f97316",
                            ]
                            segment_geo_fig = go.Figure(
                                go.Bar(
                                    x=geo_sub["label"],
                                    y=geo_sub["Revenue"],
                                    marker_color=geo_colors[: len(geo_sub)],
                                )
                            )
                            segment_geo_fig.update_layout(
                                title=f"Geographic Revenue (Latest: {segment_df['end'].max()})",
                                xaxis_title="Region",
                                yaxis=dict(title="USD"),
                                template="plotly_white",
                                paper_bgcolor="white",
                                plot_bgcolor="white",
                                font=dict(color="black"),
                                height=420,
                                showlegend=False,
                            )
                            segment_geo_fig.update_xaxes(
                                gridcolor="#e5e7eb", zerolinecolor="#e5e7eb"
                            )
                            segment_geo_fig.update_yaxes(
                                gridcolor="#e5e7eb", zerolinecolor="#e5e7eb"
                            )
                    if segment_df is None or segment_df.empty:
                        tags = _find_segment_tags(facts)
                        if tags:
                            sample = ", ".join(tags[:8])
                            more = "" if len(tags) <= 8 else f" (+{len(tags) - 8} more)"
                            segment_status = (
                                "No segment revenue found. Segment-tagged facts: "
                                f"{sample}{more}"
                            )
                        else:
                            segment_status = _summarize_segment_facts(facts)
                else:
                    segment_status = "Could not resolve CIK for segment data."
            except Exception:
                segment_status = "Segment extraction failed."

        annual_display = pd.DataFrame()
        annual_columns: list[dict] = []
        if annual_df is not None and not annual_df.empty:
            annual_display = annual_df.copy()
            annual_display = annual_display.sort_values("end")
            end_dt = pd.to_datetime(annual_display["end"], errors="coerce")
            if "fp" in annual_display.columns:
                annual_display["fp"] = "FY"
            if "fy" in annual_display.columns:
                annual_display["fy"] = end_dt.dt.year
            annual_display["YoY %"] = annual_display["Revenue"].pct_change() * 100
            annual_display["end"] = pd.to_datetime(
                annual_display["end"]
            ).dt.date.astype(str)
            annual_display["Revenue"] = annual_display["Revenue"].map(
                lambda x: f"${x:,.0f}"
            )
            annual_display["YoY %"] = annual_display["YoY %"].map(
                lambda x: "" if pd.isna(x) else f"{x:+.1f}%"
            )
            annual_display = annual_display.sort_values("end", ascending=False)
            annual_columns = [
                {"name": col, "id": col}
                for col in annual_display.columns
                if col not in {"fp", "fy", "Fiscal Year", "period_type"}
            ]
        annual_data = (
            annual_display.to_dict("records") if not annual_display.empty else []
        )

        return (
            create_figure(df, ticker, refreshed, tag_used),
            _build_annual_figure(annual_df, ticker),
            _build_annual_growth_figure(annual_df, annual_window, price_df),
            annual_data,
            annual_columns,
            _build_growth_momentum_figure(df, fy_end_month),
            _build_quality_figure(annual_df, ocf_df, _normalize_ticker(ticker)),
            _build_ocf_margin_figure(annual_df, ocf_df, _normalize_ticker(ticker)),
            segment_fig,
            segment_geo_fig,
            segment_status,
            segment_data,
            segment_columns,
            data,
            columns,
            status,
            refreshed,
            live_refreshed,
        )


register_callbacks(get_app())
