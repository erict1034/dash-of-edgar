import asyncio
import aiohttp
import hashlib
import sqlite3
import os
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import get_app, Input, Output, State, dash_table, dcc, html, ctx
from dotenv import load_dotenv
from storage_paths import CENTRAL_SQLITE_PATH, parquet_path

dash.register_page(
    __name__,
    path="/speedy-index-pull",
    name="Daily Insider Purchase List",
    order=0,
)

load_dotenv()

# -----------------------------
# SEC REQUIRED HEADERS
# -----------------------------
SEC_CONTACT_EMAIL = os.getenv("SEC_CONTACT_EMAIL", "your_email@example.com")
HEADERS = {
    "User-Agent": f"DashOfEdgar ({SEC_CONTACT_EMAIL})",
    "Accept-Encoding": "gzip, deflate",
    "Accept": "text/html,application/xml;q=0.9,*/*;q=0.8",
    "Connection": "keep-alive",
}

# Optional: display floats nicely
pd.options.display.float_format = "{:,.2f}".format

SQLITE_DB_PATH = CENTRAL_SQLITE_PATH
PARQUET_PATH = parquet_path("ef4_speedy_form4")


def _normalize_daily_index_date(value: str | None) -> str:
    if not value:
        raise ValueError("Daily index date is required.")
    parsed = pd.to_datetime(str(value), errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid daily index date: {value}")
    return parsed.strftime("%Y-%m-%d")


def _daily_index_token(value: str | None) -> str:
    return _normalize_daily_index_date(value).replace("-", "")


def _weekend_disabled_days(
    start_date: str = "2020-01-01", end_date: str = "2030-12-31"
) -> list[str]:
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    return [d.strftime("%Y-%m-%d") for d in dates if d.weekday() >= 5]


# -----------------------------
# Find the latest available SEC index
# -----------------------------
async def find_latest_available_date(session):
    test_date = datetime.now()
    for _ in range(7):  # look back up to 7 days
        date_str = test_date.strftime("%Y-%m-%d")
        token = date_str.replace("-", "")
        year = token[:4]
        month = int(token[4:6])
        quarter = (month - 1) // 3 + 1
        url = f"https://www.sec.gov/Archives/edgar/daily-index/{year}/QTR{quarter}/master.{token}.idx"
        async with session.get(url) as r:
            if r.status == 200:
                print(f"Using SEC filing date: {date_str}")
                return date_str
        test_date -= timedelta(days=1)
    raise Exception("No recent SEC index found.")


# -----------------------------
# Get Form 4 filings from index
# -----------------------------
async def get_form4_index(session, date):
    canonical_date = _normalize_daily_index_date(date)
    token = canonical_date.replace("-", "")
    year = token[:4]
    month = int(token[4:6])
    quarter = (month - 1) // 3 + 1
    url = f"https://www.sec.gov/Archives/edgar/daily-index/{year}/QTR{quarter}/master.{token}.idx"
    async with session.get(url) as resp:
        resp.raise_for_status()
        text = await resp.text()
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("CIK|"))
    records = []
    for line in lines[start + 1 :]:
        parts = line.split("|")
        if len(parts) == 5 and parts[2] == "4":
            records.append(parts)
    df = pd.DataFrame(records, columns=["CIK", "Company", "Form", "Date", "FilePath"])
    df["FilingURL"] = "https://www.sec.gov/Archives/" + df["FilePath"]
    print(f"Found {len(df)} Form 4 filings")
    return df


# -----------------------------
# Parse transactions (Code P only) from Form 4 XML
# -----------------------------
async def parse_transactions(session, filing_url):
    try:
        async with session.get(filing_url) as r:
            text = await r.text()
        start = text.find("<XML>")
        end = text.find("</XML>")
        if start == -1 or end == -1:
            return []
        xml = text[start + 5 : end]
        soup = BeautifulSoup(xml, "xml")
        issuer = soup.find("issuer")
        company = issuer.find("issuerName").text if issuer else "Unknown"
        ticker = issuer.find("issuerTradingSymbol").text if issuer else "N/A"
        rows = []
        for txn in soup.find_all("nonDerivativeTransaction"):
            code_tag = txn.find("transactionCode")
            shares_tag = txn.find("transactionShares")
            if not code_tag or not shares_tag:
                continue
            code = code_tag.text.strip()
            if code != "P":  # keep only purchases
                continue
            rows.append(
                {
                    "Company": company,
                    "Ticker": ticker,
                    "Code": code,
                    "Shares": float(shares_tag.value.text),
                }
            )
        return rows
    except Exception:
        return []


# -----------------------------
# Process a single filing
# -----------------------------
async def process_filing(session, row):
    return await parse_transactions(session, row["FilingURL"])


# -----------------------------
# Main scraper
# -----------------------------
async def scrape_purchases(
    daily_index_date: str | None = None,
) -> tuple[pd.DataFrame, str]:
    connector = aiohttp.TCPConnector(limit=5)  # SEC-safe
    async with aiohttp.ClientSession(headers=HEADERS, connector=connector) as session:
        if daily_index_date:
            daily_index_date = _normalize_daily_index_date(daily_index_date)
        else:
            daily_index_date = await find_latest_available_date(session)
        df = await get_form4_index(session, daily_index_date)
        tasks = [process_filing(session, row) for _, row in df.iterrows()]
        results = await asyncio.gather(*tasks)
    flat = [item for sublist in results for item in sublist]
    final_df = pd.DataFrame(flat)
    return final_df, daily_index_date


def query_latest_loaded_date(db_path: Path = SQLITE_DB_PATH) -> str | None:
    if not db_path.exists():
        return None
    with sqlite3.connect(db_path) as conn:
        try:
            row = conn.execute(
                "SELECT MAX(filing_date) FROM form4_purchases"
            ).fetchone()
        except sqlite3.OperationalError:
            # Centralized DB may exist before this table is created.
            return None
    if not row or not row[0]:
        return None
    return str(row[0])


def query_top_purchases(
    limit: int = 200,
    db_path: Path = SQLITE_DB_PATH,
    filing_date: str | None = None,
) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()

    target_date = filing_date or query_latest_loaded_date(db_path)
    if not target_date:
        return pd.DataFrame()

    sql = """
    SELECT
        filing_date AS daily_index_date,
        ticker,
        company,
        SUM(shares) AS total_shares,
        COUNT(*) AS filings
    FROM form4_purchases
    WHERE filing_date = ?
    GROUP BY filing_date, ticker, company
    ORDER BY total_shares DESC
    LIMIT ?
    """
    with sqlite3.connect(db_path) as conn:
        try:
            return pd.read_sql_query(sql, conn, params=(target_date, int(limit)))
        except Exception:
            return pd.DataFrame()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalize_scraped_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["daily_index_date", "ticker", "company", "shares", "code"]
        )

    def _get_series(
        frame: pd.DataFrame, names: list[str], default: str = ""
    ) -> pd.Series:
        for name in names:
            if name in frame.columns:
                return frame[name]
        return pd.Series([default] * len(frame), index=frame.index)

    out = df.copy()
    if "daily_index_date" not in out.columns and out.index.name == "daily_index_date":
        out = out.reset_index()
    out["ticker"] = (
        _get_series(out, ["Ticker", "ticker"])
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )
    out["company"] = (
        _get_series(out, ["Company", "company"]).fillna("").astype(str).str.strip()
    )
    out["shares"] = pd.to_numeric(
        _get_series(out, ["Shares", "shares"], default="0"), errors="coerce"
    ).fillna(0.0)
    out["code"] = (
        _get_series(out, ["Code", "code"], default="P")
        .fillna("P")
        .astype(str)
        .str.strip()
        .str.upper()
    )
    out["daily_index_date"] = (
        pd.to_datetime(
            _get_series(out, ["daily_index_date", "DailyIndexDate"], default=""),
            errors="coerce",
        )
        .dt.date.astype("string")
        .fillna("")
    )
    out = out[out["code"] == "P"]
    return out[["daily_index_date", "ticker", "company", "shares", "code"]]


def save_parquet_snapshot(
    data: pd.DataFrame,
    daily_index_date: str,
    parquet_path: Path = PARQUET_PATH,
) -> tuple[Path, int]:
    normalized = _normalize_scraped_rows(data)
    normalized["daily_index_date"] = _normalize_daily_index_date(daily_index_date)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    if parquet_path.exists():
        try:
            existing = pd.read_parquet(parquet_path)
        except Exception:
            existing = pd.DataFrame()
        if existing is not None and not existing.empty:
            normalized = pd.concat([existing, normalized], ignore_index=True)

    normalized = normalized.drop_duplicates(
        subset=["daily_index_date", "ticker", "company", "shares", "code"],
        keep="last",
    )
    normalized["daily_index_date"] = pd.to_datetime(
        normalized["daily_index_date"], errors="coerce"
    )
    normalized = normalized.dropna(subset=["daily_index_date"]).sort_values(
        ["daily_index_date", "ticker", "company"]
    )
    normalized = normalized.set_index("daily_index_date")
    normalized.to_parquet(parquet_path)
    return parquet_path, len(normalized)


def load_parquet_snapshot(parquet_path: Path = PARQUET_PATH) -> pd.DataFrame:
    if not parquet_path.exists():
        return pd.DataFrame()
    out = pd.read_parquet(parquet_path)
    if out.index.name == "daily_index_date":
        out = out.reset_index()
    return out


def upsert_sqlite_purchases(
    data: pd.DataFrame, db_path: Path = SQLITE_DB_PATH
) -> tuple[int, int]:
    normalized = _normalize_scraped_rows(data)
    if normalized.empty:
        return 0, 0

    pulled_at_utc = _utc_now_iso()

    # Keep row granularity by creating a deterministic id from row content.
    normalized = normalized.reset_index(drop=True)
    normalized["record_id"] = normalized.apply(
        lambda row: hashlib.sha1(
            f"{row['daily_index_date']}|{row['ticker']}|{row['company']}|{row['code']}|{row['shares']}|{row.name}".encode(
                "utf-8"
            )
        ).hexdigest(),
        axis=1,
    )

    rows = [
        (
            row.record_id,
            row.daily_index_date,
            row.ticker,
            row.company,
            row.code,
            float(row.shares),
            pulled_at_utc,
        )
        for row in normalized.itertuples(index=False)
    ]

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS form4_purchases (
                record_id TEXT PRIMARY KEY,
                filing_date TEXT NOT NULL,
                ticker TEXT,
                company TEXT,
                code TEXT,
                shares REAL,
                pulled_at_utc TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_form4_filing_date ON form4_purchases(filing_date)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_form4_ticker ON form4_purchases(ticker)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_form4_company ON form4_purchases(company)"
        )

        before = conn.execute("SELECT COUNT(*) FROM form4_purchases").fetchone()[0]
        conn.executemany(
            """
            INSERT OR IGNORE INTO form4_purchases
            (record_id, filing_date, ticker, company, code, shares, pulled_at_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        after = conn.execute("SELECT COUNT(*) FROM form4_purchases").fetchone()[0]

    return len(rows), after - before


def _run_scrape_sync(daily_index_date: str | None = None) -> tuple[pd.DataFrame, str]:
    try:
        return asyncio.run(scrape_purchases(daily_index_date))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(scrape_purchases(daily_index_date))
        finally:
            loop.close()


def _totals_columns(df: pd.DataFrame | None = None):
    if df is not None and not df.empty:
        return [{"name": col, "id": col} for col in df.columns]
    return [
        {"name": "daily_index_date", "id": "daily_index_date"},
        {"name": "ticker", "id": "ticker"},
        {"name": "company", "id": "company"},
        {"name": "total_shares", "id": "total_shares"},
        {"name": "filings", "id": "filings"},
    ]


def _find_ticker_col(columns):
    return next((col for col in columns if str(col).lower() == "ticker"), None)


def build_table(df: pd.DataFrame | None = None):
    if df is None:
        df = query_top_purchases(limit=200)
    if df is None:
        df = pd.DataFrame()

    ticker_col = _find_ticker_col(df.columns)

    return dash_table.DataTable(
        id="speedy-totals-table",
        data=df.to_dict("records"),
        columns=_totals_columns(df),
        page_size=25,
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
        style_data_conditional=(
            [
                {
                    "if": {"column_id": ticker_col},
                    "color": "#0088ff",
                    "textDecoration": "underline",
                    "cursor": "pointer",
                }
            ]
            if ticker_col
            else []
        )
        + [
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
    )


def build_layout():
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H2("EF4 Speedy Pull", className="mb-1"),
                            html.Div(
                                "Fast SEC Form 4 purchase pull with SQLite-backed totals.",
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
                                dbc.CardHeader("Latest Purchases"),
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.ButtonGroup(
                                                        [
                                                            dbc.Button(
                                                                "SQLite Data Load",
                                                                id="speedy-sqlite-load-btn",
                                                                color="secondary",
                                                                n_clicks=0,
                                                            ),
                                                            dbc.Button(
                                                                "Edgar Data Pull",
                                                                id="speedy-edgar-pull-btn",
                                                                color="primary",
                                                                n_clicks=0,
                                                            ),
                                                        ],
                                                        className="w-100",
                                                    ),
                                                    width=12,
                                                    lg=3,
                                                ),
                                                dbc.Col(
                                                    dcc.DatePickerSingle(
                                                        id="speedy-date-input",
                                                        placeholder="YYYY-MM-DD",
                                                        display_format="YYYY-MM-DD",
                                                        max_date_allowed=datetime.now().date(),
                                                        disabled_days=_weekend_disabled_days(),
                                                        style={"width": "100%"},
                                                    ),
                                                    width=12,
                                                    lg=3,
                                                ),
                                                dbc.Col(
                                                    dbc.Button(
                                                        "Pull Selected Date",
                                                        id="speedy-date-pull-btn",
                                                        color="warning",
                                                        className="w-100",
                                                        n_clicks=0,
                                                    ),
                                                    width=12,
                                                    lg=2,
                                                ),
                                                dbc.Col(
                                                    html.Div(
                                                        "Use SQLite Data Load, Edgar Data Pull, or pull a selected SEC daily index date.",
                                                        id="speedy-load-status",
                                                        className="text-muted small mt-2 mt-lg-0",
                                                    ),
                                                    width=12,
                                                    lg=4,
                                                ),
                                            ],
                                            className="g-2 align-items-center p-2",
                                        ),
                                        build_table(),
                                    ],
                                    className="p-0",
                                ),
                            ],
                            className="shadow-sm",
                        ),
                        width=12,
                    )
                ]
            ),
        ],
        fluid=True,
        className="py-4",
    )


layout = build_layout()


def register_callbacks(app):
    @app.callback(
        Output("speedy-totals-table", "data"),
        Output("speedy-totals-table", "columns"),
        Output("speedy-load-status", "children"),
        Input("speedy-sqlite-load-btn", "n_clicks"),
        Input("speedy-edgar-pull-btn", "n_clicks"),
        Input("speedy-date-pull-btn", "n_clicks"),
        State("speedy-date-input", "date"),
        prevent_initial_call=True,
    )
    def load_data(sqlite_clicks, pull_clicks, date_pull_clicks, selected_date):
        if ctx.triggered_id in {"speedy-edgar-pull-btn", "speedy-date-pull-btn"}:
            if ctx.triggered_id == "speedy-date-pull-btn":
                if not selected_date:
                    return (
                        [],
                        _totals_columns(),
                        "Selected date pull failed: choose a date first.",
                    )
                target_date = _normalize_daily_index_date(selected_date)
            else:
                target_date = None

            scraped, daily_index_date = _run_scrape_sync(target_date)
            parquet_path, parquet_rows = save_parquet_snapshot(
                scraped, daily_index_date
            )
            parquet_df = load_parquet_snapshot(parquet_path)
            attempted, inserted = upsert_sqlite_purchases(parquet_df)
            df = query_top_purchases(limit=200, filing_date=daily_index_date)
            action_label = (
                "Selected date pull"
                if ctx.triggered_id == "speedy-date-pull-btn"
                else "Edgar pull"
            )
            return (
                df.to_dict("records"),
                _totals_columns(df),
                f"{action_label} complete for daily index {daily_index_date}. Parquet rows: {parquet_rows}; SQLite attempted: {attempted}, inserted: {inserted}, rows: {len(df)}.",
            )

        df = query_top_purchases(limit=200)
        latest_date = query_latest_loaded_date()
        if df is None or df.empty:
            return [], _totals_columns(df), "SQLite load complete. No rows found."
        return (
            df.to_dict("records"),
            _totals_columns(df),
            f"SQLite load complete for latest daily index {latest_date}. Rows: {len(df)}.",
        )


register_callbacks(get_app())
