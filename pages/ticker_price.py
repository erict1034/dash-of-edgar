# ticker_price_dash.py
"""
Plotly Dash price dashboard using Yahoo Finance (default) or Alpha Vantage.
Bootstrap styling consistent with us_gdp_dash.
"""

import dash
import sqlite3
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import yfinance as yf
from dash import get_app, Input, Output, State, dcc, html, no_update, dash_table, ctx
import dash_bootstrap_components as dbc
from dotenv import load_dotenv
from storage_paths import CENTRAL_SQLITE_PATH, parquet_path


dash.register_page(__name__, path="/ticker-price", name="Ticker Price", order=5)

load_dotenv()


EARNINGS_SQLITE_PATH = CENTRAL_SQLITE_PATH
PARQUET_PATH = parquet_path("ticker_prices")
SQLITE_PATH = CENTRAL_SQLITE_PATH
PARQUET_PATH_AV = parquet_path("ticker_prices_av")
PARQUET_PATH_YF = parquet_path("ticker_prices_yf")

ALPHAVANTAGE_BASE_URL = "https://www.alphavantage.co/query"
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")

YF_RANGE_OPTIONS = [
    {"label": "1D", "value": "1D"},
    {"label": "5D", "value": "5D"},
    {"label": "1M", "value": "1M"},
    {"label": "3M", "value": "3M"},
    {"label": "YTD", "value": "YTD"},
    {"label": "1Y", "value": "1Y"},
    {"label": "5Y", "value": "5Y"},
    {"label": "ALL", "value": "ALL"},
]
AV_RANGE_OPTIONS = [
    {"label": "1D", "value": "1D"},
    {"label": "5D", "value": "5D"},
    {"label": "1M", "value": "1M"},
    {"label": "3M", "value": "3M"},
    {"label": "YTD", "value": "YTD"},
]

BENCHMARK_OPTIONS = [
    {"label": "S&P 500", "value": "^GSPC"},
    {"label": "Nasdaq", "value": "^IXIC"},
    {"label": "Dow", "value": "^DJI"},
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalize_prices(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = df.copy()
    df["ticker"] = ticker.upper()
    df["pulled_at_utc"] = _utc_now_iso()
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
    return df[
        ["ticker", "date", "open", "high", "low", "close", "volume", "pulled_at_utc"]
    ]


def fetch_alphavantage(ticker: str) -> pd.DataFrame:
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": ticker.upper(),
        "outputsize": "compact",
        "apikey": ALPHAVANTAGE_API_KEY,
    }
    response = requests.get(ALPHAVANTAGE_BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    for err_key in ("Error Message", "Note", "Information"):
        if err_key in payload:
            raise RuntimeError(f"Alpha Vantage [{err_key}]: {payload[err_key]}")

    series = payload.get("Time Series (Daily)")
    if not series:
        raise RuntimeError(f"No daily time series returned for {ticker}.")

    df = pd.DataFrame.from_dict(series, orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().rename(
        columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. volume": "volume",
        }
    )
    df["date"] = df.index.strftime("%Y-%m-%d")
    return _normalize_prices(df, ticker)


def fetch_yahoo(ticker: str, period: str = "max") -> pd.DataFrame:
    raw = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if raw.empty:
        raise RuntimeError(f"Yahoo Finance returned no data for {ticker}.")
    raw = raw.reset_index()
    raw.columns = [c.lower() for c in raw.columns]
    raw = raw.rename(
        columns={"date": "date", "stock splits": "splits", "capital gains": "capgains"}
    )
    raw["date"] = pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d")
    return _normalize_prices(
        raw[["date", "open", "high", "low", "close", "volume"]], ticker
    )


def save_parquet(df: pd.DataFrame, path: Path = PARQUET_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_parquet(path)
        existing["date"] = pd.to_datetime(existing["date"])
        df = pd.concat([existing, df]).drop_duplicates(
            subset=["ticker", "date"], keep="last"
        )
    df.to_parquet(path, index=False)


def upsert_sqlite(df: pd.DataFrame, path: Path = SQLITE_PATH) -> tuple[int, int]:
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


def load_prices(ticker: str, path: Path = SQLITE_PATH) -> pd.DataFrame:
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

    if PARQUET_PATH.exists():
        df = pd.read_parquet(PARQUET_PATH)
        df = df[df["ticker"] == ticker.upper()].copy()
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date")

    return pd.DataFrame()


# ----------------------------------------------------------
# Figure builder
# ----------------------------------------------------------
def create_price_figure(
    df: pd.DataFrame,
    ticker: str,
    compare_df: pd.DataFrame | None = None,
    compare_ticker: str | None = None,
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["close"],
            mode="lines",
            name="Close",
            line=dict(width=2, color="#2563eb"),
        )
    )

    # 50-day and 200-day moving averages
    if len(df) >= 50:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["close"].rolling(50).mean(),
                mode="lines",
                name="50-day MA",
                line=dict(width=1.5, dash="dot", color="#f59e0b"),
            )
        )
    if len(df) >= 200:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["close"].rolling(200).mean(),
                mode="lines",
                name="200-day MA",
                line=dict(width=1.5, dash="dash", color="#10b981"),
            )
        )

    if compare_df is not None and not compare_df.empty and compare_ticker:
        fig.add_trace(
            go.Scatter(
                x=compare_df["date"],
                y=compare_df["close"],
                mode="lines",
                name=f"{compare_ticker} Close",
                line=dict(width=2, color="#ef4444"),
            )
        )

    # Volume on secondary axis
    fig.add_trace(
        go.Bar(
            x=df["date"],
            y=df["volume"],
            name="Volume",
            yaxis="y2",
            marker_color="rgba(14,116,144,0.25)",
        )
    )
    if len(df) >= 50:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["volume"].rolling(50).mean(),
                mode="lines",
                name="50-day Vol MA",
                line=dict(width=1.5, dash="dot", color="#0ea5a4"),
                yaxis="y2",
            )
        )
    if len(df) >= 200:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["volume"].rolling(200).mean(),
                mode="lines",
                name="200-day Vol MA",
                line=dict(width=1.5, dash="dash", color="#0891b2"),
                yaxis="y2",
            )
        )

    title = f"{ticker} Daily Price"
    if compare_ticker:
        title = f"{ticker} vs {compare_ticker} Daily Price"
    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        xaxis_title="Date",
        yaxis=dict(title="Price (USD)"),
        yaxis2=dict(
            title="Volume",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=90, l=60, r=60, b=60),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#000000"),
    )
    fig.update_xaxes(gridcolor="#cccccc", zerolinecolor="#cccccc")
    fig.update_yaxes(gridcolor="#cccccc", zerolinecolor="#cccccc")
    return fig


def _filter_by_range(df: pd.DataFrame, range_value: str) -> pd.DataFrame:
    if range_value in {"ALL", "MAX"} or df.empty:
        return df
    max_date = df["date"].max()
    if range_value == "YTD":
        cutoff = pd.Timestamp(year=max_date.year, month=1, day=1)
        return df[df["date"] >= cutoff]
    if range_value.endswith("D"):
        days = int(range_value.replace("D", ""))
        cutoff = max_date - pd.Timedelta(days=days)
        return df[df["date"] >= cutoff]
    if range_value.endswith("M"):
        months = int(range_value.replace("M", ""))
        cutoff = max_date - pd.DateOffset(months=months)
    else:
        years = int(range_value.replace("Y", ""))
        cutoff = max_date - pd.DateOffset(years=years)
    return df[df["date"] >= cutoff]


def _summary_stats(df: pd.DataFrame) -> tuple[str, str, str, str]:
    if df.empty:
        return "—", "—", "—", "—"
    latest = f"${df['close'].iloc[-1]:,.2f}"
    high = f"${df['high'].max():,.2f}"
    low = f"${df['low'].min():,.2f}"
    chg = df["close"].iloc[-1] - df["close"].iloc[0]
    chg_pct = (chg / df["close"].iloc[0]) * 100 if df["close"].iloc[0] else 0
    change = f"${chg:+,.2f} ({chg_pct:+.1f}%)"
    return latest, high, low, change


def _format_stat(
    primary_label: str,
    primary_value: str,
    compare_label: str | None = None,
    compare_value: str | None = None,
):
    if compare_label and compare_value:
        return html.Div(
            [
                html.Div(f"{primary_label}: {primary_value}", className="fw-semibold"),
                html.Div(f"{compare_label}: {compare_value}", className="text-muted"),
            ]
        )
    return primary_value


def _parse_ticker_list(raw: str | None, fallback: list[str] | None = None) -> list[str]:
    if not raw:
        return fallback or []
    tokens = [
        t.strip().upper()
        for t in raw.replace("\n", ",").replace(" ", ",").split(",")
        if t.strip()
    ]
    # De-duplicate while preserving order
    seen: set[str] = set()
    unique = []
    for t in tokens:
        if t not in seen:
            unique.append(t)
            seen.add(t)
    return unique


def _build_normalized_compare_figure(
    price_frames: dict[str, pd.DataFrame],
) -> go.Figure:
    fig = go.Figure()
    for ticker, df in price_frames.items():
        if df.empty:
            continue
        base = df["close"].iloc[0]
        if base == 0:
            continue
        norm = (df["close"] / base) * 100
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=norm,
                mode="lines",
                name=ticker,
                line=dict(width=2),
            )
        )
    fig.update_layout(
        title="Normalized Performance (Base = 100)",
        template="plotly_white",
        hovermode="x unified",
        xaxis_title="Date",
        yaxis=dict(title="Indexed Price"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=90, l=60, r=60, b=60),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#000000"),
    )
    fig.update_xaxes(gridcolor="#cccccc", zerolinecolor="#cccccc")
    fig.update_yaxes(gridcolor="#cccccc", zerolinecolor="#cccccc")
    return fig


def _compute_metrics(df: pd.DataFrame) -> dict[str, str]:
    if df.empty:
        return {
            "Start": "—",
            "End": "—",
            "Change": "—",
            "Volatility": "—",
            "Max Drawdown": "—",
        }
    start = df["close"].iloc[0]
    end = df["close"].iloc[-1]
    chg = end - start
    chg_pct = (chg / start) * 100 if start else 0
    returns = df["close"].pct_change().dropna()
    vol = returns.std() * 100 if not returns.empty else 0
    cum = (df["close"] / start).cummax()
    drawdown = (df["close"] / cum - 1).min() * 100 if start else 0
    return {
        "Start": f"${start:,.2f}",
        "End": f"${end:,.2f}",
        "Change": f"{chg_pct:+.1f}%",
        "Volatility": f"{vol:.2f}%",
        "Max Drawdown": f"{drawdown:.1f}%",
    }


def _load_earnings_quality_latest(ticker: str) -> dict[str, str]:
    if not EARNINGS_SQLITE_PATH.exists():
        return {
            "Earnings Period": "—",
            "Accrual Ratio": "—",
            "Net Income": "—",
            "Operating Cash Flow": "—",
        }
    sql = """
    SELECT
        period_end,
        net_income,
        operating_cash_flow,
        accrual_ratio
    FROM earnings_quality
    WHERE ticker = ?
    ORDER BY period_end DESC
    LIMIT 1
    """
    with sqlite3.connect(EARNINGS_SQLITE_PATH) as conn:
        row = conn.execute(sql, (ticker.upper(),)).fetchone()
    if not row:
        return {
            "Earnings Period": "—",
            "Accrual Ratio": "—",
            "Net Income": "—",
            "Operating Cash Flow": "—",
        }
    period_end, net_income, operating_cash_flow, accrual_ratio = row
    return {
        "Earnings Period": str(period_end),
        "Accrual Ratio": f"{accrual_ratio:,.3f}" if accrual_ratio is not None else "—",
        "Net Income": f"${net_income:,.0f}" if net_income is not None else "—",
        "Operating Cash Flow": (
            f"${operating_cash_flow:,.0f}" if operating_cash_flow is not None else "—"
        ),
    }


def _load_or_fetch_prices(ticker: str, source: str) -> pd.DataFrame:
    if source == "av":
        parquet_path = PARQUET_PATH_AV
        sqlite_path = SQLITE_PATH
    else:
        parquet_path = PARQUET_PATH_YF
        sqlite_path = SQLITE_PATH

    df = load_prices(ticker, path=sqlite_path)
    if not df.empty:
        return df
    if source == "av":
        df = fetch_alphavantage(ticker)
    else:
        df = fetch_yahoo(ticker)
    save_parquet(df, path=parquet_path)
    upsert_sqlite(df, path=sqlite_path)
    return load_prices(ticker, path=sqlite_path)


# ----------------------------------------------------------
# App layout
# ----------------------------------------------------------
def build_layout():
    return dbc.Container(
        [
            # Header
            dbc.Row(
                dbc.Col(
                    [
                        html.H2("Ticker Price Dashboard", className="mb-1"),
                        html.Div(
                            "Daily OHLCV prices via Yahoo Finance (default) or Alpha Vantage.",
                            className="text-muted mb-3",
                        ),
                    ],
                    width=12,
                ),
                className="mb-2",
            ),
            # Input row
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Input(
                            id="ticker-input",
                            placeholder="Enter ticker (e.g. AAPL)",
                            type="text",
                            debounce=False,
                        ),
                        md=3,
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Advanced",
                            id="advanced-toggle",
                            color="secondary",
                            outline=True,
                            size="sm",
                        ),
                        md="auto",
                    ),
                    dbc.Collapse(
                        dbc.Col(
                            dcc.Dropdown(
                                id="source-select",
                                options=[
                                    {"label": "Yahoo Finance (default)", "value": "yf"},
                                    {
                                        "label": "Alpha Vantage (free: ~100 days)",
                                        "value": "av",
                                    },
                                ],
                                value="yf",
                                clearable=False,
                                style={"minWidth": "220px"},
                            ),
                            md="auto",
                        ),
                        id="advanced-collapse",
                        is_open=False,
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Load", id="ticker-load-btn", color="primary", n_clicks=0
                        ),
                        md="auto",
                    ),
                ],
                className="mb-3 align-items-center",
            ),
            # Watchlist row
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Textarea(
                            id="watchlist-input",
                            placeholder="Watchlist (up to 20 tickers, comma/space/newline separated)",
                            rows=2,
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Load Watchlist",
                            id="watchlist-load-btn",
                            color="info",
                            n_clicks=0,
                        ),
                        md="auto",
                    ),
                    dbc.Col(
                        html.Div(
                            id="watchlist-status-msg", className="text-muted small"
                        ),
                        md=True,
                    ),
                ],
                className="mb-3 align-items-center",
            ),
            # Watchlist table
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                "Watchlist (Click a row to load the chart; check Compare to include)"
                            ),
                            dbc.CardBody(
                                dash_table.DataTable(
                                    id="watchlist-table",
                                    columns=[
                                        {
                                            "name": "Ticker",
                                            "id": "ticker",
                                            "editable": False,
                                        },
                                        {
                                            "name": "Latest",
                                            "id": "latest",
                                            "editable": False,
                                        },
                                        {
                                            "name": "Change",
                                            "id": "change",
                                            "editable": False,
                                        },
                                        {
                                            "name": "Volatility",
                                            "id": "volatility",
                                            "editable": False,
                                        },
                                        {
                                            "name": "Max Drawdown",
                                            "id": "max_drawdown",
                                            "editable": False,
                                        },
                                        {
                                            "name": "Earnings Period",
                                            "id": "earnings_period",
                                            "editable": False,
                                        },
                                        {
                                            "name": "Accrual Ratio",
                                            "id": "accrual_ratio",
                                            "editable": False,
                                        },
                                        {
                                            "name": "Net Income",
                                            "id": "net_income",
                                            "editable": False,
                                        },
                                        {
                                            "name": "Operating Cash Flow",
                                            "id": "operating_cash_flow",
                                            "editable": False,
                                        },
                                        {
                                            "name": "Compare",
                                            "id": "compare",
                                            "type": "any",
                                            "editable": True,
                                        },
                                    ],
                                    data=[],
                                    editable=True,
                                    row_selectable="single",
                                    dropdown={
                                        "compare": {
                                            "options": [
                                                {"label": "Yes", "value": True},
                                                {"label": "No", "value": False},
                                            ]
                                        }
                                    },
                                    style_table={"overflowX": "auto"},
                                    style_cell={
                                        "fontSize": 12,
                                        "padding": "6px",
                                        "whiteSpace": "nowrap",
                                    },
                                    style_header={
                                        "fontWeight": "600",
                                        "backgroundColor": "#f8fafc",
                                    },
                                ),
                                className="p-0",
                            ),
                        ],
                        className="shadow-sm",
                    ),
                    width=12,
                ),
                className="mb-4",
            ),
            # Comparison row
            dbc.Row(
                [],
                className="mb-3 align-items-center",
            ),
            dcc.Store(id="status-store"),
            # Status
            dbc.Row(
                dbc.Col(
                    html.Div(id="status-msg", className="text-muted small"), width=12
                ),
                className="mb-3",
            ),
            # Summary cards
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Latest Close"),
                                dbc.CardBody(html.H4(id="stat-latest", children="—")),
                            ],
                            className="shadow-sm",
                        ),
                        md=3,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Period High"),
                                dbc.CardBody(html.H4(id="stat-high", children="—")),
                            ],
                            className="shadow-sm",
                        ),
                        md=3,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Period Low"),
                                dbc.CardBody(html.H4(id="stat-low", children="—")),
                            ],
                            className="shadow-sm",
                        ),
                        md=3,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Period Change"),
                                dbc.CardBody(html.H4(id="stat-change", children="—")),
                            ],
                            className="shadow-sm",
                        ),
                        md=3,
                    ),
                ],
                className="mb-4",
            ),
            # Chart
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dcc.RadioItems(
                                                id="range-select",
                                                options=YF_RANGE_OPTIONS,
                                                value="ALL",
                                                inline=True,
                                                inputStyle={"marginRight": "4px"},
                                                labelStyle={
                                                    "marginRight": "12px",
                                                    "fontWeight": 500,
                                                },
                                            ),
                                        ),
                                    ],
                                    className="g-2 align-items-center",
                                ),
                                className="py-2",
                            ),
                            dbc.CardBody(
                                dcc.Graph(id="price-chart", style={"height": "500px"})
                            ),
                        ],
                        className="shadow-sm",
                    ),
                    width=12,
                ),
                className="mb-4",
            ),
            # Multi-compare chart
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Textarea(
                                                id="compare-list-input",
                                                placeholder="Compare up to 10 tickers (comma, space, or newline separated)",
                                                rows=2,
                                            ),
                                        ),
                                        dbc.Col(
                                            dcc.RadioItems(
                                                id="compare-range-select",
                                                options=YF_RANGE_OPTIONS,
                                                value="ALL",
                                                inline=True,
                                                inputStyle={"marginRight": "4px"},
                                                labelStyle={
                                                    "marginRight": "12px",
                                                    "fontWeight": 500,
                                                },
                                            ),
                                        ),
                                    ],
                                    className="g-2 align-items-center",
                                ),
                                className="py-2",
                            ),
                            dbc.CardBody(
                                dcc.Graph(id="compare-chart", style={"height": "450px"})
                            ),
                        ],
                        className="shadow-sm",
                    ),
                    width=12,
                ),
                className="mb-4",
            ),
            # Comparison metrics table
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Comparison Metrics"),
                            dbc.CardBody(
                                html.Div(id="compare-table-container"), className="p-0"
                            ),
                        ],
                        className="shadow-sm",
                    ),
                    width=12,
                ),
                className="mt-4",
            ),
        ],
        fluid=True,
        className="py-4",
    )


layout = build_layout()


def register_callbacks(app):
    if hasattr(app, "_ticker_price_callbacks_registered"):
        return
    app._ticker_price_callbacks_registered = True

    @app.callback(
        Output("advanced-collapse", "is_open"),
        Input("advanced-toggle", "n_clicks"),
        State("advanced-collapse", "is_open"),
    )
    def toggle_advanced(n_clicks, is_open):
        if not n_clicks:
            return is_open
        return not is_open

    @app.callback(
        Output("range-select", "options"),
        Output("range-select", "value"),
        Input("source-select", "value"),
        State("range-select", "value"),
    )
    def update_range_options(source, current_value):
        options = AV_RANGE_OPTIONS if source == "av" else YF_RANGE_OPTIONS
        allowed = {opt["value"] for opt in options}
        default_value = "3M" if source == "av" else "ALL"
        next_value = current_value if current_value in allowed else default_value
        return options, next_value

    @app.callback(
        Output("watchlist-table", "data"),
        Output("watchlist-table", "selected_rows"),
        Output("watchlist-status-msg", "children"),
        Input("watchlist-load-btn", "n_clicks"),
        State("watchlist-input", "value"),
        State("range-select", "value"),
        State("source-select", "value"),
        prevent_initial_call=True,
    )
    def update_watchlist_table(n_clicks, watchlist_input, range_value, source):
        tickers = _parse_ticker_list(watchlist_input)
        if not tickers:
            return [], [], "Enter up to 20 tickers to load a watchlist."
        tickers = tickers[:20]

        rows = []
        for t in tickers:
            try:
                df = _load_or_fetch_prices(t, source)
            except Exception:
                df = pd.DataFrame()
            df = _filter_by_range(df, range_value) if not df.empty else df
            metrics = _compute_metrics(df)
            latest = f"${df['close'].iloc[-1]:,.2f}" if not df.empty else "—"
            earnings = _load_earnings_quality_latest(t)
            rows.append(
                {
                    "ticker": t,
                    "latest": latest,
                    "change": metrics["Change"],
                    "volatility": metrics["Volatility"],
                    "max_drawdown": metrics["Max Drawdown"],
                    "earnings_period": earnings["Earnings Period"],
                    "accrual_ratio": earnings["Accrual Ratio"],
                    "net_income": earnings["Net Income"],
                    "operating_cash_flow": earnings["Operating Cash Flow"],
                    "compare": False,
                }
            )

        msg = f"Loaded {len(rows)} tickers. Click a row to view the chart."
        return rows, [0] if rows else [], msg

    @app.callback(
        Output("price-chart", "figure"),
        Output("stat-latest", "children"),
        Output("stat-high", "children"),
        Output("stat-low", "children"),
        Output("stat-change", "children"),
        Output("status-store", "data"),
        Output("compare-chart", "figure"),
        Output("compare-table-container", "children"),
        Input("watchlist-table", "data"),
        Input("watchlist-table", "selected_rows"),
        Input("ticker-load-btn", "n_clicks"),
        Input("range-select", "value"),
        Input("compare-list-input", "value"),
        Input("compare-range-select", "value"),
        State("ticker-input", "value"),
        State("source-select", "value"),
        prevent_initial_call=True,
    )
    def update_dashboard(
        watchlist_data,
        watchlist_selected_rows,
        n_clicks,
        range_value,
        compare_list_input,
        compare_range_value,
        ticker_input,
        source,
    ):

        source = source or "yf"
        ticker = (ticker_input or "").strip().upper()
        if watchlist_data and watchlist_selected_rows:
            row_idx = watchlist_selected_rows[0]
            if 0 <= row_idx < len(watchlist_data):
                ticker = str(watchlist_data[row_idx].get("ticker", ticker)).upper()
        if not ticker:
            empty = go.Figure()
            return (
                empty,
                "--",
                "--",
                "--",
                "--",
                "Enter a ticker and click Load.",
                empty,
                no_update,
            )

        df = pd.DataFrame()
        status = ""
        try:
            df = _load_or_fetch_prices(ticker, source)
            source_label = "Alpha Vantage" if source == "av" else "Yahoo Finance"
            status = f"{source_label}: {len(df)} rows loaded."
        except Exception as exc:
            sqlite_path = SQLITE_PATH
            df = load_prices(ticker, path=sqlite_path)
            if df.empty:
                empty = go.Figure()
                empty.update_layout(title=f"Error loading {ticker}: {exc}")
                return empty, "--", "--", "--", "--", f"Error: {exc}", empty, no_update
            status = f"Fetch error ({exc}) — loaded {len(df)} rows from sqlite."

        if df.empty:
            empty = go.Figure()
            empty.update_layout(title=f"No data for {ticker}.")
            return (
                empty,
                "--",
                "--",
                "--",
                "--",
                f"No data returned for {ticker}.",
                empty,
                no_update,
            )

        filtered = _filter_by_range(df, range_value)
        fig = create_price_figure(filtered, ticker)
        latest, high, low, change = _summary_stats(filtered)
        latest_display = _format_stat(ticker, latest)
        high_display = _format_stat(ticker, high)
        low_display = _format_stat(ticker, low)
        change_display = _format_stat(ticker, change)

        # Multi-ticker comparison (up to 10)
        compare_list: list[str] = []
        if watchlist_data:
            compare_list = [
                str(row.get("ticker", "")).upper()
                for row in watchlist_data
                if row.get("compare")
            ]
            compare_list = [t for t in compare_list if t]
        if not compare_list:
            compare_list = _parse_ticker_list(compare_list_input, fallback=[])
        if not compare_list:
            compare_list = [ticker]
        if ticker not in compare_list:
            compare_list = [ticker] + compare_list
        compare_list = compare_list[:10]

        price_frames: dict[str, pd.DataFrame] = {}
        compare_range = compare_range_value or "ALL"
        for t in compare_list:
            try:
                t_df = _load_or_fetch_prices(t, source)
            except Exception:
                t_df = load_prices(t, path=SQLITE_PATH)
            t_df = _filter_by_range(t_df, compare_range)
            if not t_df.empty:
                price_frames[t] = t_df

        for opt in BENCHMARK_OPTIONS:
            try:
                b_df = _load_or_fetch_prices(opt["value"], "yf")
            except Exception:
                b_df = load_prices(opt["value"], path=SQLITE_PATH)
            b_df = _filter_by_range(b_df, compare_range) if not b_df.empty else b_df
            if not b_df.empty:
                price_frames[opt["label"]] = b_df
        if not any(
            label in price_frames
            for label in [opt["label"] for opt in BENCHMARK_OPTIONS]
        ):
            status = (
                f"{status} (Benchmark data unavailable for selected index.)".strip()
            )

        compare_fig = (
            _build_normalized_compare_figure(price_frames)
            if price_frames
            else go.Figure()
        )

        metrics_rows = []
        for t, t_df in price_frames.items():
            metrics = _compute_metrics(t_df)
            metrics_rows.append(
                {
                    "Ticker": t,
                    "Start": metrics["Start"],
                    "End": metrics["End"],
                    "Change": metrics["Change"],
                    "Volatility": metrics["Volatility"],
                    "Max Drawdown": metrics["Max Drawdown"],
                }
            )
        metrics_df = pd.DataFrame(metrics_rows)
        compare_table = (
            dbc.Table.from_dataframe(
                metrics_df,
                striped=True,
                bordered=True,
                hover=True,
                responsive=True,
                style={"fontSize": 12},
            )
            if not metrics_df.empty
            else no_update
        )

        return (
            fig,
            latest_display,
            high_display,
            low_display,
            change_display,
            status,
            compare_fig,
            compare_table,
        )

    @app.callback(
        Output("status-msg", "children"),
        Input("status-store", "data"),
    )
    def render_status(data):
        return data or ""


register_callbacks(get_app())
