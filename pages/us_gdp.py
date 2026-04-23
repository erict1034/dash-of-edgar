# ==========================================================
# US GDP DASHBOARD (Single File)
# Loads API key securely from .env
# ==========================================================


import os
import sqlite3
import pandas as pd
import dash
from dash import get_app, ctx, html, dcc, Input, Output, dash_table, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

dash.register_page(__name__, path="/gdp", name="GDP Dashboard")


# ----------------------------------------------------------
# 1. DATA LOADING HELPERS (assume these are defined elsewhere in the file)
# ----------------------------------------------------------
def load_gdp_data(force_refresh=False):
    from pathlib import Path
    from dotenv import load_dotenv
    from fredapi import Fred

    DATA_DIR = Path(__file__).with_name("data")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PARQUET_PATH = DATA_DIR / "us_gdp.parquet"
    SQLITE_PATH = DATA_DIR / "us_gdp.sqlite"

    def normalize_gdp(df):
        df.columns = df.columns.str.lower()

        df.rename(
            columns={
                "date": "Date",
                "gdp": "GDP",
                "recession": "Recession",
                "gdp_growth_pct": "GDP_Growth_Pct",
            },
            inplace=True,
        )

        if "Date" not in df.columns:
            raise ValueError("Expected 'Date' column not found")

        df["Date"] = pd.to_datetime(df["Date"])
        return df[["Date", "GDP", "Recession", "GDP_Growth_Pct"]]

    # Try SQLite first
    if SQLITE_PATH.exists() and not force_refresh:
        with sqlite3.connect(SQLITE_PATH) as conn:
            df = pd.read_sql_query("SELECT * FROM us_gdp ORDER BY date", conn)

        return normalize_gdp(df)

    # Try Parquet next
    if PARQUET_PATH.exists() and not force_refresh:
        df = pd.read_parquet(PARQUET_PATH)

        return normalize_gdp(df)

    # Fallback: fetch from FRED API
    load_dotenv()
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise ValueError("FRED_API_KEY not found in .env file")
    fred = Fred(api_key=api_key)
    gdp = fred.get_series("GDPC1")
    recession = fred.get_series("USREC")
    df = pd.DataFrame({"Date": gdp.index, "GDP": gdp.values})
    df["Recession"] = recession.reindex(df["Date"]).values
    df.dropna(inplace=True)
    df["GDP_Growth_Pct"] = df["GDP"].pct_change(4) * 100

    # 👉 WRITE CACHE (this is what you're missing)
    with sqlite3.connect(SQLITE_PATH) as conn:
        df.to_sql("us_gdp", conn, if_exists="replace", index=False)

    df.to_parquet(PARQUET_PATH, index=False)
    return normalize_gdp(df)


def create_gdp_figure(df, avg_gap_months: float | None, months_since_last: int | None):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["Date"], y=df["GDP"], mode="lines", name="Real GDP", line=dict(width=3)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["GDP_Growth_Pct"],
            mode="lines",
            name="YoY GDP Growth (%)",
            yaxis="y2",
            line=dict(width=2, dash="dot", color="#2563eb"),
        )
    )

    # ----- Recession shading -----
    in_recession = False
    start_date = None

    for i in range(len(df)):
        if df["Recession"].iloc[i] == 1 and not in_recession:
            in_recession = True
            start_date = df["Date"].iloc[i]

        elif df["Recession"].iloc[i] == 0 and in_recession:
            fig.add_vrect(
                x0=start_date,
                x1=df["Date"].iloc[i],
                fillcolor="LightSalmon",
                opacity=0.25,
                line_width=0,
            )
            in_recession = False

    annotations = []
    if avg_gap_months is not None:
        annotations.append(
            dict(
                x=0.99,
                y=1.16,
                xref="paper",
                yref="paper",
                text=f"Avg months between recessions: {avg_gap_months:.1f}",
                showarrow=False,
                xanchor="right",
                align="right",
                font=dict(size=12, color="#4b5563"),
            )
        )
    if months_since_last is not None:
        annotations.append(
            dict(
                x=0.99,
                y=1.09,
                xref="paper",
                yref="paper",
                text=f"Months since last recession: {months_since_last}",
                showarrow=False,
                xanchor="right",
                align="right",
                font=dict(size=12, color="#4b5563"),
            )
        )

    fig.update_layout(
        title="US Real GDP (Inflation Adjusted)",
        template="plotly_white",
        hovermode="x unified",
        xaxis_title="Date",
        yaxis=dict(title="Billions of Chained 2017 Dollars"),
        yaxis2=dict(
            title="YoY GDP Growth (%)",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        annotations=annotations,
        margin=dict(t=90, l=60, r=60, b=60),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#000000"),
    )
    fig.update_xaxes(gridcolor="#cccccc", zerolinecolor="#cccccc")
    fig.update_yaxes(gridcolor="#cccccc", zerolinecolor="#cccccc")

    return fig


def _compute_recession_metrics(frame: pd.DataFrame) -> tuple[float | None, int | None]:
    recession_starts = []
    recession_ends = []
    prev_recession = 0
    for date, rec in zip(frame["Date"], frame["Recession"]):
        if rec == 1 and prev_recession == 0:
            recession_starts.append(date)
        if rec == 0 and prev_recession == 1:
            recession_ends.append(date)
        prev_recession = rec

    avg_recession_gap_months = None
    gaps = []
    for end_date, start_date in zip(recession_ends, recession_starts[1:]):
        end_month = end_date.year * 12 + end_date.month
        start_month = start_date.year * 12 + start_date.month
        gaps.append(start_month - end_month)
    if gaps:
        avg_recession_gap_months = sum(gaps) / len(gaps)

    months_since_last_recession = None
    if recession_ends:
        last_end = recession_ends[-1]
        current = frame["Date"].max()
        months_since_last_recession = (current.year * 12 + current.month) - (
            last_end.year * 12 + last_end.month
        )

    return avg_recession_gap_months, months_since_last_recession


def _load_data():
    df = None
    try:
        df = load_gdp_data()
    except Exception:
        import numpy as np

        dates = pd.date_range("2010-01-01", periods=64, freq="QE")
        df = pd.DataFrame(
            {
                "Date": dates,
                "GDP": np.random.normal(20000, 500, len(dates)),
                "Recession": [0, 0, 0, 1, 1, 0, 0, 0] * 8,
                "GDP_Growth_Pct": np.random.normal(2, 0.5, len(dates)),
            }
        )
    return df


TABLE_COLUMNS = [
    {"name": "Date", "id": "Date"},
    {"name": "GDP", "id": "GDP", "type": "numeric", "format": {"specifier": ",.0f"}},
    {"name": "Recession", "id": "Recession"},
    {
        "name": "GDP Growth (%)",
        "id": "GDP_Growth_Pct",
        "type": "numeric",
        "format": {"specifier": ".2f"},
    },
]

TABLE_STYLE = {
    "css": [
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
    "style_data_conditional": [
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
    "style_table": {"overflowX": "auto"},
    "style_cell": {
        "fontFamily": "Segoe UI, Arial, sans-serif",
        "fontSize": 12,
        "padding": "6px",
        "textAlign": "left",
    },
    "style_header": {
        "backgroundColor": "#e9ecef",
        "fontWeight": "bold",
        "color": "#1f2937",
    },
}


def build_layout():
    df = _load_data()
    avg_gap, months_since = _compute_recession_metrics(df)
    fig = create_gdp_figure(df, avg_gap, months_since)
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H2("US GDP Dashboard", className="mb-1"),
                            html.Div(
                                "Real GDP and growth with recession shading.",
                                className="text-muted mb-3",
                            ),
                        ],
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Select Date Range"),
                                dbc.CardBody(
                                    [
                                        dcc.RadioItems(
                                            id="gdp-range",
                                            options=[
                                                {"label": v, "value": v}
                                                for v in [
                                                    "4Q",
                                                    "8Q",
                                                    "12Q",
                                                    "1Y",
                                                    "5Y",
                                                    "10Y",
                                                    "MAX",
                                                ]
                                            ],
                                            value="MAX",
                                            inline=True,
                                            inputStyle={"marginRight": "4px"},
                                            labelStyle={
                                                "marginRight": "12px",
                                                "fontWeight": 500,
                                            },
                                        ),
                                    ]
                                ),
                            ],
                            className="shadow-sm mb-3",
                        ),
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Controls"),
                                dbc.CardBody(
                                    [
                                        dbc.ButtonGroup(
                                            [
                                                dbc.Button(
                                                    "SQLite Load",
                                                    id="gdp-sqlite-load-btn",
                                                    color="secondary",
                                                    n_clicks=0,
                                                ),
                                                dbc.Button(
                                                    "Refresh Load",
                                                    id="gdp-refresh-load-btn",
                                                    color="primary",
                                                    n_clicks=0,
                                                ),
                                            ],
                                        )
                                    ]
                                ),
                            ],
                            className="shadow-sm mb-3",
                        )
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("US Real GDP Over Time"),
                                dbc.CardBody(
                                    [
                                        dcc.Graph(
                                            id="gdp-chart",
                                            figure=fig,
                                            style={"height": "500px"},
                                        ),
                                        dcc.Store(
                                            id="gdp-store", data=df.to_dict("records")
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
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Recent GDP Data"),
                            dbc.CardBody(
                                dash_table.DataTable(
                                    id="gdp-table",
                                    columns=TABLE_COLUMNS,
                                    data=df.tail(20).to_dict("records"),
                                    page_size=10,
                                    sort_action="native",
                                    filter_action="native",
                                    **TABLE_STYLE,
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
        ],
        fluid=True,
        className="py-4",
    )


layout = build_layout()


def register_callbacks(app):
    @app.callback(
        Output("gdp-chart", "figure"),
        Output("gdp-table", "data"),
        Output("gdp-store", "data"),
        Input("gdp-sqlite-load-btn", "n_clicks"),
        Input("gdp-refresh-load-btn", "n_clicks"),
        Input("gdp-range", "value"),
        Input("gdp-store", "data"),
    )
    def update_gdp_range(sqlite_clicks, refresh_clicks, range_value, stored):
        if not ctx.triggered_id:
            return no_update, no_update, no_update

        trigger = ctx.triggered_id

        # --- DATA LOADING LOGIC ---
        if trigger == "gdp-sqlite-load-btn":
            df = load_gdp_data(force_refresh=False)  # you define this

        elif trigger == "gdp-refresh-load-btn":
            df = load_gdp_data(force_refresh=True)  # you define this

        else:
            # Range change → use stored data
            if not stored:
                return no_update, no_update, no_update
            df = pd.DataFrame(stored)
        df["Date"] = pd.to_datetime(df["Date"])
        if range_value == "MAX":
            filtered = df
        elif range_value.endswith("Q"):
            filtered = df.tail(int(range_value.replace("Q", "")))
        else:
            cutoff = df["Date"].max() - pd.DateOffset(
                years=int(range_value.replace("Y", ""))
            )
            filtered = df[df["Date"] >= cutoff]
        avg_gap, months_since = _compute_recession_metrics(filtered)
        return (
            create_gdp_figure(filtered, avg_gap, months_since),
            filtered.to_dict("records"),
            df.to_dict("records"),
        )


register_callbacks(get_app())
