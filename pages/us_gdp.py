# ==========================================================
# US GDP DASHBOARD (FULL VERSION - Pages Safe)
# ==========================================================

import os
import sqlite3
import pandas as pd
import dash
from dash import html, dcc, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

dash.register_page(__name__, path="/gdp", name="GDP Dashboard")

# ----------------------------------------------------------
# DATA
# ----------------------------------------------------------


def load_gdp_data():
    from pathlib import Path
    from dotenv import load_dotenv
    from fredapi import Fred  # ✅ lazy import

    DATA_DIR = Path(__file__).with_name("data")
    PARQUET_PATH = DATA_DIR / "us_gdp.parquet"
    SQLITE_PATH = DATA_DIR / "us_gdp.sqlite"

    if SQLITE_PATH.exists():
        with sqlite3.connect(SQLITE_PATH) as conn:
            df = pd.read_sql_query("SELECT * FROM us_gdp ORDER BY date", conn)
            df["Date"] = pd.to_datetime(df["date"])
            df["GDP"] = df["gdp"]
            df["Recession"] = df["recession"]
            df["GDP_Growth_Pct"] = df["gdp_growth_pct"]
            return df[["Date", "GDP", "Recession", "GDP_Growth_Pct"]]

    if PARQUET_PATH.exists():
        df = pd.read_parquet(PARQUET_PATH)
        df["Date"] = pd.to_datetime(df["date"])
        df["GDP"] = df["gdp"]
        df["Recession"] = df["recession"]
        df["GDP_Growth_Pct"] = df["gdp_growth_pct"]
        return df[["Date", "GDP", "Recession", "GDP_Growth_Pct"]]

    # FRED fallback
    load_dotenv()
    api_key = os.getenv("FRED_API_KEY")

    fred = Fred(api_key=api_key)

    gdp = fred.get_series("GDPC1")
    rec = fred.get_series("USREC")

    df = pd.DataFrame({"Date": gdp.index, "GDP": gdp.values})
    df["Recession"] = rec.reindex(df["Date"]).values
    df.dropna(inplace=True)

    df["GDP_Growth_Pct"] = df["GDP"].pct_change(4) * 100

    return df[["Date", "GDP", "Recession", "GDP_Growth_Pct"]]


def compute_metrics(df):
    starts, ends = [], []
    prev = 0

    for d, r in zip(df["Date"], df["Recession"]):
        if r == 1 and prev == 0:
            starts.append(d)
        if r == 0 and prev == 1:
            ends.append(d)
        prev = r

    gaps = []
    for end, start in zip(ends, starts[1:]):
        gaps.append((start.year * 12 + start.month) - (end.year * 12 + end.month))

    avg_gap = sum(gaps) / len(gaps) if gaps else None

    months_since = None
    if ends:
        last = ends[-1]
        now = df["Date"].max()
        months_since = (now.year * 12 + now.month) - (last.year * 12 + last.month)

    return avg_gap, months_since


def build_figure(df):
    avg_gap, months_since = compute_metrics(df)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=df["Date"], y=df["GDP"], name="Real GDP", line=dict(width=3))
    )

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["GDP_Growth_Pct"],
            name="YoY Growth (%)",
            yaxis="y2",
            line=dict(dash="dot"),
        )
    )

    # recession shading
    in_rec = False
    start = None

    for i in range(len(df)):
        if df["Recession"].iloc[i] == 1 and not in_rec:
            in_rec = True
            start = df["Date"].iloc[i]

        elif df["Recession"].iloc[i] == 0 and in_rec:
            fig.add_vrect(
                x0=start,
                x1=df["Date"].iloc[i],
                fillcolor="LightSalmon",
                opacity=0.25,
                line_width=0,
            )
            in_rec = False

    annotations = []

    if avg_gap:
        annotations.append(
            dict(
                x=0.99,
                y=1.15,
                xref="paper",
                yref="paper",
                text=f"Avg months between recessions: {avg_gap:.1f}",
                showarrow=False,
                xanchor="right",
            )
        )

    if months_since:
        annotations.append(
            dict(
                x=0.99,
                y=1.08,
                xref="paper",
                yref="paper",
                text=f"Months since last recession: {months_since}",
                showarrow=False,
                xanchor="right",
            )
        )

    fig.update_layout(
        title="US Real GDP",
        hovermode="x unified",
        yaxis=dict(title="GDP"),
        yaxis2=dict(title="Growth %", overlaying="y", side="right"),
        annotations=annotations,
    )

    return fig


# ----------------------------------------------------------
# LAYOUT
# ----------------------------------------------------------

layout = dbc.Container(
    [
        html.H2("US GDP Dashboard"),
        dbc.Button("Load GDP Data", id="gdp-load-btn", color="primary"),
        dcc.RadioItems(
            id="gdp-range",
            options=[{"label": v, "value": v} for v in ["4Q", "8Q", "1Y", "5Y", "MAX"]],
            value="8Q",
            inline=True,
        ),
        dcc.Store(id="gdp-store"),
        dcc.Graph(id="gdp-chart"),
        dash_table.DataTable(id="gdp-table", page_size=10),
    ],
    fluid=True,
)

# ----------------------------------------------------------
# CALLBACKS
# ----------------------------------------------------------


@dash.callback(
    Output("gdp-store", "data"),
    Input("gdp-load-btn", "n_clicks"),
    prevent_initial_call=True,
)
def load_data(n):
    df = load_gdp_data()
    return df.to_dict("records")


@dash.callback(
    Output("gdp-chart", "figure"),
    Output("gdp-table", "data"),
    Input("gdp-range", "value"),
    Input("gdp-store", "data"),
)
def update(range_val, stored):

    if not stored:
        return go.Figure(), []

    df = pd.DataFrame(stored)
    df["Date"] = pd.to_datetime(df["Date"])

    if range_val == "MAX":
        filtered = df
    elif range_val.endswith("Q"):
        filtered = df.tail(int(range_val.replace("Q", "")))
    else:
        cutoff = df["Date"].max() - pd.DateOffset(years=int(range_val.replace("Y", "")))
        filtered = df[df["Date"] >= cutoff]

    fig = build_figure(filtered)

    return fig, filtered.tail(20).to_dict("records")
