# oil_macro_dashboard.py
# -------------------------------------------------------
# Macro Oil Dashboard
# FRED -> Parquet -> SQLite -> PyTorch -> Plotly Dash
# -------------------------------------------------------

from os import getenv
from pathlib import Path
import requests
import sqlite3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
import dash
from dash import get_app, dcc, html, Input, Output, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

# =============================
# CONFIG
# =============================


dash.register_page(__name__, path="/oil-macro", name="Oil Macro Dashboard")


load_dotenv()

DEVICE = "cuda"
FRED_API_KEY = getenv("FRED_API_KEY")
if not FRED_API_KEY:
    raise RuntimeError("FRED_API_KEY is not set. Add it to your .env file.")
HORMUZ_START_DATE = pd.Timestamp("2026-02-28")
HORMUZ_TRANSIT_SHARE = 0.20

SERIES = {
    "wti": "DCOILWTICO",
    "industrial_production": "INDPRO",
    "real_gdp": "GDPC1",
    "us_oil_production": "IPG211S",  # your corrected label
    "oil_inventories": "A33DTI",  # your corrected label
    "10y_yield": "DGS10",
}

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PARQUET_FILE = DATA_DIR / "oil_macro.parquet"
SQLITE_FILE = DATA_DIR / "oil_macro.db"

# =============================
# FRED FETCH
# =============================


def fetch_fred_series(series_id):

    url = "https://api.stlouisfed.org/fred/series/observations"

    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
    }

    r = requests.get(url, params=params)
    data = r.json()

    if "observations" not in data:
        raise ValueError(data)

    df = pd.DataFrame(data["observations"])
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    return df.set_index("date")["value"]


# =============================
# DATA COLLECTION
# =============================


def collect_data():

    print("Collecting macro oil data...")

    dfs = []

    for name, sid in SERIES.items():
        print("Downloading", name)
        s = fetch_fred_series(sid)
        s.name = name
        dfs.append(s)

    data = pd.concat(dfs, axis=1, sort=False)

    return data


# =============================
# FREQUENCY NORMALIZATION
# =============================


def normalize_frequency(df):

    df.index = pd.to_datetime(df.index)

    # weekly alignment (oil-friendly frequency)
    df = df.resample("W-FRI").last()

    df = df.ffill()

    return df


# =============================
# FEATURE ENGINEERING
# =============================


def build_features(df):

    df["demand_growth"] = df["industrial_production"].pct_change(12)

    df["supply_growth"] = df["us_oil_production"].pct_change(12)

    inv_mean = df["oil_inventories"].rolling(52).mean()
    inv_std = df["oil_inventories"].rolling(52).std()

    df["inventory_z"] = (df["oil_inventories"] - inv_mean) / inv_std

    df["real_rate_proxy"] = df["10y_yield"]

    # Regime feature for the Strait of Hormuz shutdown shock.
    df["hormuz_shock"] = (df.index >= HORMUZ_START_DATE).astype(float)

    # Weeks since shutdown to allow the model to learn persistence/decay.
    weeks_since = np.maximum((df.index - HORMUZ_START_DATE).days / 7.0, 0.0)
    df["hormuz_weeks_since"] = weeks_since

    # Supply shock proxy: 20% transit loss scaled by active-shock regime.
    df["hormuz_supply_loss"] = HORMUZ_TRANSIT_SHARE * df["hormuz_shock"]

    # Inventory draw pressure proxy during disruption windows.
    df["hormuz_inventory_draw"] = df["inventory_z"] - (
        HORMUZ_TRANSIT_SHARE * df["hormuz_shock"]
    )

    return df


# =============================
# PYTORCH MODEL
# =============================

BASE_FEATURES = [
    "demand_growth",
    "supply_growth",
    "inventory_z",
    "real_rate_proxy",
]

SHOCK_FEATURES = BASE_FEATURES + [
    "hormuz_shock",
    "hormuz_weeks_since",
    "hormuz_supply_loss",
    "hormuz_inventory_draw",
]


class OilModel(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_model(df, feature_cols):

    print("Training PyTorch model...")

    train_df = df.copy()
    train_df["wti_future"] = train_df["wti"].shift(-4)
    train_df = train_df.dropna(subset=feature_cols + ["wti_future"])

    X = torch.tensor(train_df[feature_cols].values, dtype=torch.float32).to(DEVICE)
    y = (
        torch.tensor(train_df["wti_future"].values, dtype=torch.float32)
        .view(-1, 1)
        .to(DEVICE)
    )

    model = OilModel(len(feature_cols)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(300):
        pred = model(X)
        loss = loss_fn(pred, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 50 == 0:
            print("Epoch", epoch, "Loss:", loss.item())

    return model


def add_fair_value(df, model, feature_cols, fair_value_col, mispricing_col):

    DEVICE = next(model.parameters()).device

    out = df.dropna(subset=feature_cols).copy()

    X = torch.tensor(out[feature_cols].values, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        preds = model(X)

    out[fair_value_col] = preds.cpu().numpy().flatten()

    out[mispricing_col] = out["wti"] - out[fair_value_col]

    return out


# =============================
# STORAGE
# =============================


def save_data(df):

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df.to_parquet(PARQUET_FILE)
    print("Saved parquet")

    conn = sqlite3.connect(SQLITE_FILE)
    df.to_sql("oil_macro", conn, if_exists="replace")
    conn.close()

    print("Saved SQLite DB")


# =============================
# DASH APP
# =============================


PERIOD_OPTIONS = [
    {"label": "1D", "value": "1D"},
    {"label": "5D", "value": "5D"},
    {"label": "1M", "value": "1M"},
    {"label": "3M", "value": "3M"},
    {"label": "6M", "value": "6M"},
    {"label": "YTD", "value": "YTD"},
    {"label": "2026", "value": "2026"},
    {"label": "1Y", "value": "1Y"},
    {"label": "2Y", "value": "2Y"},
    {"label": "5Y", "value": "5Y"},
    {"label": "All", "value": "all"},
]

MODEL_OPTIONS = [
    {"label": "Shock-Aware", "value": "shock"},
    {"label": "Baseline", "value": "base"},
    {"label": "Both", "value": "both"},
]


def _apply_period(df, period):
    today = pd.Timestamp.today()
    if period == "1D":
        start = today - pd.DateOffset(days=1)
    elif period == "5D":
        start = today - pd.DateOffset(days=5)
    elif period == "1M":
        start = today - pd.DateOffset(months=1)
    elif period == "3M":
        start = today - pd.DateOffset(months=3)
    elif period == "6M":
        start = today - pd.DateOffset(months=6)
    elif period == "YTD":
        start = pd.Timestamp(today.year, 1, 1)
    elif period == "2026":
        start = pd.Timestamp("2026-01-01")
    elif period == "1Y":
        start = today - pd.DateOffset(years=1)
    elif period == "2Y":
        start = today - pd.DateOffset(years=2)
    elif period == "5Y":
        start = today - pd.DateOffset(years=5)
    else:
        return df
    return df[df.index >= start]


def _empty_figure(title: str, message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        margin=dict(t=50, b=20),
        annotations=[
            dict(
                text=message,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=14, color="#6b7280"),
            )
        ],
    )
    return fig


_OIL_DF_CACHE = None


def _build_model_frame():

    df = collect_data()
    df = normalize_frequency(df)
    df = build_features(df)

    base_model = train_model(df, BASE_FEATURES)
    shock_model = train_model(df, SHOCK_FEATURES)

    df_base = add_fair_value(
        df,
        base_model,
        BASE_FEATURES,
        fair_value_col="oil_fair_value_base",
        mispricing_col="mispricing_base",
    )
    df_shock = add_fair_value(
        df,
        shock_model,
        SHOCK_FEATURES,
        fair_value_col="oil_fair_value_shock",
        mispricing_col="mispricing_shock",
    )

    df = df_base.join(
        df_shock[["oil_fair_value_shock", "mispricing_shock"]],
        how="inner",
    )

    save_data(df)
    return df


def get_oil_macro_df():

    global _OIL_DF_CACHE

    if _OIL_DF_CACHE is None:
        _OIL_DF_CACHE = _build_model_frame()

    return _OIL_DF_CACHE


def refresh_oil_macro_df():

    global _OIL_DF_CACHE
    _OIL_DF_CACHE = _build_model_frame()
    return _OIL_DF_CACHE


def load_oil_macro_df_from_storage():

    if SQLITE_FILE.exists():
        with sqlite3.connect(SQLITE_FILE) as conn:
            df = pd.read_sql_query("SELECT * FROM oil_macro", conn)
        if not df.empty:
            date_col = None
            for candidate in ["date", "index", "Date"]:
                if candidate in df.columns:
                    date_col = candidate
                    break
            if date_col is not None:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
            return df

    if PARQUET_FILE.exists():
        df = pd.read_parquet(PARQUET_FILE)
        if isinstance(df.index, pd.DatetimeIndex):
            return df.sort_index()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            return df.dropna(subset=["date"]).set_index("date").sort_index()
        return df

    raise FileNotFoundError(
        f"No cached oil macro data found in {SQLITE_FILE} or {PARQUET_FILE}."
    )


def cache_oil_macro_df(df):

    global _OIL_DF_CACHE
    _OIL_DF_CACHE = df
    return _OIL_DF_CACHE


def build_layout():

    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H2("Macro Oil Dashboard", className="mb-1"),
                            html.Div(
                                "FRED macro data → PyTorch fair value model → WTI mispricing.",
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
                                dbc.CardHeader("Time Period"),
                                dbc.CardBody(
                                    dcc.RadioItems(
                                        id="oil-period",
                                        options=PERIOD_OPTIONS,
                                        value="2026",
                                        inputStyle={"marginRight": "4px"},
                                        labelStyle={"marginRight": "16px"},
                                        inline=True,
                                    )
                                ),
                            ],
                            className="shadow-sm",
                        ),
                        width=12,
                        lg=5,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Model View"),
                                dbc.CardBody(
                                    dcc.RadioItems(
                                        id="oil-model-mode",
                                        options=MODEL_OPTIONS,
                                        value="shock",
                                        inputStyle={"marginRight": "4px"},
                                        labelStyle={"marginRight": "16px"},
                                        inline=True,
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
                                dbc.CardHeader("Run"),
                                dbc.CardBody(
                                    [
                                        dbc.ButtonGroup(
                                            [
                                                dbc.Button(
                                                    "SQLite Load",
                                                    id="oil-sqlite-load-btn",
                                                    color="secondary",
                                                    n_clicks=0,
                                                ),
                                                dbc.Button(
                                                    "Refresh Load",
                                                    id="oil-refresh-load-btn",
                                                    color="primary",
                                                    n_clicks=0,
                                                ),
                                            ],
                                            className="w-100",
                                        ),
                                        html.Div(
                                            "Use SQLite Load for cached data or Refresh Load for a full retrain.",
                                            id="oil-run-status",
                                            className="text-muted small mt-2",
                                        ),
                                    ]
                                ),
                            ],
                            className="shadow-sm",
                        ),
                        width=12,
                        lg=3,
                    ),
                ],
                className="mb-3 g-2",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("WTI Price vs Macro Fair Value"),
                                dbc.CardBody(
                                    dcc.Graph(id="oil-price-chart"),
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
                                dbc.CardHeader("Oil Mispricing (WTI − Fair Value)"),
                                dbc.CardBody(
                                    dcc.Graph(id="oil-mispricing-chart"),
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
        className="py-3",
    )


layout = build_layout()


def register_callbacks(app):

    @app.callback(
        Output("oil-price-chart", "figure"),
        Output("oil-mispricing-chart", "figure"),
        Output("oil-run-status", "children"),
        Input("oil-sqlite-load-btn", "n_clicks"),
        Input("oil-refresh-load-btn", "n_clicks"),
        Input("oil-period", "value"),
        Input("oil-model-mode", "value"),
    )
    def update_charts(sqlite_clicks, refresh_clicks, period, model_mode):
        if (
            _OIL_DF_CACHE is None
            and (sqlite_clicks or 0) < 1
            and (refresh_clicks or 0) < 1
        ):
            return (
                _empty_figure(
                    "WTI Price vs Macro Fair Value",
                    "Click SQLite Load or Refresh Load to build charts.",
                ),
                _empty_figure(
                    "Oil Mispricing (WTI - Fair Value)",
                    "Click SQLite Load or Refresh Load to build charts.",
                ),
                "Waiting to run.",
            )

        try:
            if ctx.triggered_id == "oil-refresh-load-btn" and (refresh_clicks or 0) > 0:
                df = refresh_oil_macro_df()
                status = f"Refresh complete. Rows: {len(df)}"
            elif ctx.triggered_id == "oil-sqlite-load-btn" and (sqlite_clicks or 0) > 0:
                df = cache_oil_macro_df(load_oil_macro_df_from_storage())
                status = f"SQLite load complete. Rows: {len(df)}"
            else:
                df = get_oil_macro_df()
                status = f"Ready. Last run rows: {len(df)}"
        except Exception as exc:
            return (
                _empty_figure("WTI Price vs Macro Fair Value", f"Run failed: {exc}"),
                _empty_figure(
                    "Oil Mispricing (WTI - Fair Value)", f"Run failed: {exc}"
                ),
                f"Run failed: {exc}",
            )

        dff = _apply_period(df, period)

        fig = go.Figure()
        fig.add_scatter(x=dff.index, y=dff["wti"], name="WTI Market Price")

        if model_mode in ["shock", "both"]:
            fig.add_scatter(
                x=dff.index,
                y=dff["oil_fair_value_shock"],
                name="Macro Fair Value (Shock-Aware)",
                line=dict(dash="dash"),
            )
        if model_mode in ["base", "both"]:
            fig.add_scatter(
                x=dff.index,
                y=dff["oil_fair_value_base"],
                name="Macro Fair Value (Baseline)",
                line=dict(dash="dot"),
            )
        fig.update_layout(margin=dict(t=20, b=20))
        if not dff.empty and dff.index.min() <= HORMUZ_START_DATE <= dff.index.max():
            fig.add_vline(x=HORMUZ_START_DATE, line_dash="dot", line_color="crimson")
            fig.add_annotation(
                x=HORMUZ_START_DATE,
                y=1,
                yref="paper",
                text="Hormuz shutdown",
                showarrow=False,
                xanchor="left",
                font=dict(color="crimson"),
            )

        fig2 = go.Figure()
        if model_mode in ["shock", "both"]:
            fig2.add_scatter(
                x=dff.index,
                y=dff["mispricing_shock"],
                name="Oil Mispricing (Shock-Aware)",
            )
        if model_mode in ["base", "both"]:
            fig2.add_scatter(
                x=dff.index,
                y=dff["mispricing_base"],
                name="Oil Mispricing (Baseline)",
            )
        fig2.update_layout(margin=dict(t=20, b=20))
        if not dff.empty and dff.index.min() <= HORMUZ_START_DATE <= dff.index.max():
            fig2.add_vline(x=HORMUZ_START_DATE, line_dash="dot", line_color="crimson")

        return fig, fig2, status


register_callbacks(get_app())

# =============================
# MAIN PIPELINE
# =============================
