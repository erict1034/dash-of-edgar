import dash
import dash_bootstrap_components as dbc
from dash import html


dash.register_page(__name__, path="/", name="Home", order=-1)

MACRO_CARDS = [
    ("US GDP", "/gdp", "U.S. Gross Domestic Product trends and growth rates."),
    (
        "US Oil Prices",
        "/oil-macro",
        "FRED-based oil macro fair value and WTI mispricing dashboard.",
    ),
]

MICRO_CARDS = [
    ("Stock Price", "/ticker-price", "Historical price charts and moving averages."),
    (
        "Public Company Revenue",
        "/revenue",
        "Annual and quarterly revenue analysis by ticker.",
    ),
    (
        "Public Company Liabilities",
        "/liabilities",
        "Quarterly and annual liability trends from SEC filings.",
    ),
    (
        "Public Company Earnings",
        "/earnings-quality",
        "Accruals, OCF margin, and earnings quality metrics.",
    ),
    (
        "10-K Sentiment",
        "/sentiment",
        "Sentiment signals from 10-K filings.",
    ),
    (
        "Intrinsic Value",
        "/intrinsic-value",
        "Intrinsic value model with SEC fundamentals and market data.",
    ),
    (
        "Daily Insider Purchase List",
        "/speedy-index-pull",
        "Fast SEC Form 4 purchase pull with SQLite-backed totals.",
    ),
    (
        "Public Company Insider Purchases",
        "/pull",
        "Deep-dive ticker dashboard powered by EDGAR filings.",
    ),
]


def _card(title, href, text):
    return dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                    html.H5(title, className="card-title"),
                    html.P(text, className="card-text text-muted small"),
                    dbc.ButtonGroup(
                        [
                            dbc.Button("Open", href=href, color="primary", size="sm"),
                            html.A(
                                dbc.Button("New Tab", color="secondary", size="sm"),
                                href=href,
                                target="_blank",
                            ),
                        ]
                    ),
                ]
            ),
            className="h-100 shadow-sm",
        ),
        xs=12,
        sm=6,
        lg=4,
        className="mb-4",
    )


layout = html.Div(
    [
        dbc.Container(
            [
                html.Div(
                    [
                        html.H1("Dash of Edgar", className="display-5 fw-bold"),
                        html.P(
                            "Economic and financial dashboards built on SEC EDGAR and FRED data.",
                            className="lead text-muted",
                        ),
                    ],
                    className="py-4",
                ),
                html.H5("Macro", className="mb-3 text-uppercase text-secondary"),
                dbc.Row([_card(*c) for c in MACRO_CARDS], className="mb-4"),
                html.H5("Micro", className="mb-3 text-uppercase text-secondary"),
                dbc.Row([_card(*c) for c in MICRO_CARDS]),
            ],
            fluid=True,
        ),
    ]
)
