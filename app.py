import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
from dotenv import load_dotenv
# keep if you still want navbar

load_dotenv()

app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
)

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        # Optional navbar (remove if you don’t want it)
        # This automatically renders pages from /pages
        dash.page_container,
    ]
)

if __name__ == "__main__":
    app.run(debug=True, port=8053)
