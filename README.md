# Dash of EDGAR

Multi-page Dash application for SEC EDGAR and Fred analytics.
## Features

- EDGAR Earnings dashboard
- EDGAR Revenue dashboard
- EDGAR Liabilities dashboard
- Form 4 insider dashboards 
- Ticker Price dashboard (Yahoo Finance and Alpha Vantage)
- US GDP dashboard (FRED)
- Oil and macro dashboard
- SQLite and Parquet storage

## Requirements

- Python 3.11+ recommended
- Windows PowerShell commands below (works similarly on macOS/Linux)

## Setup

pip install -r requirements.txt


## Environment Variables

Create a `.env` file in the project root and set at least:

```dotenv
SEC_CONTACT_EMAIL=your_email@domain.com
ALPHAVANTAGE_API_KEY=your_alpha_vantage_key
FRED_API_KEY=your_fred_key
```

Notes:

- `SEC_CONTACT_EMAIL` is used in SEC/EDGAR user-agent identity.
- `ALPHAVANTAGE_API_KEY` is used by Form 4 and ticker price lookups.
- `FRED_API_KEY` is required for GDP and macro data pulls.

## Run the App

python app.py

Default URL:

- http://127.0.0.1:8053/

## Storage Model

Local data is under the `data` folders:

- Shared SQLite database: `data/dash_storage.sqlite`
- Shared Parquet cache directory: `data/parquet/`

These paths are defined in `storage_paths.py`.

## GitHub / Security Notes

- `.env` is ignored by git.


## Troubleshooting

- If a dashboard shows empty cache data, run a live load once from that page.
- If Alpha Vantage times out, retry; the app includes retries and cache fallback.
- If dependencies fail, reinstall from `requirements.txt` inside the 
