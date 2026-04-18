import sqlite3
from datetime import datetime, timezone
from pathlib import Path
import re

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import requests
import dash
from dash import get_app, Input, Output, State, ctx, dash_table, dcc, html
from storage_paths import CENTRAL_SQLITE_PATH, parquet_path


dash.register_page(__name__, path="/liabilities", name="Liabilities", order=5)

HEADERS = {"User-Agent": "EarlyWarningDashboard your_email@example.com"}

TOTAL_LIABILITY_TAGS = [
    "Liabilities",
    "LiabilitiesNetMinorityInterest",
]
REVENUE_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
]
EARNINGS_TAGS = [
    "NetIncomeLoss",
    "ProfitLoss",
]
OPERATING_CASH_FLOW_TAGS = [
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
]
CURRENT_LIABILITY_TAGS = [
    "LiabilitiesCurrent",
]
NONCURRENT_LIABILITY_TAGS = [
    "LiabilitiesNoncurrent",
]
LIABILITY_COMPONENT_KEYWORDS = (
    "liabilit",
    "payable",
    "debt",
    "accrued",
    "lease",
    "deferredrevenue",
    "unearnedrevenue",
    "contractwithcustomerliability",
)
LIABILITY_COMPONENT_EXCLUDE_KEYWORDS = (
    "asset",
    "equity",
    "stockholder",
    "shareholder",
    "receivable",
    "commitment",
    "contingenc",
)
LIABILITY_COMPONENT_BLACKLIST = {
    "Liabilities",
    "LiabilitiesCurrent",
    "LiabilitiesNoncurrent",
    "LiabilitiesNetMinorityInterest",
    "LiabilitiesAndStockholdersEquity",
    "LiabilitiesAndPartnersCapital",
    "LiabilitiesAndCapital",
}
LIABILITY_COMPONENT_LABELS = {
    "AccountsPayableCurrent": "Accounts Payable",
    "AccruedLiabilitiesCurrent": "Accrued Liabilities",
    "OtherAccruedLiabilitiesCurrent": "Other Accrued Liabilities",
    "EmployeeRelatedLiabilitiesCurrent": "Employee-related Liabilities",
    "TaxesPayableCurrent": "Taxes Payable",
    "LongTermDebtCurrent": "Current Debt",
    "LongTermDebtNoncurrent": "Long-term Debt",
    "LongTermDebtAndCapitalLeaseObligationsCurrent": "Current Debt and Capital Lease Obligations",
    "LongTermDebtAndCapitalLeaseObligationsNoncurrent": "Long-term Debt and Capital Lease Obligations",
    "OperatingLeaseLiabilityCurrent": "Current Operating Lease Liability",
    "OperatingLeaseLiabilityNoncurrent": "Non-current Operating Lease Liability",
    "FinanceLeaseLiabilityCurrent": "Current Finance Lease Liability",
    "FinanceLeaseLiabilityNoncurrent": "Non-current Finance Lease Liability",
    "DeferredRevenueCurrent": "Current Deferred Revenue",
    "DeferredRevenueNoncurrent": "Non-current Deferred Revenue",
    "ContractWithCustomerLiabilityCurrent": "Current Contract Liability",
    "ContractWithCustomerLiabilityNoncurrent": "Non-current Contract Liability",
}
LIABILITY_SEGMENT_DEFINITIONS = [
    {
        "label": "Accounts payable",
        "tags": ["AccountsPayableCurrent", "AccountsPayable"],
        "mode": "alias",
    },
    {
        "label": "Current portion of long-term debt",
        "tags": [
            "LongTermDebtCurrent",
            "LongTermDebtAndCapitalLeaseObligationsCurrent",
            "LongTermDebtAndFinanceLeaseObligationsCurrent",
        ],
        "mode": "alias",
    },
    {
        "label": "Accrued compensation",
        "tags": [
            "AccruedCompensationCurrent",
            "EmployeeRelatedLiabilitiesCurrent",
            "AccruedSalariesCurrent",
            "AccruedPayrollCurrent",
            "AccruedEmployeeBenefitsCurrent",
        ],
        "mode": "alias",
    },
    {
        "label": "Short-term income taxes",
        "tags": [
            "IncomeTaxesPayableCurrent",
            "TaxesPayableCurrent",
            "AccruedIncomeTaxesCurrent",
        ],
        "mode": "alias",
    },
    {
        "label": "Short-term unearned revenue",
        "tags": [
            "DeferredRevenueCurrent",
            "UnearnedRevenueCurrent",
            "ContractWithCustomerLiabilityCurrent",
        ],
        "mode": "alias",
    },
    {
        "label": "Other current liabilities",
        "tags": [
            "OtherLiabilitiesCurrent",
            "OtherCurrentLiabilities",
            "OtherAccruedLiabilitiesCurrent",
        ],
        "mode": "alias",
    },
    {
        "label": "Total current liabilities",
        "series": "current_total",
        "tag": "LiabilitiesCurrent",
        "mode": "series",
    },
    {
        "label": "Long-term debt",
        "tags": [
            "LongTermDebtNoncurrent",
            "LongTermDebtAndCapitalLeaseObligationsNoncurrent",
            "LongTermDebtAndFinanceLeaseObligationsNoncurrent",
        ],
        "mode": "alias",
    },
    {
        "label": "Long-term income taxes",
        "tags": [
            "IncomeTaxesPayableNoncurrent",
            "LongTermIncomeTaxesPayable",
        ],
        "mode": "alias",
    },
    {
        "label": "Long-term unearned revenue",
        "tags": [
            "DeferredRevenueNoncurrent",
            "UnearnedRevenueNoncurrent",
            "ContractWithCustomerLiabilityNoncurrent",
        ],
        "mode": "alias",
    },
    {
        "label": "Deferred income taxes",
        "tags": [
            "DeferredTaxLiabilitiesNoncurrent",
            "DeferredIncomeTaxLiabilitiesNet",
            "DeferredTaxLiabilitiesNetNoncurrent",
        ],
        "mode": "alias",
    },
    {
        "label": "Operating lease liabilities",
        "tags": [
            "OperatingLeaseLiabilityCurrent",
            "OperatingLeaseLiabilityNoncurrent",
        ],
        "mode": "combine",
    },
    {
        "label": "Other long-term liabilities",
        "tags": [
            "OtherLiabilitiesNoncurrent",
            "OtherNoncurrentLiabilities",
            "AccruedLiabilitiesNoncurrent",
        ],
        "mode": "alias",
    },
    {
        "label": "Total liabilities",
        "series": "total_liabilities",
        "tag": "Liabilities",
        "mode": "series",
    },
]
LIABILITY_SEGMENT_ORDER = {
    definition["label"]: index
    for index, definition in enumerate(LIABILITY_SEGMENT_DEFINITIONS)
}
INTEREST_PAYABLE_TAGS = [
    "InterestPayableCurrent",
    "InterestPayable",
    "AccruedInterestCurrent",
    "AccruedInterest",
    "InterestExpensePayableCurrent",
]
INTEREST_EXPENSE_TAGS = [
    "InterestExpenseAndDebtExpense",
    "InterestExpense",
    "InterestAndDebtExpense",
    "InterestExpenseOther",
    "InterestExpenseAndOther",
]
DEBT_TOTAL_TAGS = [
    "LongTermDebtAndFinanceLeaseObligations",
    "LongTermDebtAndCapitalLeaseObligations",
    "LongTermDebt",
    "DebtInstrumentFaceAmount",
]
DEBT_CURRENT_TAGS = [
    "LongTermDebtCurrent",
    "LongTermDebtAndCapitalLeaseObligationsCurrent",
    "LongTermDebtAndFinanceLeaseObligationsCurrent",
]
DEBT_NONCURRENT_TAGS = [
    "LongTermDebtNoncurrent",
    "LongTermDebtAndCapitalLeaseObligationsNoncurrent",
    "LongTermDebtAndFinanceLeaseObligationsNoncurrent",
]
TOTAL_ASSETS_TAGS = [
    "Assets",
]
TOTAL_EQUITY_TAGS = [
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    "StockholdersEquity",
]
LIABILITIES_AND_EQUITY_TAGS = [
    "LiabilitiesAndStockholdersEquity",
]
DEBT_MATURITY_BUCKETS = [
    (
        "Next 12 Months",
        [
            "LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths",
            "LongTermDebtAndCapitalLeaseObligationsMaturitiesRepaymentsOfPrincipalInNextTwelveMonths",
            "LesseeOperatingLeaseLiabilityPaymentsDueNextTwelveMonths",
            "FinanceLeaseLiabilityPaymentsDueNextTwelveMonths",
            "LesseeOperatingLeaseLiabilityPaymentsRemainderOfFiscalYear",
        ],
    ),
    (
        "Year 2",
        [
            "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearTwo",
            "LongTermDebtAndCapitalLeaseObligationsMaturitiesRepaymentsOfPrincipalInYearTwo",
            "LesseeOperatingLeaseLiabilityPaymentsDueYearTwo",
            "FinanceLeaseLiabilityPaymentsDueYearTwo",
        ],
    ),
    (
        "Year 3",
        [
            "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearThree",
            "LongTermDebtAndCapitalLeaseObligationsMaturitiesRepaymentsOfPrincipalInYearThree",
            "LesseeOperatingLeaseLiabilityPaymentsDueYearThree",
            "FinanceLeaseLiabilityPaymentsDueYearThree",
        ],
    ),
    (
        "Year 4",
        [
            "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearFour",
            "LongTermDebtAndCapitalLeaseObligationsMaturitiesRepaymentsOfPrincipalInYearFour",
            "LesseeOperatingLeaseLiabilityPaymentsDueYearFour",
            "FinanceLeaseLiabilityPaymentsDueYearFour",
        ],
    ),
    (
        "Year 5",
        [
            "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearFive",
            "LongTermDebtAndCapitalLeaseObligationsMaturitiesRepaymentsOfPrincipalInYearFive",
            "LesseeOperatingLeaseLiabilityPaymentsDueYearFive",
            "FinanceLeaseLiabilityPaymentsDueYearFive",
        ],
    ),
    (
        "Thereafter",
        [
            "LongTermDebtMaturitiesRepaymentsOfPrincipalAfterYearFive",
            "LongTermDebtAndCapitalLeaseObligationsMaturitiesRepaymentsOfPrincipalAfterYearFive",
            "LesseeOperatingLeaseLiabilityPaymentsDueAfterYearFive",
            "FinanceLeaseLiabilityPaymentsDueAfterYearFive",
        ],
    ),
]

SERIES_STYLES = {
    "liabilities": {"color": "#2563eb", "width": 2.5, "dash": "solid"},
    "revenue": {"color": "#10b981", "width": 2.5, "dash": "solid"},
    "earnings": {"color": "#f97316", "width": 2.5, "dash": "dot"},
    "operating_cash_flow": {"color": "#8b5cf6", "width": 2.5, "dash": "dash"},
}
_CIK_CACHE = {}
_FACTS_CACHE = {}

LIAB_PARQUET_PATH = parquet_path("edgar_liabilities")
LIAB_SQLITE_PATH = CENTRAL_SQLITE_PATH
LIAB_COMPONENT_PARQUET_PATH = LIAB_PARQUET_PATH
LIAB_DEBT_SNAPSHOT_PARQUET_PATH = LIAB_PARQUET_PATH
LIAB_DEBT_SCHEDULE_PARQUET_PATH = LIAB_PARQUET_PATH
LIAB_DEBT_DETAIL_PARQUET_PATH = LIAB_PARQUET_PATH

_callbacks_registered = False


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_unified_parquet_cache(
    parquet_path: Path = LIAB_PARQUET_PATH,
) -> pd.DataFrame:
    if not parquet_path.exists():
        return pd.DataFrame()

    cache_df = pd.read_parquet(parquet_path)
    if cache_df.empty:
        return cache_df

    if "dataset_type" not in cache_df.columns:
        legacy_df = cache_df.copy()
        if "series_name" in legacy_df.columns:
            legacy_df["dataset_type"] = "liabilities_series"
            return legacy_df
    return cache_df


def _save_unified_parquet_dataset(
    data: pd.DataFrame,
    dataset_type: str,
    key_columns: list[str],
    parquet_path: Path = LIAB_PARQUET_PATH,
) -> int:
    if data is None or data.empty:
        return 0

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    write_df = data.copy()
    write_df["dataset_type"] = dataset_type

    existing = _load_unified_parquet_cache(parquet_path)
    if not existing.empty:
        write_df = pd.concat([existing, write_df], ignore_index=True, sort=False)

    dedupe_columns = ["dataset_type", *key_columns]
    for column in dedupe_columns:
        if column not in write_df.columns:
            write_df[column] = pd.NA

    write_df = write_df.drop_duplicates(subset=dedupe_columns, keep="last")
    write_df.to_parquet(parquet_path, index=False)
    return len(data)


def _load_unified_parquet_dataset(
    dataset_type: str,
    parquet_path: Path = LIAB_PARQUET_PATH,
) -> pd.DataFrame:
    cache_df = _load_unified_parquet_cache(parquet_path)
    if cache_df.empty or "dataset_type" not in cache_df.columns:
        return pd.DataFrame()

    dataset_df = cache_df[cache_df["dataset_type"] == dataset_type].copy()
    if dataset_df.empty:
        return pd.DataFrame()

    return dataset_df.drop(columns=["dataset_type"], errors="ignore")


def _normalize_series_for_persist(
    df: pd.DataFrame, ticker: str, series_name: str
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "series_name",
                "period_end",
                "value",
                "form",
                "filed",
                "pulled_at_utc",
            ]
        )

    working = df.copy()
    working["period_end"] = pd.to_datetime(working["end"], errors="coerce")
    if "filed" in working.columns:
        working["filed"] = pd.to_datetime(working["filed"], errors="coerce")
    else:
        working["filed"] = pd.NaT
    working["value"] = pd.to_numeric(working["value"], errors="coerce")
    working = working.dropna(subset=["period_end", "value"])

    if working.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "series_name",
                "period_end",
                "value",
                "form",
                "filed",
                "pulled_at_utc",
            ]
        )

    working["ticker"] = (ticker or "").strip().upper()
    working["series_name"] = series_name
    working["period_end"] = working["period_end"].dt.strftime("%Y-%m-%d")
    working["filed"] = working["filed"].dt.strftime("%Y-%m-%d")
    working["pulled_at_utc"] = _utc_now_iso()

    if "form" not in working.columns:
        working["form"] = ""

    return working[
        [
            "ticker",
            "series_name",
            "period_end",
            "value",
            "form",
            "filed",
            "pulled_at_utc",
        ]
    ]


def _save_liabilities_parquet(
    data: pd.DataFrame, parquet_path: Path = LIAB_PARQUET_PATH
) -> int:
    return _save_unified_parquet_dataset(
        data=data,
        dataset_type="liabilities_series",
        key_columns=["ticker", "series_name", "period_end"],
        parquet_path=parquet_path,
    )


def _upsert_liabilities_sqlite(
    data: pd.DataFrame, db_path: Path = LIAB_SQLITE_PATH
) -> int:
    if data is None or data.empty:
        return 0

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS liabilities_series (
                ticker TEXT NOT NULL,
                series_name TEXT NOT NULL,
                period_end TEXT NOT NULL,
                value REAL NOT NULL,
                form TEXT,
                filed TEXT,
                pulled_at_utc TEXT NOT NULL,
                PRIMARY KEY (ticker, series_name, period_end)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_liab_ticker ON liabilities_series(ticker)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_liab_period ON liabilities_series(period_end)"
        )

        rows = [
            (
                row.ticker,
                row.series_name,
                row.period_end,
                float(row.value),
                row.form,
                row.filed,
                row.pulled_at_utc,
            )
            for row in data.itertuples(index=False)
        ]
        conn.executemany(
            """
            INSERT OR REPLACE INTO liabilities_series
            (ticker, series_name, period_end, value, form, filed, pulled_at_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    return len(data)


def _normalize_components_for_persist(
    component_df: pd.DataFrame,
    ticker: str,
    component_source: str,
) -> pd.DataFrame:
    if component_df is None or component_df.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "tag",
                "label",
                "period_end",
                "value",
                "form",
                "filed",
                "component_source",
                "pulled_at_utc",
            ]
        )

    working = component_df.copy()
    working["period_end"] = pd.to_datetime(working["end"], errors="coerce")
    working["filed"] = pd.to_datetime(working.get("filed"), errors="coerce")
    working["value"] = pd.to_numeric(working.get("value"), errors="coerce")
    working = working.dropna(subset=["period_end", "value"])
    if working.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "tag",
                "label",
                "period_end",
                "value",
                "form",
                "filed",
                "component_source",
                "pulled_at_utc",
            ]
        )

    working["ticker"] = (ticker or "").strip().upper()
    working["tag"] = working.get("tag", "").astype(str)
    working["label"] = working.get("label", working["tag"]).astype(str)
    working["period_end"] = working["period_end"].dt.strftime("%Y-%m-%d")
    working["filed"] = working["filed"].dt.strftime("%Y-%m-%d")
    working["form"] = working.get("form", "").fillna("").astype(str)
    working["component_source"] = str(component_source or "")
    working["pulled_at_utc"] = _utc_now_iso()

    return working[
        [
            "ticker",
            "tag",
            "label",
            "period_end",
            "value",
            "form",
            "filed",
            "component_source",
            "pulled_at_utc",
        ]
    ]


def _save_component_parquet(
    data: pd.DataFrame,
    parquet_path: Path = LIAB_COMPONENT_PARQUET_PATH,
) -> int:
    return _save_unified_parquet_dataset(
        data=data,
        dataset_type="liability_components",
        key_columns=["ticker", "tag", "period_end"],
        parquet_path=parquet_path,
    )


def _normalize_debt_snapshot_for_persist(
    ticker: str,
    debt_balance_display: str,
    accrued_interest_display: str,
    implied_rate_display: str,
    return_on_debt_display: str,
    implied_rate_raw: float | None,
) -> pd.DataFrame:
    ticker_norm = (ticker or "").strip().upper()
    if not ticker_norm:
        return pd.DataFrame(
            columns=[
                "ticker",
                "debt_balance_display",
                "accrued_interest_display",
                "implied_rate_display",
                "return_on_debt_display",
                "implied_rate_raw",
                "pulled_at_utc",
            ]
        )

    return pd.DataFrame(
        [
            {
                "ticker": ticker_norm,
                "debt_balance_display": debt_balance_display or "-",
                "accrued_interest_display": accrued_interest_display or "-",
                "implied_rate_display": implied_rate_display or "-",
                "return_on_debt_display": return_on_debt_display or "-",
                "implied_rate_raw": float(implied_rate_raw)
                if implied_rate_raw is not None
                else None,
                "pulled_at_utc": _utc_now_iso(),
            }
        ]
    )


def _save_debt_snapshot_parquet(
    data: pd.DataFrame,
    parquet_path: Path = LIAB_DEBT_SNAPSHOT_PARQUET_PATH,
) -> int:
    return _save_unified_parquet_dataset(
        data=data,
        dataset_type="debt_snapshot",
        key_columns=["ticker"],
        parquet_path=parquet_path,
    )


def _upsert_debt_snapshot_sqlite(
    data: pd.DataFrame,
    db_path: Path = LIAB_SQLITE_PATH,
) -> int:
    if data is None or data.empty:
        return 0

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS liabilities_debt_snapshot (
                ticker TEXT NOT NULL PRIMARY KEY,
                debt_balance_display TEXT,
                accrued_interest_display TEXT,
                implied_rate_display TEXT,
                return_on_debt_display TEXT,
                implied_rate_raw REAL,
                pulled_at_utc TEXT NOT NULL
            )
            """
        )

        row = data.iloc[-1]
        conn.execute(
            """
            INSERT OR REPLACE INTO liabilities_debt_snapshot
            (ticker, debt_balance_display, accrued_interest_display, implied_rate_display,
             return_on_debt_display, implied_rate_raw, pulled_at_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.get("ticker"),
                row.get("debt_balance_display"),
                row.get("accrued_interest_display"),
                row.get("implied_rate_display"),
                row.get("return_on_debt_display"),
                row.get("implied_rate_raw"),
                row.get("pulled_at_utc"),
            ),
        )
        conn.commit()
    return len(data)


def _normalize_debt_schedule_for_persist(
    schedule_df: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    if schedule_df is None or schedule_df.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "bucket",
                "tag",
                "label",
                "period_end",
                "value",
                "form",
                "filed",
                "unit",
                "pulled_at_utc",
            ]
        )

    working = schedule_df.copy()
    working["period_end"] = pd.to_datetime(working.get("end"), errors="coerce")
    working["filed"] = pd.to_datetime(working.get("filed"), errors="coerce")
    working["value"] = pd.to_numeric(working.get("value"), errors="coerce")
    working = working.dropna(subset=["bucket", "period_end", "value"])
    if working.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "bucket",
                "tag",
                "label",
                "period_end",
                "value",
                "form",
                "filed",
                "unit",
                "pulled_at_utc",
            ]
        )

    working["ticker"] = (ticker or "").strip().upper()
    working["period_end"] = working["period_end"].dt.strftime("%Y-%m-%d")
    working["filed"] = working["filed"].dt.strftime("%Y-%m-%d")
    working["tag"] = working.get("tag", "").fillna("").astype(str)
    working["label"] = working.get("label", "").fillna("").astype(str)
    working["form"] = working.get("form", "").fillna("").astype(str)
    working["unit"] = working.get("unit", "").fillna("").astype(str)
    working["pulled_at_utc"] = _utc_now_iso()

    return working[
        [
            "ticker",
            "bucket",
            "tag",
            "label",
            "period_end",
            "value",
            "form",
            "filed",
            "unit",
            "pulled_at_utc",
        ]
    ]


def _save_debt_schedule_parquet(
    data: pd.DataFrame,
    parquet_path: Path = LIAB_DEBT_SCHEDULE_PARQUET_PATH,
) -> int:
    return _save_unified_parquet_dataset(
        data=data,
        dataset_type="debt_schedule",
        key_columns=["ticker", "bucket", "period_end"],
        parquet_path=parquet_path,
    )


def _upsert_debt_schedule_sqlite(
    data: pd.DataFrame,
    db_path: Path = LIAB_SQLITE_PATH,
) -> int:
    if data is None or data.empty:
        return 0

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS liabilities_debt_schedule (
                ticker TEXT NOT NULL,
                bucket TEXT NOT NULL,
                tag TEXT,
                label TEXT,
                period_end TEXT NOT NULL,
                value REAL NOT NULL,
                form TEXT,
                filed TEXT,
                unit TEXT,
                pulled_at_utc TEXT NOT NULL,
                PRIMARY KEY (ticker, bucket, period_end)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_liab_debt_schedule_ticker ON liabilities_debt_schedule(ticker)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_liab_debt_schedule_period ON liabilities_debt_schedule(period_end)"
        )

        rows = [
            (
                row.ticker,
                row.bucket,
                row.tag,
                row.label,
                row.period_end,
                float(row.value),
                row.form,
                row.filed,
                row.unit,
                row.pulled_at_utc,
            )
            for row in data.itertuples(index=False)
        ]
        conn.executemany(
            """
            INSERT OR REPLACE INTO liabilities_debt_schedule
            (ticker, bucket, tag, label, period_end, value, form, filed, unit, pulled_at_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    return len(data)


def _normalize_debt_detail_for_persist(
    ticker: str,
    detail_rows: list[dict],
) -> pd.DataFrame:
    if not detail_rows:
        return pd.DataFrame(
            columns=[
                "ticker",
                "category",
                "metric",
                "value_text",
                "period_end",
                "form",
                "source_tag",
                "pulled_at_utc",
            ]
        )

    working = pd.DataFrame(detail_rows).copy()
    if working.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "category",
                "metric",
                "value_text",
                "period_end",
                "form",
                "source_tag",
                "pulled_at_utc",
            ]
        )

    working["ticker"] = (ticker or "").strip().upper()
    working["category"] = working.get("Category", "").fillna("").astype(str)
    working["metric"] = working.get("Metric", "").fillna("").astype(str)
    working["value_text"] = working.get("Value", "").fillna("").astype(str)
    working["period_end"] = working.get("Period End", "").fillna("").astype(str)
    working["form"] = working.get("Form", "").fillna("").astype(str)
    working["source_tag"] = working.get("Source Tag", "").fillna("").astype(str)
    working["pulled_at_utc"] = _utc_now_iso()

    return working[
        [
            "ticker",
            "category",
            "metric",
            "value_text",
            "period_end",
            "form",
            "source_tag",
            "pulled_at_utc",
        ]
    ]


def _save_debt_detail_parquet(
    data: pd.DataFrame,
    parquet_path: Path = LIAB_DEBT_DETAIL_PARQUET_PATH,
) -> int:
    return _save_unified_parquet_dataset(
        data=data,
        dataset_type="debt_detail",
        key_columns=["ticker", "category", "metric", "period_end"],
        parquet_path=parquet_path,
    )


def _upsert_debt_detail_sqlite(
    data: pd.DataFrame,
    db_path: Path = LIAB_SQLITE_PATH,
) -> int:
    if data is None or data.empty:
        return 0

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS liabilities_debt_detail (
                ticker TEXT NOT NULL,
                category TEXT NOT NULL,
                metric TEXT NOT NULL,
                value_text TEXT,
                period_end TEXT,
                form TEXT,
                source_tag TEXT,
                pulled_at_utc TEXT NOT NULL,
                PRIMARY KEY (ticker, category, metric, period_end)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_liab_debt_detail_ticker ON liabilities_debt_detail(ticker)"
        )

        rows = [
            (
                row.ticker,
                row.category,
                row.metric,
                row.value_text,
                row.period_end,
                row.form,
                row.source_tag,
                row.pulled_at_utc,
            )
            for row in data.itertuples(index=False)
        ]
        conn.executemany(
            """
            INSERT OR REPLACE INTO liabilities_debt_detail
            (ticker, category, metric, value_text, period_end, form, source_tag, pulled_at_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    return len(data)


def _persist_debt_analytics(
    ticker: str,
    debt_balance_display: str,
    accrued_interest_display: str,
    implied_rate_display: str,
    return_on_debt_display: str,
    implied_rate_raw: float | None,
    debt_schedule_df: pd.DataFrame,
    debt_detail_rows: list[dict],
) -> dict:
    snapshot_df = _normalize_debt_snapshot_for_persist(
        ticker=ticker,
        debt_balance_display=debt_balance_display,
        accrued_interest_display=accrued_interest_display,
        implied_rate_display=implied_rate_display,
        return_on_debt_display=return_on_debt_display,
        implied_rate_raw=implied_rate_raw,
    )
    schedule_df = _normalize_debt_schedule_for_persist(debt_schedule_df, ticker=ticker)
    detail_df = _normalize_debt_detail_for_persist(
        ticker=ticker, detail_rows=debt_detail_rows
    )

    snapshot_parquet_rows = _save_debt_snapshot_parquet(snapshot_df)
    snapshot_sqlite_rows = _upsert_debt_snapshot_sqlite(snapshot_df)
    schedule_parquet_rows = _save_debt_schedule_parquet(schedule_df)
    schedule_sqlite_rows = _upsert_debt_schedule_sqlite(schedule_df)
    detail_parquet_rows = _save_debt_detail_parquet(detail_df)
    detail_sqlite_rows = _upsert_debt_detail_sqlite(detail_df)

    return {
        "debt_snapshot_rows": int(len(snapshot_df)),
        "debt_schedule_rows": int(len(schedule_df)),
        "debt_detail_rows": int(len(detail_df)),
        "debt_snapshot_parquet_rows": int(snapshot_parquet_rows),
        "debt_snapshot_sqlite_rows": int(snapshot_sqlite_rows),
        "debt_schedule_parquet_rows": int(schedule_parquet_rows),
        "debt_schedule_sqlite_rows": int(schedule_sqlite_rows),
        "debt_detail_parquet_rows": int(detail_parquet_rows),
        "debt_detail_sqlite_rows": int(detail_sqlite_rows),
        "debt_snapshot_parquet_path": str(LIAB_PARQUET_PATH),
        "debt_schedule_parquet_path": str(LIAB_PARQUET_PATH),
        "debt_detail_parquet_path": str(LIAB_PARQUET_PATH),
    }


def _upsert_components_sqlite(
    data: pd.DataFrame,
    db_path: Path = LIAB_SQLITE_PATH,
) -> int:
    if data is None or data.empty:
        return 0

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS liabilities_components (
                ticker TEXT NOT NULL,
                tag TEXT NOT NULL,
                label TEXT NOT NULL,
                period_end TEXT NOT NULL,
                value REAL NOT NULL,
                form TEXT,
                filed TEXT,
                component_source TEXT,
                pulled_at_utc TEXT NOT NULL,
                PRIMARY KEY (ticker, tag, period_end)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_liab_components_ticker ON liabilities_components(ticker)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_liab_components_period ON liabilities_components(period_end)"
        )

        rows = [
            (
                row.ticker,
                row.tag,
                row.label,
                row.period_end,
                float(row.value),
                row.form,
                row.filed,
                row.component_source,
                row.pulled_at_utc,
            )
            for row in data.itertuples(index=False)
        ]
        conn.executemany(
            """
            INSERT OR REPLACE INTO liabilities_components
            (ticker, tag, label, period_end, value, form, filed, component_source, pulled_at_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    return len(data)


def _persist_liability_components(
    ticker: str,
    component_df: pd.DataFrame,
    component_source: str,
) -> dict:
    persist_df = _normalize_components_for_persist(
        component_df=component_df,
        ticker=ticker,
        component_source=component_source,
    )
    parquet_rows = _save_component_parquet(persist_df, LIAB_COMPONENT_PARQUET_PATH)
    sqlite_rows = _upsert_components_sqlite(persist_df, LIAB_SQLITE_PATH)

    return {
        "component_rows_attempted": int(len(persist_df)),
        "component_parquet_rows": int(parquet_rows),
        "component_sqlite_rows": int(sqlite_rows),
        "component_parquet_path": str(LIAB_PARQUET_PATH),
        "sqlite_path": str(LIAB_SQLITE_PATH),
    }


def _persist_liabilities(
    ticker: str,
    q_total: pd.DataFrame,
    a_total: pd.DataFrame,
    q_curr: pd.DataFrame,
    q_noncurr: pd.DataFrame,
    q_revenue: pd.DataFrame,
    a_revenue: pd.DataFrame,
    q_earnings: pd.DataFrame,
    a_earnings: pd.DataFrame,
    q_operating_cash_flow: pd.DataFrame,
    a_operating_cash_flow: pd.DataFrame,
) -> dict:
    persist_df = pd.concat(
        [
            _normalize_series_for_persist(q_total, ticker, "total_quarterly"),
            _normalize_series_for_persist(a_total, ticker, "total_annual"),
            _normalize_series_for_persist(q_curr, ticker, "current_quarterly"),
            _normalize_series_for_persist(q_noncurr, ticker, "noncurrent_quarterly"),
            _normalize_series_for_persist(q_revenue, ticker, "revenue_quarterly"),
            _normalize_series_for_persist(a_revenue, ticker, "revenue_annual"),
            _normalize_series_for_persist(q_earnings, ticker, "earnings_quarterly"),
            _normalize_series_for_persist(a_earnings, ticker, "earnings_annual"),
            _normalize_series_for_persist(
                q_operating_cash_flow,
                ticker,
                "operating_cash_flow_quarterly",
            ),
            _normalize_series_for_persist(
                a_operating_cash_flow,
                ticker,
                "operating_cash_flow_annual",
            ),
        ],
        ignore_index=True,
    )

    parquet_rows = _save_liabilities_parquet(persist_df, LIAB_PARQUET_PATH)
    sqlite_rows = _upsert_liabilities_sqlite(persist_df, LIAB_SQLITE_PATH)

    return {
        "rows_attempted": int(len(persist_df)),
        "parquet_rows": int(parquet_rows),
        "sqlite_rows": int(sqlite_rows),
        "parquet_path": str(LIAB_PARQUET_PATH),
        "sqlite_path": str(LIAB_SQLITE_PATH),
    }


def _fetch_cached_liabilities_series(ticker: str, series_name: str) -> pd.DataFrame:
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return pd.DataFrame(columns=["end", "value", "form", "filed"])

    if LIAB_SQLITE_PATH.exists():
        with sqlite3.connect(LIAB_SQLITE_PATH) as conn:
            df = pd.read_sql_query(
                """
                SELECT
                    period_end AS end,
                    value,
                    form,
                    filed
                FROM liabilities_series
                WHERE ticker = ? AND series_name = ?
                ORDER BY period_end
                """,
                conn,
                params=(ticker, series_name),
            )
        if not df.empty:
            df["end"] = pd.to_datetime(df["end"], errors="coerce")
            if "filed" in df.columns:
                df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            return df.dropna(subset=["end", "value"]).sort_values("end")

    if LIAB_PARQUET_PATH.exists():
        df = _load_unified_parquet_dataset("liabilities_series", LIAB_PARQUET_PATH)
        if not df.empty:
            df = df[
                (df["ticker"].astype(str).str.upper() == ticker)
                & (df["series_name"] == series_name)
            ].copy()
            if not df.empty:
                df = df.rename(columns={"period_end": "end"})
                df["end"] = pd.to_datetime(df["end"], errors="coerce")
                if "filed" in df.columns:
                    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
                return (
                    df[["end", "value", "form", "filed"]]
                    .dropna(subset=["end", "value"])
                    .sort_values("end")
                )

    return pd.DataFrame(columns=["end", "value", "form", "filed"])


def _fetch_cached_liability_components(ticker: str) -> pd.DataFrame:
    history_df = _fetch_cached_liability_component_history(ticker)
    if history_df.empty:
        return pd.DataFrame(columns=["tag", "label", "end", "value", "form", "filed"])
    return _select_latest_segment_rows(history_df)


def _fetch_cached_liability_component_history(ticker: str) -> pd.DataFrame:
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return pd.DataFrame(columns=["tag", "label", "end", "value", "form", "filed"])

    if LIAB_SQLITE_PATH.exists():
        try:
            with sqlite3.connect(LIAB_SQLITE_PATH) as conn:
                df = pd.read_sql_query(
                    """
                    SELECT
                        tag,
                        label,
                        period_end AS end,
                        value,
                        form,
                        filed
                    FROM liabilities_components
                    WHERE ticker = ?
                    ORDER BY period_end, label
                    """,
                    conn,
                    params=(ticker,),
                )
        except Exception:
            df = pd.DataFrame()
        if not df.empty:
            df["end"] = pd.to_datetime(df["end"], errors="coerce")
            df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["end", "value"])
            if not df.empty:
                return df.sort_values(["label", "end", "filed"]).reset_index(drop=True)

    if LIAB_COMPONENT_PARQUET_PATH.exists():
        df = _load_unified_parquet_dataset(
            "liability_components", LIAB_COMPONENT_PARQUET_PATH
        )
        if not df.empty:
            df = df[df["ticker"].astype(str).str.upper() == ticker].copy()
            if not df.empty:
                df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
                df = df.rename(columns={"period_end": "end"})
                df["end"] = pd.to_datetime(df["end"], errors="coerce")
                df["filed"] = pd.to_datetime(df.get("filed"), errors="coerce")
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
                df = df.dropna(subset=["end", "value"])
                if not df.empty:
                    return (
                        df[["tag", "label", "end", "value", "form", "filed"]]
                        .sort_values(["label", "end", "filed"])
                        .reset_index(drop=True)
                    )

    return pd.DataFrame(columns=["tag", "label", "end", "value", "form", "filed"])


def _fetch_cached_debt_snapshot(ticker: str) -> dict:
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return {}

    if LIAB_SQLITE_PATH.exists():
        try:
            with sqlite3.connect(LIAB_SQLITE_PATH) as conn:
                df = pd.read_sql_query(
                    """
                    SELECT
                        ticker,
                        debt_balance_display,
                        accrued_interest_display,
                        implied_rate_display,
                        return_on_debt_display,
                        implied_rate_raw
                    FROM liabilities_debt_snapshot
                    WHERE ticker = ?
                    """,
                    conn,
                    params=(ticker,),
                )
            if not df.empty:
                row = df.iloc[-1].to_dict()
                return {
                    "debt_balance_display": row.get("debt_balance_display", "-"),
                    "accrued_interest_display": row.get(
                        "accrued_interest_display", "-"
                    ),
                    "implied_rate_display": row.get("implied_rate_display", "-"),
                    "return_on_debt_display": row.get("return_on_debt_display", "-"),
                    "implied_rate_raw": float(row["implied_rate_raw"])
                    if row.get("implied_rate_raw") is not None
                    and pd.notna(row.get("implied_rate_raw"))
                    else None,
                }
        except Exception:
            pass

    if LIAB_DEBT_SNAPSHOT_PARQUET_PATH.exists():
        df = _load_unified_parquet_dataset(
            "debt_snapshot", LIAB_DEBT_SNAPSHOT_PARQUET_PATH
        )
        if not df.empty:
            df = df[df["ticker"].astype(str).str.upper() == ticker].copy()
            if not df.empty:
                if "pulled_at_utc" in df.columns:
                    df["pulled_at_utc"] = pd.to_datetime(
                        df["pulled_at_utc"], errors="coerce"
                    )
                    df = df.sort_values("pulled_at_utc")
                row = df.iloc[-1].to_dict()
                return {
                    "debt_balance_display": row.get("debt_balance_display", "-"),
                    "accrued_interest_display": row.get(
                        "accrued_interest_display", "-"
                    ),
                    "implied_rate_display": row.get("implied_rate_display", "-"),
                    "return_on_debt_display": row.get("return_on_debt_display", "-"),
                    "implied_rate_raw": float(row["implied_rate_raw"])
                    if row.get("implied_rate_raw") is not None
                    and pd.notna(row.get("implied_rate_raw"))
                    else None,
                }

    return {}


def _fetch_cached_debt_schedule(ticker: str) -> pd.DataFrame:
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return pd.DataFrame(
            columns=["bucket", "tag", "label", "end", "value", "form", "filed", "unit"]
        )

    if LIAB_SQLITE_PATH.exists():
        try:
            with sqlite3.connect(LIAB_SQLITE_PATH) as conn:
                df = pd.read_sql_query(
                    """
                    SELECT
                        bucket,
                        tag,
                        label,
                        period_end AS end,
                        value,
                        form,
                        filed,
                        unit
                    FROM liabilities_debt_schedule
                    WHERE ticker = ?
                    ORDER BY period_end, bucket
                    """,
                    conn,
                    params=(ticker,),
                )
            if not df.empty:
                df["end"] = pd.to_datetime(df["end"], errors="coerce")
                df["filed"] = pd.to_datetime(df.get("filed"), errors="coerce")
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
                df = df.dropna(subset=["end", "value"])
                if not df.empty:
                    latest_end = df["end"].max()
                    return (
                        df[df["end"] == latest_end]
                        .sort_values("value", ascending=False)
                        .reset_index(drop=True)
                    )
        except Exception:
            pass

    if LIAB_DEBT_SCHEDULE_PARQUET_PATH.exists():
        df = _load_unified_parquet_dataset(
            "debt_schedule", LIAB_DEBT_SCHEDULE_PARQUET_PATH
        )
        if not df.empty:
            df = df[df["ticker"].astype(str).str.upper() == ticker].copy()
            if not df.empty:
                df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
                df["filed"] = pd.to_datetime(df.get("filed"), errors="coerce")
                df = df.dropna(subset=["period_end", "value"])
                if not df.empty:
                    latest_end = df["period_end"].max()
                    df = df[df["period_end"] == latest_end].copy()
                    df = df.rename(columns={"period_end": "end"})
                    return (
                        df[
                            [
                                "bucket",
                                "tag",
                                "label",
                                "end",
                                "value",
                                "form",
                                "filed",
                                "unit",
                            ]
                        ]
                        .sort_values("value", ascending=False)
                        .reset_index(drop=True)
                    )

    return pd.DataFrame(
        columns=["bucket", "tag", "label", "end", "value", "form", "filed", "unit"]
    )


def _fetch_cached_debt_detail(ticker: str) -> list[dict]:
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return []

    if LIAB_SQLITE_PATH.exists():
        try:
            with sqlite3.connect(LIAB_SQLITE_PATH) as conn:
                df = pd.read_sql_query(
                    """
                    SELECT
                        category,
                        metric,
                        value_text,
                        period_end,
                        form,
                        source_tag
                    FROM liabilities_debt_detail
                    WHERE ticker = ?
                    ORDER BY category, metric
                    """,
                    conn,
                    params=(ticker,),
                )
            if not df.empty:
                return [
                    {
                        "Category": row.category,
                        "Metric": row.metric,
                        "Value": row.value_text,
                        "Period End": row.period_end or "",
                        "Form": row.form or "",
                        "Source Tag": row.source_tag or "",
                    }
                    for row in df.itertuples(index=False)
                ]
        except Exception:
            pass

    if LIAB_DEBT_DETAIL_PARQUET_PATH.exists():
        df = _load_unified_parquet_dataset("debt_detail", LIAB_DEBT_DETAIL_PARQUET_PATH)
        if not df.empty:
            df = df[df["ticker"].astype(str).str.upper() == ticker].copy()
            if not df.empty:
                if "pulled_at_utc" in df.columns:
                    df["pulled_at_utc"] = pd.to_datetime(
                        df["pulled_at_utc"], errors="coerce"
                    )
                    latest_pull = df["pulled_at_utc"].max()
                    if pd.notna(latest_pull):
                        df = df[df["pulled_at_utc"] == latest_pull]
                return [
                    {
                        "Category": row.category,
                        "Metric": row.metric,
                        "Value": row.value_text,
                        "Period End": row.period_end or "",
                        "Form": row.form or "",
                        "Source Tag": row.source_tag or "",
                    }
                    for row in df.itertuples(index=False)
                ]

    return []


def _empty_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#111827"),
        margin=dict(l=30, r=30, t=60, b=40),
    )
    return fig


def _get_cik(ticker: str):
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return None

    cached = _CIK_CACHE.get(ticker)
    if cached:
        return cached

    url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    for item in data.values():
        if str(item.get("ticker", "")).upper() == ticker:
            cik = str(item.get("cik_str", "")).zfill(10)
            _CIK_CACHE[ticker] = cik
            return cik
    return None


def _get_company_facts(cik: str):
    cached = _FACTS_CACHE.get(cik)
    if cached is not None:
        return cached

    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    payload = resp.json()
    _FACTS_CACHE[cik] = payload
    return payload


def _extract_tag_units(facts: dict, tag: str) -> pd.DataFrame:
    try:
        unit_map = facts["facts"]["us-gaap"][tag]["units"]
    except Exception:
        return pd.DataFrame()

    frames = []
    for unit_name, entries in unit_map.items():
        if "usd" not in unit_name.lower():
            continue
        df = pd.DataFrame(entries)
        if df.empty or "val" not in df.columns or "end" not in df.columns:
            continue
        df["unit"] = unit_name
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    merged["tag"] = tag
    return merged


def _extract_series_for_forms(
    facts: dict, tags: list[str], forms: set[str]
) -> pd.DataFrame:
    tag_frames = []
    for priority, tag in enumerate(tags):
        raw = _extract_tag_units(facts, tag)
        if raw.empty:
            continue

        working = raw.copy()
        if "form" in working.columns:
            working = working[working["form"].isin(forms)]
        if working.empty:
            continue

        if "filed" in working.columns:
            working["filed"] = pd.to_datetime(working["filed"], errors="coerce")
        else:
            working["filed"] = pd.NaT

        working["end"] = pd.to_datetime(working["end"], errors="coerce")
        working["val"] = pd.to_numeric(working["val"], errors="coerce")
        working = working.dropna(subset=["end", "val"])
        if working.empty:
            continue

        # Keep latest filing per period within each tag before cross-tag stitching.
        working = working.sort_values(["end", "filed"]).drop_duplicates(
            subset=["end"], keep="last"
        )
        working["tag_priority"] = priority
        tag_frames.append(working[["end", "val", "form", "filed", "tag_priority"]])

    if not tag_frames:
        return pd.DataFrame(columns=["end", "value", "form", "filed"])

    stitched = pd.concat(tag_frames, ignore_index=True)
    # Prefer earlier tags in the configured tag list; use filed recency as tie-breaker.
    stitched = stitched.sort_values(
        ["end", "tag_priority", "filed"], ascending=[True, True, False]
    ).drop_duplicates(subset=["end"], keep="first")

    stitched = stitched.sort_values("end")
    return stitched[["end", "val", "form", "filed"]].rename(columns={"val": "value"})


def _derive_total_liabilities_series(
    facts: dict,
    forms: set[str],
) -> pd.DataFrame:
    assets_df = _extract_series_for_forms(facts, TOTAL_ASSETS_TAGS, forms)
    equity_df = _extract_series_for_forms(facts, TOTAL_EQUITY_TAGS, forms)
    liab_plus_equity_df = _extract_series_for_forms(
        facts, LIABILITIES_AND_EQUITY_TAGS, forms
    )

    # Preferred: LiabilitiesAndStockholdersEquity - StockholdersEquity
    if not liab_plus_equity_df.empty and not equity_df.empty:
        merged = liab_plus_equity_df.merge(
            equity_df,
            on="end",
            how="inner",
            suffixes=("_lpe", "_eq"),
        )
        if not merged.empty:
            merged["value"] = merged["value_lpe"] - merged["value_eq"]
            merged["form"] = merged.get("form_lpe", "").fillna(
                merged.get("form_eq", "")
            )
            filed_cols = [
                col for col in ["filed_lpe", "filed_eq"] if col in merged.columns
            ]
            merged["filed"] = merged[filed_cols].max(axis=1) if filed_cols else pd.NaT
            merged = merged[["end", "value", "form", "filed"]]
            merged["value"] = pd.to_numeric(merged["value"], errors="coerce")
            merged = merged.dropna(subset=["end", "value"]).sort_values("end")
            if not merged.empty:
                return merged

    # Secondary: Assets - StockholdersEquity
    if not assets_df.empty and not equity_df.empty:
        merged = assets_df.merge(
            equity_df,
            on="end",
            how="inner",
            suffixes=("_assets", "_eq"),
        )
        if not merged.empty:
            merged["value"] = merged["value_assets"] - merged["value_eq"]
            merged["form"] = merged.get("form_assets", "").fillna(
                merged.get("form_eq", "")
            )
            filed_cols = [
                col for col in ["filed_assets", "filed_eq"] if col in merged.columns
            ]
            merged["filed"] = merged[filed_cols].max(axis=1) if filed_cols else pd.NaT
            merged = merged[["end", "value", "form", "filed"]]
            merged["value"] = pd.to_numeric(merged["value"], errors="coerce")
            merged = merged.dropna(subset=["end", "value"]).sort_values("end")
            if not merged.empty:
                return merged

    return pd.DataFrame(columns=["end", "value", "form", "filed"])


def _fmt_money(value):
    if value is None or pd.isna(value):
        return "-"
    abs_val = abs(float(value))
    if abs_val >= 1_000_000_000:
        return f"${value / 1_000_000_000:,.2f}B"
    if abs_val >= 1_000_000:
        return f"${value / 1_000_000:,.2f}M"
    return f"${value:,.0f}"


def _fmt_pct(value):
    if value is None or pd.isna(value):
        return "-"
    return f"{value * 100:,.2f}%"


def _prettify_xbrl_tag(tag: str) -> str:
    label = LIABILITY_COMPONENT_LABELS.get(tag)
    if label:
        return label

    pretty = re.sub(r"(?<!^)(?=[A-Z])", " ", str(tag or "")).strip()
    pretty = pretty.replace("Noncurrent", "Non-current")
    pretty = pretty.replace("And", " and ")
    return " ".join(pretty.split()) or str(tag)


def _extract_liability_component_row(
    payload: dict,
    tag: str,
    forms: set[str] | None,
    target_end: pd.Timestamp | None,
) -> pd.DataFrame:
    if payload is None or not isinstance(payload, dict):
        return pd.DataFrame(columns=["tag", "label", "end", "value", "form", "filed"])

    units = payload.get("units", {})
    if not units:
        return pd.DataFrame(columns=["tag", "label", "end", "value", "form", "filed"])

    for unit_name, entries in units.items():
        if "usd" not in str(unit_name).lower():
            continue

        df = pd.DataFrame(entries)
        if df.empty or "val" not in df.columns or "end" not in df.columns:
            continue

        if forms and "form" in df.columns:
            df = df[df["form"].isin(forms)]
        if df.empty:
            continue

        df["end"] = pd.to_datetime(df["end"], errors="coerce")
        df["filed"] = pd.to_datetime(df.get("filed"), errors="coerce")
        df["val"] = pd.to_numeric(df["val"], errors="coerce")
        df = df.dropna(subset=["end", "val"])
        if df.empty:
            continue

        if target_end is not None:
            exact = df[df["end"] == target_end]
            if not exact.empty:
                df = exact
            else:
                deltas = (df["end"] - target_end).abs()
                df = df[deltas <= pd.Timedelta(days=35)]
                if df.empty:
                    continue

        df = df.sort_values(["end", "filed"]).tail(1).copy()
        df["tag"] = tag
        df["label"] = _prettify_xbrl_tag(tag)
        return df[["tag", "label", "end", "val", "form", "filed"]].rename(
            columns={"val": "value"}
        )

    return pd.DataFrame(columns=["tag", "label", "end", "value", "form", "filed"])


def _extract_liability_components(
    facts: dict,
    forms: set[str] | None = None,
    target_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    try:
        us_gaap = facts["facts"]["us-gaap"]
    except Exception:
        return pd.DataFrame(columns=["tag", "label", "end", "value", "form", "filed"])

    frames = []
    for tag, payload in us_gaap.items():
        tag_name = str(tag)
        tag_lower = tag_name.lower()
        if tag_name in LIABILITY_COMPONENT_BLACKLIST:
            continue
        if not any(keyword in tag_lower for keyword in LIABILITY_COMPONENT_KEYWORDS):
            continue
        if any(
            keyword in tag_lower for keyword in LIABILITY_COMPONENT_EXCLUDE_KEYWORDS
        ):
            continue

        row = _extract_liability_component_row(payload, tag_name, forms, target_end)
        if not row.empty:
            frames.append(row)

    if not frames:
        return pd.DataFrame(columns=["tag", "label", "end", "value", "form", "filed"])

    components = pd.concat(frames, ignore_index=True)
    components = components.sort_values(["value", "filed"], ascending=[False, False])
    components = components.drop_duplicates(subset=["label"], keep="first")
    return components.sort_values("value", ascending=False).reset_index(drop=True)


def _latest_series_row(series_df: pd.DataFrame) -> pd.DataFrame:
    if series_df is None or series_df.empty:
        return pd.DataFrame(columns=["end", "value", "form", "filed"])

    working = series_df.copy()
    working["end"] = pd.to_datetime(working["end"], errors="coerce")
    working["filed"] = pd.to_datetime(working.get("filed"), errors="coerce")
    working["value"] = pd.to_numeric(working.get("value"), errors="coerce")
    working = working.dropna(subset=["end", "value"]).sort_values(["end", "filed"])
    if working.empty:
        return pd.DataFrame(columns=["end", "value", "form", "filed"])
    return working.tail(1)[["end", "value", "form", "filed"]].copy()


def _build_segment_row_from_series(
    label: str,
    tag: str,
    series_df: pd.DataFrame,
) -> pd.DataFrame:
    row = _latest_series_row(series_df)
    if row.empty:
        return pd.DataFrame(columns=["tag", "label", "end", "value", "form", "filed"])

    row["tag"] = tag
    row["label"] = label
    return row[["tag", "label", "end", "value", "form", "filed"]]


def _build_segment_row_from_component_source(
    component_df: pd.DataFrame,
    definition: dict,
) -> pd.DataFrame:
    if component_df is None or component_df.empty:
        return pd.DataFrame(columns=["tag", "label", "end", "value", "form", "filed"])

    working = component_df.copy()
    working["end"] = pd.to_datetime(working.get("end"), errors="coerce")
    working["filed"] = pd.to_datetime(working.get("filed"), errors="coerce")
    working["value"] = pd.to_numeric(working.get("value"), errors="coerce")
    working = working.dropna(subset=["end", "value"])
    if working.empty:
        return pd.DataFrame(columns=["tag", "label", "end", "value", "form", "filed"])

    tags = definition.get("tags", [])
    mode = definition.get("mode")
    matched = working[working["tag"].isin(tags)].copy()
    if matched.empty:
        return pd.DataFrame(columns=["tag", "label", "end", "value", "form", "filed"])

    latest_end = matched["end"].max()
    matched = matched[matched["end"] == latest_end].copy()
    if matched.empty:
        return pd.DataFrame(columns=["tag", "label", "end", "value", "form", "filed"])

    if mode == "combine":
        forms = sorted(
            {
                str(value)
                for value in matched.get("form", pd.Series(dtype=str)).fillna("")
                if str(value)
            }
        )
        filed = matched["filed"].max() if "filed" in matched.columns else pd.NaT
        return pd.DataFrame(
            [
                {
                    "tag": "/".join(tags),
                    "label": definition["label"],
                    "end": latest_end,
                    "value": matched["value"].sum(),
                    "form": "/".join(forms),
                    "filed": filed,
                }
            ]
        )

    for tag in tags:
        tag_match = matched[matched["tag"] == tag].copy()
        if tag_match.empty:
            continue
        tag_match = tag_match.sort_values(["end", "filed"]).tail(1).copy()
        tag_match["label"] = definition["label"]
        return tag_match[["tag", "label", "end", "value", "form", "filed"]]

    return pd.DataFrame(columns=["tag", "label", "end", "value", "form", "filed"])


def _build_selected_liability_segments(
    component_df: pd.DataFrame,
    q_curr: pd.DataFrame,
    total_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for definition in LIABILITY_SEGMENT_DEFINITIONS:
        mode = definition.get("mode")
        if mode == "series":
            if definition.get("series") == "current_total":
                row = _build_segment_row_from_series(
                    label=definition["label"],
                    tag=definition["tag"],
                    series_df=q_curr,
                )
            else:
                row = _build_segment_row_from_series(
                    label=definition["label"],
                    tag=definition["tag"],
                    series_df=total_df,
                )
        else:
            row = _build_segment_row_from_component_source(component_df, definition)

        if not row.empty:
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["tag", "label", "end", "value", "form", "filed"])

    selected = pd.concat(rows, ignore_index=True)
    selected["segment_order"] = selected["label"].map(LIABILITY_SEGMENT_ORDER)
    selected = selected.sort_values(["segment_order", "end"]).drop(
        columns=["segment_order"]
    )
    return selected.reset_index(drop=True)


def _extract_selected_liability_segments(
    facts: dict,
    forms: set[str] | None,
    target_end: pd.Timestamp | None,
    q_curr: pd.DataFrame,
    q_total: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for definition in LIABILITY_SEGMENT_DEFINITIONS:
        mode = definition.get("mode")
        if mode == "series":
            if definition.get("series") == "current_total":
                row = _build_segment_row_from_series(
                    label=definition["label"],
                    tag=definition["tag"],
                    series_df=q_curr,
                )
            else:
                row = _build_segment_row_from_series(
                    label=definition["label"],
                    tag=definition["tag"],
                    series_df=q_total,
                )
        elif mode == "combine":
            combine_rows = []
            for tag in definition.get("tags", []):
                tag_row = _extract_latest_metric_by_tags(
                    facts,
                    [tag],
                    forms=forms,
                    target_end=target_end,
                )
                if not tag_row.empty:
                    combine_rows.append(tag_row)
            if combine_rows:
                combined = pd.concat(combine_rows, ignore_index=True)
                filed = pd.to_datetime(combined.get("filed"), errors="coerce").max()
                forms_joined = sorted(
                    {
                        str(value)
                        for value in combined.get("form", pd.Series(dtype=str)).fillna(
                            ""
                        )
                        if str(value)
                    }
                )
                row = pd.DataFrame(
                    [
                        {
                            "tag": "/".join(definition.get("tags", [])),
                            "label": definition["label"],
                            "end": pd.to_datetime(
                                combined["end"], errors="coerce"
                            ).max(),
                            "value": pd.to_numeric(
                                combined["value"], errors="coerce"
                            ).sum(),
                            "form": "/".join(forms_joined),
                            "filed": filed,
                        }
                    ]
                )
            else:
                row = pd.DataFrame(
                    columns=["tag", "label", "end", "value", "form", "filed"]
                )
        else:
            row = _extract_latest_metric_by_tags(
                facts,
                definition.get("tags", []),
                forms=forms,
                target_end=target_end,
            )
            if not row.empty:
                row = row[["tag", "end", "value", "form", "filed"]].copy()
                row["label"] = definition["label"]
                row = row[["tag", "label", "end", "value", "form", "filed"]]

        if not row.empty:
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["tag", "label", "end", "value", "form", "filed"])

    selected = pd.concat(rows, ignore_index=True)
    selected["segment_order"] = selected["label"].map(LIABILITY_SEGMENT_ORDER)
    selected = selected.sort_values(["segment_order", "end"]).drop(
        columns=["segment_order"]
    )
    return selected.reset_index(drop=True)


def _fallback_liability_components_from_split(
    q_curr: pd.DataFrame,
    q_noncurr: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    for frame, label, tag in [
        (q_curr, "Current Liabilities", "LiabilitiesCurrent"),
        (q_noncurr, "Non-current Liabilities", "LiabilitiesNoncurrent"),
    ]:
        if frame is None or frame.empty:
            continue
        working = frame.copy().sort_values("end")
        row = working.tail(1).copy()
        row["label"] = label
        row["tag"] = tag
        frames.append(row[["tag", "label", "end", "value", "form", "filed"]])

    if not frames:
        return pd.DataFrame(columns=["tag", "label", "end", "value", "form", "filed"])

    return pd.concat(frames, ignore_index=True).sort_values("value", ascending=False)


def _build_component_figure(
    component_df: pd.DataFrame,
    ticker: str,
    total_value: float | None = None,
) -> go.Figure:
    if component_df is None or component_df.empty:
        return _empty_figure(f"No liability component breakdown found for {ticker}")

    chart_df = component_df.copy()
    chart_df["segment_order"] = chart_df["label"].map(LIABILITY_SEGMENT_ORDER)
    chart_df = chart_df.sort_values(["segment_order", "value"], ascending=[False, True])
    chart_df = chart_df.drop(columns=["segment_order"])
    share = None
    if total_value is not None and pd.notna(total_value) and total_value != 0:
        share = (chart_df["value"] / total_value).fillna(0)

    fig = go.Figure(
        go.Bar(
            x=chart_df["value"],
            y=chart_df["label"],
            orientation="h",
            marker_color="#0f766e",
            text=chart_df["value"].apply(_fmt_money),
            textposition="outside",
            customdata=(share.to_frame(name="share") if share is not None else None),
            hovertemplate=(
                "%{y}<br>Value: %{x:$,.0f}<br>Share of total: %{customdata[0]:.1%}<extra></extra>"
                if share is not None
                else "%{y}<br>Value: %{x:$,.0f}<extra></extra>"
            ),
        )
    )
    latest_end = pd.to_datetime(chart_df["end"], errors="coerce").max()
    fig.update_layout(
        title=(
            f"Liability Components - {ticker} ({latest_end.strftime('%Y-%m-%d')})"
            if pd.notna(latest_end)
            else f"Liability Components - {ticker}"
        ),
        template="plotly_white",
        xaxis_title="USD",
        yaxis_title="",
        margin=dict(l=30, r=30, t=60, b=40),
        height=max(260, 60 + (len(chart_df) * 28)),
    )
    return fig


def _build_component_table_data(
    component_df: pd.DataFrame,
    total_value: float | None = None,
    component_history_df: pd.DataFrame | None = None,
) -> list[dict]:
    if component_df is None or component_df.empty:
        return []

    table_df = component_df.copy()
    table_df["segment_order"] = table_df["label"].map(LIABILITY_SEGMENT_ORDER)
    table_df = table_df.sort_values(["segment_order", "end"]).drop(
        columns=["segment_order"]
    )
    table_df["Period End"] = pd.to_datetime(
        table_df["end"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    table_df["Liability Component"] = table_df["label"]
    table_df["Value"] = table_df["value"].apply(_fmt_money)
    if total_value is not None and pd.notna(total_value) and total_value != 0:
        table_df["Share of Total"] = (table_df["value"] / total_value).apply(_fmt_pct)
    else:
        table_df["Share of Total"] = "-"
    change_map = _compute_segment_change_values(component_history_df)
    table_df["Qtr % Change"] = table_df["label"].map(
        lambda label: change_map.get(label, {}).get("Qtr % Change", "-")
    )
    table_df["Year % Change"] = table_df["label"].map(
        lambda label: change_map.get(label, {}).get("Year % Change", "-")
    )
    table_df["Form"] = table_df.get("form", "")
    table_df["Filed"] = pd.to_datetime(
        table_df.get("filed"), errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    table_df["Filed"] = table_df["Filed"].fillna("")
    table_df["Tag"] = table_df["tag"]

    return table_df[
        [
            "Liability Component",
            "Value",
            "Share of Total",
            "Qtr % Change",
            "Year % Change",
            "Period End",
            "Form",
            "Filed",
            "Tag",
        ]
    ].to_dict("records")


def _combine_series_frames(
    left_df: pd.DataFrame, right_df: pd.DataFrame
) -> pd.DataFrame:
    if left_df is None or left_df.empty:
        return right_df.copy() if right_df is not None else pd.DataFrame()
    if right_df is None or right_df.empty:
        return left_df.copy()

    left = left_df.copy()
    right = right_df.copy()
    left["end"] = pd.to_datetime(left["end"], errors="coerce")
    right["end"] = pd.to_datetime(right["end"], errors="coerce")
    left = left.dropna(subset=["end"])
    right = right.dropna(subset=["end"])

    merged = left.merge(right, on="end", how="outer", suffixes=("_left", "_right"))
    merged["value"] = merged[["value_left", "value_right"]].fillna(0).sum(axis=1)
    merged["form"] = merged.get("form_left", "").fillna(merged.get("form_right", ""))
    filed_cols = [col for col in ["filed_left", "filed_right"] if col in merged.columns]
    if filed_cols:
        merged["filed"] = merged[filed_cols].max(axis=1)
    else:
        merged["filed"] = pd.NaT
    return merged[["end", "value", "form", "filed"]].sort_values("end")


def _extract_debt_series(facts: dict, forms: set[str]) -> pd.DataFrame:
    total_df = _extract_series_for_forms(facts, DEBT_TOTAL_TAGS, forms)
    if total_df is not None and not total_df.empty:
        return total_df.sort_values("end")

    current_df = _extract_series_for_forms(facts, DEBT_CURRENT_TAGS, forms)
    noncurrent_df = _extract_series_for_forms(facts, DEBT_NONCURRENT_TAGS, forms)
    combined = _combine_series_frames(current_df, noncurrent_df)
    return (
        combined
        if combined is not None
        else pd.DataFrame(columns=["end", "value", "form", "filed"])
    )


def _extract_latest_metric_by_tags(
    facts: dict,
    tags: list[str],
    forms: set[str] | None = None,
    target_end: pd.Timestamp | None = None,
    unit_keywords: tuple[str, ...] = ("usd",),
) -> pd.DataFrame:
    try:
        us_gaap = facts["facts"]["us-gaap"]
    except Exception:
        return pd.DataFrame(
            columns=["tag", "label", "end", "value", "form", "filed", "unit"]
        )

    for tag in tags:
        payload = us_gaap.get(tag)
        if not isinstance(payload, dict):
            continue
        units = payload.get("units", {})
        for unit_name, entries in units.items():
            unit_lower = str(unit_name).lower()
            if unit_keywords and not any(
                keyword in unit_lower for keyword in unit_keywords
            ):
                continue

            df = pd.DataFrame(entries)
            if df.empty or "val" not in df.columns or "end" not in df.columns:
                continue
            if forms and "form" in df.columns:
                df = df[df["form"].isin(forms)]
            if df.empty:
                continue

            df["end"] = pd.to_datetime(df["end"], errors="coerce")
            df["filed"] = pd.to_datetime(df.get("filed"), errors="coerce")
            df["val"] = pd.to_numeric(df["val"], errors="coerce")
            df = df.dropna(subset=["end", "val"])
            if df.empty:
                continue

            if target_end is not None:
                exact = df[df["end"] == target_end]
                if not exact.empty:
                    df = exact
                else:
                    deltas = (df["end"] - target_end).abs()
                    df = df[deltas <= pd.Timedelta(days=35)]
                    if df.empty:
                        continue

            df = df.sort_values(["end", "filed"]).tail(1).copy()
            df["tag"] = tag
            df["label"] = _prettify_xbrl_tag(tag)
            df["unit"] = unit_name
            return df[["tag", "label", "end", "val", "form", "filed", "unit"]].rename(
                columns={"val": "value"}
            )

    return pd.DataFrame(
        columns=["tag", "label", "end", "value", "form", "filed", "unit"]
    )


def _extract_metric_history_by_tags(
    facts: dict,
    tags: list[str],
    forms: set[str] | None = None,
    unit_keywords: tuple[str, ...] = ("usd",),
) -> pd.DataFrame:
    try:
        us_gaap = facts["facts"]["us-gaap"]
    except Exception:
        return pd.DataFrame(columns=["tag", "label", "end", "value", "form", "filed"])

    for tag in tags:
        payload = us_gaap.get(tag)
        if not isinstance(payload, dict):
            continue
        units = payload.get("units", {})
        frames = []
        for unit_name, entries in units.items():
            unit_lower = str(unit_name).lower()
            if unit_keywords and not any(
                keyword in unit_lower for keyword in unit_keywords
            ):
                continue

            df = pd.DataFrame(entries)
            if df.empty or "val" not in df.columns or "end" not in df.columns:
                continue
            if forms and "form" in df.columns:
                df = df[df["form"].isin(forms)]
            if df.empty:
                continue

            df["end"] = pd.to_datetime(df["end"], errors="coerce")
            df["filed"] = pd.to_datetime(df.get("filed"), errors="coerce")
            df["val"] = pd.to_numeric(df["val"], errors="coerce")
            df = df.dropna(subset=["end", "val"])
            if df.empty:
                continue

            df["tag"] = tag
            df["label"] = _prettify_xbrl_tag(tag)
            frames.append(df[["tag", "label", "end", "val", "form", "filed"]])

        if frames:
            history = pd.concat(frames, ignore_index=True).rename(
                columns={"val": "value"}
            )
            history = history.sort_values(["end", "filed"]).drop_duplicates(
                subset=["end"], keep="last"
            )
            return history.reset_index(drop=True)

    return pd.DataFrame(columns=["tag", "label", "end", "value", "form", "filed"])


def _build_segment_history_from_series(
    label: str,
    tag: str,
    series_df: pd.DataFrame,
) -> pd.DataFrame:
    if series_df is None or series_df.empty:
        return pd.DataFrame(columns=["tag", "label", "end", "value", "form", "filed"])

    working = series_df.copy()
    working["end"] = pd.to_datetime(working.get("end"), errors="coerce")
    working["filed"] = pd.to_datetime(working.get("filed"), errors="coerce")
    working["value"] = pd.to_numeric(working.get("value"), errors="coerce")
    working = working.dropna(subset=["end", "value"]).sort_values(["end", "filed"])
    if working.empty:
        return pd.DataFrame(columns=["tag", "label", "end", "value", "form", "filed"])

    working["tag"] = tag
    working["label"] = label
    return working[["tag", "label", "end", "value", "form", "filed"]].reset_index(
        drop=True
    )


def _extract_selected_liability_segment_history(
    facts: dict,
    forms: set[str] | None,
    q_curr: pd.DataFrame,
    q_total: pd.DataFrame,
) -> pd.DataFrame:
    histories = []
    for definition in LIABILITY_SEGMENT_DEFINITIONS:
        mode = definition.get("mode")
        if mode == "series":
            series_df = (
                q_curr if definition.get("series") == "current_total" else q_total
            )
            history = _build_segment_history_from_series(
                label=definition["label"],
                tag=definition["tag"],
                series_df=series_df,
            )
        elif mode == "combine":
            combine_frames = []
            for tag in definition.get("tags", []):
                tag_history = _extract_metric_history_by_tags(facts, [tag], forms=forms)
                if not tag_history.empty:
                    combine_frames.append(tag_history)

            if combine_frames:
                combined = pd.concat(combine_frames, ignore_index=True)
                combined["end"] = pd.to_datetime(combined["end"], errors="coerce")
                combined["filed"] = pd.to_datetime(combined["filed"], errors="coerce")
                combined["value"] = pd.to_numeric(combined["value"], errors="coerce")
                combined = combined.dropna(subset=["end", "value"])
                history = (
                    combined.groupby("end", as_index=False)
                    .agg(
                        value=("value", "sum"),
                        filed=("filed", "max"),
                        form=(
                            "form",
                            lambda values: "/".join(
                                sorted({str(v) for v in values if str(v)})
                            ),
                        ),
                    )
                    .sort_values("end")
                )
                history["tag"] = "/".join(definition.get("tags", []))
                history["label"] = definition["label"]
                history = history[["tag", "label", "end", "value", "form", "filed"]]
            else:
                history = pd.DataFrame(
                    columns=["tag", "label", "end", "value", "form", "filed"]
                )
        else:
            history = _extract_metric_history_by_tags(
                facts,
                definition.get("tags", []),
                forms=forms,
            )
            if not history.empty:
                history = history[["tag", "end", "value", "form", "filed"]].copy()
                history["label"] = definition["label"]
                history = history[["tag", "label", "end", "value", "form", "filed"]]

        if not history.empty:
            histories.append(history)

    if not histories:
        return pd.DataFrame(columns=["tag", "label", "end", "value", "form", "filed"])

    history_df = pd.concat(histories, ignore_index=True)
    history_df["segment_order"] = history_df["label"].map(LIABILITY_SEGMENT_ORDER)
    history_df = history_df.sort_values(["segment_order", "end", "filed"]).drop(
        columns=["segment_order"]
    )
    return history_df.reset_index(drop=True)


def _select_latest_segment_rows(history_df: pd.DataFrame) -> pd.DataFrame:
    if history_df is None or history_df.empty:
        return pd.DataFrame(columns=["tag", "label", "end", "value", "form", "filed"])

    working = history_df.copy()
    working["end"] = pd.to_datetime(working.get("end"), errors="coerce")
    working["filed"] = pd.to_datetime(working.get("filed"), errors="coerce")
    working["value"] = pd.to_numeric(working.get("value"), errors="coerce")
    working = working.dropna(subset=["label", "end", "value"])
    if working.empty:
        return pd.DataFrame(columns=["tag", "label", "end", "value", "form", "filed"])

    latest = (
        working.sort_values(["label", "end", "filed"])
        .groupby("label", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    latest["segment_order"] = latest["label"].map(LIABILITY_SEGMENT_ORDER)
    latest = latest.sort_values(["segment_order", "end"]).drop(
        columns=["segment_order"]
    )
    return latest.reset_index(drop=True)


def _compute_segment_change_values(history_df: pd.DataFrame) -> dict:
    if history_df is None or history_df.empty:
        return {}

    working = history_df.copy()
    working["end"] = pd.to_datetime(working.get("end"), errors="coerce")
    working["filed"] = pd.to_datetime(working.get("filed"), errors="coerce")
    working["value"] = pd.to_numeric(working.get("value"), errors="coerce")
    working = working.dropna(subset=["label", "end", "value"])
    if working.empty:
        return {}

    results = {}
    for label, group in working.groupby("label"):
        group = group.sort_values(["end", "filed"]).reset_index(drop=True)
        latest = group.iloc[-1]
        latest_end = latest["end"]
        latest_value = latest["value"]
        latest_form = str(latest.get("form", ""))

        qtr_change = None
        previous_rows = group[group["end"] < latest_end]
        if not previous_rows.empty and "10-Q" in latest_form:
            previous_value = previous_rows.iloc[-1]["value"]
            if pd.notna(previous_value) and previous_value not in (0, None):
                qtr_change = (latest_value / previous_value) - 1

        year_change = None
        if not previous_rows.empty:
            if "10-Q" in latest_form:
                target_end = latest_end - pd.DateOffset(years=1)
                candidate_rows = previous_rows[
                    (previous_rows["end"] >= target_end - pd.Timedelta(days=45))
                    & (previous_rows["end"] <= target_end + pd.Timedelta(days=45))
                ].copy()
                if not candidate_rows.empty:
                    candidate_rows["delta_days"] = (
                        candidate_rows["end"] - target_end
                    ).abs()
                    year_value = candidate_rows.sort_values("delta_days").iloc[0][
                        "value"
                    ]
                    if pd.notna(year_value) and year_value not in (0, None):
                        year_change = (latest_value / year_value) - 1
            else:
                year_value = previous_rows.iloc[-1]["value"]
                if pd.notna(year_value) and year_value not in (0, None):
                    year_change = (latest_value / year_value) - 1

        results[label] = {
            "Qtr % Change": _fmt_pct(qtr_change),
            "Year % Change": _fmt_pct(year_change),
        }

    return results


def _extract_debt_maturity_schedule(
    facts: dict,
    target_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    for forms in ({"10-K"}, {"10-K", "10-Q"}, None):
        rows = []
        for bucket_label, tags in DEBT_MATURITY_BUCKETS:
            row = _extract_latest_metric_by_tags(
                facts,
                tags,
                forms=forms,
                target_end=target_end,
            )
            if row.empty:
                continue
            row = row.copy()
            row["bucket"] = bucket_label
            rows.append(row)

        if rows:
            return pd.concat(rows, ignore_index=True)

    return pd.DataFrame(
        columns=["bucket", "tag", "label", "end", "value", "form", "filed", "unit"]
    )


def _build_debt_schedule_figure(
    schedule_df: pd.DataFrame,
    ticker: str,
    stress_enabled: bool = False,
    stress_spread_bps: float = 0.0,
) -> go.Figure:
    if schedule_df is None or schedule_df.empty:
        return _empty_figure(f"No debt maturity schedule found for {ticker}")

    chart_df = schedule_df.copy()
    chart_df["value"] = pd.to_numeric(chart_df["value"], errors="coerce")
    chart_df = chart_df.dropna(subset=["value"])
    if chart_df.empty:
        return _empty_figure(f"No debt maturity schedule found for {ticker}")

    total_due = float(chart_df["value"].sum())
    chart_df["share"] = chart_df["value"] / total_due if total_due > 0 else 0.0
    chart_df["share_label"] = chart_df["share"].apply(lambda x: f"{x * 100:,.1f}%")
    chart_df["cum_share"] = chart_df["share"].cumsum()

    stress_rate = max(float(stress_spread_bps), 0.0) / 10_000.0
    chart_df["stress_cost"] = (
        chart_df["value"] * stress_rate if stress_enabled and stress_rate > 0 else 0.0
    )

    fig = go.Figure(
        go.Bar(
            x=chart_df["bucket"],
            y=chart_df["value"],
            marker_color="#7c3aed",
            text=chart_df.apply(
                lambda row: f"{_fmt_money(row['value'])}<br>{row['share_label']}",
                axis=1,
            ),
            textposition="outside",
            customdata=chart_df[["share"]],
            hovertemplate="%{x}<br>Principal due: %{y:$,.0f}<br>Share: %{customdata[0]:.1%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart_df["bucket"],
            y=chart_df["cum_share"],
            mode="lines+markers",
            name="Cumulative %",
            yaxis="y2",
            line=dict(color="#0f766e", width=2.5),
            hovertemplate="%{x}<br>Cumulative: %{y:.1%}<extra></extra>",
        )
    )

    if stress_enabled and stress_rate > 0 and chart_df["stress_cost"].sum() > 0:
        fig.add_trace(
            go.Bar(
                x=chart_df["bucket"],
                y=chart_df["stress_cost"],
                name="Stress Interest (annual)",
                marker_color="#f59e0b",
                opacity=0.6,
                hovertemplate="%{x}<br>Est. annual interest impact: %{y:$,.0f}<extra></extra>",
            )
        )

    latest_end = pd.to_datetime(chart_df["end"], errors="coerce").max()
    fig.update_layout(
        title=(
            f"Debt Payment Schedule - {ticker} ({latest_end.strftime('%Y-%m-%d')})"
            if pd.notna(latest_end)
            else f"Debt Payment Schedule - {ticker}"
        ),
        template="plotly_white",
        xaxis_title="Maturity Bucket",
        yaxis_title="Principal Due",
        yaxis2=dict(
            title="Cumulative Share",
            overlaying="y",
            side="right",
            tickformat=".0%",
            range=[0, 1.05],
            showgrid=False,
        ),
        barmode="group",
        margin=dict(l=30, r=30, t=60, b=40),
    )
    return fig


def _build_debt_detail_table_data(
    metric_rows: list[dict],
    schedule_df: pd.DataFrame,
) -> list[dict]:
    rows = list(metric_rows or [])
    if schedule_df is not None and not schedule_df.empty:
        for row in schedule_df.itertuples(index=False):
            rows.append(
                {
                    "Category": "Maturity Schedule",
                    "Metric": row.bucket,
                    "Value": _fmt_money(row.value),
                    "Period End": pd.to_datetime(row.end, errors="coerce").strftime(
                        "%Y-%m-%d"
                    )
                    if pd.notna(pd.to_datetime(row.end, errors="coerce"))
                    else "",
                    "Form": row.form or "",
                    "Source Tag": row.tag,
                }
            )
    return rows


def _build_total_liabilities_figure(
    q_df: pd.DataFrame, a_df: pd.DataFrame, ticker: str
) -> go.Figure:
    fig = go.Figure()

    if not q_df.empty:
        fig.add_trace(
            go.Scatter(
                x=q_df["end"],
                y=q_df["value"],
                mode="lines+markers",
                name="Quarterly (10-Q)",
                line=dict(color="#2563eb", width=2.5),
            )
        )

    if not a_df.empty:
        fig.add_trace(
            go.Scatter(
                x=a_df["end"],
                y=a_df["value"],
                mode="lines+markers",
                name="Annual (10-K)",
                line=dict(color="#0f766e", width=2.5, dash="dash"),
            )
        )

    if q_df.empty and a_df.empty:
        return _empty_figure(f"No liabilities data found for {ticker}")

    fig.update_layout(
        title=f"Total Liabilities Trend - {ticker}",
        template="plotly_white",
        yaxis_title="USD",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=30, r=30, t=60, b=100),
    )
    return fig


def _build_rolling_average_series(
    df: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    if df is None or df.empty or window <= 1:
        return pd.DataFrame(columns=["end", "rolling_value"])

    working = df.copy()
    working["end"] = pd.to_datetime(working["end"], errors="coerce")
    working["value"] = pd.to_numeric(working["value"], errors="coerce")
    working = working.dropna(subset=["end", "value"]).sort_values("end")
    if len(working) < window:
        return pd.DataFrame(columns=["end", "rolling_value"])

    working["rolling_value"] = working["value"].rolling(window=window).mean()
    working = working.dropna(subset=["rolling_value"])
    return working[["end", "rolling_value"]]


def _build_comparison_figure(
    liabilities_df: pd.DataFrame,
    revenue_df: pd.DataFrame,
    earnings_df: pd.DataFrame,
    operating_cash_flow_df: pd.DataFrame,
    ticker: str,
    period_label: str,
) -> go.Figure:
    fig = go.Figure()
    liab_style = SERIES_STYLES["liabilities"]
    rev_style = SERIES_STYLES["revenue"]
    earn_style = SERIES_STYLES["earnings"]
    ocf_style = SERIES_STYLES["operating_cash_flow"]
    period_label_lower = (period_label or "").lower()
    rolling_window = 4 if "quarter" in period_label_lower else 3
    rolling_label = (
        f"{rolling_window}-Q Rolling Avg Liabilities"
        if "quarter" in period_label_lower
        else f"{rolling_window}-Y Rolling Avg Liabilities"
    )

    if liabilities_df is not None and not liabilities_df.empty:
        fig.add_trace(
            go.Scatter(
                x=liabilities_df["end"],
                y=liabilities_df["value"],
                mode="lines+markers",
                name="Liabilities",
                line=dict(
                    color=liab_style["color"],
                    width=liab_style["width"],
                    dash=liab_style["dash"],
                ),
            )
        )
        rolling_df = _build_rolling_average_series(liabilities_df, rolling_window)
        if not rolling_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=rolling_df["end"],
                    y=rolling_df["rolling_value"],
                    mode="lines",
                    name=rolling_label,
                    line=dict(
                        color="#1d4ed8",
                        width=2,
                        dash="dash",
                    ),
                )
            )

    if revenue_df is not None and not revenue_df.empty:
        fig.add_trace(
            go.Scatter(
                x=revenue_df["end"],
                y=revenue_df["value"],
                mode="lines+markers",
                name="Revenue",
                line=dict(
                    color=rev_style["color"],
                    width=rev_style["width"],
                    dash=rev_style["dash"],
                ),
            )
        )

    if earnings_df is not None and not earnings_df.empty:
        fig.add_trace(
            go.Scatter(
                x=earnings_df["end"],
                y=earnings_df["value"],
                mode="lines+markers",
                name="Earnings",
                line=dict(
                    color=earn_style["color"],
                    width=earn_style["width"],
                    dash=earn_style["dash"],
                ),
            )
        )

    if operating_cash_flow_df is not None and not operating_cash_flow_df.empty:
        fig.add_trace(
            go.Scatter(
                x=operating_cash_flow_df["end"],
                y=operating_cash_flow_df["value"],
                mode="lines+markers",
                name="Operating Cash Flow",
                line=dict(
                    color=ocf_style["color"],
                    width=ocf_style["width"],
                    dash=ocf_style["dash"],
                ),
            )
        )

    if (
        (liabilities_df is None or liabilities_df.empty)
        and (revenue_df is None or revenue_df.empty)
        and (earnings_df is None or earnings_df.empty)
        and (operating_cash_flow_df is None or operating_cash_flow_df.empty)
    ):
        return _empty_figure(
            f"No {period_label} liabilities/revenue/earnings/operating cash flow data for {ticker}"
        )

    fig.update_layout(
        title=f"{ticker} {period_label} Liabilities vs Revenue vs Earnings vs Operating Cash Flow",
        template="plotly_white",
        yaxis=dict(title="USD", side="right"),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=30, r=30, t=60, b=100),
    )
    return fig


def _build_quarterly_table_data(q_total: pd.DataFrame) -> list[dict]:
    if q_total is None or q_total.empty:
        return []

    table_df = q_total.copy()
    table_df["end"] = pd.to_datetime(table_df["end"], errors="coerce")
    table_df["value"] = pd.to_numeric(table_df["value"], errors="coerce")
    table_df = table_df.dropna(subset=["end", "value"]).sort_values("end")
    table_df["qoq_change"] = table_df["value"].pct_change()
    table_df = table_df.sort_values("end", ascending=False)
    table_df["Period End"] = pd.to_datetime(
        table_df["end"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    table_df["Total Liabilities"] = table_df["value"].apply(_fmt_money)
    table_df["QoQ % Increase"] = table_df["qoq_change"].apply(_fmt_pct)
    table_df["Form"] = table_df.get("form", "")
    table_df["Filed"] = pd.to_datetime(
        table_df.get("filed"), errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    table_df["Filed"] = table_df["Filed"].fillna("")

    return table_df[
        ["Period End", "Total Liabilities", "QoQ % Increase", "Form", "Filed"]
    ].to_dict("records")


def _build_annual_table_data(a_total: pd.DataFrame) -> list[dict]:
    if a_total is None or a_total.empty:
        return []

    table_df = a_total.copy()
    table_df["end"] = pd.to_datetime(table_df["end"], errors="coerce")
    table_df["value"] = pd.to_numeric(table_df["value"], errors="coerce")
    table_df = table_df.dropna(subset=["end", "value"]).sort_values("end")
    table_df["yoy_change"] = table_df["value"].pct_change()
    table_df = table_df.sort_values("end", ascending=False)
    table_df["Year"] = pd.to_datetime(table_df["end"], errors="coerce").dt.year
    table_df["Period End"] = pd.to_datetime(
        table_df["end"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    table_df["Total Liabilities"] = table_df["value"].apply(_fmt_money)
    table_df["YoY % Increase"] = table_df["yoy_change"].apply(_fmt_pct)
    table_df["Form"] = table_df.get("form", "")
    table_df["Filed"] = pd.to_datetime(
        table_df.get("filed"), errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    table_df["Filed"] = table_df["Filed"].fillna("")

    return table_df[
        ["Year", "Period End", "Total Liabilities", "YoY % Increase", "Form", "Filed"]
    ].to_dict("records")


def _filter_series_by_period(df: pd.DataFrame, period_value: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["end", "value", "form", "filed"])

    period_value = (period_value or "max").lower()
    if period_value == "max":
        return df.copy()

    years_map = {"1y": 1, "2y": 2, "3y": 3, "4y": 4, "5y": 5, "10y": 10}
    years = years_map.get(period_value)
    if not years:
        return df.copy()

    working = df.copy()
    working["end"] = pd.to_datetime(working["end"], errors="coerce")
    working = working.dropna(subset=["end"])
    if working.empty:
        return working

    cutoff = working["end"].max() - pd.DateOffset(years=years)
    return working[working["end"] >= cutoff].copy()


def build_layout():
    return dbc.Container(
        [
            dcc.Store(id="liab-series-store"),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Input(
                            id="liab-ticker-input",
                            type="text",
                            placeholder="Enter ticker (e.g., MSFT)",
                            value="",
                        ),
                        md=4,
                    ),
                    dbc.Col(
                        dbc.Button(
                            "EDGAR Pull",
                            id="liab-pull-live-btn",
                            color="primary",
                            n_clicks=0,
                            className="w-100",
                        ),
                        md=2,
                    ),
                    dbc.Col(
                        dbc.Button(
                            "DB Pull",
                            id="liab-pull-db-btn",
                            color="secondary",
                            outline=True,
                            n_clicks=0,
                            className="w-100",
                        ),
                        md=2,
                    ),
                    dbc.Col(
                        html.Div(
                            id="liab-status-msg",
                            className="text-muted small",
                        ),
                        md=6,
                    ),
                ],
                className="mb-3 align-items-center",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H6("Latest Total Liabilities"),
                                    html.H4("-", id="liab-latest-total"),
                                ]
                            ),
                            className="shadow-sm",
                        ),
                        md=4,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [html.H6("QoQ Growth"), html.H4("-", id="liab-qoq")]
                            ),
                            className="shadow-sm",
                        ),
                        md=4,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [html.H6("YoY Growth"), html.H4("-", id="liab-yoy")]
                            ),
                            className="shadow-sm",
                        ),
                        md=4,
                    ),
                ],
                className="mb-3",
            ),
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                dbc.RadioItems(
                                    id="liab-q-period-radio",
                                    options=[
                                        {"label": "1Y", "value": "1y"},
                                        {"label": "2Y", "value": "2y"},
                                        {"label": "3Y", "value": "3y"},
                                        {"label": "4Y", "value": "4y"},
                                        {"label": "5Y", "value": "5y"},
                                        {"label": "10Y", "value": "10y"},
                                        {"label": "All", "value": "max"},
                                    ],
                                    value="max",
                                    inline=True,
                                    className="mb-2",
                                ),
                                dcc.Graph(
                                    id="liab-quarterly-chart",
                                    figure=_empty_figure(
                                        "Enter a ticker and click EDGAR Pull or DB Pull"
                                    ),
                                ),
                            ]
                        ),
                        className="shadow-sm",
                    ),
                    width=12,
                ),
                className="mb-3",
            ),
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5(
                                    "Quarterly 10-Q Liabilities",
                                    className="mb-3",
                                ),
                                dash_table.DataTable(
                                    id="liab-quarterly-table",
                                    columns=[
                                        {"name": "Period End", "id": "Period End"},
                                        {
                                            "name": "Total Liabilities",
                                            "id": "Total Liabilities",
                                        },
                                        {
                                            "name": "QoQ % Increase",
                                            "id": "QoQ % Increase",
                                        },
                                        {"name": "Form", "id": "Form"},
                                        {"name": "Filed", "id": "Filed"},
                                    ],
                                    data=[],
                                    sort_action="native",
                                    page_size=12,
                                    style_table={"overflowX": "auto"},
                                    style_cell={
                                        "fontFamily": "Segoe UI, Arial, sans-serif",
                                        "fontSize": "0.9rem",
                                        "padding": "0.45rem",
                                        "textAlign": "left",
                                    },
                                    style_header={
                                        "fontWeight": "600",
                                        "backgroundColor": "#f8fafc",
                                    },
                                ),
                            ]
                        ),
                        className="shadow-sm",
                    ),
                    width=12,
                ),
                className="mb-3",
            ),
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                dbc.RadioItems(
                                    id="liab-a-period-radio",
                                    options=[
                                        {"label": "1Y", "value": "1y"},
                                        {"label": "2Y", "value": "2y"},
                                        {"label": "3Y", "value": "3y"},
                                        {"label": "4Y", "value": "4y"},
                                        {"label": "5Y", "value": "5y"},
                                        {"label": "10Y", "value": "10y"},
                                        {"label": "All", "value": "max"},
                                    ],
                                    value="max",
                                    inline=True,
                                    className="mb-2",
                                ),
                                dcc.Graph(
                                    id="liab-annual-chart",
                                    figure=_empty_figure(
                                        "Annual (10-K) liabilities vs revenue vs earnings"
                                    ),
                                ),
                            ]
                        ),
                        className="shadow-sm",
                    ),
                    width=12,
                ),
                className="mb-3",
            ),
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5(
                                    "Yearly 10-K Liabilities",
                                    className="mb-3",
                                ),
                                dash_table.DataTable(
                                    id="liab-annual-table",
                                    columns=[
                                        {"name": "Year", "id": "Year"},
                                        {"name": "Period End", "id": "Period End"},
                                        {
                                            "name": "Total Liabilities",
                                            "id": "Total Liabilities",
                                        },
                                        {
                                            "name": "YoY % Increase",
                                            "id": "YoY % Increase",
                                        },
                                        {"name": "Form", "id": "Form"},
                                        {"name": "Filed", "id": "Filed"},
                                    ],
                                    data=[],
                                    sort_action="native",
                                    page_size=10,
                                    style_table={"overflowX": "auto"},
                                    style_cell={
                                        "fontFamily": "Segoe UI, Arial, sans-serif",
                                        "fontSize": "0.9rem",
                                        "padding": "0.45rem",
                                        "textAlign": "left",
                                    },
                                    style_header={
                                        "fontWeight": "600",
                                        "backgroundColor": "#f8fafc",
                                    },
                                ),
                            ]
                        ),
                        className="shadow-sm",
                    ),
                    width=12,
                ),
                className="mb-3",
            ),
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            dcc.Graph(
                                id="liab-component-chart",
                                figure=_empty_figure(
                                    "Liability component breakdown requires a pull"
                                ),
                            )
                        ),
                        className="shadow-sm",
                    ),
                    width=12,
                ),
                className="mb-3",
            ),
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Liability Component Detail", className="mb-3"),
                                dash_table.DataTable(
                                    id="liab-component-table",
                                    columns=[
                                        {
                                            "name": "Liability Component",
                                            "id": "Liability Component",
                                        },
                                        {"name": "Value", "id": "Value"},
                                        {
                                            "name": "Share of Total",
                                            "id": "Share of Total",
                                        },
                                        {
                                            "name": "Qtr % Change",
                                            "id": "Qtr % Change",
                                        },
                                        {
                                            "name": "Year % Change",
                                            "id": "Year % Change",
                                        },
                                        {"name": "Period End", "id": "Period End"},
                                        {"name": "Form", "id": "Form"},
                                        {"name": "Filed", "id": "Filed"},
                                        {"name": "Tag", "id": "Tag"},
                                    ],
                                    data=[],
                                    sort_action="native",
                                    page_size=18,
                                    style_table={"overflowX": "auto"},
                                    style_cell={
                                        "fontFamily": "Segoe UI, Arial, sans-serif",
                                        "fontSize": "0.9rem",
                                        "padding": "0.45rem",
                                        "textAlign": "left",
                                    },
                                    style_header={
                                        "fontWeight": "600",
                                        "backgroundColor": "#f8fafc",
                                    },
                                ),
                            ]
                        ),
                        className="shadow-sm",
                    ),
                    width=12,
                )
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H6("Interest-bearing Debt"),
                                    html.H4("-", id="liab-debt-balance"),
                                ]
                            ),
                            className="shadow-sm",
                        ),
                        md=3,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H6("Accrued Interest Liability"),
                                    html.H4("-", id="liab-accrued-interest"),
                                ]
                            ),
                            className="shadow-sm",
                        ),
                        md=3,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H6("Implied Interest Rate"),
                                    html.H4("-", id="liab-implied-rate"),
                                ]
                            ),
                            className="shadow-sm",
                        ),
                        md=3,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H6("Earnings / Avg Debt"),
                                    html.H4("-", id="liab-return-on-debt"),
                                ]
                            ),
                            className="shadow-sm",
                        ),
                        md=3,
                    ),
                ],
                className="mb-3",
            ),
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Switch(
                                                id="liab-stress-toggle",
                                                label="Refinance Stress Mode",
                                                value=False,
                                            ),
                                            md=4,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    "Stress spread (bps)",
                                                    className="small text-muted",
                                                ),
                                                dcc.Slider(
                                                    id="liab-stress-bps-slider",
                                                    min=0,
                                                    max=600,
                                                    step=25,
                                                    value=100,
                                                    marks={
                                                        0: "0",
                                                        100: "100",
                                                        200: "200",
                                                        300: "300",
                                                        400: "400",
                                                        500: "500",
                                                        600: "600",
                                                    },
                                                ),
                                            ],
                                            md=8,
                                        ),
                                    ],
                                    className="mb-2",
                                ),
                                dcc.Graph(
                                    id="liab-debt-schedule-chart",
                                    figure=_empty_figure(
                                        "Debt maturity schedule requires a live pull"
                                    ),
                                ),
                                html.Div(
                                    id="liab-stress-impact-msg",
                                    className="small text-muted",
                                ),
                            ]
                        ),
                        className="shadow-sm",
                    ),
                    width=12,
                ),
                className="mb-3",
            ),
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Debt and Interest Detail", className="mb-3"),
                                dash_table.DataTable(
                                    id="liab-debt-detail-table",
                                    columns=[
                                        {"name": "Category", "id": "Category"},
                                        {"name": "Metric", "id": "Metric"},
                                        {"name": "Value", "id": "Value"},
                                        {"name": "Period End", "id": "Period End"},
                                        {"name": "Form", "id": "Form"},
                                        {"name": "Source Tag", "id": "Source Tag"},
                                    ],
                                    data=[],
                                    sort_action="native",
                                    page_size=12,
                                    style_table={"overflowX": "auto"},
                                    style_cell={
                                        "fontFamily": "Segoe UI, Arial, sans-serif",
                                        "fontSize": "0.9rem",
                                        "padding": "0.45rem",
                                        "textAlign": "left",
                                    },
                                    style_header={
                                        "fontWeight": "600",
                                        "backgroundColor": "#f8fafc",
                                    },
                                ),
                            ]
                        ),
                        className="shadow-sm",
                    ),
                    width=12,
                ),
                className="mb-3",
            ),
        ],
        fluid=True,
        className="py-3",
    )


layout = build_layout()


def register_callbacks(app):
    @app.callback(
        Output("liab-series-store", "data"),
        Output("liab-quarterly-chart", "figure"),
        Output("liab-annual-chart", "figure"),
        Output("liab-component-chart", "figure"),
        Output("liab-debt-schedule-chart", "figure"),
        Output("liab-latest-total", "children"),
        Output("liab-debt-balance", "children"),
        Output("liab-accrued-interest", "children"),
        Output("liab-implied-rate", "children"),
        Output("liab-return-on-debt", "children"),
        Output("liab-qoq", "children"),
        Output("liab-yoy", "children"),
        Output("liab-quarterly-table", "data"),
        Output("liab-annual-table", "data"),
        Output("liab-component-table", "data"),
        Output("liab-debt-detail-table", "data"),
        Output("liab-status-msg", "children"),
        Input("liab-pull-live-btn", "n_clicks"),
        Input("liab-pull-db-btn", "n_clicks"),
        State("liab-ticker-input", "value"),
        prevent_initial_call=True,
    )
    def pull_liabilities(_live_n_clicks, _db_n_clicks, ticker_value):
        ticker = (ticker_value or "").strip().upper()
        if not ticker:
            return (
                None,
                _empty_figure("No ticker provided"),
                _empty_figure("No ticker provided"),
                _empty_figure("No ticker provided"),
                _empty_figure("No ticker provided"),
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                [],
                [],
                [],
                [],
                "Please enter a ticker.",
            )

        try:
            triggered = ctx.triggered_id
            use_db = triggered == "liab-pull-db-btn"

            if use_db:
                q_total = _fetch_cached_liabilities_series(ticker, "total_quarterly")
                a_total = _fetch_cached_liabilities_series(ticker, "total_annual")
                q_curr = _fetch_cached_liabilities_series(ticker, "current_quarterly")
                q_noncurr = _fetch_cached_liabilities_series(
                    ticker, "noncurrent_quarterly"
                )
                q_revenue = _fetch_cached_liabilities_series(
                    ticker, "revenue_quarterly"
                )
                a_revenue = _fetch_cached_liabilities_series(ticker, "revenue_annual")
                q_earnings = _fetch_cached_liabilities_series(
                    ticker, "earnings_quarterly"
                )
                a_earnings = _fetch_cached_liabilities_series(ticker, "earnings_annual")
                q_operating_cash_flow = _fetch_cached_liabilities_series(
                    ticker,
                    "operating_cash_flow_quarterly",
                )
                a_operating_cash_flow = _fetch_cached_liabilities_series(
                    ticker,
                    "operating_cash_flow_annual",
                )
                if q_total.empty and a_total.empty:
                    return (
                        None,
                        _empty_figure(f"No cached liabilities data for {ticker}"),
                        _empty_figure(f"No cached liabilities data for {ticker}"),
                        _empty_figure(f"No cached liabilities data for {ticker}"),
                        _empty_figure(f"No cached liabilities data for {ticker}"),
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        [],
                        [],
                        [],
                        [],
                        f"No cached data found for {ticker}. Run EDGAR Pull first.",
                    )
                persist_result = {"rows_attempted": 0}
                status_source = "SQLite/Parquet cache"
                component_history_df = _fetch_cached_liability_component_history(ticker)
                component_df = _select_latest_segment_rows(component_history_df)
                if component_df.empty:
                    component_note = "no cached segment detail"
                else:
                    component_note = "cached selected liability segments"
                debt_schedule_df = _fetch_cached_debt_schedule(ticker)
                debt_detail_rows = _fetch_cached_debt_detail(ticker)
                debt_snapshot = _fetch_cached_debt_snapshot(ticker)
                debt_balance_display = debt_snapshot.get("debt_balance_display", "-")
                accrued_interest_display = debt_snapshot.get(
                    "accrued_interest_display", "-"
                )
                implied_rate_display = debt_snapshot.get("implied_rate_display", "-")
                return_on_debt_display = debt_snapshot.get(
                    "return_on_debt_display", "-"
                )
                implied_rate = debt_snapshot.get("implied_rate_raw")
                if (
                    debt_schedule_df.empty
                    and not debt_detail_rows
                    and not debt_snapshot
                ):
                    debt_note = "debt interest detail unavailable from cache"
                else:
                    debt_note = "cached debt, interest, and maturity analytics"
            else:
                cik = _get_cik(ticker)
                if not cik:
                    return (
                        None,
                        _empty_figure(f"Ticker not found: {ticker}"),
                        _empty_figure(f"Ticker not found: {ticker}"),
                        _empty_figure(f"Ticker not found: {ticker}"),
                        _empty_figure(f"Ticker not found: {ticker}"),
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        [],
                        [],
                        [],
                        [],
                        f"Ticker not found in SEC list: {ticker}",
                    )

                facts = _get_company_facts(cik)

                q_total = _extract_series_for_forms(
                    facts, TOTAL_LIABILITY_TAGS, {"10-Q"}
                ).sort_values("end")
                a_total = _extract_series_for_forms(
                    facts, TOTAL_LIABILITY_TAGS, {"10-K"}
                ).sort_values("end")
                if q_total.empty:
                    q_total = _derive_total_liabilities_series(facts, {"10-Q"})
                if a_total.empty:
                    a_total = _derive_total_liabilities_series(facts, {"10-K"})

                q_curr = _extract_series_for_forms(
                    facts, CURRENT_LIABILITY_TAGS, {"10-Q"}
                ).sort_values("end")
                q_noncurr = _extract_series_for_forms(
                    facts, NONCURRENT_LIABILITY_TAGS, {"10-Q"}
                ).sort_values("end")
                q_revenue = _extract_series_for_forms(
                    facts, REVENUE_TAGS, {"10-Q"}
                ).sort_values("end")
                a_revenue = _extract_series_for_forms(
                    facts, REVENUE_TAGS, {"10-K"}
                ).sort_values("end")
                q_earnings = _extract_series_for_forms(
                    facts, EARNINGS_TAGS, {"10-Q"}
                ).sort_values("end")
                a_earnings = _extract_series_for_forms(
                    facts, EARNINGS_TAGS, {"10-K"}
                ).sort_values("end")
                q_operating_cash_flow = _extract_series_for_forms(
                    facts, OPERATING_CASH_FLOW_TAGS, {"10-Q"}
                ).sort_values("end")
                a_operating_cash_flow = _extract_series_for_forms(
                    facts, OPERATING_CASH_FLOW_TAGS, {"10-K"}
                ).sort_values("end")

                persist_result = _persist_liabilities(
                    ticker=ticker,
                    q_total=q_total,
                    a_total=a_total,
                    q_curr=q_curr,
                    q_noncurr=q_noncurr,
                    q_revenue=q_revenue,
                    a_revenue=a_revenue,
                    q_earnings=q_earnings,
                    a_earnings=a_earnings,
                    q_operating_cash_flow=q_operating_cash_flow,
                    a_operating_cash_flow=a_operating_cash_flow,
                )
                component_forms: set[str] | None = None
                component_current_total = q_curr
                component_total_df = q_total if not q_total.empty else a_total
                if not q_total.empty:
                    component_forms = {"10-Q"}
                elif not a_total.empty:
                    component_forms = {"10-K"}
                    component_current_total = _extract_series_for_forms(
                        facts, CURRENT_LIABILITY_TAGS, {"10-K"}
                    ).sort_values("end")

                component_history_df = _extract_selected_liability_segment_history(
                    facts,
                    forms=component_forms,
                    q_curr=component_current_total,
                    q_total=component_total_df,
                )
                component_df = _select_latest_segment_rows(component_history_df)
                if component_df.empty:
                    component_note = "no selected liability segments found"
                    component_source = "fallback_split"
                else:
                    component_note = "selected liability segments"
                    component_source = "edgar_components"

                persist_result.update(
                    _persist_liability_components(
                        ticker=ticker,
                        component_df=component_df,
                        component_source=component_source,
                    )
                )
                status_source = "live EDGAR"

                q_debt = _extract_debt_series(facts, {"10-Q"}).sort_values("end")
                a_debt = _extract_debt_series(facts, {"10-K"}).sort_values("end")

                debt_target_end = None
                if not q_debt.empty:
                    debt_target_end = pd.to_datetime(q_debt["end"].max())
                elif not a_debt.empty:
                    debt_target_end = pd.to_datetime(a_debt["end"].max())
                elif not q_total.empty:
                    debt_target_end = pd.to_datetime(q_total["end"].max())
                elif not a_total.empty:
                    debt_target_end = pd.to_datetime(a_total["end"].max())

                annual_target_end = (
                    pd.to_datetime(a_total["end"].max())
                    if not a_total.empty
                    else debt_target_end
                )
                accrued_interest_row = _extract_latest_metric_by_tags(
                    facts,
                    INTEREST_PAYABLE_TAGS,
                    forms={"10-Q", "10-K"},
                    target_end=debt_target_end,
                )
                interest_expense_annual = _extract_series_for_forms(
                    facts, INTEREST_EXPENSE_TAGS, {"10-K"}
                ).sort_values("end")
                debt_schedule_df = _extract_debt_maturity_schedule(
                    facts,
                    target_end=annual_target_end,
                )

                latest_debt_value = None
                latest_debt_end = None
                latest_debt_form = ""
                latest_debt_tag = ""
                debt_source_df = q_debt if not q_debt.empty else a_debt
                if debt_source_df is not None and not debt_source_df.empty:
                    debt_source_df = debt_source_df.sort_values("end")
                    latest_debt_value = float(debt_source_df["value"].iloc[-1])
                    latest_debt_end = pd.to_datetime(
                        debt_source_df["end"].iloc[-1], errors="coerce"
                    )
                    latest_debt_form = (
                        debt_source_df.get("form", pd.Series(dtype=str)).iloc[-1]
                        if "form" in debt_source_df.columns
                        else ""
                    )
                    latest_debt_tag = "/".join(DEBT_TOTAL_TAGS)
                debt_balance_display = _fmt_money(latest_debt_value)

                accrued_interest_value = None
                accrued_interest_display = "-"
                if not accrued_interest_row.empty:
                    accrued_interest_value = float(
                        accrued_interest_row["value"].iloc[-1]
                    )
                    accrued_interest_display = _fmt_money(accrued_interest_value)

                implied_rate = None
                annual_interest_expense_value = None
                annual_interest_end = None
                if not interest_expense_annual.empty:
                    interest_expense_annual = interest_expense_annual.sort_values("end")
                    annual_interest_expense_value = abs(
                        float(interest_expense_annual["value"].iloc[-1])
                    )
                    annual_interest_end = pd.to_datetime(
                        interest_expense_annual["end"].iloc[-1], errors="coerce"
                    )
                average_debt_value = None
                if not a_debt.empty:
                    annual_debt = a_debt.sort_values("end").tail(2)
                    if len(annual_debt) == 2:
                        average_debt_value = float(annual_debt["value"].mean())
                    elif len(annual_debt) == 1:
                        average_debt_value = float(annual_debt["value"].iloc[-1])
                elif latest_debt_value is not None:
                    average_debt_value = latest_debt_value
                if annual_interest_expense_value is not None and average_debt_value:
                    implied_rate = annual_interest_expense_value / average_debt_value
                implied_rate_display = _fmt_pct(implied_rate)

                return_on_debt = None
                latest_earnings_value = None
                latest_earnings_end = None
                if not a_earnings.empty:
                    annual_earnings = a_earnings.sort_values("end")
                    latest_earnings_value = float(annual_earnings["value"].iloc[-1])
                    latest_earnings_end = pd.to_datetime(
                        annual_earnings["end"].iloc[-1], errors="coerce"
                    )
                if latest_earnings_value is not None and average_debt_value:
                    return_on_debt = latest_earnings_value / average_debt_value
                return_on_debt_display = _fmt_pct(return_on_debt)

                debt_detail_rows = []
                if latest_debt_value is not None:
                    debt_detail_rows.append(
                        {
                            "Category": "Debt Snapshot",
                            "Metric": "Interest-bearing debt outstanding",
                            "Value": _fmt_money(latest_debt_value),
                            "Period End": latest_debt_end.strftime("%Y-%m-%d")
                            if pd.notna(latest_debt_end)
                            else "",
                            "Form": latest_debt_form,
                            "Source Tag": latest_debt_tag,
                        }
                    )
                if accrued_interest_value is not None:
                    accrued_end = pd.to_datetime(
                        accrued_interest_row["end"].iloc[-1], errors="coerce"
                    )
                    debt_detail_rows.append(
                        {
                            "Category": "Debt Snapshot",
                            "Metric": "Accrued interest liability",
                            "Value": _fmt_money(accrued_interest_value),
                            "Period End": accrued_end.strftime("%Y-%m-%d")
                            if pd.notna(accrued_end)
                            else "",
                            "Form": accrued_interest_row["form"].iloc[-1]
                            if "form" in accrued_interest_row.columns
                            else "",
                            "Source Tag": accrued_interest_row["tag"].iloc[-1],
                        }
                    )
                if annual_interest_expense_value is not None:
                    debt_detail_rows.append(
                        {
                            "Category": "Interest",
                            "Metric": "Annual interest expense",
                            "Value": _fmt_money(annual_interest_expense_value),
                            "Period End": annual_interest_end.strftime("%Y-%m-%d")
                            if pd.notna(annual_interest_end)
                            else "",
                            "Form": interest_expense_annual["form"].iloc[-1]
                            if "form" in interest_expense_annual.columns
                            else "",
                            "Source Tag": "/".join(INTEREST_EXPENSE_TAGS),
                        }
                    )
                if implied_rate is not None:
                    debt_detail_rows.append(
                        {
                            "Category": "Interest",
                            "Metric": "Implied interest rate (interest expense / avg debt)",
                            "Value": _fmt_pct(implied_rate),
                            "Period End": annual_interest_end.strftime("%Y-%m-%d")
                            if pd.notna(annual_interest_end)
                            else "",
                            "Form": "10-K",
                            "Source Tag": "Derived from interest expense and debt",
                        }
                    )
                if return_on_debt is not None:
                    debt_detail_rows.append(
                        {
                            "Category": "Return Proxy",
                            "Metric": "Earnings / average debt",
                            "Value": _fmt_pct(return_on_debt),
                            "Period End": latest_earnings_end.strftime("%Y-%m-%d")
                            if pd.notna(latest_earnings_end)
                            else "",
                            "Form": "10-K",
                            "Source Tag": "Derived from earnings and debt",
                        }
                    )

                debt_note = "live debt, interest, and maturity analytics"
                persist_result.update(
                    _persist_debt_analytics(
                        ticker=ticker,
                        debt_balance_display=debt_balance_display,
                        accrued_interest_display=accrued_interest_display,
                        implied_rate_display=implied_rate_display,
                        return_on_debt_display=return_on_debt_display,
                        implied_rate_raw=implied_rate,
                        debt_schedule_df=debt_schedule_df,
                        debt_detail_rows=debt_detail_rows,
                    )
                )

            series_store = {
                "ticker": ticker,
                "q_total": q_total.assign(
                    end=q_total["end"].dt.strftime("%Y-%m-%d")
                    if not q_total.empty
                    else q_total.get("end", pd.Series(dtype=str))
                ).to_dict("records"),
                "a_total": a_total.assign(
                    end=a_total["end"].dt.strftime("%Y-%m-%d")
                    if not a_total.empty
                    else a_total.get("end", pd.Series(dtype=str))
                ).to_dict("records"),
                "q_revenue": q_revenue.assign(
                    end=q_revenue["end"].dt.strftime("%Y-%m-%d")
                    if not q_revenue.empty
                    else q_revenue.get("end", pd.Series(dtype=str))
                ).to_dict("records"),
                "a_revenue": a_revenue.assign(
                    end=a_revenue["end"].dt.strftime("%Y-%m-%d")
                    if not a_revenue.empty
                    else a_revenue.get("end", pd.Series(dtype=str))
                ).to_dict("records"),
                "q_earnings": q_earnings.assign(
                    end=q_earnings["end"].dt.strftime("%Y-%m-%d")
                    if not q_earnings.empty
                    else q_earnings.get("end", pd.Series(dtype=str))
                ).to_dict("records"),
                "a_earnings": a_earnings.assign(
                    end=a_earnings["end"].dt.strftime("%Y-%m-%d")
                    if not a_earnings.empty
                    else a_earnings.get("end", pd.Series(dtype=str))
                ).to_dict("records"),
                "q_operating_cash_flow": q_operating_cash_flow.assign(
                    end=q_operating_cash_flow["end"].dt.strftime("%Y-%m-%d")
                    if not q_operating_cash_flow.empty
                    else q_operating_cash_flow.get("end", pd.Series(dtype=str))
                ).to_dict("records"),
                "a_operating_cash_flow": a_operating_cash_flow.assign(
                    end=a_operating_cash_flow["end"].dt.strftime("%Y-%m-%d")
                    if not a_operating_cash_flow.empty
                    else a_operating_cash_flow.get("end", pd.Series(dtype=str))
                ).to_dict("records"),
                "debt_schedule": debt_schedule_df.assign(
                    end=pd.to_datetime(
                        debt_schedule_df["end"], errors="coerce"
                    ).dt.strftime("%Y-%m-%d")
                ).to_dict("records")
                if debt_schedule_df is not None and not debt_schedule_df.empty
                else [],
                "implied_rate_raw": float(implied_rate)
                if implied_rate is not None
                else None,
            }

            quarterly_chart = _build_comparison_figure(
                liabilities_df=q_total,
                revenue_df=q_revenue,
                earnings_df=q_earnings,
                operating_cash_flow_df=q_operating_cash_flow,
                ticker=ticker,
                period_label="Quarterly 10-Q",
            )
            annual_chart = _build_comparison_figure(
                liabilities_df=a_total,
                revenue_df=a_revenue,
                earnings_df=a_earnings,
                operating_cash_flow_df=a_operating_cash_flow,
                ticker=ticker,
                period_label="Annual 10-K",
            )

            quarterly_table_data = _build_quarterly_table_data(q_total)
            annual_table_data = _build_annual_table_data(a_total)

            latest_total = "-"
            latest_total_value = None
            qoq = None
            yoy = None
            if not q_total.empty:
                q_total = q_total.sort_values("end").tail(20).reset_index(drop=True)
                latest_total_value = float(q_total["value"].iloc[-1])
                latest_total = _fmt_money(latest_total_value)
                if len(q_total) >= 2:
                    prev = q_total["value"].iloc[-2]
                    curr = q_total["value"].iloc[-1]
                    if prev and pd.notna(prev):
                        qoq = (curr / prev) - 1
                if len(q_total) >= 5:
                    prev_y = q_total["value"].iloc[-5]
                    curr = q_total["value"].iloc[-1]
                    if prev_y and pd.notna(prev_y):
                        yoy = (curr / prev_y) - 1
            elif not a_total.empty:
                latest_total_value = float(a_total.sort_values("end")["value"].iloc[-1])

            component_fig = _build_component_figure(
                component_df=component_df,
                ticker=ticker,
                total_value=latest_total_value,
            )
            component_table_data = _build_component_table_data(
                component_df=component_df,
                total_value=latest_total_value,
                component_history_df=component_history_df,
            )
            debt_schedule_fig = _build_debt_schedule_figure(debt_schedule_df, ticker)
            debt_detail_table_data = _build_debt_detail_table_data(
                debt_detail_rows,
                debt_schedule_df,
            )

            status = (
                f"Loaded liabilities for {ticker} from {status_source} | "
                f"Quarterly points: {len(q_total)} | Annual points: {len(a_total)} | "
                f"Persisted rows: {persist_result.get('rows_attempted', 0)} | "
                f"Component rows: {persist_result.get('component_rows_attempted', 0)} | "
                f"Component breakdown: {component_note} | "
                f"Debt analytics: {debt_note}"
            )

            return (
                series_store,
                quarterly_chart,
                annual_chart,
                component_fig,
                debt_schedule_fig,
                latest_total,
                debt_balance_display,
                accrued_interest_display,
                implied_rate_display,
                return_on_debt_display,
                _fmt_pct(qoq),
                _fmt_pct(yoy),
                quarterly_table_data,
                annual_table_data,
                component_table_data,
                debt_detail_table_data,
                status,
            )
        except Exception as exc:
            return (
                None,
                _empty_figure(f"Unable to load liabilities for {ticker}"),
                _empty_figure(f"Unable to load liabilities for {ticker}"),
                _empty_figure(f"Unable to load liabilities for {ticker}"),
                _empty_figure(f"Unable to load liabilities for {ticker}"),
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                [],
                [],
                [],
                [],
                f"Error: {exc}",
            )

    @app.callback(
        Output("liab-debt-schedule-chart", "figure", allow_duplicate=True),
        Output("liab-stress-impact-msg", "children"),
        Input("liab-series-store", "data"),
        Input("liab-stress-toggle", "value"),
        Input("liab-stress-bps-slider", "value"),
        prevent_initial_call=True,
    )
    def update_debt_stress_chart(series_store, stress_enabled, stress_bps):
        if not series_store:
            return _empty_figure("Debt maturity schedule requires a pull"), ""

        ticker = series_store.get("ticker", "")
        schedule_df = pd.DataFrame(series_store.get("debt_schedule", []))
        if schedule_df.empty:
            return _empty_figure(f"No debt maturity schedule found for {ticker}"), ""

        if "end" in schedule_df.columns:
            schedule_df["end"] = pd.to_datetime(schedule_df["end"], errors="coerce")
        if "value" in schedule_df.columns:
            schedule_df["value"] = pd.to_numeric(schedule_df["value"], errors="coerce")
        schedule_df = schedule_df.dropna(subset=["value"])

        fig = _build_debt_schedule_figure(
            schedule_df=schedule_df,
            ticker=ticker,
            stress_enabled=bool(stress_enabled),
            stress_spread_bps=float(stress_bps or 0.0),
        )

        if not stress_enabled:
            return fig, "Stress mode is off."

        total_due = float(schedule_df["value"].sum()) if not schedule_df.empty else 0.0
        spread = max(float(stress_bps or 0.0), 0.0) / 10_000.0
        added_interest = total_due * spread
        implied_rate_raw = series_store.get("implied_rate_raw")
        if implied_rate_raw is not None and pd.notna(implied_rate_raw):
            stressed_rate = float(implied_rate_raw) + spread
            msg = (
                f"Stress at +{float(stress_bps or 0):,.0f} bps implies about "
                f"{_fmt_money(added_interest)} extra annual interest on scheduled maturities. "
                f"Implied rate moves from {_fmt_pct(implied_rate_raw)} to {_fmt_pct(stressed_rate)}."
            )
        else:
            msg = (
                f"Stress at +{float(stress_bps or 0):,.0f} bps implies about "
                f"{_fmt_money(added_interest)} extra annual interest on scheduled maturities."
            )
        return fig, msg

    @app.callback(
        Output("liab-quarterly-chart", "figure", allow_duplicate=True),
        Input("liab-series-store", "data"),
        Input("liab-q-period-radio", "value"),
        prevent_initial_call=True,
    )
    def render_quarterly_chart(series_store, period_value):
        if not series_store:
            return _empty_figure("Enter a ticker and click EDGAR Pull or DB Pull")

        ticker = series_store.get("ticker", "")
        q_total = pd.DataFrame(series_store.get("q_total", []))
        q_revenue = pd.DataFrame(series_store.get("q_revenue", []))
        q_earnings = pd.DataFrame(series_store.get("q_earnings", []))
        q_operating_cash_flow = pd.DataFrame(
            series_store.get("q_operating_cash_flow", [])
        )

        for df in [q_total, q_revenue, q_earnings, q_operating_cash_flow]:
            if not df.empty and "end" in df.columns:
                df["end"] = pd.to_datetime(df["end"], errors="coerce")

        return _build_comparison_figure(
            liabilities_df=_filter_series_by_period(q_total, period_value),
            revenue_df=_filter_series_by_period(q_revenue, period_value),
            earnings_df=_filter_series_by_period(q_earnings, period_value),
            operating_cash_flow_df=_filter_series_by_period(
                q_operating_cash_flow, period_value
            ),
            ticker=ticker,
            period_label="Quarterly 10-Q",
        )

    @app.callback(
        Output("liab-annual-chart", "figure", allow_duplicate=True),
        Input("liab-series-store", "data"),
        Input("liab-a-period-radio", "value"),
        prevent_initial_call=True,
    )
    def render_annual_chart(series_store, period_value):
        if not series_store:
            return _empty_figure("Annual (10-K) liabilities vs revenue vs earnings")

        ticker = series_store.get("ticker", "")
        a_total = pd.DataFrame(series_store.get("a_total", []))
        a_revenue = pd.DataFrame(series_store.get("a_revenue", []))
        a_earnings = pd.DataFrame(series_store.get("a_earnings", []))
        a_operating_cash_flow = pd.DataFrame(
            series_store.get("a_operating_cash_flow", [])
        )

        for df in [a_total, a_revenue, a_earnings, a_operating_cash_flow]:
            if not df.empty and "end" in df.columns:
                df["end"] = pd.to_datetime(df["end"], errors="coerce")

        return _build_comparison_figure(
            liabilities_df=_filter_series_by_period(a_total, period_value),
            revenue_df=_filter_series_by_period(a_revenue, period_value),
            earnings_df=_filter_series_by_period(a_earnings, period_value),
            operating_cash_flow_df=_filter_series_by_period(
                a_operating_cash_flow, period_value
            ),
            ticker=ticker,
            period_label="Annual 10-K",
        )


register_callbacks(get_app())
