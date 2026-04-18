from pathlib import Path


DATA_DIR = Path(__file__).with_name("data")
PARQUET_DIR = DATA_DIR / "parquet"
CENTRAL_SQLITE_PATH = DATA_DIR / "dash_storage.sqlite"

DATA_DIR.mkdir(parents=True, exist_ok=True)
PARQUET_DIR.mkdir(parents=True, exist_ok=True)


def parquet_path(name: str) -> Path:
    return PARQUET_DIR / f"{name}.parquet"
