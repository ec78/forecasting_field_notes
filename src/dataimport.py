"""Refresh the raw CSVs under data/raw/.

The checked-in CSVs are sufficient to run every notebook, so this script is
only needed to pull more recent history.

    python -m src.dataimport --dataset spy
    python -m src.dataimport --dataset spy --out spy_blind_2026-11-18.csv
    python -m src.dataimport --dataset electricity --balancing-authority CISO
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def _save(data, metadata, csv_filename, metadata_filename):
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(RAW_DATA_DIR / csv_filename)
    pd.Series(metadata).to_json(RAW_DATA_DIR / metadata_filename, indent=2)


def fetch_spy(csv_filename="spy.csv"):
    """Download SPY adjusted-close history from Yahoo Finance.

    ``csv_filename`` lets a refresh go to a new file instead of overwriting
    ``spy.csv``, which earlier posts' results depend on (Post 05 fetches its
    blind window this way). The metadata file is named after it.
    """
    import yfinance as yf

    data = yf.download("SPY", period="max", auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data[["Close"]].rename(columns={"Close": "y"}).dropna().copy()
    data.columns.name = None
    data.index = pd.to_datetime(data.index)
    data.index.name = "date"

    metadata = {
        "dataset": "SPY",
        "source": "Yahoo Finance",
        "ticker": "SPY",
        "frequency": "daily",
        "download_date": pd.Timestamp.now().isoformat(),
        "n_observations": len(data),
        "start_date": str(data.index.min().date()),
        "end_date": str(data.index.max().date()),
    }
    metadata_filename = csv_filename.removesuffix(".csv") + "_metadata.json"
    _save(data, metadata, csv_filename, metadata_filename)
    return data


def fetch_electricity(balancing_authority="CISO", start_year=2019, end_year=None):
    """Download hourly electricity demand for one EIA balancing authority."""
    load_dotenv()
    api_key = os.getenv("EIA_API_KEY")
    if api_key is None:
        raise ValueError("EIA_API_KEY not found. Add it to a .env file in the project root.")

    end_year = end_year or pd.Timestamp.now().year
    base_url = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
    page_length = 5000
    all_rows = []

    for year in range(start_year, end_year + 1):
        print(f"Downloading electricity data for {year}...")
        offset = 0
        while True:
            params = {
                "api_key": api_key,
                "frequency": "hourly",
                "data[0]": "value",
                "facets[respondent][]": balancing_authority,
                "facets[type][]": "D",
                "start": f"{year}-01-01",
                "end": f"{year}-12-31T23",
                "length": page_length,
                "offset": offset,
                "sort[0][column]": "period",
                "sort[0][direction]": "asc",
            }
            response = requests.get(base_url, params=params, timeout=60)
            response.raise_for_status()
            rows = response.json()["response"]["data"]
            if not rows:
                break
            all_rows.extend(rows)
            if len(rows) < page_length:
                break
            offset += page_length

    raw = pd.DataFrame(all_rows)
    if raw.empty:
        raise ValueError("No electricity data were returned for the requested parameters.")

    data = raw[["period", "value"]].rename(columns={"period": "date", "value": "y"})
    data["date"] = pd.to_datetime(data["date"], utc=True)
    data["y"] = pd.to_numeric(data["y"], errors="coerce")
    data = data.dropna().drop_duplicates(subset="date").sort_values("date").set_index("date")
    data.index.name = "date"

    metadata = {
        "dataset": "EIA hourly electricity demand",
        "source": "U.S. Energy Information Administration",
        "balancing_authority": balancing_authority,
        "series_type": "demand",
        "frequency": "hourly",
        "start_year": start_year,
        "end_year": end_year,
        "download_date": pd.Timestamp.now().isoformat(),
        "n_observations": len(data),
        "start_date": str(data.index.min()),
        "end_date": str(data.index.max()),
    }
    ba = balancing_authority.lower()
    _save(data, metadata, f"electricity_{ba}.csv", f"electricity_{ba}_metadata.json")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["spy", "electricity"], required=True)
    parser.add_argument("--balancing-authority", default="CISO")
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--out", default=None,
                        help="SPY only: CSV filename under data/raw/ (default spy.csv).")
    args = parser.parse_args()

    if args.dataset == "spy":
        fetch_spy(args.out or "spy.csv")
    else:
        fetch_electricity(args.balancing_authority, args.start_year)
