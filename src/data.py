"""Loaders for the raw series stored under data/raw/."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def _load_csv(filename):
    """Load a date,y CSV from data/raw/ into a Series indexed by date."""
    filepath = RAW_DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n"
            "See data/README.md, or run src/dataimport.py to fetch it."
        )
    data = pd.read_csv(filepath, parse_dates=["date"], index_col="date")
    data.index.name = "date"
    return data["y"]


def load_electricity(balancing_authority="CISO"):
    """Hourly electricity demand (MWh) for one EIA balancing authority."""
    return _load_csv(f"electricity_{balancing_authority.lower()}.csv")


def load_spy(filename="spy.csv"):
    """Daily SPY adjusted close (``filename`` selects a refreshed copy, e.g. Post 05's blind file)."""
    return _load_csv(filename)


def train_val_test_split(series, val_size, test_size):
    """Split a series into contiguous train / validation / test segments.

    The split is chronological: train comes first, then validation, then
    test. Validation is meant for anything that needs to be tuned (hybrid
    weights, residual-memory scaling) before touching the test period.
    """
    n = len(series)
    if val_size + test_size >= n:
        raise ValueError("val_size + test_size must be smaller than len(series).")

    train_end = n - val_size - test_size
    val_end = n - test_size

    train = series.iloc[:train_end]
    val = series.iloc[train_end:val_end]
    test = series.iloc[val_end:]
    return train, val, test
