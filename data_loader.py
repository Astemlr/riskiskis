"""
Functions
---------
load_full_data(fp: str, date_col='Дата', numeric_cols=None) -> pd.DataFrame
    Reads a CSV, parses the date column, converts numeric columns to float, sorts by date,
    re‑indexes to business‑day frequency, forward‑fills gaps and drops residual NaNs.

set_seed(seed: int = 42) -> np.random.Generator
    Fixes the global NumPy random seed (for reproducibility) and returns a Generator instance.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

__all__ = [
    "set_seed",
    "load_full_data",
]

# Reproducibility helpers

def set_seed(seed: int = 42) -> np.random.Generator:
    np.random.seed(seed)  # legacy RNG (affects many libs that rely on it)
    return np.random.default_rng(seed)  # modern, independent RNG

# Data loading / cleaning

def _clean_numeric(series: pd.Series) -> pd.Series:
    """Remove spaces/non‑breaking spaces as thousand‑separators and replace commas → dots.

    Returned series is float64 with *NaN* when conversion fails.
    """
    return (
        pd.to_numeric(
            series.astype(str)
            .str.replace(r"[ \u00A0]", "", regex=True)  # delete spaces / NBSP
            .str.replace(",", "."),
            errors="coerce",
        ).astype("float64")
    )


def load_full_data(
    fp: str | Path,
    *,
    date_col: str = "Дата",
    numeric_cols: Optional[List[str]] = None,
    dayfirst: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(fp)

    # Автоопределение колонки с датой, если 'Дата' отсутствует
    if date_col not in df.columns:
        if "Unnamed: 0" in df.columns:
            date_col = "Unnamed: 0"
        else:
            raise ValueError(f"Колонка даты '{date_col}' не найдена в CSV")

    df[date_col] = pd.to_datetime(df[date_col], dayfirst=dayfirst)

    if numeric_cols is None:
        numeric_cols = df.columns.drop(date_col).tolist()

    for col in numeric_cols:
        df[col] = _clean_numeric(df[col])

    df = (
        df.sort_values(date_col)
        .set_index(date_col)
        .asfreq("B")  # бизнес-сетка
        .ffill()
        .dropna()
    )
    return df


# Quick CLI test

if __name__ == "__main__":
    rng = set_seed(42)
    data_path = Path(__file__).with_name("full_data.csv")
    if data_path.exists():
        df = load_full_data(data_path, numeric_cols=["moex", "rts", "Евро", "Доллар США"])
        print(df.head())
        print("\nColumns → dtypes:\n", df.dtypes)
