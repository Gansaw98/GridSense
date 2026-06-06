"""
Utils: Data Loading & Preprocessing
- UCI Household Electric Power Consumption dataset loader
- Train/val/test splitting with no data leakage
- Synthetic data generator for testing without the real dataset
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def load_uci_dataset(filepath: str) -> pd.DataFrame:
    """
    Load and parse the UCI Household Electric Power Consumption dataset.
    
    Download from:
    https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption
    
    Args:
        filepath: Path to the raw .txt or .csv file
    
    Returns:
        DataFrame with DatetimeIndex and cleaned numeric columns
    """
    df = pd.read_csv(
        filepath,
        sep=";",
        na_values=["?"],
        parse_dates={"datetime": ["Date", "Time"]},
        dayfirst=True,
        infer_datetime_format=True,
    )
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)

    # Drop rows with missing target
    df.dropna(subset=["Global_active_power"], inplace=True)

    # Fill remaining NaNs with forward fill
    df.fillna(method="ffill", inplace=True)

    # Resample from 1-minute to 1-hour (mean)
    df = df.resample("1H").mean()

    return df


def generate_synthetic_dataset(
    n_hours: int = 8760,   # 1 year
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a realistic synthetic smart grid dataset for testing.
    
    Simulates:
      - Daily load curve (residential: peak morning + evening)
      - Weekly seasonality (lower weekend consumption)
      - Annual seasonality (higher winter + summer HVAC)
      - Random anomalies (~2% of windows)
      - Correlated temperature and occupancy signals
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n_hours, freq="H")

    hour = idx.hour.values
    dow = idx.dayofweek.values
    month = idx.month.values

    # Daily load curve: double hump (07:00 and 19:00)
    daily = (
        1.5 * np.exp(-0.5 * ((hour - 7) / 2.5) ** 2) +
        2.0 * np.exp(-0.5 * ((hour - 19) / 2.0) ** 2)
    )

    # Weekly effect: weekdays higher
    weekly = np.where(dow < 5, 1.0, 0.75)

    # Annual effect: winter + summer peaks
    annual = 1.0 + 0.3 * np.cos(2 * np.pi * (month - 1) / 12)

    # Temperature: correlated with load
    temp = 15 + 10 * np.cos(2 * np.pi * (month - 7) / 12)
    temp += rng.normal(0, 2, n_hours)

    # Occupancy: 1–4 people, higher evenings
    occupancy = (
        2 + np.round(np.clip(daily / daily.max() * 3, 0, 4))
    ).astype(float)
    occupancy += rng.normal(0, 0.3, n_hours)
    occupancy = np.clip(occupancy, 0, 5)

    # Compose load signal
    load = daily * weekly * annual
    load += 0.5 * (temp - 15) / 10   # temperature effect
    load += rng.normal(0, 0.15, n_hours)    # noise
    load = np.clip(load, 0.1, None)

    # Inject anomalies (~2%): sudden spikes or drops
    n_anomalies = int(0.02 * n_hours)
    anom_idx = rng.choice(n_hours, n_anomalies, replace=False)
    anom_type = rng.choice(["spike", "drop"], n_anomalies)
    for i, t in zip(anom_idx, anom_type):
        if t == "spike":
            load[i] *= rng.uniform(2.5, 5.0)
        else:
            load[i] *= rng.uniform(0.05, 0.2)

    df = pd.DataFrame({
        "Global_active_power": load,
        "temperature": temp,
        "occupancy": occupancy,
    }, index=idx)

    return df


def train_val_test_split(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> Tuple:
    """
    Chronological split — NO shuffling to prevent data leakage.
    
    Returns:
        (X_train, X_val, X_test) or
        (X_train, X_val, X_test, y_train, y_val, y_test) if y is given
    """
    n = len(X)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    n_train = n - n_val - n_test

    X_train = X[:n_train]
    X_val = X[n_train: n_train + n_val]
    X_test = X[n_train + n_val:]

    if y is None:
        return X_train, X_val, X_test

    y_train = y[:n_train]
    y_val = y[n_train: n_train + n_val]
    y_test = y[n_train + n_val:]
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = generate_synthetic_dataset(n_hours=2000)
    print(f"✅ Synthetic dataset: {df.shape}")
    print(df.head(3))
    print(f"\nLoad stats:\n{df['Global_active_power'].describe()}")
