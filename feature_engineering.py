"""
Module 1: Feature Engineering
- Cyclical sine/cosine encoding for temporal features
- Exogenous event markers (holidays, special events)
- Dynamic pricing integration
- Lag & rolling statistical features
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Optional, List
import warnings
warnings.filterwarnings('ignore')


class CyclicalEncoder:
    """Encodes periodic features using sine/cosine transforms."""

    @staticmethod
    def encode(series: pd.Series, period: float) -> pd.DataFrame:
        """
        Encode a periodic feature into sine and cosine components.
        
        Args:
            series: Raw periodic values (e.g., hour 0–23)
            period: The full cycle length (e.g., 24 for hours)
        
        Returns:
            DataFrame with _sin and _cos columns
        """
        name = series.name or "feature"
        angle = 2 * np.pi * series / period
        return pd.DataFrame({
            f"{name}_sin": np.sin(angle),
            f"{name}_cos": np.cos(angle),
        }, index=series.index)


class SmartGridFeatureEngineer:
    """
    Full feature engineering pipeline for smart grid time series.
    
    Steps:
        1. Cyclical encoding (hour, day-of-week, month)
        2. Lag features (t-1, t-24, t-168)
        3. Rolling statistics (mean, std)
        4. Exogenous event markers
        5. Dynamic pricing signal
        6. StandardScaler normalization
    """

    def __init__(
        self,
        target_col: str = "Global_active_power",
        lag_hours: List[int] = [1, 2, 3, 6, 12, 24, 48, 168],
        rolling_windows: List[int] = [6, 12, 24, 48],
        holiday_dates: Optional[List[str]] = None,
    ):
        self.target_col = target_col
        self.lag_hours = lag_hours
        self.rolling_windows = rolling_windows
        self.holiday_dates = holiday_dates or []
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit scaler and return transformed feature matrix."""
        features = self._build_features(df)
        self.feature_names = features.columns.tolist()
        scaled = self.scaler.fit_transform(features.values)
        self._fitted = True
        return scaled

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform using already-fitted scaler."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform first.")
        features = self._build_features(df)
        return self.scaler.transform(features.values)

    def inverse_scale_target(self, values: np.ndarray) -> np.ndarray:
        """Inverse-scale target column predictions."""
        target_idx = self.feature_names.index(self.target_col)
        mean = self.scaler.mean_[target_idx]
        std = np.sqrt(self.scaler.var_[target_idx])
        return values * std + mean

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.index = pd.to_datetime(df.index)

        parts = []

        # 1. Cyclical time features
        parts.append(CyclicalEncoder.encode(df.index.hour.to_series(index=df.index).rename("hour"), 24))
        parts.append(CyclicalEncoder.encode(df.index.dayofweek.to_series(index=df.index).rename("dow"), 7))
        parts.append(CyclicalEncoder.encode(df.index.month.to_series(index=df.index).rename("month"), 12))

        # 2. Raw target (will be scaled)
        parts.append(df[[self.target_col]])

        # 3. Lag features
        for lag in self.lag_hours:
            col = df[self.target_col].shift(lag).rename(f"{self.target_col}_lag{lag}")
            parts.append(col.to_frame())

        # 4. Rolling statistics
        for w in self.rolling_windows:
            rolled = df[self.target_col].rolling(window=w, min_periods=1)
            parts.append(rolled.mean().rename(f"{self.target_col}_roll_mean_{w}").to_frame())
            parts.append(rolled.std().rename(f"{self.target_col}_roll_std_{w}").fillna(0).to_frame())

        # 5. Exogenous event marker
        is_holiday = df.index.normalize().isin(pd.to_datetime(self.holiday_dates))
        parts.append(pd.Series(is_holiday.astype(float), index=df.index, name="is_holiday").to_frame())

        # 6. Synthetic dynamic pricing (sinusoidal peak-pricing pattern)
        pricing = self._synthetic_pricing(df.index)
        parts.append(pricing.to_frame())

        # 7. Any extra numeric columns already in df (e.g., temperature, occupancy)
        extra_cols = [c for c in df.columns if c != self.target_col]
        if extra_cols:
            parts.append(df[extra_cols])

        combined = pd.concat(parts, axis=1)
        combined = combined.fillna(method="bfill").fillna(0)
        return combined

    @staticmethod
    def _synthetic_pricing(index: pd.DatetimeIndex) -> pd.Series:
        """
        Simulate a time-of-use electricity price signal.
        Peak: 08:00–20:00 weekdays → higher price.
        Off-peak: nights & weekends → lower price.
        """
        hour = index.hour
        is_weekday = index.dayofweek < 5
        price = np.where((hour >= 8) & (hour < 20) & is_weekday, 1.0, 0.3)
        # Add a mild sinusoidal modulation
        price = price + 0.1 * np.sin(2 * np.pi * hour / 24)
        return pd.Series(price, index=index, name="dynamic_price")


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Simulate 2 weeks of hourly data
    idx = pd.date_range("2024-01-01", periods=336, freq="H")
    df = pd.DataFrame({
        "Global_active_power": np.random.rand(336) * 3 + 1,
        "temperature": np.random.rand(336) * 15 + 5,
        "occupancy": np.random.randint(0, 5, 336).astype(float),
    }, index=idx)

    eng = SmartGridFeatureEngineer(holiday_dates=["2024-01-01", "2024-01-15"])
    X = eng.fit_transform(df)
    print(f"✅ Feature matrix shape: {X.shape}")
    print(f"   Features: {eng.feature_names[:8]} ... ({len(eng.feature_names)} total)")
