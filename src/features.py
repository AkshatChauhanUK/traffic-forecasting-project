"""
Feature engineering for Smart City Traffic Forecasting.

Builds time-based, lag, rolling, and holiday features from the raw
(DateTime, Junction, Vehicles, ID) dataset.
"""

import pandas as pd
import numpy as np

# Indian public holidays observed in the dataset window (Nov 2015 - Jun 2017).
# Hand-curated list (major national holidays) since the project targets an
# Indian "smart city" use case. Extend this list if you adapt to other years.
#
# NOTE: This list only covers 2015-2017. If you extend the dataset or run
# next-24h forecasts that roll into an unlisted year, is_holiday will
# silently default to 0 for those dates rather than raising an error.
# That's acceptable for this project's date range, but worth knowing if
# you adapt this for a live, ongoing deployment.
INDIA_HOLIDAYS = [
    "2015-11-11", "2015-11-25", "2015-12-25",
    "2016-01-01", "2016-01-26", "2016-03-07", "2016-03-24", "2016-04-14",
    "2016-08-15", "2016-08-25", "2016-10-02", "2016-10-11", "2016-10-30",
    "2016-11-14", "2016-12-25",
    "2017-01-01", "2017-01-26", "2017-03-13", "2017-04-14", "2017-08-15",
]
INDIA_HOLIDAYS = pd.to_datetime(INDIA_HOLIDAYS)


def load_raw(path: str) -> pd.DataFrame:
    """Load the raw Kaggle CSV and parse datetime."""
    df = pd.read_csv(path)
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df = df.sort_values(["Junction", "DateTime"]).reset_index(drop=True)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar/time-of-day features, including cyclical encodings."""
    df = df.copy()
    dt = df["DateTime"]

    df["hour"] = dt.dt.hour
    df["day"] = dt.dt.day
    df["dayofweek"] = dt.dt.dayofweek  # Monday=0
    df["month"] = dt.dt.month
    df["year"] = dt.dt.year
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_holiday"] = dt.dt.normalize().isin(INDIA_HOLIDAYS).astype(int)

    # Cyclical encodings so the model understands hour 23 is close to hour 0,
    # and December is close to January.
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Rush-hour flags (typical Indian urban traffic peaks)
    df["is_morning_rush"] = df["hour"].between(8, 10).astype(int)
    df["is_evening_rush"] = df["hour"].between(17, 20).astype(int)

    return df


def add_lag_and_rolling_features(df: pd.DataFrame, target_col: str = "Vehicles") -> pd.DataFrame:
    """
    Add lag and rolling-window features, computed PER JUNCTION so history
    from one junction never leaks into another.
    """
    df = df.copy()
    df = df.sort_values(["Junction", "DateTime"])

    grp = df.groupby("Junction")[target_col]

    # Lag features: same hour 1, 24 (yesterday), and 168 (last week) hours ago
    df["lag_1"] = grp.shift(1)
    df["lag_24"] = grp.shift(24)
    df["lag_168"] = grp.shift(168)

    # Rolling statistics over the trailing window (shifted by 1 to avoid leakage)
    shifted = df.groupby("Junction")[target_col].shift(1)
    df["roll_mean_24"] = shifted.groupby(df["Junction"]).rolling(24, min_periods=1).mean().reset_index(drop=True)
    df["roll_std_24"] = shifted.groupby(df["Junction"]).rolling(24, min_periods=1).std().reset_index(drop=True)
    df["roll_mean_168"] = shifted.groupby(df["Junction"]).rolling(168, min_periods=1).mean().reset_index(drop=True)

    return df


def build_feature_frame(raw_csv_path: str) -> pd.DataFrame:
    """Full pipeline: load -> time features -> lag/rolling features -> dropna."""
    df = load_raw(raw_csv_path)
    df = add_time_features(df)
    df = add_lag_and_rolling_features(df)
    # Drop rows with NaNs introduced by lag_168 (first week per junction)
    df = df.dropna().reset_index(drop=True)
    return df


FEATURE_COLUMNS = [
    "hour", "dayofweek", "month", "is_weekend", "is_holiday",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_morning_rush", "is_evening_rush",
    "lag_1", "lag_24", "lag_168",
    "roll_mean_24", "roll_std_24", "roll_mean_168",
]

TARGET_COLUMN = "Vehicles"


if __name__ == "__main__":
    df = build_feature_frame("data/traffic.csv")
    print(df.shape)
    print(df.head())
    df.to_csv("data/features.csv", index=False)
    print("Saved data/features.csv")