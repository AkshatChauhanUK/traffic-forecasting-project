"""
FastAPI backend for Smart City Traffic Forecasting.

Serves predictions from the best trained model per junction, plus
historical data and basic congestion alerting.

Run with: uvicorn api.main:app --reload --port 8000
(from project root, so it can find the models/ and data/ folders)
"""

import json
import os
import sys
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Allow importing src/features.py when running from project root
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from features import build_feature_frame, FEATURE_COLUMNS, get_sarima_exog  # noqa: E402

app = FastAPI(title="Smart City Traffic Forecasting API")

ALLOWED_ORIGINS = [
    "https://traffic-forecasting-project.vercel.app",
    "http://localhost:5173",  # local dev
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["*"],
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "traffic.csv")
BEST_MODELS_PATH = os.path.join(os.path.dirname(__file__), "..", "reports", "best_models.csv")

CONGESTION_PERCENTILE = 0.90

# All three models are now servable — SARIMA saves a .pkl artifact in models/
# and has a full inference path below.
SERVABLE_MODELS = {"XGBoost", "LSTM", "SARIMA"}

_feature_cache = {}
_best_model_cache = {}
_threshold_cache = {}


def get_feature_frame():
    if "df" not in _feature_cache:
        _feature_cache["df"] = build_feature_frame(DATA_PATH)
    return _feature_cache["df"]


def get_known_junctions() -> set[int]:
    df = get_feature_frame()
    return set(int(j) for j in df["Junction"].unique())


def get_best_model_name(junction: int) -> str:
    if not _best_model_cache:
        best_df = pd.read_csv(BEST_MODELS_PATH)
        for _, row in best_df.iterrows():
            model_name = row["model"]
            if model_name not in SERVABLE_MODELS:
                # Defensive guard: best_models.csv should already exclude
                # non-servable models (see train_models.py), but if it ever
                # doesn't, fall back instead of trying to load a model with
                # no inference path.
                model_name = "XGBoost"
            _best_model_cache[int(row["junction"])] = model_name
    return _best_model_cache.get(junction, "XGBoost")


def get_congestion_threshold(junction: int) -> float:
    if junction not in _threshold_cache:
        df = get_feature_frame()
        sub = df[df["Junction"] == junction]["Vehicles"]
        _threshold_cache[junction] = float(sub.quantile(CONGESTION_PERCENTILE))
    return _threshold_cache[junction]


class PredictionResponse(BaseModel):
    junction: int
    datetime: str
    predicted_vehicles: float
    model_used: str
    congestion_risk: str
    threshold: float




class HistoryPoint(BaseModel):
    datetime: str
    vehicles: int


@app.get("/")
def root():
    return {
        "message": "Smart City Traffic Forecasting API",
        "endpoints": ["/junctions", "/history/{junction}", "/predict/{junction}", "/predict/{junction}/next24", "/anomalies/{junction}"],
    }


@app.get("/junctions")
def list_junctions():
    """Return basic info about each junction: data range, avg traffic, best model."""
    df = get_feature_frame()
    result = []
    for j in sorted(df["Junction"].unique()):
        sub = df[df["Junction"] == j]
        result.append({
            "junction": int(j),
            "data_start": str(sub["DateTime"].min()),
            "data_end": str(sub["DateTime"].max()),
            "avg_vehicles": round(float(sub["Vehicles"].mean()), 2),
            "max_vehicles": int(sub["Vehicles"].max()),
            "best_model": get_best_model_name(int(j)),
        })
    return result


@app.get("/history/{junction}", response_model=list[HistoryPoint])
def get_history(junction: int, hours: int = 168):
    """Return the last `hours` of actual historical traffic for a junction."""
    df = get_feature_frame()
    sub = df[df["Junction"] == junction].sort_values("DateTime")
    if sub.empty:
        raise HTTPException(status_code=404, detail=f"Junction {junction} not found")
    sub = sub.tail(hours)
    return [
        {"datetime": str(row["DateTime"]), "vehicles": int(row["Vehicles"])}
        for _, row in sub.iterrows()
    ]


def load_model_for_junction(junction: int, model_name: str):
    """Load the saved model object for a given junction + model name."""
    if model_name == "XGBoost":
        path = os.path.join(MODELS_DIR, f"xgb_junction_{junction}.pkl")
        if not os.path.exists(path):
            return None, None
        return joblib.load(path), "xgboost"

    if model_name == "LSTM":
        import tensorflow as tf
        path = os.path.join(MODELS_DIR, f"lstm_junction_{junction}.keras")
        norm_path = os.path.join(MODELS_DIR, f"lstm_junction_{junction}_norm.json")
        if not os.path.exists(path):
            return None, None
        model = tf.keras.models.load_model(path)
        with open(norm_path) as f:
            norm_stats = json.load(f)
        return (model, norm_stats), "lstm"

    if model_name == "SARIMA":
        path = os.path.join(MODELS_DIR, f"sarima_junction_{junction}.pkl")
        if not os.path.exists(path):
            return None, None
        return joblib.load(path), "sarima"

    return None, None


def predict_next_hour_xgb(model, junction: int):
    """Predict the next hour using the XGBoost model and latest available features."""
    df = get_feature_frame()
    sub = df[df["Junction"] == junction].sort_values("DateTime")
    last_row = sub.iloc[[-1]][FEATURE_COLUMNS]
    pred = model.predict(last_row)[0]
    next_dt = sub["DateTime"].iloc[-1] + timedelta(hours=1)
    return float(max(pred, 0)), next_dt


def predict_next_hour_lstm(model_bundle, junction: int, window: int = 24):
    """Predict the next hour using the LSTM model and latest 24-hour window."""
    model, norm_stats = model_bundle
    mu, sigma = norm_stats["mu"], norm_stats["sigma"]

    df = get_feature_frame()
    sub = df[df["Junction"] == junction].sort_values("DateTime")
    last_values = sub["Vehicles"].tail(window).values.astype(float)
    norm = (last_values - mu) / sigma
    X = norm.reshape(1, window, 1)
    pred_norm = model.predict(X, verbose=0)[0, 0]
    pred = pred_norm * sigma + mu
    next_dt = sub["DateTime"].iloc[-1] + timedelta(hours=1)
    return float(max(pred, 0)), next_dt


def predict_next_hour_sarima(fit, junction: int):
    """Predict the next hour using the fitted SARIMAX model with exog features."""
    df = get_feature_frame()
    sub = df[df["Junction"] == junction].sort_values("DateTime")
    next_dt = sub["DateTime"].iloc[-1] + timedelta(hours=1)
    # Build a one-row exog frame for the next hour using its calendar features
    next_row = pd.DataFrame({"DateTime": [next_dt]})
    next_row["DateTime"] = pd.to_datetime(next_row["DateTime"])
    from features import add_time_features, INDIA_HOLIDAYS
    next_row["Junction"] = junction
    next_row["Vehicles"] = 0
    next_row["ID"] = 0
    next_row = add_time_features(next_row)
    exog = get_sarima_exog(next_row).values
    pred = float(max(fit.forecast(steps=1, exog=exog)[0], 0))
    return pred, next_dt


def _require_known_junction(junction: int):
    """Raise a clean 404 for unknown junctions instead of letting a bad
    junction id fall through to an unhandled error deeper in the pipeline."""
    if junction not in get_known_junctions():
        raise HTTPException(status_code=404, detail=f"Junction {junction} not found")


@app.get("/predict/{junction}", response_model=PredictionResponse)
def predict_next_hour(junction: int):
    """Predict traffic for the next hour at a junction, using its best model."""
    _require_known_junction(junction)

    model_name = get_best_model_name(junction)
    model_obj, kind = load_model_for_junction(junction, model_name)

    if model_obj is None:
        raise HTTPException(status_code=404, detail=f"No trained model found for junction {junction}")

    if kind == "xgboost":
        pred_value, next_dt = predict_next_hour_xgb(model_obj, junction)
    elif kind == "sarima":
        pred_value, next_dt = predict_next_hour_sarima(model_obj, junction)
    else:
        pred_value, next_dt = predict_next_hour_lstm(model_obj, junction)

    threshold = get_congestion_threshold(junction)
    risk = "high" if pred_value >= threshold else ("medium" if pred_value >= threshold * 0.75 else "low")

    return {
        "junction": junction,
        "datetime": str(next_dt),
        "predicted_vehicles": round(pred_value, 2),
        "model_used": model_name,
        "congestion_risk": risk,
        "threshold": round(threshold, 2),
    }


@app.get("/predict/{junction}/next24")
def predict_next_24(junction: int):
    """
    Predict the next 24 hours iteratively (walk-forward), feeding each
    prediction back in as input for the next step. This is the realistic
    deployment scenario: at inference time you don't have ground truth
    for future hours, so the model must rely on its own prior outputs.

    (This matches how the LSTM is now evaluated in src/train_models.py too -
    that function previously leaked ground-truth test values into its
    walk-forward loop, which has been fixed so reported metrics reflect
    this same real walk-forward behavior.)
    """
    _require_known_junction(junction)

    model_name = get_best_model_name(junction)
    model_obj, kind = load_model_for_junction(junction, model_name)

    if model_obj is None:
        raise HTTPException(status_code=404, detail=f"No trained model found for junction {junction}")

    df = get_feature_frame()
    sub = df[df["Junction"] == junction].sort_values("DateTime").copy()
    threshold = get_congestion_threshold(junction)

    predictions = []
    last_dt = sub["DateTime"].iloc[-1]

    if kind == "sarima":
        from features import add_time_features
        for step in range(24):
            next_dt = last_dt + timedelta(hours=step + 1)
            next_row = pd.DataFrame({"DateTime": [next_dt], "Junction": junction,
                                     "Vehicles": 0, "ID": 0})
            next_row = add_time_features(next_row)
            exog = get_sarima_exog(next_row).values
            pred = float(max(model_obj.forecast(steps=1, exog=exog)[0], 0))
            risk = "high" if pred >= threshold else ("medium" if pred >= threshold * 0.75 else "low")
            predictions.append({
                "datetime": str(next_dt),
                "predicted_vehicles": round(pred, 2),
                "congestion_risk": risk,
            })

    elif kind == "xgboost":
        working = sub.copy()
        for step in range(24):
            last_row = working.iloc[[-1]][FEATURE_COLUMNS]
            pred = float(max(model_obj.predict(last_row)[0], 0))
            next_dt = working["DateTime"].iloc[-1] + timedelta(hours=1)

            risk = "high" if pred >= threshold else ("medium" if pred >= threshold * 0.75 else "low")
            predictions.append({
                "datetime": str(next_dt),
                "predicted_vehicles": round(pred, 2),
                "congestion_risk": risk,
            })

            new_row = working.iloc[[-1]].copy()
            new_row["DateTime"] = next_dt
            new_row["Vehicles"] = pred
            new_row["hour"] = next_dt.hour
            new_row["dayofweek"] = next_dt.dayofweek
            working = pd.concat([working, new_row], ignore_index=True)
            from features import add_time_features, add_lag_and_rolling_features
            working = add_time_features(working[["DateTime", "Junction", "Vehicles", "ID"]].ffill())
            working = add_lag_and_rolling_features(working)
            working = working.ffill()

    else:
        model, norm_stats = model_obj
        mu, sigma = norm_stats["mu"], norm_stats["sigma"]
        history = list(((sub["Vehicles"].tail(24).values.astype(float)) - mu) / sigma)

        for step in range(24):
            X = np.array(history[-24:]).reshape(1, 24, 1)
            pred_norm = model.predict(X, verbose=0)[0, 0]
            pred = float(max(pred_norm * sigma + mu, 0))
            next_dt = last_dt + timedelta(hours=step + 1)

            risk = "high" if pred >= threshold else ("medium" if pred >= threshold * 0.75 else "low")
            predictions.append({
                "datetime": str(next_dt),
                "predicted_vehicles": round(pred, 2),
                "congestion_risk": risk,
            })
            history.append((pred - mu) / sigma)

    return {"junction": junction, "model_used": model_name, "threshold": round(threshold, 2), "forecast": predictions}


@app.get("/anomalies/{junction}")
def get_anomalies(junction: int, hours: int = 72):
    """
    Detect unusual traffic spikes in the last `hours` of historical data.

    An hour is flagged as an anomaly when its vehicle count deviates more than
    2 standard deviations from the historical mean for that same weekday+hour
    combination (e.g. comparing Monday 8am to all other Monday 8ams in history).
    This separates genuine surprises (accidents, events) from routine congestion,
    which the congestion threshold already handles.
    """
    _require_known_junction(junction)

    df = get_feature_frame()
    sub = df[df["Junction"] == junction].sort_values("DateTime").copy()

    if sub.empty:
        raise HTTPException(status_code=404, detail=f"Junction {junction} not found")

    # Compute historical mean and std per (dayofweek, hour) group
    stats = (
        sub.groupby(["dayofweek", "hour"])["Vehicles"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "hist_mean", "std": "hist_std"})
    )
    # Fill zero std (only one data point for that slot) with global std
    global_std = sub["Vehicles"].std()
    stats["hist_std"] = stats["hist_std"].fillna(global_std).replace(0, global_std)

    # Check last `hours` rows
    recent = sub.tail(hours).copy()
    recent = recent.merge(stats, on=["dayofweek", "hour"], how="left")
    recent["z_score"] = (recent["Vehicles"] - recent["hist_mean"]) / recent["hist_std"]
    recent["is_anomaly"] = recent["z_score"].abs() > 2.0
    recent["anomaly_type"] = recent.apply(
        lambda r: "spike" if r["z_score"] > 2.0 else ("dip" if r["z_score"] < -2.0 else "normal"),
        axis=1
    )

    anomalies = recent[recent["is_anomaly"]].copy()

    return {
        "junction": junction,
        "hours_checked": hours,
        "total_anomalies": int(len(anomalies)),
        "anomalies": [
            {
                "datetime": str(row["DateTime"]),
                "vehicles": int(row["Vehicles"]),
                "expected": round(float(row["hist_mean"]), 1),
                "z_score": round(float(row["z_score"]), 2),
                "type": row["anomaly_type"],
            }
            for _, row in anomalies.iterrows()
        ],
    }