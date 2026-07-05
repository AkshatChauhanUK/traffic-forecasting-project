"""
Model training & comparison for Smart City Traffic Forecasting.

Trains three model families PER JUNCTION:
  1. SARIMA       - classical statistical baseline
  2. XGBoost      - gradient boosting on engineered features
  3. LSTM         - sequence model on raw lag windows

Uses a strict time-based (walk-forward style) train/test split -
NEVER a random split, since shuffling time series leaks the future
into the past during evaluation.

Saves the best model per junction to models/, and a comparison
report to reports/model_comparison.csv
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from features import (build_feature_frame, FEATURE_COLUMNS, TARGET_COLUMN,
                      SARIMA_EXOG_COLUMNS, get_sarima_exog)

tf.random.set_seed(42)
np.random.seed(42)

TEST_HOURS = 24 * 14  # last 14 days held out as test set, per junction
LSTM_WINDOW = 24      # use trailing 24 hours to predict next hour


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate(y_true, y_pred, model_name, junction):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mp = mape(y_true, y_pred)
    return {"junction": junction, "model": model_name, "rmse": rmse, "mae": mae, "mape": mp}


def split_train_test(df_j: pd.DataFrame):
    """Time-based split: last TEST_HOURS rows are test, rest is train."""
    df_j = df_j.sort_values("DateTime").reset_index(drop=True)
    split_idx = len(df_j) - TEST_HOURS
    train = df_j.iloc[:split_idx]
    test = df_j.iloc[split_idx:]
    return train, test

def train_sarima(train: pd.DataFrame, test: pd.DataFrame):
    """
    SARIMAX with exogenous features (is_holiday, is_weekend, rush-hour flags).

    Adding exogenous features gives SARIMA context it previously lacked —
    it can now account for holiday dips and rush-hour peaks rather than
    treating every hour identically. This makes the three-way comparison
    (SARIMA vs XGBoost vs LSTM) fairer, since XGBoost already exploits
    these calendar signals via its engineered feature set.

    Memory notes (same as before):
      - Trained on only the trailing 60 days to avoid a huge state-space matrix.
      - seasonal D=0, simple_differencing=True, method="powell" for stability.
    """
    SARIMA_TRAIN_HOURS = 24 * 60
    y_full = train[TARGET_COLUMN].values
    exog_full = get_sarima_exog(train).values

    if len(y_full) > SARIMA_TRAIN_HOURS:
        y_train = y_full[-SARIMA_TRAIN_HOURS:]
        exog_train = exog_full[-SARIMA_TRAIN_HOURS:]
    else:
        y_train = y_full
        exog_train = exog_full

    exog_test = get_sarima_exog(test).values

    model = SARIMAX(
        y_train,
        exog=exog_train,
        order=(1, 1, 1),
        seasonal_order=(0, 0, 0, 0),
        enforce_stationarity=False,
        enforce_invertibility=False,
        simple_differencing=True,
    )
    fit = model.fit(disp=False, method="powell", maxiter=50)
    preds = fit.forecast(steps=len(test), exog=exog_test)
    preds = np.clip(preds, 0, None)
    return preds, fit


def train_xgboost(train: pd.DataFrame, test: pd.DataFrame):
    X_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN]
    X_test, y_test = test[FEATURE_COLUMNS], test[TARGET_COLUMN]

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    preds = np.clip(preds, 0, None)
    return preds, model


def make_lstm_sequences(values: np.ndarray, window: int):
    X, y = [], []
    for i in range(window, len(values)):
        X.append(values[i - window:i])
        y.append(values[i])
    return np.array(X), np.array(y)


def train_lstm(train: pd.DataFrame, test: pd.DataFrame):
    """LSTM on normalized Vehicles sequences.

    Walk-forward evaluation: at each test step, the model only sees its OWN
    prior predictions plus the real training history that came before the
    test period - never the true future value for the step it's currently
    predicting. This matches how /predict/{junction}/next24 actually serves
    predictions in production (it can't know the real future either), so the
    reported test-set RMSE/MAPE reflect genuine deployment performance.

    (Earlier version of this function fed the true test-set value into
    `history` after each step instead of the prediction - that's teacher
    forcing, not walk-forward, and it made LSTM's test metrics look better
    than the model would actually perform in /predict/{junction}/next24.
    Fixed here so training-time eval matches serving-time behavior.)
    """
    full = pd.concat([train, test])[TARGET_COLUMN].values.astype(float)
    train_len = len(train)

    # Normalize using train stats only (avoid leakage)
    mu, sigma = full[:train_len].mean(), full[:train_len].std()
    sigma = sigma if sigma > 0 else 1.0
    norm = (full - mu) / sigma

    X_train, y_train = make_lstm_sequences(norm[:train_len], LSTM_WINDOW)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    model = Sequential([
        LSTM(64, activation="tanh", return_sequences=True, input_shape=(LSTM_WINDOW, 1)),
        Dropout(0.2),
        LSTM(32, activation="tanh"),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    es = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=30, batch_size=64, verbose=0, callbacks=[es])

    # True walk-forward prediction over the test period: feed the model's
    # OWN prediction back in as history, never the real future value.
    history = list(norm[:train_len])
    preds_norm = []
    for i in range(len(test)):
        window = np.array(history[-LSTM_WINDOW:]).reshape(1, LSTM_WINDOW, 1)
        pred = model.predict(window, verbose=0)[0, 0]
        preds_norm.append(pred)
        history.append(pred)  # <-- fixed: was `norm[train_len + i]` (ground truth leak)

    preds = np.array(preds_norm) * sigma + mu
    preds = np.clip(preds, 0, None)
    return preds, model, (mu, sigma)


def run_all():
    print("Building feature frame...")
    df = build_feature_frame("data/traffic.csv")

    os.makedirs(r"E:\traffic-forecasting-project\traffic-forecasting-project\models", exist_ok=True)
    os.makedirs(r"E:\traffic-forecasting-project\traffic-forecasting-project\reports", exist_ok=True)
    MODELS_OUT = r"E:\traffic-forecasting-project\traffic-forecasting-project\models"
    REPORTS_OUT = r"E:\traffic-forecasting-project\traffic-forecasting-project\reports"

    results = []
    predictions_store = {}

    for junction in sorted(df["Junction"].unique()):
        print(f"\n=== Junction {junction} ===")
        df_j = df[df["Junction"] == junction]

        if len(df_j) < TEST_HOURS * 3:
            print(f"  Limited history ({len(df_j)} rows) for this junction")

        train, test = split_train_test(df_j)
        print(f"  Train: {len(train)} rows | Test: {len(test)} rows")

        y_test = test[TARGET_COLUMN].values
        junction_preds = {"dates": test["DateTime"].astype(str).tolist(), "actual": y_test.tolist()}

        try:
            print("  Training SARIMA...")
            preds_sarima, sarima_fit = train_sarima(train, test)
            results.append(evaluate(y_test, preds_sarima, "SARIMA", junction))
            junction_preds["sarima"] = preds_sarima.tolist()
            # Save the fitted SARIMA model so api/main.py can load it for inference.
            joblib.dump(sarima_fit, f"{MODELS_OUT}\\sarima_junction_{junction}.pkl")
        except Exception as e:
            print(f"  SARIMA failed: {e}")

        print("  Training XGBoost...")
        preds_xgb, xgb_model = train_xgboost(train, test)
        results.append(evaluate(y_test, preds_xgb, "XGBoost", junction))
        junction_preds["xgboost"] = preds_xgb.tolist()
        joblib.dump(xgb_model, f"{MODELS_OUT}\\xgb_junction_{junction}.pkl")

        try:
            print("  Training LSTM...")
            preds_lstm, lstm_model, norm_stats = train_lstm(train, test)
            results.append(evaluate(y_test, preds_lstm, "LSTM", junction))
            junction_preds["lstm"] = preds_lstm.tolist()
            lstm_model.save(f"{MODELS_OUT}\\lstm_junction_{junction}.keras")
            with open(f"{MODELS_OUT}\\lstm_junction_{junction}_norm.json", "w") as f:
                json.dump({"mu": float(norm_stats[0]), "sigma": float(norm_stats[1])}, f)
        except Exception as e:
            print(f"  LSTM failed: {e}")

        predictions_store[str(junction)] = junction_preds

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{REPORTS_OUT}\\model_comparison.csv", index=False)
    print("\n=== Final comparison ===")
    print(results_df.sort_values(["junction", "rmse"]))

    with open(f"{REPORTS_OUT}\\predictions.json", "w") as f:
        json.dump(predictions_store, f)

    servable = results_df[results_df["model"].isin(["XGBoost", "LSTM", "SARIMA"])]
    best = servable.loc[servable.groupby("junction")["rmse"].idxmin()]
    best.to_csv(f"{REPORTS_OUT}\\best_models.csv", index=False)
    print("\nBest model per junction:")
    print(best)

    return results_df


if __name__ == "__main__":
    run_all()