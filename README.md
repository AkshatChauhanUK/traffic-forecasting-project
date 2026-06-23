# Smart City Traffic Forecasting & Junction Monitoring System

An end-to-end machine learning system that forecasts hourly traffic volume across 4 city junctions, compares classical statistical, gradient boosting, and deep learning models, and serves live predictions through a REST API and an interactive dashboard.

## Live Demo
*(Add deployed link here once hosted)*

## Problem Statement
Government planners need to anticipate traffic congestion before it happens, not just react to it. This project builds a forecasting system that predicts vehicle counts one hour and twenty-four hours ahead at each junction, flags congestion risk, and surfaces it through a real-time monitoring dashboard â€” turning historical traffic data into a decision-support tool for infrastructure planning.

## Dataset
- **Source:** [Smart City Traffic Patterns](https://www.kaggle.com/datasets/utathya/smart-city-traffic-patterns) (Kaggle, originally from a McKinsey Analytics hackathon)
- **Size:** 48,120 hourly readings across 4 junctions (Nov 2015 â€“ Jun 2017)
- **Columns:** `DateTime`, `Junction`, `Vehicles`, `ID`
- **Note:** Junction 4 has a shorter history (Janâ€“Jun 2017 only) â€” handled explicitly in the modeling approach rather than ignored.

## Approach

### 1. Feature Engineering (`src/features.py`)
- Cyclical time encodings (hour, day-of-week, month) so the model understands that hour 23 is close to hour 0
- Lag features (1h, 24h, 168h ago) and rolling statistics (24h, 168h mean/std)
- Holiday flags and rush-hour indicators
- All lag/rolling features computed **per junction** to prevent data leakage across junctions

### 2. Model Comparison (`src/train_models.py`)
Three model families trained and evaluated per junction, using a strict **time-based train/test split** (never random â€” shuffling time series leaks the future into the past):

| Model | Approach |
|---|---|
| SARIMA | Classical statistical baseline on the raw series |
| XGBoost | Gradient boosting on engineered lag/time features |
| LSTM | Sequence model on raw trailing windows |

**Result:** XGBoost won on high-traffic junctions (1, 2) where engineered features captured the pattern well; LSTM won on lower-volume junctions (3, 4) where sequence modeling generalized better. SARIMA underperformed everywhere â€” a useful finding: pure statistical models struggle without the engineered context that the other two approaches exploit.

| Junction | Best Model | RMSE | MAPE |
|---|---|---|---|
| 1 | XGBoost | 4.36 | 4.5% |
| 2 | XGBoost | 2.80 | 10.0% |
| 3 | LSTM | 8.58 | 18.2% |
| 4 | LSTM | 3.41 | 31.1% |

### 3. API (`api/main.py`)
FastAPI backend serving:
- `GET /junctions` â€” metadata and best model per junction
- `GET /history/{junction}` â€” recent actual traffic
- `GET /predict/{junction}` â€” next-hour forecast with congestion risk
- `GET /predict/{junction}/next24` â€” 24-hour walk-forward forecast

Each junction automatically uses whichever model performed best for it during evaluation.

### 4. Dashboard (`dashboard/`)
A React + Recharts console showing live predictions, a 72h-actual vs 24h-forecast chart with a congestion threshold line, and an hour-by-hour risk strip â€” built for a 2-minute live demo, not just a notebook printout.

## Tech Stack
Python, pandas, scikit-learn, XGBoost, statsmodels, TensorFlow/Keras, FastAPI, React, Recharts, Vite

## Running Locally

**Backend:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/features.py
python src/train_models.py
uvicorn api.main:app --reload --port 8000
```

**Dashboard:**
```bash
cd dashboard
npm install
npm run dev
```

Visit `http://localhost:5173`.

## What I'd Improve Next
- Add exogenous features to SARIMA (holidays, rush hour) for a fairer three-way comparison
- Deploy with a scheduled retraining pipeline instead of static model files
- Add anomaly detection for unusual traffic spikes (accidents, events) separate from routine congestion

## Author
Akshat Chauhan — built as part of a 3rd-year B.Tech CSE ML internship project.
