# Smart City Traffic Forecasting & Junction Monitoring System

An end-to-end machine learning system that forecasts hourly traffic volume across 4 city junctions, compares classical statistical, gradient boosting, and deep learning models, and serves live predictions through a REST API and an interactive dashboard.

## Live Demo
- **Dashboard:** [traffic-forecasting-project.vercel.app](https://traffic-forecasting-project.vercel.app)
- **API:** [traffic-forecasting-project-production.up.railway.app/junctions](https://traffic-forecasting-project-production.up.railway.app/junctions)

## Problem Statement
Government planners need to anticipate traffic congestion before it happens, not just react to it. This project builds a forecasting system that predicts vehicle counts one hour and twenty-four hours ahead at each junction, flags congestion risk, and surfaces it through a real-time monitoring dashboard — turning historical traffic data into a decision-support tool for infrastructure planning.

## Dataset
- **Source:** [Smart City Traffic Patterns](https://www.kaggle.com/datasets/utathya/smart-city-traffic-patterns) (Kaggle, originally from a McKinsey Analytics hackathon)
- **Size:** 48,120 hourly readings across 4 junctions (Nov 2015 – Jun 2017)
- **Columns:** `DateTime`, `Junction`, `Vehicles`, `ID`
- **Note:** Junction 4 has a shorter history (Jan–Jun 2017 only) — handled explicitly in the modeling approach rather than ignored.

## Approach

### 1. Feature Engineering (`src/features.py`)
- Cyclical time encodings (hour, day-of-week, month) so the model understands that hour 23 is close to hour 0
- Lag features (1h, 24h, 168h ago) and rolling statistics (24h, 168h mean/std)
- Holiday flags and rush-hour indicators
- All lag/rolling features computed **per junction** to prevent data leakage across junctions

### 2. Model Comparison (`src/train_models.py`)
Three model families trained and evaluated **per junction** (separate train/test split and models for each — junctions are never mixed), using a strict **time-based split**: the last 14 days (336 hours) of each junction's data is held out as test, everything before is training. Train size differs a lot by junction since each has a different amount of history (Junction 1 has ~14k+ training rows, Junction 4 only ~3.8k) — that's expected, not a bug.

| Model | Approach |
|---|---|
| SARIMA | Classical statistical baseline on the raw series |
| XGBoost | Gradient boosting on engineered lag/time features |
| LSTM | Sequence model on raw trailing windows |

*SARIMA note: trained on only the trailing 60 days with seasonal differencing off and a lighter optimizer, to avoid the huge memory footprint a full seasonal model hits on 14,000+ hourly rows.*

**Result:** XGBoost won on **all four junctions** — both LSTM and SARIMA underperformed it, with SARIMA the weakest.

**Evaluation note:** LSTM's test-set loop used to leak the true future value back into its own history at each step ("teacher forcing"), which made its scores look better than real deployment performance. Fixed now — it only feeds back its own predictions, matching exactly how `/predict/{junction}/next24` works live. Post-fix, LSTM's error rose a lot (e.g. Junction 3's MAPE: ~18% → ~114%) since its mistakes now compound over the walk-forward window — that's the honest number, and it's why XGBoost wins everywhere once compared fairly.

| Junction | Best Model | RMSE | MAPE |
|---|---|---|---|
| 1 | XGBoost | 4.44 | 4.6% |
| 2 | XGBoost | 2.74 | 9.8% |
| 3 | XGBoost | 8.77 | 17.7% |
| 4 | XGBoost | 3.78 | 34.4% |

<details>
<summary>Full model comparison (all 3 models, all 4 junctions)</summary>

| Junction | Model | RMSE | MAE | MAPE |
|---|---|---|---|---|
| 1 | XGBoost | 4.44 | 3.25 | 4.6% |
| 1 | LSTM | 42.44 | 35.24 | 43.6% |
| 1 | SARIMA | 74.74 | 69.36 | 94.6% |
| 2 | XGBoost | 2.74 | 2.18 | 9.8% |
| 2 | LSTM | 13.83 | 11.03 | 43.2% |
| 2 | SARIMA | 26.20 | 24.21 | 94.9% |
| 3 | XGBoost | 8.77 | 3.51 | 17.7% |
| 3 | LSTM | 19.22 | 14.36 | 114.2% |
| 3 | SARIMA | 23.08 | 19.17 | 96.5% |
| 4 | XGBoost | 3.78 | 2.53 | 34.4% |
| 4 | LSTM | 4.94 | 3.30 | 39.0% |
| 4 | SARIMA | 9.83 | 8.51 | 95.2% |

</details>

### 3. API (`api/main.py`)
FastAPI backend serving:
- `GET /junctions` — metadata and best model per junction
- `GET /history/{junction}` — recent actual traffic
- `GET /predict/{junction}` — next-hour forecast with congestion risk
- `GET /predict/{junction}/next24` — 24-hour walk-forward forecast

Each junction automatically uses whichever model performed best for it during evaluation — restricted to models the API can actually serve (XGBoost, LSTM). SARIMA is trained and scored for the comparison above but deliberately excluded from "best model" selection, since there's no SARIMA inference path implemented yet (see "What I'd Improve Next"). In practice this exclusion currently has no effect on serving anyway, since XGBoost now wins on every junction.

### 4. Dashboard (`dashboard/`)
A React + Recharts console showing live predictions, a 72h-actual vs 24h-forecast chart with a congestion threshold line, and an hour-by-hour risk strip — built for a 2-minute live demo, not just a notebook printout.

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
- Add a SARIMA inference path to the API (currently trained/scored for comparison only, but not servable)
- Add exogenous features to SARIMA (holidays, rush hour) — it currently underperforms everywhere, partly because it only sees the raw series with no calendar/rush-hour context
- Investigate why LSTM's walk-forward error compounds so much over 24h (junction 3 in particular) — possibly a shorter lookback window, retraining on residuals, or a hybrid that blends LSTM with XGBoost's lag features
- Deploy with a scheduled retraining pipeline instead of static model files
- Add anomaly detection for unusual traffic spikes (accidents, events) separate from routine congestion
- Lock down CORS (`allow_origins=["*"]`) to the dashboard's actual origin before treating this as production-ready

## Author
Akshat Chauhan — built as part of a 3rd-year B.Tech CSE ML internship project.