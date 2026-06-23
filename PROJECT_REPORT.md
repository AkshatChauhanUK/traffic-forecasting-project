# Project Report: Smart City Traffic Forecasting & Junction Monitoring System

**Submitted as part of:** Machine Learning Internship Program — UpSkill Campus / UniConverge Technologies (UCT)
**Author:** Akshat Chauhan, B.Tech CSE (3rd Year)
**Code Repository:** https://github.com/AkshatChauhanUK/traffic-forecasting-project
**Live Dashboard:** https://traffic-forecasting-project.vercel.app
**Live API:** https://traffic-forecasting-project-production.up.railway.app/junctions

---

## 1. Background

UniConverge Technologies (UCT) runs an internship program structured around real-world ML problems across four domains: Agriculture, Predictive Maintenance, Smart City, and Industrial Manufacturing. This project was selected from the **Smart City** category — the "Smart City Traffic Patterns" problem, where a city government wants to better understand and forecast traffic at its junctions in order to plan infrastructure and manage congestion proactively rather than reactively.

## 2. Problem Statement & Relevance

Traffic congestion is a growing problem in Indian cities, and most municipal traffic systems are reactive — signals and interventions respond *after* congestion has already built up. If a city could reliably forecast traffic volume an hour or a day ahead at key junctions, planners could:

- Adjust signal timing proactively before rush-hour peaks
- Flag junctions likely to exceed safe congestion thresholds
- Make more informed infrastructure investment decisions (which junctions need widening, new signals, etc.)

This project directly addresses that need: given ~20 months of historical hourly traffic counts at 4 junctions, build a system that forecasts future traffic and flags congestion risk in real time, accessible through a live dashboard.

## 3. Dataset

- **Source:** Smart City Traffic Patterns dataset (Kaggle, originally from a McKinsey Analytics Online Hackathon)
- **Volume:** 48,120 hourly records across 4 junctions, November 2015 to June 2017
- **Schema:** `DateTime`, `Junction` (1–4), `Vehicles` (hourly count), `ID`
- **Data quality:** Verified zero missing values, zero duplicate timestamps, and zero gaps in the hourly sequence per junction — an unusually clean real-world dataset.
- **Notable asymmetry:** Junction 4 only has data from January 2017 onward (4,344 rows vs. ~14,592 for Junctions 1–3). Rather than discard this junction or pretend the gap doesn't exist, the modeling pipeline treats each junction independently, so Junction 4's shorter history doesn't corrupt the others.

## 4. Design & Approach

### 4.1 Feature Engineering
Implemented in `src/features.py`. Key design decisions:
- **Cyclical encoding** of hour, day-of-week, and month (sine/cosine transforms) so the model understands that hour 23 and hour 0 are adjacent, rather than treating them as numerically distant.
- **Lag features** at 1 hour, 24 hours (same hour yesterday), and 168 hours (same hour last week) — these capture daily and weekly seasonality directly as inputs.
- **Rolling statistics** (24h and 168h trailing mean/std), shifted by one step to avoid leaking the current value into its own features.
- **Holiday and rush-hour flags**, since traffic on Indian national holidays and during 8–10 AM / 5–8 PM windows behaves differently from an average hour.
- All lag/rolling computations are grouped **per junction**, so history from one junction never leaks into another junction's features — an easy mistake to make with `groupby` operations that I deliberately guarded against.

### 4.2 Model Comparison
Implemented in `src/train_models.py`. Three model families were trained and evaluated **per junction**, using a strict **time-based train/test split** — the last 14 days of each junction's data held out as test, never a random split (random splits leak future information into training for time series, which would make the evaluation meaningless).

| Model | Description |
|---|---|
| **SARIMA** | Classical seasonal ARIMA on the raw series, no exogenous features |
| **XGBoost** | Gradient boosting on the full engineered feature set |
| **LSTM** | Two-layer LSTM network on raw 24-hour trailing windows |

### 4.3 API
Implemented in `src/api/main.py` using FastAPI. Exposes four endpoints (`/junctions`, `/history/{junction}`, `/predict/{junction}`, `/predict/{junction}/next24`), and automatically routes each junction's prediction request to whichever model performed best for that junction during evaluation — rather than forcing one model architecture on all four junctions.

### 4.4 Dashboard
Built with React + Recharts (`dashboard/`). Shows live next-hour predictions, a 72-hour-actual-to-24-hour-forecast chart with a congestion threshold line, and an hour-by-hour risk strip — designed to be demoable in under two minutes rather than just a set of notebook printouts.

## 5. Results

| Junction | Best Model | RMSE | MAE | MAPE |
|---|---|---|---|---|
| 1 | XGBoost | 4.36 | 3.16 | 4.5% |
| 2 | XGBoost | 2.80 | 2.23 | 10.0% |
| 3 | LSTM | 8.58 | 3.51 | 18.2% |
| 4 | LSTM | 3.41 | 2.26 | 31.1% |

SARIMA underperformed substantially at every junction (RMSE 9.8–74.7, MAPE consistently near 95%) — close to simply predicting the historical mean. This is a genuinely informative result rather than a failure to fix: SARIMA has no access to the engineered features (lag windows, holiday flags, rush-hour indicators) that the other two models exploit, and pure statistical models without exogenous regressors struggle on traffic data with strong daily/weekly structure.

**Pattern observed:** XGBoost won on the higher-traffic, longer-history junctions (1, 2), where engineered lag/rolling features had enough data to be reliable. LSTM won on the lower-volume junctions (3, 4), where sequence modeling generalized slightly better — plausibly because Junction 4's short history (4 months) gave the rolling-window features less time to stabilize, while the LSTM's sequence-based approach adapted faster.

## 6. Implementation Details

- **Stack:** Python (pandas, scikit-learn, XGBoost, statsmodels, TensorFlow/Keras), FastAPI, React, Recharts, Vite
- **Deployment:** Backend deployed on Railway (containerized, auto-restart on failure); dashboard deployed on Vercel, configured to talk to the live backend rather than localhost
- **Version control:** Full project history tracked in Git, public repository on GitHub with a structured README

## 7. Learnings

- **Time-series evaluation requires different discipline than standard ML.** A random train/test split would have given misleadingly optimistic numbers; the time-based split was essential to get a trustworthy estimate of real forecasting performance.
- **Per-group feature engineering is easy to get subtly wrong.** Lag and rolling features had to be computed strictly within each junction's own history — an oversight here would silently corrupt every downstream model without throwing an error.
- **Model choice should be data-driven, not assumed.** Going in, I expected LSTM to dominate as the "more sophisticated" model. The actual results — XGBoost winning on two junctions — were a useful reminder that engineered features can beat a deep model when the underlying patterns (daily/weekly cycles) are well-captured by lags and rolling stats.
- **Memory constraints are a real engineering tradeoff, not just a bug to "fix."** Full-history seasonal SARIMAX on hourly data exhausted available memory; windowing the training data to the most recent 60 days was a deliberate, explainable design choice rather than a workaround to hide.
- **Deployment surfaces problems notebooks never would.** Getting the backend and dashboard actually live (Railway + Vercel) surfaced real issues — CORS, environment-specific file paths, build configuration — that never come up when a project only runs locally.

## 8. What I'd Improve Next
- Add exogenous regressors (holidays, rush-hour flags) to SARIMA for a fairer three-way comparison
- Replace static model files with a scheduled retraining pipeline as new traffic data arrives
- Add anomaly detection for irregular spikes (accidents, events) as distinct from routine rush-hour congestion