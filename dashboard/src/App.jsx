import { useState, useEffect, useCallback } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from "recharts";
import axios from "axios";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000";

const RISK_COLORS = {
  low: "#3DDC84",
  medium: "#FFB020",
  high: "#FF4D4F",
};

function RiskBadge({ risk }) {
  return (
    <span className="risk-badge" style={{ "--risk-color": RISK_COLORS[risk] || "#888" }}>
      {risk}
    </span>
  );
}

export default function App() {
  const [junctions, setJunctions] = useState([]);
  const [selected, setSelected] = useState(1);
  const [history, setHistory] = useState([]);
  const [forecast, setForecast] = useState(null);
  const [current, setCurrent] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    axios.get(`${API_BASE}/junctions`)
      .then((res) => setJunctions(res.data))
      .catch(() => setError("Can't reach the API. Is uvicorn running on port 8000?"));
  }, []);

  const loadJunctionData = useCallback((junctionId) => {
    setLoading(true);
    setError(null);
    Promise.all([
      axios.get(`${API_BASE}/history/${junctionId}?hours=72`),
      axios.get(`${API_BASE}/predict/${junctionId}/next24`),
      axios.get(`${API_BASE}/predict/${junctionId}`),
    ])
      .then(([histRes, forecastRes, predictRes]) => {
        setHistory(histRes.data);
        setForecast(forecastRes.data);
        setCurrent(predictRes.data);
      })
      .catch(() => setError("Couldn't load data for this junction."))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    loadJunctionData(selected);
  }, [selected, loadJunctionData]);

  const chartData = [
    ...history.map((h) => ({
      time: h.datetime.slice(5, 16),
      actual: h.vehicles,
      predicted: null,
    })),
    ...(forecast?.forecast || []).map((f) => ({
      time: f.datetime.slice(5, 16),
      actual: null,
      predicted: f.predicted_vehicles,
      risk: f.congestion_risk,
    })),
  ];

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <div className="header-title">
          <span className="signal-dot" />
          <h1>Traffic Forecast Console</h1>
        </div>
        <p className="header-sub">Smart City Junction Monitoring &amp; 24h Forecasting</p>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <div className="junction-tabs">
        {junctions.map((j) => (
          <button
            key={j.junction}
            className={`junction-tab ${selected === j.junction ? "active" : ""}`}
            onClick={() => setSelected(j.junction)}
          >
            Junction {j.junction}
            <span className="tab-model">{j.best_model}</span>
          </button>
        ))}
      </div>

      {loading ? (
        <div className="loading-state">Loading junction data…</div>
      ) : (
        <>
          <div className="stat-row">
            <div className="stat-card highlight">
              <span className="stat-label">Next Hour Forecast</span>
              <span className="stat-value">{current?.predicted_vehicles ?? "--"}</span>
              <span className="stat-unit">vehicles</span>
              {current && <RiskBadge risk={current.congestion_risk} />}
            </div>
            <div className="stat-card">
              <span className="stat-label">Congestion Threshold</span>
              <span className="stat-value">{current?.threshold ?? "--"}</span>
              <span className="stat-unit">vehicles / hr (p90)</span>
            </div>
            <div className="stat-card">
              <span className="stat-label">Model In Use</span>
              <span className="stat-value model-value">{current?.model_used ?? "--"}</span>
              <span className="stat-unit">best performer for this junction</span>
            </div>
          </div>

          <div className="chart-panel">
            <div className="chart-panel-header">
              <h2>Last 72h actuals → next 24h forecast</h2>
              <div className="legend">
                <span className="legend-item"><i className="dot actual" />Actual</span>
                <span className="legend-item"><i className="dot predicted" />Predicted</span>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={340}>
              <LineChart data={chartData} margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
                <CartesianGrid stroke="#2A3142" strokeDasharray="3 3" />
                <XAxis dataKey="time" tick={{ fill: "#8B95A8", fontSize: 11 }} interval={Math.floor(chartData.length / 8)} />
                <YAxis tick={{ fill: "#8B95A8", fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ background: "#1A1F2B", border: "1px solid #2A3142", borderRadius: 8 }}
                  labelStyle={{ color: "#8B95A8" }}
                />
                {current && (
                  <ReferenceLine
                    y={current.threshold}
                    stroke="#FF4D4F"
                    strokeDasharray="4 4"
                    label={{ value: "congestion threshold", fill: "#FF4D4F", fontSize: 11, position: "insideTopRight" }}
                  />
                )}
                <Line type="monotone" dataKey="actual" stroke="#5B9DFF" strokeWidth={2} dot={false} connectNulls={false} />
                <Line type="monotone" dataKey="predicted" stroke="#FFB020" strokeWidth={2} strokeDasharray="5 3" dot={false} connectNulls={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="forecast-strip">
            <h2>Next 24 hours</h2>
            <div className="hour-cards">
              {(forecast?.forecast || []).map((f) => (
                <div key={f.datetime} className="hour-card" style={{ "--risk-color": RISK_COLORS[f.congestion_risk] }}>
                  <span className="hour-time">{f.datetime.slice(11, 16)}</span>
                  <span className="hour-value">{Math.round(f.predicted_vehicles)}</span>
                  <span className="hour-risk-dot" />
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}