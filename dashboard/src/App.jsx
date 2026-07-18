import { useState, useEffect, useCallback } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Scatter, ScatterChart,
  ComposedChart, ZAxis,
} from "recharts";
import axios from "axios";
import "./App.css";

const API_BASE = "https://traffic-forecasting-project-production.up.railway.app";
const RISK_COLORS = {
  low: "#3DDC84",
  medium: "#FFB020",
  high: "#FF4D4F",
};

const DAY_LABELS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

function RiskBadge({ risk }) {
  return (
    <span className="risk-badge" style={{ "--risk-color": RISK_COLORS[risk] || "#888" }}>
      {risk}
    </span>
  );
}

// Custom dot renderer — shows a red triangle on anomaly points, nothing on normal points
function AnomalyDot(props) {
  const { cx, cy, payload } = props;
  if (!payload?.isAnomaly) return null;
  return (
    <g>
      <polygon
        points={`${cx},${cy - 10} ${cx - 7},${cy + 4} ${cx + 7},${cy + 4}`}
        fill="#FF4D4F"
        opacity={0.9}
      />
      <text x={cx} y={cy - 14} textAnchor="middle" fill="#FF4D4F" fontSize={10}>⚠</text>
    </g>
  );
}

// Custom tooltip — shows anomaly info when hovering over a flagged point
function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  return (
    <div style={{
      background: "#1A1F2B", border: "1px solid #2A3142", borderRadius: 8,
      padding: "8px 12px", fontSize: 12,
    }}>
      <p style={{ color: "#8B95A8", margin: 0 }}>{label}</p>
      {d?.actual != null && <p style={{ color: "#5B9DFF", margin: "4px 0 0" }}>Actual: {d.actual}</p>}
      {d?.predicted != null && <p style={{ color: "#FFB020", margin: "4px 0 0" }}>Predicted: {d.predicted}</p>}
      {d?.isAnomaly && (
        <p style={{ color: "#FF4D4F", margin: "6px 0 0", fontWeight: "bold" }}>
          ⚠ Anomaly — {d.anomalyType} (z={d.zScore})
        </p>
      )}
    </div>
  );
}

// Maps an average-vehicles value to a heatmap cell color (blue = low, red = high)
function heatmapColor(value, max) {
  if (max <= 0) return "#1A1F2B";
  const ratio = Math.min(value / max, 1);
  // interpolate from cool blue (#1E3A5F) to hot red (#FF4D4F)
  const r = Math.round(30 + ratio * (255 - 30));
  const g = Math.round(58 + ratio * (77 - 58));
  const b = Math.round(95 + ratio * (79 - 95));
  return `rgb(${r},${g},${b})`;
}

export default function App() {
  const [junctions, setJunctions] = useState([]);
  const [selected, setSelected] = useState(1);
  const [history, setHistory] = useState([]);
  const [forecast, setForecast] = useState(null);
  const [current, setCurrent] = useState(null);
  const [anomalies, setAnomalies] = useState([]);
  const [junctionAnomalyCounts, setJunctionAnomalyCounts] = useState({});
  const [heatmapData, setHeatmapData] = useState([]);
  const [cityOverview, setCityOverview] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    axios.get(`${API_BASE}/junctions`)
      .then((res) => {
        setJunctions(res.data);

        Promise.all(
          res.data.map((j) =>
            axios.get(`${API_BASE}/anomalies/${j.junction}`)
              .then((r) => ({ junction: j.junction, count: r.data.total_anomalies }))
              .catch(() => ({ junction: j.junction, count: 0 }))
          )
        ).then((results) => {
          const counts = {};
          results.forEach((r) => { counts[r.junction] = r.count; });
          setJunctionAnomalyCounts(counts);
        });

        Promise.all(
          res.data.map((j) =>
            axios.get(`${API_BASE}/predict/${j.junction}`)
              .then((r) => ({ junction: j.junction, ...r.data }))
              .catch(() => ({ junction: j.junction, predicted_vehicles: null, congestion_risk: "low" }))
          )
        ).then((results) => {
          const overview = {};
          results.forEach((r) => { overview[r.junction] = r; });
          setCityOverview(overview);
        });
      })
      .catch(() => setError("Can't reach the API. Is uvicorn running on port 8000?"));
  }, []);

  const loadJunctionData = useCallback((junctionId) => {
    setLoading(true);
    setError(null);
    Promise.all([
      axios.get(`${API_BASE}/history/${junctionId}?hours=72`),
      axios.get(`${API_BASE}/predict/${junctionId}/next24`),
      axios.get(`${API_BASE}/predict/${junctionId}`),
      axios.get(`${API_BASE}/anomalies/${junctionId}`),
      axios.get(`${API_BASE}/heatmap/${junctionId}`),
    ])
      .then(([histRes, forecastRes, predictRes, anomalyRes, heatmapRes]) => {
        setHistory(histRes.data);
        setForecast(forecastRes.data);
        setCurrent(predictRes.data);
        setAnomalies(anomalyRes.data.anomalies || []);
        setHeatmapData(heatmapRes.data.data || []);
      })
      .catch(() => setError("Couldn't load data for this junction."))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    loadJunctionData(selected);
  }, [selected, loadJunctionData]);

  // Build a set of anomaly datetimes for fast lookup
  const anomalyMap = {};
  anomalies.forEach((a) => {
    anomalyMap[a.datetime] = a;
  });

  const chartData = [
    ...history.map((h) => {
      const anomaly = anomalyMap[h.datetime];
      return {
        time: h.datetime.slice(5, 16),
        actual: h.vehicles,
        predicted: null,
        isAnomaly: !!anomaly,
        anomalyType: anomaly?.type || null,
        zScore: anomaly?.z_score || null,
      };
    }),
    ...(forecast?.forecast || []).map((f) => ({
      time: f.datetime.slice(5, 16),
      actual: null,
      predicted: f.predicted_vehicles,
      risk: f.congestion_risk,
      isAnomaly: false,
    })),
  ];

  const anomalyCount = anomalies.length;

  // Build a quick lookup grid for the heatmap: heatmapGrid[dayofweek][hour] = avg_vehicles
  const heatmapGrid = {};
  let heatmapMax = 0;
  heatmapData.forEach((d) => {
    if (!heatmapGrid[d.dayofweek]) heatmapGrid[d.dayofweek] = {};
    heatmapGrid[d.dayofweek][d.hour] = d.avg_vehicles;
    if (d.avg_vehicles > heatmapMax) heatmapMax = d.avg_vehicles;
  });

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

      {/* City Overview — all junctions at a glance */}
      {junctions.length > 0 && (
        <div className="city-overview">
          <h2>City Overview</h2>
          <div className="city-overview-cards">
            {junctions.map((j) => {
              const info = cityOverview[j.junction];
              const risk = info?.congestion_risk || "low";
              return (
                <button
                  key={j.junction}
                  className={`city-card ${selected === j.junction ? "active" : ""}`}
                  style={{ "--risk-color": RISK_COLORS[risk] }}
                  onClick={() => setSelected(j.junction)}
                >
                  <span className="city-card-header">
                    <span className="city-card-title">Junction {j.junction}</span>
                    <span className="city-card-risk-dot" />
                  </span>
                  <span className="city-card-value">
                    {info?.predicted_vehicles ?? "--"}
                    <span className="city-card-unit">vehicles/hr</span>
                  </span>
                  <span className="city-card-footer">
                    <span className="city-card-model">{j.best_model}</span>
                    {junctionAnomalyCounts[j.junction] > 0 && (
                      <span className="city-card-anomaly">⚠️ {junctionAnomalyCounts[j.junction]}</span>
                    )}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      )}

      <div className="junction-tabs">
        {junctions.map((j) => (
          <button
            key={j.junction}
            className={`junction-tab ${selected === j.junction ? "active" : ""}`}
            onClick={() => setSelected(j.junction)}
          >
            Junction {j.junction}
            <span className="tab-model">{j.best_model}</span>
            {junctionAnomalyCounts[j.junction] > 0 && (
              <span className="tab-anomaly-badge">⚠️ {junctionAnomalyCounts[j.junction]}</span>
            )}
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
            <div className="stat-card" style={{ borderColor: anomalyCount > 0 ? "#FF4D4F44" : undefined }}>
              <span className="stat-label">Anomalies (last 72h)</span>
              <span className="stat-value" style={{ color: anomalyCount > 0 ? "#FF4D4F" : "#3DDC84" }}>
                {anomalyCount > 0 ? `⚠ ${anomalyCount}` : "✓ 0"}
              </span>
              <span className="stat-unit">unusual spikes detected</span>
            </div>
          </div>

          <div className="chart-panel">
            <div className="chart-panel-header">
              <h2>Last 72h actuals → next 24h forecast</h2>
              <div className="legend">
                <span className="legend-item"><i className="dot actual" />Actual</span>
                <span className="legend-item"><i className="dot predicted" />Predicted</span>
                {anomalyCount > 0 && (
                  <span className="legend-item" style={{ color: "#FF4D4F" }}>▲ Anomaly</span>
                )}
              </div>
            </div>
            <ResponsiveContainer width="100%" height={340}>
              <ComposedChart data={chartData} margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
                <CartesianGrid stroke="#2A3142" strokeDasharray="3 3" />
                <XAxis dataKey="time" tick={{ fill: "#8B95A8", fontSize: 11 }} interval={Math.floor(chartData.length / 8)} />
                <YAxis tick={{ fill: "#8B95A8", fontSize: 11 }} />
                <Tooltip content={<CustomTooltip />} />
                {current && (
                  <ReferenceLine
                    y={current.threshold}
                    stroke="#FF4D4F"
                    strokeDasharray="4 4"
                    label={{ value: "congestion threshold", fill: "#FF4D4F", fontSize: 11, position: "insideTopRight" }}
                  />
                )}
                <Line
                  type="monotone"
                  dataKey="actual"
                  stroke="#5B9DFF"
                  strokeWidth={2}
                  dot={<AnomalyDot />}
                  activeDot={{ r: 4 }}
                  connectNulls={false}
                />
                <Line
                  type="monotone"
                  dataKey="predicted"
                  stroke="#FFB020"
                  strokeWidth={2}
                  strokeDasharray="5 3"
                  dot={false}
                  connectNulls={false}
                />
              </ComposedChart>
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

          {/* Weekly Traffic Heatmap */}
          {heatmapData.length > 0 && (
            <div className="heatmap-panel">
              <h2>Weekly Traffic Pattern</h2>
              <p className="heatmap-sub">Average vehicles per hour, by day of week — Junction {selected}</p>
              <div className="heatmap-grid-wrapper">
                <div className="heatmap-grid">
                  {/* Header row: hour labels */}
                  <div className="heatmap-corner" />
                  {Array.from({ length: 24 }, (_, h) => (
                    <div key={`h-${h}`} className="heatmap-hour-label">
                      {h % 3 === 0 ? h : ""}
                    </div>
                  ))}

                  {/* One row per day */}
                  {DAY_LABELS.map((label, dow) => (
                    <div className="heatmap-row" key={`row-${dow}`}>
                      <div className="heatmap-day-label">{label}</div>
                      <div className="heatmap-row-cells">
                        {Array.from({ length: 24 }, (_, h) => {
                          const val = heatmapGrid[dow]?.[h];
                          return (
                            <div
                              key={`cell-${dow}-${h}`}
                              className="heatmap-cell"
                              style={{ background: val != null ? heatmapColor(val, heatmapMax) : "#1A1F2B" }}
                              title={val != null ? `${label} ${h}:00 — avg ${val} vehicles` : ""}
                            />
                          );
                        })}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              <div className="heatmap-legend">
                <span>Low</span>
                <div className="heatmap-legend-gradient" />
                <span>High</span>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}