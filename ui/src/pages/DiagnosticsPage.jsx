import { useState, useEffect } from "react";
import { api } from "../services/api";
import {
  Activity,
  CheckCircle,
  AlertTriangle,
  Database,
  Cpu,
  RefreshCw,
  Zap,
  DollarSign,
} from "lucide-react";

export default function DiagnosticsPage() {
  const [health, setHealth] = useState(null);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchDiagnostics = async () => {
    setLoading(true);
    try {
      const [hData, sData] = await Promise.all([
        api.getHealth().catch(() => null),
        api.getGatewayStats().catch(() => null),
      ]);
      setHealth(hData);
      setStats(sData);
    } catch (err) {
      console.error("Failed to load diagnostics:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDiagnostics();
  }, []);

  return (
    <div className="page-diagnostics">
      <div className="page-header flex-between">
        <div>
          <h1>System Diagnostics & Gateway</h1>
          <p>Real-time infrastructure health, vector store index status, and cache telemetry</p>
        </div>
        <button onClick={fetchDiagnostics} className="btn-secondary" disabled={loading}>
          <RefreshCw size={16} className={loading ? "spin" : ""} />
          <span>Refresh</span>
        </button>
      </div>

      {/* Services Health Grid */}
      <h2 className="section-title">Subsystem Health</h2>
      <div className="metrics-grid">
        {health?.services &&
          Object.entries(health.services).map(([key, svc]) => (
            <div key={key} className="panel-card">
              <div className="service-card-header">
                <span className="service-name">{key.replace("_", " ").toUpperCase()}</span>
                <span className={`status-pill ${svc.status}`}>
                  {svc.status === "healthy" ? <CheckCircle size={14} /> : <AlertTriangle size={14} />}
                  {svc.status}
                </span>
              </div>
              <div className="service-latency">{svc.latency_ms} ms response latency</div>
              <p className="service-detail">{svc.detail || "Operating normally."}</p>
            </div>
          ))}
      </div>

      {/* Gateway Cache Telemetry */}
      {stats?.cache && (
        <div className="panel-card" style={{ marginTop: "1.5rem" }}>
          <div className="panel-header">
            <Zap className="icon-accent" size={22} />
            <h2>LLM Response Cache Metrics</h2>
          </div>
          <div className="stats-inline-grid">
            <div className="stat-box">
              <span className="stat-num">{stats.cache.entries_count || 0}</span>
              <span className="stat-label">Cached Queries</span>
            </div>
            <div className="stat-box">
              <span className="stat-num">{stats.cache.hits || 0}</span>
              <span className="stat-label">Cache Hits</span>
            </div>
            <div className="stat-box">
              <span className="stat-num">{stats.cache.misses || 0}</span>
              <span className="stat-label">Cache Misses</span>
            </div>
            <div className="stat-box">
              <span className="stat-num">{stats.cache.hit_ratio_percent || 0}%</span>
              <span className="stat-label">Hit Ratio</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
