import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { api } from "../services/api";
import {
  MessageSquare,
  FileText,
  Video,
  GitBranch,
  Activity,
  ArrowRight,
  ShieldCheck,
  Zap,
  Layers,
  Database,
} from "lucide-react";

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let isMounted = true;
    Promise.all([api.getGatewayStats().catch(() => null), api.getHealth().catch(() => null)]).then(
      ([statsData, healthData]) => {
        if (isMounted) {
          setStats(statsData);
          setHealth(healthData);
          setLoading(false);
        }
      }
    );
    return () => {
      isMounted = false;
    };
  }, []);

  return (
    <div className="page-dashboard">
      <div className="page-header">
        <div>
          <h1>Insight AI Knowledge Center</h1>
          <p>Adaptive Multimodal RAG, GitHub Intelligence & Memory Orchestration</p>
        </div>
      </div>

      {/* Metrics Row */}
      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-icon accent">
            <Zap size={22} />
          </div>
          <div className="metric-info">
            <span className="metric-label">Cache Hit Ratio</span>
            <span className="metric-value">
              {stats?.cache?.hit_ratio_percent ? `${stats.cache.hit_ratio_percent}%` : "0.0%"}
            </span>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-icon success">
            <Database size={22} />
          </div>
          <div className="metric-info">
            <span className="metric-label">Vector Store</span>
            <span className="metric-value">
              {health?.services?.vector_store?.status === "healthy" ? "Active" : "Ready"}
            </span>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-icon info">
            <Layers size={22} />
          </div>
          <div className="metric-info">
            <span className="metric-label">LLM Provider</span>
            <span className="metric-value">Gemini / Ollama</span>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-icon warning">
            <ShieldCheck size={22} />
          </div>
          <div className="metric-info">
            <span className="metric-label">Security Boundary</span>
            <span className="metric-value">PII & Guardrails Active</span>
          </div>
        </div>
      </div>

      {/* Capabilities / Feature Cards */}
      <h2 className="section-title">AI Workspaces</h2>
      <div className="features-grid">
        <div className="feature-card">
          <div className="feature-header">
            <div className="feature-icon">
              <MessageSquare size={24} />
            </div>
            <h3>Agentic Chat & Streaming</h3>
          </div>
          <p>
            Deterministic query routing with sub-millisecond chit-chat bypass, CRAG quality
            evaluation, and Server-Sent Events (SSE) token streaming.
          </p>
          <Link to="/chat" className="btn-secondary">
            Open Chat <ArrowRight size={16} />
          </Link>
        </div>

        <div className="feature-card">
          <div className="feature-header">
            <div className="feature-icon">
              <FileText size={24} />
            </div>
            <h3>Multimodal Document RAG</h3>
          </div>
          <p>
            Upload PDF, Word, Excel, PPTX, image OCR, audio, or video files into persistent Chroma
            embeddings with Cross-Encoder reranking.
          </p>
          <Link to="/documents" className="btn-secondary">
            Upload Documents <ArrowRight size={16} />
          </Link>
        </div>

        <div className="feature-card">
          <div className="feature-header">
            <div className="feature-icon">
              <Video size={24} />
            </div>
            <h3>YouTube & Web Ingestion</h3>
          </div>
          <p>
            Index full video transcripts or public web articles and generate hierarchical summaries
            (quick, standard, or deep).
          </p>
          <Link to="/ingest-media" className="btn-secondary">
            Ingest URLs <ArrowRight size={16} />
          </Link>
        </div>

        <div className="feature-card">
          <div className="feature-header">
            <div className="feature-icon">
              <GitBranch size={24} />
            </div>
            <h3>GitHub Repository Analyzer</h3>
          </div>
          <p>
            Inspect GitHub repositories or topics, analyze architecture, detect weaknesses, and
            generate evidence-based engineering reports.
          </p>
          <Link to="/github" className="btn-secondary">
            Analyze Repos <ArrowRight size={16} />
          </Link>
        </div>
      </div>
    </div>
  );
}
