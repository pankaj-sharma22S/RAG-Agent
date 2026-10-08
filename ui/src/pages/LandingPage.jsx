import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import {
  Sparkles,
  ArrowRight,
  MessageSquare,
  FileText,
  Video,
  GitBranch,
  ShieldCheck,
  Zap,
  Layers,
  Database,
  CheckCircle,
  Cpu,
  Lock,
} from "lucide-react";

export default function LandingPage() {
  const { user, continueAsGuest } = useAuth();
  const navigate = useNavigate();

  const handleQuickLaunch = () => {
    if (!user) {
      continueAsGuest("Demo Explorer");
    }
    navigate("/dashboard");
  };

  return (
    <div className="landing-page">
      {/* Navigation Bar */}
      <header className="landing-nav">
        <div className="landing-brand">
          <div className="brand-badge">
            <Sparkles className="icon-accent" size={24} />
          </div>
          <span className="brand-title">Insight AI</span>
        </div>

        <nav className="landing-nav-links">
          <a href="#features" className="nav-link-landing">Features</a>
          <a href="#architecture" className="nav-link-landing">Architecture</a>
          <a href="#diagnostics" className="nav-link-landing">Diagnostics</a>
        </nav>

        <div className="landing-nav-actions">
          {user ? (
            <Link to="/dashboard" className="btn-primary btn-sm">
              Go to Dashboard <ArrowRight size={14} />
            </Link>
          ) : (
            <>
              <Link to="/login" className="btn-secondary btn-sm">
                Sign In
              </Link>
              <button onClick={handleQuickLaunch} className="btn-primary btn-sm">
                Launch Demo <Sparkles size={14} />
              </button>
            </>
          )}
        </div>
      </header>

      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-badge">
          <span className="hero-badge-dot" />
          <span>Production Multimodal Knowledge AI & Agentic RAG</span>
        </div>

        <h1 className="hero-title">
          Intelligent Knowledge Engine for <span className="gradient-text">Documents, Code & Media</span>
        </h1>

        <p className="hero-subtitle">
          Next-generation knowledge orchestration powered by deterministic pre-RAG routing,
          hybrid Dense + BM25 retrieval, Cross-Encoder reranking, and read-only GitHub analysis.
        </p>

        <div className="hero-cta-group">
          <button onClick={handleQuickLaunch} className="btn-primary btn-lg">
            Launch Interactive Workspace <ArrowRight size={18} />
          </button>
          <Link to="/login" className="btn-secondary btn-lg">
            Sign In with Supabase
          </Link>
        </div>

        {/* Live Metrics Strip */}
        <div className="hero-metrics-strip">
          <div className="metric-strip-item">
            <span className="strip-num">&lt; 0.05 ms</span>
            <span className="strip-label">Deterministic Pre-RAG Routing</span>
          </div>
          <div className="metric-strip-item">
            <span className="strip-num">768 / 384 Dim</span>
            <span className="strip-label">Dual Text & Vision Embeddings</span>
          </div>
          <div className="metric-strip-item">
            <span className="strip-num">3-Tier Cache</span>
            <span className="strip-label">Embedding, Score & Response Caching</span>
          </div>
          <div className="metric-strip-item">
            <span className="strip-num">100% Guarded</span>
            <span className="strip-label">PII, Secret & Prompt Injection Defense</span>
          </div>
        </div>
      </section>

      {/* Capabilities Section */}
      <section id="features" className="section-container">
        <div className="section-header-center">
          <span className="section-tag">POWERFUL AGENT WORKSPACES</span>
          <h2>Engineered for Enterprise Knowledge Intelligence</h2>
          <p>Execute document analysis, YouTube transcript extractions, and deep repository architecture audits.</p>
        </div>

        <div className="features-grid">
          <div className="feature-card highlighted">
            <div className="feature-header">
              <div className="feature-icon">
                <MessageSquare size={26} />
              </div>
              <h3>Agentic Chat & Streaming</h3>
            </div>
            <p>
              Real-time Server-Sent Events (SSE) token streaming with sub-millisecond route classification,
              CRAG quality assessment, and grounded evidence attribution.
            </p>
            <div className="card-tag">SSE Streaming + LangGraph</div>
          </div>

          <div className="feature-card highlighted">
            <div className="feature-header">
              <div className="feature-icon">
                <FileText size={26} />
              </div>
              <h3>Multimodal Document RAG</h3>
            </div>
            <p>
              Index PDFs, Word documents, Excel spreadsheets, PowerPoint presentations, OCR images, and
              audio/video transcripts into persistent Chroma vector DB.
            </p>
            <div className="card-tag">PaddleOCR + SigLIP + BGE</div>
          </div>

          <div className="feature-card highlighted">
            <div className="feature-header">
              <div className="feature-icon">
                <Video size={26} />
              </div>
              <h3>YouTube & Web Ingestion</h3>
            </div>
            <p>
              Extract and index full video transcripts and public web articles with hierarchical
              summarization (Quick, Standard, and Deep 7-section summaries).
            </p>
            <div className="card-tag">Transcript API + Web Scraping</div>
          </div>

          <div className="feature-card highlighted">
            <div className="feature-header">
              <div className="feature-icon">
                <GitBranch size={26} />
              </div>
              <h3>GitHub Repository Analyzer</h3>
            </div>
            <p>
              Read-only GitHub discovery engine parsing directory trees, README docs, dependency files,
              and generating architectural synchronization reports.
            </p>
            <div className="card-tag">Topic Discovery + Architecture Reports</div>
          </div>
        </div>
      </section>

      {/* Architecture Flow Section */}
      <section id="architecture" className="section-container">
        <div className="section-header-center">
          <span className="section-tag">EXECUTION PIPELINE</span>
          <h2>High Performance Adaptive RAG Graph</h2>
          <p>Every query follows an optimized zero-redundancy path designed for low latency.</p>
        </div>

        <div className="architecture-diagram-card">
          <div className="flow-step">
            <div className="step-num">01</div>
            <h4>Input Guard & PII Scanner</h4>
            <p>Sanitizes queries, masks credentials, and encapsulates untrusted external text.</p>
          </div>
          <div className="flow-arrow">→</div>
          <div className="flow-step">
            <div className="step-num">02</div>
            <h4>Deterministic Router</h4>
            <p>Sub-millisecond routing bypassing RAG for chit-chat, math, or general knowledge.</p>
          </div>
          <div className="flow-arrow">→</div>
          <div className="flow-step">
            <div className="step-num">03</div>
            <h4>Hybrid Retrieval & Reranking</h4>
            <p>Parallel Dense (Chroma) + Sparse (BM25) with MS-Marco Cross-Encoder scoring.</p>
          </div>
          <div className="flow-arrow">→</div>
          <div className="flow-step">
            <div className="step-num">04</div>
            <h4>CRAG Quality & SSE Output</h4>
            <p>Self-RAG evaluation, single-loop refinement, and token-by-token streaming.</p>
          </div>
        </div>
      </section>

      {/* CTA Footer Banner */}
      <section className="cta-banner">
        <h2>Ready to Explore Insight AI?</h2>
        <p>Launch the interactive workspace and start querying documents, analyzing repos, and chatting in real-time.</p>
        <button onClick={handleQuickLaunch} className="btn-primary btn-lg">
          Launch Workspace Now <ArrowRight size={18} />
        </button>
      </section>

      <footer className="landing-footer">
        <div className="footer-content">
          <span>Insight AI Knowledge Platform v1.0.0</span>
          <span>FastAPI Backend • Supabase Auth • Chroma Vector DB</span>
        </div>
      </footer>
    </div>
  );
}
