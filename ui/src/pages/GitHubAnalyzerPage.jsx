import { useState, useRef, useEffect } from "react";
import { useSearchParams, useNavigate } from "react-router-dom";
import { api, resolveImageUrl } from "../services/api";
import MarkdownRenderer from "../components/MarkdownRenderer";
import {
  GitBranch,
  Search,
  Star,
  GitFork,
  CheckCircle,
  AlertCircle,
  FileCode,
  Layers,
  ArrowRight,
  ExternalLink,
  ShieldCheck,
  FolderTree,
  Send,
  Square,
  Bot,
  User,
  Sparkles,
  Copy,
  Check,
  MessageSquare,
  Clock,
  RotateCcw,
} from "lucide-react";

const SAMPLE_REPOS = [
  {
    name: "langchain-ai/langgraph",
    url: "https://github.com/langchain-ai/langgraph",
    desc: "Multi-agent graph workflows",
  },
  {
    name: "fastapi/fastapi",
    url: "https://github.com/fastapi/fastapi",
    desc: "High performance Python web framework",
  },
  {
    name: "topics/weather-data",
    url: "https://github.com/topics/weather-data",
    desc: "GitHub Topic Discovery",
  },
];

export default function GitHubAnalyzerPage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const urlThreadId = searchParams.get("thread");

  const [repoUrl, setRepoUrl] = useState("");
  const [focusQuery, setFocusQuery] = useState("");
  const [branch, setBranch] = useState("");
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  // Embedded Repository Chat State
  const [messages, setMessages] = useState([]);
  const [chatInput, setChatInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentStage, setCurrentStage] = useState(null);
  const [threadId, setThreadId] = useState(
    () => urlThreadId || `gh_thread_${Math.random().toString(36).substring(2, 9)}`
  );
  const [copiedId, setCopiedId] = useState(null);

  const abortControllerRef = useRef(null);
  const messagesEndRef = useRef(null);

  // Restore thread history if thread ID changes
  useEffect(() => {
    let isSubscribed = true;
    const fetchHistory = async () => {
      if (!threadId) return;
      try {
        const threadData = await api.getThread(threadId);
        if (isSubscribed && threadData?.turns && threadData.turns.length > 0) {
          const restored = [];
          threadData.turns.forEach((turn, idx) => {
            restored.push({
              id: `user_${turn.turn_id || idx}`,
              role: "user",
              content: turn.user_query,
            });
            restored.push({
              id: `ai_${turn.turn_id || idx}`,
              role: "assistant",
              content: turn.agent_response,
              route: turn.route || "github_agent",
            });
          });
          setMessages(restored);
        }
      } catch (err) {
        console.debug("No previous thread history for this session:", err);
      }
    };
    fetchHistory();
    return () => {
      isSubscribed = false;
    };
  }, [threadId]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, currentStage]);

  const handleAnalyze = async (e, urlOverride) => {
    if (e) e.preventDefault();
    const targetUrl = (urlOverride || repoUrl).trim();
    if (!targetUrl) return;

    if (urlOverride) setRepoUrl(urlOverride);
    setAnalyzing(true);
    setResult(null);
    setError("");

    try {
      const res = await api.analyzeGitHub({
        repo_url: targetUrl,
        focus_query: focusQuery.trim() || undefined,
        branch: branch.trim() || undefined,
        thread_id: threadId,
      });

      if (res.error) {
        setError(res.error);
      } else {
        setResult(res);
        setMessages([
          {
            id: "welcome_gh",
            role: "assistant",
            content: `### 🔍 Repository Inspection Ready\n\nI have loaded architectural metadata for **\`${res.repo_info?.full_name || targetUrl}\`**.\n\nYou can ask any specific questions about execution flow, key files, authentication, or dependencies below!`,
            route: "github_agent",
          },
        ]);
      }
    } catch (err) {
      setError(err.message || "Repository analysis failed.");
    } finally {
      setAnalyzing(false);
    }
  };

  const handleSendChat = async (queryText) => {
    const query = (queryText || chatInput).trim();
    if (!query || isStreaming) return;

    setChatInput("");
    const userMsgId = `user_${Date.now()}`;
    const assistantMsgId = `ai_${Date.now()}`;

    setMessages((prev) => [
      ...prev,
      { id: userMsgId, role: "user", content: query },
      { id: assistantMsgId, role: "assistant", content: "", route: "evaluating" },
    ]);

    setIsStreaming(true);
    setCurrentStage("Inspecting repository code and architecture...");

    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      await api.streamChatSSE({
        message: `${query} (in context of GitHub repository ${repoUrl})`,
        thread_id: threadId,
        signal: controller.signal,
        onStatus: (statusData) => {
          setCurrentStage(statusData.message || `Pipeline: ${statusData.route}`);
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId
                ? { ...m, route: statusData.route, confidence: statusData.confidence }
                : m
            )
          );
        },
        onToken: (token) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId ? { ...m, content: m.content + token } : m
            )
          );
        },
        onDone: (doneData) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId
                ? {
                    ...m,
                    content: doneData.answer || m.content,
                    route: doneData.route || m.route,
                    latency_ms: doneData.latency_ms,
                    sources: doneData.sources || [],
                  }
                : m
            )
          );
          setIsStreaming(false);
          setCurrentStage(null);
        },
        onError: (errData) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId
                ? {
                    ...m,
                    content: m.content || `⚠️ Error: ${errData.message || "Failed to analyze repository."}`,
                    isError: true,
                  }
                : m
            )
          );
          setIsStreaming(false);
          setCurrentStage(null);
        },
      });
    } catch (err) {
      if (err.name !== "AbortError") {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsgId
              ? { ...m, content: m.content || `⚠️ Network Error: ${err.message}`, isError: true }
              : m
          )
        );
      }
    } finally {
      setIsStreaming(false);
      setCurrentStage(null);
      abortControllerRef.current = null;
    }
  };

  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setIsStreaming(false);
      setCurrentStage(null);
    }
  };

  const handleCopy = (id, text) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  return (
    <div className="page-github">
      <div className="page-header">
        <h1>GitHub Repository & Topic Analyzer</h1>
        <p>Read-only deep architectural discovery, synchronization inspection & weakness audits</p>
      </div>

      <div className="panel-card max-width-card" style={{ marginBottom: "1.75rem" }}>
        {/* Sample Quick Pickers */}
        <div className="sample-repos-bar">
          <span className="sample-label">Quick Samples:</span>
          <div className="sample-list">
            {SAMPLE_REPOS.map((sample, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => handleAnalyze(null, sample.url)}
                className="sample-pill"
                disabled={analyzing}
              >
                <GitBranch size={13} />
                <span>{sample.name}</span>
              </button>
            ))}
          </div>
        </div>

        <form onSubmit={handleAnalyze} className="github-form">
          <div className="form-group">
            <label htmlFor="repoUrl">GitHub URL (Repository or Topic)</label>
            <div className="input-with-icon">
              <GitBranch className="input-icon" size={18} />
              <input
                id="repoUrl"
                type="text"
                value={repoUrl}
                onChange={(e) => setRepoUrl(e.target.value)}
                placeholder="https://github.com/owner/repo or https://github.com/topics/..."
                disabled={analyzing}
                required
              />
            </div>
          </div>

          <div className="grid-2-col">
            <div className="form-group">
              <label htmlFor="focusQuery">Focus Inspection Area (Optional)</label>
              <input
                id="focusQuery"
                type="text"
                value={focusQuery}
                onChange={(e) => setFocusQuery(e.target.value)}
                placeholder="e.g. Inspect architecture, auth, or rate limits"
                disabled={analyzing}
              />
            </div>

            <div className="form-group">
              <label htmlFor="branch">Target Branch (Optional)</label>
              <input
                id="branch"
                type="text"
                value={branch}
                onChange={(e) => setBranch(e.target.value)}
                placeholder="main / master"
                disabled={analyzing}
              />
            </div>
          </div>

          {error && (
            <div className="auth-alert error">
              <AlertCircle size={18} />
              <span>{error}</span>
            </div>
          )}

          <button type="submit" className="btn-primary full-width" disabled={!repoUrl.trim() || analyzing}>
            {analyzing ? "Running Read-Only Architecture Analysis..." : "Analyze Repository"}
            {!analyzing && <ArrowRight size={16} />}
          </button>
        </form>
      </div>

      {result && (
        <div className="analysis-results">
          {result.repo_info && (
            <div className="repo-meta-card">
              <div className="repo-title-row">
                <div className="repo-title-group">
                  <GitBranch size={22} className="icon-accent" />
                  <h2>{result.repo_info.full_name || repoUrl}</h2>
                </div>
                <div className="repo-badges">
                  {result.repo_info.stargazers_count !== undefined && (
                    <span className="badge">
                      <Star size={14} /> {result.repo_info.stargazers_count.toLocaleString()} Stars
                    </span>
                  )}
                  {result.repo_info.forks_count !== undefined && (
                    <span className="badge">
                      <GitFork size={14} /> {result.repo_info.forks_count.toLocaleString()} Forks
                    </span>
                  )}
                  {result.repo_info.language && (
                    <span className="badge language-badge">
                      {result.repo_info.language}
                    </span>
                  )}
                </div>
              </div>
              <p className="repo-desc">{result.repo_info.description || "No repository description provided."}</p>
            </div>
          )}

          {result.markdown_report && (
            <div className="panel-card report-panel" style={{ marginBottom: "1.75rem" }}>
              <div className="panel-header">
                <FileCode className="icon-accent" size={22} />
                <h2>Engineering Architecture Report</h2>
              </div>
              <div className="report-markdown-content">
                <MarkdownRenderer content={result.markdown_report} />
              </div>
            </div>
          )}

          {/* Interactive GitHub Follow-Up Chat */}
          <div className="panel-card embedded-chat-panel">
            <div className="panel-header flex-between">
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <MessageSquare className="icon-accent" size={22} />
                <h2>Ask Follow-Up Questions About This Codebase</h2>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <span className="badge">Session: <code>{threadId}</code></span>
                <button
                  type="button"
                  onClick={() => navigate(`/chat?thread=${threadId}`)}
                  className="btn-secondary btn-sm"
                  title="Open full conversation history in primary chat workspace"
                >
                  <MessageSquare size={13} />
                  <span>Full Chat</span>
                  <ArrowRight size={13} />
                </button>
              </div>
            </div>

            {/* Quick Starters */}
            <div className="quick-prompts-bar">
              <button
                type="button"
                onClick={() => handleSendChat("Explain the main entry point and execution flow in detail.")}
                className="sample-pill"
                disabled={isStreaming}
              >
                <FolderTree size={13} /> Main Entry Point
              </button>
              <button
                type="button"
                onClick={() => handleSendChat("What are the key dependencies and external APIs integrated?")}
                className="sample-pill"
                disabled={isStreaming}
              >
                <Layers size={13} /> Dependencies & APIs
              </button>
              <button
                type="button"
                onClick={() => handleSendChat("Are there any potential security risks, unhandled errors, or edge cases?")}
                className="sample-pill"
                disabled={isStreaming}
              >
                <ShieldCheck size={13} /> Security & Edge Cases
              </button>
            </div>

            {/* Messages Feed */}
            <div className="embedded-messages-feed">
              {messages.map((msg) => (
                <div key={msg.id} className={`pro-message-row ${msg.role}`}>
                  <div className={`pro-avatar ${msg.role}`}>
                    {msg.role === "user" ? <User size={16} /> : <Bot size={16} />}
                  </div>

                  <div className="pro-message-body" style={{ width: "100%" }}>
                    {msg.role === "assistant" && msg.route && msg.route !== "system" && (
                      <div className="pro-telemetry-header">
                        <span className="telemetry-pill">Route: {msg.route}</span>
                        {msg.latency_ms && <span className="telemetry-latency">{msg.latency_ms} ms</span>}
                      </div>
                    )}

                    <div className={`pro-bubble ${msg.role}`}>
                      <MarkdownRenderer content={msg.content || (isStreaming ? "Analyzing repository..." : "")} />
                    </div>

                    {msg.content && msg.role === "assistant" && (
                      <div className="pro-message-actions" style={{ marginTop: "0.25rem" }}>
                        <button onClick={() => handleCopy(msg.id, msg.content)} className="btn-action-sm">
                          {copiedId === msg.id ? <Check size={12} className="icon-success" /> : <Copy size={12} />}
                          <span>{copiedId === msg.id ? "Copied" : "Copy"}</span>
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {currentStage && (
                <div className="pro-stage-banner" style={{ margin: "0.5rem 0" }}>
                  <div className="stage-pulse-ring" />
                  <span className="stage-text">{currentStage}</span>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Embedded Input Bar */}
            <form
              onSubmit={(e) => {
                e.preventDefault();
                handleSendChat();
              }}
              className="embedded-input-form"
            >
              <input
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                placeholder="Ask about functions, logic, architecture, or performance in this repo..."
                disabled={isStreaming}
              />
              {isStreaming ? (
                <button type="button" onClick={handleStop} className="btn-stop-stream">
                  <Square size={15} /> Stop
                </button>
              ) : (
                <button type="submit" disabled={!chatInput.trim()} className="btn-send-pro">
                  <Send size={15} /> Send
                </button>
              )}
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
