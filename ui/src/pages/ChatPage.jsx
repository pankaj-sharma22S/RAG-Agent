import { useState, useRef, useEffect } from "react";
import { Link } from "react-router-dom";
import { api } from "../services/api";
import MarkdownRenderer from "../components/MarkdownRenderer";
import {
  Send,
  Square,
  Sparkles,
  Bot,
  User,
  Shield,
  Layers,
  ArrowRight,
  Info,
  Clock,
  CheckCircle,
  Copy,
  Check,
  RotateCcw,
  Zap,
  Globe,
  FileText,
  GitBranch,
  Paperclip,
  ChevronDown,
  ChevronUp,
  Cpu,
} from "lucide-react";

const STARTER_PROMPTS = [
  {
    category: "Document Intelligence",
    label: "Summarize Uploaded PDF",
    query: "Give me a detailed hierarchical summary of the uploaded document.",
    icon: FileText,
  },
  {
    category: "Banker's Algorithm",
    label: "Explain Deadlock Avoidance",
    query: "How does the Banker's Algorithm calculate safe states in operating systems?",
    icon: Shield,
  },
  {
    category: "GitHub Architecture",
    label: "Analyze LangGraph Workflow",
    query: "Analyze the multi-agent architecture of https://github.com/langchain-ai/langgraph",
    icon: GitBranch,
  },
  {
    category: "Real-time Discovery",
    label: "Current Tokyo Weather",
    query: "What is the current weather forecast in Tokyo?",
    icon: Globe,
  },
];

export default function ChatPage() {
  const [messages, setMessages] = useState([
    {
      id: "welcome",
      role: "assistant",
      content:
        "### Welcome to Insight AI Knowledge Assistant\n\nI am your unified enterprise knowledge assistant connected to **multimodal document retrieval**, **read-only GitHub repository audits**, **YouTube video transcripts**, and **real-time web search**.\n\n* **Deterministic pre-RAG routing**: Evaluates queries in `< 0.05 ms` to bypass unnecessary RAG loops.\n* **Hybrid dense + sparse retrieval**: Chroma vector search with BM25 keyword matching and Cross-Encoder reranking.\n* **Security boundary**: Automatic PII masking, credential shielding, and injection guards.\n\nType a question below or pick a quick starter to begin.",
      route: "system",
      latency_ms: 0,
    },
  ]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentStage, setCurrentStage] = useState(null);
  const [threadId, setThreadId] = useState(
    () => `thread_${Math.random().toString(36).substring(2, 9)}`
  );
  const [copiedId, setCopiedId] = useState(null);
  const [copiedThread, setCopiedThread] = useState(false);
  const [expandedSources, setExpandedSources] = useState({});

  const abortControllerRef = useRef(null);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, currentStage]);

  // Auto-resize textarea as user types
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 180)}px`;
    }
  }, [input]);

  const handleCopy = (id, text) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const handleCopyThreadId = () => {
    navigator.clipboard.writeText(threadId);
    setCopiedThread(true);
    setTimeout(() => setCopiedThread(false), 2000);
  };

  const toggleSources = (msgId) => {
    setExpandedSources((prev) => ({
      ...prev,
      [msgId]: !prev[msgId],
    }));
  };

  const handleSend = async (textToSend) => {
    const query = (textToSend || input).trim();
    if (!query || isStreaming) return;

    setInput("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }

    const userMsgId = `user_${Date.now()}`;
    const assistantMsgId = `ai_${Date.now()}`;

    setMessages((prev) => [
      ...prev,
      { id: userMsgId, role: "user", content: query, timestamp: new Date().toLocaleTimeString() },
      {
        id: assistantMsgId,
        role: "assistant",
        content: "",
        route: "evaluating",
        sources: [],
        timestamp: new Date().toLocaleTimeString(),
      },
    ]);

    setIsStreaming(true);
    setCurrentStage("Evaluating query intent & deterministic routing (<0.05ms)...");

    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      await api.streamChatSSE({
        message: query,
        thread_id: threadId,
        signal: controller.signal,
        onStatus: (statusData) => {
          setCurrentStage(statusData.message || `Executing pipeline: ${statusData.route}`);
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
                    sources: doneData.sources || m.sources,
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
                    content:
                      m.content ||
                      `⚠️ **Pipeline Error:** ${errData.message || "Failed to process request."}`,
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
              ? {
                  ...m,
                  content:
                    m.content ||
                    `⚠️ **Connection Error:** Could not stream response from FastAPI backend (${err.message}).`,
                  isError: true,
                }
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

  const handleNewChat = () => {
    setThreadId(`thread_${Math.random().toString(36).substring(2, 9)}`);
    setMessages([
      {
        id: "welcome",
        role: "assistant",
        content: "Started a fresh conversation session. How can I assist your workflow today?",
        route: "system",
        latency_ms: 0,
      },
    ]);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="pro-chat-wrapper">
      {/* Top Session Control Deck */}
      <div className="pro-chat-topbar">
        <div className="topbar-left">
          <div className="model-chip">
            <Cpu size={15} className="chip-icon" />
            <span className="model-name">Gemini 2.5 Flash</span>
            <span className="chip-divider">•</span>
            <span className="routing-tag">Sub-ms Adaptive RAG</span>
          </div>

          <button
            onClick={handleCopyThreadId}
            className="session-tag-btn"
            title="Click to copy Session Thread ID"
          >
            <span className="tag-label">Thread:</span>
            <code>{threadId}</code>
            {copiedThread ? <Check size={12} className="icon-success" /> : <Copy size={12} />}
          </button>
        </div>

        <div className="topbar-right">
          <Link to="/documents" className="topbar-link-btn" title="Manage Document Vector Store">
            <Paperclip size={14} /> Upload Knowledge
          </Link>
          <button
            onClick={handleNewChat}
            className="btn-secondary btn-sm"
            title="Clear context and start fresh thread"
          >
            <RotateCcw size={14} /> New Chat
          </button>
        </div>
      </div>

      {/* Main Conversation Stream */}
      <div className="pro-chat-feed">
        <div className="feed-container">
          {messages.map((msg) => (
            <div key={msg.id} className={`pro-message-row ${msg.role}`}>
              <div className={`pro-avatar ${msg.role}`}>
                {msg.role === "user" ? <User size={18} /> : <Bot size={18} />}
              </div>

              <div className="pro-message-body">
                {/* Route & Telemetry Header for AI */}
                {msg.role === "assistant" && msg.route && msg.route !== "system" && (
                  <div className="pro-telemetry-header">
                    <span className="telemetry-pill">
                      <span className="pill-dot" />
                      <span className="pill-route">Route: {msg.route}</span>
                    </span>
                    {msg.latency_ms !== undefined && msg.latency_ms > 0 && (
                      <span className="telemetry-latency">
                        <Clock size={12} /> {msg.latency_ms} ms
                      </span>
                    )}
                    {msg.confidence !== undefined && (
                      <span className="telemetry-conf">
                        Confidence: {Math.round(msg.confidence * 100)}%
                      </span>
                    )}
                  </div>
                )}

                {/* Message Content with Markdown & Code formatting */}
                <div className={`pro-bubble ${msg.role} ${msg.isError ? "error-bubble" : ""}`}>
                  {msg.role === "user" ? (
                    <div className="user-text">{msg.content}</div>
                  ) : (
                    <MarkdownRenderer
                      content={
                        msg.content ||
                        (isStreaming && msg.id.startsWith("ai_")
                          ? "Generating response..."
                          : "")
                      }
                    />
                  )}

                  {/* Grounded Citations Drawer */}
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="pro-sources-drawer">
                      <button
                        type="button"
                        onClick={() => toggleSources(msg.id)}
                        className="sources-toggle-btn"
                      >
                        <Info size={14} />
                        <span>Grounded Citations & Evidence ({msg.sources.length})</span>
                        {expandedSources[msg.id] ? (
                          <ChevronUp size={14} />
                        ) : (
                          <ChevronDown size={14} />
                        )}
                      </button>

                      {expandedSources[msg.id] && (
                        <div className="sources-expanded-grid">
                          {msg.sources.map((src, sIdx) => (
                            <div key={sIdx} className="source-citation-card">
                              <div className="citation-header">
                                <span className="citation-num">#{sIdx + 1}</span>
                                <span className="citation-filename">
                                  {src.source || src.filename || "Document Chunk"}
                                </span>
                              </div>
                              {src.rerank_score !== undefined && (
                                <div className="citation-score">
                                  Rerank Score: {src.rerank_score.toFixed(3)}
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {/* Bottom Actions Bar */}
                {msg.content && msg.role === "assistant" && (
                  <div className="pro-message-actions">
                    <button
                      onClick={() => handleCopy(msg.id, msg.content)}
                      className="btn-action-sm"
                      title="Copy response markdown"
                    >
                      {copiedId === msg.id ? (
                        <Check size={13} className="icon-success" />
                      ) : (
                        <Copy size={13} />
                      )}
                      <span>{copiedId === msg.id ? "Copied" : "Copy"}</span>
                    </button>
                    {msg.timestamp && (
                      <span className="msg-timestamp">{msg.timestamp}</span>
                    )}
                  </div>
                )}
              </div>
            </div>
          ))}

          {/* Active Pipeline Stage Animation */}
          {currentStage && (
            <div className="pro-stage-banner">
              <div className="stage-pulse-ring" />
              <span className="stage-text">{currentStage}</span>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Quick Starters (shown when thread is fresh) */}
      {messages.length <= 2 && (
        <div className="pro-starters-section">
          <div className="starters-container">
            <span className="starters-heading">Suggested Knowledge Workflows:</span>
            <div className="starters-grid">
              {STARTER_PROMPTS.map((p, idx) => {
                const Icon = p.icon;
                return (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => handleSend(p.query)}
                    className="starter-card"
                    disabled={isStreaming}
                  >
                    <div className="starter-header">
                      <div className="starter-icon-box">
                        <Icon size={16} />
                      </div>
                      <span className="starter-category">{p.category}</span>
                    </div>
                    <span className="starter-label">{p.label}</span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Floating Modern Input Deck */}
      <div className="pro-input-dock">
        <div className="dock-container">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleSend();
            }}
            className="pro-input-form"
          >
            <div className="input-textarea-wrapper">
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask about documents, analyze GitHub repos, summarize videos... (Press Enter to send, Shift+Enter for newline)"
                rows={1}
                disabled={isStreaming}
              />
            </div>

            <div className="input-controls-row">
              <div className="controls-left">
                <Link
                  to="/documents"
                  className="btn-attach"
                  title="Upload files to Chroma Vector Store"
                >
                  <Paperclip size={16} />
                  <span>Attach Document</span>
                </Link>
                <Link
                  to="/github"
                  className="btn-attach"
                  title="Inspect GitHub Repository"
                >
                  <GitBranch size={16} />
                  <span>GitHub</span>
                </Link>
              </div>

              <div className="controls-right">
                {isStreaming ? (
                  <button
                    type="button"
                    onClick={handleStop}
                    className="btn-stop-stream"
                    title="Stop Generating"
                  >
                    <Square size={16} />
                    <span>Stop</span>
                  </button>
                ) : (
                  <button
                    type="submit"
                    disabled={!input.trim()}
                    className="btn-send-pro"
                    title="Send message (Enter)"
                  >
                    <span>Send</span>
                    <Send size={15} />
                  </button>
                )}
              </div>
            </div>
          </form>

          <div className="input-disclaimer">
            <span>Insight AI v1.0 • Responses grounded in Chroma embeddings & MS-Marco Cross-Encoder</span>
          </div>
        </div>
      </div>
    </div>
  );
}
