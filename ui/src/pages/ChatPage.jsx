import { useState, useRef, useEffect } from "react";
import { useSearchParams, useNavigate } from "react-router-dom";
import { api, resolveImageUrl } from "../services/api";
import MarkdownRenderer from "../components/MarkdownRenderer";
import {
  Send,
  Square,
  Sparkles,
  Bot,
  User,
  Shield,
  Layers,
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
  RefreshCw,
  Video,
  X,
  Image as ImageIcon,
  ExternalLink,
  Download,
  ZoomIn,
  ZoomOut,
  Maximize2,
} from "lucide-react";

const CAPABILITIES = [
  { id: "auto", label: "Auto-Adaptive", icon: Sparkles, desc: "Sub-ms dynamic intent routing" },
  { id: "docs", label: "Document RAG", icon: FileText, desc: "Hybrid dense & sparse retrieval" },
  { id: "github", label: "GitHub Agent", icon: GitBranch, desc: "Codebase architecture & audit" },
  { id: "youtube", label: "YouTube Video", icon: Video, desc: "Video transcript analysis" },
  { id: "web", label: "Web Discovery", icon: Globe, desc: "Real-time web search" },
];

const ROUTE_BADGES = {
  document_rag: { label: "Document Intelligence", icon: FileText, styleClass: "badge-docs" },
  document_question: { label: "Document Intelligence", icon: FileText, styleClass: "badge-docs" },
  web_research: { label: "Live Web Discovery", icon: Globe, styleClass: "badge-web" },
  web_search: { label: "Live Web Discovery", icon: Globe, styleClass: "badge-web" },
  semantic_cache: { label: "Instant Cache", icon: Zap, styleClass: "badge-cache" },
  github_analyzer: { label: "GitHub Codebase", icon: GitBranch, styleClass: "badge-github" },
  github_repo: { label: "GitHub Repository", icon: GitBranch, styleClass: "badge-github" },
  github_topic: { label: "GitHub Topic", icon: GitBranch, styleClass: "badge-github" },
  youtube_question: { label: "YouTube Video", icon: Video, styleClass: "badge-youtube" },
  youtube_summary: { label: "YouTube Summary", icon: Video, styleClass: "badge-youtube" },
  hybrid_web_rag: { label: "Hybrid Doc & Web", icon: Layers, styleClass: "badge-hybrid" },
  fast_chat: { label: "Instant Assistant", icon: Sparkles, styleClass: "badge-chat" },
  chat: { label: "AI Assistant", icon: Sparkles, styleClass: "badge-chat" },
  general_chat: { label: "AI Assistant", icon: Sparkles, styleClass: "badge-chat" },
};

function formatRouteBadge(route) {
  if (!route) return null;
  const key = String(route).toLowerCase().trim();
  return (
    ROUTE_BADGES[key] || {
      label: route.replace(/[_-]/g, " "),
      icon: Sparkles,
      styleClass: "badge-default",
    }
  );
}

const STARTER_PROMPTS = [
  {
    category: "Document Intelligence",
    label: "Summarize Uploaded Document",
    query: "Give me a structured hierarchical summary of the uploaded document with key takeaways.",
    icon: FileText,
  },
  {
    category: "System Architecture",
    label: "Banker's Algorithm & Safe States",
    query: "Explain the Banker's Algorithm and how deadlock avoidance calculates safe states.",
    icon: Shield,
  },
  {
    category: "GitHub Codebase Audit",
    label: "Inspect FastAPI Architecture",
    query: "Analyze the architecture and routing mechanisms of https://github.com/fastapi/fastapi",
    icon: GitBranch,
  },
  {
    category: "Real-Time Intelligence",
    label: "Current AI Agent Trends",
    query: "What are the latest breakthrough developments in autonomous AI agent frameworks this year?",
    icon: Globe,
  },
];

export default function ChatPage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();

  // Thread Identification
  const urlThreadId = searchParams.get("thread");
  const isNewChat = searchParams.get("new") === "true";

  const [threadId, setThreadId] = useState(
    () => urlThreadId || `thread_${Math.random().toString(36).substring(2, 9)}`
  );

  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [selectedMode, setSelectedMode] = useState("auto");
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentStage, setCurrentStage] = useState(null);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [copiedId, setCopiedId] = useState(null);
  const [copiedThread, setCopiedThread] = useState(false);
  const [expandedSources, setExpandedSources] = useState({});
  const [uploadingFile, setUploadingFile] = useState(false);
  const [uploadNotice, setUploadNotice] = useState(null);
  const [lightboxData, setLightboxData] = useState(null);
  const [lightboxZoom, setLightboxZoom] = useState(false);
  const [dynamicSuggestions, setDynamicSuggestions] = useState([]);

  const abortControllerRef = useRef(null);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);
  const fileInputRef = useRef(null);

  // Close lightbox on Escape key
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === "Escape" && lightboxData) {
        setLightboxData(null);
        setLightboxZoom(false);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [lightboxData]);

  // Sync state when URL search param changes
  useEffect(() => {
    if (urlThreadId && urlThreadId !== threadId) {
      setThreadId(urlThreadId);
      setMessages([]);
    }
  }, [urlThreadId]);

  // Load thread history from SQLite store
  useEffect(() => {
    let isSubscribed = true;

    const fetchThreadHistory = async () => {
      if (isNewChat) {
        setMessages([]);
        return;
      }

      if (!threadId) return;

      try {
        setLoadingHistory(true);
        const threadData = await api.getThread(threadId);

        if (!isSubscribed) return;

        if (threadData && threadData.turns && threadData.turns.length > 0) {
          const restoredMessages = [];
          threadData.turns.forEach((turn, idx) => {
            restoredMessages.push({
              id: `user_${turn.turn_id || idx}`,
              role: "user",
              content: turn.user_query,
              timestamp: new Date(turn.created_at || Date.now()).toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
              }),
            });
            restoredMessages.push({
              id: `ai_${turn.turn_id || idx}`,
              role: "assistant",
              content: turn.agent_response,
              route: turn.route || "memory_chat",
              latency_ms: turn.latency_ms || 0,
              timestamp: new Date(turn.created_at || Date.now()).toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
              }),
            });
          });
          setMessages(restoredMessages);
        } else {
          setMessages([]);
        }
      } catch (err) {
        console.warn("Could not restore thread history:", err);
        if (isSubscribed) setMessages([]);
      } finally {
        if (isSubscribed) setLoadingHistory(false);
      }
    };

    fetchThreadHistory();

    return () => {
      isSubscribed = false;
    };
  }, [threadId, isNewChat]);

  // Load document and thread-grounded suggestions dynamically
  useEffect(() => {
    let isSubscribed = true;
    const fetchSuggestions = async () => {
      try {
        const res = await api.getThreadSuggestions(threadId);
        if (isSubscribed && res?.suggestions && res.suggestions.length > 0) {
          setDynamicSuggestions(res.suggestions);
        }
      } catch (err) {
        console.debug("Could not load dynamic suggestions:", err);
      }
    };
    fetchSuggestions();
    return () => {
      isSubscribed = false;
    };
  }, [threadId, messages.length]);

  // Auto-scroll on new tokens
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, currentStage]);

  // Auto-resize composer textarea
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

  const handleNewChat = () => {
    const newId = `thread_${Math.random().toString(36).substring(2, 9)}`;
    setMessages([]);
    setThreadId(newId);
    navigate(`/chat?thread=${newId}&new=true`);
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      setUploadingFile(true);
      setUploadNotice({ type: "info", text: `Uploading & indexing "${file.name}"...` });
      const res = await api.uploadFile(file, threadId);
      setUploadNotice({
        type: "success",
        text: `Indexed "${res.filename}" (${res.chunks_indexed} chunks created). You can now ask questions about it!`,
      });
      setTimeout(() => setUploadNotice(null), 6000);
    } catch (err) {
      setUploadNotice({ type: "error", text: `Upload failed: ${err.message}` });
    } finally {
      setUploadingFile(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handleSend = async (textToSend) => {
    const query = (textToSend || input).trim();
    if (!query || isStreaming) return;

    if (isNewChat) {
      navigate(`/chat?thread=${threadId}`, { replace: true });
    }

    setInput("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }

    const userMsgId = `user_${Date.now()}`;
    const assistantMsgId = `ai_${Date.now()}`;

    const userMessage = {
      id: userMsgId,
      role: "user",
      content: query,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };

    const initialAssistantMessage = {
      id: assistantMsgId,
      role: "assistant",
      content: "",
      route: selectedMode === "auto" ? "evaluating" : selectedMode,
      confidence: 1.0,
      sources: [],
      latency_ms: 0,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };

    setMessages((prev) => [...prev, userMessage, initialAssistantMessage]);
    setIsStreaming(true);
    setCurrentStage("Analyzing intent & routing query...");

    const controller = new AbortController();
    abortControllerRef.current = controller;

    let collectedContent = "";
    const startTime = performance.now();

    try {
      await api.streamChatSSE({
        message: query,
        thread_id: threadId,
        signal: controller.signal,
        onStatus: (statusData) => {
          if (statusData.message) {
            setCurrentStage(statusData.message);
          }
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId
                ? {
                    ...m,
                    route: statusData.route || m.route,
                    confidence: statusData.confidence ?? m.confidence,
                  }
                : m
            )
          );
        },
        onToken: (token) => {
          collectedContent += token;
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId
                ? {
                    ...m,
                    content: collectedContent,
                  }
                : m
            )
          );
        },
        onDone: (doneData) => {
          const totalLatency = Math.round(performance.now() - startTime);
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId
                ? {
                    ...m,
                    content: doneData.answer || collectedContent,
                    route: doneData.route || m.route,
                    sources: doneData.sources || m.sources || [],
                    latency_ms: doneData.latency_ms || totalLatency,
                  }
                : m
            )
          );
        },
        onError: (errData) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId
                ? {
                    ...m,
                    content:
                      m.content ||
                      `⚠️ **Pipeline Error:** ${errData.message || "An error occurred during query execution."}`,
                    isError: true,
                  }
                : m
            )
          );
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

  const handleRetry = (msgIndex) => {
    if (msgIndex <= 0) return;
    const prevUserMsg = messages[msgIndex - 1];
    if (prevUserMsg && prevUserMsg.role === "user") {
      handleSend(prevUserMsg.content);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="pro-chat-wrapper">
      {/* Top Session Deck */}
      <div className="pro-chat-topbar">
        <div className="topbar-left">
          <div className="model-chip">
            <Cpu size={14} className="chip-icon" />
            <span>Gemini 2.5 Flash</span>
            <span className="chip-divider">•</span>
            <span className="routing-tag">Sub-ms Adaptive RAG</span>
          </div>

          <button
            onClick={handleCopyThreadId}
            className="session-tag-btn"
            title="Click to copy Thread Session ID"
          >
            <span>Thread:</span>
            <code>{threadId}</code>
            {copiedThread ? <Check size={12} className="icon-success" /> : <Copy size={12} />}
          </button>
        </div>

        <div className="topbar-right">
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploadingFile}
            className="topbar-link-btn"
            title="Upload Document into Thread Memory"
          >
            <Paperclip size={13} />
            <span>{uploadingFile ? "Indexing..." : "Attach File"}</span>
          </button>
          <input
            ref={fileInputRef}
            type="file"
            onChange={handleFileUpload}
            style={{ display: "none" }}
            accept=".pdf,.docx,.doc,.pptx,.xlsx,.csv,.txt,.png,.jpg,.jpeg"
          />

          <button
            onClick={handleNewChat}
            className="topbar-link-btn primary-btn"
            title="Start a fresh conversation thread"
          >
            <RotateCcw size={13} />
            <span>New Chat</span>
          </button>
        </div>
      </div>

      {/* Upload Notification Banner */}
      {uploadNotice && (
        <div className={`notice-banner notice-${uploadNotice.type}`}>
          <span>{uploadNotice.text}</span>
          <button onClick={() => setUploadNotice(null)} className="notice-close-btn">
            <X size={13} />
          </button>
        </div>
      )}

      {/* Main Conversation Feed */}
      <div className="pro-chat-feed custom-scrollbar">
        <div className="feed-container">
          {loadingHistory ? (
            <div className="feed-loading-box">
              <RefreshCw size={24} className="spin icon-accent" />
              <span>Loading conversation history...</span>
            </div>
          ) : messages.length === 0 ? (
            /* Polished Empty State / Starter Prompts */
            <div className="empty-welcome-screen">
              <div className="welcome-badge">
                <Sparkles size={28} />
              </div>

              <div className="welcome-content">
                <h1 className="welcome-title">Insight AI Knowledge Workspace</h1>
                <p className="welcome-desc">
                  Enterprise-grade agentic assistant with <strong>Multimodal PDF RAG</strong>,{" "}
                  <strong>GitHub Codebase Audits</strong>, <strong>YouTube Transcripts</strong>, and{" "}
                  <strong>Real-time Discovery</strong>.
                </p>
              </div>

              <div className="starter-prompts-grid">
                {(dynamicSuggestions.length > 0 ? dynamicSuggestions : STARTER_PROMPTS).map((prompt, idx) => {
                  let Icon = Sparkles;
                  if (prompt.icon === "FileText" || prompt.icon === FileText) Icon = FileText;
                  else if (prompt.icon === "Layers" || prompt.icon === Layers) Icon = Layers;
                  else if (prompt.icon === "ImageIcon" || prompt.icon === ImageIcon) Icon = ImageIcon;
                  else if (prompt.icon === "Video" || prompt.icon === Video) Icon = Video;
                  else if (prompt.icon === "GitBranch" || prompt.icon === GitBranch) Icon = GitBranch;
                  else if (prompt.icon === "Shield" || prompt.icon === Shield) Icon = Shield;
                  else if (prompt.icon === "Globe" || prompt.icon === Globe) Icon = Globe;

                  return (
                    <div
                      key={idx}
                      onClick={() => handleSend(prompt.query)}
                      className="starter-prompt-card"
                    >
                      <div className="prompt-header">
                        <Icon size={16} className="icon-accent" />
                        <span className="prompt-label">{prompt.label}</span>
                      </div>
                      <p className="prompt-query">{prompt.query}</p>
                    </div>
                  );
                })}
              </div>
            </div>
          ) : (
            /* Render Chronological Messages */
            messages.map((msg, idx) => {
              const isUser = msg.role === "user";
              return (
                <div key={msg.id || idx} className={`pro-message-row ${msg.role}`}>
                  <div className={`pro-avatar ${msg.role}`}>
                    {isUser ? <User size={16} /> : <Bot size={16} />}
                  </div>

                  <div className="pro-message-body">
                    {/* Telemetry Header for AI Assistant */}
                    {!isUser && msg.route && (() => {
                      const badge = formatRouteBadge(msg.route);
                      if (!badge) return null;
                      const BadgeIcon = badge.icon;
                      return (
                        <div className="pro-telemetry-header">
                          <span className={`telemetry-pill ${badge.styleClass}`}>
                            <BadgeIcon size={12} className="telemetry-icon" />
                            <span>{badge.label}</span>
                          </span>
                          {msg.latency_ms > 0 && (
                            <span className="telemetry-latency">
                              <Clock size={11} /> {msg.latency_ms} ms
                            </span>
                          )}
                          {msg.confidence !== undefined && (
                            <span className="telemetry-latency">
                              {Math.round(msg.confidence * 100)}% Match
                            </span>
                          )}
                        </div>
                      );
                    })()}

                    {/* Message Bubble Surface */}
                    <div className={`pro-bubble ${msg.role} ${msg.isError ? "error-bubble" : ""}`}>
                      {isUser ? (
                        <div className="user-text">{msg.content}</div>
                      ) : (
                        <MarkdownRenderer
                          content={
                            msg.content ||
                            (isStreaming && msg.id.startsWith("ai_")
                              ? "Generating verified response..."
                              : "")
                          }
                          onImageClick={(url, alt) =>
                            setLightboxData({
                              url: resolveImageUrl(url),
                              title: alt || "Visual Citation",
                            })
                          }
                        />
                      )}

                      {/* Multimodal Visuals Quick-Bar (if sources have images) */}
                      {!isUser &&
                        msg.sources &&
                        msg.sources.some((s) => s.image_url) && (
                          <div className="visual-evidence-strip">
                            <div className="visual-strip-header">
                              <ImageIcon size={13} className="icon-accent" />
                              <span>Retrieved Multimodal Document Visuals:</span>
                            </div>
                            <div className="visual-thumbs-row custom-scrollbar">
                              {msg.sources
                                .filter((s) => s.image_url)
                                .map((s, imgIdx) => (
                                  <div
                                    key={imgIdx}
                                    className="visual-thumb-item"
                                    onClick={() =>
                                      setLightboxData({
                                        url: resolveImageUrl(s.image_url),
                                        title: s.filename || (s.source ? s.source.split(/[\\/]/).pop() : "Extracted Visual"),
                                        page: s.page,
                                        score: s.rerank_score,
                                      })
                                    }
                                    title="Click to view full-resolution image"
                                  >
                                    <img
                                      src={resolveImageUrl(s.image_url)}
                                      alt="Extracted figure"
                                      className="visual-thumb-img"
                                      loading="lazy"
                                      onError={(e) => {
                                        e.target.parentElement.style.display = "none";
                                      }}
                                    />
                                    {s.page && <span className="visual-page-badge">Pg {s.page}</span>}
                                  </div>
                                ))}
                            </div>
                          </div>
                        )}

                      {/* Grounded Citations Drawer */}
                      {msg.sources && msg.sources.length > 0 && (
                        <div className="citations-drawer">
                          <button
                            type="button"
                            onClick={() => toggleSources(msg.id)}
                            className="citation-toggle-btn"
                          >
                            <Info size={13} />
                            <span>Grounded Citations & Sources ({msg.sources.length})</span>
                            {expandedSources[msg.id] ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
                          </button>

                          {expandedSources[msg.id] && (
                            <div className="citation-grid">
                              {msg.sources.map((src, sIdx) => (
                                <div key={sIdx} className="citation-card">
                                  <div className="citation-card-header">
                                    <span className="citation-title" title={src.filename || src.source}>
                                      #{sIdx + 1} {src.filename || (src.source ? src.source.split(/[\\/]/).pop() : "Source Chunk")}
                                    </span>
                                    {src.page && (
                                      <span className="citation-page-tag">Page {src.page}</span>
                                    )}
                                  </div>

                                  {src.image_url && (
                                    <div
                                      className="citation-img-wrap"
                                      onClick={() =>
                                        setLightboxData({
                                          url: resolveImageUrl(src.image_url),
                                          title: src.filename || (src.source ? src.source.split(/[\\/]/).pop() : "Citation Visual"),
                                          page: src.page,
                                          score: src.rerank_score,
                                        })
                                      }
                                      title="Click to zoom image"
                                    >
                                      <img
                                        src={resolveImageUrl(src.image_url)}
                                        alt="Citation visual"
                                        className="citation-img-thumb"
                                        loading="lazy"
                                        onError={(e) => {
                                          e.target.parentElement.style.display = "none";
                                        }}
                                      />
                                      <div className="citation-zoom-hint">
                                        <ZoomIn size={12} />
                                        <span>Inspect</span>
                                      </div>
                                    </div>
                                  )}

                                  {src.rerank_score !== undefined && (
                                    <div className="citation-score">
                                      Score: {Number(src.rerank_score).toFixed(3)}
                                    </div>
                                  )}
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      )}
                    </div>

                    {/* Action Toolbar */}
                    {!isUser && msg.content && (
                      <div className="pro-message-actions">
                        <button
                          onClick={() => handleCopy(msg.id, msg.content)}
                          className="action-btn"
                          title="Copy markdown content"
                        >
                          {copiedId === msg.id ? (
                            <Check size={12} className="icon-success" />
                          ) : (
                            <Copy size={12} />
                          )}
                          <span>{copiedId === msg.id ? "Copied" : "Copy"}</span>
                        </button>
                        <button
                          onClick={() => handleRetry(idx)}
                          className="action-btn"
                          title="Retry query"
                        >
                          <RotateCcw size={12} />
                          <span>Retry</span>
                        </button>
                        {msg.timestamp && <span className="action-timestamp">{msg.timestamp}</span>}
                      </div>
                    )}
                  </div>
                </div>
              );
            })
          )}

          {/* Active Pipeline Stage Pulse */}
          {currentStage && (
            <div className="pro-stage-banner">
              <div className="stage-pulse-ring" />
              <span className="stage-text">{currentStage}</span>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Bottom Composer Deck */}
      <div className="pro-composer-wrapper">
        <div className="composer-container">
          {/* Capabilities Selector */}
          <div className="capabilities-strip custom-scrollbar">
            {CAPABILITIES.map((cap) => {
              const Icon = cap.icon;
              const isSelected = selectedMode === cap.id;
              return (
                <button
                  key={cap.id}
                  onClick={() => setSelectedMode(cap.id)}
                  className={`cap-pill ${isSelected ? "active" : ""}`}
                  title={cap.desc}
                >
                  <Icon size={12} />
                  <span>{cap.label}</span>
                </button>
              );
            })}
          </div>

          {/* Composer Input Box */}
          <div className="pro-composer-box">
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={uploadingFile || isStreaming}
              className="composer-attach-btn"
              title="Attach document or image"
            >
              <Paperclip size={18} />
            </button>

            <textarea
              ref={textareaRef}
              rows={1}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask anything across documents, GitHub repositories, YouTube, or the web... (Shift+Enter for newline)"
              disabled={isStreaming}
              className="composer-textarea custom-scrollbar"
            />

            {isStreaming ? (
              <button
                onClick={handleStop}
                className="composer-stop-btn"
                title="Stop generation"
              >
                <Square size={16} />
              </button>
            ) : (
              <button
                onClick={() => handleSend()}
                disabled={!input.trim()}
                className={`composer-send-btn ${input.trim() ? "active" : ""}`}
                title="Send query"
              >
                <Send size={16} />
              </button>
            )}
          </div>

          <div className="composer-footer-text">
            <span>Press <strong>Enter</strong> to send, <strong>Shift + Enter</strong> for new line</span>
            <span className="security-tag">Protected by PII Masking & Injection Defense</span>
          </div>
        </div>
      </div>

      {/* Multimodal Fullscreen Lightbox Modal */}
      {lightboxData && (
        <div className="lightbox-overlay" onClick={() => { setLightboxData(null); setLightboxZoom(false); }}>
          <div className="lightbox-container" onClick={(e) => e.stopPropagation()}>
            <div className="lightbox-header">
              <div className="lightbox-title-wrap">
                <ImageIcon size={16} className="icon-accent" />
                <span className="lightbox-title">{lightboxData.title || "Visual Evidence"}</span>
                {lightboxData.page && (
                  <span className="lightbox-page-tag">Page {lightboxData.page}</span>
                )}
                {lightboxData.score !== undefined && (
                  <span className="lightbox-score-tag">
                    Score: {Number(lightboxData.score).toFixed(3)}
                  </span>
                )}
              </div>

              <div className="lightbox-controls">
                <button
                  onClick={() => setLightboxZoom((prev) => !prev)}
                  className={`lightbox-btn ${lightboxZoom ? "active" : ""}`}
                  title={lightboxZoom ? "Fit to window" : "Zoom 100%"}
                >
                  {lightboxZoom ? <ZoomOut size={15} /> : <ZoomIn size={15} />}
                  <span>{lightboxZoom ? "Fit" : "Zoom"}</span>
                </button>

                <a
                  href={lightboxData.url}
                  target="_blank"
                  rel="noreferrer"
                  className="lightbox-btn"
                  title="Open image in new tab"
                >
                  <ExternalLink size={15} />
                  <span>Open Tab</span>
                </a>

                <a
                  href={lightboxData.url}
                  download={lightboxData.title || "extracted_figure.png"}
                  className="lightbox-btn"
                  title="Download image"
                >
                  <Download size={15} />
                  <span>Download</span>
                </a>

                <button
                  onClick={() => { setLightboxData(null); setLightboxZoom(false); }}
                  className="lightbox-close-btn"
                  title="Close viewer (Esc)"
                >
                  <X size={18} />
                </button>
              </div>
            </div>

            <div className={`lightbox-body ${lightboxZoom ? "zoomed" : ""}`}>
              <img
                src={lightboxData.url}
                alt={lightboxData.title || "Document Visual"}
                className="lightbox-image"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
