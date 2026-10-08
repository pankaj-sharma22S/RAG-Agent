import { useState, useRef, useEffect } from "react";
import { api } from "../services/api";
import MarkdownRenderer from "../components/MarkdownRenderer";
import {
  Video,
  Globe,
  CheckCircle,
  AlertCircle,
  ArrowRight,
  Send,
  Square,
  Bot,
  User,
  Sparkles,
  RotateCcw,
  Clock,
  Layers,
  Copy,
  Check,
  FileText,
} from "lucide-react";

export default function MediaIngestPage() {
  const [url, setUrl] = useState("");
  const [ingesting, setIngesting] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  // Embedded Media Chat State
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentStage, setCurrentStage] = useState(null);
  const [threadId, setThreadId] = useState(
    () => `yt_thread_${Math.random().toString(36).substring(2, 9)}`
  );
  const [copiedId, setCopiedId] = useState(null);

  const abortControllerRef = useRef(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, currentStage]);

  const handleIngest = async (e) => {
    e.preventDefault();
    if (!url.trim()) return;

    setIngesting(true);
    setResult(null);
    setError("");

    try {
      const res = await api.ingestUrl(url.trim(), threadId);
      setResult(res);
      setMessages([
        {
          id: "welcome_media",
          role: "assistant",
          content: `### ✅ Successfully Indexed ${res.source_type === "youtube" ? "YouTube Video" : "Web Page"}\n\n**Source:** \`${res.url}\`\n**Indexed Chunks:** ${res.chunks_indexed} vectors in ChromaDB.\n\nYou can now ask specific questions about this transcript or request summaries below!`,
          route: "system",
        },
      ]);
    } catch (err) {
      setError(err.message || "Failed to ingest link.");
    } finally {
      setIngesting(false);
    }
  };

  const handleSend = async (queryText) => {
    const query = (queryText || input).trim();
    if (!query || isStreaming) return;

    setInput("");
    const userMsgId = `user_${Date.now()}`;
    const assistantMsgId = `ai_${Date.now()}`;

    setMessages((prev) => [
      ...prev,
      { id: userMsgId, role: "user", content: query },
      { id: assistantMsgId, role: "assistant", content: "", route: "evaluating" },
    ]);

    setIsStreaming(true);
    setCurrentStage("Retrieving video transcript context...");

    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      await api.streamChatSSE({
        message: query,
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
                    content: m.content || `⚠️ Error: ${errData.message || "Failed to process query."}`,
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
    <div className="page-media">
      <div className="page-header">
        <h1>YouTube & Web Ingestion</h1>
        <p>Extract transcripts and clean article text from public links into the knowledge base</p>
      </div>

      <div className="panel-card max-width-card" style={{ marginBottom: "1.75rem" }}>
        <div className="panel-header">
          <Globe className="icon-accent" size={22} />
          <h2>Index Public Link or Video</h2>
        </div>
        <p className="panel-desc">
          Enter a YouTube video URL (e.g. <code>https://youtube.com/watch?v=...</code>) or any article link to index and converse with its transcript.
        </p>

        <form onSubmit={handleIngest} className="ingest-form">
          <div className="form-group">
            <input
              type="url"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://www.youtube.com/watch?v=... or https://example.com/article"
              disabled={ingesting}
              required
            />
          </div>

          {error && (
            <div className="auth-alert error">
              <AlertCircle size={18} />
              <span>{error}</span>
            </div>
          )}

          {result && (
            <div className="auth-alert success">
              <CheckCircle size={18} />
              <span>
                Successfully indexed {result.source_type} link ({result.chunks_indexed} chunks created). You can now chat with it below!
              </span>
            </div>
          )}

          <button type="submit" className="btn-primary full-width" disabled={!url.trim() || ingesting}>
            {ingesting ? "Extracting & Embedding Transcript..." : "Ingest & Index Link"}
          </button>
        </form>
      </div>

      {/* Embedded Transcript Chat Interface */}
      {(result || messages.length > 0) && (
        <div className="panel-card embedded-chat-panel">
          <div className="panel-header flex-between">
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <Video className="icon-accent" size={22} />
              <h2>Chat with Ingested Media</h2>
            </div>
            <span className="badge">Active Session: <code>{threadId}</code></span>
          </div>

          {/* Quick Prompts */}
          <div className="quick-prompts-bar">
            <button
              type="button"
              onClick={() => handleSend("Give me a comprehensive structured summary of this video transcript with key takeaways.")}
              className="sample-pill"
              disabled={isStreaming}
            >
              <Sparkles size={13} /> Summarize Video
            </button>
            <button
              type="button"
              onClick={() => handleSend("What are the main arguments and conclusions discussed?")}
              className="sample-pill"
              disabled={isStreaming}
            >
              <FileText size={13} /> Main Takeaways
            </button>
            <button
              type="button"
              onClick={() => handleSend("List all specific steps, tools, or techniques mentioned in order.")}
              className="sample-pill"
              disabled={isStreaming}
            >
              <ArrowRight size={13} /> Actionable Steps
            </button>
          </div>

          {/* Messages Container */}
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
                    <MarkdownRenderer content={msg.content || (isStreaming ? "Thinking..." : "")} />
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
              handleSend();
            }}
            className="embedded-input-form"
          >
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask anything about the video or article content..."
              disabled={isStreaming}
            />
            {isStreaming ? (
              <button type="button" onClick={handleStop} className="btn-stop-stream">
                <Square size={15} /> Stop
              </button>
            ) : (
              <button type="submit" disabled={!input.trim()} className="btn-send-pro">
                <Send size={15} /> Send
              </button>
            )}
          </form>
        </div>
      )}
    </div>
  );
}
