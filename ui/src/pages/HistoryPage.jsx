import { useState, useEffect } from "react";
import { api } from "../services/api";
import { History, MessageSquare, Clock, User, Bot, Layers, ChevronRight } from "lucide-react";

export default function HistoryPage() {
  const [threads, setThreads] = useState([]);
  const [selectedThread, setSelectedThread] = useState(null);
  const [threadDetails, setThreadDetails] = useState(null);
  const [loading, setLoading] = useState(true);
  const [loadingDetails, setLoadingDetails] = useState(false);

  useEffect(() => {
    let isMounted = true;
    api
      .listThreads()
      .then((data) => {
        if (isMounted) {
          setThreads(data.threads || []);
          setLoading(false);
        }
      })
      .catch((err) => {
        console.error("Failed to load threads:", err);
        if (isMounted) setLoading(false);
      });
    return () => {
      isMounted = false;
    };
  }, []);

  const handleSelectThread = async (threadId) => {
    setSelectedThread(threadId);
    setLoadingDetails(true);
    try {
      const details = await api.getThread(threadId);
      setThreadDetails(details);
    } catch (err) {
      console.error("Failed to fetch thread details:", err);
    } finally {
      setLoadingDetails(false);
    }
  };

  return (
    <div className="page-history">
      <div className="page-header">
        <h1>Conversation Threads & Memory</h1>
        <p>Persisted SQLite conversation sessions and episodic memory turns</p>
      </div>

      <div className="grid-history-split">
        {/* Thread List Column */}
        <div className="threads-sidebar-card">
          <div className="panel-header">
            <History className="icon-accent" size={20} />
            <h2>Saved Threads ({threads.length})</h2>
          </div>

          {loading ? (
            <div className="loading-placeholder">Loading threads...</div>
          ) : threads.length === 0 ? (
            <div className="empty-placeholder">No conversation threads found.</div>
          ) : (
            <div className="threads-list">
              {threads.map((t) => (
                <div
                  key={t.thread_id}
                  onClick={() => handleSelectThread(t.thread_id)}
                  className={`thread-item-card ${selectedThread === t.thread_id ? "active" : ""}`}
                >
                  <div className="thread-item-header">
                    <span className="thread-title">{t.title || t.thread_id}</span>
                    <ChevronRight size={16} />
                  </div>
                  <div className="thread-meta-row">
                    <span>{t.turn_count} turns</span>
                    <span>{new Date(t.updated_at).toLocaleDateString()}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Turn Inspector Column */}
        <div className="thread-content-card">
          {!selectedThread ? (
            <div className="empty-content-state">
              <MessageSquare size={48} className="icon-muted" />
              <h3>Select a thread from the list</h3>
              <p>View full conversation turns, agent reasoning routes, and timestamps</p>
            </div>
          ) : loadingDetails ? (
            <div className="loading-placeholder">Loading thread turns...</div>
          ) : (
            <div className="thread-turns-view">
              <div className="turns-header">
                <h2>{threadDetails?.title || selectedThread}</h2>
                <span className="badge-meta">{threadDetails?.turns?.length || 0} Turns</span>
              </div>

              <div className="turns-list">
                {threadDetails?.turns?.map((turn, idx) => (
                  <div key={turn.turn_id || idx} className="turn-block">
                    <div className="turn-bubble user">
                      <div className="turn-avatar">
                        <User size={14} />
                      </div>
                      <div className="turn-text">{turn.user_query}</div>
                    </div>
                    <div className="turn-bubble assistant">
                      <div className="turn-avatar">
                        <Bot size={14} />
                      </div>
                      <div className="turn-text">{turn.agent_response}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
