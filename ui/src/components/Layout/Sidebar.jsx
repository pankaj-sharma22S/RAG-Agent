import { useState, useEffect } from "react";
import { NavLink, useNavigate, useLocation, useSearchParams } from "react-router-dom";
import {
  LayoutDashboard,
  MessageSquare,
  FileText,
  Video,
  GitBranch,
  Activity,
  User,
  Sparkles,
  Plus,
  Search,
  Trash2,
  Edit2,
  Sun,
  Moon,
  LogOut,
  Palette,
  MessageCircle,
  X,
  Check,
} from "lucide-react";
import { useAuth } from "../../context/AuthContext";
import { useTheme, ACCENT_THEMES } from "../../context/ThemeContext";
import { api } from "../../services/api";

const navItems = [
  { path: "/chat", label: "AI Workspace", icon: MessageSquare },
  { path: "/dashboard", label: "Overview", icon: LayoutDashboard },
  { path: "/documents", label: "Document RAG", icon: FileText },
  { path: "/ingest-media", label: "YouTube & Web", icon: Video },
  { path: "/github", label: "GitHub Analyzer", icon: GitBranch },
  { path: "/diagnostics", label: "Diagnostics", icon: Activity },
  { path: "/profile", label: "Profile Memory", icon: User },
];

export default function Sidebar() {
  const { user, logout } = useAuth();
  const { theme, accent, toggleTheme, setAccent } = useTheme();
  const navigate = useNavigate();
  const location = useLocation();
  const [searchParams] = useSearchParams();
  const activeThreadId = searchParams.get("thread");

  const [threads, setThreads] = useState([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [editingThreadId, setEditingThreadId] = useState(null);
  const [editTitle, setEditTitle] = useState("");
  const [loadingThreads, setLoadingThreads] = useState(false);
  const [showPaletteMenu, setShowPaletteMenu] = useState(false);

  // Load threads on mount and location changes
  const loadThreads = async () => {
    try {
      setLoadingThreads(true);
      const res = await api.listThreads(40, 0);
      setThreads(res?.threads || []);
    } catch (err) {
      console.warn("Could not load threads:", err);
    } finally {
      setLoadingThreads(false);
    }
  };

  useEffect(() => {
    loadThreads();
  }, [location.key]);

  const handleNewChat = () => {
    const newThreadId = `thread_${Math.random().toString(36).substring(2, 9)}`;
    navigate(`/chat?thread=${newThreadId}&new=true`);
  };

  const handleSelectThread = (threadId) => {
    navigate(`/chat?thread=${threadId}`);
  };

  const handleDeleteThread = async (e, threadId) => {
    e.stopPropagation();
    if (!window.confirm("Are you sure you want to delete this conversation thread?")) return;
    try {
      await api.deleteThread(threadId);
      setThreads((prev) => prev.filter((t) => t.thread_id !== threadId));
      if (activeThreadId === threadId) {
        handleNewChat();
      }
    } catch (err) {
      alert("Failed to delete thread: " + err.message);
    }
  };

  const handleStartRename = (e, thread) => {
    e.stopPropagation();
    setEditingThreadId(thread.thread_id);
    setEditTitle(thread.title || "");
  };

  const handleSaveRename = async (e, threadId) => {
    if (e) e.stopPropagation();
    if (!editTitle.trim()) {
      setEditingThreadId(null);
      return;
    }
    try {
      await api.renameThread(threadId, editTitle.trim());
      setThreads((prev) =>
        prev.map((t) => (t.thread_id === threadId ? { ...t, title: editTitle.trim() } : t))
      );
    } catch (err) {
      console.warn("Could not rename thread:", err);
    } finally {
      setEditingThreadId(null);
    }
  };

  const filteredThreads = threads.filter((t) =>
    (t.title || t.thread_id || "").toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <aside className="sidebar">
      {/* Brand Header */}
      <div className="sidebar-brand">
        <div className="brand-badge">
          <Sparkles size={20} />
        </div>
        <div className="brand-text">
          <span className="brand-title">Insight AI</span>
          <span className="brand-subtitle">Enterprise Agentic Workspace</span>
        </div>
      </div>

      {/* New Conversation Button */}
      <div className="sidebar-action-wrap">
        <button onClick={handleNewChat} className="sidebar-new-btn" title="Create a new conversation thread">
          <Plus size={16} />
          <span>New Conversation</span>
        </button>
      </div>

      {/* Navigation Links */}
      <nav className="sidebar-nav">
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) => `sidebar-nav-item ${isActive ? "active" : ""}`}
            >
              <Icon size={16} className="nav-icon" />
              <span>{item.label}</span>
            </NavLink>
          );
        })}
      </nav>

      {/* Thread History Section */}
      <div className="sidebar-threads-section">
        <div className="sidebar-threads-header">
          <span>Conversations</span>
          <span className="thread-count-badge">{threads.length}</span>
        </div>

        {/* Thread Search Box */}
        <div className="sidebar-search-box">
          <Search size={13} className="search-icon" />
          <input
            type="text"
            placeholder="Search threads..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="sidebar-search-input"
          />
          {searchQuery && (
            <button onClick={() => setSearchQuery("")} className="clear-search-btn">
              <X size={12} />
            </button>
          )}
        </div>

        {/* Scrollable Threads List */}
        <div className="sidebar-threads-list custom-scrollbar">
          {loadingThreads && threads.length === 0 ? (
            <div className="sidebar-empty-hint">Loading conversations...</div>
          ) : filteredThreads.length === 0 ? (
            <div className="sidebar-empty-hint">
              {searchQuery ? "No matching conversations" : "No conversations yet"}
            </div>
          ) : (
            filteredThreads.map((thread) => {
              const isActive = activeThreadId === thread.thread_id;
              const isEditing = editingThreadId === thread.thread_id;

              return (
                <div
                  key={thread.thread_id}
                  onClick={() => handleSelectThread(thread.thread_id)}
                  className={`sidebar-thread-item ${isActive ? "active" : ""}`}
                  title={thread.title || thread.thread_id}
                >
                  <MessageCircle size={14} className="thread-icon" />

                  {isEditing ? (
                    <input
                      type="text"
                      value={editTitle}
                      onChange={(e) => setEditTitle(e.target.value)}
                      onBlur={(e) => handleSaveRename(e, thread.thread_id)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") handleSaveRename(e, thread.thread_id);
                        if (e.key === "Escape") setEditingThreadId(null);
                      }}
                      autoFocus
                      onClick={(e) => e.stopPropagation()}
                      className="sidebar-rename-input"
                    />
                  ) : (
                    <span className="sidebar-thread-title">
                      {thread.title || "Conversation"}
                    </span>
                  )}

                  {!isEditing && (
                    <div className="sidebar-thread-actions">
                      <button
                        onClick={(e) => handleStartRename(e, thread)}
                        className="thread-action-btn"
                        title="Rename conversation"
                      >
                        <Edit2 size={12} />
                      </button>
                      <button
                        onClick={(e) => handleDeleteThread(e, thread.thread_id)}
                        className="thread-action-btn delete"
                        title="Delete conversation"
                      >
                        <Trash2 size={12} />
                      </button>
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>
      </div>

      {/* Theme & Palette Selector Popover */}
      {showPaletteMenu && (
        <div className="sidebar-palette-modal">
          <div className="palette-header">
            <span>Choose Accent Theme</span>
            <button onClick={() => setShowPaletteMenu(false)} className="palette-close-btn">
              <X size={13} />
            </button>
          </div>
          <div className="palette-grid">
            {ACCENT_THEMES.map((t) => (
              <button
                key={t.id}
                onClick={() => {
                  setAccent(t.id);
                  setShowPaletteMenu(false);
                }}
                className={`palette-item-btn ${accent === t.id ? "active" : ""}`}
                title={t.label}
              >
                <span className="palette-color-swatch" style={{ backgroundColor: t.color }} />
                <span className="palette-label">{t.label}</span>
                {accent === t.id && <Check size={13} className="palette-check" />}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Sidebar Footer */}
      <div className="sidebar-footer">
        <div className="sidebar-user-block">
          <div className="user-avatar-mini">
            {user?.email ? user.email.charAt(0).toUpperCase() : "A"}
          </div>
          <div className="user-info-text">
            <span className="user-email-text" title={user?.email || "Guest Analyst"}>
              {user?.email || "Guest Analyst"}
            </span>
            <span className="user-role-text">{user?.role || "analyst"}</span>
          </div>
        </div>

        <div className="sidebar-footer-controls">
          <button
            onClick={() => setShowPaletteMenu((prev) => !prev)}
            className={`footer-icon-btn ${showPaletteMenu ? "active" : ""}`}
            title="Choose Color Accent Theme"
          >
            <Palette size={15} />
          </button>

          <button
            onClick={toggleTheme}
            className="footer-icon-btn"
            title={`Switch to ${theme === "dark" ? "Light" : "Dark"} Mode`}
          >
            {theme === "dark" ? <Sun size={15} /> : <Moon size={15} />}
          </button>

          {user && logout && (
            <button onClick={logout} className="footer-icon-btn logout" title="Sign Out">
              <LogOut size={15} />
            </button>
          )}
        </div>
      </div>
    </aside>
  );
}
