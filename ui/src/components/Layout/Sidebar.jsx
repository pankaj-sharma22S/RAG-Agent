import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  MessageSquare,
  FileText,
  Video,
  GitBranch,
  History,
  Activity,
  User,
  Sparkles,
} from "lucide-react";

const navItems = [
  { path: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { path: "/chat", label: "AI Agent Chat", icon: MessageSquare },
  { path: "/documents", label: "Document RAG", icon: FileText },
  { path: "/ingest-media", label: "YouTube & Web", icon: Video },
  { path: "/github", label: "GitHub Analyzer", icon: GitBranch },
  { path: "/history", label: "Threads & Memory", icon: History },
  { path: "/diagnostics", label: "System Diagnostics", icon: Activity },
  { path: "/profile", label: "User Profile", icon: User },
];

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="brand-badge">
          <Sparkles className="icon-accent" size={22} />
        </div>
        <div className="brand-text">
          <span className="brand-title">Insight AI</span>
          <span className="brand-subtitle">Knowledge Platform</span>
        </div>
      </div>

      <nav className="sidebar-nav">
        <div className="nav-section-title">INTELLIGENCE SUITE</div>
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}
            >
              <Icon size={19} className="nav-icon" />
              <span>{item.label}</span>
            </NavLink>
          );
        })}
      </nav>

      <div className="sidebar-footer">
        <div className="status-pill">
          <span className="status-dot healthy" />
          <span>FastAPI v1.0.0 Online</span>
        </div>
      </div>
    </aside>
  );
}
