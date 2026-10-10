import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../../context/AuthContext";
import { useTheme, ACCENT_THEMES } from "../../context/ThemeContext";
import { api } from "../../services/api";
import { LogOut, User as UserIcon, Shield, Sun, Moon, Palette, Check, X } from "lucide-react";

export default function Header() {
  const { user, logout } = useAuth();
  const { theme, accent, toggleTheme, setAccent } = useTheme();
  const navigate = useNavigate();
  const [healthStatus, setHealthStatus] = useState("checking");
  const [showPalette, setShowPalette] = useState(false);

  useEffect(() => {
    let isMounted = true;
    const checkHealth = () => {
      api
        .getHealth()
        .then((data) => {
          if (isMounted) setHealthStatus(data?.status || "offline");
        })
        .catch(() => {
          if (isMounted) setHealthStatus("degraded");
        });
    };

    checkHealth();
    const intervalId = window.setInterval(checkHealth, 15000);

    return () => {
      isMounted = false;
      window.clearInterval(intervalId);
    };
  }, []);

  const handleLogout = async () => {
    await logout();
    navigate("/login");
  };

  return (
    <header className="app-header">
      <div className="header-left">
        <div className={`health-indicator ${healthStatus}`}>
          <span className="dot" />
          <span className="label">
            Backend: {healthStatus === "healthy" ? "Connected" : healthStatus}
          </span>
        </div>
      </div>

      <div className="header-right" style={{ position: "relative" }}>
        {/* Color Palette Selector Dropdown */}
        <button
          onClick={() => setShowPalette((prev) => !prev)}
          className={`footer-icon-btn ${showPalette ? "active" : ""}`}
          title="Change Color Theme Accent"
        >
          <Palette size={16} />
        </button>

        {showPalette && (
          <div
            style={{
              position: "absolute",
              top: "44px",
              right: "80px",
              background: "var(--bg-card)",
              border: "1px solid var(--border-color)",
              borderRadius: "var(--radius-lg)",
              padding: "0.75rem",
              boxShadow: "var(--shadow-lg)",
              zIndex: 1000,
              width: "220px",
              display: "flex",
              flexDirection: "column",
              gap: "0.35rem",
            }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                fontSize: "0.75rem",
                fontWeight: 700,
                marginBottom: "0.25rem",
              }}
            >
              <span>Accent Color Palette</span>
              <button
                onClick={() => setShowPalette(false)}
                style={{ background: "transparent", border: "none", color: "var(--text-muted)", cursor: "pointer" }}
              >
                <X size={12} />
              </button>
            </div>
            {ACCENT_THEMES.map((t) => (
              <button
                key={t.id}
                onClick={() => {
                  setAccent(t.id);
                  setShowPalette(false);
                }}
                className={`palette-item-btn ${accent === t.id ? "active" : ""}`}
                style={{ width: "100%" }}
              >
                <span className="palette-color-swatch" style={{ backgroundColor: t.color }} />
                <span className="palette-label">{t.label}</span>
                {accent === t.id && <Check size={13} className="palette-check" />}
              </button>
            ))}
          </div>
        )}

        {/* Theme Dark/Light Toggle */}
        <button
          onClick={toggleTheme}
          className="footer-icon-btn"
          title={`Switch to ${theme === "dark" ? "Light" : "Dark"} Mode`}
        >
          {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
        </button>

        {/* User Identity Pill */}
        <div className="user-badge">
          <div className="user-avatar">
            <UserIcon size={15} />
          </div>
          <div className="user-meta">
            <span className="user-email">{user?.email || "Authenticated User"}</span>
            <span className="user-role">
              <Shield size={10} /> {user?.role || "analyst"}
            </span>
          </div>
        </div>

        {/* Sign Out Button */}
        <button onClick={handleLogout} className="btn-icon-danger" title="Sign Out">
          <LogOut size={16} />
        </button>
      </div>
    </header>
  );
}
