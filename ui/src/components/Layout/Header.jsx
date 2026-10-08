import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../../context/AuthContext";
import { api } from "../../services/api";
import { LogOut, User as UserIcon, Shield, RefreshCw } from "lucide-react";

export default function Header() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [healthStatus, setHealthStatus] = useState("checking");

  useEffect(() => {
    let isMounted = true;
    api
      .getHealth()
      .then((data) => {
        if (isMounted) setHealthStatus(data?.status || "offline");
      })
      .catch(() => {
        if (isMounted) setHealthStatus("degraded");
      });
    return () => {
      isMounted = false;
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

      <div className="header-right">
        <div className="user-badge">
          <div className="user-avatar">
            <UserIcon size={16} />
          </div>
          <div className="user-meta">
            <span className="user-email">{user?.email || "Authenticated User"}</span>
            <span className="user-role">
              <Shield size={11} /> {user?.role || "authenticated"}
            </span>
          </div>
        </div>

        <button onClick={handleLogout} className="btn-icon-danger" title="Sign Out">
          <LogOut size={18} />
        </button>
      </div>
    </header>
  );
}
