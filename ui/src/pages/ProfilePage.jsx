import { useState, useEffect } from "react";
import { api } from "../services/api";
import { useAuth } from "../context/AuthContext";
import { User, Shield, Code, Tag, Clock, CheckCircle } from "lucide-react";

export default function ProfilePage() {
  const { user } = useAuth();
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let isMounted = true;
    api
      .getUserProfile()
      .then((data) => {
        if (isMounted) {
          setProfile(data);
          setLoading(false);
        }
      })
      .catch((err) => {
        console.error("Failed to load user profile:", err);
        if (isMounted) setLoading(false);
      });
    return () => {
      isMounted = false;
    };
  }, []);

  return (
    <div className="page-profile">
      <div className="page-header">
        <h1>User Profile & Semantic Memory</h1>
        <p>Verified cryptographic identity and durable personalized memory preferences</p>
      </div>

      <div className="grid-2-col">
        {/* Identity Card */}
        <div className="panel-card">
          <div className="panel-header">
            <Shield className="icon-accent" size={22} />
            <h2>Verified Identity</h2>
          </div>
          <div className="profile-details-list">
            <div className="detail-item">
              <span className="label">Verified User ID (JWT sub)</span>
              <code className="code-tag">{user?.id || "N/A"}</code>
            </div>
            <div className="detail-item">
              <span className="label">Email Address</span>
              <span className="value">{user?.email || "N/A"}</span>
            </div>
            <div className="detail-item">
              <span className="label">Role</span>
              <span className="value">{user?.role || "authenticated"}</span>
            </div>
            <div className="detail-item">
              <span className="label">Last Sign In</span>
              <span className="value">
                {user?.last_sign_in_at
                  ? new Date(user.last_sign_in_at).toLocaleString()
                  : "Active Session"}
              </span>
            </div>
          </div>
        </div>

        {/* Semantic Memory Card */}
        <div className="panel-card">
          <div className="panel-header">
            <Code className="icon-accent" size={22} />
            <h2>Personalized Memory Facts</h2>
          </div>
          <p className="panel-desc">
            Automatically extracted facts from user interactions stored in durable episodic memory.
          </p>

          <div className="profile-details-list">
            <div className="detail-item">
              <span className="label">Stated Name</span>
              <span className="value">{profile?.name || "Not specified"}</span>
            </div>
            <div className="detail-item">
              <span className="label">Profession</span>
              <span className="value">{profile?.profession || "Not specified"}</span>
            </div>
            <div className="detail-item">
              <span className="label">Technical Stack</span>
              <div className="tags-container">
                {profile?.technical_stack?.length > 0 ? (
                  profile.technical_stack.map((t, idx) => (
                    <span key={idx} className="tech-tag">
                      {t}
                    </span>
                  ))
                ) : (
                  <span className="text-muted">None extracted yet</span>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
