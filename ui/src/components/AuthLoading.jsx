import { Sparkles } from "lucide-react";

export default function AuthLoading() {
  return (
    <div className="auth-loading-screen">
      <div className="loading-card">
        <div className="auth-logo pulse">
          <Sparkles className="icon-accent" size={32} />
        </div>
        <h3>Loading Insight AI...</h3>
        <p>Connecting to knowledge engine</p>
        <div className="spinner-bar">
          <div className="spinner-bar-inner" />
        </div>
      </div>
    </div>
  );
}