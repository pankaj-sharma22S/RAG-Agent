import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { Sparkles, Lock, Mail, AlertCircle, ArrowRight, UserCheck, Shield } from "lucide-react";

export default function Login() {
  const navigate = useNavigate();
  const { login, continueAsGuest } = useAuth();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    if (!email.trim()) {
      setError("Please enter your email address.");
      return;
    }
    if (!password) {
      setError("Please enter your password.");
      return;
    }

    try {
      setLoading(true);
      const { error } = await login(email, password);
      if (error) {
        setError(error.message || "Invalid email or password.");
        return;
      }
      navigate("/dashboard");
    } catch (err) {
      setError("An unexpected error occurred. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleGuestLogin = () => {
    continueAsGuest();
    navigate("/dashboard");
  };

  return (
    <div className="auth-container">
      <div className="auth-card">
        <div className="auth-header">
          <div className="auth-logo">
            <Sparkles className="icon-accent" size={28} />
          </div>
          <h1>Insight AI Knowledge Center</h1>
          <p>Sign in to your account or launch instant demo access</p>
        </div>

        {error && (
          <div className="auth-alert error">
            <AlertCircle size={18} />
            <span>{error}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="auth-form">
          <div className="form-group">
            <label htmlFor="email">Email Address</label>
            <div className="input-with-icon">
              <Mail className="input-icon" size={18} />
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="name@company.com"
                disabled={loading}
              />
            </div>
          </div>

          <div className="form-group">
            <div className="label-with-action">
              <label htmlFor="password">Password</label>
              <Link to="/forgot-password" className="link-muted">
                Forgot password?
              </Link>
            </div>
            <div className="input-with-icon">
              <Lock className="input-icon" size={18} />
              <input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                disabled={loading}
              />
            </div>
          </div>

          <button type="submit" className="btn-primary full-width" disabled={loading}>
            {loading ? "Signing in..." : "Sign In with Supabase"}
            {!loading && <ArrowRight size={18} />}
          </button>
        </form>

        <div className="auth-divider">
          <span>OR</span>
        </div>

        <button
          type="button"
          onClick={handleGuestLogin}
          className="btn-secondary full-width btn-guest"
        >
          <UserCheck size={18} className="icon-accent" />
          <span>Launch Workspace (Instant Guest Access)</span>
        </button>

        <div className="auth-footer">
          <p>
            Don't have an account?{" "}
            <Link to="/signup" className="link-accent">
              Create an account
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}