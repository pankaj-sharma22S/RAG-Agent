import { useState } from "react";
import { supabase } from "../lib/supabase";
import { getAuthErrorMessage } from "../utils/authErrors";
import { RefreshCw, CheckCircle, AlertCircle } from "lucide-react";

function ResendVerification({ email }) {
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  const handleResend = async () => {
    setMessage("");
    setError("");

    if (!email) {
      setError("Email address is required.");
      return;
    }

    try {
      setLoading(true);

      const { error } = await supabase.auth.resend({
        type: "signup",
        email: email.trim(),
      });

      if (error) {
        setError(getAuthErrorMessage(error));
        return;
      }

      setMessage("If the account exists, a verification email has been sent.");
    } catch (err) {
      console.error("Resend verification error:", err);
      setError("Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ marginTop: "1rem" }}>
      {message && (
        <div className="auth-alert success" style={{ marginBottom: "0.75rem" }}>
          <CheckCircle size={16} />
          <span>{message}</span>
        </div>
      )}

      {error && (
        <div className="auth-alert error" style={{ marginBottom: "0.75rem" }}>
          <AlertCircle size={16} />
          <span>{error}</span>
        </div>
      )}

      <button
        type="button"
        onClick={handleResend}
        disabled={loading || !email}
        className="btn-secondary full-width"
      >
        <RefreshCw size={15} className={loading ? "spin" : ""} />
        <span>{loading ? "Sending link..." : "Resend Verification Email"}</span>
      </button>
    </div>
  );
}

export default ResendVerification;