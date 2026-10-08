import { useLocation, useNavigate, Link } from "react-router-dom";
import ResendVerification from "../components/ResendVerification";
import { Sparkles, MailCheck, ArrowRight } from "lucide-react";

export default function VerifyEmail() {
  const location = useLocation();
  const navigate = useNavigate();
  const email = location.state?.email;

  return (
    <div className="auth-container">
      <div className="auth-card text-center">
        <div className="auth-header">
          <div className="auth-logo">
            <MailCheck className="icon-accent" size={32} />
          </div>
          <h1>Verify your email</h1>
          <p>
            We've sent a verification link to{" "}
            {email ? <strong className="text-white">{email}</strong> : "your inbox"}.
          </p>
        </div>

        <div className="auth-info-box">
          <p>
            Please check your inbox and click the verification link to activate your Insight AI account.
          </p>
        </div>

        <ResendVerification email={email} />

        <div className="auth-footer" style={{ marginTop: "1.5rem" }}>
          <button onClick={() => navigate("/login")} className="btn-primary full-width">
            Proceed to Sign In <ArrowRight size={16} />
          </button>
        </div>
      </div>
    </div>
  );
}
