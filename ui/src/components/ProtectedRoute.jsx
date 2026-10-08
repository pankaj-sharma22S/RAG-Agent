import { useEffect } from "react";
import { Outlet } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import AuthLoading from "./AuthLoading";

function ProtectedRoute() {
  const { user, loading, continueAsGuest } = useAuth();

  useEffect(() => {
    if (!loading && !user) {
      continueAsGuest("Demo Analyst");
    }
  }, [loading, user, continueAsGuest]);

  if (loading) {
    return <AuthLoading />;
  }

  return <Outlet />;
}

export default ProtectedRoute;