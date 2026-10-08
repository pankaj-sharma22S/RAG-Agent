import { Outlet } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import AuthLoading from "./AuthLoading";

function PublicRoute() {
  const { loading } = useAuth();

  if (loading) {
    return <AuthLoading />;
  }

  return <Outlet />;
}

export default PublicRoute;