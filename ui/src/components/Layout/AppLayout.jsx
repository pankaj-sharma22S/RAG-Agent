import { Outlet, useLocation } from "react-router-dom";
import Sidebar from "./Sidebar";
import Header from "./Header";

export default function AppLayout() {
  const location = useLocation();
  const isChat = location.pathname === "/chat";

  return (
    <div className="app-shell">
      <Sidebar />
      <div className="app-main">
        <Header />
        <main className={`content-container ${isChat ? "chat-full-bleed" : ""}`}>
          <Outlet />
        </main>
      </div>
    </div>
  );
}
