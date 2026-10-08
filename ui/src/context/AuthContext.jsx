import { createContext, useContext, useEffect, useState } from "react";
import { supabase } from "../lib/supabase";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let mounted = true;

    const initializeAuth = async () => {
      try {
        // Check for local guest session first
        const savedGuest = localStorage.getItem("insight_ai_guest_user");
        if (savedGuest) {
          try {
            const guestData = JSON.parse(savedGuest);
            if (mounted) {
              setUser(guestData);
              setSession({ access_token: "guest_demo_jwt_token", user: guestData });
              setLoading(false);
              return;
            }
          } catch (e) {
            localStorage.removeItem("insight_ai_guest_user");
          }
        }

        const {
          data: { session },
          error,
        } = await supabase.auth.getSession();

        if (error) {
          console.warn("Supabase session check:", error.message);
        }

        if (!mounted) return;

        setSession(session);
        setUser(session?.user ?? null);
      } catch (error) {
        console.warn("Auth initialization fallback:", error);
      } finally {
        if (mounted) {
          setLoading(false);
        }
      }
    };

    initializeAuth();

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((event, session) => {
      setSession(session);
      setUser(session?.user ?? null);
      if (session) {
        localStorage.removeItem("insight_ai_guest_user");
      }
    });

    return () => {
      mounted = false;
      subscription?.unsubscribe?.();
    };
  }, []);

  const login = async (email, password) => {
    const { data, error } = await supabase.auth.signInWithPassword({
      email: email.trim(),
      password,
    });
    if (!error) {
      localStorage.removeItem("insight_ai_guest_user");
    }
    return { data, error };
  };

  const signup = async (email, password) => {
    const { data, error } = await supabase.auth.signUp({
      email: email.trim(),
      password,
    });
    return { data, error };
  };

  const continueAsGuest = (guestName = "Demo Analyst") => {
    const guestUser = {
      id: "guest_" + Math.random().toString(36).substring(2, 9),
      email: "demo.analyst@insight.ai",
      role: "guest_analyst",
      user_metadata: { name: guestName },
    };
    localStorage.setItem("insight_ai_guest_user", JSON.stringify(guestUser));
    setUser(guestUser);
    setSession({ access_token: "guest_demo_token", user: guestUser });
    return guestUser;
  };

  const logout = async () => {
    localStorage.removeItem("insight_ai_guest_user");
    try {
      await supabase.auth.signOut();
    } catch (e) {
      // Ignore if offline
    }
    setSession(null);
    setUser(null);
    return { error: null };
  };

  const resetPassword = async (email) => {
    const { data, error } = await supabase.auth.resetPasswordForEmail(email.trim(), {
      redirectTo: `${window.location.origin}/reset-password`,
    });
    return { data, error };
  };

  const updatePassword = async (newPassword) => {
    const { data, error } = await supabase.auth.updateUser({
      password: newPassword,
    });
    return { data, error };
  };

  const value = {
    user,
    session,
    token: session?.access_token || null,
    loading,
    login,
    signup,
    continueAsGuest,
    logout,
    resetPassword,
    updatePassword,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}