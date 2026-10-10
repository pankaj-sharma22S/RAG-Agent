import { createContext, useContext, useEffect, useState } from "react";

export const ACCENT_THEMES = [
  { id: "violet", label: "Violet Iris", color: "#7c3aed" },
  { id: "emerald", label: "Emerald Matrix", color: "#10b981" },
  { id: "cyan", label: "Cyan Electric", color: "#06b6d4" },
  { id: "blue", label: "Ocean Blue", color: "#3b82f6" },
  { id: "amber", label: "Amber Sunset", color: "#f59e0b" },
  { id: "rose", label: "Rose Crimson", color: "#f43f5e" },
];

const ThemeContext = createContext({
  theme: "dark",
  accent: "violet",
  toggleTheme: () => {},
  setTheme: () => {},
  setAccent: () => {},
});

export function ThemeProvider({ children }) {
  const [theme, setThemeState] = useState(() => {
    const saved = localStorage.getItem("insight_ai_theme");
    if (saved === "light" || saved === "dark") return saved;
    return window.matchMedia?.("(prefers-color-scheme: light)").matches ? "light" : "dark";
  });

  const [accent, setAccentState] = useState(() => {
    const saved = localStorage.getItem("insight_ai_accent");
    return saved || "violet";
  });

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("insight_ai_theme", theme);
  }, [theme]);

  useEffect(() => {
    document.documentElement.setAttribute("data-accent", accent);
    localStorage.setItem("insight_ai_accent", accent);
  }, [accent]);

  const toggleTheme = () => {
    setThemeState((prev) => (prev === "dark" ? "light" : "dark"));
  };

  const setTheme = (newTheme) => {
    if (newTheme === "light" || newTheme === "dark") {
      setThemeState(newTheme);
    }
  };

  const setAccent = (newAccent) => {
    if (ACCENT_THEMES.some((t) => t.id === newAccent)) {
      setAccentState(newAccent);
    }
  };

  return (
    <ThemeContext.Provider value={{ theme, accent, toggleTheme, setTheme, setAccent }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  return useContext(ThemeContext);
}
