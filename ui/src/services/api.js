/**
 * Centralized API Service Layer connecting React to the FastAPI backend.
 * Automatically injects Supabase JWT Bearer token, correlates request IDs,
 * handles structured error envelopes, and supports Server-Sent Events (SSE).
 */

import { supabase } from "../lib/supabase";

const configuredApiBaseUrl = import.meta.env.VITE_API_BASE_URL;
const apiBaseUrl = configuredApiBaseUrl
  ? new URL(configuredApiBaseUrl, window.location.origin)
  : null;
const loopbackHosts = ["localhost", "127.0.0.1", "::1", "[::1]"];

if (
  apiBaseUrl &&
  loopbackHosts.includes(apiBaseUrl.hostname) &&
  !loopbackHosts.includes(window.location.hostname)
) {
  apiBaseUrl.hostname = window.location.hostname;
}

export const API_BASE_URL = apiBaseUrl
  ? apiBaseUrl.toString().replace(/\/$/, "")
  : "/api/v1";
export const BACKEND_SERVER_URL = API_BASE_URL.startsWith("http")
  ? API_BASE_URL.replace(/\/api\/v1\/?$/, "")
  : "";

/**
 * Robustly resolve image URLs (handling relative /api paths, absolute URLs, and data URIs)
 */
export function resolveImageUrl(url) {
  if (!url || typeof url !== "string") return "";
  const trimmed = url.trim();
  if (
    trimmed.startsWith("http://") ||
    trimmed.startsWith("https://") ||
    trimmed.startsWith("data:") ||
    trimmed.startsWith("blob:")
  ) {
    return trimmed;
  }
  const cleanUrl = trimmed.startsWith("/") ? trimmed : `/${trimmed}`;
  return `${BACKEND_SERVER_URL}${cleanUrl}`;
}

/**
 * Extract active JWT access token from Supabase session.
 */
async function getAuthHeader() {
  try {
    const { data: { session } } = await supabase.auth.getSession();
    if (session?.access_token) {
      return { Authorization: `Bearer ${session.access_token}` };
    }
  } catch (err) {
    console.warn("Could not retrieve Supabase session token:", err);
  }
  return {};
}

/**
 * Standard fetch wrapper with centralized error handling.
 */
async function request(endpoint, options = {}) {
  const authHeaders = await getAuthHeader();
  const url = `${API_BASE_URL}${endpoint}`;

  const headers = {
    ...authHeaders,
    ...options.headers,
  };

  // Do not set Content-Type if sending FormData (browser sets boundary)
  if (!(options.body instanceof FormData) && !headers["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }

  const config = {
    ...options,
    headers,
  };

  try {
    const response = await fetch(url, config);

    if (!response.ok) {
      let errorPayload;
      try {
        errorPayload = await response.json();
      } catch {
        errorPayload = { error: { message: `HTTP ${response.status}: ${response.statusText}` } };
      }

      const errorMessage =
        errorPayload?.error?.message ||
        errorPayload?.detail ||
        `Request failed with status ${response.status}`;

      const error = new Error(errorMessage);
      error.status = response.status;
      error.code = errorPayload?.error?.code || "HTTP_ERROR";
      error.details = errorPayload?.error?.details || null;
      error.requestId = errorPayload?.error?.request_id || response.headers.get("X-Request-ID");
      throw error;
    }

    return await response.json();
  } catch (err) {
    console.error(`API Error on [${options.method || "GET"} ${endpoint}]:`, err);
    if (err.name === "TypeError" && (err.message === "Failed to fetch" || err.message.includes("fetch"))) {
      const netErr = new Error("Unable to connect to the backend server. Please verify the server is running on port 8000.");
      netErr.status = 0;
      netErr.code = "NETWORK_ERROR";
      throw netErr;
    }
    throw err;
  }
}

export const api = {
  // ==========================================
  // Health & Diagnostics
  // ==========================================
  async getHealth() {
    return request("/health");
  },

  async getGatewayStats() {
    return request("/gateway/stats");
  },

  // ==========================================
  // Authentication & Identity
  // ==========================================
  async getMe() {
    return request("/auth/me");
  },

  async getUserProfile() {
    return request("/profile");
  },

  // ==========================================
  // Agentic Chat & Streaming
  // ==========================================
  async sendMessage({ message, thread_id, active_source }) {
    return request("/chat", {
      method: "POST",
      body: JSON.stringify({ message, thread_id, active_source }),
    });
  },

  /**
   * Stream agent response using Server-Sent Events (SSE).
   * Calls onStatus, onToken, onDone, and onError handlers.
   */
  async streamChatSSE({ message, thread_id, active_source, onStatus, onToken, onDone, onError, signal }) {
    const authHeaders = await getAuthHeader();
    const url = `${API_BASE_URL}/chat/sse`;

    try {
      const response = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...authHeaders,
        },
        body: JSON.stringify({ message, thread_id, active_source }),
        signal,
      });

      if (!response.ok) {
        const errJson = await response.json().catch(() => ({}));
        throw new Error(errJson?.error?.message || `SSE failed with HTTP ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";

        for (const block of lines) {
          if (!block.trim()) continue;

          let eventType = "message";
          let dataStr = "";

          const blockLines = block.split("\n");
          for (const line of blockLines) {
            if (line.startsWith("event: ")) {
              eventType = line.slice(7).trim();
            } else if (line.startsWith("data: ")) {
              dataStr = line.slice(6).trim();
            }
          }

          if (!dataStr) continue;

          try {
            const data = JSON.parse(dataStr);
            if (eventType === "status" && onStatus) {
              onStatus(data);
            } else if (eventType === "token" && onToken) {
              onToken(data.content || "");
            } else if (eventType === "done" && onDone) {
              onDone(data);
            } else if (eventType === "error" && onError) {
              onError(data);
            }
          } catch (parseErr) {
            console.warn("Failed to parse SSE data frame:", parseErr, dataStr);
          }
        }
      }
    } catch (err) {
      if (err.name === "AbortError") {
        console.log("Chat SSE stream aborted by user.");
        return;
      }
      if (onError) onError(err);
      throw err;
    }
  },

  // ==========================================
  // Ingestion & Documents
  // ==========================================
  async uploadFile(file, thread_id = "default") {
    const formData = new FormData();
    formData.append("file", file);
    return request(`/upload?thread_id=${encodeURIComponent(thread_id)}`, {
      method: "POST",
      body: formData,
    });
  },

  async ingestUrl(url, thread_id = "default") {
    return request("/url", {
      method: "POST",
      body: JSON.stringify({ url, thread_id }),
    });
  },

  async queryDocuments(question, thread_id = "default") {
    return request("/query", {
      method: "POST",
      body: JSON.stringify({ question, thread_id }),
    });
  },

  // ==========================================
  // GitHub Analysis
  // ==========================================
  async analyzeGitHub({ repo_url, focus_query = "", branch = null, thread_id = "default" }) {
    return request("/github/analyze", {
      method: "POST",
      body: JSON.stringify({ repo_url, focus_query, branch, thread_id }),
    });
  },

  async getGitHubStatus() {
    return request("/github/status");
  },

  async listThreads(limit = 50, offset = 0) {
    return request(`/threads?limit=${limit}&offset=${offset}`);
  },

  async getThread(thread_id) {
    return request(`/threads/${encodeURIComponent(thread_id)}`);
  },

  async deleteThread(thread_id) {
    return request(`/threads/${encodeURIComponent(thread_id)}`, {
      method: "DELETE",
    });
  },

  async renameThread(thread_id, title) {
    return request(`/threads/${encodeURIComponent(thread_id)}`, {
      method: "PATCH",
      body: JSON.stringify({ title }),
    });
  },

  async getThreadSuggestions(thread_id = "default") {
    return request(`/threads/${encodeURIComponent(thread_id)}/suggestions`);
  },
};
