/**
 * Centralized API Service Layer connecting React to the FastAPI backend.
 * Automatically injects Supabase JWT Bearer token, correlates request IDs,
 * handles structured error envelopes, and supports Server-Sent Events (SSE).
 */

import { supabase } from "../lib/supabase";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api/v1";

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

  // ==========================================
  // Threads & Conversation History
  // ==========================================
  async listThreads(limit = 20, offset = 0) {
    return request(`/threads?limit=${limit}&offset=${offset}`);
  },

  async getThread(thread_id) {
    return request(`/threads/${encodeURIComponent(thread_id)}`);
  },
};
