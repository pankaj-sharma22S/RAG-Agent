"""Comprehensive test suite for the FastAPI API layer, authentication, routing, error handling, and streaming."""

import os
import sys
import time
import jwt
from fastapi.testclient import TestClient

current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.dirname(current_dir)
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

from core.config import settings
from main import app

client = TestClient(app)


def generate_test_jwt(sub: str = "test_user_123", email: str = "test@example.com", role: str = "authenticated", exp_seconds: int = 3600, secret: str = "test_jwt_secret_key_1234567890123456") -> str:
    """Generate a test JWT matching Supabase token structure."""
    payload = {
        "sub": sub,
        "email": email,
        "role": role,
        "aud": "authenticated",
        "app_metadata": {"role": role},
        "user_metadata": {"name": "Test User"},
        "exp": int(time.time()) + exp_seconds,
        "iat": int(time.time()),
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def test_root_and_health_endpoints():
    print("\n--- 1. Testing Root and Health Endpoints ---")
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert data["api_v1"] == "/api/v1"
    assert "X-Request-ID" in res.headers
    assert "X-Response-Time-Ms" in res.headers
    print("  Root [/] -> PASS (Request-ID: %s)" % res.headers["X-Request-ID"])

    res_health = client.get("/api/v1/health")
    assert res_health.status_code == 200
    hdata = res_health.json()
    assert hdata["status"] in {"healthy", "degraded"}
    assert "vector_store" in hdata["services"]
    assert "database" in hdata["services"]
    assert "llm_gateway" in hdata["services"]
    print("  Health [/api/v1/health] -> PASS (Overall: %s)" % hdata["status"])


def test_authentication_and_authorization():
    print("\n--- 2. Testing Authentication & Authorization ---")
    # A. Missing token on protected route -> 401
    res = client.get("/api/v1/auth/me")
    assert res.status_code == 401
    err = res.json()["error"]
    assert err["code"] == "AUTHENTICATION_FAILED"
    print("  Missing token -> 401 PASS (%s)" % err["message"])

    # B. Invalid / Malformed token -> 401
    res_bad = client.get("/api/v1/auth/me", headers={"Authorization": "Bearer invalid.garbage.token"})
    assert res_bad.status_code == 401
    assert res_bad.json()["error"]["code"] == "AUTHENTICATION_FAILED"
    print("  Invalid token -> 401 PASS")

    # C. Expired token -> 401
    expired_token = generate_test_jwt(exp_seconds=-300)
    res_exp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {expired_token}"})
    assert res_exp.status_code == 401
    assert "expired" in res_exp.json()["error"]["message"].lower()
    print("  Expired token -> 401 PASS")

    # D. Valid Token -> 200 with extracted sub
    valid_token = generate_test_jwt(sub="user_secure_999", email="alice@test.com")
    res_valid = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {valid_token}"})
    assert res_valid.status_code == 200
    user_data = res_valid.json()
    assert user_data["user_id"] == "user_secure_999"
    assert user_data["email"] == "alice@test.com"
    assert user_data["is_anonymous"] is False
    print("  Valid token -> 200 PASS (Verified User: %s)" % user_data["user_id"])


def test_validation_and_error_handling():
    print("\n--- 3. Testing Validation and Centralized Error Handling ---")
    # A. Missing required fields in ChatRequest -> 422 with structured envelope
    res = client.post("/api/v1/chat", json={})
    assert res.status_code == 422
    err_env = res.json()["error"]
    assert err_env["code"] == "VALIDATION_ERROR"
    assert len(err_env["details"]) > 0
    assert "X-Request-ID" in res.headers
    print("  Empty chat payload -> 422 Validation Error PASS")

    # B. Non-existent route -> 404 with structured envelope
    res_404 = client.get("/api/v1/non_existent_endpoint_xyz")
    assert res_404.status_code == 404
    assert res_404.json()["error"]["code"] == "NOT_FOUND"
    print("  Non-existent route -> 404 Not Found PASS")


def test_chat_and_streaming_endpoints():
    print("\n--- 4. Testing Chat, Token Streaming, and SSE Endpoints ---")
    valid_token = generate_test_jwt(sub="user_chat_tester")
    headers = {"Authorization": f"Bearer {valid_token}"}

    # A. Standard Non-Streaming Chat
    res_chat = client.post(
        "/api/v1/chat",
        json={"message": "hello", "thread_id": "test_thread_api_01"},
        headers=headers,
    )
    assert res_chat.status_code == 200
    chat_data = res_chat.json()
    assert "answer" in chat_data
    assert chat_data["thread_id"] == "test_thread_api_01"
    assert chat_data["latency_ms"] > 0
    print("  Standard Chat -> 200 PASS (Answer: %s, Route: %s)" % (chat_data["answer"][:40], chat_data["route"]))

    # B. Plain Token Streaming
    res_stream = client.post(
        "/api/v1/chat/stream",
        json={"message": "hello", "thread_id": "test_thread_api_stream"},
        headers=headers,
    )
    assert res_stream.status_code == 200
    assert len(res_stream.text) > 0
    print("  Token Stream -> 200 PASS (Stream length: %d chars)" % len(res_stream.text))

    # C. Server-Sent Events (SSE)
    res_sse = client.post(
        "/api/v1/chat/sse",
        json={"message": "hello", "thread_id": "test_thread_api_sse"},
        headers=headers,
    )
    assert res_sse.status_code == 200
    sse_text = res_sse.text
    assert "event: status" in sse_text
    assert "event: token" in sse_text or "event: done" in sse_text
    print("  SSE Stream -> 200 PASS (Contains event: status and done)")


def test_threads_and_memory_endpoints():
    print("\n--- 5. Testing Threads and Memory Endpoints ---")
    valid_token = generate_test_jwt(sub="user_memory_tester")
    headers = {"Authorization": f"Bearer {valid_token}"}

    # List threads for user
    res_threads = client.get("/api/v1/threads", headers=headers)
    assert res_threads.status_code == 200
    assert "threads" in res_threads.json()
    print("  List User Threads -> 200 PASS")

    # Get user profile memory
    res_profile = client.get("/api/v1/profile", headers=headers)
    assert res_profile.status_code == 200
    prof = res_profile.json()
    assert prof["user_id"] == "user_memory_tester"
    print("  Get Profile Memory -> 200 PASS")


def test_github_and_gateway_endpoints():
    print("\n--- 6. Testing GitHub and Gateway Stats Endpoints ---")
    # GitHub Analyzer Status
    res_gh = client.get("/api/v1/github/status")
    assert res_gh.status_code == 200
    assert res_gh.json()["status"] == "online"
    print("  GitHub Analyzer Status -> 200 PASS")

    # Gateway Stats
    res_gw = client.get("/api/v1/gateway/stats")
    assert res_gw.status_code == 200
    gw_data = res_gw.json()
    assert "cost_and_usage" in gw_data
    assert "cache" in gw_data
    print("  Gateway Stats -> 200 PASS")


def test_backward_compatibility_routes():
    print("\n--- 7. Testing Backward Compatibility with Legacy /api Routes ---")
    res_leg_gh = client.get("/api/github/status")
    assert res_leg_gh.status_code == 200

    res_leg_gw = client.get("/api/gateway/stats")
    assert res_leg_gw.status_code == 200
    print("  Legacy /api/github/status & /api/gateway/stats -> 200 PASS")


if __name__ == "__main__":
    print("=" * 90)
    print("               FASTAPI PRODUCTION LAYER VERIFICATION TEST SUITE               ")
    print("=" * 90)
    test_root_and_health_endpoints()
    test_authentication_and_authorization()
    test_validation_and_error_handling()
    test_chat_and_streaming_endpoints()
    test_threads_and_memory_endpoints()
    test_github_and_gateway_endpoints()
    test_backward_compatibility_routes()
    print("\n" + "=" * 90)
    print("             ALL FASTAPI PRODUCTION LAYER TESTS VALIDATED 100%!               ")
    print("=" * 90)
