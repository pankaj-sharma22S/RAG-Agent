"""Comprehensive verification suite for Redis rate limiting, semantic caching, and non-critical degradation."""

import hashlib
import json
import time
import uuid
import fakeredis
import numpy as np
from fastapi.testclient import TestClient

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.dirname(current_dir)
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

from core.config import settings
from main import app
from services.redis_service import RedisService, redis_service

client = TestClient(app)


def test_redis_service_non_critical_fallback_when_disconnected():
    """Verify that when Redis is disconnected, operations gracefully fail-open without raising errors."""
    print("\n--- 1. Testing Redis Non-Critical Fallback (Disconnected State) ---")
    disconnected_service = RedisService()
    # Mock unreachable port
    disconnected_service._client = None
    
    # A. Rate limit check should fail-open
    allowed, remaining, retry_after = disconnected_service.check_rate_limit(
        identifier="test_user_offline",
        endpoint="chat",
        max_requests=5,
        window_seconds=60,
    )
    assert allowed is True
    assert remaining == 5
    assert retry_after == 0.0
    print("  Rate limit fallback when Redis disconnected -> PASS (allowed=True)")

    # B. Semantic cache lookup should gracefully return None
    cached = disconnected_service.get_semantic_cache(
        query="what is artificial intelligence?",
        context={"user_id": "test_user_offline"},
    )
    assert cached is None
    print("  Semantic cache GET fallback when Redis disconnected -> PASS (returned None)")

    # C. Semantic cache write should gracefully return False without crashing
    saved = disconnected_service.set_semantic_cache(
        query="what is artificial intelligence?",
        response_data={"answer": "AI is the simulation of human intelligence."},
        context={"user_id": "test_user_offline"},
    )
    assert saved is False
    print("  Semantic cache SET fallback when Redis disconnected -> PASS (returned False)")


def test_redis_rate_limiting_with_active_redis():
    """Verify sliding-window rate limiting with an active Redis instance."""
    print("\n--- 2. Testing Redis Sliding-Window Rate Limiting ---")
    fake_client = fakeredis.FakeRedis(decode_responses=True)
    svc = RedisService()
    svc._client = fake_client

    user_id = f"user_limiter_{uuid.uuid4().hex[:6]}"

    # Test limit of 3 requests per 60 seconds
    # Request 1
    allowed1, rem1, retry1 = svc.check_rate_limit(user_id, endpoint="test_ep", max_requests=3, window_seconds=60)
    assert allowed1 is True
    assert rem1 == 2
    assert retry1 == 0.0

    # Request 2
    allowed2, rem2, retry2 = svc.check_rate_limit(user_id, endpoint="test_ep", max_requests=3, window_seconds=60)
    assert allowed2 is True
    assert rem2 == 1

    # Request 3
    allowed3, rem3, retry3 = svc.check_rate_limit(user_id, endpoint="test_ep", max_requests=3, window_seconds=60)
    assert allowed3 is True
    assert rem3 == 0

    # Request 4 -> Should be REJECTED (HTTP 429 condition)
    allowed4, rem4, retry4 = svc.check_rate_limit(user_id, endpoint="test_ep", max_requests=3, window_seconds=60)
    assert allowed4 is False
    assert rem4 == 0
    assert retry4 > 0.0
    print("  Rate limit 3/3 exceeded on 4th call -> PASS (allowed=False, retry_after=%.2fs)" % retry4)


def test_redis_semantic_caching_and_similarity():
    """Verify semantic caching similarity matching and context isolation."""
    print("\n--- 3. Testing Semantic Response Caching & Context Isolation ---")
    fake_client = fakeredis.FakeRedis(decode_responses=True)
    svc = RedisService()
    svc._client = fake_client

    user_a = "user_alpha"
    user_b = "user_beta"

    q1 = "What is the capital of France?"
    ans1 = {"answer": "The capital of France is Paris."}

    # 1. Initially Cache MISS
    miss = svc.get_semantic_cache(query=q1, context={"user_id": user_a, "endpoint": "chat"})
    assert miss is None
    print("  Initial query -> CACHE MISS PASS")

    # 2. Store response in Redis semantic cache
    stored = svc.set_semantic_cache(
        query=q1,
        response_data=ans1,
        context={"user_id": user_a, "endpoint": "chat"},
        ttl_seconds=3600,
    )
    assert stored is True
    print("  Stored in Redis semantic cache -> PASS")

    # 3. Exact Query HIT
    hit_exact = svc.get_semantic_cache(query=q1, context={"user_id": user_a, "endpoint": "chat"})
    assert hit_exact is not None
    assert hit_exact["answer"] == ans1["answer"]
    print("  Exact query -> CACHE HIT PASS: '%s'" % hit_exact["answer"])

    # 4. Semantically Similar Query HIT (Threshold >= 0.88)
    q1_similar = "Can you tell me the capital city of France?"
    hit_similar = svc.get_semantic_cache(
        query=q1_similar,
        context={"user_id": user_a, "endpoint": "chat"},
        threshold=0.88,
    )
    assert hit_similar is not None
    assert hit_similar["answer"] == ans1["answer"]
    print("  Semantically similar query -> CACHE HIT PASS (Query: '%s')" % q1_similar)

    # 5. Context Isolation: User B query should NOT return User A's private cache
    user_b_lookup = svc.get_semantic_cache(query=q1, context={"user_id": user_b, "endpoint": "chat"})
    assert user_b_lookup is None
    print("  Context Isolation (User A vs User B) -> PASS (No cross-user cache leakage)")

    # 6. Semantically Disparate Query MISS
    q_diff = "How do quantum computers execute superposition?"
    diff_lookup = svc.get_semantic_cache(query=q_diff, context={"user_id": user_a, "endpoint": "chat"})
    assert diff_lookup is None
    print("  Semantically different query -> CACHE MISS PASS")


def test_fastapi_chat_flow_with_redis():
    """Verify end-to-end FastAPI chat endpoint request flow with Redis rate-limiting and semantic caching."""
    print("\n--- 4. Testing FastAPI /api/v1/chat Request Flow with Redis ---")
    fake_client = fakeredis.FakeRedis(decode_responses=True)
    redis_service._client = fake_client

    test_uid = "user_e2e_tester"
    headers = {"X-Session-ID": test_uid, "X-Dev-User-ID": test_uid}

    # First Call: CACHE MISS -> Processes -> Caches
    res1 = client.post(
        "/api/v1/chat",
        json={"message": "What is Python programming language?", "thread_id": "thread_e2e_01"},
        headers=headers,
    )
    assert res1.status_code == 200
    data1 = res1.json()
    assert "answer" in data1
    assert data1["route"] == "memory_chat"
    print("  1st Request (MISS) -> Status 200, Route: %s, Latency: %.2fms" % (data1["route"], data1["latency_ms"]))

    # Second Call (Same Query): CACHE HIT -> Returns cached instantly with route="semantic_cache"
    res2 = client.post(
        "/api/v1/chat",
        json={"message": "What is Python programming language?", "thread_id": "thread_e2e_02"},
        headers=headers,
    )
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["route"] == "semantic_cache"
    assert data2["answer"] == data1["answer"]
    print("  2nd Request (HIT) -> Status 200, Route: %s, Latency: %.2fms (Skipped LLM)" % (data2["route"], data2["latency_ms"]))

    # Rate Limit Exceeded Test: Set strict limit on endpoint
    for _ in range(settings.DEFAULT_RATE_LIMIT_PER_MINUTE):
        redis_service.check_rate_limit(test_uid, endpoint="chat")

    res_blocked = client.post(
        "/api/v1/chat",
        json={"message": "Are you there?", "thread_id": "thread_e2e_blocked"},
        headers=headers,
    )
    assert res_blocked.status_code == 429
    err = res_blocked.json()["error"]
    assert err["code"] == "RATE_LIMIT_EXCEEDED"
    print("  Rate Limit Exceeded Request -> HTTP 429 PASS (Code: %s)" % err["code"])


if __name__ == "__main__":
    print("=" * 90)
    print("               REDIS INTEGRATION & VERIFICATION TEST SUITE                    ")
    print("=" * 90)
    test_redis_service_non_critical_fallback_when_disconnected()
    test_redis_rate_limiting_with_active_redis()
    test_redis_semantic_caching_and_similarity()
    test_fastapi_chat_flow_with_redis()
    print("\n" + "=" * 90)
    print("             ALL REDIS INTEGRATION TESTS PASSED 100% SUCCESSFUL!               ")
    print("=" * 90)
