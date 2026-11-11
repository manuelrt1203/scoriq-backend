import os
import sys
import unittest
from unittest import mock

import requests
from fastapi import HTTPException

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import security  # noqa: E402


class FakeResponse:
    def __init__(self, status=200, body=None):
        self.status_code = status
        self._body = body or {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


def reset_state():
    security._token_cache.clear()
    security._validation_budget = security.SlidingWindowLimiter(security.VALIDATIONS_PER_MINUTE, 60)
    security._agent_limiter = security.SlidingWindowLimiter(security.AGENT_REQUESTS, security.AGENT_WINDOW)


class SlidingWindowTests(unittest.TestCase):
    def test_blocks_after_limit_then_frees(self):
        now = [0.0]
        lim = security.SlidingWindowLimiter(2, 10, clock=lambda: now[0])
        self.assertTrue(lim.allow("a"))
        self.assertTrue(lim.allow("a"))
        self.assertFalse(lim.allow("a"))
        self.assertTrue(lim.allow("b"))  # clés indépendantes
        now[0] = 11
        self.assertTrue(lim.allow("a"))


class AuthenticateTests(unittest.TestCase):
    def setUp(self):
        reset_state()

    def status(self, header):
        try:
            security.authenticate_bearer(header)
        except HTTPException as e:
            return e.status_code
        return 200

    def test_missing_or_malformed_header_is_401_without_calling_supabase(self):
        with mock.patch.object(security.requests, "get") as get:
            for h in (None, "", "Bearer", "Bearer ", "Basic abc", "x" * 10):
                self.assertEqual(self.status(h), 401, h)
            self.assertEqual(self.status("Bearer " + "a" * 5000), 401)
            get.assert_not_called()

    def test_fake_bearer_is_refused(self):
        # Le cas signalé par la revue : « Bearer x » passait avant.
        with mock.patch.object(security.requests, "get", return_value=FakeResponse(401)):
            self.assertEqual(self.status("Bearer x"), 401)

    def test_valid_token_returns_user_id_and_is_cached(self):
        with mock.patch.object(security.requests, "get", return_value=FakeResponse(200, {"id": "u-1"})) as get:
            self.assertEqual(security.authenticate_bearer("Bearer good"), "u-1")
            self.assertEqual(security.authenticate_bearer("bearer good"), "u-1")
            self.assertEqual(get.call_count, 1)
            _, kwargs = get.call_args
            self.assertEqual(kwargs["headers"]["Authorization"], "Bearer good")

    def test_refused_token_is_cached_too(self):
        with mock.patch.object(security.requests, "get", return_value=FakeResponse(401)) as get:
            self.assertEqual(self.status("Bearer bad"), 401)
            self.assertEqual(self.status("Bearer bad"), 401)
            self.assertEqual(get.call_count, 1)

    def test_raw_token_is_not_stored(self):
        with mock.patch.object(security.requests, "get", return_value=FakeResponse(200, {"id": "u-1"})):
            security.authenticate_bearer("Bearer secret-token-value")
        self.assertFalse(any("secret-token-value" in k for k in security._token_cache))

    def test_supabase_down_fails_closed_with_503(self):
        with mock.patch.object(security.requests, "get", side_effect=requests.ConnectionError("down")):
            self.assertEqual(self.status("Bearer t"), 503)
        with mock.patch.object(security.requests, "get", return_value=FakeResponse(500)):
            self.assertEqual(self.status("Bearer t2"), 503)

    def test_response_without_id_is_refused(self):
        with mock.patch.object(security.requests, "get", return_value=FakeResponse(200, {})):
            self.assertEqual(self.status("Bearer t"), 401)

    def test_validation_budget_caps_calls_to_supabase(self):
        security._validation_budget = security.SlidingWindowLimiter(3, 60)
        with mock.patch.object(security.requests, "get", return_value=FakeResponse(401)) as get:
            codes = [self.status(f"Bearer t{i}") for i in range(5)]
            self.assertEqual(codes, [401, 401, 401, 429, 429])
            self.assertEqual(get.call_count, 3)


class QuotaAndPayloadTests(unittest.TestCase):
    def setUp(self):
        reset_state()

    def test_quota_is_per_user(self):
        security._agent_limiter = security.SlidingWindowLimiter(2, 600)
        security.check_agent_quota("a")
        security.check_agent_quota("a")
        with self.assertRaises(HTTPException) as ctx:
            security.check_agent_quota("a")
        self.assertEqual(ctx.exception.status_code, 429)
        security.check_agent_quota("b")

    def test_payload_limits(self):
        security.check_agent_payload([{"role": "user", "content": "salut"}])
        with self.assertRaises(HTTPException) as ctx:
            security.check_agent_payload([{"role": "user", "content": "x"}] * (security.AGENT_MAX_MESSAGES + 1))
        self.assertEqual(ctx.exception.status_code, 413)
        with self.assertRaises(HTTPException) as ctx:
            security.check_agent_payload([{"role": "user", "content": "x" * (security.AGENT_MAX_CHARS + 1)}])
        self.assertEqual(ctx.exception.status_code, 413)


if __name__ == "__main__":
    unittest.main()
