"""Garde-fous de main.py : cache de /predict/today et protection de /agent/chat.

Les dépendances lourdes absentes de l'environnement de test (pandas, groq, psycopg2, ...) sont
remplacées par des doublures vides ; si elles sont installées, les vraies sont utilisées.
"""
import importlib.util
import os
import sys
import threading
import time
import types
import unittest
from unittest import mock

from fastapi import HTTPException

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _stub_missing_modules():
    # Paquets tiers absents de l'environnement de test.
    for name in ("pandas", "groq", "psycopg2", "psycopg2.extras", "sklearn", "lightgbm"):
        if name in sys.modules:
            continue
        try:
            if importlib.util.find_spec(name) is not None:
                continue
        except (ImportError, ValueError):
            pass
        sys.modules[name] = types.ModuleType(name)
    if not hasattr(sys.modules["groq"], "APIError"):
        sys.modules["groq"].APIError = type("APIError", (Exception,), {})
        sys.modules["groq"].Groq = object
    # Modules du dépôt : réels si leurs dépendances sont là, sinon doublures minimales.
    for name, attrs in (
        ("db_conn", {"get_conn": lambda: None}),
        ("scoriq_models", {"MIN_PROBA": 0.02, "plancher_probas": lambda *a, **k: a}),
    ):
        try:
            importlib.import_module(name)
        except Exception:
            mod = types.ModuleType(name)
            for k, v in attrs.items():
                setattr(mod, k, v)
            sys.modules[name] = mod


_stub_missing_modules()
import main  # noqa: E402
import security  # noqa: E402


class PredictTodayCacheTests(unittest.TestCase):
    def setUp(self):
        main._predict_today_cache.update(day=None, at=0.0, value=None)

    def test_computed_once_within_ttl_then_again_after(self):
        with mock.patch.object(main, "_compute_predict_today", side_effect=[{"count": 1}, {"count": 2}]) as comp:
            self.assertEqual(main.predict_today(), {"count": 1})
            self.assertEqual(main.predict_today(), {"count": 1})
            self.assertEqual(comp.call_count, 1)
            main._predict_today_cache["at"] -= main.PREDICT_TODAY_TTL + 1
            self.assertEqual(main.predict_today(), {"count": 2})
            self.assertEqual(comp.call_count, 2)

    def test_new_day_invalidates(self):
        with mock.patch.object(main, "_compute_predict_today", side_effect=[{"d": 1}, {"d": 2}]) as comp:
            main.predict_today()
            main._predict_today_cache["day"] = "1999-01-01"
            self.assertEqual(main.predict_today(), {"d": 2})
            self.assertEqual(comp.call_count, 2)

    def test_error_is_not_cached(self):
        with mock.patch.object(main, "_compute_predict_today", side_effect=[RuntimeError("db"), {"ok": True}]):
            with self.assertRaises(RuntimeError):
                main.predict_today()
            self.assertEqual(main.predict_today(), {"ok": True})

    def test_concurrent_visitors_trigger_a_single_computation(self):
        calls = []

        def slow():
            calls.append(1)
            time.sleep(0.2)
            return {"count": 7}

        results = []
        with mock.patch.object(main, "_compute_predict_today", side_effect=slow):
            threads = [threading.Thread(target=lambda: results.append(main.predict_today())) for _ in range(8)]
            [t.start() for t in threads]
            [t.join() for t in threads]
        self.assertEqual(len(calls), 1)
        self.assertEqual(results, [{"count": 7}] * 8)


class AgentChatGuardTests(unittest.TestCase):
    def setUp(self):
        security._token_cache.clear()
        security._validation_budget = security.SlidingWindowLimiter(security.VALIDATIONS_PER_MINUTE, 60)
        security._agent_limiter = security.SlidingWindowLimiter(security.AGENT_REQUESTS, security.AGENT_WINDOW)

    def call(self, header, messages=None):
        req = main.AgentChatRequest(messages=messages or [{"role": "user", "content": "salut"}])
        return main.agent_chat(req, authorization=header)

    def test_fake_bearer_never_reaches_groq(self):
        with mock.patch.object(security, "_fetch_user_id", return_value=None), \
             mock.patch.object(main, "get_groq_client", side_effect=AssertionError("Groq appelé")):
            with self.assertRaises(HTTPException) as ctx:
                self.call("Bearer x")
            self.assertEqual(ctx.exception.status_code, 401)

    def test_no_header_is_401(self):
        with mock.patch.object(main, "get_groq_client", side_effect=AssertionError("Groq appelé")):
            with self.assertRaises(HTTPException) as ctx:
                self.call(None)
            self.assertEqual(ctx.exception.status_code, 401)

    def test_valid_user_reaches_groq(self):
        class Reached(Exception):
            pass

        with mock.patch.object(security, "_fetch_user_id", return_value="u-1"), \
             mock.patch.object(main, "get_groq_client", side_effect=Reached):
            with self.assertRaises(Reached):
                self.call("Bearer good")

    def test_quota_and_size_are_enforced_before_groq(self):
        security._agent_limiter = security.SlidingWindowLimiter(1, 600)

        class Reached(Exception):
            pass

        with mock.patch.object(security, "_fetch_user_id", return_value="u-1"), \
             mock.patch.object(main, "get_groq_client", side_effect=Reached):
            with self.assertRaises(Reached):
                self.call("Bearer good")
            with self.assertRaises(HTTPException) as ctx:
                self.call("Bearer good")
            self.assertEqual(ctx.exception.status_code, 429)
            security._agent_limiter = security.SlidingWindowLimiter(100, 600)
            big = [{"role": "user", "content": "x" * (security.AGENT_MAX_CHARS + 1)}]
            with self.assertRaises(HTTPException) as ctx:
                self.call("Bearer good", big)
            self.assertEqual(ctx.exception.status_code, 413)


if __name__ == "__main__":
    unittest.main()
