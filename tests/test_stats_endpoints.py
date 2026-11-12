"""/stats/calibration (cache, repli, concurrence) et le champ correct_1x2 de /stats/summary, avec une base simulée."""
import os
import sys
import threading
import time
import unittest
from unittest import mock

from fastapi import HTTPException

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_main_guards  # noqa: E402,F401  (installe les doublures des modules lourds, puis importe main)
import main  # noqa: E402


def rows_80pct_overconfident():
    return [{"proba_home_win": 0.8, "proba_draw": 0.1, "proba_away_win": 0.1, "real_result": "1" if i < 7 else "2",
             "market_proba_home": None, "market_proba_draw": None, "market_proba_away": None} for i in range(10)]


class FakeConn:
    def __init__(self, rows=None, fetchone_values=None):
        self.rows, self.queries, self.closed = rows or [], [], False
        self._fetchone = list(fetchone_values or [])

    def execute(self, sql, params=()):
        self.queries.append(sql)
        return self

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self._fetchone.pop(0)

    def close(self):
        self.closed = True


class CalibrationEndpointTests(unittest.TestCase):
    def setUp(self):
        main._stats_calibration_cache["entry"] = None

    def test_payload_and_sql(self):
        conn = FakeConn(rows_80pct_overconfident())
        with mock.patch.object(main, "get_conn", return_value=conn):
            out = main.stats_calibration()
        self.assertEqual(out["overall"]["n"], 10)
        self.assertEqual(out["calibration"][0]["count"], 10)
        self.assertAlmostEqual(out["calibration"][0]["observed"], 0.7)
        self.assertIsNone(out["market"])
        self.assertIn("evaluation_status = 'OK'", conn.queries[0])
        self.assertIn("proba_home_win", conn.queries[0])
        self.assertTrue(conn.closed)

    def test_cached_within_ttl_then_refreshed(self):
        with mock.patch.object(main, "get_conn", side_effect=lambda: FakeConn(rows_80pct_overconfident())) as gc:
            main.stats_calibration()
            main.stats_calibration()
            self.assertEqual(gc.call_count, 1)
            at, value = main._stats_calibration_cache["entry"]
            main._stats_calibration_cache["entry"] = (at - main.STATS_CALIBRATION_TTL - 1, value)
            main.stats_calibration()
            self.assertEqual(gc.call_count, 2)

    def test_database_failure_without_cache_is_a_503_and_the_lock_is_released(self):
        with mock.patch.object(main, "get_conn", side_effect=RuntimeError("db down")):
            with self.assertRaises(HTTPException) as ctx:
                main.stats_calibration()
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertNotIn("db down", str(ctx.exception.detail))
        self.assertFalse(main._stats_calibration_lock.locked())

    def test_database_failure_serves_the_last_known_value(self):
        with mock.patch.object(main, "get_conn", side_effect=lambda: FakeConn(rows_80pct_overconfident())):
            first = main.stats_calibration()
        at, value = main._stats_calibration_cache["entry"]
        main._stats_calibration_cache["entry"] = (at - main.STATS_CALIBRATION_TTL - 1, value)
        with mock.patch.object(main, "get_conn", side_effect=RuntimeError("db down")):
            self.assertEqual(main.stats_calibration(), first)

    def test_concurrent_visitors_trigger_a_single_query(self):
        def slow_conn():
            time.sleep(0.2)
            return FakeConn(rows_80pct_overconfident())

        results = []
        with mock.patch.object(main, "get_conn", side_effect=slow_conn) as gc:
            threads = [threading.Thread(target=lambda: results.append(main.stats_calibration())) for _ in range(8)]
            [t.start() for t in threads]
            [t.join() for t in threads]
        self.assertEqual(gc.call_count, 1)
        self.assertEqual(len(results), 8)

    def test_cold_visitor_gives_up_instead_of_waiting_forever(self):
        main._stats_calibration_lock.acquire()
        try:
            with mock.patch.object(main, "STATS_CALIBRATION_WAIT", 0.1):
                with self.assertRaises(HTTPException) as ctx:
                    main.stats_calibration()
                self.assertEqual(ctx.exception.status_code, 503)
        finally:
            main._stats_calibration_lock.release()


class SummaryTests(unittest.TestCase):
    def summary(self, correct):
        agg = {"accuracy_1x2": 0.55, "correct_1x2": correct, "accuracy_btts": 0.5, "accuracy_over_1_5": 0.7,
               "accuracy_over_2_5": 0.5, "mae_total_goals": 1.2}
        conn = FakeConn(fetchone_values=[{"n": 2000}, {"n": 1000}, agg])
        with mock.patch.object(main, "get_conn", return_value=conn):
            return main.stats_summary(), conn

    def test_exact_correct_count_is_returned(self):
        out, conn = self.summary(550)
        self.assertEqual((out.evaluated_predictions, out.correct_1x2), (1000, 550))
        self.assertIn("is_correct_1x2 = 1", conn.queries[2])

    def test_decimal_and_null_counts(self):
        from decimal import Decimal
        out, _ = self.summary(Decimal("550"))
        self.assertEqual(out.correct_1x2, 550)
        out, _ = self.summary(None)
        self.assertIsNone(out.correct_1x2)  # base vide : pas d'erreur


if __name__ == "__main__":
    unittest.main()
