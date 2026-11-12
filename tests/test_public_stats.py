import math
import os
import sys
import unittest
from decimal import Decimal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import public_stats as ps  # noqa: E402


def row(p1, px, p2, result, m=None):
    r = {"proba_home_win": p1, "proba_draw": px, "proba_away_win": p2, "real_result": result}
    if m:
        r.update(market_proba_home=m[0], market_proba_draw=m[1], market_proba_away=m[2])
    return r


class NumberTests(unittest.TestCase):
    def test_unusable_values_are_none(self):
        for v in (None, True, False, "abc", "", float("nan"), float("inf"), -0.1, [], {}):
            self.assertIsNone(ps._number(v), repr(v))
        self.assertEqual(ps._number(Decimal("0.25")), 0.25)
        self.assertEqual(ps._number("0.5"), 0.5)
        self.assertEqual(ps._number(0), 0.0)

    def test_probabilities_are_normalised_and_validated(self):
        p = ps.model_probabilities(row(2, 1, 1, "1"))
        self.assertAlmostEqual(sum(p.values()), 1.0)
        self.assertAlmostEqual(p["1"], 0.5)
        self.assertIsNone(ps.model_probabilities(row(0, 0, 0, "1")))
        self.assertIsNone(ps.model_probabilities(row(0.5, None, 0.5, "1")))
        self.assertIsNone(ps.model_probabilities({"proba_home_win": 0.5}))


class SummarizeTests(unittest.TestCase):
    def test_perfect_prediction(self):
        s = ps.summarize([row(1, 0, 0, "1"), row(0, 0, 1, "2")], ps.model_probabilities)
        self.assertEqual((s["n"], s["accuracy"], s["brier"]), (2, 1.0, 0.0))
        self.assertAlmostEqual(s["log_loss"], 0.0, places=4)

    def test_uniform_prediction_matches_the_chance_references(self):
        rows = [row(1, 1, 1, o) for o in ("1", "X", "2")]
        s = ps.summarize(rows, ps.model_probabilities)
        self.assertAlmostEqual(s["brier"], 2 / 3, places=3)
        self.assertAlmostEqual(s["log_loss"], math.log(3), places=3)

    def test_known_values(self):
        # p = (0.5, 0.3, 0.2), résultat « 1 » : Brier = 0.25 + 0.09 + 0.04 = 0.38 ; log-loss = -ln 0.5
        s = ps.summarize([row(0.5, 0.3, 0.2, "1")], ps.model_probabilities)
        self.assertAlmostEqual(s["brier"], 0.38, places=4)
        self.assertAlmostEqual(s["log_loss"], 0.6931, places=4)

    def test_rows_without_result_or_probabilities_are_ignored_and_empty_is_none(self):
        rows = [row(0.5, 0.3, 0.2, None), row(0.5, 0.3, 0.2, "?"), row(None, 0.3, 0.2, "1")]
        self.assertIsNone(ps.summarize(rows, ps.model_probabilities))
        self.assertIsNone(ps.summarize([], ps.model_probabilities))

    def test_a_zero_probability_for_the_real_result_does_not_blow_up(self):
        s = ps.summarize([row(1, 0, 0, "2")], ps.model_probabilities)
        self.assertTrue(math.isfinite(s["log_loss"]))


class CalibrationTests(unittest.TestCase):
    def test_overconfident_bin(self):
        rows = [row(0.8, 0.1, 0.1, "1" if i < 7 else "2") for i in range(10)]  # annoncé 80 %, 7 réussis sur 10
        bins = ps.calibration_bins(rows)
        self.assertEqual(len(bins), 1)
        b = bins[0]
        self.assertEqual((b["low"], b["high"], b["count"]), (0.75, 0.85, 10))
        self.assertAlmostEqual(b["predicted"], 0.8)
        self.assertAlmostEqual(b["observed"], 0.7)

    def test_bins_cover_the_whole_range_including_certainty(self):
        for top in (1 / 3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.97, 1.0):
            rest = (1 - top) / 2
            bins = ps.calibration_bins([row(top, rest, rest, "1")])
            self.assertEqual(sum(b["count"] for b in bins), 1, top)

    def test_the_pick_is_the_most_probable_outcome(self):
        # pronostic « 2 » (0.6) et le résultat est « 2 » : réussi
        b = ps.calibration_bins([row(0.2, 0.2, 0.6, "2")])[0]
        self.assertEqual((b["count"], b["observed"]), (1, 1.0))

    def test_empty_bins_are_omitted(self):
        self.assertEqual(ps.calibration_bins([]), [])


class MarketTests(unittest.TestCase):
    def rows(self, n, with_odds):
        out = []
        for i in range(n):
            result = "1" if i % 2 == 0 else "2"
            out.append(row(0.5, 0.2, 0.3, result, m=(0.45, 0.25, 0.30) if i < with_odds else None))
        return out

    def test_not_published_below_the_minimum(self):
        self.assertIsNone(ps.market_comparison(self.rows(100, ps.MIN_MARKET_MATCHES - 1), 100))

    def test_compared_on_the_same_matches_only(self):
        rows = self.rows(100, 40)
        m = ps.market_comparison(rows, len(rows))
        self.assertEqual(m["n"], 40)
        self.assertEqual(m["model"]["n"], 40)
        self.assertEqual(m["market"]["n"], 40)
        self.assertAlmostEqual(m["coverage"], 0.4)

    def test_market_probabilities_are_renormalised(self):
        rows = [row(0.5, 0.2, 0.3, "1", m=(0.9, 0.5, 0.6)) for _ in range(ps.MIN_MARKET_MATCHES)]  # somme 2.0
        m = ps.market_comparison(rows, len(rows))
        self.assertAlmostEqual(ps.market_probabilities(rows[0])["1"], 0.45)
        self.assertEqual(m["market"]["accuracy"], 1.0)

    def test_null_market_fields_are_skipped(self):
        rows = [row(0.5, 0.2, 0.3, "1", m=(None, 0.3, 0.3)) for _ in range(50)]
        self.assertIsNone(ps.market_comparison(rows, 50))


class BuildTests(unittest.TestCase):
    def test_full_payload_shape(self):
        rows = [row(0.6, 0.25, 0.15, "1") for _ in range(5)] + [row(0.5, 0.3, 0.2, None)]
        out = ps.build_public_stats(rows)
        self.assertEqual(set(out), {"calibration", "overall", "market", "reference"})
        self.assertEqual(out["overall"]["n"], 5)  # la ligne sans résultat réel est écartée
        self.assertIsNone(out["market"])
        self.assertEqual(out["reference"], {"brier_chance": 0.667, "log_loss_chance": 1.099})

    def test_empty_history(self):
        out = ps.build_public_stats([])
        self.assertEqual((out["calibration"], out["overall"], out["market"]), ([], None, None))


if __name__ == "__main__":
    unittest.main()
