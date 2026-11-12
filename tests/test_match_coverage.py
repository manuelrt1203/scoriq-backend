"""Non-régressions du 12/11/2026 : « l'app n'affichait aucun match du jour ».

1. La UEFA Nations League 2026-27 n'existe à la source que sous la saison « 2026-2027 » ; le pipeline ne
   testait que des années simples et n'en récupérait aucun match.
2. /top-picks reprenait les prédictions de la VEILLE (déjà jouées) faute de filtre sur la date.
"""
import os
import sqlite3
import sys
import unittest
from datetime import datetime
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_main_guards  # noqa: E402,F401  (doublures des modules lourds, puis import de main)
import main  # noqa: E402
import update_base  # noqa: E402


class InternationalSeasonsTests(unittest.TestCase):
    def test_november_2026_includes_the_two_year_season(self):
        seasons = update_base.get_international_seasons(datetime(2026, 11, 12))
        self.assertEqual(seasons, ["2025", "2026", "2027", "2026-2027"])

    def test_spring_uses_the_season_that_started_last_year(self):
        self.assertIn("2025-2026", update_base.get_international_seasons(datetime(2026, 3, 26)))

    def test_club_seasons_are_unchanged(self):
        self.assertEqual(update_base.get_seasons_for_comp("LEAGUE"), update_base.get_club_seasons())

    def test_nations_league_is_reachable_when_the_source_only_knows_the_hyphenated_season(self):
        comp = {"idLeague": 4480, "competition_type": "INTERNATIONAL", "rounds": 6, "name": "UEFA Nations League"}
        asked = []

        def season_events(league_id, season):
            asked.append(season)
            return [{"idEvent": "1", "intRound": "1"}] if season == "2026-2027" else []

        def round_events(league_id, season, round_no):
            # La vraie source publie les journées 1 à 6 de la phase de ligue sous « 2026-2027 », et rien sous « 2026 ».
            return [{"idEvent": f"r{round_no}", "intRound": str(round_no)}] if season == "2026-2027" and round_no <= 6 else []

        found = {}
        with mock.patch.object(update_base, "fetch_events_for_league_season", side_effect=season_events), \
             mock.patch.object(update_base, "fetch_events_for_round", side_effect=round_events), \
             mock.patch.object(update_base.time, "sleep"):
            for season in update_base.get_seasons_for_comp("INTERNATIONAL"):
                events, by_round = update_base.fetch_events_for_competition(comp, season)
                found[season] = (events, by_round)
        self.assertIn("2026-2027", asked)
        ids = {e["idEvent"] for events, _ in found.values() for e in events}
        self.assertIn("r5", ids)  # la journée 5 (12-13 novembre) est bien récupérée
        # Avec les seules années simples (ancien comportement), rien n'aurait été trouvé :
        self.assertEqual({s for s, (events, _) in found.items() if events}, {"2026-2027"})


class RefreshSeasonTests(unittest.TestCase):
    def conn_with(self, rows):
        db = sqlite3.connect(":memory:")
        db.row_factory = sqlite3.Row
        db.execute("CREATE TABLE matches (idLeague INTEGER, season TEXT, date TEXT)")
        db.executemany("INSERT INTO matches VALUES (?,?,?)", rows)
        return db

    def test_uses_the_season_recorded_for_the_competition(self):
        db = self.conn_with([(4480, "2026-2027", "2026-11-11T17:00:00"), (4480, "2025-2026", "2026-03-26T19:45:00")])
        comp = {"idLeague": 4480, "competition_type": "INTERNATIONAL"}
        self.assertEqual(update_base.latest_season_for_league(db, 4480), "2026-2027")
        self.assertEqual(update_base.season_for_refresh(db, comp, datetime(2026, 11, 12)), "2026-2027")

    def test_plain_year_competitions_keep_their_plain_year(self):
        db = self.conn_with([(4496, "2026", "2026-11-11T17:00:00")])
        comp = {"idLeague": 4496, "competition_type": "INTERNATIONAL"}
        self.assertEqual(update_base.season_for_refresh(db, comp, datetime(2026, 11, 12)), "2026")

    def test_unknown_competition_falls_back_to_the_calendar_year_and_clubs_to_the_football_season(self):
        db = self.conn_with([])
        self.assertEqual(update_base.season_for_refresh(db, {"idLeague": 1, "competition_type": "INTERNATIONAL"}, datetime(2026, 11, 12)), "2026")
        self.assertEqual(update_base.season_for_refresh(db, {"idLeague": 2, "competition_type": "LEAGUE"}, datetime(2026, 11, 12)), "2026-2027")


class FixedNow(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 11, 12, 9, 0, 0)


class SharedConn:
    """Enveloppe une base SQLite en mémoire : close() ne la ferme pas (plusieurs appels dans un test)."""

    def __init__(self, db):
        self.db = db

    def execute(self, sql, params=()):
        return self.db.execute(sql, params)

    def close(self):
        pass


def pred(id_, date, home, away, confidence, status="PREDICTED"):
    return (id_, date, home, away, status, confidence)


class TopPicksTests(unittest.TestCase):
    def setUp(self):
        db = sqlite3.connect(":memory:")
        db.row_factory = sqlite3.Row
        db.execute("""CREATE TABLE predictions_history (id INTEGER PRIMARY KEY, match_date TEXT, home_team TEXT,
                      away_team TEXT, status_prediction TEXT, confidence REAL, competition_name TEXT)""")
        db.execute("CREATE TABLE teams (strTeam TEXT, badge_url TEXT, strLeague TEXT)")
        self.db = db

    def add(self, *rows):
        self.db.executemany(
            "INSERT INTO predictions_history (id, match_date, home_team, away_team, status_prediction, confidence, competition_name) "
            "VALUES (?,?,?,?,?,?, 'Test')", rows)

    def picks(self, limit=5):
        with mock.patch.object(main, "get_conn", return_value=SharedConn(self.db)), \
             mock.patch.object(main, "datetime", FixedNow):
            return main.top_picks(limit)

    def test_yesterdays_matches_are_not_top_picks_today(self):
        # Le cas réel du 12/11 : seules des prédictions de la veille existaient (96 %, 77 %...).
        self.add(pred(1, "2026-11-11", "Benin", "Central African Republic", 0.96),
                 pred(2, "2026-11-11", "Ivory Coast", "Gambia", 0.767),
                 pred(3, "2026-11-10", "A", "B", 0.99))
        self.assertEqual(self.picks(), [])

    def test_today_and_upcoming_are_kept_sorted_by_confidence(self):
        self.add(pred(1, "2026-11-11", "Hier", "Hier2", 0.99),
                 pred(2, "2026-11-12", "Aujourd'hui", "Faible", 0.55),
                 pred(3, "2026-11-12", "Aujourd'hui", "Fort", 0.80),
                 pred(4, "2026-11-13", "Demain", "Moyen", 0.70))
        out = self.picks()
        self.assertEqual([r["away_team"] for r in out], ["Fort", "Moyen", "Faible"])
        self.assertTrue(all(r["match_date"] >= "2026-11-12" for r in out))

    def test_limit_and_non_predicted_rows(self):
        self.add(*[pred(i, "2026-11-12", f"H{i}", f"A{i}", 0.5 + i / 100) for i in range(1, 9)],
                 pred(20, "2026-11-12", "X", "Y", 0.99, status="INSUFFICIENT_HISTORY"))
        out = self.picks(limit=3)
        self.assertEqual(len(out), 3)
        self.assertEqual([r["home_team"] for r in out], ["H8", "H7", "H6"])  # l'« INSUFFICIENT_HISTORY » est exclu

    def test_limit_is_bounded_on_the_public_route(self):
        self.add(*[pred(i, "2026-11-12", f"H{i}", f"A{i}", 0.5 + i / 1000) for i in range(1, 80)])
        self.assertEqual(len(self.picks(limit=10_000_000)), main.MAX_TOP_PICKS)
        self.assertEqual(len(self.picks(limit=-5)), 1)
        self.assertEqual(len(self.picks(limit=0)), 1)

    def test_history_page_is_bounded(self):
        db = self.db
        db.execute("CREATE TABLE IF NOT EXISTS predictions_history_x (id INTEGER)")
        captured = {}

        class Conn(SharedConn):
            def execute(self, sql, params=()):
                captured["params"] = params
                return self.db.execute("SELECT 1 WHERE 0")

        with mock.patch.object(main, "get_conn", return_value=Conn(db)):
            main.predictions_history(limit=10**9, offset=-7, only_evaluated=True)
        self.assertEqual(captured["params"], (main.MAX_HISTORY_PAGE, 0))
        with mock.patch.object(main, "get_conn", return_value=Conn(db)):
            main.predictions_history(limit=-3, offset=5, only_evaluated=True)
        self.assertEqual(captured["params"], (1, 5))

    def test_only_the_latest_prediction_per_match_is_kept(self):
        self.add(pred(1, "2026-11-12", "A", "B", 0.60), pred(2, "2026-11-12", "A", "B", 0.75))
        out = self.picks()
        self.assertEqual([(r["id"], r["confidence"]) for r in out], [(2, 0.75)])


if __name__ == "__main__":
    unittest.main()
