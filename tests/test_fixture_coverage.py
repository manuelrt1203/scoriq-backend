"""Contrôle de couverture des matchs (limite 1) et coût du sondage de saison (limite 2)."""
import os
import sqlite3
import sys
import unittest
from datetime import date
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_main_guards  # noqa: E402,F401  (doublures des modules lourds)
import check_fixture_coverage as cov  # noqa: E402
import update_base  # noqa: E402

TODAY = date(2026, 11, 12)
NL = {"idLeague": 4490, "name": "UEFA Nations League", "competition_type": "INTERNATIONAL", "country": None}
CL = {"idLeague": 4480, "name": "UEFA Champions League", "competition_type": "EUROPE", "country": None}


def make_conn(known_ids=()):
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE matches (id INTEGER PRIMARY KEY)")
    db.executemany("INSERT INTO matches VALUES (?)", [(i,) for i in known_ids])
    return db


def next_event(idevent, day, season="2026-2027", round_no="5", home="Turkey", away="Belgium"):
    return {"events": [{"idEvent": str(idevent), "dateEvent": day, "strSeason": season, "intRound": round_no,
                        "strHomeTeam": home, "strAwayTeam": away}]}


class GapDetectionTests(unittest.TestCase):
    def run_check(self, conn, comps, responses, heal=None):
        heal = heal or mock.Mock(return_value=0)
        fetch = mock.Mock(side_effect=lambda url, params: responses[params["id"]])
        gaps, unchecked = cov.find_gaps(conn, comps, today=TODAY, fetch=fetch, heal=heal, sleep=lambda s: None)
        return gaps, unchecked, heal, fetch

    def test_the_12_november_incident_is_detected_and_healed(self):
        # Turquie–Belgique (12/11) absent de la base : la source la donne sous la saison « 2026-2027 », journée 5.
        heal = mock.Mock(return_value=26)
        gaps, unchecked, heal, fetch = self.run_check(
            make_conn(), [NL], {4490: next_event(2442822, "2026-11-12")}, heal)
        self.assertEqual(len(gaps), 1)
        self.assertEqual((gaps[0]["competition"], gaps[0]["season"], gaps[0]["round"], gaps[0]["healed"]),
                         ("UEFA Nations League", "2026-2027", 5, 26))
        args = heal.call_args.args
        self.assertEqual(args[2], [4, 5, 6])      # journées voisines
        self.assertEqual(args[3], "2026-2027")    # avec la VRAIE saison, celle que le pipeline ne demandait pas
        self.assertEqual(unchecked, [])

    def test_no_gap_when_the_next_match_is_known(self):
        gaps, _, heal, _ = self.run_check(make_conn([2442822]), [NL], {4490: next_event(2442822, "2026-11-12")})
        self.assertEqual(gaps, [])
        heal.assert_not_called()

    def test_only_the_missing_competition_is_reported(self):
        gaps, _, heal, _ = self.run_check(
            make_conn([777]), [CL, NL],
            {4480: next_event(777, "2026-11-24", round_no="5"), 4490: next_event(2442822, "2026-11-12")})
        self.assertEqual([g["competition"] for g in gaps], ["UEFA Nations League"])
        self.assertEqual(heal.call_count, 1)

    def test_far_future_and_off_season_are_ignored(self):
        responses = {4490: next_event(1, "2027-08-15"), 4480: {"events": None}}
        gaps, unchecked, heal, _ = self.run_check(make_conn(), [NL, CL], responses)
        self.assertEqual((gaps, unchecked), ([], []))
        heal.assert_not_called()

    def test_source_errors_are_reported_as_unchecked_not_as_gaps(self):
        def boom(url, params):
            raise RuntimeError("429")

        gaps, unchecked = cov.find_gaps(make_conn(), [NL], today=TODAY, fetch=boom, heal=mock.Mock(),
                                        sleep=lambda s: None)
        self.assertEqual((gaps, unchecked), ([], ["UEFA Nations League"]))

    def test_missing_round_or_season_is_reported_without_healing(self):
        heal = mock.Mock(return_value=0)
        gaps, _, heal, _ = self.run_check(
            make_conn(), [NL], {4490: next_event(5, "2026-11-12", round_no=None)}, heal)
        self.assertEqual(len(gaps), 1)
        heal.assert_not_called()

    def test_malformed_events_do_not_crash(self):
        for bad in ({"events": [{"idEvent": "x", "dateEvent": "2026-11-12"}]},
                    {"events": [{"idEvent": "1", "dateEvent": "pas une date"}]}):
            gaps, unchecked, *_ = self.run_check(make_conn(), [NL], {4490: bad})
            self.assertEqual((gaps, unchecked), ([], []))

    def test_empty_source_response_is_unchecked_not_ok(self):
        # get_json_with_retry renvoie {} (sans lever) quand la source est en panne ou limitée : ce n'est PAS « rien à venir ».
        for empty in ({}, None):
            gaps, unchecked, *_ = self.run_check(make_conn(), [NL], {4490: empty})
            self.assertEqual((gaps, unchecked), ([], ["UEFA Nations League"]), f"réponse {empty!r}")
        # une vraie réponse « aucun match à venir » reste un cas normal
        gaps, unchecked, *_ = self.run_check(make_conn(), [NL], {4490: {"events": None}})
        self.assertEqual((gaps, unchecked), ([], []))

    def test_a_transient_failure_is_retried_once_then_trusted(self):
        calls = []

        def flaky(url, params):
            calls.append(params["id"])
            if len(calls) == 1:
                raise RuntimeError("429")
            return {"events": None}

        pauses = []
        gaps, unchecked = cov.find_gaps(make_conn(), [NL], today=TODAY, fetch=flaky, heal=mock.Mock(), sleep=pauses.append)
        self.assertEqual((gaps, unchecked), ([], []))
        self.assertEqual(len(calls), 2)
        self.assertIn(cov.RETRY_PAUSE_S, pauses)

    def test_a_competition_silent_twice_stays_unchecked_and_is_asked_only_twice(self):
        calls = []

        def down(url, params):
            calls.append(params["id"])
            raise RuntimeError("panne")

        gaps, unchecked = cov.find_gaps(make_conn(), [NL, CL], today=TODAY, fetch=down, heal=mock.Mock(), sleep=lambda s: None)
        self.assertEqual((gaps, sorted(unchecked)), ([], ["UEFA Champions League", "UEFA Nations League"]))
        self.assertEqual(len(calls), 4)  # une passe + une seconde passe, pas de boucle sans fin

    def test_team_names_cannot_inject_workflow_commands_in_the_log(self):
        import contextlib
        import io
        evil = next_event(9, "2026-11-12", home="A\n::add-mask::secret", away="B\r\n::stop-commands::x")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.run_check(make_conn(), [NL], {4490: evil})
        self.assertTrue(all(not line.startswith("::") for line in out.getvalue().splitlines()), out.getvalue())

    def test_summary_is_clean_single_line_text(self):
        text = cov.summary([{"competition": "Ligue\nX; rm -rf $(id)", "season": "2026-2027", "healed": 3,
                             "round": 5, "next_match": "a"}])
        self.assertNotIn("\n", text)
        self.assertNotIn("$", text)
        self.assertNotIn("(id)", text.replace("match(s)", ""))
        self.assertIn("3 match(s) rattrapé(s)", text)

    def test_summary_only_keeps_a_plausible_season(self):
        # Une saison venue de la source ne doit pas pouvoir devenir un lien dans l'issue publique.
        text = cov.summary([{"competition": "Ligue", "season": "www.exemple.com", "healed": 0, "round": 1, "next_match": "a"}])
        self.assertNotIn("exemple", text)
        self.assertIn("saison inconnue", text)
        for ok in ("2026", "2026-2027"):
            self.assertIn(f"saison {ok}", cov.summary([{"competition": "L", "season": ok, "healed": 0, "round": 1, "next_match": "a"}]))


class MainExitCodeTests(unittest.TestCase):
    """main() : une panne de la source sur la majorité des compétitions doit échouer, pas afficher « tout va bien »."""

    def run_main(self, n_competitions, unchecked, gaps=()):
        import tempfile
        comps = [{"idLeague": i, "name": f"Comp {i}"} for i in range(n_competitions)]
        with tempfile.NamedTemporaryFile("r+", suffix=".out") as out, \
                mock.patch.dict(os.environ, {"GITHUB_OUTPUT": out.name}), \
                mock.patch.object(cov.update_base, "connect_db", return_value=mock.Mock()), \
                mock.patch.object(cov.update_base, "get_competitions", return_value=comps), \
                mock.patch.object(cov, "find_gaps", return_value=(list(gaps), list(unchecked))):
            code = cov.main()
            out.seek(0)
            return code, out.read()

    def test_source_down_on_most_competitions_fails_with_a_reason(self):
        code, written = self.run_main(4, ["Comp 0", "Comp 1", "Comp 2"])
        self.assertEqual(code, 1)
        self.assertTrue(written.startswith("reason="))
        self.assertEqual(len(written.strip().splitlines()), 1)
        self.assertIn("3 compétition(s) sur 4", written)

    def test_a_single_silent_competition_fails_instead_of_looking_fine(self):
        # Sinon le pipeline dirait « rétabli » (et fermerait l'alerte) pour une compétition jamais vérifiée.
        code, written = self.run_main(10, ["Comp 3"])
        self.assertEqual(code, 1)
        self.assertIn("Comp 3", written)
        self.assertEqual(len(written.strip().splitlines()), 1)

    def test_everything_checked_exits_cleanly(self):
        self.assertEqual(self.run_main(10, []), (0, ""))

    def test_gaps_still_fail_and_keep_their_message(self):
        gap = {"competition": "UEFA Nations League", "season": "2026-2027", "round": 5, "healed": 26, "next_match": "a - b"}
        code, written = self.run_main(10, [], gaps=[gap])
        self.assertEqual(code, 1)
        self.assertIn("UEFA Nations League", written)


class ProbeCostTests(unittest.TestCase):
    """Limite 2 : la saison « à cheval » ne coûte plus qu'un appel quand elle est vide."""

    def run_fetch(self, comp, season, season_events, **kw):
        calls = {"rounds": 0}

        def rounds(league_id, s, r):
            calls["rounds"] += 1
            return []

        with mock.patch.object(update_base, "fetch_events_for_league_season", return_value=season_events), \
             mock.patch.object(update_base, "fetch_events_for_round", side_effect=rounds), \
             mock.patch.object(update_base.time, "sleep"):
            out = update_base.fetch_events_for_competition(comp, season, **kw)
        return out, calls["rounds"]

    def test_empty_probe_season_costs_a_single_call(self):
        out, rounds = self.run_fetch(NL, "2026-2027", [], probe=True)
        self.assertEqual((out, rounds), (([], {}), 0))

    def test_non_empty_probe_season_still_walks_the_rounds(self):
        events = [{"idEvent": "1", "intRound": "1"}]
        (_, _), rounds = self.run_fetch(NL, "2026-2027", events, probe=True)
        self.assertGreater(rounds, 0)

    def test_default_behaviour_is_unchanged_for_plain_seasons(self):
        # Sans `probe`, une saison vide parcourt encore les journées (4 appels vides avant de s'arrêter).
        (_, _), rounds = self.run_fetch(NL, "2026", [])
        self.assertEqual(rounds, 4)

    def test_which_seasons_are_probes(self):
        self.assertTrue(update_base.is_probe_season("INTERNATIONAL", "2026-2027"))
        self.assertFalse(update_base.is_probe_season("INTERNATIONAL", "2026"))
        self.assertFalse(update_base.is_probe_season("LEAGUE", "2026-2027"))  # les clubs gardent leur comportement

    def test_only_the_two_year_season_of_international_competitions_is_a_probe(self):
        flags = [update_base.is_probe_season("INTERNATIONAL", s) for s in update_base.get_seasons_for_comp("INTERNATIONAL")]
        self.assertEqual(flags, [False, False, False, True])


if __name__ == "__main__":
    unittest.main()
