"""Fusion des alias d'équipes (team_aliases.apply_all) : doublons de shots_data, et jamais de suppression ailleurs.

Cas réel du 12/11/2026 : « Paris SG », « FC Koln » et « Ipswich » existaient aussi sous leur nom canonique dans
shots_data pour les mêmes matchs ; l'UPDATE de fusion violait la contrainte d'unicité et bloquait toute la fusion.
Les tests tournent sur SQLite en mémoire (la requête de suppression est du SQL portable), sans toucher une vraie base.
"""
import sqlite3
import sys
import types

import pytest

try:
    import db_conn  # noqa: F401
except ImportError:  # psycopg2 absent en local : team_aliases n'a besoin que du type Connection
    _stub = types.ModuleType("db_conn")
    _stub.Connection = object
    sys.modules["db_conn"] = _stub

import team_aliases  # noqa: E402


class SqliteConn:
    """Le sous-ensemble de db_conn.Connection utilisé par team_aliases, sur SQLite."""

    def __init__(self):
        self.c = sqlite3.connect(":memory:")
        self.c.row_factory = sqlite3.Row

    def execute(self, sql, params=()):
        return self.c.execute(sql, params)

    def execute_script(self, sql):
        self.c.executescript(sql)

    def commit(self):
        self.c.commit()

    def rollback(self):
        self.c.rollback()


@pytest.fixture
def conn():
    c = SqliteConn()
    c.execute_script("""
        CREATE TABLE matches (id INTEGER PRIMARY KEY, home TEXT, away TEXT);
        CREATE TABLE odds (id INTEGER PRIMARY KEY, home_team TEXT, away_team TEXT);
        CREATE TABLE predictions_history (id INTEGER PRIMARY KEY, match_date TEXT, home_team TEXT, away_team TEXT,
                                          UNIQUE (match_date, home_team, away_team));
        CREATE TABLE shots_data (id INTEGER PRIMARY KEY, idleague INTEGER, season TEXT, match_date TEXT,
                                 home_team TEXT, away_team TEXT, shots INTEGER,
                                 UNIQUE (idleague, season, match_date, home_team, away_team));
    """)
    team_aliases.ensure_table(c)
    return c


def add_aliases(conn, *pairs):
    for alias, canonical in pairs:  # ordre d'insertion = ordre de traitement
        conn.execute("INSERT INTO team_aliases (alias, canonical_name) VALUES (?, ?)", (alias, canonical))
    conn.commit()


def shot(conn, home, away, shots, date="2014-08-16"):
    conn.execute("INSERT INTO shots_data (idleague, season, match_date, home_team, away_team, shots) VALUES (4334, '2014-2015', ?, ?, ?, ?)",
                 (date, home, away, shots))
    conn.commit()


def rows(conn, table, cols):
    return [tuple(r) for r in conn.execute(f"SELECT {cols} FROM {table} ORDER BY id").fetchall()]


def test_doublon_a_domicile_supprime_et_la_ligne_canonique_est_gardee(conn):
    add_aliases(conn, ("Paris SG", "Paris Saint-Germain"))
    shot(conn, "Paris SG", "Bastia", 11)
    shot(conn, "Paris Saint-Germain", "Bastia", 22)
    team_aliases.apply_all(conn)
    assert rows(conn, "shots_data", "home_team, away_team, shots") == [("Paris Saint-Germain", "Bastia", 22)]


def test_doublon_a_l_exterieur(conn):
    add_aliases(conn, ("Ipswich", "Ipswich Town"))
    shot(conn, "Lyon", "Ipswich", 5)
    shot(conn, "Lyon", "Ipswich Town", 7)
    team_aliases.apply_all(conn)
    assert rows(conn, "shots_data", "home_team, away_team, shots") == [("Lyon", "Ipswich Town", 7)]


def test_une_ligne_d_alias_sans_jumeau_est_renommee_pas_supprimee(conn):
    add_aliases(conn, ("FC Koln", "FC Köln"))
    shot(conn, "FC Koln", "Mainz", 9)
    team_aliases.apply_all(conn)
    assert rows(conn, "shots_data", "home_team, away_team, shots") == [("FC Köln", "Mainz", 9)]


def test_un_autre_match_n_est_pas_pris_pour_un_doublon(conn):
    add_aliases(conn, ("Paris SG", "Paris Saint-Germain"))
    shot(conn, "Paris SG", "Bastia", 11, date="2014-08-16")
    shot(conn, "Paris Saint-Germain", "Bastia", 22, date="2015-01-10")  # même affiche, autre date
    team_aliases.apply_all(conn)
    assert rows(conn, "shots_data", "home_team, match_date, shots") == [
        ("Paris Saint-Germain", "2014-08-16", 11), ("Paris Saint-Germain", "2015-01-10", 22)]


def test_deux_alias_du_meme_club_sans_ligne_canonique_gardent_une_seule_ligne(conn):
    add_aliases(conn, ("Paris SG", "Paris Saint-Germain"), ("Paris Saint-Germain FC", "Paris Saint-Germain"))
    shot(conn, "Paris SG", "Bastia", 11)
    shot(conn, "Paris Saint-Germain FC", "Bastia", 33)
    team_aliases.apply_all(conn)
    assert rows(conn, "shots_data", "home_team, away_team") == [("Paris Saint-Germain", "Bastia")]


def test_les_deux_equipes_sont_des_alias(conn):
    add_aliases(conn, ("Paris SG", "Paris Saint-Germain"), ("FC Koln", "FC Köln"))
    shot(conn, "Paris SG", "FC Koln", 1)
    team_aliases.apply_all(conn)
    assert rows(conn, "shots_data", "home_team, away_team") == [("Paris Saint-Germain", "FC Köln")]


def test_idempotent(conn):
    add_aliases(conn, ("Paris SG", "Paris Saint-Germain"))
    shot(conn, "Paris SG", "Bastia", 11)
    shot(conn, "Paris Saint-Germain", "Bastia", 22)
    team_aliases.apply_all(conn)
    first = rows(conn, "shots_data", "home_team, away_team, shots")
    team_aliases.apply_all(conn)
    assert rows(conn, "shots_data", "home_team, away_team, shots") == first


def test_un_alias_identique_au_nom_canonique_ne_supprime_rien(conn):
    """Ligne insérée à la main (SQL, CLI) : sans garde-fou, chaque ligne serait son propre « doublon » et disparaîtrait."""
    add_aliases(conn, ("PSG", "PSG"))
    shot(conn, "PSG", "Nice", 1, date="2014-08-16")
    shot(conn, "PSG", "Lyon", 2, date="2014-08-23")
    team_aliases.apply_all(conn)
    assert rows(conn, "shots_data", "home_team, away_team") == [("PSG", "Nice"), ("PSG", "Lyon")]


def test_les_autres_tables_sont_renommees_normalement(conn):
    add_aliases(conn, ("Paris SG", "Paris Saint-Germain"))
    conn.execute("INSERT INTO matches (home, away) VALUES ('Paris SG', 'Nice')")
    conn.execute("INSERT INTO odds (home_team, away_team) VALUES ('Nice', 'Paris SG')")
    conn.commit()
    team_aliases.apply_all(conn)
    assert rows(conn, "matches", "home, away") == [("Paris Saint-Germain", "Nice")]
    assert rows(conn, "odds", "home_team, away_team") == [("Nice", "Paris Saint-Germain")]


def test_predictions_history_n_est_jamais_supprimee_un_conflit_reste_une_erreur(conn):
    """Historique public : un doublon y est signalé (RuntimeError), jamais supprimé en silence. Les autres alias
    sont quand même traités, et le travail de l'alias en échec est annulé en entier."""
    add_aliases(conn, ("Paris SG", "Paris Saint-Germain"), ("FC Koln", "FC Köln"))
    conn.execute("INSERT INTO matches (home, away) VALUES ('Paris SG', 'Nice')")
    conn.execute("INSERT INTO matches (home, away) VALUES ('FC Koln', 'Mainz')")
    conn.execute("INSERT INTO predictions_history (match_date, home_team, away_team) VALUES ('2026-04-19', 'Paris SG', 'Nice')")
    conn.execute("INSERT INTO predictions_history (match_date, home_team, away_team) VALUES ('2026-04-19', 'Paris Saint-Germain', 'Nice')")
    conn.commit()
    with pytest.raises(RuntimeError, match="Paris SG"):
        team_aliases.apply_all(conn)
    # aucune ligne de predictions_history supprimée
    assert rows(conn, "predictions_history", "home_team") == [("Paris SG",), ("Paris Saint-Germain",)]
    # le travail de l'alias en échec est annulé en entier (pas de renommage à moitié) ; l'alias suivant, lui, est bien traité
    assert rows(conn, "matches", "home") == [("Paris SG",), ("FC Köln",)]
