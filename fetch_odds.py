#!/usr/bin/env python3
"""
Récupère les cotes bookmakers depuis The Odds API et les stocke en base.
Calcule automatiquement les value bets par rapport aux prédictions ScorIQ.

Usage : python fetch_odds.py
Consommation : ~11 requêtes API par exécution (500/mois en gratuit)
"""

import os
import time
from datetime import datetime

import requests

import db_conn

API_KEY = os.environ.get("ODDS_API_KEY", "de0ccf75fbaf200ac373e1407b2a2762")
BASE_URL = "https://api.the-odds-api.com/v4/sports"

# Compétitions suivies : idLeague ScorIQ → clé Odds API
COMPETITIONS = {
    4328: "soccer_epl",
    4331: "soccer_germany_bundesliga",
    4332: "soccer_italy_serie_a",
    4334: "soccer_france_ligue_one",
    4335: "soccer_spain_la_liga",
    4337: "soccer_netherlands_eredivisie",
    4344: "soccer_portugal_primeira_liga",
    4339: "soccer_turkey_super_league",
    4480: "soccer_uefa_champs_league",
    4481: "soccer_uefa_europa_league",
    4484: "soccer_france_coupe_de_france",
    4485: "soccer_germany_dfb_pokal",
}

# Bookmakers prioritaires (les mieux disponibles en Europe)
PREFERRED_BOOKS = ["betclic", "winamax", "unibet", "bet365", "william_hill",
                   "pinnacle", "betfair", "1xbet"]

# Normalisation Odds API → TheSportsDB
TEAM_NAME_MAP = {
    # Premier League
    "Brighton and Hove Albion": "Brighton",
    "Newcastle United":         "Newcastle",
    "Tottenham Hotspur":        "Tottenham",
    "West Ham United":          "West Ham",
    "Wolverhampton Wanderers":  "Wolves",
    "Nottingham Forest":        "Nottingham Forest",
    "Leeds United":             "Leeds United",
    "Leicester City":           "Leicester",
    # Bundesliga
    "Bayer Leverkusen":         "Bayer Leverkusen",
    "Borussia Dortmund":        "Borussia Dortmund",
    "Borussia Mönchengladbach": "Borussia Mönchengladbach",
    "Eintracht Frankfurt":      "Eintracht Frankfurt",
    "FC Augsburg":              "Augsburg",
    "FC Cologne":               "FC Koln",
    "FC Heidenheim":            "FC Heidenheim",
    "Hamburger SV":             "Hamburg",
    "RB Leipzig":               "RasenBallsport Leipzig",
    "SC Freiburg":              "Freiburg",
    "TSG Hoffenheim":           "Hoffenheim",
    "VfB Stuttgart":            "Stuttgart",
    "VfL Bochum":               "Bochum",
    "VfL Wolfsburg":            "Wolfsburg",
    "Werder Bremen":            "Werder Bremen",
    # Serie A
    "Atalanta BC":              "Atalanta",
    "AC Milan":                 "Milan",
    "Hellas Verona":            "Hellas Verona",
    "Inter Milan":              "Inter Milan",
    "Juventus":                 "Juventus",
    "SS Lazio":                 "Lazio",
    "SSC Napoli":               "Napoli",
    "Udinese Calcio":           "Udinese",
    # Ligue 1
    "Olympique Lyonnais":       "Lyon",
    "Olympique de Marseille":   "Marseille",
    "Paris Saint-Germain":      "Paris SG",
    "Paris Saint Germain":      "Paris SG",
    "AS Monaco":                "Monaco",
    "AJ Auxerre":               "Auxerre",
    "Stade de Reims":           "Stade de Reims",
    "Stade Brestois 29":        "Brest",
    "Stade Rennais":            "Rennes",
    "RC Lens":                  "Lens",
    # La Liga
    "Athletic Club":            "Ath Bilbao",
    "Athletic Bilbao":          "Ath Bilbao",
    "Atletico Madrid":          "Ath Madrid",
    "Atlético Madrid":          "Ath Madrid",
    "Alavés":                   "Alaves",
    "Deportivo Alavés":         "Alaves",
    "CA Osasuna":               "Osasuna",
    "Deportivo Alavés":         "Alaves",
    "Girona FC":                "Girona",
    "RCD Mallorca":             "Mallorca",
    "Rayo Vallecano":           "Rayo Vallecano",
    "Real Betis":               "Real Betis",
    "Real Sociedad":            "Sociedad",
    "Villarreal CF":            "Villarreal",
    # Eredivisie
    "PSV Eindhoven":            "PSV Eindhoven",
    "AFC Ajax":                 "Ajax",
    "Feyenoord":                "Feyenoord",
    # CL/EL
    "Manchester City":          "Manchester City",
    "Real Madrid":              "Real Madrid",
    "Barcelona":                "Barcelona",
    "Arsenal":                  "Arsenal",
    "Liverpool":                "Liverpool",
    "Bayern Munich":            "Bayern Munich",
    "Sporting CP":              "Sporting CP",
    "Benfica":                  "Benfica",
    "Club Brugge":              "Club Brugge",
    "Borussia Dortmund":        "Borussia Dortmund",
}


def normalize_team(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)


def ensure_table(conn: db_conn.Connection) -> None:
    conn.execute_script("""
        CREATE TABLE IF NOT EXISTS odds (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id        TEXT NOT NULL,
            idLeague        INTEGER,
            match_date      TEXT,
            home_team       TEXT,
            away_team       TEXT,
            bookmaker       TEXT,
            odds_home       REAL,
            odds_draw       REAL,
            odds_away       REAL,
            implied_home    REAL,
            implied_draw    REAL,
            implied_away    REAL,
            fetched_at      TEXT,
            UNIQUE(match_id, bookmaker)
        );
        CREATE INDEX IF NOT EXISTS idx_odds_date_teams
        ON odds(match_date, home_team, away_team)
    """)
    conn.commit()


def fetch_sport_odds(sport_key: str) -> list[dict]:
    try:
        r = requests.get(
            f"{BASE_URL}/{sport_key}/odds/",
            params={
                "apiKey":      API_KEY,
                "regions":     "eu",
                "markets":     "h2h",
                "oddsFormat":  "decimal",
                "dateFormat":  "iso",
            },
            timeout=15,
        )
        remaining = r.headers.get("x-requests-remaining", "?")
        if r.status_code == 200:
            print(f"  {sport_key}: {len(r.json())} matchs (restantes: {remaining})")
            return r.json()
        print(f"  {sport_key}: HTTP {r.status_code}")
        return []
    except requests.RequestException as e:
        print(f"  {sport_key}: erreur {e}")
        return []


def best_odds(match: dict) -> dict | None:
    """Retourne les meilleures cotes moyennes parmi tous les bookmakers."""
    home_odds_list, draw_odds_list, away_odds_list = [], [], []

    for book in match.get("bookmakers", []):
        h2h = next((m for m in book["markets"] if m["key"] == "h2h"), None)
        if not h2h:
            continue
        outcomes = {o["name"]: o["price"] for o in h2h["outcomes"]}
        h = outcomes.get(match["home_team"])
        d = outcomes.get("Draw")
        a = outcomes.get(match["away_team"])
        if h and d and a:
            home_odds_list.append((h, book["key"]))
            draw_odds_list.append((d, book["key"]))
            away_odds_list.append((a, book["key"]))

    if not home_odds_list:
        return None

    # Meilleure cote disponible pour chaque issue
    best_h = max(home_odds_list, key=lambda x: x[0])
    best_d = max(draw_odds_list, key=lambda x: x[0])
    best_a = max(away_odds_list, key=lambda x: x[0])

    # Cote moyenne (marché consensus)
    avg_h = sum(o for o, _ in home_odds_list) / len(home_odds_list)
    avg_d = sum(o for o, _ in draw_odds_list) / len(draw_odds_list)
    avg_a = sum(o for o, _ in away_odds_list) / len(away_odds_list)

    return {
        "best_home": best_h[0], "best_draw": best_d[0], "best_away": best_a[0],
        "avg_home":  avg_h,     "avg_draw":  avg_d,     "avg_away":  avg_a,
        "bookmaker": best_h[1],
    }


def upsert_odds(conn: db_conn.Connection, match: dict, id_league: int) -> int:
    fetched_at = datetime.now().isoformat()
    match_date = match["commence_time"][:10]
    inserted = 0

    for book in match.get("bookmakers", []):
        h2h = next((m for m in book["markets"] if m["key"] == "h2h"), None)
        if not h2h:
            continue
        outcomes = {o["name"]: o["price"] for o in h2h["outcomes"]}
        oh = outcomes.get(match["home_team"])
        od = outcomes.get("Draw")
        oa = outcomes.get(match["away_team"])
        if not (oh and od and oa):
            continue

        # Probabilités implicites (sans marge)
        total = 1/oh + 1/od + 1/oa
        ih = (1/oh) / total
        id_ = (1/od) / total
        ia = (1/oa) / total

        conn.execute("""
            INSERT INTO odds (
                match_id, idLeague, match_date, home_team, away_team,
                bookmaker, odds_home, odds_draw, odds_away,
                implied_home, implied_draw, implied_away, fetched_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(match_id, bookmaker) DO UPDATE SET
                odds_home=excluded.odds_home, odds_draw=excluded.odds_draw,
                odds_away=excluded.odds_away, implied_home=excluded.implied_home,
                implied_draw=excluded.implied_draw, implied_away=excluded.implied_away,
                fetched_at=excluded.fetched_at
        """, (
            match["id"], id_league, match_date,
            normalize_team(match["home_team"]),
            normalize_team(match["away_team"]),
            book["key"], oh, od, oa, ih, id_, ia, fetched_at
        ))
        inserted += 1

    return inserted


def print_value_bets(conn: db_conn.Connection) -> None:
    """Affiche les value bets : prédictions ScorIQ vs cotes bookmakers."""
    rows = conn.execute("""
        SELECT
            p.match_date, p.home_team, p.away_team, p.competition_name,
            p.proba_home_win, p.proba_draw, p.proba_away_win,
            o.bookmaker, o.odds_home, o.odds_draw, o.odds_away,
            o.implied_home, o.implied_draw, o.implied_away
        FROM predictions_history p
        JOIN odds o ON o.match_date = substr(p.match_date,1,10)
            AND (lower(o.home_team) LIKE '%' || lower(substr(p.home_team,1,6)) || '%'
              OR lower(p.home_team) LIKE '%' || lower(substr(o.home_team,1,6)) || '%')
        WHERE p.real_result IS NULL
          AND o.bookmaker IN ('betclic','winamax','unibet','bet365','william_hill','pinnacle')
        ORDER BY p.match_date, p.home_team
        LIMIT 30
    """).fetchall()

    if not rows:
        print("\nAucun match avec cotes ET prédictions trouvé (vérifier les noms d'équipes).")
        return

    print(f"\n{'='*70}")
    print("VALUE BETS — ScorIQ vs Bookmakers")
    print(f"{'='*70}")

    for r in rows:
        date, home, away, comp = r[0], r[1], r[2], r[3]
        ph, pd, pa = r[4], r[5], r[6]
        book, oh, od, oa = r[7], r[8], r[9], r[10]
        ih, id_, ia = r[11], r[12], r[13]

        value_h = (ph or 0) - ih
        value_d = (pd or 0) - id_
        value_a = (pa or 0) - ia

        values = [(value_h, "DOM", oh), (value_d, "NUL", od), (value_a, "EXT", oa)]
        best_value = max(values, key=lambda x: x[0])

        if best_value[0] > 0.05:
            print(f"\n  {date} | {home} vs {away} ({comp})")
            print(f"  ScorIQ: Dom={ph:.0%} Nul={pd:.0%} Ext={pa:.0%}")
            print(f"  {book}: {oh:.2f} / {od:.2f} / {oa:.2f}")
            marker = "🔥" if best_value[0] > 0.10 else "✓"
            print(f"  {marker} VALUE {best_value[1]}: +{best_value[0]:.1%} (cote {best_value[2]:.2f})")


def main() -> None:
    conn = db_conn.get_connection()
    ensure_table(conn)

    print("=== RÉCUPÉRATION DES COTES ===\n")
    total = 0

    for id_league, sport_key in COMPETITIONS.items():
        matches = fetch_sport_odds(sport_key)
        for m in matches:
            total += upsert_odds(conn, m, id_league)
        conn.commit()
        time.sleep(0.5)

    print(f"\n{total} entrées de cotes stockées.")
    print_value_bets(conn)
    conn.close()


if __name__ == "__main__":
    main()
