"""
Normalise les noms d'équipes divergents (ex: "Paris SG" vs "Paris Saint-Germain")
et va chercher sur TheSportsDB les badges des équipes qui n'en ont pas encore,
sur toutes les tables qui stockent un nom d'équipe (matches, odds,
predictions_history, shots_data, teams).

Pensé pour tourner régulièrement (voir daily_update.yml) : idempotent, ne touche
que les lignes concernées, ne devine jamais un badge sans nom clairement lié
(pas de fallback "premier résultat dispo" — un badge faux est pire que pas de badge).
"""
import re
import time
import unicodedata

import requests

import db_conn

API_KEY = "123"
SEARCH_URL = f"https://www.thesportsdb.com/api/v1/json/{API_KEY}/searchteams.php"
REQUEST_TIMEOUT = 15
SLEEP_BETWEEN_REQUESTS = 4.0
MAX_RETRIES = 4

# Variantes de noms connues qui désignent la même équipe mais fragmentent son
# historique dans `matches` (ce qui casse à la fois les badges ET les features
# de prédiction, cf. incident Lille-PSG du 28/08/2026). Clé = variante à
# éliminer, valeur = nom canonique à conserver (celui utilisé par les fixtures
# à venir / le plus fréquent).
TEAM_NAME_MERGES = {
    "Paris SG": "Paris Saint-Germain",
    "Paris Saint-Germain FC": "Paris Saint-Germain",
    "FC Koln": "FC Köln",
    "Köln": "FC Köln",
    "Porto": "FC Porto",
    "Manchester United FC": "Manchester United",
    "Sporting Clube de Braga": "Braga",
    "Sporting Clube de Portugal": "Sporting CP",
}

# Alias de recherche TheSportsDB pour les noms qui ne matchent pas tels quels.
SEARCH_ALIASES = {
    "Bosnia-Herzegovina": ["Bosnia and Herzegovina"],
    "Cape Verde Islands": ["Cape Verde"],
    "Congo DR": ["DR Congo", "Congo DR"],
    "Górnik Zabrze": ["Gornik Zabrze"],
    "Lech Poznań": ["Lech Poznan"],
    "Kauno Žalgiris": ["Zalgiris Kaunas", "Kauno Zalgiris"],
    "Győri ETO": ["Gyori ETO FC", "Gyor"],
    "Jagiellonia Białystok": ["Jagiellonia Bialystok"],
    "Hradec Králové": ["Hradec Kralove"],
    "Universitatea Craiova": ["FC Universitatea Craiova", "Universitatea Craiova 1948"],
    "Universitatea Cluj": ["FC Universitatea Cluj"],
    "KÍ Klaksvík": ["KI Klaksvik"],
    "Víkingur Reykjavík": ["Vikingur"],
    "Petrocub Hîncești": ["Petrocub Hincesti"],
    "Sabah Baku": ["Sabah"],
    "Hapoel Be'er Sheva": ["Hapoel Beer Sheva"],
    "Marítimo": ["CS Maritimo"],
    "Académico de Viseu": ["Academico Viseu FC", "Academica de Viseu"],
    "Erzurumspor": ["BB Erzurumspor"],
    "Çorum": ["Corum FK"],
    "Bodø/Glimt": ["Bodo Glimt", "Bodo/Glimt"],
    "Heart of Midlothian": ["Hearts"],
    "Ararat-Armenia": ["FC Ararat-Armenia", "Ararat Armenia"],
    "Union Saint-Gilloise": ["Royale Union Saint-Gilloise"],
    "Shamrock Rovers": ["Shamrock Rovers FC"],
    "Dynamo Kyiv": ["Dynamo Kiev"],
    "Riga FC": ["FC Riga"],
}


def normalize(text: str) -> str:
    text = text or ""
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_str = nfkd.encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_str.lower().split())


def get_variants(name: str) -> list[str]:
    variants = [name, re.sub(r"[’']", "", name)]
    variants.extend(SEARCH_ALIASES.get(name, []))
    seen, out = set(), []
    for v in variants:
        k = normalize(v)
        if k and k not in seen:
            seen.add(k)
            out.append(v)
    return out


def search_team(name: str) -> list[dict]:
    wait = 8
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(SEARCH_URL, params={"t": name}, timeout=REQUEST_TIMEOUT)
            if r.status_code == 429 and attempt < MAX_RETRIES - 1:
                time.sleep(wait)
                wait *= 2
                continue
            r.raise_for_status()
            return r.json().get("teams") or []
        except requests.RequestException:
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait)
                wait *= 2
                continue
            return []
    return []


def choose_best(results: list[dict], wanted_name: str, wanted_league: str) -> dict | None:
    """Ne matche que si le nom retourné est réellement lié au nom recherché.
    Volontairement pas de fallback "premier résultat dispo" : un mauvais match
    (vu en pratique : "Union SG" -> club équatorien sans rapport) est pire
    qu'une case vide dans l'app."""
    wn, wl = normalize(wanted_name), normalize(wanted_league)
    soccer = [t for t in results if (t.get("strSport") or "").lower() == "soccer" and t.get("strBadge")]
    for t in soccer:
        if normalize(t.get("strTeam", "")) == wn and normalize(t.get("strLeague", "")) == wl:
            return t
    for t in soccer:
        if normalize(t.get("strTeam", "")) == wn:
            return t
    for t in soccer:
        an = normalize(t.get("strTeam", ""))
        alt = normalize(t.get("strTeamAlternate", "") or "")
        if (wn in an or an in wn) and min(len(wn), len(an)) >= 4:
            return t
        if wn and wn in alt:
            return t
    return None


def merge_team_name_variants(conn: db_conn.Connection) -> None:
    for old_name, canonical in TEAM_NAME_MERGES.items():
        for table, col in [
            ("matches", "home"), ("matches", "away"),
            ("odds", "home_team"), ("odds", "away_team"),
            ("predictions_history", "home_team"), ("predictions_history", "away_team"),
            ("shots_data", "home_team"), ("shots_data", "away_team"),
        ]:
            conn.execute(f"UPDATE {table} SET {col} = ? WHERE {col} = ?", (canonical, old_name))
    conn.commit()


def find_teams_missing_badges(conn: db_conn.Connection) -> list[dict]:
    rows = conn.execute("""
        SELECT DISTINCT team, competition_name, competition_type FROM (
            SELECT home AS team, competition_name, competition_type FROM matches
            WHERE date >= CURRENT_DATE - INTERVAL '90 days'
            UNION
            SELECT away AS team, competition_name, competition_type FROM matches
            WHERE date >= CURRENT_DATE - INTERVAL '90 days'
        ) recent
        WHERE NOT EXISTS (
            SELECT 1 FROM teams t
            WHERE LOWER(TRIM(t.strTeam)) = LOWER(TRIM(recent.team))
              AND t.badge_url IS NOT NULL AND TRIM(t.badge_url) <> ''
        )
        ORDER BY team
    """ if conn.is_pg else """
        SELECT DISTINCT team, competition_name, competition_type FROM (
            SELECT home AS team, competition_name, competition_type FROM matches
            WHERE date >= date('now', '-90 days')
            UNION
            SELECT away AS team, competition_name, competition_type FROM matches
            WHERE date >= date('now', '-90 days')
        ) recent
        WHERE NOT EXISTS (
            SELECT 1 FROM teams t
            WHERE LOWER(TRIM(t.strTeam)) = LOWER(TRIM(recent.team))
              AND t.badge_url IS NOT NULL AND TRIM(t.badge_url) <> ''
        )
        ORDER BY team
    """)
    return [dict(r) for r in rows.fetchall()]


def backfill_badges(conn: db_conn.Connection) -> tuple[int, int]:
    missing = find_teams_missing_badges(conn)
    found, not_found = 0, 0

    for item in missing:
        name, league, ctype = item["team"], item["competition_name"], item["competition_type"]
        match = None
        for variant in get_variants(name):
            results = search_team(variant)
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            match = choose_best(results, variant, league)
            if match:
                break

        if not match:
            not_found += 1
            continue

        badge = match.get("strBadge")
        country = match.get("strCountry")
        team_type = "NATIONAL" if ctype == "INTERNATIONAL" else "CLUB"

        conn.execute("""
            INSERT INTO teams (strteam, idleague, strleague, competition_type, badge_url, strcountry, strsport, team_type)
            SELECT ?, NULL, ?, ?, ?, ?, 'Soccer', ?
            WHERE NOT EXISTS (
                SELECT 1 FROM teams WHERE LOWER(TRIM(strteam)) = LOWER(TRIM(?)) AND strleague = ?
            )
        """, (name, league, ctype, badge, country, team_type, name, league))
        conn.execute("""
            UPDATE teams SET badge_url = ?, strcountry = ?, strsport = 'Soccer'
            WHERE LOWER(TRIM(strteam)) = LOWER(TRIM(?)) AND (badge_url IS NULL OR TRIM(badge_url) = '')
        """, (badge, country, name))
        conn.commit()
        found += 1

    return found, not_found


def main():
    conn = db_conn.get_connection()
    try:
        merge_team_name_variants(conn)
        found, not_found = backfill_badges(conn)
        print(f"Badges ajoutés : {found}")
        print(f"Toujours introuvables (probablement absents de TheSportsDB) : {not_found}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
