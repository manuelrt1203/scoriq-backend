#!/usr/bin/env python3
"""
Import historical match data from OpenFootball into the production database.

Compatible SQLite (local) and PostgreSQL (Railway via DATABASE_URL).

Usage:
  Local SQLite  : python import_openfootball.py
  PostgreSQL    : DATABASE_URL=postgres://... python import_openfootball.py

Sources:
  - football.json (JSON): Premier League, Bundesliga, Serie A, La Liga, Ligue 1
  - champions-league repo (text): UEFA Champions League
  Seasons: 2012-13 to 2024-25
"""

import hashlib
import json
import re
import time

import requests

import db_conn

REQUEST_SLEEP = 0.3
REQUEST_TIMEOUT = 20

MONTH_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}

LEAGUE_CONFIGS = [
    {"code": "en.1", "idLeague": 4328, "name": "Premier League", "type": "LEAGUE", "country": "England"},
    {"code": "de.1", "idLeague": 4331, "name": "Bundesliga",      "type": "LEAGUE", "country": "Germany"},
    {"code": "it.1", "idLeague": 4332, "name": "Serie A",         "type": "LEAGUE", "country": "Italy"},
    {"code": "es.1", "idLeague": 4335, "name": "La Liga",         "type": "LEAGUE", "country": "Spain"},
    {"code": "fr.1", "idLeague": 4334, "name": "Ligue 1",         "type": "LEAGUE", "country": "France"},
]

SEASONS = [
    "2012-13", "2013-14", "2014-15", "2015-16", "2016-17", "2017-18",
    "2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
]

CL_CONFIG = {
    "idLeague": 4480,
    "name": "UEFA Champions League",
    "type": "EUROPE",
    "country": "Europe",
}

JSON_BASE = "https://raw.githubusercontent.com/openfootball/football.json/master"
CL_BASE   = "https://raw.githubusercontent.com/openfootball/champions-league/master"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def of_season_to_db(of_season: str) -> str:
    parts = of_season.split("-")
    if len(parts) == 2 and len(parts[0]) == 4 and len(parts[1]) == 2:
        start = int(parts[0])
        return f"{start}-{start + 1}"
    return of_season


def make_id(competition: str, season: str, date: str, team1: str, team2: str) -> int:
    key = f"{competition}|{season}|{date}|{team1.lower()}|{team2.lower()}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:12], 16)


def fetch(url: str) -> str | None:
    try:
        # (connect_timeout, read_timeout) — évite les blocages infinis
        r = requests.get(url, timeout=(10, 30))
        if r.status_code == 404:
            return None
        r.raise_for_status()
        time.sleep(REQUEST_SLEEP)
        return r.text
    except requests.RequestException as e:
        print(f"  [ERREUR] {url} -> {e}", flush=True)
        return None


# ---------------------------------------------------------------------------
# DB setup
# ---------------------------------------------------------------------------

def ensure_schema(conn: db_conn.Connection) -> None:
    # PostgreSQL INTEGER max = 2.1 milliards, nos hash IDs peuvent dépasser → BIGINT
    if conn.is_pg:
        cur = conn.execute("""
            SELECT data_type FROM information_schema.columns
            WHERE table_name='matches' AND column_name='id'
        """)
        row = cur.fetchone()
        if row and row["data_type"] == "integer":
            conn.execute("ALTER TABLE matches ALTER COLUMN id TYPE BIGINT")
            conn.commit()
            print("Colonne 'id' convertie INTEGER → BIGINT.")

    if not conn.column_exists("matches", "source"):
        conn.execute("ALTER TABLE matches ADD COLUMN source TEXT DEFAULT 'thesportsdb'")
        conn.commit()
        print("Colonne 'source' ajoutée à matches.")

    conn.execute("UPDATE matches SET source = 'thesportsdb' WHERE source IS NULL")
    conn.commit()


_first_error_printed = False

def upsert_match(conn: db_conn.Connection, match_id: int, id_league: int, season: str,
                 round_no: int | None, date: str, home: str, away: str,
                 home_score: int | None, away_score: int | None,
                 comp_name: str, comp_type: str, comp_country: str) -> bool:
    global _first_error_printed
    status = "FINISHED" if home_score is not None and away_score is not None else "NS"
    # Convertir round en str pour PostgreSQL (colonne TEXT dans PG, INTEGER dans SQLite)
    round_val = str(round_no) if round_no is not None else None
    try:
        if conn.is_pg:
            conn.execute("SAVEPOINT upsert_sp")
        cur = conn.execute(
            """
            INSERT INTO matches (
                id, idLeague, season, round, date, home, away,
                home_score, away_score, status,
                competition_name, competition_type, competition_country, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'openfootball')
            ON CONFLICT(id) DO NOTHING
            """,
            (match_id, id_league, season, round_val, date, home, away,
             home_score, away_score, status, comp_name, comp_type, comp_country),
        )
        inserted = cur.rowcount > 0
        if conn.is_pg:
            conn.execute("RELEASE SAVEPOINT upsert_sp")
        return inserted
    except Exception as e:
        if not _first_error_printed:
            print(f"  [ERREUR INSERT] {home} vs {away} ({date}): {e}")
            _first_error_printed = True
        if conn.is_pg:
            try:
                conn.execute("ROLLBACK TO SAVEPOINT upsert_sp")
            except Exception:
                pass
        return False


# ---------------------------------------------------------------------------
# JSON league import
# ---------------------------------------------------------------------------

def import_json_season(conn: db_conn.Connection, cfg: dict, of_season: str) -> int:
    url = f"{JSON_BASE}/{of_season}/{cfg['code']}.json"
    text = fetch(url)
    if not text:
        return 0

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  [JSON invalide] {url}: {e}")
        return 0

    matches = data.get("matches", [])
    db_season = of_season_to_db(of_season)
    inserted = 0

    for m in matches:
        date = m.get("date")
        team1 = m.get("team1", "")
        team2 = m.get("team2", "")
        if not date or not team1 or not team2:
            continue

        if isinstance(team1, dict):
            team1 = team1.get("name", "")
        if isinstance(team2, dict):
            team2 = team2.get("name", "")

        score = m.get("score", {})
        ft = score.get("ft") if isinstance(score, dict) else None
        home_score = int(ft[0]) if ft and len(ft) == 2 else None
        away_score = int(ft[1]) if ft and len(ft) == 2 else None

        round_str = m.get("round", "")
        round_no = None
        if round_str:
            rn = re.search(r"\d+", str(round_str))
            round_no = int(rn.group()) if rn else None

        match_id = make_id(cfg["name"], db_season, date, team1, team2)
        if upsert_match(conn, match_id, cfg["idLeague"], db_season, round_no,
                        date, team1, team2, home_score, away_score,
                        cfg["name"], cfg["type"], cfg["country"]):
            inserted += 1

    conn.commit()
    return inserted


def import_leagues(conn: db_conn.Connection) -> None:
    print("\n=== IMPORT LIGUES (football.json) ===")
    for cfg in LEAGUE_CONFIGS:
        total = 0
        for season in SEASONS:
            n = import_json_season(conn, cfg, season)
            if n > 0:
                print(f"  {cfg['name']} {season}: +{n} matchs")
            total += n
        print(f"  → {cfg['name']} total: {total} matchs importés\n")


# ---------------------------------------------------------------------------
# Champions League text parser
# ---------------------------------------------------------------------------

def parse_cl_date(line: str) -> str | None:
    m = re.search(r"([A-Z][a-z]{2})/(\d{1,2})\s+(\d{4})", line)
    if m:
        month = MONTH_MAP.get(m.group(1))
        day = int(m.group(2))
        year = int(m.group(3))
        if month:
            return f"{year}-{month:02d}-{day:02d}"
    return None


def parse_cl_round(line: str) -> int | None:
    content = line.lstrip("»").strip().lower()
    m = re.search(r"matchday\s+(\d+)", content)
    if m:
        return int(m.group(1))
    if "round of 16" in content:
        return 16
    if "round of 32" in content or "playoff" in content or "round 1" in content:
        return 32
    if "quarter" in content:
        return 128
    if "semi" in content:
        return 256
    if re.search(r"\bfinal\b", content) and "semi" not in content and "quarter" not in content:
        return 400
    return None


def parse_cl_match_line(line: str) -> dict | None:
    stripped = line.strip()
    if " v " not in stripped:
        return None
    stripped = re.sub(r"^\d{1,2}\.\d{2}\s+", "", stripped)
    v_idx = stripped.find(" v ")
    if v_idx == -1:
        return None
    team1_raw = stripped[:v_idx].strip()
    rest = stripped[v_idx + 3:].strip()
    score_m = re.search(r"(\d+)-(\d+)\s*(?:\(\d+-\d+\))?\s*$", rest)
    if score_m:
        home_score = int(score_m.group(1))
        away_score = int(score_m.group(2))
        team2_raw = rest[: score_m.start()].strip()
    else:
        home_score = None
        away_score = None
        team2_raw = rest
    team1 = re.sub(r"\s*\([A-Z]{2,3}\)\s*$", "", team1_raw).strip()
    team2 = re.sub(r"\s*\([A-Z]{2,3}\)\s*$", "", team2_raw).strip()
    if not team1 or not team2:
        return None
    return {"team1": team1, "team2": team2, "home_score": home_score, "away_score": away_score}


def parse_cl_text(text: str, of_season: str) -> list[dict]:
    db_season = of_season_to_db(of_season)
    matches = []
    current_round: int | None = None
    current_date: str | None = None
    for line in text.splitlines():
        if not line.strip():
            continue
        if "»" in line:
            r = parse_cl_round(line)
            if r is not None:
                current_round = r
            continue
        if re.match(r"^\s{1,3}[A-Z][a-z]{2}\s+[A-Z][a-z]{2}/\d", line):
            d = parse_cl_date(line)
            if d:
                current_date = d
            continue
        if re.match(r"^\s{4,}", line) and " v " in line:
            m = parse_cl_match_line(line)
            if m and current_date:
                matches.append({"season": db_season, "round": current_round,
                                "date": current_date, **m})
    return matches


def import_cl_season(conn: db_conn.Connection, of_season: str) -> tuple[int, int]:
    url = f"{CL_BASE}/{of_season}/cl.txt"
    text = fetch(url)
    if not text:
        return 0, 0
    matches = parse_cl_text(text, of_season)
    cfg = CL_CONFIG
    inserted = 0
    knockout = 0
    KNOCKOUT = {16, 32, 64, 128, 256, 400}
    for m in matches:
        match_id = make_id(cfg["name"], m["season"], m["date"], m["team1"], m["team2"])
        ok = upsert_match(conn, match_id, cfg["idLeague"], m["season"], m["round"],
                          m["date"], m["team1"], m["team2"],
                          m["home_score"], m["away_score"],
                          cfg["name"], cfg["type"], cfg["country"])
        if ok:
            inserted += 1
            if m["round"] in KNOCKOUT:
                knockout += 1
    conn.commit()
    return inserted, knockout


def import_champions_league(conn: db_conn.Connection) -> None:
    print("\n=== IMPORT CHAMPIONS LEAGUE (text) ===")
    total, total_ko = 0, 0
    for season in SEASONS:
        n, ko = import_cl_season(conn, season)
        if n > 0:
            print(f"  CL {season}: +{n} matchs dont {ko} phases finales")
        total += n
        total_ko += ko
    print(f"  → Champions League total: {total} matchs dont {total_ko} phases finales\n")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(conn: db_conn.Connection) -> None:
    print("=== RÉSUMÉ GLOBAL ===")
    rows = conn.execute("""
        SELECT competition_name,
               COUNT(DISTINCT season) as saisons,
               COUNT(*) as matchs,
               SUM(CASE WHEN status='FINISHED' THEN 1 ELSE 0 END) as termines
        FROM matches
        WHERE source = 'openfootball'
        GROUP BY competition_name
        ORDER BY competition_name
    """).fetchall()
    for r in rows:
        print(f"  {r['competition_name']}: {r['saisons']} saisons, "
              f"{r['matchs']} matchs ({r['termines']} terminés)")

    total = conn.execute(
        "SELECT COUNT(*) FROM matches WHERE source='openfootball'"
    ).fetchone()
    total_all = conn.execute("SELECT COUNT(*) FROM matches").fetchone()
    print(f"\n  OpenFootball: {list(total.values())[0]} matchs  |  "
          f"Total en base: {list(total_all.values())[0]}")

    print("\n--- Phases finales CL (OpenFootball) ---")
    labels = {16: "R16", 32: "R32", 128: "QF", 256: "SF", 400: "Finale"}
    rows = conn.execute("""
        SELECT season, round, COUNT(*) as nb
        FROM matches
        WHERE competition_name='UEFA Champions League'
          AND source='openfootball'
          AND round IN (16, 32, 128, 256, 400)
        GROUP BY season, round
        ORDER BY season, round
    """).fetchall()
    for r in rows:
        label = labels.get(r["round"], str(r["round"]))
        print(f"  {r['season']} {label}: {r['nb']} matchs")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    conn = db_conn.get_connection()
    db_type = "PostgreSQL" if conn.is_pg else "SQLite"
    print(f"Connexion : {db_type}")

    try:
        ensure_schema(conn)
        import_leagues(conn)
        import_champions_league(conn)
        print_summary(conn)
    finally:
        conn.close()

    print("\nImport OpenFootball terminé.")


if __name__ == "__main__":
    main()
