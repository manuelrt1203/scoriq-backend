"""
Complète les phases à élimination directe manquantes (Coupe du Monde, C1)
via football-data.org — TheSportsDB (gratuit) ne les fournit pas.

Ne touche jamais aux matchs déjà présents via TheSportsDB (phase de poules
du Mondial, phase de ligue + 8es de C1) : n'ajoute que les rounds absents,
avec un id dédié (offset) pour ne jamais entrer en collision.
"""
import os
import time

import requests

import db_conn

API_TOKEN = os.environ.get("FOOTBALL_DATA_KEY", "")
BASE_URL = "https://api.football-data.org/v4"
SLEEP_BETWEEN_REQUESTS = 6  # 10 req/min max sur le plan gratuit

ID_OFFSET = 900_000_000_000_000  # namespace dédié, hors de portée de thesportsdb/openfootball

# Compétitions couvertes + mapping stage -> round (évite toute collision avec
# les rounds déjà utilisés par thesportsdb pour la même compétition)
COMPETITIONS = {
    "WC": {
        "idLeague": 4429,
        "name": "FIFA World Cup",
        "competition_type": "INTERNATIONAL",
        "competition_country": None,
        "skip_stages": {"GROUP_STAGE"},  # déjà en base via thesportsdb
        "round_map": {
            "LAST_32": "4",
            "LAST_16": "5",
            "QUARTER_FINALS": "6",
            "SEMI_FINALS": "7",
            "THIRD_PLACE": "8",
            "FINAL": "9",
        },
    },
    "CL": {
        "idLeague": 4480,
        "name": "UEFA Champions League",
        "competition_type": "EUROPE",
        "competition_country": "Europe",
        "skip_stages": {"LEAGUE_STAGE", "LAST_16"},  # déjà en base via thesportsdb
        "round_map": {
            "PLAYOFFS": "32",
            "QUARTER_FINALS": "128",
            "SEMI_FINALS": "256",
            "FINAL": "512",
        },
    },
}


def get_json_with_retry(url: str, retries: int = 4):
    headers = {"X-Auth-Token": API_TOKEN}
    for attempt in range(retries):
        resp = requests.get(url, headers=headers, timeout=25)
        if resp.status_code == 429:
            wait_s = 15 * (attempt + 1)
            print(f"[429] pause {wait_s}s")
            time.sleep(wait_s)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"Échec après {retries} tentatives : {url}")


def normalize_status(fd_status: str, home_score, away_score) -> str:
    mapping = {
        "FINISHED": "FINISHED",
        "IN_PLAY": "LIVE",
        "PAUSED": "LIVE",
        "SCHEDULED": "NS",
        "TIMED": "NS",
        "POSTPONED": "POSTPONED",
        "SUSPENDED": "POSTPONED",
        "CANCELLED": "CANCELLED",
        "AWARDED": "FINISHED",
    }
    return mapping.get(fd_status, "UNKNOWN")


def build_row(match: dict, comp_conf: dict):
    stage = match.get("stage")
    round_value = comp_conf["round_map"].get(stage)
    if round_value is None:
        return None  # stage non mappée (déjà couverte ailleurs ou inconnue) -> ignorée

    home = match["homeTeam"].get("name")
    away = match["awayTeam"].get("name")
    date_value = match.get("utcDate")
    if not home or not away or not date_value:
        return None

    score = match.get("score", {}).get("fullTime", {})
    home_score = score.get("home")
    away_score = score.get("away")
    status = normalize_status(match.get("status"), home_score, away_score)

    season_info = match.get("season", {})
    start_year = str(season_info.get("startDate", ""))[:4]
    end_year = str(season_info.get("endDate", ""))[:4]
    season = f"{start_year}-{end_year}" if end_year and end_year != start_year else start_year

    return (
        ID_OFFSET + int(match["id"]),
        comp_conf["idLeague"],
        season,
        round_value,
        date_value,
        home,
        away,
        home_score,
        away_score,
        status,
        comp_conf["name"],
        comp_conf["competition_type"],
        comp_conf["competition_country"],
    )


SOURCE = "football-data-org"


def upsert_match(conn: db_conn.Connection, row: tuple) -> None:
    conn.execute("""
        INSERT INTO matches (
            id, idLeague, season, round, date, home, away,
            home_score, away_score, status, competition_name, competition_type, competition_country, source
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            round = excluded.round,
            date = excluded.date,
            home = excluded.home,
            away = excluded.away,
            home_score = excluded.home_score,
            away_score = excluded.away_score,
            status = excluded.status
    """, row + (SOURCE,))


def main():
    if not API_TOKEN:
        raise RuntimeError("FOOTBALL_DATA_KEY manquant")

    conn = db_conn.get_connection()

    for code, conf in COMPETITIONS.items():
        print(f"=== {conf['name']} ({code}) ===")
        data = get_json_with_retry(f"{BASE_URL}/competitions/{code}/matches")
        matches = data.get("matches", [])
        print(f"  {len(matches)} matchs reçus")

        inserted = 0
        skipped = 0
        for m in matches:
            if m.get("stage") in conf["skip_stages"]:
                continue
            row = build_row(m, conf)
            if row is None:
                skipped += 1
                continue
            upsert_match(conn, row)
            inserted += 1

        conn.commit()
        print(f"  -> {inserted} match(s) upsertés (phases à élimination directe)")
        if skipped:
            print(f"  -> {skipped} match(s) ignoré(s) (stage non mappée)")

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    conn.close()


if __name__ == "__main__":
    main()
