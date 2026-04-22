import requests
import time
from datetime import datetime, timedelta
from typing import Any

import db_conn
API_KEY = "123"
BASE_URL = f"https://www.thesportsdb.com/api/v1/json/{API_KEY}"

EVENTS_SEASON_URL = f"{BASE_URL}/eventsseason.php"
EVENTS_ROUND_URL = f"{BASE_URL}/eventsround.php"

REQUEST_TIMEOUT = 25
SLEEP_BETWEEN_REQUESTS = 4
MAX_RETRIES = 4

ALLOWED_COMP_TYPES = {"LEAGUE", "DOMESTIC_CUP", "EUROPE", "INTERNATIONAL"}

# Si une compétition n'a rien via eventsseason.php, on peut tenter quelques rounds
DEFAULT_ROUNDS_BY_TYPE = {
    "LEAGUE": 40,
    "DOMESTIC_CUP": 20,
    "EUROPE": 25,
    "INTERNATIONAL": 20,
}


def connect_db() -> db_conn.Connection:
    return db_conn.get_connection()


def ensure_matches_table(conn: db_conn.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY,
            idLeague INTEGER,
            season TEXT,
            round TEXT,
            date TEXT,
            home TEXT,
            away TEXT,
            home_score INTEGER,
            away_score INTEGER,
            status TEXT NOT NULL,
            competition_name TEXT,
            competition_type TEXT,
            competition_country TEXT
        )
    """)

    if not conn.column_exists("matches", "competition_country"):
        conn.execute("ALTER TABLE matches ADD COLUMN competition_country TEXT")

    conn.commit()


def get_competitions(conn: db_conn.Connection) -> list[dict[str, Any]]:
    has_country = conn.column_exists("competitions", "country")
    has_rounds  = conn.column_exists("competitions", "rounds")

    select_cols = ["idLeague", "name", "competition_type"]
    if has_rounds:
        select_cols.append("rounds")
    if has_country:
        select_cols.append("country")

    query = f"""
        SELECT {", ".join(select_cols)}
        FROM competitions
        WHERE idLeague IS NOT NULL
          AND name IS NOT NULL
        ORDER BY name
    """

    rows = conn.execute(query).fetchall()

    competitions = []
    for row in rows:
        comp_type = (row["competition_type"] or "").strip().upper()
        if comp_type not in ALLOWED_COMP_TYPES:
            continue

        competitions.append({
            "idLeague": row["idLeague"],
            "name": row["name"],
            "competition_type": comp_type,
            "rounds": row["rounds"] if has_rounds and row["rounds"] is not None else DEFAULT_ROUNDS_BY_TYPE.get(comp_type, 20),
            "country": row["country"] if has_country else None,
        })

    return competitions


def current_football_season(today: datetime | None = None) -> str:
    today = today or datetime.now()
    if today.month >= 8:
        return f"{today.year}-{today.year + 1}"
    return f"{today.year - 1}-{today.year}"


def previous_football_season(today: datetime | None = None) -> str:
    today = today or datetime.now()
    current = current_football_season(today)
    start = int(current[:4])
    end = int(current[-4:])
    return f"{start - 1}-{end - 1}"


def get_club_seasons() -> list[str]:
    return [
        previous_football_season(),
        current_football_season(),
    ]


def get_international_seasons() -> list[str]:
    now = datetime.now()
    # Années simples pour les compétitions internationales
    return [
        str(now.year - 1),
        str(now.year),
        str(now.year + 1),
    ]


def get_seasons_for_comp(comp_type: str) -> list[str]:
    if comp_type == "INTERNATIONAL":
        return get_international_seasons()
    return get_club_seasons()


def get_json_with_retry(url: str, params: dict[str, Any], retries: int = MAX_RETRIES) -> dict[str, Any]:
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)

            if response.status_code == 429:
                wait_s = 8 * (attempt + 1)
                print(f"[429] Pause {wait_s}s pour {params}")
                time.sleep(wait_s)
                continue

            response.raise_for_status()
            return response.json()

        except requests.RequestException as e:
            if attempt == retries - 1:
                print(f"[ERREUR API] {params} -> {e}")
                return {}
            wait_s = 5 * (attempt + 1)
            print(f"[RETRY] {e} -> pause {wait_s}s")
            time.sleep(wait_s)

    return {}


def fetch_events_for_league_season(league_id: int, season: str) -> list[dict[str, Any]]:
    data = get_json_with_retry(EVENTS_SEASON_URL, {"id": league_id, "s": season})
    events = data.get("events")
    return events if isinstance(events, list) else []


def fetch_events_for_round(league_id: int, season: str, round_no: int) -> list[dict[str, Any]]:
    data = get_json_with_retry(EVENTS_ROUND_URL, {"id": league_id, "r": round_no, "s": season})
    events = data.get("events")
    return events if isinstance(events, list) else []


def normalize_status(status: Any, date_value: Any = None, home_score: Any = None, away_score: Any = None) -> str:
    if home_score is not None and away_score is not None:
        return "FINISHED"

    if status is None:
        # si pas de statut mais date passée -> score manquant
        if date_value:
            try:
                d = str(date_value)[:19].replace("T", " ")
                for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                    try:
                        dt = datetime.strptime(d[:len(fmt)], fmt)
                        if dt < datetime.now():
                            return "MISSING_SCORE"
                        return "NS"
                    except ValueError:
                        continue
            except Exception:
                pass
        return "UNKNOWN"

    s = str(status).strip().upper()
    if not s:
        return "UNKNOWN"

    mapping = {
        "FT": "FINISHED",
        "AET": "FINISHED",
        "PEN": "FINISHED",
        "NS": "NS",
        "NOT STARTED": "NS",
        "SCHEDULED": "NS",
        "TIMED": "NS",
        "POSTPONED": "POSTPONED",
        "CANCELLED": "CANCELLED",
    }
    return mapping.get(s, s)


def build_match_row(event: dict[str, Any], fallback_name: str, fallback_type: str, fallback_country: str | None, requested_season: str) -> tuple | None:
    event_id = event.get("idEvent")
    league_id = event.get("idLeague")
    season = event.get("strSeason") or requested_season

    round_value = event.get("intRound") or event.get("strRound") or event.get("round")

    date_value = (
        event.get("strTimestamp")
        or event.get("dateEvent")
        or event.get("strDate")
    )

    home = event.get("strHomeTeam")
    away = event.get("strAwayTeam")

    home_score = event.get("intHomeScore")
    away_score = event.get("intAwayScore")

    try:
        home_score = int(home_score) if home_score not in (None, "") else None
    except (TypeError, ValueError):
        home_score = None

    try:
        away_score = int(away_score) if away_score not in (None, "") else None
    except (TypeError, ValueError):
        away_score = None

    status = normalize_status(
        event.get("strStatus"),
        date_value=date_value,
        home_score=home_score,
        away_score=away_score,
    )

    competition_name = event.get("strLeague") or fallback_name

    if event_id in (None, "") or not home or not away or not date_value:
        return None

    return (
        int(event_id),
        int(league_id) if league_id not in (None, "") else None,
        season,
        str(round_value) if round_value not in (None, "") else None,
        date_value,
        home,
        away,
        home_score,
        away_score,
        status,
        competition_name,
        fallback_type,
        fallback_country,
    )


def upsert_match(conn: db_conn.Connection, row: tuple) -> None:
    conn.execute("""
        INSERT INTO matches (
            id, idLeague, season, round, date, home, away,
            home_score, away_score, status, competition_name, competition_type, competition_country
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            idLeague = excluded.idLeague,
            season = excluded.season,
            round = excluded.round,
            date = excluded.date,
            home = excluded.home,
            away = excluded.away,
            home_score = excluded.home_score,
            away_score = excluded.away_score,
            status = excluded.status,
            competition_name = excluded.competition_name,
            competition_type = excluded.competition_type,
            competition_country = excluded.competition_country
    """, row)


def dedupe_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    clean = []
    for e in events:
        event_id = e.get("idEvent")
        if event_id in (None, ""):
            continue
        if event_id in seen:
            continue
        seen.add(event_id)
        clean.append(e)
    return clean


KNOCKOUT_ROUNDS = [16, 32, 64, 128, 256, 400]


def fetch_events_for_competition(comp: dict[str, Any], season: str) -> list[dict[str, Any]]:
    league_id = comp["idLeague"]
    comp_type = comp.get("competition_type", "")
    rounds = int(comp.get("rounds") or DEFAULT_ROUNDS_BY_TYPE.get(comp_type, 20))

    collected: list[dict[str, Any]] = []

    # 1) Saison complète
    season_events = fetch_events_for_league_season(league_id, season)
    collected.extend(dedupe_events(season_events))

    # 2) Rounds séquentiels (phase de ligue, coupe domestique, etc.)
    empty_streak = 0
    for round_no in range(1, rounds + 1):
        round_events = fetch_events_for_round(league_id, season, round_no)
        if not round_events:
            empty_streak += 1
            if empty_streak >= 4:
                break
            continue
        empty_streak = 0
        collected.extend(round_events)

    # 3) Pour les compétitions européennes : rounds knockout fixes
    #    (play-offs R32, R16, QF, SF, Finale utilisent des rounds non séquentiels)
    if comp_type == "EUROPE":
        for round_no in KNOCKOUT_ROUNDS:
            if round_no <= rounds:
                continue  # déjà couvert par la boucle séquentielle
            round_events = fetch_events_for_round(league_id, season, round_no)
            if round_events:
                collected.extend(round_events)

    return dedupe_events(collected)


def main() -> None:
    conn = connect_db()
    ensure_matches_table(conn)

    competitions = get_competitions(conn)
    if not competitions:
        print("Aucune compétition utilisable trouvée dans competitions.")
        conn.close()
        return

    print(f"{len(competitions)} compétition(s) à mettre à jour.\n")

    total_events_seen = 0
    total_upserted = 0
    total_skipped = 0

    try:
        for idx, comp in enumerate(competitions, start=1):
            league_id = comp["idLeague"]
            name = comp["name"]
            comp_type = comp["competition_type"]
            comp_country = comp.get("country")
            seasons = get_seasons_for_comp(comp_type)

            print(f"[{idx}/{len(competitions)}] {name} ({league_id}) - {comp_type}")
            print(f"  Saisons testées : {', '.join(seasons)}")

            inserted_here = 0
            skipped_here = 0
            seen_here = 0

            for season in seasons:
                try:
                    events = fetch_events_for_competition(comp, season)
                except Exception as e:
                    print(f"  -> erreur API sur {season} : {e}")
                    time.sleep(SLEEP_BETWEEN_REQUESTS)
                    continue

                if not events:
                    print(f"  -> aucun événement renvoyé pour {season}")
                    time.sleep(SLEEP_BETWEEN_REQUESTS)
                    continue

                print(f"  -> {len(events)} événement(s) trouvés pour {season}")
                seen_here += len(events)

                for event in events:
                    row = build_match_row(
                        event,
                        fallback_name=name,
                        fallback_type=comp_type,
                        fallback_country=comp_country,
                        requested_season=season,
                    )

                    if row is None:
                        skipped_here += 1
                        total_skipped += 1
                        continue

                    try:
                        upsert_match(conn, row)
                        inserted_here += 1
                        total_upserted += 1
                    except Exception as e:
                        print(f"  -> match ignoré (erreur): {e}")
                        conn.rollback()
                        skipped_here += 1
                        total_skipped += 1

                conn.commit()
                time.sleep(SLEEP_BETWEEN_REQUESTS)

            total_events_seen += seen_here

            print(f"  -> {inserted_here} match(s) upsertés")
            if skipped_here:
                print(f"  -> {skipped_here} match(s) ignoré(s)")
            print()

        print("Terminé.")
        print(f"Événements vus : {total_events_seen}")
        print(f"Lignes upsertées : {total_upserted}")
        print(f"Lignes ignorées : {total_skipped}")

        refresh_pass(conn, competitions)

    finally:
        conn.close()


def get_stale_rounds(conn: db_conn.Connection, league_id: int) -> list[int]:
    """Rounds in DB with status SCHEDULED but date already passed (résultats manquants)."""
    cutoff = (datetime.now() - timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%S")
    rows = conn.execute("""
        SELECT DISTINCT round FROM matches
        WHERE idLeague = ?
          AND status NOT IN ('FINISHED', 'CANCELLED', 'POSTPONED')
          AND date < ?
          AND round IS NOT NULL
    """, (league_id, cutoff)).fetchall()
    result = []
    for r in rows:
        try:
            result.append(int(r["round"]))
        except (ValueError, TypeError):
            pass
    return result


def get_max_round(conn: db_conn.Connection, league_id: int) -> int:
    """Plus grand numéro de round en base pour cette ligue."""
    row = conn.execute("""
        SELECT MAX(CAST(round AS INTEGER)) as max_r FROM matches
        WHERE idLeague = ? AND round IS NOT NULL
    """, (league_id,)).fetchone()
    return int(row["max_r"]) if row and row["max_r"] else 0


def refresh_rounds(conn: db_conn.Connection, comp: dict, rounds_to_fetch: list[int], season: str) -> int:
    """Re-fetche des rounds spécifiques et met à jour la DB. Retourne le nb de matchs mis à jour."""
    updated = 0
    name = comp["name"]
    comp_type = comp["competition_type"]
    comp_country = comp.get("country")

    for round_no in sorted(set(rounds_to_fetch)):
        events = fetch_events_for_round(comp["idLeague"], season, round_no)
        if not events:
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            continue
        for event in dedupe_events(events):
            row = build_match_row(event, name, comp_type, comp_country, season)
            if row:
                upsert_match(conn, row)
                updated += 1
        conn.commit()
        time.sleep(SLEEP_BETWEEN_REQUESTS)
    return updated


def refresh_pass(conn: db_conn.Connection, competitions: list[dict]) -> None:
    """
    Second passage après la mise à jour principale :
    1. Re-fetche les rounds SCHEDULED dont la date est passée (pour récupérer les scores).
    2. Fetche les 2 rounds suivants le max connu (pour les rounds non encore publiés par l'API).
    """
    print("\n=== PASSAGE DE RAFRAÎCHISSEMENT ===")
    for comp in competitions:
        if comp["competition_type"] == "INTERNATIONAL":
            season = str(datetime.now().year)
        else:
            season = current_football_season()

        league_id = comp["idLeague"]
        rounds_to_fetch = []

        # 1. Rounds passés encore SCHEDULED
        stale = get_stale_rounds(conn, league_id)
        if stale:
            print(f"  {comp['name']} — rounds passés non terminés : {stale}")
            rounds_to_fetch.extend(stale)

        # 2. Rounds manquants (max_round + 1 et + 2)
        max_r = get_max_round(conn, league_id)
        if max_r > 0:
            rounds_to_fetch.extend([max_r + 1, max_r + 2])

        if not rounds_to_fetch:
            continue

        updated = refresh_rounds(conn, comp, rounds_to_fetch, season)
        if updated:
            print(f"  {comp['name']} — {updated} match(s) mis à jour")


if __name__ == "__main__":
    main()