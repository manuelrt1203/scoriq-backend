"""
Migration one-shot : copie football.db (SQLite) vers PostgreSQL.
Usage : DATABASE_URL=postgres://... python migrate_to_pg.py
"""
import os
import sqlite3
import psycopg2
import psycopg2.extras

SQLITE_PATH = os.environ.get("SQLITE_PATH", "football.db")
DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    raise SystemExit("DATABASE_URL non défini.")


def create_pg_schema(pg):
    cur = pg.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS competitions (
            idLeague  INTEGER,
            name      TEXT NOT NULL,
            competition_type TEXT NOT NULL,
            rounds    INTEGER NOT NULL,
            strLeague TEXT,
            country   TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            id                 INTEGER PRIMARY KEY,
            idLeague           INTEGER,
            season             TEXT,
            round              TEXT,
            date               TEXT,
            home               TEXT,
            away               TEXT,
            home_score         INTEGER,
            away_score         INTEGER,
            status             TEXT NOT NULL,
            competition_name   TEXT,
            competition_type   TEXT,
            competition_country TEXT,
            year               INTEGER
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS teams (
            id               INTEGER,
            strTeam          TEXT,
            idLeague         INTEGER,
            strLeague        TEXT,
            competition_type TEXT,
            badge_url        TEXT,
            strCountry       TEXT,
            strSport         TEXT,
            team_type        TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions_history (
            id                    SERIAL PRIMARY KEY,
            prediction_run_date   TEXT NOT NULL,
            match_date            TEXT NOT NULL,
            competition_name      TEXT,
            competition_type      TEXT,
            home_team             TEXT NOT NULL,
            away_team             TEXT NOT NULL,
            status_prediction     TEXT NOT NULL,
            top_pick              TEXT,
            confidence            REAL,
            trust_level           TEXT,
            proba_home_win        REAL,
            proba_draw            REAL,
            proba_away_win        REAL,
            pred_home_goals       REAL,
            pred_away_goals       REAL,
            pred_total_goals      REAL,
            over_1_5              REAL,
            over_2_5              REAL,
            over_3_5              REAL,
            btts_yes              REAL,
            most_likely_score     TEXT,
            most_likely_score_prob REAL,
            value_pick            TEXT,
            value_edge            REAL,
            evaluation_status     TEXT,
            real_home_goals       INTEGER,
            real_away_goals       INTEGER,
            real_result           TEXT,
            real_total_goals      INTEGER,
            real_btts             INTEGER,
            real_over_2_5         INTEGER,
            is_correct_1x2        INTEGER,
            is_correct_score      INTEGER,
            is_correct_btts       INTEGER,
            is_correct_over_2_5   INTEGER,
            abs_error_home_goals  REAL,
            abs_error_away_goals  REAL,
            abs_error_total_goals REAL,
            model_used            TEXT
        )
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_ph_run_date
        ON predictions_history(prediction_run_date)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_ph_match_date
        ON predictions_history(match_date)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_ph_match_key
        ON predictions_history(match_date, home_team, away_team)
    """)
    pg.commit()
    print("Schema PostgreSQL créé.")


def migrate_table(sqlite_conn, pg, table: str, conflict_col: str | None = None):
    rows = sqlite_conn.execute(f"SELECT * FROM {table}").fetchall()
    if not rows:
        print(f"  {table}: vide, ignoré.")
        return

    cols = list(rows[0].keys())
    placeholders = ", ".join(["%s"] * len(cols))
    col_list = ", ".join(cols)

    if conflict_col:
        updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c != conflict_col)
        sql = (
            f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
            f"ON CONFLICT ({conflict_col}) DO UPDATE SET {updates}"
        )
    else:
        sql = f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"

    cur = pg.cursor()
    batch = [tuple(row[c] for c in cols) for row in rows]
    psycopg2.extras.execute_batch(cur, sql, batch, page_size=500)
    pg.commit()
    print(f"  {table}: {len(batch)} lignes migrées.")


def main():
    sqlite_conn = sqlite3.connect(SQLITE_PATH)
    sqlite_conn.row_factory = sqlite3.Row

    pg = psycopg2.connect(DATABASE_URL)

    print(f"Source SQLite : {SQLITE_PATH}")
    print(f"Cible PG      : {DATABASE_URL[:40]}...\n")

    create_pg_schema(pg)

    migrate_table(sqlite_conn, pg, "competitions")
    migrate_table(sqlite_conn, pg, "matches", conflict_col="id")
    migrate_table(sqlite_conn, pg, "predictions_history")
    migrate_table(sqlite_conn, pg, "teams")

    sqlite_conn.close()
    pg.close()
    print("\nMigration terminée.")


if __name__ == "__main__":
    main()
