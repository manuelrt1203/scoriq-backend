"""
Sync Railway PostgreSQL → football.db (SQLite local).
Usage : DATABASE_URL=postgres://... python sync_from_pg.py
"""
import os
import sqlite3
import psycopg2
import psycopg2.extras

SQLITE_PATH = os.environ.get("SQLITE_PATH", "football.db")
DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    raise SystemExit("DATABASE_URL non défini. Exemple :\n  DATABASE_URL=postgres://... python sync_from_pg.py")

TABLES = [
    ("competitions",       None),
    ("matches",            "id"),
    ("teams",              None),
    ("predictions_history","id"),
]


def migrate_sqlite_schema(sq):
    """Ajoute les colonnes manquantes sur les tables existantes."""
    migrations = [
        ("competitions", "country", "TEXT"),
        ("matches", "home_badge", "TEXT"),
        ("matches", "away_badge", "TEXT"),
    ]
    for table, col, col_type in migrations:
        existing = [row[1] for row in sq.execute(f"PRAGMA table_info({table})").fetchall()]
        if col not in existing:
            try:
                sq.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
                print(f"  Migration : {table}.{col} ajouté.")
            except Exception:
                pass
    sq.commit()


def ensure_sqlite_schema(sq):
    sq.executescript("""
    CREATE TABLE IF NOT EXISTS competitions (
        idLeague         INTEGER,
        name             TEXT NOT NULL,
        competition_type TEXT NOT NULL,
        rounds           INTEGER NOT NULL,
        strLeague        TEXT,
        country          TEXT
    );
    CREATE TABLE IF NOT EXISTS matches (
        id                  INTEGER PRIMARY KEY,
        idLeague            INTEGER,
        season              TEXT,
        round               TEXT,
        date                TEXT,
        home                TEXT,
        away                TEXT,
        home_score          INTEGER,
        away_score          INTEGER,
        status              TEXT NOT NULL,
        competition_name    TEXT,
        competition_type    TEXT,
        competition_country TEXT,
        year                INTEGER
    );
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
    );
    CREATE TABLE IF NOT EXISTS predictions_history (
        id                    INTEGER PRIMARY KEY,
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
    );
    """)
    sq.commit()


def sync_table(pg_cur, sq, table: str, conflict_col: str | None):
    pg_cur.execute(f"SELECT * FROM {table}")
    rows = pg_cur.fetchall()
    if not rows:
        print(f"  {table}: vide sur PG, ignoré.")
        return

    cols = [desc[0] for desc in pg_cur.description]
    placeholders = ", ".join(["?"] * len(cols))
    col_list = ", ".join(cols)

    if conflict_col:
        updates = ", ".join(f"{c} = excluded.{c}" for c in cols if c != conflict_col)
        sql = (
            f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
            f"ON CONFLICT({conflict_col}) DO UPDATE SET {updates}"
        )
    else:
        sql = f"INSERT OR IGNORE INTO {table} ({col_list}) VALUES ({placeholders})"

    batch = [tuple(row) for row in rows]
    sq.executemany(sql, batch)
    sq.commit()
    print(f"  {table}: {len(batch)} lignes synchronisées.")


def main():
    print(f"Source PG    : {DATABASE_URL[:50]}...")
    print(f"Cible SQLite : {SQLITE_PATH}\n")

    pg = psycopg2.connect(DATABASE_URL)
    pg_cur = pg.cursor()

    sq = sqlite3.connect(SQLITE_PATH)
    ensure_sqlite_schema(sq)
    migrate_sqlite_schema(sq)

    for table, conflict_col in TABLES:
        sync_table(pg_cur, sq, table, conflict_col)

    pg_cur.close()
    pg.close()
    sq.close()
    print("\nSync terminé.")


if __name__ == "__main__":
    main()
