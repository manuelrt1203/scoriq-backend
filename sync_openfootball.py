#!/usr/bin/env python3
"""
Copie les matchs OpenFootball depuis SQLite local vers PostgreSQL Railway.
Usage : DATABASE_URL=postgres://... python sync_openfootball.py
"""
import os
import sqlite3

import psycopg2
import psycopg2.extras

SQLITE_PATH = os.environ.get("SQLITE_PATH", "football.db")
DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    raise SystemExit("DATABASE_URL non défini.")

BATCH_SIZE = 500

# ---------------------------------------------------------------------------
# Lecture SQLite
# ---------------------------------------------------------------------------

def load_openfootball_matches():
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT id, idLeague, season, round, date, home, away,
               home_score, away_score, status,
               competition_name, competition_type, competition_country, source
        FROM matches
        WHERE source = 'openfootball'
        ORDER BY date
    """).fetchall()
    conn.close()
    print(f"{len(rows)} matchs OpenFootball lus depuis SQLite.")
    return rows


# ---------------------------------------------------------------------------
# Écriture PostgreSQL
# ---------------------------------------------------------------------------

def ensure_bigint(pg):
    cur = pg.cursor()
    cur.execute("""
        SELECT data_type FROM information_schema.columns
        WHERE table_name='matches' AND column_name='id'
    """)
    row = cur.fetchone()
    if row and row[0] == "integer":
        cur.execute("ALTER TABLE matches ALTER COLUMN id TYPE BIGINT")
        pg.commit()
        print("Colonne 'id' convertie INTEGER → BIGINT.")


def ensure_source_column(pg):
    cur = pg.cursor()
    cur.execute("""
        SELECT COUNT(*) FROM information_schema.columns
        WHERE table_name='matches' AND column_name='source'
    """)
    if cur.fetchone()[0] == 0:
        cur.execute("ALTER TABLE matches ADD COLUMN source TEXT DEFAULT 'thesportsdb'")
        pg.commit()
        print("Colonne 'source' ajoutée.")
    cur.execute("UPDATE matches SET source = 'thesportsdb' WHERE source IS NULL")
    pg.commit()


def insert_batch(pg, batch):
    cur = pg.cursor()
    psycopg2.extras.execute_values(
        cur,
        """
        INSERT INTO matches (
            id, idLeague, season, round, date, home, away,
            home_score, away_score, status,
            competition_name, competition_type, competition_country, source
        )
        VALUES %s
        ON CONFLICT (id) DO NOTHING
        """,
        [
            (
                row["id"], row["idLeague"], row["season"],
                str(row["round"]) if row["round"] is not None else None,
                row["date"], row["home"], row["away"],
                row["home_score"], row["away_score"], row["status"],
                row["competition_name"], row["competition_type"],
                row["competition_country"], row["source"],
            )
            for row in batch
        ],
        page_size=BATCH_SIZE,
    )
    pg.commit()
    return cur.rowcount


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    matches = load_openfootball_matches()
    if not matches:
        print("Rien à importer.")
        return

    pg = psycopg2.connect(DATABASE_URL)
    ensure_bigint(pg)
    ensure_source_column(pg)

    total_inserted = 0
    total = len(matches)

    for i in range(0, total, BATCH_SIZE):
        batch = matches[i: i + BATCH_SIZE]
        n = insert_batch(pg, batch)
        total_inserted += n
        print(f"  [{i + len(batch)}/{total}] +{n} insérés", flush=True)

    pg.close()

    print(f"\nTerminé. {total_inserted} matchs insérés sur {total}.")


if __name__ == "__main__":
    main()
