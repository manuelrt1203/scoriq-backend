"""
Database abstraction layer.
- DATABASE_URL set → PostgreSQL via psycopg2
- DATABASE_URL absent → SQLite (football.db)
"""
import os
import re
import sqlite3

DATABASE_URL = os.environ.get("DATABASE_URL")
SQLITE_PATH = os.environ.get("SQLITE_PATH", "football.db")


# ── SQL translation helpers ────────────────────────────────────────────────────

def _pg_sql(sql: str) -> str:
    return sql.replace("?", "%s")


def _pg_ddl(sql: str) -> str:
    return re.sub(
        r"\bINTEGER\s+PRIMARY\s+KEY\s+AUTOINCREMENT\b",
        "SERIAL PRIMARY KEY",
        sql,
        flags=re.IGNORECASE,
    )


# ── Case-insensitive dict (PostgreSQL folds column names to lowercase) ─────────

class _CIRow(dict):
    """Dict whose keys are accessible in any case (idLeague == idleague)."""
    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            return super().__getitem__(key.lower())

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


# ── Cursor wrapper (normalises row access) ─────────────────────────────────────

class _PgCursor:
    def __init__(self, cur):
        self._cur = cur

    @property
    def rowcount(self):
        return self._cur.rowcount

    def fetchall(self):
        rows = self._cur.fetchall()
        return [_CIRow(r) for r in rows] if rows else []

    def fetchone(self):
        row = self._cur.fetchone()
        return _CIRow(row) if row is not None else None

    def __iter__(self):
        for row in self._cur:
            yield _CIRow(row)


# ── Connection wrapper ─────────────────────────────────────────────────────────

class Connection:
    def __init__(self):
        self.is_pg = DATABASE_URL is not None
        if self.is_pg:
            import psycopg2
            import psycopg2.extras
            self._conn = psycopg2.connect(DATABASE_URL)
            self._extras = psycopg2.extras
        else:
            self._conn = sqlite3.connect(SQLITE_PATH)
            self._conn.row_factory = sqlite3.Row

    # ── queries ────────────────────────────────────────────────────────────────

    def execute(self, sql: str, params=()):
        if self.is_pg:
            cur = self._conn.cursor(cursor_factory=self._extras.RealDictCursor)
            cur.execute(_pg_sql(sql), params or None)
            return _PgCursor(cur)
        return self._conn.execute(sql, params)

    def executemany(self, sql: str, params_list):
        if self.is_pg:
            cur = self._conn.cursor()
            cur.executemany(_pg_sql(sql), params_list)
        else:
            self._conn.executemany(sql, params_list)

    # ── DDL ────────────────────────────────────────────────────────────────────

    def execute_script(self, sql: str):
        """Run multi-statement DDL; handles AUTOINCREMENT → SERIAL."""
        if self.is_pg:
            cur = self._conn.cursor()
            for stmt in sql.split(";"):
                stmt = stmt.strip()
                if stmt:
                    cur.execute(_pg_ddl(_pg_sql(stmt)))
        else:
            self._conn.executescript(sql)

    # ── schema inspection ──────────────────────────────────────────────────────

    def column_exists(self, table: str, column: str) -> bool:
        if self.is_pg:
            cur = self._conn.cursor()
            cur.execute("""
                SELECT COUNT(*) FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s AND column_name = %s
            """, (table, column))
            return cur.fetchone()[0] > 0
        rows = self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        return any(r["name"] == column for r in rows)

    def table_exists(self, table: str) -> bool:
        if self.is_pg:
            cur = self._conn.cursor()
            cur.execute("""
                SELECT COUNT(*) FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = %s
            """, (table,))
            return cur.fetchone()[0] > 0
        cur = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
        )
        return cur.fetchone() is not None

    # ── transaction ────────────────────────────────────────────────────────────

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.rollback()
        else:
            self.commit()
        self.close()
        return False


def get_connection() -> Connection:
    return Connection()


def ensure_matches_indexes(conn: "Connection") -> None:
    """Index requis par les requêtes de predict_v3.py / update_base.py sur la table matches.
    Idempotent — safe à rappeler à chaque démarrage de script."""
    conn.execute_script("""
        CREATE INDEX IF NOT EXISTS idx_matches_source ON matches(source);
        CREATE INDEX IF NOT EXISTS idx_matches_idleague_season_round ON matches(idLeague, season, round);
        CREATE INDEX IF NOT EXISTS idx_matches_status_comptype ON matches(status, competition_type);
        CREATE INDEX IF NOT EXISTS idx_matches_date_trunc ON matches(date(date));
        CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(date);
        CREATE INDEX IF NOT EXISTS idx_matches_status ON matches(status);
        CREATE INDEX IF NOT EXISTS idx_matches_home ON matches(home);
        CREATE INDEX IF NOT EXISTS idx_matches_away ON matches(away);
        CREATE INDEX IF NOT EXISTS idx_matches_league_season ON matches(idLeague, season);
        CREATE INDEX IF NOT EXISTS idx_matches_league_date ON matches(idLeague, date);
    """)
    conn.commit()
