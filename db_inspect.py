"""Diagnostic en lecture seule — état des compétitions internationales / C1."""
import db_conn


def main():
    conn = db_conn.get_connection()

    print("=== Compétitions INTERNATIONAL ===")
    for r in conn.execute("""
        SELECT idLeague, name FROM competitions WHERE competition_type='INTERNATIONAL' ORDER BY name
    """).fetchall():
        print(dict(r) if not isinstance(r, dict) else r)

    print("\n=== Matches par compétition INTERNATIONAL (comptage) ===")
    for r in conn.execute("""
        SELECT idLeague, competition_name, COUNT(*) as n,
               SUM(CASE WHEN status='FINISHED' THEN 1 ELSE 0 END) as finished,
               MIN(date) as min_date, MAX(date) as max_date
        FROM matches WHERE competition_type='INTERNATIONAL'
        GROUP BY idLeague, competition_name ORDER BY n DESC
    """).fetchall():
        print(dict(r) if not isinstance(r, dict) else r)

    print("\n=== FIFA World Cup (idLeague=4429) par round ===")
    for r in conn.execute("""
        SELECT round, status, COUNT(*) as n, MIN(date) as min_date, MAX(date) as max_date
        FROM matches WHERE idLeague=4429
        GROUP BY round, status ORDER BY CAST(round AS INTEGER)
    """).fetchall():
        print(dict(r) if not isinstance(r, dict) else r)

    print("\n=== UEFA Champions League (idLeague=4480) par round, saison 2025-2026 ===")
    for r in conn.execute("""
        SELECT round, status, COUNT(*) as n
        FROM matches WHERE idLeague=4480 AND season='2025-2026'
        GROUP BY round, status ORDER BY CAST(round AS INTEGER)
    """).fetchall():
        print(dict(r) if not isinstance(r, dict) else r)

    print("\n=== Round 400 C1 — échantillon (pollution suspectée) ===")
    for r in conn.execute("""
        SELECT date, home, away, home_score, away_score FROM matches
        WHERE idLeague=4480 AND round='400' ORDER BY date LIMIT 5
    """).fetchall():
        print(dict(r) if not isinstance(r, dict) else r)
    total400 = conn.execute("SELECT COUNT(*) as n FROM matches WHERE idLeague=4480 AND round='400'").fetchone()
    print("total round=400:", dict(total400) if not isinstance(total400, dict) else total400)

    conn.close()


if __name__ == "__main__":
    main()
