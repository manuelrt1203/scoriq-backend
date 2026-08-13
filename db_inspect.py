"""Diagnostic en lecture seule — recherche match PSG vs Aston Villa."""
import db_conn


def main():
    conn = db_conn.get_connection()

    print("=== Equipes correspondant à PSG / Paris ===")
    for r in conn.execute("""
        SELECT DISTINCT home as team FROM matches WHERE home ILIKE '%psg%' OR home ILIKE '%paris%'
        UNION
        SELECT DISTINCT away as team FROM matches WHERE away ILIKE '%psg%' OR away ILIKE '%paris%'
    """).fetchall():
        print(dict(r) if not isinstance(r, dict) else r)

    print("\n=== Equipes correspondant à Aston Villa ===")
    for r in conn.execute("""
        SELECT DISTINCT home as team FROM matches WHERE home ILIKE '%aston%villa%'
        UNION
        SELECT DISTINCT away as team FROM matches WHERE away ILIKE '%aston%villa%'
    """).fetchall():
        print(dict(r) if not isinstance(r, dict) else r)

    print("\n=== Match(s) PSG vs Aston Villa (dans les deux sens) ===")
    for r in conn.execute("""
        SELECT id, home, away, home_score, away_score, date, status, competition_name, season, round, source
        FROM matches
        WHERE (
            (home ILIKE '%psg%' OR home ILIKE '%paris%') AND away ILIKE '%aston%villa%'
        ) OR (
            (away ILIKE '%psg%' OR away ILIKE '%paris%') AND home ILIKE '%aston%villa%'
        )
        ORDER BY date DESC
    """).fetchall():
        print(dict(r) if not isinstance(r, dict) else r)

    conn.close()


if __name__ == "__main__":
    main()
