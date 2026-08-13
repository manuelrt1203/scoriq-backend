"""Diagnostic en lecture seule — recherche match PSG vs Aston Villa récent (Supercoupe ?)."""
import db_conn


def main():
    conn = db_conn.get_connection()

    print("=== Tous les matches PSG vs Aston Villa (toutes compétitions, tous statuts) ===")
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

    print("\n=== Matches PSG (toutes compétitions) autour du 2026-08-11 / 12 / 13 ===")
    for r in conn.execute("""
        SELECT id, home, away, home_score, away_score, date, status, competition_name, season, round, source
        FROM matches
        WHERE (home ILIKE '%psg%' OR home ILIKE '%paris%' OR away ILIKE '%psg%' OR away ILIKE '%paris%')
          AND date >= '2026-08-08' AND date <= '2026-08-13'
        ORDER BY date DESC
    """).fetchall():
        print(dict(r) if not isinstance(r, dict) else r)

    print("\n=== Matches Aston Villa (toutes compétitions) autour du 2026-08-11 / 12 / 13 ===")
    for r in conn.execute("""
        SELECT id, home, away, home_score, away_score, date, status, competition_name, season, round, source
        FROM matches
        WHERE (home ILIKE '%aston%villa%' OR away ILIKE '%aston%villa%')
          AND date >= '2026-08-08' AND date <= '2026-08-13'
        ORDER BY date DESC
    """).fetchall():
        print(dict(r) if not isinstance(r, dict) else r)

    print("\n=== Compétitions contenant 'super' (Supercoupe / Super Cup) ===")
    for r in conn.execute("""
        SELECT idLeague, name, competition_type FROM competitions WHERE name ILIKE '%super%'
    """).fetchall():
        print(dict(r) if not isinstance(r, dict) else r)

    conn.close()


if __name__ == "__main__":
    main()
