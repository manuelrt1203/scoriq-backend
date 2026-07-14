import db_conn


def main():
    conn = db_conn.get_connection()

    print("=== Tout match France <-> Spain, sans filtre de statut ===")
    for r in conn.execute("""
        SELECT home, away, date, status, home_score, away_score, competition_name, source
        FROM matches
        WHERE (home='France' AND away='Spain') OR (home='Spain' AND away='France')
        ORDER BY date DESC
    """).fetchall():
        print(dict(r))

    print("\n=== Variantes de noms contenant 'France' ou 'Spain' ===")
    for r in conn.execute("""
        SELECT DISTINCT home as team, source FROM matches
        WHERE home LIKE '%France%' OR home LIKE '%Spain%'
        ORDER BY team
    """).fetchall():
        print(dict(r))

    conn.close()


if __name__ == "__main__":
    main()
