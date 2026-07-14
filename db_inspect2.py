import db_conn


def main():
    conn = db_conn.get_connection()

    print("=== Plage de dates globale (matches) ===")
    r = conn.execute("SELECT MIN(date) as min_d, MAX(date) as max_d, COUNT(*) as n FROM matches").fetchone()
    print(dict(r))

    print("\n=== Plage de dates par source ===")
    for r in conn.execute("""
        SELECT source, MIN(date) as min_d, MAX(date) as max_d, COUNT(*) as n
        FROM matches GROUP BY source
    """).fetchall():
        print(dict(r))

    print("\n=== Noms d'équipes Real Madrid par source (test H2H) ===")
    for r in conn.execute("""
        SELECT DISTINCT home as team, source FROM matches
        WHERE home LIKE '%Real Madrid%' OR home LIKE '%Bayern%'
    """).fetchall():
        print(dict(r))

    print("\n=== Exemple H2H cassé : Real Madrid vs Bayern, toutes variantes ===")
    for r in conn.execute("""
        SELECT home, away, date, source FROM matches
        WHERE (home LIKE '%Real Madrid%' AND away LIKE '%Bayern%')
           OR (home LIKE '%Bayern%' AND away LIKE '%Real Madrid%')
        ORDER BY date DESC LIMIT 10
    """).fetchall():
        print(dict(r))

    conn.close()


if __name__ == "__main__":
    main()
