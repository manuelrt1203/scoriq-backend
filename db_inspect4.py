import db_conn


def main():
    conn = db_conn.get_connection()
    print("=== Équipes CL saison 2025-2026, thesportsdb uniquement ===")
    for r in conn.execute("""
        SELECT DISTINCT home as team FROM matches
        WHERE idLeague=4480 AND season='2025-2026' AND source='thesportsdb'
        ORDER BY team
    """).fetchall():
        print(dict(r))
    conn.close()


if __name__ == "__main__":
    main()
