import db_conn


def main():
    conn = db_conn.get_connection()

    print("=== Équipes WC (idLeague=4429) par source ===")
    for r in conn.execute("""
        SELECT DISTINCT home as team, source FROM matches WHERE idLeague=4429 ORDER BY source, team
    """).fetchall():
        print(dict(r))

    print("\n=== Équipes CL (idLeague=4480) par source ===")
    for r in conn.execute("""
        SELECT DISTINCT home as team, source FROM matches WHERE idLeague=4480 ORDER BY source, team
    """).fetchall():
        print(dict(r))

    conn.close()


if __name__ == "__main__":
    main()
