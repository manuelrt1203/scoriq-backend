import db_conn

def main():
    conn = db_conn.get_connection()

    dupes = conn.execute("""
        SELECT match_date, home_team, away_team, COUNT(*) as cnt
        FROM predictions_history
        GROUP BY match_date, home_team, away_team
        HAVING COUNT(*) > 1
    """).fetchall()
    print(f"{len(dupes)} groupes avec doublons trouves.")

    placeholder = "%s" if conn.is_pg else "?"

    for row in dupes:
        ids = conn.execute(
            f"SELECT id FROM predictions_history WHERE match_date = {placeholder} AND home_team = {placeholder} AND away_team = {placeholder} ORDER BY prediction_run_date DESC",
            (row["match_date"], row["home_team"], row["away_team"])
        ).fetchall()
        ids_to_delete = [r["id"] for r in ids[1:]]
        if ids_to_delete:
            if conn.is_pg:
                conn.execute("DELETE FROM predictions_history WHERE id = ANY(%s)", (ids_to_delete,))
            else:
                ph = ",".join(["?"]*len(ids_to_delete))
                conn.execute(f"DELETE FROM predictions_history WHERE id IN ({ph})", ids_to_delete)
            home = row["home_team"]
            away = row["away_team"]
            date = row["match_date"]
            n = len(ids_to_delete)
            print(f"  Supprime {n} doublon(s) pour {home} vs {away} ({date})")

    conn.commit()
    print("Nettoyage termine.")
    conn.close()

if __name__ == "__main__":
    main()
