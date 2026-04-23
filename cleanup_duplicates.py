import db_conn

def main():
    conn = db_conn.get_connection()

    if conn.is_pg:
        dupes = conn.execute("""
            SELECT match_date, home_team, away_team, COUNT(*) as cnt
            FROM predictions_history
            GROUP BY match_date, home_team, away_team
            HAVING COUNT(*) > 1
        """).fetchall()
        print(f"{len(dupes)} groupes avec doublons trouvés.")

        for row in dupes:
            ids = conn.execute("""
                SELECT id FROM predictions_history
                WHERE match_date = %s AND home_team = %s AND away_team = %s
                ORDER BY prediction_run_date DESC
            """, (row["match_date"], row["home_team"], row["away_team"])).fetchall()
            ids_to_delete = [r["id"] for r in ids[1:]]
            if ids_to_delete:
                conn.execute(
                    "DELETE FROM predictions_history WHERE id = ANY(%s)",
                    (ids_to_delete,)
                )
                print(f"  Supprimé {len(ids_to_delete)} doublon(s) pour {row[home_team]} vs {row[away_team]} ({row[match_date]})")
    else:
        dupes = conn.execute("""
            SELECT match_date, home_team, away_team, COUNT(*) as cnt
            FROM predictions_history
            GROUP BY match_date, home_team, away_team
            HAVING COUNT(*) > 1
        """).fetchall()
        print(f"{len(dupes)} groupes avec doublons trouvés.")

        for row in dupes:
            ids = conn.execute("""
                SELECT id FROM predictions_history
                WHERE match_date = ? AND home_team = ? AND away_team = ?
                ORDER BY prediction_run_date DESC
            """, (row["match_date"], row["home_team"], row["away_team"])).fetchall()
            ids_to_delete = [r["id"] for r in ids[1:]]
            if ids_to_delete:
                placeholders = ",".join("?" * len(ids_to_delete))
                conn.execute(
                    f"DELETE FROM predictions_history WHERE id IN ({placeholders})",
                    ids_to_delete
                )
                print(f"  Supprimé {len(ids_to_delete)} doublon(s) pour {row[\"home_team\"]} vs {row[\"away_team\"]} ({row[\"match_date\"]})")

    conn.commit()
    print("Nettoyage terminé.")
    conn.close()

if __name__ == "__main__":
    main()

