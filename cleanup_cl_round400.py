"""
Nettoyage ponctuel : supprime les lignes round='400' de la Ligue des
Champions (idLeague=4480), polluées par un bug de l'endpoint TheSportsDB
(retourne d'anciennes finales avec des noms d'équipes corrompus au lieu
d'un résultat vide). Le round 400 a été retiré de KNOCKOUT_ROUNDS dans
update_base.py pour empêcher que ça ne se reproduise.
"""
import db_conn


def main():
    conn = db_conn.get_connection()

    before = conn.execute(
        "SELECT COUNT(*) as n FROM matches WHERE idLeague=4480 AND round='400'"
    ).fetchone()
    print(f"Lignes round=400 avant nettoyage : {before['n']}")

    conn.execute("DELETE FROM matches WHERE idLeague=4480 AND round='400'")
    conn.commit()

    after = conn.execute(
        "SELECT COUNT(*) as n FROM matches WHERE idLeague=4480 AND round='400'"
    ).fetchone()
    print(f"Lignes round=400 après nettoyage : {after['n']}")

    conn.close()


if __name__ == "__main__":
    main()
