"""
Table de correspondance des noms d'équipes fragmentés (même club stocké sous
plusieurs orthographes — ex: "Porto" / "FC Porto", "Ipswich" / "Ipswich Town" —
qui casse silencieusement les badges, l'historique de features (build_features
dans predict_v3.py) et l'entraînement Dixon-Coles).

Remplace le dict TEAM_NAME_MERGES codé en dur dans sync_team_badges.py : ajouter
un alias devient une opération de données (`python team_aliases.py add "Ancien nom"
"Nom canonique"`), plus besoin de modifier du code et redéployer sur Render.

Convention du nom canonique : celui déjà utilisé par TheSportsDB pour les
fixtures à venir (source de référence pour la dédup, voir mémoire
feedback_workflow — "Noms d'équipes divergent entre les 3 sources").

Table `team_aliases` (alias -> canonical_name). Consommée par :
- sync_team_badges.py (fusion nightly des données déjà en base)
- update_base.py (résolution à l'ingestion, empêche la fragmentation de se
  reproduire au prochain import TheSportsDB — cf. récidive Porto du 02/09/2026)
"""
import sys
from datetime import date, timedelta

import db_conn

TABLES_COLUMNS = [
    ("matches", "home"), ("matches", "away"),
    ("odds", "home_team"), ("odds", "away_team"),
    ("predictions_history", "home_team"), ("predictions_history", "away_team"),
    ("shots_data", "home_team"), ("shots_data", "away_team"),
]

# Tables où renommer « alias » -> « canonique » peut doublonner une ligne canonique déjà présente pour la même
# rencontre (contrainte d'unicité) : on supprime alors la ligne d'alias avant de renommer. Valeur = colonnes qui
# identifient la rencontre, hors équipes. Uniquement shots_data (tirs : donnée dérivée). JAMAIS predictions_history
# (historique public), matches ni odds : un conflit y reste une erreur visible, pas une suppression silencieuse.
DEDUP_KEYS = {"shots_data": ("idleague", "season", "match_date")}

# Alias déjà identifiés manuellement (incidents PSG 28/08, Köln/Porto/Man Utd/
# Braga/Sporting 29/08, Ipswich 02/09) — seed de la table à sa création.
KNOWN_ALIASES = {
    "Paris SG": "Paris Saint-Germain",
    "Paris Saint-Germain FC": "Paris Saint-Germain",
    "FC Koln": "FC Köln",
    "Köln": "FC Köln",
    "Porto": "FC Porto",
    "Manchester United FC": "Manchester United",
    "Sporting Clube de Braga": "Braga",
    "Sporting Clube de Portugal": "Sporting CP",
    "Ipswich": "Ipswich Town",
}


def ensure_table(conn: db_conn.Connection) -> None:
    conn.execute_script("""
        CREATE TABLE IF NOT EXISTS team_aliases (
            alias TEXT PRIMARY KEY,
            canonical_name TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


def seed_known_aliases(conn: db_conn.Connection) -> None:
    ensure_table(conn)
    for alias, canonical in KNOWN_ALIASES.items():
        conn.execute("""
            INSERT INTO team_aliases (alias, canonical_name) VALUES (?, ?)
            ON CONFLICT (alias) DO NOTHING
        """, (alias, canonical))
    conn.commit()


def load_alias_map(conn: db_conn.Connection) -> dict[str, str]:
    ensure_table(conn)
    rows = conn.execute("SELECT alias, canonical_name FROM team_aliases").fetchall()
    return {r["alias"]: r["canonical_name"] for r in rows}


def drop_alias_duplicates(conn: db_conn.Connection, table: str, col: str, alias: str, canonical: str) -> None:
    """Supprime les lignes `col = alias` qui existent déjà sous `col = canonical` pour la même rencontre.
    Sans cela, l'UPDATE de fusion viole la contrainte d'unicité de la table (cas réel : shots_data, 743 doublons
    pour « Paris SG », « FC Koln » et « Ipswich », qui bloquaient sync_team_badges depuis le 12/11/2026)."""
    other = "away_team" if col == "home_team" else "home_team"
    same = " AND ".join(f"b.{k} = {table}.{k}" for k in (*DEDUP_KEYS[table], other))
    conn.execute(
        f"DELETE FROM {table} WHERE {col} = ? AND EXISTS (SELECT 1 FROM {table} b WHERE {same} AND b.{col} = ?)",
        (alias, canonical),
    )


def merge_alias(conn: db_conn.Connection, alias: str, canonical: str) -> None:
    if alias == canonical:
        # Sans ce garde-fou, le DELETE de drop_alias_duplicates retrouverait chaque ligne comme son propre « doublon »
        # et effacerait toute l'équipe de shots_data (seul add_alias refusait ce cas ; apply_all lit la table telle quelle).
        # On le dit : une ligne pareille est une erreur de saisie à corriger dans team_aliases, pas à cacher.
        print(f"[ALIAS] ligne ignorée : alias identique au nom canonique ({alias!r}), à corriger dans team_aliases")
        return
    for table, col in TABLES_COLUMNS:
        if table in DEDUP_KEYS:
            drop_alias_duplicates(conn, table, col, alias, canonical)
        conn.execute(f"UPDATE {table} SET {col} = ? WHERE {col} = ?", (canonical, alias))
    conn.commit()


def apply_all(conn: db_conn.Connection) -> None:
    """Rejoue tous les alias connus sur les 4 tables. Idempotent (no-op si déjà fusionné).
    Un alias en échec n'empêche plus les suivants : on annule SON travail, on continue, puis on lève une erreur
    récapitulative à la fin (le pipeline la signale par une issue, comme avant)."""
    errors = []
    for alias, canonical in load_alias_map(conn).items():
        try:
            merge_alias(conn, alias, canonical)
        except Exception as e:  # noqa: BLE001 : on veut tous les alias traités avant d'échouer
            conn.rollback()
            errors.append(f"{alias!r} -> {canonical!r} : {type(e).__name__}: {e}")
    if errors:
        raise RuntimeError(f"Fusion impossible pour {len(errors)} alias :\n" + "\n".join(errors))


def add_alias(conn: db_conn.Connection, alias: str, canonical: str) -> None:
    if alias == canonical:
        raise ValueError("alias et nom canonique identiques")
    ensure_table(conn)
    conn.execute("""
        INSERT INTO team_aliases (alias, canonical_name) VALUES (?, ?)
        ON CONFLICT (alias) DO UPDATE SET canonical_name = excluded.canonical_name
    """, (alias, canonical))
    conn.commit()
    merge_alias(conn, alias, canonical)


def scan_candidates(conn: db_conn.Connection) -> list[tuple[str, str, int, int]]:
    """Détecte des paires de noms probablement fragmentées, sans jamais fusionner
    automatiquement (cf. incident "Union SG -> club équatorien sans rapport" dans
    sync_team_badges.py : un mauvais merge est pire qu'un merge manquant).
    Heuristique : un nom est strictement contenu dans un autre (ex: "Porto" dans
    "FC Porto"), les deux ont joué dans les 2 dernières saisons, et ne sont pas
    déjà résolus par un alias existant."""
    known = load_alias_map(conn)
    # matches.date est du texte ISO : limite calculée ici (valable pour SQLite et PostgreSQL).
    since = (date.today() - timedelta(days=730)).isoformat()
    rows = conn.execute("""
        SELECT team, COUNT(*) AS n FROM (
            SELECT home AS team FROM matches WHERE date >= ?
            UNION ALL
            SELECT away FROM matches WHERE date >= ?
        ) t
        GROUP BY team
        ORDER BY team
    """, (since, since)).fetchall()

    teams = [(r["team"], r["n"]) for r in rows if r["team"]]
    candidates = []
    for i, (name_a, n_a) in enumerate(teams):
        for name_b, n_b in teams[i + 1:]:
            if name_a == name_b:
                continue
            if name_a in known or name_b in known:
                continue
            short, long_ = (name_a, name_b) if len(name_a) < len(name_b) else (name_b, name_a)
            if len(short) < 4:
                continue
            if short == long_[: len(short)] or f" {short}" in long_ or long_.startswith(f"{short} "):
                n_short = n_a if short == name_a else n_b
                n_long = n_b if short == name_a else n_a
                candidates.append((short, long_, n_short, n_long))
    return candidates


def _cli():
    if len(sys.argv) < 2:
        print("Usage:")
        print('  python team_aliases.py add "Ancien nom" "Nom canonique"')
        print("  python team_aliases.py list")
        print("  python team_aliases.py scan")
        print("  python team_aliases.py apply")
        sys.exit(1)

    conn = db_conn.get_connection()
    try:
        cmd = sys.argv[1]
        if cmd == "add":
            alias, canonical = sys.argv[2], sys.argv[3]
            add_alias(conn, alias, canonical)
            print(f"Alias ajouté et fusionné : {alias!r} -> {canonical!r}")
        elif cmd == "list":
            for alias, canonical in sorted(load_alias_map(conn).items()):
                print(f"  {alias!r} -> {canonical!r}")
        elif cmd == "scan":
            candidates = scan_candidates(conn)
            if not candidates:
                print("Aucun candidat trouvé.")
            for short, long_, n_short, n_long in candidates:
                print(f'  "{short}" ({n_short} matchs) vs "{long_}" ({n_long} matchs) — à vérifier manuellement avant merge')
        elif cmd == "apply":
            apply_all(conn)
            print("Alias connus réappliqués sur matches/odds/predictions_history/shots_data.")
        else:
            print(f"Commande inconnue : {cmd}")
            sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    _cli()
