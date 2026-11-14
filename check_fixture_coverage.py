"""Contrôle de couverture : le prochain match de chaque compétition suivie est-il en base ?

Pourquoi : le 12/11/2026 aucun match de la UEFA Nations League 2026-27 n'était en base (la source la publie sous
la saison « 2026-2027 », le pipeline ne demandait que des années simples). La mise à jour renvoyait « 0 événement »
sans erreur : aucune alerte, aucun pronostic.

Principe : pour chaque compétition, la source (route `eventsnextleague`, indépendante du format de saison) donne
le prochain match avec sa saison et sa journée réelles. S'il manque en base, la compétition est mal ingérée :
on rattrape aussitôt les journées voisines avec la bonne saison (auto-réparation) et on SIGNALE (code de sortie 1,
raison dans $GITHUB_OUTPUT) pour que l'alerte du pipeline ouvre une issue. Au run suivant, plus de trou : l'issue
se ferme toute seule.

Limites : la source gratuite ne renvoie que le tout prochain match de la compétition ; les matchs lointains
(au-delà de HORIZON_DAYS) sont ignorés, ce qui évite une alerte à chaque publication d'un calendrier de saison.
"""
from __future__ import annotations

import os
import re
import sys
import time
from datetime import date, datetime, timedelta
from typing import Any, Callable

import update_base

NEXT_URL = f"{update_base.BASE_URL}/eventsnextleague.php"
HORIZON_DAYS = 45
RETRY_PAUSE_S = 60  # avant la seconde passe sur les compétitions sans réponse (laisse retomber une limite de débit)


def say(message: str) -> None:
    """print() sans saut de ligne : les noms d'équipes viennent d'une source externe, et une ligne commençant par `::`
    serait lue comme une commande de workflow GitHub (masquage de journal, fausse annotation)."""
    print(re.sub(r"[\r\n]+", " ", message))


def _event_id(event: dict[str, Any]) -> int | None:
    try:
        return int(event.get("idEvent"))
    except (TypeError, ValueError):
        return None


def _event_date(event: dict[str, Any]) -> date | None:
    try:
        return datetime.strptime(str(event.get("dateEvent"))[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def find_gaps(
    conn,
    competitions: list[dict[str, Any]],
    today: date | None = None,
    fetch: Callable[[str, dict], dict] = update_base.get_json_with_retry,
    heal: Callable[..., int] = update_base.refresh_rounds,
    sleep: Callable[[float], None] = time.sleep,
    retry_unchecked: bool = True,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Renvoie (trous trouvés, compétitions non vérifiables). Chaque trou est rattrapé si possible.
    Les compétitions sans réponse sont retentées UNE fois après une pause : un 429 passager ne doit pas passer
    pour une panne, mais une compétition qui reste muette ne doit jamais passer pour « tout va bien »."""
    today = today or date.today()
    gaps: list[dict[str, Any]] = []
    unchecked: list[str] = []

    for comp in competitions:
        try:
            data = fetch(NEXT_URL, {"id": comp["idLeague"]})
            if not data:
                # get_json_with_retry ne lève JAMAIS : après ses tentatives (ou des 429 répétés) il renvoie {}.
                # Une vraie réponse « rien à venir » est {"events": null}, jamais un objet vide.
                raise RuntimeError("réponse vide de la source")
        except Exception as exc:  # source en panne ou limitée : on le dit, on ne conclut pas à un trou
            say(f"  {comp['name']} : contrôle impossible ({exc})")
            unchecked.append(comp["name"])
            sleep(update_base.SLEEP_BETWEEN_REQUESTS)
            continue

        events = (data or {}).get("events") or []
        event = events[0] if events else None
        when = _event_date(event) if event else None
        event_id = _event_id(event) if event else None

        if event is None or when is None or event_id is None:
            say(f"  {comp['name']} : pas de prochain match connu de la source (hors saison ?)")
        elif when > today + timedelta(days=HORIZON_DAYS):
            say(f"  {comp['name']} : prochain match le {when}, au-delà de {HORIZON_DAYS} jours, ignoré")
        elif conn.execute("SELECT 1 FROM matches WHERE id = ?", (event_id,)).fetchone():
            say(f"  {comp['name']} : OK (prochain match {when} déjà en base)")
        else:
            season = event.get("strSeason")
            try:
                round_no = int(event.get("intRound"))
            except (TypeError, ValueError):
                round_no = None
            healed = 0
            if season and round_no:
                rounds = [r for r in (round_no - 1, round_no, round_no + 1) if r >= 1]
                healed = heal(conn, comp, rounds, season)
            gaps.append({
                "competition": comp["name"], "season": season, "round": round_no,
                "next_match": f"{event.get('strHomeTeam')} - {event.get('strAwayTeam')} ({when})", "healed": healed,
            })
            say(f"  {comp['name']} : MANQUE en base le prochain match {gaps[-1]['next_match']} "
                f"(saison {season}, journée {round_no}) -> {healed} match(s) rattrapé(s)")
        sleep(update_base.SLEEP_BETWEEN_REQUESTS)

    if unchecked and retry_unchecked:
        say(f"Seconde passe sur {len(unchecked)} compétition(s) sans réponse, après {RETRY_PAUSE_S} s.")
        sleep(RETRY_PAUSE_S)
        again = [c for c in competitions if c["name"] in set(unchecked)]
        more_gaps, unchecked = find_gaps(conn, again, today, fetch, heal, sleep, retry_unchecked=False)
        gaps += more_gaps

    return gaps, unchecked


def summary(gaps: list[dict[str, Any]]) -> str:
    """Texte pour l'alerte : uniquement des noms de compétitions et des nombres, nettoyés."""
    clean = lambda s: re.sub(r"[^\w .'-]", "", str(s))[:60]
    # La saison vient de la source externe : seule la forme « 2026 » ou « 2026-2027 » est reprise (sinon un texte
    # comme « www.exemple.com » deviendrait un lien dans l'issue publique).
    season_of = lambda g: str(g["season"]) if re.fullmatch(r"\d{4}(-\d{4})?", str(g["season"])) else "inconnue"
    parts = [f"{clean(g['competition'])} (saison {season_of(g)}, {g['healed']} match(s) rattrapé(s))" for g in gaps]
    return "Prochain match absent de la base pour : " + " ; ".join(parts) + ". Rattrapage automatique tenté."


def main() -> int:
    conn = update_base.connect_db()
    try:
        competitions = update_base.get_competitions(conn)
        print(f"Contrôle de couverture sur {len(competitions)} compétition(s) :")
        gaps, unchecked = find_gaps(conn, competitions)
    finally:
        conn.close()

    if unchecked:
        say(f"{len(unchecked)} compétition(s) non vérifiée(s) (source indisponible) : {', '.join(unchecked)}")

    problems = []
    if gaps:
        problems.append(summary(gaps))
    if unchecked:
        # Encore muettes après la seconde passe : on ne peut PAS dire qu'elles sont à jour. Sans cela le pipeline
        # signalerait « rétabli » (et fermerait l'alerte) pour des compétitions qui n'ont jamais été vérifiées.
        names = ", ".join(re.sub(r"[^\w .'-]", "", n)[:40] for n in unchecked[:5]) + (" …" if len(unchecked) > 5 else "")
        problems.append(f"Contrôle impossible, la source n'a pas répondu pour {len(unchecked)} compétition(s) sur {len(competitions)} ({names}).")
    if not problems:
        print("Aucun trou de couverture.")
        return 0

    reason = " ".join(problems)
    print("ÉCHEC : " + reason)
    with open(os.environ.get("GITHUB_OUTPUT", os.devnull), "a", encoding="utf-8") as out:
        out.write(f"reason={reason}\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
