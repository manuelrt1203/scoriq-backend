"""
Rattrapage ponctuel : le pipeline quotidien (daily_update.yml) est resté
silencieux ~2,5 mois (04/05/2026 → 14/07/2026, désactivation GitHub après
60 jours sans activité sur le dépôt) puis n'a tourné que sporadiquement
jusqu'à mi-août. Résultat : des centaines de matchs joués n'ont jamais reçu
de prédiction.

Ce script rejoue predict_v3.py --backdate sur chaque jour manquant identifié
(matchs FINISHED présents dans `matches` mais absents de `predictions_history`
pour cette date). Pas de fuite de données : predict_v3.py en mode backdate
construit l'historique de chaque équipe strictement AVANT la date ciblée.

Usage : python backfill_missed_predictions.py
"""
import sys
import time

import predict_v3

# Jours identifiés le 29/08/2026 comme ayant des matchs FINISHED (LEAGUE,
# DOMESTIC_CUP, EUROPE, INTERNATIONAL) sans aucune ligne dans
# predictions_history pour cette match_date.
MISSING_DATES = [
    "2026-03-16", "2026-03-18", "2026-03-19", "2026-03-20", "2026-03-21",
    "2026-03-22", "2026-03-25", "2026-03-26", "2026-03-27", "2026-03-29",
    "2026-03-31", "2026-04-03", "2026-04-04", "2026-04-05", "2026-04-06",
    "2026-04-07", "2026-04-08", "2026-04-09", "2026-04-10", "2026-04-11",
    "2026-04-12", "2026-04-13", "2026-04-14", "2026-04-15", "2026-04-17",
    "2026-04-22", "2026-04-23", "2026-04-24", "2026-04-25", "2026-04-26",
    "2026-04-27", "2026-04-28", "2026-04-29", "2026-05-01", "2026-05-02",
    "2026-05-03", "2026-05-05", "2026-05-06", "2026-05-08", "2026-05-09",
    "2026-05-10", "2026-05-11", "2026-05-12", "2026-05-13", "2026-05-14",
    "2026-05-15", "2026-05-16", "2026-05-17", "2026-05-18", "2026-05-19",
    "2026-05-22", "2026-05-23", "2026-05-24", "2026-05-30", "2026-06-11",
    "2026-06-12", "2026-06-13", "2026-06-14", "2026-06-15", "2026-06-16",
    "2026-06-17", "2026-06-18", "2026-06-19", "2026-06-20", "2026-06-21",
    "2026-06-22", "2026-06-23", "2026-06-24", "2026-06-25", "2026-06-26",
    "2026-06-27", "2026-06-28", "2026-06-29", "2026-06-30", "2026-07-01",
    "2026-07-02", "2026-07-03", "2026-07-04", "2026-07-05", "2026-07-06",
    "2026-07-07", "2026-07-08", "2026-07-09", "2026-07-10", "2026-07-11",
    "2026-07-12", "2026-07-16", "2026-07-21", "2026-07-22", "2026-07-23",
    "2026-07-28", "2026-07-29", "2026-07-30", "2026-08-04", "2026-08-05",
    "2026-08-11", "2026-08-13", "2026-08-17",
]


def main():
    print(f"{len(MISSING_DATES)} jours à rattraper", flush=True)
    for i, d in enumerate(MISSING_DATES, 1):
        print(f"\n=== [{i}/{len(MISSING_DATES)}] {d} ===", flush=True)
        sys.argv = ["predict_v3.py", d]
        try:
            predict_v3.main()
        except Exception as e:
            print(f"  ERREUR sur {d} : {e}", flush=True)
        time.sleep(0.5)
    print("\nRattrapage terminé.", flush=True)


if __name__ == "__main__":
    main()
