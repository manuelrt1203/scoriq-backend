"""Mesures publiques de la qualité des pronostics 1X2, calculées sur l'historique évalué.

Fonctions pures (aucun accès base) : `main.py` leur passe les lignes de `predictions_history`.

- Calibration : quand le modèle annonce X % pour son pronostic principal, quelle part se réalise ?
- Scores : exactitude, Brier et log-loss du modèle (références du hasard : 0,667 et 1,099).
- Comparaison au marché : modèle et bookmakers sur les MÊMES matchs (ceux qui ont des cotes), cotes
  renormalisées pour que la somme fasse 1.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Iterable

OUTCOMES = ("1", "X", "2")
CALIBRATION_BINS = [(0.33, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 0.75), (0.75, 0.85), (0.85, 0.95), (0.95, 1.0001)]
MIN_MARKET_MATCHES = 30  # en dessous, la comparaison n'est pas assez solide pour être publiée


def _number(value: Any) -> float | None:
    """float fini et positif ou nul, sinon None (NULL, texte, NaN, inf ; Decimal accepté)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) and x >= 0 else None


def _probabilities(row: dict, keys: tuple[str, str, str]) -> dict[str, float] | None:
    """Probabilités 1/X/2 normalisées à 1, ou None si une valeur manque ou si leur somme est nulle."""
    values = [_number(row.get(k)) for k in keys]
    if any(v is None for v in values):
        return None
    total = sum(values)
    if total <= 0:
        return None
    return {o: v / total for o, v in zip(OUTCOMES, values)}


MODEL_KEYS = ("proba_home_win", "proba_draw", "proba_away_win")
MARKET_KEYS = ("market_proba_home", "market_proba_draw", "market_proba_away")


def model_probabilities(row: dict) -> dict[str, float] | None:
    return _probabilities(row, MODEL_KEYS)


def market_probabilities(row: dict) -> dict[str, float] | None:
    return _probabilities(row, MARKET_KEYS)


def _scorable(rows: Iterable[dict]) -> list[dict]:
    """Lignes dont le résultat réel est connu (1, X ou 2) et qui ont des probabilités de modèle utilisables."""
    return [r for r in rows if r.get("real_result") in OUTCOMES and model_probabilities(r) is not None]


def _pick(p: dict[str, float]) -> str:
    return max(OUTCOMES, key=lambda o: p[o])


def summarize(rows: list[dict], probabilities: Callable[[dict], dict[str, float] | None]) -> dict | None:
    """Exactitude, Brier (somme des écarts quadratiques sur 1/X/2) et log-loss moyens. None s'il n'y a aucune ligne."""
    n = brier = loss = correct = 0
    for r in rows:
        p = probabilities(r)
        if p is None or r.get("real_result") not in OUTCOMES:
            continue
        y = r["real_result"]
        n += 1
        brier += sum((p[o] - (1.0 if o == y else 0.0)) ** 2 for o in OUTCOMES)
        loss += -math.log(max(p[y], 1e-6))
        correct += _pick(p) == y
    if n == 0:
        return None
    return {"n": n, "accuracy": round(correct / n, 4), "brier": round(brier / n, 4), "log_loss": round(loss / n, 4)}


def calibration_bins(rows: list[dict]) -> list[dict]:
    """Par tranche de probabilité annoncée pour le pronostic principal : effectif, probabilité annoncée
    moyenne et fréquence observée. Les tranches vides sont omises."""
    out = []
    for low, high in CALIBRATION_BINS:
        predicted_sum = hits = count = 0
        for r in rows:
            p = model_probabilities(r)
            if p is None or r.get("real_result") not in OUTCOMES:
                continue
            pick = _pick(p)
            if low <= p[pick] < high:
                count += 1
                predicted_sum += p[pick]
                hits += pick == r["real_result"]
        if count:
            out.append({
                "low": low, "high": min(high, 1.0), "count": count,
                "predicted": round(predicted_sum / count, 4), "observed": round(hits / count, 4),
            })
    return out


def market_comparison(rows: list[dict], total_evaluated: int) -> dict | None:
    """Modèle et marché sur les mêmes matchs. None si trop peu de matchs ont des cotes."""
    with_odds = [r for r in rows if market_probabilities(r) is not None]
    if len(with_odds) < MIN_MARKET_MATCHES:
        return None
    return {
        "n": len(with_odds),
        "coverage": round(len(with_odds) / total_evaluated, 4) if total_evaluated else None,
        "model": summarize(with_odds, model_probabilities),
        "market": summarize(with_odds, market_probabilities),
    }


def build_public_stats(rows: list[dict]) -> dict:
    scorable = _scorable(rows)
    return {
        "calibration": calibration_bins(scorable),
        "overall": summarize(scorable, model_probabilities),
        "market": market_comparison(scorable, len(scorable)),
        "reference": {"brier_chance": round(2 / 3, 3), "log_loss_chance": round(math.log(3), 3)},
    }
