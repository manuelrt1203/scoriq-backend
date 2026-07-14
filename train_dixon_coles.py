"""
Modèle Dixon-Coles (1997) avec pondération temporelle — version Postgres/SQLite
(db_conn) + vectorisée numpy, destinée à tourner régulièrement dans le pipeline
(contrairement à l'original, ponctuel et bien trop lent pour un cron quotidien :
il recalculait le poids temporel de chaque match — indépendant des paramètres
optimisés — à chaque évaluation de la fonction, en boucle Python pure).

Pour chaque équipe i on estime :
  α_i  : force d'attaque
  δ_i  : force de défense (négatif = bonne défense)
  γ    : avantage domicile (par type de compétition)
  ρ    : correction pour les faibles scores (0-0, 1-0, 0-1, 1-1)

Lambdas :
  λ_home = exp(α_home + δ_away + γ)
  λ_away = exp(α_away + δ_home)

La pondération temporelle exponentielle (XI) fait que les matchs récents
(ex : le tournoi en cours) dominent naturellement l'estimation par rapport
à l'historique ancien, sans logique spécifique par compétition.
"""

import json
import math
from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy.special import gammaln
from scipy.optimize import minimize

import db_conn

PARAMS_PATH = "dixon_coles_params.json"

XI = 0.0065          # décroissance exponentielle par jour (~365j → poids ≈ 0.09)
MIN_MATCHES = 5      # matchs minimum pour inclure une équipe
MIN_WEIGHT = 1e-10   # matchs plus anciens que ça sont exclus (poids négligeable)
L2_REG = 0.03        # shrinkage sur attack/defense — évite les valeurs extrêmes
                     # pour les équipes à faible échantillon (MLE non régularisé diverge)

TARGET_COMPETITION_TYPES = ("LEAGUE", "DOMESTIC_CUP", "EUROPE", "INTERNATIONAL")
COMP_TYPES = ["LEAGUE", "DOMESTIC_CUP", "EUROPE", "INTERNATIONAL"]


def load_matches(conn):
    placeholders = ",".join("?" for _ in TARGET_COMPETITION_TYPES)
    return conn.execute(f"""
        WITH sdb_rounds AS (
            SELECT DISTINCT idLeague, season, round
            FROM matches
            WHERE source = 'thesportsdb'
        )
        SELECT m.home, m.away, m.home_score, m.away_score, m.date, m.competition_type
        FROM matches m
        WHERE m.status = 'FINISHED'
          AND m.home_score IS NOT NULL
          AND m.away_score IS NOT NULL
          AND m.competition_type IN ({placeholders})
          AND (
              COALESCE(m.source, 'thesportsdb') = 'thesportsdb'
              OR NOT EXISTS (
                  SELECT 1 FROM sdb_rounds s
                  WHERE s.idLeague = m.idLeague
                    AND s.season   = m.season
                    AND (s.round = m.round OR (s.round IS NULL AND m.round IS NULL))
              )
          )
        ORDER BY m.date
    """, TARGET_COMPETITION_TYPES).fetchall()


def time_weight(date_str, ref_date):
    try:
        dt = datetime.strptime(str(date_str)[:10], "%Y-%m-%d")
    except ValueError:
        return 0.0
    days_ago = (ref_date - dt).days
    return math.exp(-XI * days_ago) if days_ago >= 0 else 0.0


def neg_log_likelihood_and_grad(params, n, nc, home_idx, away_idx, ct_idx, hg, ag, lgamma_hg1, lgamma_ag1, weights):
    """Version vectorisée numpy + gradient analytique — équivalente à la boucle
    Python d'origine mais sans reparcourir les matchs un par un à chaque appel,
    et sans le gradient numérique par différences finies (~n_params évaluations
    par étape) : L-BFGS-B converge alors en quelques dizaines d'évaluations."""
    attack    = params[:n]
    defense   = params[n:2 * n]
    home_advs = params[2 * n: 2 * n + nc]
    rho       = params[2 * n + nc]

    lam_h = np.exp(attack[home_idx] + defense[away_idx] + home_advs[ct_idx])
    lam_a = np.exp(attack[away_idx] + defense[home_idx])

    ll_h = hg * np.log(lam_h) - lam_h - lgamma_hg1
    ll_a = ag * np.log(lam_a) - lam_a - lgamma_ag1

    tau = np.ones_like(lam_h)
    m00 = (hg == 0) & (ag == 0)
    m10 = (hg == 1) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m11 = (hg == 1) & (ag == 1)
    tau[m00] = 1 - lam_h[m00] * lam_a[m00] * rho
    tau[m10] = 1 + lam_a[m10] * rho
    tau[m01] = 1 + lam_h[m01] * rho
    tau[m11] = 1 - rho

    valid = tau > 0
    nll = -np.sum(weights[valid] * (ll_h[valid] + ll_a[valid] + np.log(tau[valid])))
    nll += 0.5 * L2_REG * np.sum(attack ** 2 + defense ** 2)

    # ── Gradient analytique ────────────────────────────────────────────────
    # d(ll_h)/d(lam_h) * d(lam_h)/d(attack[home]) = (hg/lam_h - 1) * lam_h = hg - lam_h
    # (même terme pour defense[away] et home_adv[ct], qui entrent linéairement dans lam_h)
    dlogtau_dlamh = np.zeros_like(lam_h)
    dlogtau_dlama = np.zeros_like(lam_a)
    dlogtau_drho  = np.zeros_like(lam_h)

    vm00 = m00 & valid
    dlogtau_dlamh[vm00] = -lam_a[vm00] * rho / tau[vm00]
    dlogtau_dlama[vm00] = -lam_h[vm00] * rho / tau[vm00]
    dlogtau_drho[vm00]  = -lam_h[vm00] * lam_a[vm00] / tau[vm00]

    vm10 = m10 & valid
    dlogtau_dlama[vm10] = rho / tau[vm10]
    dlogtau_drho[vm10]  = lam_a[vm10] / tau[vm10]

    vm01 = m01 & valid
    dlogtau_dlamh[vm01] = rho / tau[vm01]
    dlogtau_drho[vm01]  = lam_h[vm01] / tau[vm01]

    vm11 = m11 & valid
    dlogtau_drho[vm11] = -1.0 / tau[vm11]

    w = np.where(valid, weights, 0.0)
    contrib_h = w * ((hg - lam_h) + dlogtau_dlamh * lam_h)  # -> attack[home], defense[away], home_adv[ct]
    contrib_a = w * ((ag - lam_a) + dlogtau_dlama * lam_a)  # -> attack[away], defense[home]

    grad = np.zeros_like(params)
    np.add.at(grad[:n], home_idx, contrib_h)          # d/d(attack[home])
    np.add.at(grad[:n], away_idx, contrib_a)          # d/d(attack[away])
    np.add.at(grad[n:2 * n], away_idx, contrib_h)     # d/d(defense[away])
    np.add.at(grad[n:2 * n], home_idx, contrib_a)     # d/d(defense[home])
    np.add.at(grad[2 * n: 2 * n + nc], ct_idx, contrib_h)  # d/d(home_adv[ct])
    grad[2 * n + nc] = np.sum(w * dlogtau_drho)       # d/d(rho)

    grad = -grad  # gradient de la NLL = -gradient de la log-vraisemblance
    grad[:2 * n] += L2_REG * params[:2 * n]
    return nll, grad


def train():
    conn = db_conn.get_connection()
    matches = load_matches(conn)
    conn.close()

    print(f"{len(matches)} matchs chargés.")
    ref_date = datetime.now()

    # Poids temporel = statique (ne dépend pas des paramètres optimisés) :
    # calculé une seule fois, et les matchs négligeables sont exclus d'emblée.
    weighted = []
    for m in matches:
        w = time_weight(m["date"], ref_date)
        if w >= MIN_WEIGHT:
            weighted.append((m, w))

    counts = defaultdict(int)
    for m, _ in weighted:
        counts[m["home"]] += 1
        counts[m["away"]] += 1

    teams = sorted(t for t, c in counts.items() if c >= MIN_MATCHES)
    team_set = set(teams)
    team_index = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    print(f"{n} équipes incluses (>= {MIN_MATCHES} matchs).")

    weighted = [(m, w) for m, w in weighted if m["home"] in team_set and m["away"] in team_set]
    print(f"{len(weighted)} matchs après filtre.")

    ct_index = {ct: i for i, ct in enumerate(COMP_TYPES)}
    home_idx = np.array([team_index[m["home"]] for m, _ in weighted], dtype=np.int64)
    away_idx = np.array([team_index[m["away"]] for m, _ in weighted], dtype=np.int64)
    ct_idx   = np.array([ct_index.get(m["competition_type"], ct_index["LEAGUE"]) for m, _ in weighted], dtype=np.int64)
    hg       = np.array([int(m["home_score"]) for m, _ in weighted], dtype=np.float64)
    ag       = np.array([int(m["away_score"]) for m, _ in weighted], dtype=np.float64)
    weights  = np.array([w for _, w in weighted], dtype=np.float64)
    lgamma_hg1 = gammaln(hg + 1)
    lgamma_ag1 = gammaln(ag + 1)

    nc = len(COMP_TYPES)

    x0 = np.zeros(2 * n + nc + 1)
    for i in range(nc):
        x0[2 * n + i] = 0.30
    x0[2 * n + nc] = -0.10

    print("Optimisation MLE en cours...")
    t0 = datetime.now()
    result = minimize(
        neg_log_likelihood_and_grad,
        x0,
        args=(n, nc, home_idx, away_idx, ct_idx, hg, ag, lgamma_hg1, lgamma_ag1, weights),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 5000, "maxfun": 50000, "ftol": 1e-10},
    )
    print(f"Optimisation terminée en {(datetime.now() - t0).total_seconds():.1f}s")

    if not result.success:
        print(f"Avertissement optimisation : {result.message}")

    p = result.x
    attack   = dict(zip(teams, p[:n]))
    defense  = dict(zip(teams, p[n:2 * n]))
    home_advs = dict(zip(COMP_TYPES, p[2 * n: 2 * n + nc]))
    rho      = float(p[2 * n + nc])

    mu_a = float(np.mean(list(attack.values())))
    attack  = {t: v - mu_a for t, v in attack.items()}
    defense = {t: v + mu_a for t, v in defense.items()}

    output = {
        "trained_at":     ref_date.strftime("%Y-%m-%d %H:%M:%S"),
        "n_teams":        n,
        "n_matches":      len(weighted),
        "home_advantage": home_advs,
        "rho":            rho,
        "attack":         attack,
        "defense":        defense,
    }

    with open(PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nParamètres sauvegardés -> {PARAMS_PATH}")
    for ct, v in home_advs.items():
        print(f"Avantage domicile {ct:<15} : {v:+.4f}")
    print(f"Rho               : {rho:+.4f}")

    top_atk = sorted(attack.items(), key=lambda x: -x[1])[:5]
    top_def = sorted(defense.items(), key=lambda x: x[1])[:5]
    print("\nTop 5 attaques :")
    for t, v in top_atk:
        print(f"  {t:<30} {v:+.4f}")
    print("Top 5 défenses (plus bas = meilleur) :")
    for t, v in top_def:
        print(f"  {t:<30} {v:+.4f}")


if __name__ == "__main__":
    train()
