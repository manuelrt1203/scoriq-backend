"""
predict_v3.py — backend Railway (SQLite local ou PostgreSQL via db_conn)

Même logique que le predict_v3.py local mais utilise db_conn
pour fonctionner indifféremment en SQLite et PostgreSQL.
"""

import json
import math
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, date
from statistics import mean, variance
from collections import defaultdict

import db_conn
from scoriq_models import LeagueCalibrator  # noqa: F401 — requis pour désérialisation joblib

DC_PARAMS_PATH  = "dixon_coles_params.json"
CALIB_PATH      = "calibrators_v2.pkl"
LGBM_PATH_TMPL  = "lgbm_1x2_{}.pkl"
HGBR_HOME_TMPL  = "model_home_goals_{}.pkl"
HGBR_AWAY_TMPL  = "model_away_goals_{}.pkl"

LOOKBACK           = 5
LOOKBACK_VENUE     = 5
LOOKBACK_DRAW      = 10
MIN_LEAGUE_MATCHES = 3
MIN_VENUE_MATCHES  = 3
H2H_LOOKBACK       = 5
TARGET_COMPETITION_TYPES = ("LEAGUE", "DOMESTIC_CUP", "EUROPE", "INTERNATIONAL")

ELO_BASE       = 1500.0
ELO_K          = 20.0
HOME_ELO_BOOST = 50.0

TRUST_FORTE   = 0.60
TRUST_MOYENNE = 0.50

W_LGB_NO_MARKET = 0.65
W_DC_NO_MARKET  = 0.35
W_LGB_MARKET    = 0.45
W_DC_MARKET     = 0.15
W_MKT_MARKET    = 0.40


# ── Utilitaires ────────────────────────────────────────────────────────────────

def parse_date(d):
    d = str(d)
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(d, fmt)
        except ValueError:
            pass
    raise ValueError(f"Format non géré : {d}")


def avg(values):
    return mean(values) if values else 0.0


def var10(values):
    return variance(values) if len(values) >= 2 else 0.0


def points_from_result(gf, ga):
    return 3 if gf > ga else (1 if gf == ga else 0)


def result_score(gf, ga):
    return 1.0 if gf > ga else (0.5 if gf == ga else 0.0)


def expected_score(elo_a, elo_b):
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def update_elo(elo_home, elo_away, hg, ag):
    he = expected_score(elo_home + HOME_ELO_BOOST, elo_away)
    hr = result_score(hg, ag)
    new_h = elo_home + ELO_K * (hr - he)
    new_a = elo_away + ELO_K * ((1 - hr) - (1 - he))
    return new_h, new_a


def competition_type_code(ct):
    return {"LEAGUE": 0, "DOMESTIC_CUP": 1, "EUROPE": 2, "INTERNATIONAL": 3}.get(ct, -1)


def make_team_entry(m, team_name, is_home, hg, ag, comp_type):
    gf = hg if is_home else ag
    ga = ag if is_home else hg
    return {
        "date": m["date"],
        "date_dt": parse_date(m["date"]),
        "goals_for": gf, "goals_against": ga,
        "points": points_from_result(gf, ga),
        "win": int(gf > ga), "draw": int(gf == ga), "loss": int(gf < ga),
        "was_home": is_home,
        "competition_type": comp_type,
        "opponent": m["away"] if is_home else m["home"],
    }


def get_recent(hist, before_dt, limit, comp_type=None):
    out = []
    for item in reversed(hist):
        if item["date_dt"] >= before_dt:
            continue
        if comp_type and item["competition_type"] != comp_type:
            continue
        out.append(item)
        if len(out) == limit:
            break
    return out


def get_recent_home(hist, before_dt, limit):
    out = []
    for item in reversed(hist):
        if item["date_dt"] >= before_dt:
            continue
        if not item["was_home"]:
            continue
        out.append(item)
        if len(out) == limit:
            break
    return out


def get_recent_away(hist, before_dt, limit):
    out = []
    for item in reversed(hist):
        if item["date_dt"] >= before_dt:
            continue
        if item["was_home"]:
            continue
        out.append(item)
        if len(out) == limit:
            break
    return out


def get_h2h(home_hist, opponent, before_dt, limit):
    out = []
    for item in reversed(home_hist):
        if item["date_dt"] >= before_dt:
            continue
        if item["opponent"] != opponent:
            continue
        out.append(item)
        if len(out) == limit:
            break
    return out


def days_since(hist, before_dt):
    for item in reversed(hist):
        if item["date_dt"] < before_dt:
            return (before_dt - item["date_dt"]).days
    return 99


def matches_in_days(hist, before_dt, days=14):
    count = 0
    for item in reversed(hist):
        if item["date_dt"] >= before_dt:
            continue
        delta = (before_dt - item["date_dt"]).days
        if 0 < delta <= days:
            count += 1
        elif delta > days:
            break
    return count


def get_recent_sot(sot_hist, before_dt, limit=5):
    out = []
    for item in reversed(sot_hist):
        if item[0] >= before_dt:
            continue
        out.append(item)
        if len(out) == limit:
            break
    if not out:
        return 0.0, 0.0
    return (sum(x[1] for x in out) / len(out), sum(x[2] for x in out) / len(out))


# ── Dixon-Coles ────────────────────────────────────────────────────────────────

def dc_prob(dc, home_team, away_team, comp_type="LEAGUE", max_goals=8):
    attack   = dc["attack"]
    defense  = dc["defense"]
    rho      = dc["rho"]
    home_adv = dc["home_advantage"].get(comp_type, dc["home_advantage"].get("LEAGUE", 0.23))

    if home_team not in attack or away_team not in attack:
        return None

    lh = math.exp(attack[home_team] + defense[away_team] + home_adv)
    la = math.exp(attack[away_team] + defense[home_team])

    def poisson(lam, k):
        return math.exp(-lam) * (lam ** k) / math.factorial(k)

    def tau(x, y):
        if x == 0 and y == 0: return 1 - lh * la * rho
        if x == 0 and y == 1: return 1 + lh * rho
        if x == 1 and y == 0: return 1 + la * rho
        if x == 1 and y == 1: return 1 - rho
        return 1.0

    ph = pd_val = pa = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = poisson(lh, i) * poisson(la, j) * tau(i, j)
            if i > j:   ph     += p
            elif i == j: pd_val += p
            else:        pa     += p

    return ph, pd_val, pa, lh, la


# ── Marché ─────────────────────────────────────────────────────────────────────

def shin_normalize(odds_home, odds_draw, odds_away):
    raw = [1/odds_home, 1/odds_draw, 1/odds_away]
    total = sum(raw)
    return [r / total for r in raw]


def get_market_proba(conn, home_team, away_team, match_date):
    row = conn.execute("""
        SELECT AVG(odds_home) as oh, AVG(odds_draw) as od, AVG(odds_away) as oa, COUNT(*) as n
        FROM odds
        WHERE LOWER(home_team) = LOWER(?)
          AND LOWER(away_team) = LOWER(?)
          AND match_date = ?
          AND bookmaker NOT IN ('betfair_ex_eu','matchbook','betfair_ex_uk')
          AND odds_home > 1.0 AND odds_draw > 1.0 AND odds_away > 1.0
    """, (home_team, away_team, match_date)).fetchone()

    if row and row["n"] >= 2 and row["oh"] and row["od"] and row["oa"]:
        return shin_normalize(row["oh"], row["od"], row["oa"]), row["n"]
    return None, 0


# ── Poisson pour buts / Over / BTTS ───────────────────────────────────────────

def poisson_prob(lam, k):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def goals_to_markets(lh, la, max_goals=8):
    ph = pd_val = pa = 0.0
    over15 = over25 = over35 = btts = 0.0
    best_score, best_score_prob = (0, 0), 0.0

    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = poisson_prob(lh, i) * poisson_prob(la, j)
            if i > j:   ph     += p
            elif i == j: pd_val += p
            else:        pa     += p
            if i + j > 1.5: over15 += p
            if i + j > 2.5: over25 += p
            if i + j > 3.5: over35 += p
            if i > 0 and j > 0: btts += p
            if p > best_score_prob:
                best_score_prob = p
                best_score = (i, j)

    return {
        "ph": ph, "pd": pd_val, "pa": pa,
        "over_1_5": over15, "over_2_5": over25, "over_3_5": over35,
        "btts_yes": btts,
        "most_likely_score": f"{best_score[0]}-{best_score[1]}",
        "most_likely_score_prob": best_score_prob,
    }


# ── Features ───────────────────────────────────────────────────────────────────

def build_features(home_team, away_team, match_date_str, comp_type, id_league,
                   season, round_val, home_hist, away_hist, elo_state,
                   league_stats, season_draw_tracker, shots_hist_home, shots_hist_away):

    dt = parse_date(match_date_str)

    h_all = get_recent(home_hist, dt, LOOKBACK)
    a_all = get_recent(away_hist, dt, LOOKBACK)
    is_intl = comp_type == "INTERNATIONAL"
    h_lg = h_all if is_intl else get_recent(home_hist, dt, LOOKBACK, "LEAGUE")
    a_lg = a_all if is_intl else get_recent(away_hist, dt, LOOKBACK, "LEAGUE")
    h_home_spec = get_recent_home(home_hist, dt, LOOKBACK_VENUE)
    a_away_spec  = get_recent_away(away_hist,  dt, LOOKBACK_VENUE)
    h2h = get_h2h(home_hist, away_team, dt, H2H_LOOKBACK)

    if (len(h_all) < LOOKBACK or len(a_all) < LOOKBACK
            or (not is_intl and (len(h_lg) < MIN_LEAGUE_MATCHES or len(a_lg) < MIN_LEAGUE_MATCHES))
            or len(h_home_spec) < MIN_VENUE_MATCHES or len(a_away_spec) < MIN_VENUE_MATCHES):
        return None

    h_scored_all  = avg([x["goals_for"]    for x in h_all])
    h_conceded_all = avg([x["goals_against"] for x in h_all])
    a_scored_all  = avg([x["goals_for"]    for x in a_all])
    a_conceded_all = avg([x["goals_against"] for x in a_all])

    h_scored_lg  = avg([x["goals_for"]    for x in h_lg])
    h_conceded_lg = avg([x["goals_against"] for x in h_lg])
    a_scored_lg  = avg([x["goals_for"]    for x in a_lg])
    a_conceded_lg = avg([x["goals_against"] for x in a_lg])

    h_home_scored   = avg([x["goals_for"]    for x in h_home_spec])
    h_home_conceded = avg([x["goals_against"] for x in h_home_spec])
    h_home_points   = sum(x["points"] for x in h_home_spec)
    h_home_wins     = sum(x["win"]    for x in h_home_spec)

    a_away_scored   = avg([x["goals_for"]    for x in a_away_spec])
    a_away_conceded = avg([x["goals_against"] for x in a_away_spec])
    a_away_points   = sum(x["points"] for x in a_away_spec)
    a_away_wins     = sum(x["win"]    for x in a_away_spec)

    h2h_n    = len(h2h)
    h2h_wins = sum(x["win"]  for x in h2h)
    h2h_draws= sum(x["draw"] for x in h2h)
    h2h_loss = sum(x["loss"] for x in h2h)
    h2h_scored   = avg([x["goals_for"]    for x in h2h])
    h2h_conceded = avg([x["goals_against"] for x in h2h])
    h2h_points   = sum(x["points"] for x in h2h)

    h_last10 = get_recent(home_hist, dt, LOOKBACK_DRAW)
    a_last10 = get_recent(away_hist, dt, LOOKBACK_DRAW)
    h_home_last10 = get_recent_home(home_hist, dt, LOOKBACK_DRAW)
    a_away_last10 = get_recent_away(away_hist, dt, LOOKBACK_DRAW)

    home_draw_rate_last10      = sum(x["draw"] for x in h_last10) / len(h_last10) if h_last10 else 0.25
    away_draw_rate_last10      = sum(x["draw"] for x in a_last10) / len(a_last10) if a_last10 else 0.25
    home_draw_rate_home_last10 = sum(x["draw"] for x in h_home_last10) / len(h_home_last10) if h_home_last10 else 0.25
    away_draw_rate_away_last10 = sum(x["draw"] for x in a_away_last10) / len(a_away_last10) if a_away_last10 else 0.25
    home_clean_sheet_last10 = sum(1 for x in h_last10 if x["goals_against"] == 0) / len(h_last10) if h_last10 else 0.0
    away_clean_sheet_last10 = sum(1 for x in a_last10 if x["goals_against"] == 0) / len(a_last10) if a_last10 else 0.0
    home_no_score_last10    = sum(1 for x in h_last10 if x["goals_for"] == 0) / len(h_last10) if h_last10 else 0.0
    away_no_score_last10    = sum(1 for x in a_last10 if x["goals_for"] == 0) / len(a_last10) if a_last10 else 0.0
    h2h_draw_rate   = h2h_draws / h2h_n if h2h_n > 0 else 0.25
    draw_propensity = (home_draw_rate_last10 + away_draw_rate_last10) / 2

    home_scored_var   = var10([x["goals_for"]    for x in h_last10])
    away_scored_var   = var10([x["goals_for"]    for x in a_last10])
    home_conceded_var = var10([x["goals_against"] for x in h_last10])
    away_conceded_var = var10([x["goals_against"] for x in a_last10])

    h_rest = days_since(home_hist, dt)
    a_rest = days_since(away_hist, dt)

    elo_h  = elo_state["global"][home_team]
    elo_a  = elo_state["global"][away_team]
    elo_hh = elo_state["home"][home_team]
    elo_aa = elo_state["away"][away_team]

    ls = league_stats.get(id_league, {})
    league_draw_rate     = ls.get("draw_rate", 0.25)
    league_home_win_rate = ls.get("home_win_rate", 0.44)
    league_away_win_rate = ls.get("away_win_rate", 0.31)
    league_avg_goals     = ls.get("avg_goals", 2.6)

    season_key = (id_league, season)
    sd = season_draw_tracker.get(season_key, {"n": 0, "draws": 0})
    season_draw_rate = sd["draws"] / sd["n"] if sd["n"] >= 10 else league_draw_rate

    h_sot_for, h_sot_ag = get_recent_sot(shots_hist_home, dt)
    a_sot_for, a_sot_ag = get_recent_sot(shots_hist_away, dt)

    return {
        "idLeague": id_league,
        "season":   season,
        "round":    round_val if round_val is not None else 0,
        "competition_type": comp_type,
        "competition_type_code": competition_type_code(comp_type),

        "home_all_last5_scored_avg":   h_scored_all,
        "home_all_last5_conceded_avg": h_conceded_all,
        "home_all_last5_points": sum(x["points"] for x in h_all),
        "home_all_last5_wins":   sum(x["win"]    for x in h_all),
        "home_all_last5_draws":  sum(x["draw"]   for x in h_all),
        "home_all_last5_losses": sum(x["loss"]   for x in h_all),

        "away_all_last5_scored_avg":   a_scored_all,
        "away_all_last5_conceded_avg": a_conceded_all,
        "away_all_last5_points": sum(x["points"] for x in a_all),
        "away_all_last5_wins":   sum(x["win"]    for x in a_all),
        "away_all_last5_draws":  sum(x["draw"]   for x in a_all),
        "away_all_last5_losses": sum(x["loss"]   for x in a_all),

        "home_league_last5_scored_avg":   h_scored_lg,
        "home_league_last5_conceded_avg": h_conceded_lg,
        "home_league_last5_points": sum(x["points"] for x in h_lg),
        "home_league_last5_wins":   sum(x["win"]    for x in h_lg),
        "home_league_last5_draws":  sum(x["draw"]   for x in h_lg),
        "home_league_last5_losses": sum(x["loss"]   for x in h_lg),

        "away_league_last5_scored_avg":   a_scored_lg,
        "away_league_last5_conceded_avg": a_conceded_lg,
        "away_league_last5_points": sum(x["points"] for x in a_lg),
        "away_league_last5_wins":   sum(x["win"]    for x in a_lg),
        "away_league_last5_draws":  sum(x["draw"]   for x in a_lg),
        "away_league_last5_losses": sum(x["loss"]   for x in a_lg),

        "home_specific_home_scored_avg":   h_home_scored,
        "home_specific_home_conceded_avg": h_home_conceded,
        "home_specific_home_points": h_home_points,
        "home_specific_home_wins":   h_home_wins,

        "away_specific_away_scored_avg":   a_away_scored,
        "away_specific_away_conceded_avg": a_away_conceded,
        "away_specific_away_points": a_away_points,
        "away_specific_away_wins":   a_away_wins,

        "h2h_n":        h2h_n,
        "h2h_home_wins": h2h_wins,
        "h2h_draws":     h2h_draws,
        "h2h_away_wins": h2h_loss,
        "h2h_home_scored_avg":   h2h_scored,
        "h2h_home_conceded_avg": h2h_conceded,
        "h2h_home_points": h2h_points,
        "h2h_home_win_rate": h2h_wins / h2h_n if h2h_n > 0 else 0.5,

        "home_draw_rate_last10":      home_draw_rate_last10,
        "away_draw_rate_last10":      away_draw_rate_last10,
        "home_draw_rate_home_last10": home_draw_rate_home_last10,
        "away_draw_rate_away_last10": away_draw_rate_away_last10,
        "home_clean_sheet_last10":    home_clean_sheet_last10,
        "away_clean_sheet_last10":    away_clean_sheet_last10,
        "home_no_score_last10":       home_no_score_last10,
        "away_no_score_last10":       away_no_score_last10,
        "h2h_draw_rate":              h2h_draw_rate,
        "draw_propensity":            draw_propensity,

        "home_matches_last14d":       matches_in_days(home_hist, dt),
        "away_matches_last14d":       matches_in_days(away_hist, dt),
        "home_days_since_last_match": h_rest,
        "away_days_since_last_match": a_rest,

        "elo_home_global": elo_h,
        "elo_away_global": elo_a,
        "elo_diff_global": elo_h - elo_a,
        "elo_home_home":   elo_hh,
        "elo_away_away":   elo_aa,
        "elo_diff_home_away": elo_hh - elo_aa,

        "home_attack_vs_away_defense": h_home_scored   - a_away_conceded,
        "away_attack_vs_home_defense": a_away_scored   - h_home_conceded,

        "diff_all_points_last5":          sum(x["points"] for x in h_all) - sum(x["points"] for x in a_all),
        "diff_league_points_last5":       sum(x["points"] for x in h_lg)  - sum(x["points"] for x in a_lg),
        "diff_all_scored_avg_last5":      h_scored_all  - a_scored_all,
        "diff_all_conceded_avg_last5":    h_conceded_all - a_conceded_all,
        "diff_league_scored_avg_last5":   h_scored_lg   - a_scored_lg,
        "diff_league_conceded_avg_last5": h_conceded_lg  - a_conceded_lg,
        "diff_venue_scored":              h_home_scored  - a_away_scored,
        "diff_venue_conceded":            h_home_conceded - a_away_conceded,
        "diff_rest_days":                 h_rest - a_rest,

        "home_scored_var_last10":   round(home_scored_var,   4),
        "away_scored_var_last10":   round(away_scored_var,   4),
        "home_conceded_var_last10": round(home_conceded_var, 4),
        "away_conceded_var_last10": round(away_conceded_var, 4),

        "league_draw_rate":     round(league_draw_rate,     4),
        "league_home_win_rate": round(league_home_win_rate, 4),
        "league_away_win_rate": round(league_away_win_rate, 4),
        "league_avg_goals":     round(league_avg_goals,     4),

        "season_draw_rate": round(season_draw_rate, 4),

        "home_sot_last5":         round(h_sot_for, 3),
        "away_sot_last5":         round(a_sot_for, 3),
        "home_sot_against_last5": round(h_sot_ag,  3),
        "away_sot_against_last5": round(a_sot_ag,  3),
        "diff_sot_last5":         round(h_sot_for - a_sot_for, 3),
    }


# ── Modèles ML ─────────────────────────────────────────────────────────────────

def load_models(comp_type):
    safe = comp_type.replace(" ", "_")
    lgbm_bundle = hgbr_home = hgbr_away = None
    for path_tmpl, key in [(LGBM_PATH_TMPL, "lgbm"), (HGBR_HOME_TMPL, "home"), (HGBR_AWAY_TMPL, "away")]:
        try:
            b = joblib.load(path_tmpl.format(safe))
            if key == "lgbm":   lgbm_bundle = b
            elif key == "home": hgbr_home   = b
            else:               hgbr_away   = b
        except FileNotFoundError:
            pass
    return lgbm_bundle, hgbr_home, hgbr_away


def predict_lgbm(bundle, features_dict):
    model    = bundle["model"]
    feat_cols = bundle["features"]
    df = pd.DataFrame([features_dict])
    if "season" in df.columns:
        df["season"] = (df["season"].astype(str)
                        .str.extract(r"(\d{4})", expand=False)
                        .fillna("0").astype(int))
    df = pd.get_dummies(df, drop_first=False)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    for col in feat_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feat_cols]
    p = model.predict(df)[0]
    return p[0], p[1], p[2]


def predict_hgbr(bundle, features_dict):
    model    = bundle["model"]
    feat_cols = bundle["features"]
    df = pd.DataFrame([features_dict])
    if "season" in df.columns:
        df["season"] = (df["season"].astype(str)
                        .str.extract(r"(\d{4})", expand=False)
                        .fillna("0").astype(int))
    df = pd.get_dummies(df, drop_first=False)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    for col in feat_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feat_cols]
    return float(bundle["model"].predict(df)[0])


# ── Blend + calibration + value ────────────────────────────────────────────────

def blend(lgb_h, lgb_d, lgb_a, dc_h, dc_d, dc_a, mkt=None):
    if mkt is not None:
        mkt_h, mkt_d, mkt_a = mkt
        bh = W_LGB_MARKET * lgb_h + W_DC_MARKET * dc_h + W_MKT_MARKET * mkt_h
        bd = W_LGB_MARKET * lgb_d + W_DC_MARKET * dc_d + W_MKT_MARKET * mkt_d
        ba = W_LGB_MARKET * lgb_a + W_DC_MARKET * dc_a + W_MKT_MARKET * mkt_a
    else:
        bh = W_LGB_NO_MARKET * lgb_h + W_DC_NO_MARKET * dc_h
        bd = W_LGB_NO_MARKET * lgb_d + W_DC_NO_MARKET * dc_d
        ba = W_LGB_NO_MARKET * lgb_a + W_DC_NO_MARKET * dc_a
    total = bh + bd + ba
    return bh / total, bd / total, ba / total


def apply_calibration(calibrators, comp_name, comp_type, ph, pd_val, pa):
    calib = calibrators.get(comp_name) or calibrators.get(comp_type)
    if calib is None:
        return ph, pd_val, pa
    result = calib.predict_one([ph, pd_val, pa])
    return result[0], result[1], result[2]


def compute_value_edge(final_ph, final_pd, final_pa, mkt_proba):
    if mkt_proba is None:
        return None, None
    diffs = {"1": final_ph - mkt_proba[0], "X": final_pd - mkt_proba[1], "2": final_pa - mkt_proba[2]}
    best_pick = max(diffs, key=diffs.get)
    best_edge = diffs[best_pick]
    if best_edge >= 0.04:
        return best_pick, round(best_edge, 3)
    return None, None


# ── Migration table ─────────────────────────────────────────────────────────────

def ensure_schema(conn):
    conn.execute_script("""
    CREATE TABLE IF NOT EXISTS predictions_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prediction_run_date TEXT NOT NULL,
        match_date TEXT NOT NULL,
        competition_name TEXT,
        competition_type TEXT,
        home_team TEXT NOT NULL,
        away_team TEXT NOT NULL,
        status_prediction TEXT NOT NULL,
        top_pick TEXT, confidence REAL, trust_level TEXT,
        proba_home_win REAL, proba_draw REAL, proba_away_win REAL,
        pred_home_goals REAL, pred_away_goals REAL, pred_total_goals REAL,
        over_1_5 REAL, over_2_5 REAL, over_3_5 REAL, btts_yes REAL,
        most_likely_score TEXT, most_likely_score_prob REAL,
        value_pick TEXT, value_edge REAL,
        market_proba_home REAL, market_proba_draw REAL, market_proba_away REAL,
        n_bookmakers INTEGER, model_used TEXT,
        evaluation_status TEXT,
        real_home_goals INTEGER, real_away_goals INTEGER, real_result TEXT,
        real_total_goals INTEGER, real_btts INTEGER, real_over_2_5 INTEGER,
        is_correct_1x2 INTEGER, is_correct_score INTEGER,
        is_correct_btts INTEGER, is_correct_over_2_5 INTEGER,
        abs_error_home_goals REAL, abs_error_away_goals REAL, abs_error_total_goals REAL
    )
    """)
    conn.commit()

    # Migrations pour tables existantes
    for col, col_type in [
        ("market_proba_home", "REAL"),
        ("market_proba_draw", "REAL"),
        ("market_proba_away", "REAL"),
        ("n_bookmakers", "INTEGER"),
        ("model_used", "TEXT"),
    ]:
        if not conn.column_exists("predictions_history", col):
            conn.execute(f"ALTER TABLE predictions_history ADD COLUMN {col} {col_type}")
            conn.commit()

    if conn.is_pg:
        conn.execute("""
            SELECT setval('predictions_history_id_seq',
                COALESCE((SELECT MAX(id) FROM predictions_history), 0))
        """)
        conn.commit()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    target_date = date.today().isoformat()
    run_date    = datetime.now().isoformat(timespec="seconds")
    print(f"predict_v3 — {target_date}", flush=True)

    with open(DC_PARAMS_PATH) as f:
        dc = json.load(f)
    print(f"DC : {dc['n_teams']} équipes", flush=True)

    try:
        calibrators = joblib.load(CALIB_PATH)
        print(f"Calibrateurs : {len(calibrators)}", flush=True)
    except FileNotFoundError:
        calibrators = {}
        print("Calibrateurs absents", flush=True)

    conn = db_conn.get_connection()
    ensure_schema(conn)

    ph = ",".join("?" for _ in TARGET_COMPETITION_TYPES)
    today_matches = conn.execute(f"""
        WITH sdb_rounds AS (
            SELECT DISTINCT idLeague, season, round FROM matches WHERE source='thesportsdb'
        )
        SELECT m.id, m.idLeague, m.season, m.round, m.date, m.home, m.away,
               m.competition_type, m.competition_name, m.status
        FROM matches m
        WHERE DATE(m.date) = ?
          AND m.competition_type IN ({ph})
          AND m.status != 'FINISHED'
          AND (
              COALESCE(m.source,'thesportsdb') = 'thesportsdb'
              OR NOT EXISTS (
                  SELECT 1 FROM sdb_rounds s
                  WHERE s.idLeague=m.idLeague AND s.season=m.season
                    AND (s.round=m.round OR (s.round IS NULL AND m.round IS NULL))
              )
          )
    """, (target_date,) + TARGET_COMPETITION_TYPES).fetchall()

    print(f"{len(today_matches)} matchs à prédire", flush=True)
    if not today_matches:
        conn.close()
        return

    hist_matches = conn.execute(f"""
        WITH sdb_rounds AS (
            SELECT DISTINCT idLeague, season, round FROM matches WHERE source='thesportsdb'
        )
        SELECT m.id, m.idLeague, m.season, m.round, m.date, m.home, m.away,
               m.home_score, m.away_score, m.competition_type
        FROM matches m
        WHERE m.status = 'FINISHED'
          AND m.home_score IS NOT NULL AND m.away_score IS NOT NULL
          AND m.competition_type IN ({ph})
          AND DATE(m.date) < ?
          AND (
              COALESCE(m.source,'thesportsdb') = 'thesportsdb'
              OR NOT EXISTS (
                  SELECT 1 FROM sdb_rounds s
                  WHERE s.idLeague=m.idLeague AND s.season=m.season
                    AND (s.round=m.round OR (s.round IS NULL AND m.round IS NULL))
              )
          )
        ORDER BY m.date, m.id
    """, TARGET_COMPETITION_TYPES + (target_date,)).fetchall()

    shots_rows = conn.execute(
        "SELECT match_date, home_team, away_team, home_sot, away_sot FROM shots_data"
    ).fetchall()
    shots_index = {(r["match_date"][:10], r["home_team"], r["away_team"]): (r["home_sot"], r["away_sot"])
                   for r in shots_rows}

    league_stats_rows = conn.execute("""
        SELECT idLeague,
               COUNT(*) as n,
               SUM(CASE WHEN home_score=away_score THEN 1 ELSE 0 END) as draws,
               SUM(CASE WHEN home_score>away_score THEN 1 ELSE 0 END) as home_wins,
               SUM(CASE WHEN home_score<away_score THEN 1 ELSE 0 END) as away_wins,
               AVG(CAST(home_score AS REAL)+CAST(away_score AS REAL)) as avg_goals
        FROM matches
        WHERE status='FINISHED' AND home_score IS NOT NULL AND competition_type='LEAGUE'
        GROUP BY idLeague HAVING COUNT(*)>=50
    """).fetchall()
    league_stats = {r["idLeague"]: {
        "draw_rate":     r["draws"] / r["n"],
        "home_win_rate": r["home_wins"] / r["n"],
        "away_win_rate": r["away_wins"] / r["n"],
        "avg_goals":     r["avg_goals"] or 2.6,
    } for r in league_stats_rows}

    print(f"Historique : {len(hist_matches)} matchs", flush=True)

    team_histories      = {}
    shots_histories     = {}
    season_draw_tracker = {}
    elo_state = {
        "global": defaultdict(lambda: ELO_BASE),
        "home":   defaultdict(lambda: ELO_BASE),
        "away":   defaultdict(lambda: ELO_BASE),
    }

    for m in hist_matches:
        ht, at = m["home"], m["away"]
        hg, ag = int(m["home_score"]), int(m["away_score"])
        dt = parse_date(m["date"])

        team_histories.setdefault(ht, []).append(make_team_entry(m, ht, True,  hg, ag, m["competition_type"]))
        team_histories.setdefault(at, []).append(make_team_entry(m, at, False, hg, ag, m["competition_type"]))

        sk = (str(m["date"])[:10], ht, at)
        if sk in shots_index:
            hsot, asot = shots_index[sk]
            if hsot is not None:
                shots_histories.setdefault(ht, []).append((dt, hsot, asot or 0))
                shots_histories.setdefault(at, []).append((dt, asot or 0, hsot))

        season_key = (m["idLeague"], m["season"])
        if season_key not in season_draw_tracker:
            season_draw_tracker[season_key] = {"n": 0, "draws": 0}
        season_draw_tracker[season_key]["n"] += 1
        if hg == ag:
            season_draw_tracker[season_key]["draws"] += 1

        new_h, new_a = update_elo(elo_state["global"][ht], elo_state["global"][at], hg, ag)
        elo_state["global"][ht] = new_h
        elo_state["global"][at] = new_a
        new_hh, new_aa = update_elo(elo_state["home"][ht], elo_state["away"][at], hg, ag)
        elo_state["home"][ht] = new_hh
        elo_state["away"][at] = new_aa

    written = skipped = 0

    for match in today_matches:
        ht        = match["home"]
        at        = match["away"]
        comp_type = match["competition_type"]
        comp_name = match["competition_name"]
        match_date_str = str(match["date"])[:10]

        existing = conn.execute("""
            SELECT id FROM predictions_history
            WHERE prediction_run_date LIKE ? AND home_team=? AND away_team=? AND match_date=?
        """, (run_date[:10] + "%", ht, at, match_date_str)).fetchone()
        if existing:
            continue

        feats = build_features(
            ht, at, match_date_str, comp_type,
            match["idLeague"], match["season"], match["round"],
            team_histories.get(ht, []), team_histories.get(at, []),
            elo_state, league_stats, season_draw_tracker,
            shots_histories.get(ht, []), shots_histories.get(at, []),
        )

        if feats is None:
            print(f"  SKIP (historique): {ht} vs {at}", flush=True)
            skipped += 1
            conn.execute("""
                INSERT INTO predictions_history
                (prediction_run_date, match_date, competition_name, competition_type,
                 home_team, away_team, status_prediction, model_used)
                VALUES (?,?,?,?,?,?,'INSUFFICIENT_HISTORY','v3')
            """, (run_date, match_date_str, comp_name, comp_type, ht, at))
            conn.commit()
            continue

        lgbm_bundle, hgbr_home_bundle, hgbr_away_bundle = load_models(comp_type)
        if lgbm_bundle is None:
            lgbm_bundle, hgbr_home_bundle, hgbr_away_bundle = load_models("LEAGUE")
        if lgbm_bundle is None:
            print(f"  SKIP (pas de modèle): {ht} vs {at}", flush=True)
            skipped += 1
            continue

        lgb_h, lgb_d, lgb_a = predict_lgbm(lgbm_bundle, feats)

        dc_result = dc_prob(dc, ht, at, comp_type)
        if dc_result:
            dc_h, dc_d, dc_a, lh_dc, la_dc = dc_result
        else:
            dc_h, dc_d, dc_a = lgb_h, lgb_d, lgb_a
            lh_dc = la_dc = None

        if hgbr_home_bundle and hgbr_away_bundle:
            pred_hg = max(0.3, predict_hgbr(hgbr_home_bundle, feats))
            pred_ag = max(0.3, predict_hgbr(hgbr_away_bundle, feats))
        elif lh_dc:
            pred_hg, pred_ag = lh_dc, la_dc
        else:
            pred_hg = feats.get("home_all_last5_scored_avg", 1.3)
            pred_ag = feats.get("away_all_last5_scored_avg", 1.1)

        markets = goals_to_markets(pred_hg, pred_ag)

        mkt_proba, n_bookmakers = get_market_proba(conn, ht, at, match_date_str)

        final_ph, final_pd, final_pa = blend(lgb_h, lgb_d, lgb_a, dc_h, dc_d, dc_a, mkt_proba)
        final_ph, final_pd, final_pa = apply_calibration(
            calibrators, comp_name, comp_type, final_ph, final_pd, final_pa
        )

        probs = {"1": final_ph, "X": final_pd, "2": final_pa}
        top_pick   = max(probs, key=probs.get)
        confidence = probs[top_pick]
        trust_level = "FORTE" if confidence >= TRUST_FORTE else ("MOYENNE" if confidence >= TRUST_MOYENNE else "FAIBLE")

        value_pick, value_edge = compute_value_edge(final_ph, final_pd, final_pa, mkt_proba)
        model_tag = "v3_lgbm+dc" + ("+mkt" if mkt_proba else "")

        conn.execute("""
            INSERT INTO predictions_history (
                prediction_run_date, match_date, competition_name, competition_type,
                home_team, away_team, status_prediction,
                top_pick, confidence, trust_level,
                proba_home_win, proba_draw, proba_away_win,
                pred_home_goals, pred_away_goals, pred_total_goals,
                over_1_5, over_2_5, over_3_5, btts_yes,
                most_likely_score, most_likely_score_prob,
                value_pick, value_edge,
                market_proba_home, market_proba_draw, market_proba_away, n_bookmakers,
                model_used
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            run_date, match_date_str, comp_name, comp_type,
            ht, at, "PREDICTED",
            top_pick, round(confidence, 4), trust_level,
            round(final_ph, 4), round(final_pd, 4), round(final_pa, 4),
            round(pred_hg, 3), round(pred_ag, 3), round(pred_hg + pred_ag, 3),
            round(markets["over_1_5"], 4), round(markets["over_2_5"], 4), round(markets["over_3_5"], 4),
            round(markets["btts_yes"], 4),
            markets["most_likely_score"], round(markets["most_likely_score_prob"], 4),
            value_pick, value_edge,
            round(mkt_proba[0], 4) if mkt_proba else None,
            round(mkt_proba[1], 4) if mkt_proba else None,
            round(mkt_proba[2], 4) if mkt_proba else None,
            n_bookmakers, model_tag,
        ))
        conn.commit()
        written += 1

        mkt_str = (f" | mkt({n_bookmakers}bk): {mkt_proba[0]:.2f}/{mkt_proba[1]:.2f}/{mkt_proba[2]:.2f}"
                   if mkt_proba else "")
        vb_str  = f" ★ VALUE {value_pick}+{value_edge:.3f}" if value_pick else ""
        print(f"  [{trust_level}] {ht} vs {at} → {top_pick} ({confidence:.2f}) "
              f"| {final_ph:.2f}/{final_pd:.2f}/{final_pa:.2f}{mkt_str}{vb_str}", flush=True)

    conn.close()
    print(f"\nTerminé : {written} prédictions, {skipped} skipped", flush=True)


if __name__ == "__main__":
    main()
