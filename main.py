from __future__ import annotations

import math
import sqlite3
from collections import defaultdict
from datetime import datetime
from statistics import mean
from typing import Any

import joblib
import pandas as pd
import requests as http_requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import db_conn

DC_PARAMS_PATH     = "dixon_coles_params.json"
CALIBRATORS_PATH   = "calibrators.pkl"
ML_GLOBAL_HOME     = "model_home_goals_v3.pkl"
ML_GLOBAL_AWAY     = "model_away_goals_v3.pkl"

LOOKBACK = 5
LOOKBACK_VENUE = 5
LOOKBACK_DRAW = 10
H2H_LOOKBACK = 5
MIN_LEAGUE_MATCHES = 3
TARGET_COMPETITION_TYPES = ("LEAGUE", "DOMESTIC_CUP", "EUROPE", "INTERNATIONAL")

ELO_BASE      = 1500.0
ELO_K         = 20.0
HOME_ELO_BOOST = 50.0

MIN_CONFIDENCE    = 0.45
STRONG_CONFIDENCE = 0.55

app = FastAPI(title="Pronostics API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




class SummaryResponse(BaseModel):
    total_predictions: int
    evaluated_predictions: int
    accuracy_1x2: float | None
    accuracy_btts: float | None
    accuracy_over_1_5: float | None
    accuracy_over_2_5: float | None
    mae_total_goals: float | None


class ActionResponse(BaseModel):
    ok: bool
    message: str


import json
_cached_ai: dict[str, Any] | None = None


def get_ai() -> dict[str, Any]:
    """Charge (une seule fois) DC params, calibrateurs et modèles ML."""
    global _cached_ai
    if _cached_ai is not None:
        return _cached_ai

    ai: dict[str, Any] = {"dc": None, "calibrators": None, "ml": {}}

    # Dixon-Coles
    try:
        with open(DC_PARAMS_PATH, encoding="utf-8") as f:
            ai["dc"] = json.load(f)
    except FileNotFoundError:
        pass

    # Calibrateurs
    try:
        ai["calibrators"] = joblib.load(CALIBRATORS_PATH)
    except FileNotFoundError:
        pass

    # Modèles ML par compétition + global
    for ct in ["LEAGUE", "EUROPE", "INTERNATIONAL"]:
        try:
            ai["ml"][ct] = {
                "home": joblib.load(f"model_home_goals_{ct}.pkl"),
                "away": joblib.load(f"model_away_goals_{ct}.pkl"),
            }
        except FileNotFoundError:
            pass
    try:
        ai["ml"]["_global"] = {
            "home": joblib.load(ML_GLOBAL_HOME),
            "away": joblib.load(ML_GLOBAL_AWAY),
        }
    except FileNotFoundError:
        pass

    _cached_ai = ai
    return ai


def get_conn() -> db_conn.Connection:
    return db_conn.get_connection()


def parse_date(d: str) -> datetime:
    d = str(d)
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(d, fmt)
        except ValueError:
            pass
    raise ValueError(f"Format de date non géré : {d}")


def avg(values: list[float]) -> float:
    return mean(values) if values else 0.0


def points_from_result(gf: int, ga: int) -> int:
    if gf > ga:
        return 3
    if gf == ga:
        return 1
    return 0


def result_score(gf: int, ga: int) -> float:
    if gf > ga:
        return 1.0
    if gf == ga:
        return 0.5
    return 0.0


def expected_score(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def update_elo(elo_home: float, elo_away: float, home_goals: int, away_goals: int) -> tuple[float, float]:
    home_exp = expected_score(elo_home + HOME_ELO_BOOST, elo_away)
    away_exp = 1.0 - home_exp

    home_real = result_score(home_goals, away_goals)
    away_real = 1.0 - home_real

    new_home = elo_home + ELO_K * (home_real - home_exp)
    new_away = elo_away + ELO_K * (away_real - away_exp)
    return new_home, new_away


def competition_type_code(comp_type: str) -> int:
    mapping = {"LEAGUE": 0, "DOMESTIC_CUP": 1, "EUROPE": 2, "INTERNATIONAL": 3}
    return mapping.get(comp_type, -1)



def get_finished_matches(conn: sqlite3.Connection):
    placeholders = ",".join("?" for _ in TARGET_COMPETITION_TYPES)
    return conn.execute(
        f"""
        SELECT
            id, idLeague, season, round, date, home, away,
            home_score, away_score, status, competition_type, competition_name
        FROM matches
        WHERE status = 'FINISHED'
          AND home_score IS NOT NULL
          AND away_score IS NOT NULL
          AND competition_type IN ({placeholders})
        ORDER BY date, round, id
        """,
        TARGET_COMPETITION_TYPES,
    ).fetchall()


def get_today_matches(conn: sqlite3.Connection):
    today = datetime.now().strftime("%Y-%m-%d")
    placeholders = ",".join("?" for _ in TARGET_COMPETITION_TYPES)

    return conn.execute(
        f"""
        SELECT
            m.id,
            m.idLeague,
            m.season,
            m.round,
            m.date,
            m.home,
            m.away,
            m.status,
            m.competition_type,
            m.competition_name,

            COALESCE(
                -- 1) même nom + même compétition
                (
                    SELECT t.badge_url
                    FROM teams t
                    WHERE LOWER(TRIM(t.strTeam)) = LOWER(TRIM(m.home))
                      AND t.badge_url IS NOT NULL
                      AND TRIM(t.badge_url) <> ''
                      AND t.strLeague = m.competition_name
                    LIMIT 1
                ),
                -- 2) même nom + même type de compétition
                (
                    SELECT t.badge_url
                    FROM teams t
                    WHERE LOWER(TRIM(t.strTeam)) = LOWER(TRIM(m.home))
                      AND t.badge_url IS NOT NULL
                      AND TRIM(t.badge_url) <> ''
                      AND t.competition_type = m.competition_type
                    LIMIT 1
                ),
                -- 3) même nom + équipe nationale
                (
                    SELECT t.badge_url
                    FROM teams t
                    WHERE LOWER(TRIM(t.strTeam)) = LOWER(TRIM(m.home))
                      AND t.badge_url IS NOT NULL
                      AND TRIM(t.badge_url) <> ''
                      AND t.team_type = 'NATIONAL'
                    LIMIT 1
                ),
                -- 4) même nom tout court
                (
                    SELECT t.badge_url
                    FROM teams t
                    WHERE LOWER(TRIM(t.strTeam)) = LOWER(TRIM(m.home))
                      AND t.badge_url IS NOT NULL
                      AND TRIM(t.badge_url) <> ''
                    LIMIT 1
                )
            ) AS home_badge,

            COALESCE(
                -- 1) même nom + même compétition
                (
                    SELECT t.badge_url
                    FROM teams t
                    WHERE LOWER(TRIM(t.strTeam)) = LOWER(TRIM(m.away))
                      AND t.badge_url IS NOT NULL
                      AND TRIM(t.badge_url) <> ''
                      AND t.strLeague = m.competition_name
                    LIMIT 1
                ),
                -- 2) même nom + même type de compétition
                (
                    SELECT t.badge_url
                    FROM teams t
                    WHERE LOWER(TRIM(t.strTeam)) = LOWER(TRIM(m.away))
                      AND t.badge_url IS NOT NULL
                      AND TRIM(t.badge_url) <> ''
                      AND t.competition_type = m.competition_type
                    LIMIT 1
                ),
                -- 3) même nom + équipe nationale
                (
                    SELECT t.badge_url
                    FROM teams t
                    WHERE LOWER(TRIM(t.strTeam)) = LOWER(TRIM(m.away))
                      AND t.badge_url IS NOT NULL
                      AND TRIM(t.badge_url) <> ''
                      AND t.team_type = 'NATIONAL'
                    LIMIT 1
                ),
                -- 4) même nom tout court
                (
                    SELECT t.badge_url
                    FROM teams t
                    WHERE LOWER(TRIM(t.strTeam)) = LOWER(TRIM(m.away))
                      AND t.badge_url IS NOT NULL
                      AND TRIM(t.badge_url) <> ''
                    LIMIT 1
                )
            ) AS away_badge

        FROM matches m
        WHERE substr(m.date, 1, 10) = ?
          AND m.competition_type IN ({placeholders})
          AND (m.status IS NULL OR m.status IN ('SCHEDULED', 'TIMED', 'NOT_STARTED', 'NS'))
        ORDER BY m.date, m.id
        """,
        (today, *TARGET_COMPETITION_TYPES),
    ).fetchall()
def build_histories_and_elo(conn: sqlite3.Connection):
    matches = get_finished_matches(conn)
    team_histories = {}
    elo_state = {
        "global": defaultdict(lambda: ELO_BASE),
        "home": defaultdict(lambda: ELO_BASE),
        "away": defaultdict(lambda: ELO_BASE),
    }

    for match in matches:
        home_team = match["home"]
        away_team = match["away"]
        match_dt = parse_date(match["date"])
        hg = int(match["home_score"])
        ag = int(match["away_score"])

        home_entry = {
            "date": match["date"],
            "date_dt": match_dt,
            "goals_for": hg,
            "goals_against": ag,
            "points": points_from_result(hg, ag),
            "win": 1 if hg > ag else 0,
            "draw": 1 if hg == ag else 0,
            "loss": 1 if hg < ag else 0,
            "was_home": True,
            "competition_type": match["competition_type"],
            "opponent": away_team,
        }
        away_entry = {
            "date": match["date"],
            "date_dt": match_dt,
            "goals_for": ag,
            "goals_against": hg,
            "points": points_from_result(ag, hg),
            "win": 1 if ag > hg else 0,
            "draw": 1 if ag == hg else 0,
            "loss": 1 if ag < hg else 0,
            "was_home": False,
            "competition_type": match["competition_type"],
            "opponent": home_team,
        }

        team_histories.setdefault(home_team, []).append(home_entry)
        team_histories.setdefault(away_team, []).append(away_entry)

        new_home_elo, new_away_elo = update_elo(
            elo_state["global"][home_team],
            elo_state["global"][away_team],
            hg,
            ag,
        )
        elo_state["global"][home_team] = new_home_elo
        elo_state["global"][away_team] = new_away_elo

        new_home_home_elo, _ = update_elo(
            elo_state["home"][home_team],
            elo_state["away"][away_team],
            hg,
            ag,
        )
        _, new_away_away_elo = update_elo(
            elo_state["home"][home_team],
            elo_state["away"][away_team],
            hg,
            ag,
        )
        elo_state["home"][home_team] = new_home_home_elo
        elo_state["away"][away_team] = new_away_away_elo

    return team_histories, elo_state


def get_recent_history(history_list: list[dict[str, Any]], before_dt: datetime, limit: int, competition_type: str | None = None):
    out = []
    for item in reversed(history_list):
        if item["date_dt"] >= before_dt:
            continue
        if competition_type is not None and item["competition_type"] != competition_type:
            continue
        out.append(item)
        if len(out) == limit:
            break
    return out


def matches_in_last_days(history_list: list[dict[str, Any]], before_dt: datetime, days: int = 14) -> int:
    count = 0
    for item in reversed(history_list):
        if item["date_dt"] >= before_dt:
            continue
        delta = (before_dt - item["date_dt"]).days
        if 0 < delta <= days:
            count += 1
        elif delta > days:
            break
    return count


def days_since_last_match(history_list: list[dict[str, Any]], before_dt: datetime) -> int:
    for item in reversed(history_list):
        if item["date_dt"] < before_dt:
            return (before_dt - item["date_dt"]).days
    return 99


def get_recent_home_venue(history_list: list[dict[str, Any]], before_dt: datetime, limit: int) -> list:
    out = []
    for item in reversed(history_list):
        if item["date_dt"] >= before_dt:
            continue
        if not item["was_home"]:
            continue
        out.append(item)
        if len(out) == limit:
            break
    return out


def get_recent_away_venue(history_list: list[dict[str, Any]], before_dt: datetime, limit: int) -> list:
    out = []
    for item in reversed(history_list):
        if item["date_dt"] >= before_dt:
            continue
        if item["was_home"]:
            continue
        out.append(item)
        if len(out) == limit:
            break
    return out


def get_h2h(history_list: list[dict[str, Any]], opponent: str, before_dt: datetime, limit: int) -> list:
    out = []
    for item in reversed(history_list):
        if item["date_dt"] >= before_dt:
            continue
        if item.get("opponent") != opponent:
            continue
        out.append(item)
        if len(out) == limit:
            break
    return out


def build_features_for_future_match(match, team_histories, elo_state):
    match_dt = parse_date(match["date"])
    home_team = match["home"]
    away_team = match["away"]

    home_hist = team_histories.get(home_team, [])
    away_hist = team_histories.get(away_team, [])

    home_all = get_recent_history(home_hist, match_dt, LOOKBACK)
    away_all = get_recent_history(away_hist, match_dt, LOOKBACK)
    if match["competition_type"] == "INTERNATIONAL":
        home_league = home_all
        away_league = away_all
    else:
        home_league = get_recent_history(home_hist, match_dt, LOOKBACK, "LEAGUE")
        away_league = get_recent_history(away_hist, match_dt, LOOKBACK, "LEAGUE")

    if len(home_all) < LOOKBACK or len(away_all) < LOOKBACK:
        return None
    if match["competition_type"] != "INTERNATIONAL" and (
        len(home_league) < MIN_LEAGUE_MATCHES or len(away_league) < MIN_LEAGUE_MATCHES
    ):
        return None

    # Venue-specific histories (independent from general last-5)
    home_spec = get_recent_home_venue(home_hist, match_dt, LOOKBACK_VENUE)
    away_spec = get_recent_away_venue(away_hist, match_dt, LOOKBACK_VENUE)

    # H2H
    h2h = get_h2h(home_hist, away_team, match_dt, H2H_LOOKBACK)

    # General form stats
    home_all_scored_avg   = avg([x["goals_for"]    for x in home_all])
    home_all_conceded_avg = avg([x["goals_against"] for x in home_all])
    away_all_scored_avg   = avg([x["goals_for"]    for x in away_all])
    away_all_conceded_avg = avg([x["goals_against"] for x in away_all])
    home_points_all = sum(x["points"] for x in home_all)
    away_points_all = sum(x["points"] for x in away_all)

    # League form stats
    home_league_scored_avg   = avg([x["goals_for"]    for x in home_league])
    home_league_conceded_avg = avg([x["goals_against"] for x in home_league])
    away_league_scored_avg   = avg([x["goals_for"]    for x in away_league])
    away_league_conceded_avg = avg([x["goals_against"] for x in away_league])
    home_points_league = sum(x["points"] for x in home_league)
    away_points_league = sum(x["points"] for x in away_league)

    # Venue-specific stats (last 5 home / last 5 away — independent lookups)
    h_home_scored   = avg([x["goals_for"]    for x in home_spec])
    h_home_conceded = avg([x["goals_against"] for x in home_spec])
    h_home_points   = sum(x["points"] for x in home_spec)
    h_home_wins     = sum(x["win"]    for x in home_spec)

    a_away_scored   = avg([x["goals_for"]    for x in away_spec])
    a_away_conceded = avg([x["goals_against"] for x in away_spec])
    a_away_points   = sum(x["points"] for x in away_spec)
    a_away_wins     = sum(x["win"]    for x in away_spec)

    # H2H stats
    h2h_n     = len(h2h)
    h2h_wins  = sum(x["win"]  for x in h2h)
    h2h_draws = sum(x["draw"] for x in h2h)
    h2h_loss  = sum(x["loss"] for x in h2h)
    h2h_scored   = avg([x["goals_for"]    for x in h2h])
    h2h_conceded = avg([x["goals_against"] for x in h2h])
    h2h_points   = sum(x["points"] for x in h2h)

    # Rest days
    home_rest_days = days_since_last_match(home_hist, match_dt)
    away_rest_days = days_since_last_match(away_hist, match_dt)

    # Draw-specific features (window of 10)
    h_last10      = get_recent_history(home_hist, match_dt, LOOKBACK_DRAW)
    a_last10      = get_recent_history(away_hist, match_dt, LOOKBACK_DRAW)
    h_home_last10 = get_recent_home_venue(home_hist, match_dt, LOOKBACK_DRAW)
    a_away_last10 = get_recent_away_venue(away_hist, match_dt, LOOKBACK_DRAW)

    home_draw_rate_last10      = sum(x["draw"] for x in h_last10) / len(h_last10) if h_last10 else 0.25
    away_draw_rate_last10      = sum(x["draw"] for x in a_last10) / len(a_last10) if a_last10 else 0.25
    home_draw_rate_home_last10 = sum(x["draw"] for x in h_home_last10) / len(h_home_last10) if h_home_last10 else 0.25
    away_draw_rate_away_last10 = sum(x["draw"] for x in a_away_last10) / len(a_away_last10) if a_away_last10 else 0.25
    home_clean_sheet_last10    = sum(1 for x in h_last10 if x["goals_against"] == 0) / len(h_last10) if h_last10 else 0.0
    away_clean_sheet_last10    = sum(1 for x in a_last10 if x["goals_against"] == 0) / len(a_last10) if a_last10 else 0.0
    home_no_score_last10       = sum(1 for x in h_last10 if x["goals_for"] == 0) / len(h_last10) if h_last10 else 0.0
    away_no_score_last10       = sum(1 for x in a_last10 if x["goals_for"] == 0) / len(a_last10) if a_last10 else 0.0
    h2h_draw_rate              = sum(x["draw"] for x in h2h) / h2h_n if h2h_n > 0 else 0.25
    draw_propensity            = (home_draw_rate_last10 + away_draw_rate_last10) / 2

    return {
        "match_id": match["id"],
        "idLeague": match["idLeague"],
        "season": match["season"],
        "round": match["round"] if match["round"] is not None else 0,
        "competition_name": match["competition_name"],
        "competition_type": match["competition_type"],
        "competition_type_code": competition_type_code(match["competition_type"]),
        "date": match["date"],
        "home_team": home_team,
        "away_team": away_team,

        # General form
        "home_all_last5_scored_avg":   home_all_scored_avg,
        "home_all_last5_conceded_avg": home_all_conceded_avg,
        "home_all_last5_points": home_points_all,
        "home_all_last5_wins":   sum(x["win"]  for x in home_all),
        "home_all_last5_draws":  sum(x["draw"] for x in home_all),
        "home_all_last5_losses": sum(x["loss"] for x in home_all),

        "away_all_last5_scored_avg":   away_all_scored_avg,
        "away_all_last5_conceded_avg": away_all_conceded_avg,
        "away_all_last5_points": away_points_all,
        "away_all_last5_wins":   sum(x["win"]  for x in away_all),
        "away_all_last5_draws":  sum(x["draw"] for x in away_all),
        "away_all_last5_losses": sum(x["loss"] for x in away_all),

        # League form
        "home_league_last5_scored_avg":   home_league_scored_avg,
        "home_league_last5_conceded_avg": home_league_conceded_avg,
        "home_league_last5_points": home_points_league,
        "home_league_last5_wins":   sum(x["win"]  for x in home_league),
        "home_league_last5_draws":  sum(x["draw"] for x in home_league),
        "home_league_last5_losses": sum(x["loss"] for x in home_league),

        "away_league_last5_scored_avg":   away_league_scored_avg,
        "away_league_last5_conceded_avg": away_league_conceded_avg,
        "away_league_last5_points": away_points_league,
        "away_league_last5_wins":   sum(x["win"]  for x in away_league),
        "away_league_last5_draws":  sum(x["draw"] for x in away_league),
        "away_league_last5_losses": sum(x["loss"] for x in away_league),

        # Venue-specific form (naming matches training dataset)
        "home_specific_home_scored_avg":   h_home_scored,
        "home_specific_home_conceded_avg": h_home_conceded,
        "home_specific_home_points": h_home_points,
        "home_specific_home_wins":   h_home_wins,

        "away_specific_away_scored_avg":   a_away_scored,
        "away_specific_away_conceded_avg": a_away_conceded,
        "away_specific_away_points": a_away_points,
        "away_specific_away_wins":   a_away_wins,

        # H2H
        "h2h_n":             h2h_n,
        "h2h_home_wins":     h2h_wins,
        "h2h_draws":         h2h_draws,
        "h2h_away_wins":     h2h_loss,
        "h2h_home_scored_avg":   h2h_scored,
        "h2h_home_conceded_avg": h2h_conceded,
        "h2h_home_points":   h2h_points,
        "h2h_home_win_rate": h2h_wins / h2h_n if h2h_n > 0 else 0.5,

        # Rest
        "home_matches_last14d": matches_in_last_days(home_hist, match_dt, 14),
        "away_matches_last14d": matches_in_last_days(away_hist, match_dt, 14),
        "home_days_since_last_match": home_rest_days,
        "away_days_since_last_match": away_rest_days,

        # ELO
        "elo_home_global": elo_state["global"][home_team],
        "elo_away_global": elo_state["global"][away_team],
        "elo_diff_global": elo_state["global"][home_team] - elo_state["global"][away_team],
        "elo_home_home":   elo_state["home"][home_team],
        "elo_away_away":   elo_state["away"][away_team],
        "elo_diff_home_away": elo_state["home"][home_team] - elo_state["away"][away_team],

        # Attack vs defense
        "home_attack_vs_away_defense": h_home_scored - a_away_conceded,
        "away_attack_vs_home_defense": a_away_scored - h_home_conceded,

        # Diffs
        "diff_all_points_last5":          home_points_all   - away_points_all,
        "diff_league_points_last5":       home_points_league - away_points_league,
        "diff_all_scored_avg_last5":      home_all_scored_avg   - away_all_scored_avg,
        "diff_all_conceded_avg_last5":    home_all_conceded_avg - away_all_conceded_avg,
        "diff_league_scored_avg_last5":   home_league_scored_avg   - away_league_scored_avg,
        "diff_league_conceded_avg_last5": home_league_conceded_avg - away_league_conceded_avg,
        "diff_venue_scored":   h_home_scored   - a_away_scored,
        "diff_venue_conceded": h_home_conceded - a_away_conceded,
        "diff_rest_days": home_rest_days - away_rest_days,

        # Draw features (v6)
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
    }


def prepare_single_row(features_dict: dict[str, Any], model_features: list[str]) -> pd.DataFrame:
    df = pd.DataFrame([features_dict]).copy()
    if "season" in df.columns:
        df["season"] = (
            df["season"].astype(str).str.extract(r"(\d{4})", expand=False).fillna("0").astype(int)
        )
    df = pd.get_dummies(df, drop_first=False)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    for col in model_features:
        if col not in df.columns:
            df[col] = 0
    return df[model_features]


def poisson_prob(lmbda: float, k: int) -> float:
    return math.exp(-lmbda) * (lmbda ** k) / math.factorial(k)


def score_matrix(home_lambda: float, away_lambda: float, max_goals: int = 8):
    matrix = []
    for hg in range(max_goals + 1):
        row = []
        for ag in range(max_goals + 1):
            row.append(poisson_prob(home_lambda, hg) * poisson_prob(away_lambda, ag))
        matrix.append(row)
    return matrix


def score_outcome(hg: int, ag: int) -> str:
    return "1" if hg > ag else ("X" if hg == ag else "2")


def dc_tau(hg: int, ag: int, lam_h: float, lam_a: float, rho: float) -> float:
    if hg == 0 and ag == 0:
        return 1 - lam_h * lam_a * rho
    if hg == 1 and ag == 0:
        return 1 + lam_a * rho
    if hg == 0 and ag == 1:
        return 1 + lam_h * rho
    if hg == 1 and ag == 1:
        return 1 - rho
    return 1.0


def apply_calibration(summary: dict[str, Any], calibrators: Any) -> dict[str, Any]:
    if calibrators is None:
        return summary
    p1 = float(calibrators["home_win"].predict([summary["proba_home_win"]])[0])
    px = float(calibrators["draw"].predict([summary["proba_draw"]])[0])
    p2 = float(calibrators["away_win"].predict([summary["proba_away_win"]])[0])
    total = p1 + px + p2
    if total > 0:
        p1, px, p2 = p1 / total, px / total, p2 / total
    summary = dict(summary)
    summary["proba_home_win"] = p1
    summary["proba_draw"]     = px
    summary["proba_away_win"] = p2
    if calibrators.get("over_2_5") and summary.get("over_2_5") is not None:
        summary["over_2_5"] = float(calibrators["over_2_5"].predict([summary["over_2_5"]])[0])
    if calibrators.get("btts") and summary.get("btts_yes") is not None:
        summary["btts_yes"] = float(calibrators["btts"].predict([summary["btts_yes"]])[0])
    probs = {"1": p1, "X": px, "2": p2}
    summary["confidence"]  = max(probs.values())
    summary["top_pick"]    = max(probs, key=probs.get)
    c = summary["confidence"]
    summary["trust_level"] = ("FORTE" if c >= STRONG_CONFIDENCE
                               else "MOYENNE" if c >= MIN_CONFIDENCE else "FAIBLE")
    return summary


def derive_markets_from_poisson(home_lambda: float, away_lambda: float, rho: float = 0.0) -> dict[str, Any]:
    home_lambda = max(0.05, float(home_lambda))
    away_lambda = max(0.05, float(away_lambda))
    matrix = score_matrix(home_lambda, away_lambda, max_goals=8)

    p_home = p_draw = p_away = 0.0
    p_over_15 = p_over_25 = p_over_35 = 0.0
    p_btts = 0.0
    all_scores: list[tuple[int, int, float, str]] = []

    for hg, row in enumerate(matrix):
        for ag, p_raw in enumerate(row):
            p = p_raw * max(0.0, dc_tau(hg, ag, home_lambda, away_lambda, rho))
            if hg > ag:
                p_home += p
            elif hg == ag:
                p_draw += p
            else:
                p_away += p

            total = hg + ag
            if total >= 2:
                p_over_15 += p
            if total >= 3:
                p_over_25 += p
            if total >= 4:
                p_over_35 += p
            if hg >= 1 and ag >= 1:
                p_btts += p

            all_scores.append((hg, ag, p, score_outcome(hg, ag)))

    # Trier par probabilité décroissante
    all_scores.sort(key=lambda x: -x[2])

    best_hg, best_ag, best_p, _ = all_scores[0]

    probs = {"1": p_home, "X": p_draw, "2": p_away}
    confidence = max(probs.values())
    top_pick = max(probs, key=probs.get)

    if confidence >= STRONG_CONFIDENCE:
        trust_level = "FORTE"
    elif confidence >= MIN_CONFIDENCE:
        trust_level = "MOYENNE"
    else:
        trust_level = "FAIBLE"

    # Score le plus probable cohérent avec top_pick
    best_for_pick = next((s for s in all_scores if s[3] == top_pick), None)
    most_likely_score_for_pick      = f"{best_for_pick[0]}-{best_for_pick[1]}" if best_for_pick else None
    most_likely_score_for_pick_prob = best_for_pick[2] if best_for_pick else None

    # Cohérence : le score global confirme-t-il top_pick ?
    score_coherent = score_outcome(best_hg, best_ag) == top_pick

    # Top 3 scores individuels
    top3_scores = [
        {"score": f"{hg}-{ag}", "prob": round(p, 4)}
        for hg, ag, p, _ in all_scores[:3]
    ]

    return {
        "top_pick": top_pick,
        "confidence": confidence,
        "trust_level": trust_level,
        "proba_home_win": p_home,
        "proba_draw": p_draw,
        "proba_away_win": p_away,
        "pred_home_goals": home_lambda,
        "pred_away_goals": away_lambda,
        "pred_total_goals": home_lambda + away_lambda,
        "over_1_5": p_over_15,
        "over_2_5": p_over_25,
        "over_3_5": p_over_35,
        "btts_yes": p_btts,
        "most_likely_score": f"{best_hg}-{best_ag}",
        "most_likely_score_prob": best_p,
        "most_likely_score_for_pick": most_likely_score_for_pick,
        "most_likely_score_for_pick_prob": most_likely_score_for_pick_prob,
        "score_coherent": score_coherent,
        "top3_scores": top3_scores,
    }


@app.get("/")
def root():
    return {"ok": True, "message": "Pronostics API prête."}


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/teams")
def list_teams():
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT DISTINCT strTeam FROM teams WHERE strTeam IS NOT NULL ORDER BY strTeam"
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


@app.get("/competitions")
def list_competitions():
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT DISTINCT name FROM competitions WHERE name IS NOT NULL ORDER BY name"
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


@app.get("/stats/summary", response_model=SummaryResponse)
def stats_summary():
    conn = get_conn()
    try:
        total_predictions = conn.execute("SELECT COUNT(*) AS n FROM predictions_history").fetchone()["n"]
        evaluated_predictions = conn.execute(
            "SELECT COUNT(*) AS n FROM predictions_history WHERE evaluation_status = 'OK'"
        ).fetchone()["n"]

        row = conn.execute(
            """
            SELECT
                AVG(is_correct_1x2)        AS accuracy_1x2,
                AVG(is_correct_btts)       AS accuracy_btts,
                AVG(is_correct_over_2_5)   AS accuracy_over_2_5,
                AVG(abs_error_total_goals) AS mae_total_goals
            FROM predictions_history
            WHERE evaluation_status = 'OK'
            """
        ).fetchone()

        return SummaryResponse(
            total_predictions=total_predictions,
            evaluated_predictions=evaluated_predictions,
            accuracy_1x2=row["accuracy_1x2"],
            accuracy_btts=row["accuracy_btts"],
            accuracy_over_1_5=None,  # colonne absente de la table
            accuracy_over_2_5=row["accuracy_over_2_5"],
            mae_total_goals=row["mae_total_goals"],
        )
    finally:
        conn.close()


@app.get("/predictions/history")
def predictions_history(limit: int = 100, offset: int = 0, only_evaluated: bool = True):
    conn = get_conn()
    try:
        where = "WHERE ph.evaluation_status = 'OK'" if only_evaluated else ""
        rows = conn.execute(
            f"""
            SELECT
                ph.*,
                COALESCE(
                    (SELECT t.badge_url FROM teams t
                     WHERE LOWER(TRIM(t.strTeam)) = LOWER(TRIM(ph.home_team))
                       AND t.badge_url IS NOT NULL AND TRIM(t.badge_url) <> ''
                       AND t.strLeague = ph.competition_name
                     LIMIT 1),
                    (SELECT t.badge_url FROM teams t
                     WHERE LOWER(TRIM(t.strTeam)) = LOWER(TRIM(ph.home_team))
                       AND t.badge_url IS NOT NULL AND TRIM(t.badge_url) <> ''
                     LIMIT 1)
                ) AS home_badge,
                COALESCE(
                    (SELECT t.badge_url FROM teams t
                     WHERE LOWER(TRIM(t.strTeam)) = LOWER(TRIM(ph.away_team))
                       AND t.badge_url IS NOT NULL AND TRIM(t.badge_url) <> ''
                       AND t.strLeague = ph.competition_name
                     LIMIT 1),
                    (SELECT t.badge_url FROM teams t
                     WHERE LOWER(TRIM(t.strTeam)) = LOWER(TRIM(ph.away_team))
                       AND t.badge_url IS NOT NULL AND TRIM(t.badge_url) <> ''
                     LIMIT 1)
                ) AS away_badge
            FROM predictions_history ph
            {where}
            ORDER BY ph.match_date DESC, ph.id DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


@app.get("/matches/today")
def matches_today():
    conn = get_conn()
    try:
        rows = get_today_matches(conn)
        return [dict(row) for row in rows]
    finally:
        conn.close()


@app.get("/predict/today")
@app.post("/predict/today")
def predict_today():
    ai = get_ai()
    dc        = ai["dc"]
    calibrators = ai["calibrators"]
    ml_models = ai["ml"]

    conn = get_conn()
    try:
        team_histories, elo_state = build_histories_and_elo(conn)
        matches = get_today_matches(conn)
        results = []

        for match in matches:
            ht, at = match["home"], match["away"]
            ct = match["competition_type"] or "LEAGUE"
            lam_h = lam_a = None
            rho = 0.0
            model_used = None

            # ── Dixon-Coles en priorité
            if dc and ht in dc["attack"] and at in dc["attack"]:
                ha = dc["home_advantage"]
                adv = (ha.get(ct) or ha.get("LEAGUE", 0.25)) if isinstance(ha, dict) else ha
                lam_h = math.exp(dc["attack"][ht] + dc["defense"][at] + adv)
                lam_a = math.exp(dc["attack"][at] + dc["defense"][ht])
                rho   = dc.get("rho", 0.0)
                model_used = "Dixon-Coles"

            # ── Fallback ML par compétition
            else:
                features = build_features_for_future_match(match, team_histories, elo_state)
                if features is not None:
                    art = ml_models.get(ct) or ml_models.get("_global")
                    if art:
                        X_h = prepare_single_row(features, art["home"]["features"])
                        X_a = prepare_single_row(features, art["away"]["features"])
                        lam_h = max(0.05, float(art["home"]["model"].predict(X_h)[0]))
                        lam_a = max(0.05, float(art["away"]["model"].predict(X_a)[0]))
                        model_used = f"ML-{ct}"

            if lam_h is None:
                results.append({
                    "date": match["date"],
                    "competition_name": match["competition_name"],
                    "competition_type": ct,
                    "home_team": ht,
                    "away_team": at,
                    "home_badge": match["home_badge"],
                    "away_badge": match["away_badge"],
                    "status_prediction": "INSUFFICIENT_HISTORY",
                })
                continue

            summary = derive_markets_from_poisson(lam_h, lam_a, rho=rho)
            summary = apply_calibration(summary, calibrators)

            results.append({
                "date": match["date"],
                "competition_name": match["competition_name"],
                "competition_type": ct,
                "home_team": ht,
                "away_team": at,
                "home_badge": match["home_badge"],
                "away_badge": match["away_badge"],
                "status_prediction": "OK",
                "model_used": model_used,
                **summary,
            })

        return {"matches": results, "count": len(results)}

    finally:
        conn.close()

@app.get("/top-picks")
def top_picks(limit: int = 5):
    conn = get_conn()
    try:
        # On garde uniquement la prédiction la plus récente par match (évite les doublons)
        rows = conn.execute(
            """
            SELECT
                ph.*,
                COALESCE(
                    (SELECT t.badge_url FROM teams t
                     WHERE LOWER(TRIM(t.strTeam)) = LOWER(TRIM(ph.home_team))
                       AND t.badge_url IS NOT NULL AND TRIM(t.badge_url) <> ''
                       AND t.strLeague = ph.competition_name
                     LIMIT 1),
                    (SELECT t.badge_url FROM teams t
                     WHERE LOWER(TRIM(t.strTeam)) = LOWER(TRIM(ph.home_team))
                       AND t.badge_url IS NOT NULL AND TRIM(t.badge_url) <> ''
                     LIMIT 1)
                ) AS home_badge,
                COALESCE(
                    (SELECT t.badge_url FROM teams t
                     WHERE LOWER(TRIM(t.strTeam)) = LOWER(TRIM(ph.away_team))
                       AND t.badge_url IS NOT NULL AND TRIM(t.badge_url) <> ''
                       AND t.strLeague = ph.competition_name
                     LIMIT 1),
                    (SELECT t.badge_url FROM teams t
                     WHERE LOWER(TRIM(t.strTeam)) = LOWER(TRIM(ph.away_team))
                       AND t.badge_url IS NOT NULL AND TRIM(t.badge_url) <> ''
                     LIMIT 1)
                ) AS away_badge
            FROM predictions_history ph
            WHERE ph.status_prediction = 'OK'
              AND ph.id = (
                  SELECT MAX(id) FROM predictions_history
                  WHERE match_date = ph.match_date
                    AND home_team  = ph.home_team
                    AND away_team  = ph.away_team
                    AND status_prediction = 'OK'
              )
            ORDER BY ph.match_date DESC, ph.confidence DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows if r["confidence"] is not None]
    finally:
        conn.close()


def _fetch_score_from_sportsdb(home: str, away: str, expected_date: str, season: str = "2025-2026") -> dict | None:
    """Retourne le score depuis TheSportsDB si le match est terminé et que la date correspond."""
    try:
        resp = http_requests.get(
            "https://www.thesportsdb.com/api/v1/json/123/searchevents.php",
            params={"e": f"{home} vs {away}", "s": season},
            timeout=10,
        )
        events = resp.json().get("event") or []
        if not events:
            return None
        # Chercher l'événement dont la date correspond exactement
        ev = next(
            (e for e in events if str(e.get("dateEvent", "")).startswith(expected_date)),
            None
        )
        if ev is None:
            return None
        h_score = ev.get("intHomeScore")
        a_score = ev.get("intAwayScore")
        if h_score is None or a_score is None:
            return None
        return {"home_score": int(h_score), "away_score": int(a_score), "status": "FINISHED"}
    except Exception:
        return None


@app.post("/evaluate/latest", response_model=ActionResponse)
def evaluate_latest():
    import evaluate_predict_v1 as ev
    conn = get_conn()
    try:
        ev.ensure_predictions_table(conn)
        pending = conn.execute("""
            SELECT *
            FROM predictions_history
            WHERE evaluation_status IS NULL
               OR evaluation_status IN ('MATCH_NOT_FINISHED', 'REAL_RESULT_NOT_FOUND')
            ORDER BY match_date, id
        """).fetchall()

        if not pending:
            return ActionResponse(ok=True, message="Toutes les prédictions sont déjà évaluées.")

        # Pour chaque match en attente, tenter de mettre à jour la table matches
        # depuis TheSportsDB si le score n'y est pas encore
        for row in pending:
            score = _fetch_score_from_sportsdb(row["home_team"], row["away_team"], row["match_date"])
            if score:
                conn.execute("""
                    UPDATE matches
                    SET home_score = ?, away_score = ?, status = ?
                    WHERE lower(trim(home)) = lower(trim(?))
                      AND lower(trim(away)) = lower(trim(?))
                      AND substr(date, 1, 10) = ?
                """, (
                    score["home_score"], score["away_score"], score["status"],
                    row["home_team"], row["away_team"], row["match_date"],
                ))
        conn.commit()

        evaluated = 0
        not_finished = 0
        not_found = 0
        for row in pending:
            detail = ev.evaluate_prediction_row(conn, row)
            if detail is not None:
                evaluated += 1
            else:
                updated = conn.execute(
                    "SELECT evaluation_status FROM predictions_history WHERE id = ?",
                    (row["id"],)
                ).fetchone()
                status = updated["evaluation_status"] if updated else ""
                if status == "MATCH_NOT_FINISHED":
                    not_finished += 1
                elif status == "REAL_RESULT_NOT_FOUND":
                    not_found += 1
        conn.commit()
        return ActionResponse(
            ok=True,
            message=f"Évalués: {evaluated} | Non terminés: {not_finished} | Introuvables: {not_found}"
        )
    finally:
        conn.close()


@app.get("/h2h")
def head_to_head(home: str, away: str, limit: int = 10):
    """Retourne les dernières confrontations directes entre deux équipes."""
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT home, away, home_score, away_score, date, competition_name, season
            FROM matches
            WHERE home_score IS NOT NULL
              AND away_score IS NOT NULL
              AND status IN ('FINISHED', 'MATCH FINISHED')
              AND (
                (home = ? AND away = ?)
                OR (home = ? AND away = ?)
              )
            ORDER BY date DESC
            LIMIT ?
            """,
            (home, away, away, home, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


@app.get("/standings")
def standings(competition: str, season: str | None = None):
    """Calcule le classement d'une compétition à partir des matchs terminés."""
    from datetime import date as _date

    if not season:
        today = _date.today()
        y = today.year
        season = f"{y}-{y + 1}" if today.month >= 7 else f"{y - 1}-{y}"

    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT home, away, home_score, away_score
            FROM matches
            WHERE home_score IS NOT NULL
              AND away_score IS NOT NULL
              AND status IN ('FINISHED', 'MATCH FINISHED')
              AND competition_name LIKE ?
              AND season = ?
            """,
            (f"%{competition}%", season),
        ).fetchall()

        if not rows:
            return []

        teams: dict[str, dict] = {}
        for r in rows:
            for team, gf, ga in [
                (r["home"], r["home_score"], r["away_score"]),
                (r["away"], r["away_score"], r["home_score"]),
            ]:
                if team not in teams:
                    teams[team] = {"team": team, "played": 0, "won": 0, "drawn": 0, "lost": 0, "gf": 0, "ga": 0}
                t = teams[team]
                t["played"] += 1
                t["gf"] += gf
                t["ga"] += ga
                if gf > ga:
                    t["won"] += 1
                elif gf == ga:
                    t["drawn"] += 1
                else:
                    t["lost"] += 1

        result = list(teams.values())
        for t in result:
            t["points"] = t["won"] * 3 + t["drawn"]
            t["gd"] = t["gf"] - t["ga"]

        result.sort(key=lambda x: (-x["points"], -x["gd"], -x["gf"]))
        for i, t in enumerate(result, 1):
            t["rank"] = i

        return result
    finally:
        conn.close()


@app.get("/results/match")
def get_match_result(home: str, away: str, season: str = "2025-2026"):
    """Interroge TheSportsDB pour le score réel d'un match donné."""
    try:
        event_name = f"{home} vs {away}"
        resp = http_requests.get(
            "https://www.thesportsdb.com/api/v1/json/123/searchevents.php",
            params={"e": event_name, "s": season},
            timeout=10,
        )
        events = resp.json().get("event") or []
        if not events:
            return {"found": False}

        ev = events[0]
        home_score = ev.get("intHomeScore")
        away_score = ev.get("intAwayScore")
        status     = ev.get("strStatus", "")

        if home_score is None or away_score is None:
            return {"found": True, "finished": False, "status": status}

        h, a = int(home_score), int(away_score)
        real1x2  = "1" if h > a else "2" if a > h else "X"
        total    = h + a
        btts     = 1 if h > 0 and a > 0 else 0
        over25   = 1 if total > 2 else 0
        over15   = 1 if total > 1 else 0

        return {
            "found": True,
            "finished": True,
            "status": status,
            "home_score": h,
            "away_score": a,
            "real_result": real1x2,
            "real_btts": btts,
            "real_over_2_5": over25,
            "over_1_5": over15,
            "real_total_goals": total,
        }
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/results/evaluated")
def results_evaluated(days: int = 30):
    """Retourne les matchs évalués récents pour l'auto-évaluation des paris."""
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT
                home_team, away_team, match_date,
                real_home_goals, real_away_goals, real_total_goals,
                real_result, real_btts, real_over_2_5,
                over_1_5
            FROM predictions_history
            WHERE evaluation_status = 'OK'
              AND real_home_goals IS NOT NULL
              AND match_date >= date('now', ? || ' days')
            ORDER BY match_date DESC
            """,
            (f"-{days}",),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
