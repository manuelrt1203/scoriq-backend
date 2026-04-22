import math
import sqlite3
import joblib
import pandas as pd
from datetime import datetime
from statistics import mean
from collections import defaultdict

DB_PATH = "football.db"

LOOKBACK = 5
LOOKBACK_VENUE = 5
H2H_LOOKBACK = 5
MIN_LEAGUE_MATCHES = 3
TARGET_COMPETITION_TYPES = ("LEAGUE", "DOMESTIC_CUP", "EUROPE", "INTERNATIONAL")

MODEL_HOME_PATHS = {ct: f"model_home_goals_{ct}.pkl" for ct in TARGET_COMPETITION_TYPES}
MODEL_AWAY_PATHS = {ct: f"model_away_goals_{ct}.pkl" for ct in TARGET_COMPETITION_TYPES}
FALLBACK_HOME_PATH = "model_home_goals_v3.pkl"
FALLBACK_AWAY_PATH = "model_away_goals_v3.pkl"

ELO_BASE = 1500.0
ELO_K = 20.0
HOME_ELO_BOOST = 50.0

MIN_CONFIDENCE = 0.45
STRONG_CONFIDENCE = 0.55

ODDS_FILE = "bookmaker_odds_today.csv"  # optionnel


def ensure_predictions_table(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS predictions_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prediction_run_date TEXT NOT NULL,
        match_date TEXT NOT NULL,
        competition_name TEXT,
        competition_type TEXT,
        home_team TEXT NOT NULL,
        away_team TEXT NOT NULL,
        status_prediction TEXT NOT NULL,

        top_pick TEXT,
        confidence REAL,
        trust_level TEXT,

        proba_home_win REAL,
        proba_draw REAL,
        proba_away_win REAL,

        pred_home_goals REAL,
        pred_away_goals REAL,
        pred_total_goals REAL,

        over_1_5 REAL,
        over_2_5 REAL,
        over_3_5 REAL,
        btts_yes REAL,

        most_likely_score TEXT,
        most_likely_score_prob REAL,

        value_pick TEXT,
        value_edge REAL,

        evaluation_status TEXT,
        real_home_goals INTEGER,
        real_away_goals INTEGER,
        real_result TEXT,
        real_total_goals INTEGER,
        real_btts INTEGER,
        real_over_2_5 INTEGER,

        is_correct_1x2 INTEGER,
        is_correct_score INTEGER,
        is_correct_btts INTEGER,
        is_correct_over_2_5 INTEGER,

        abs_error_home_goals REAL,
        abs_error_away_goals REAL,
        abs_error_total_goals REAL
    );

    CREATE INDEX IF NOT EXISTS idx_predictions_history_run_date
    ON predictions_history(prediction_run_date);

    CREATE INDEX IF NOT EXISTS idx_predictions_history_match_date
    ON predictions_history(match_date);

    CREATE INDEX IF NOT EXISTS idx_predictions_history_match_key
    ON predictions_history(match_date, home_team, away_team);
    """)
    conn.commit()


def parse_date(d):
    d = str(d)
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(d, fmt)
        except ValueError:
            pass
    raise ValueError(f"Format de date non géré : {d}")


def avg(values):
    return mean(values) if values else 0.0


def points_from_result(gf, ga):
    if gf > ga:
        return 3
    if gf == ga:
        return 1
    return 0


def result_score(gf, ga):
    if gf > ga:
        return 1.0
    if gf == ga:
        return 0.5
    return 0.0


def expected_score(elo_a, elo_b):
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def update_elo(elo_home, elo_away, home_goals, away_goals):
    home_exp = expected_score(elo_home + HOME_ELO_BOOST, elo_away)
    away_exp = 1.0 - home_exp

    home_real = result_score(home_goals, away_goals)
    away_real = 1.0 - home_real

    new_home = elo_home + ELO_K * (home_real - home_exp)
    new_away = elo_away + ELO_K * (away_real - away_exp)
    return new_home, new_away


def competition_type_code(comp_type):
    mapping = {"LEAGUE": 0, "DOMESTIC_CUP": 1, "EUROPE": 2, "INTERNATIONAL": 3}
    return mapping.get(comp_type, -1)


def get_finished_matches(conn):
    placeholders = ",".join("?" for _ in TARGET_COMPETITION_TYPES)
    return conn.execute(f"""
        SELECT
            id, idLeague, season, round, date, home, away,
            home_score, away_score, status, competition_type, competition_name
        FROM matches
        WHERE status = 'FINISHED'
          AND home_score IS NOT NULL
          AND away_score IS NOT NULL
          AND competition_type IN ({placeholders})
        ORDER BY date, round, id
    """, TARGET_COMPETITION_TYPES).fetchall()


def get_today_matches(conn):
    today = datetime.now().strftime("%Y-%m-%d")
    placeholders = ",".join("?" for _ in TARGET_COMPETITION_TYPES)

    return conn.execute(f"""
        SELECT
            id, idLeague, season, round, date, home, away,
            status, competition_type, competition_name
        FROM matches
        WHERE substr(date, 1, 10) = ?
          AND competition_type IN ({placeholders})
          AND (
              status IS NULL OR status IN ('SCHEDULED', 'TIMED', 'NOT_STARTED', 'NS')
          )
        ORDER BY date, id
    """, (today, *TARGET_COMPETITION_TYPES)).fetchall()


def build_histories_and_elo(conn):
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
            hg, ag
        )
        elo_state["global"][home_team] = new_home_elo
        elo_state["global"][away_team] = new_away_elo

        new_home_home_elo, _ = update_elo(
            elo_state["home"][home_team],
            elo_state["away"][away_team],
            hg, ag
        )
        _, new_away_away_elo = update_elo(
            elo_state["home"][home_team],
            elo_state["away"][away_team],
            hg, ag
        )
        elo_state["home"][home_team] = new_home_home_elo
        elo_state["away"][away_team] = new_away_away_elo

    return team_histories, elo_state


def get_recent_history(history_list, before_dt, limit, competition_type=None):
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


def matches_in_last_days(history_list, before_dt, days=14):
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


def days_since_last_match(history_list, before_dt):
    for item in reversed(history_list):
        if item["date_dt"] < before_dt:
            return (before_dt - item["date_dt"]).days
    return 99


def get_recent_home_venue(history_list, before_dt, limit):
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


def get_recent_away_venue(history_list, before_dt, limit):
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


def get_h2h(history_list, opponent, before_dt, limit):
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
    h2h_n    = len(h2h)
    h2h_wins = sum(x["win"]  for x in h2h)
    h2h_draws = sum(x["draw"] for x in h2h)
    h2h_loss  = sum(x["loss"] for x in h2h)
    h2h_scored   = avg([x["goals_for"]    for x in h2h])
    h2h_conceded = avg([x["goals_against"] for x in h2h])
    h2h_points   = sum(x["points"] for x in h2h)

    # Rest days
    home_rest_days = days_since_last_match(home_hist, match_dt)
    away_rest_days = days_since_last_match(away_hist, match_dt)

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
    }


def prepare_single_row(features_dict, model_features):
    df = pd.DataFrame([features_dict]).copy()

    if "season" in df.columns:
        df["season"] = (
            df["season"]
            .astype(str)
            .str.extract(r"(\d{4})", expand=False)
            .fillna("0")
            .astype(int)
        )

    df = pd.get_dummies(df, drop_first=False)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    for col in model_features:
        if col not in df.columns:
            df[col] = 0

    df = df[model_features]
    return df


def poisson_prob(lmbda, k):
    return math.exp(-lmbda) * (lmbda ** k) / math.factorial(k)


def score_matrix(home_lambda, away_lambda, max_goals=8):
    matrix = []
    for hg in range(max_goals + 1):
        row = []
        for ag in range(max_goals + 1):
            row.append(poisson_prob(home_lambda, hg) * poisson_prob(away_lambda, ag))
        matrix.append(row)
    return matrix


def derive_markets_from_poisson(home_lambda, away_lambda):
    home_lambda = max(0.05, float(home_lambda))
    away_lambda = max(0.05, float(away_lambda))

    matrix = score_matrix(home_lambda, away_lambda, max_goals=8)

    p_home = p_draw = p_away = 0.0
    p_over_15 = p_over_25 = p_over_35 = 0.0
    p_btts = 0.0

    best_score = (0, 0)
    best_p = -1.0

    for hg, row in enumerate(matrix):
        for ag, p in enumerate(row):
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

            if p > best_p:
                best_p = p
                best_score = (hg, ag)

    probs = {"1": p_home, "X": p_draw, "2": p_away}
    confidence = max(probs.values())
    top_pick = max(probs, key=probs.get)

    if confidence >= STRONG_CONFIDENCE:
        trust_level = "FORTE"
    elif confidence >= MIN_CONFIDENCE:
        trust_level = "MOYENNE"
    else:
        trust_level = "FAIBLE"

    return {
        "home_goals": home_lambda,
        "away_goals": away_lambda,
        "total_goals": home_lambda + away_lambda,
        "proba_home_win": p_home,
        "proba_draw": p_draw,
        "proba_away_win": p_away,
        "over_1_5": p_over_15,
        "over_2_5": p_over_25,
        "over_3_5": p_over_35,
        "btts_yes": p_btts,
        "most_likely_score": f"{best_score[0]}-{best_score[1]}",
        "most_likely_score_prob": best_p,
        "confidence": confidence,
        "top_pick": top_pick,
        "trust_level": trust_level,
    }


def pct(x):
    return f"{100 * float(x):.2f}%"


def load_odds():
    try:
        df = pd.read_csv(ODDS_FILE)
    except FileNotFoundError:
        return None

    required = {"home_team", "away_team", "odd_1", "odd_x", "odd_2"}
    if not required.issubset(df.columns):
        print("Le fichier de cotes existe mais les colonnes attendues ne sont pas présentes.")
        return None

    return df


def find_match_odds(odds_df, home_team, away_team):
    if odds_df is None:
        return None
    rows = odds_df[(odds_df["home_team"] == home_team) & (odds_df["away_team"] == away_team)]
    if rows.empty:
        return None
    return rows.iloc[0]


def implied_prob(odd):
    if odd is None or pd.isna(odd) or odd <= 0:
        return None
    return 1.0 / float(odd)


def value_info(pred_summary, odds_row):
    if odds_row is None:
        return {"value_pick": None, "value_edge": None}

    implied_1 = implied_prob(odds_row["odd_1"])
    implied_x = implied_prob(odds_row["odd_x"])
    implied_2 = implied_prob(odds_row["odd_2"])

    edges = {
        "1": pred_summary["proba_home_win"] - implied_1 if implied_1 is not None else None,
        "X": pred_summary["proba_draw"] - implied_x if implied_x is not None else None,
        "2": pred_summary["proba_away_win"] - implied_2 if implied_2 is not None else None,
    }

    valid_edges = {k: v for k, v in edges.items() if v is not None}
    if not valid_edges:
        return {"value_pick": None, "value_edge": None}

    best_pick = max(valid_edges, key=valid_edges.get)
    return {
        "value_pick": best_pick,
        "value_edge": valid_edges[best_pick]
    }


def build_txt_block(match, summary, status_msg, value_pick=None, value_edge=None):
    lines = []
    lines.append("=" * 80)
    lines.append(f"Compétition : {match['competition_name']}")
    lines.append(f"Date        : {match['date']}")
    lines.append(f"Match       : {match['home']} vs {match['away']}")
    lines.append("")

    if summary is None:
        lines.append("Pronostic")
        lines.append(f"- {status_msg}")
        lines.append("=" * 80)
        lines.append("")
        return "\n".join(lines)

    lines.append("Pronostic principal")
    lines.append(f"- Choix principal : {summary['top_pick']}")
    lines.append(f"- Confiance       : {pct(summary['confidence'])}")
    lines.append(f"- Niveau          : {summary['trust_level']}")
    lines.append("")

    lines.append("Probabilités 1X2")
    lines.append(f"- 1 : {pct(summary['proba_home_win'])}")
    lines.append(f"- X : {pct(summary['proba_draw'])}")
    lines.append(f"- 2 : {pct(summary['proba_away_win'])}")
    lines.append("")

    lines.append("Buts attendus")
    lines.append(f"- Domicile : {summary['home_goals']:.2f}")
    lines.append(f"- Extérieur: {summary['away_goals']:.2f}")
    lines.append(f"- Total    : {summary['total_goals']:.2f}")
    lines.append("")

    lines.append("Marchés buts")
    lines.append(f"- Over 1.5 : {pct(summary['over_1_5'])}")
    lines.append(f"- Over 2.5 : {pct(summary['over_2_5'])}")
    lines.append(f"- Over 3.5 : {pct(summary['over_3_5'])}")
    lines.append(f"- BTTS Oui : {pct(summary['btts_yes'])}")
    lines.append("")

    lines.append("Score probable")
    lines.append(f"- Score : {summary['most_likely_score']}")
    lines.append(f"- Proba : {pct(summary['most_likely_score_prob'])}")
    lines.append("")

    if value_pick is not None and value_edge is not None:
        lines.append("Value betting")
        lines.append(f"- Meilleure value : {value_pick}")
        lines.append(f"- Edge modèle     : {pct(value_edge)}")
        lines.append("")

    lines.append("Lecture")
    if summary["trust_level"] == "FAIBLE":
        lines.append("- Match trop flou : à éviter.")
    elif summary["trust_level"] == "MOYENNE":
        lines.append("- Match jouable avec prudence.")
    else:
        lines.append("- Match parmi les plus nets du jour.")

    lines.append("=" * 80)
    lines.append("")
    return "\n".join(lines)


def insert_prediction_history(conn, row_dict):
    conn.execute("""
        INSERT INTO predictions_history (
            prediction_run_date,
            match_date,
            competition_name,
            competition_type,
            home_team,
            away_team,
            status_prediction,
            top_pick,
            confidence,
            trust_level,
            proba_home_win,
            proba_draw,
            proba_away_win,
            pred_home_goals,
            pred_away_goals,
            pred_total_goals,
            over_1_5,
            over_2_5,
            over_3_5,
            btts_yes,
            most_likely_score,
            most_likely_score_prob,
            value_pick,
            value_edge,
            evaluation_status,
            real_home_goals,
            real_away_goals,
            real_result,
            real_total_goals,
            real_btts,
            real_over_2_5,
            is_correct_1x2,
            is_correct_score,
            is_correct_btts,
            is_correct_over_2_5,
            abs_error_home_goals,
            abs_error_away_goals,
            abs_error_total_goals
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        row_dict.get("prediction_run_date"),
        row_dict.get("match_date"),
        row_dict.get("competition_name"),
        row_dict.get("competition_type"),
        row_dict.get("home_team"),
        row_dict.get("away_team"),
        row_dict.get("status_prediction"),
        row_dict.get("top_pick"),
        row_dict.get("confidence"),
        row_dict.get("trust_level"),
        row_dict.get("proba_home_win"),
        row_dict.get("proba_draw"),
        row_dict.get("proba_away_win"),
        row_dict.get("pred_home_goals"),
        row_dict.get("pred_away_goals"),
        row_dict.get("pred_total_goals"),
        row_dict.get("over_1_5"),
        row_dict.get("over_2_5"),
        row_dict.get("over_3_5"),
        row_dict.get("btts_yes"),
        row_dict.get("most_likely_score"),
        row_dict.get("most_likely_score_prob"),
        row_dict.get("value_pick"),
        row_dict.get("value_edge"),
        None, None, None, None, None, None, None, None, None, None, None, None, None, None
    ))


def main():
    today_str = datetime.now().strftime("%Y-%m-%d")
    txt_output_path = f"predictions_v2_{today_str}.txt"
    csv_output_path = f"predictions_v2_{today_str}.csv"

    home_models = {}
    away_models = {}
    for ct in TARGET_COMPETITION_TYPES:
        try:
            home_models[ct] = joblib.load(MODEL_HOME_PATHS[ct])
            away_models[ct] = joblib.load(MODEL_AWAY_PATHS[ct])
        except FileNotFoundError:
            pass

    try:
        fallback_home = joblib.load(FALLBACK_HOME_PATH)
        fallback_away = joblib.load(FALLBACK_AWAY_PATH)
    except FileNotFoundError:
        fallback_home = fallback_away = None

    if not home_models and fallback_home is None:
        print("Aucun modèle trouvé.")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    ensure_predictions_table(conn)

    team_histories, elo_state = build_histories_and_elo(conn)
    matches = get_today_matches(conn)
    odds_df = load_odds()

    if not matches:
        print("Aucun match du jour trouvé.")
        conn.close()
        return

    txt_blocks = []
    out_rows = []

    prediction_run_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for match in matches:
        features = build_features_for_future_match(match, team_histories, elo_state)

        if features is None:
            msg = "Historique insuffisant pour produire une prédiction fiable."
            txt_blocks.append(build_txt_block(match, None, msg))

            db_row = {
                "prediction_run_date": prediction_run_date,
                "match_date": str(match["date"])[:10],
                "competition_name": match["competition_name"],
                "competition_type": match["competition_type"],
                "home_team": match["home"],
                "away_team": match["away"],
                "status_prediction": "INSUFFICIENT_HISTORY",
                "top_pick": None,
                "confidence": None,
                "trust_level": None,
                "proba_home_win": None,
                "proba_draw": None,
                "proba_away_win": None,
                "pred_home_goals": None,
                "pred_away_goals": None,
                "pred_total_goals": None,
                "over_1_5": None,
                "over_2_5": None,
                "over_3_5": None,
                "btts_yes": None,
                "most_likely_score": None,
                "most_likely_score_prob": None,
                "value_pick": None,
                "value_edge": None,
            }
            insert_prediction_history(conn, db_row)

            out_rows.append({
                "date": match["date"],
                "competition_name": match["competition_name"],
                "competition_type": match["competition_type"],
                "home_team": match["home"],
                "away_team": match["away"],
                "status_prediction": "INSUFFICIENT_HISTORY",
            })
            continue

        ct = match["competition_type"]
        home_artifact = home_models.get(ct) or fallback_home
        away_artifact = away_models.get(ct) or fallback_away

        if home_artifact is None or away_artifact is None:
            msg = f"Aucun modèle disponible pour le type {ct}."
            txt_blocks.append(build_txt_block(match, None, msg))
            continue

        X_home = prepare_single_row(features, home_artifact["features"])
        X_away = prepare_single_row(features, away_artifact["features"])

        pred_home_goals = max(0.05, float(home_artifact["model"].predict(X_home)[0]))
        pred_away_goals = max(0.05, float(away_artifact["model"].predict(X_away)[0]))

        summary = derive_markets_from_poisson(pred_home_goals, pred_away_goals)

        odds_row = find_match_odds(odds_df, match["home"], match["away"])
        val = value_info(summary, odds_row)

        txt_blocks.append(
            build_txt_block(
                match,
                summary,
                status_msg="OK",
                value_pick=val["value_pick"],
                value_edge=val["value_edge"]
            )
        )

        out_row = {
            "date": match["date"],
            "competition_name": match["competition_name"],
            "competition_type": match["competition_type"],
            "home_team": match["home"],
            "away_team": match["away"],
            "status_prediction": "OK",
            "top_pick": summary["top_pick"],
            "confidence": round(summary["confidence"], 6),
            "trust_level": summary["trust_level"],
            "proba_home_win": round(summary["proba_home_win"], 6),
            "proba_draw": round(summary["proba_draw"], 6),
            "proba_away_win": round(summary["proba_away_win"], 6),
            "pred_home_goals": round(summary["home_goals"], 4),
            "pred_away_goals": round(summary["away_goals"], 4),
            "pred_total_goals": round(summary["total_goals"], 4),
            "over_1_5": round(summary["over_1_5"], 6),
            "over_2_5": round(summary["over_2_5"], 6),
            "over_3_5": round(summary["over_3_5"], 6),
            "btts_yes": round(summary["btts_yes"], 6),
            "most_likely_score": summary["most_likely_score"],
            "most_likely_score_prob": round(summary["most_likely_score_prob"], 6),
            "value_pick": val["value_pick"],
            "value_edge": round(val["value_edge"], 6) if val["value_edge"] is not None else None,
        }
        out_rows.append(out_row)

        db_row = {
            "prediction_run_date": prediction_run_date,
            "match_date": str(match["date"])[:10],
            **out_row
        }
        insert_prediction_history(conn, db_row)

    conn.commit()
    conn.close()

    with open(txt_output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_blocks))

    pd.DataFrame(out_rows).to_csv(csv_output_path, index=False, encoding="utf-8")

    print(f"Fichier texte créé : {txt_output_path}")
    print(f"Fichier CSV créé   : {csv_output_path}")
    print("Prédictions enregistrées aussi dans football.db -> table predictions_history")


if __name__ == "__main__":
    main()