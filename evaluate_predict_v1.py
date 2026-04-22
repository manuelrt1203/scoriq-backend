import db_conn

OUTPUT_FILE = "evaluation_report_db.txt"

FINISHED_STATUSES = {"FINISHED", "FT", "Match Finished", "Ended"}


def result_label(home, away):
    if home > away:
        return "1"
    elif home < away:
        return "2"
    return "X"


def ensure_predictions_table(conn):
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
    """)
    conn.commit()


def normalize_name(x):
    return str(x).strip().lower()


def row_match_exists_and_finished(conn, pred_row):
    match = conn.execute("""
        SELECT home_score, away_score, status
        FROM matches
        WHERE lower(trim(home)) = ?
          AND lower(trim(away)) = ?
          AND substr(date, 1, 10) = ?
        LIMIT 1
    """, (
        normalize_name(pred_row["home_team"]),
        normalize_name(pred_row["away_team"]),
        pred_row["match_date"],
    )).fetchone()

    if not match:
        return False

    status = str(match["status"]).strip() if match["status"] is not None else ""
    return status in FINISHED_STATUSES and match["home_score"] is not None and match["away_score"] is not None


def get_latest_evaluable_prediction_run(conn):
    runs = conn.execute("""
        SELECT DISTINCT prediction_run_date
        FROM predictions_history
        ORDER BY prediction_run_date DESC
    """).fetchall()

    for run in runs:
        run_date = run["prediction_run_date"]

        pred_rows = conn.execute("""
            SELECT *
            FROM predictions_history
            WHERE prediction_run_date = ?
        """, (run_date,)).fetchall()

        for pred_row in pred_rows:
            if pred_row["status_prediction"] == "INSUFFICIENT_HISTORY":
                continue
            if row_match_exists_and_finished(conn, pred_row):
                return run_date

    return None


def evaluate_prediction_row(conn, pred_row):
    if pred_row["status_prediction"] == "INSUFFICIENT_HISTORY":
        conn.execute("""
            UPDATE predictions_history
            SET evaluation_status = 'SKIPPED_INSUFFICIENT_HISTORY'
            WHERE id = ?
        """, (pred_row["id"],))
        return None

    # SQLite lower() ne gère pas les caractères non-ASCII (ex: İ turc)
    # On compare côté Python après récupération par date + équipe exacte
    candidates = conn.execute("""
        SELECT home_score, away_score, status, date, home, away
        FROM matches
        WHERE substr(date, 1, 10) = ?
    """, (pred_row["match_date"],)).fetchall()

    pred_home = normalize_name(pred_row["home_team"])
    pred_away = normalize_name(pred_row["away_team"])
    match = next(
        (r for r in candidates
         if normalize_name(r["home"]) == pred_home and normalize_name(r["away"]) == pred_away),
        None
    )

    if not match:
        conn.execute("""
            UPDATE predictions_history
            SET evaluation_status = 'REAL_RESULT_NOT_FOUND'
            WHERE id = ?
        """, (pred_row["id"],))
        return None

    status = str(match["status"]).strip() if match["status"] is not None else ""
    if status not in FINISHED_STATUSES:
        conn.execute("""
            UPDATE predictions_history
            SET evaluation_status = 'MATCH_NOT_FINISHED'
            WHERE id = ?
        """, (pred_row["id"],))
        return None

    home_goals = int(match["home_score"])
    away_goals = int(match["away_score"])

    real_result = result_label(home_goals, away_goals)
    real_total_goals = home_goals + away_goals
    real_btts = 1 if home_goals > 0 and away_goals > 0 else 0
    real_over_2_5 = 1 if real_total_goals >= 3 else 0

    probs = {
        "1": pred_row["proba_home_win"] if pred_row["proba_home_win"] is not None else -1,
        "X": pred_row["proba_draw"] if pred_row["proba_draw"] is not None else -1,
        "2": pred_row["proba_away_win"] if pred_row["proba_away_win"] is not None else -1,
    }
    pred_result = max(probs, key=probs.get)

    pred_score = pred_row["most_likely_score"]
    real_score = f"{home_goals}-{away_goals}"

    pred_btts = 1 if pred_row["btts_yes"] is not None and pred_row["btts_yes"] >= 0.5 else 0
    pred_over_2_5 = 1 if pred_row["over_2_5"] is not None and pred_row["over_2_5"] >= 0.5 else 0

    is_correct_1x2 = 1 if pred_result == real_result else 0
    is_correct_score = 1 if pred_score == real_score else 0
    is_correct_btts = 1 if pred_btts == real_btts else 0
    is_correct_over_2_5 = 1 if pred_over_2_5 == real_over_2_5 else 0

    abs_error_home_goals = abs(pred_row["pred_home_goals"] - home_goals) if pred_row["pred_home_goals"] is not None else None
    abs_error_away_goals = abs(pred_row["pred_away_goals"] - away_goals) if pred_row["pred_away_goals"] is not None else None
    abs_error_total_goals = abs(pred_row["pred_total_goals"] - real_total_goals) if pred_row["pred_total_goals"] is not None else None

    conn.execute("""
        UPDATE predictions_history
        SET
            evaluation_status = 'OK',
            real_home_goals = ?,
            real_away_goals = ?,
            real_result = ?,
            real_total_goals = ?,
            real_btts = ?,
            real_over_2_5 = ?,
            is_correct_1x2 = ?,
            is_correct_score = ?,
            is_correct_btts = ?,
            is_correct_over_2_5 = ?,
            abs_error_home_goals = ?,
            abs_error_away_goals = ?,
            abs_error_total_goals = ?
        WHERE id = ?
    """, (
        home_goals,
        away_goals,
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
        abs_error_total_goals,
        pred_row["id"],
    ))

    return {
        "home_team": pred_row["home_team"],
        "away_team": pred_row["away_team"],
        "match_date": pred_row["match_date"],
        "pred_result": pred_result,
        "real_result": real_result,
        "is_correct_1x2": is_correct_1x2,
        "pred_score": pred_score,
        "real_score": real_score,
        "is_correct_score": is_correct_score,
        "is_correct_btts": is_correct_btts,
        "is_correct_over_2_5": is_correct_over_2_5,
        "abs_error_home_goals": abs_error_home_goals,
        "abs_error_away_goals": abs_error_away_goals,
        "abs_error_total_goals": abs_error_total_goals,
        "trust_level": pred_row["trust_level"],
        "confidence": pred_row["confidence"],
    }


def build_summary(conn, prediction_run_date):
    rows = conn.execute("""
        SELECT *
        FROM predictions_history
        WHERE prediction_run_date = ?
          AND evaluation_status = 'OK'
    """, (prediction_run_date,)).fetchall()

    total = len(rows)
    if total == 0:
        return None, []

    correct_1x2 = sum(r["is_correct_1x2"] or 0 for r in rows)
    correct_score = sum(r["is_correct_score"] or 0 for r in rows)
    correct_btts = sum(r["is_correct_btts"] or 0 for r in rows)
    correct_over = sum(r["is_correct_over_2_5"] or 0 for r in rows)

    home_errors = [r["abs_error_home_goals"] for r in rows if r["abs_error_home_goals"] is not None]
    away_errors = [r["abs_error_away_goals"] for r in rows if r["abs_error_away_goals"] is not None]
    total_errors = [r["abs_error_total_goals"] for r in rows if r["abs_error_total_goals"] is not None]

    summary = {
        "prediction_run_date": prediction_run_date,
        "total": total,
        "accuracy_1x2": correct_1x2 / total,
        "accuracy_score": correct_score / total,
        "accuracy_btts": correct_btts / total,
        "accuracy_over_2_5": correct_over / total,
        "mae_home_goals": sum(home_errors) / len(home_errors) if home_errors else None,
        "mae_away_goals": sum(away_errors) / len(away_errors) if away_errors else None,
        "mae_total_goals": sum(total_errors) / len(total_errors) if total_errors else None,
    }

    return summary, rows


def build_global_summary(conn):
    """Résumé global sur toutes les prédictions évaluées OK."""
    rows = conn.execute("""
        SELECT *
        FROM predictions_history
        WHERE evaluation_status = 'OK'
    """).fetchall()

    total = len(rows)
    if total == 0:
        return None

    correct_1x2  = sum(r["is_correct_1x2"]    or 0 for r in rows)
    correct_score = sum(r["is_correct_score"]  or 0 for r in rows)
    correct_btts  = sum(r["is_correct_btts"]   or 0 for r in rows)
    correct_over  = sum(r["is_correct_over_2_5"] or 0 for r in rows)

    total_errors = [r["abs_error_total_goals"] for r in rows if r["abs_error_total_goals"] is not None]

    return {
        "total":            total,
        "accuracy_1x2":     correct_1x2  / total,
        "accuracy_score":   correct_score / total,
        "accuracy_btts":    correct_btts  / total,
        "accuracy_over_2_5": correct_over / total,
        "mae_total_goals":  sum(total_errors) / len(total_errors) if total_errors else None,
    }


def main():
    conn = db_conn.get_connection()
    ensure_predictions_table(conn)

    # Récupérer les prédictions non évaluées ou dont le match n'était pas terminé
    pending = conn.execute("""
        SELECT *
        FROM predictions_history
        WHERE evaluation_status IS NULL
           OR evaluation_status IN ('MATCH_NOT_FINISHED', 'REAL_RESULT_NOT_FOUND')
        ORDER BY match_date, id
    """).fetchall()

    if not pending:
        print("Toutes les prédictions sont déjà évaluées.")
        conn.close()
        return

    print(f"{len(pending)} prédictions à évaluer...\n")

    evaluated_details = []
    counts = {"OK": 0, "REAL_RESULT_NOT_FOUND": 0, "MATCH_NOT_FINISHED": 0,
              "SKIPPED_INSUFFICIENT_HISTORY": 0}

    for pred_row in pending:
        detail = evaluate_prediction_row(conn, pred_row)
        if detail is not None:
            evaluated_details.append(detail)
            counts["OK"] += 1
        else:
            # Lire le statut mis à jour
            updated = conn.execute(
                "SELECT evaluation_status FROM predictions_history WHERE id = ?",
                (pred_row["id"],)
            ).fetchone()
            status = updated["evaluation_status"] if updated else "UNKNOWN"
            if status in counts:
                counts[status] += 1

    conn.commit()

    # ── Rapport ──────────────────────────────────────────────
    global_summary = build_global_summary(conn)

    lines = []
    lines.append("=" * 80)
    lines.append("ÉVALUATION COMPLÈTE DES PRÉDICTIONS — football.db")
    lines.append("=" * 80)
    lines.append(f"Prédictions traitées cette passe : {len(pending)}")
    lines.append(f"  Évaluées avec succès  : {counts['OK']}")
    lines.append(f"  Match non terminé     : {counts['MATCH_NOT_FINISHED']}")
    lines.append(f"  Résultat introuvable  : {counts['REAL_RESULT_NOT_FOUND']}")
    lines.append(f"  Historique insuff.    : {counts['SKIPPED_INSUFFICIENT_HISTORY']}")

    if global_summary:
        lines.append("")
        lines.append("── PERFORMANCES GLOBALES (toutes évaluations confondues) ──")
        lines.append(f"Total évalués OK     : {global_summary['total']}")
        lines.append(f"Accuracy 1X2         : {global_summary['accuracy_1x2']:.2%}")
        lines.append(f"Accuracy score exact : {global_summary['accuracy_score']:.2%}")
        lines.append(f"Accuracy BTTS        : {global_summary['accuracy_btts']:.2%}")
        lines.append(f"Accuracy Over 2.5    : {global_summary['accuracy_over_2_5']:.2%}")
        lines.append(f"MAE buts totaux      : {global_summary['mae_total_goals']:.2f}" if global_summary['mae_total_goals'] else "MAE buts totaux      : —")

    lines.append("")
    lines.append("─" * 80)
    lines.append("DÉTAIL PAR MATCH")
    lines.append("─" * 80)

    for d in evaluated_details:
        ok = "✓" if d["is_correct_1x2"] else "✗"
        lines.append(
            f"{ok} {d['home_team']} vs {d['away_team']} ({d['match_date']}) | "
            f"Prévu {d['pred_result']} → Réel {d['real_result']} | "
            f"Score {d['pred_score']}→{d['real_score']} | "
            f"BTTS {'✓' if d['is_correct_btts'] else '✗'} | "
            f"O2.5 {'✓' if d['is_correct_over_2_5'] else '✗'} | "
            f"Confiance {d['confidence']:.0%} [{d['trust_level']}]"
        )

    report = "\n".join(lines)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"\nRapport sauvegardé : {OUTPUT_FILE}")

    conn.close()


if __name__ == "__main__":
    main()