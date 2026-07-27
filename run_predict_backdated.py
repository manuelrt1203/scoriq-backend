"""Wrapper pour lancer predict_toady_v2.py avec une date forcée."""
import sys
from datetime import datetime as _real_datetime

FAKE_DATE = sys.argv[1]  # ex: "2026-04-24"

class _FakeDatetime(_real_datetime):
    @classmethod
    def now(cls, tz=None):
        return _real_datetime.strptime(FAKE_DATE + " 05:00:00", "%Y-%m-%d %H:%M:%S")

import predict_toady_v2
import db_conn

predict_toady_v2.datetime = _FakeDatetime

# Patch get_today_matches pour inclure les matchs FINISHED (mode backdate)
_original_get_today_matches = predict_toady_v2.get_today_matches

def _backdated_get_today_matches(conn):
    placeholders = ",".join("?" for _ in predict_toady_v2.TARGET_COMPETITION_TYPES)
    return conn.execute(f"""
        SELECT
            id, idLeague, season, round, date, home, away,
            status, competition_type, competition_name
        FROM matches
        WHERE substr(date, 1, 10) = ?
          AND competition_type IN ({placeholders})
        ORDER BY date, id
    """, (FAKE_DATE, *predict_toady_v2.TARGET_COMPETITION_TYPES)).fetchall()

predict_toady_v2.get_today_matches = _backdated_get_today_matches

predict_toady_v2.main()
