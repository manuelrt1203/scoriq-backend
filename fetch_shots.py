#!/usr/bin/env python3
"""
Télécharge les statistiques de tirs (HST, AST, HS, AS) depuis football-data.co.uk
et les stocke dans la table shots_data de football.db.

Les tirs cadrés (shots on target) sont un proxy fiable du xG (corrélation ~0.85)
et permettent d'enrichir les features du dataset.

Données disponibles : 2014-15 à 2024-25, 6 ligues majeures
Usage : python fetch_shots.py
"""

import io
import sqlite3
import time

import pandas as pd
import requests

DB_PATH = "football.db"
BASE_URL = "https://www.football-data.co.uk/mmz4281"

LEAGUES = {
    "E0":  {"idLeague": 4328, "name": "Premier League"},
    "D1":  {"idLeague": 4331, "name": "Bundesliga"},
    "I1":  {"idLeague": 4332, "name": "Serie A"},
    "SP1": {"idLeague": 4335, "name": "La Liga"},
    "F1":  {"idLeague": 4334, "name": "Ligue 1"},
    "N1":  {"idLeague": 4337, "name": "Eredivisie"},
}

SEASONS = [
    ("1415", "2014-2015"), ("1516", "2015-2016"), ("1617", "2016-2017"),
    ("1718", "2017-2018"), ("1819", "2018-2019"), ("1920", "2019-2020"),
    ("2021", "2020-2021"), ("2122", "2021-2022"), ("2223", "2022-2023"),
    ("2324", "2023-2024"), ("2425", "2024-2025"),
]

# Normalisation noms football-data.co.uk → TheSportsDB
TEAM_MAP = {
    # Premier League
    "Man United":    "Manchester United",
    "Man City":      "Manchester City",
    "Spurs":         "Tottenham",
    "Newcastle":     "Newcastle",
    "Nott'm Forest": "Nottingham Forest",
    "Brighton":      "Brighton",
    "Wolves":        "Wolves",
    "West Ham":      "West Ham",
    "West Brom":     "West Brom",
    "Sheffield Utd": "Sheffield United",
    "Leeds":         "Leeds United",
    "Blackburn":     "Blackburn",
    "Bolton":        "Bolton",
    "QPR":           "QPR",
    # Bundesliga
    "Leverkusen":    "Bayer Leverkusen",
    "Dortmund":      "Borussia Dortmund",
    "Gladbach":      "Borussia Mönchengladbach",
    "Frankfurt":     "Eintracht Frankfurt",
    "Hertha":        "Hertha Berlin",
    "Mainz":         "Mainz",
    "Paderborn":     "Paderborn",
    "Ingolstadt":    "Ingolstadt",
    "Stuttgart":     "Stuttgart",
    "Wolfsburg":     "Wolfsburg",
    "Bochum":        "Bochum",
    "Hoffenheim":    "Hoffenheim",
    # Serie A
    "Inter":         "Inter Milan",
    "AC Milan":      "Milan",
    "Lazio":         "Lazio",
    "Roma":          "Roma",
    "Napoli":        "Napoli",
    "Juventus":      "Juventus",
    "Fiorentina":    "Fiorentina",
    "Atalanta":      "Atalanta",
    "Torino":        "Torino",
    "Bologna":       "Bologna",
    "Udinese":       "Udinese",
    "Verona":        "Hellas Verona",
    # La Liga
    "Ath Madrid":    "Ath Madrid",
    "Ath Bilbao":    "Ath Bilbao",
    "Betis":         "Real Betis",
    "Celta":         "Celta Vigo",
    "Espanol":       "Espanyol",
    "Getafe":        "Getafe",
    "Osasuna":       "Osasuna",
    "Sociedad":      "Sociedad",
    "Vallecano":     "Rayo Vallecano",
    "Villarreal":    "Villarreal",
    "Valencia":      "Valencia",
    "Alaves":        "Alaves",
    # Ligue 1
    "Paris SG":      "Paris SG",
    "Lyon":          "Lyon",
    "Marseille":     "Marseille",
    "Monaco":        "Monaco",
    "Lille":         "Lille",
    "Rennes":        "Rennes",
    "Nice":          "Nice",
    "Lens":          "Lens",
    "Nantes":        "Nantes",
    "Strasbourg":    "Strasbourg",
    "Reims":         "Stade de Reims",
    # Eredivisie
    "Ajax":          "Ajax",
    "PSV":           "PSV Eindhoven",
    "Feyenoord":     "Feyenoord",
    "Twente":        "Twente",
    "Utrecht":       "Utrecht",
}


def normalize_team(name: str) -> str:
    return TEAM_MAP.get(name, name)


def parse_date(date_str: str) -> str | None:
    """Convert DD/MM/YY or DD/MM/YYYY to YYYY-MM-DD."""
    if not date_str or pd.isna(date_str):
        return None
    for fmt in ("%d/%m/%y", "%d/%m/%Y"):
        try:
            from datetime import datetime
            return datetime.strptime(str(date_str).strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS shots_data (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            idLeague     INTEGER,
            season       TEXT,
            match_date   TEXT,
            home_team    TEXT,
            away_team    TEXT,
            home_shots   INTEGER,
            away_shots   INTEGER,
            home_sot     INTEGER,
            away_sot     INTEGER,
            UNIQUE(idLeague, season, match_date, home_team, away_team)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_shots_team_date
        ON shots_data(home_team, match_date)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_shots_away_date
        ON shots_data(away_team, match_date)
    """)
    conn.commit()


def fetch_season(code: str, fdcode: str, db_season: str, id_league: int) -> pd.DataFrame | None:
    url = f"{BASE_URL}/{fdcode}/{code}.csv"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None
        df = pd.read_csv(io.StringIO(r.text), encoding="utf-8-sig", on_bad_lines="skip")
        return df
    except Exception as e:
        print(f"  Erreur {url}: {e}")
        return None


def import_season(conn: sqlite3.Connection, df: pd.DataFrame,
                  id_league: int, db_season: str) -> int:
    inserted = 0
    for _, row in df.iterrows():
        date = parse_date(row.get("Date"))
        home = normalize_team(str(row.get("HomeTeam", "") or "").strip())
        away = normalize_team(str(row.get("AwayTeam", "") or "").strip())
        if not date or not home or not away:
            continue

        hs  = int(row["HS"])  if "HS"  in df.columns and not pd.isna(row.get("HS"))  else None
        as_ = int(row["AS"])  if "AS"  in df.columns and not pd.isna(row.get("AS"))  else None
        hst = int(row["HST"]) if "HST" in df.columns and not pd.isna(row.get("HST")) else None
        ast = int(row["AST"]) if "AST" in df.columns and not pd.isna(row.get("AST")) else None

        if hst is None and ast is None:
            continue

        try:
            conn.execute("""
                INSERT OR IGNORE INTO shots_data
                    (idLeague, season, match_date, home_team, away_team,
                     home_shots, away_shots, home_sot, away_sot)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (id_league, db_season, date, home, away, hs, as_, hst, ast))
            if conn.execute("SELECT changes()").fetchone()[0]:
                inserted += 1
        except sqlite3.IntegrityError:
            pass

    conn.commit()
    return inserted


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    ensure_table(conn)

    print("=== IMPORT SHOTS DATA (football-data.co.uk) ===\n")
    total = 0

    for code, league_info in LEAGUES.items():
        league_total = 0
        for fdcode, db_season in SEASONS:
            df = fetch_season(code, fdcode, db_season, league_info["idLeague"])
            if df is None:
                continue
            n = import_season(conn, df, league_info["idLeague"], db_season)
            if n > 0:
                print(f"  {league_info['name']} {db_season}: +{n} matchs")
            league_total += n
            time.sleep(0.2)
        print(f"  → {league_info['name']}: {league_total} matchs au total\n")
        total += league_total

    conn.close()
    print(f"Total : {total} matchs avec données de tirs importés.")


if __name__ == "__main__":
    main()
