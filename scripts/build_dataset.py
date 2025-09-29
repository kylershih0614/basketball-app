# scripts/build_dataset.py
import os, time, pandas as pd
from nba_api.stats.endpoints import teamgamelogs   # plural
from nba_api.stats.static import teams as teams_static

OUT = "src/api_basketball/data/raw/games.csv"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

SEASONS = ["2020-21","2021-22","2022-23","2023-24","2024-25"]

# --- Add this: canonical set of NBA team IDs ---
NBA_TEAMS = teams_static.get_teams()
NBA_TEAM_IDS = {t["id"] for t in NBA_TEAMS}

frames = []
for season in SEASONS:
    # --- Constrain to NBA + Regular Season ---
    tgl = teamgamelogs.TeamGameLogs(
        season_nullable=season,
        season_type_nullable="Regular Season",
        league_id_nullable="00"   # NBA only
    )
    df = tgl.get_data_frames()[0]
    df["SEASON"] = season

    # --- Filter to official NBA teams only (just in case) ---
    if "TEAM_ID" in df.columns:
        df = df[df["TEAM_ID"].isin(NBA_TEAM_IDS)]

    # If the frame has opponent ids, filter them too
    for col in ("OPPONENT_TEAM_ID", "VS_TEAM_ID", "TEAM_ID_OPP"):
        if col in df.columns:
            df = df[df[col].isin(NBA_TEAM_IDS)]

    frames.append(df)
    time.sleep(1.0)  # gentle on rate limits

raw = pd.concat(frames, ignore_index=True)

# Optional: keep only columns you actually use later
# cols_keep = ["GAME_ID","GAME_DATE","TEAM_ID","TEAM_NAME","MATCHUP","WL","PTS","FG3M","REB","SEASON"]
# raw = raw[cols_keep]

raw.to_csv(OUT, index=False)
print(f"Wrote {OUT} with {len(raw)} rows")
