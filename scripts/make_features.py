import os, pandas as pd, numpy as np

RAW = "src/api_basketball/data/raw/games.csv"
OUT = "src/api_basketball/data/processed/game_features.csv"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

df = pd.read_csv(RAW)

# Home/away from MATCHUP, e.g., "LAL vs. BOS" (home) or "LAL @ BOS" (away)
df["IS_HOME"] = df["MATCHUP"].astype(str).str.contains(" vs. ")
# Label: home win if WL == 'W' on a home game row
df["HOME_WIN"] = ((df["WL"] == "W") & df["IS_HOME"]).astype(int)

# Sort for proper rolling windows
df = df.sort_values(["TEAM_ID","GAME_DATE"])
for col in ["PTS","FG3M","REB"]:
    if col in df.columns:
        df[f"{col}_rolling5"]  = df.groupby("TEAM_ID")[col].transform(lambda s: s.shift(1).rolling(5).mean())
        df[f"{col}_rolling10"] = df.groupby("TEAM_ID")[col].transform(lambda s: s.shift(1).rolling(10).mean())

feat_cols = [c for c in df.columns if "rolling" in c]
df = df.dropna(subset=feat_cols + ["HOME_WIN"])
df[["GAME_ID","TEAM_ID","HOME_WIN"] + feat_cols].to_csv(OUT, index=False)
print(f"Wrote {OUT}")
