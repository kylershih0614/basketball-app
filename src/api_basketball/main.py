# src/api_basketball/main.py
from flask import Flask, jsonify, request
from flask_cors import CORS
import os, json, joblib, numpy as np, pandas as pd
from datetime import datetime
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players as players_static
from nba_api.stats.static import teams as teams_static  # NEW

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # handles OPTIONS automatically

# ---------- helpers ----------
def full_name_to_id(full_name: str):
    matches = players_static.find_players_by_full_name(full_name or "")
    if not matches:
        return None
    for m in matches:
        if m["full_name"].lower() == (full_name or "").lower():
            return m["id"]
    return matches[0]["id"]

# Build team name <-> id maps once
_ALL_TEAMS = teams_static.get_teams()
TEAM_NAME_TO_ID = {t["full_name"]: t["id"] for t in _ALL_TEAMS}
TEAM_ID_TO_NAME = {t["id"]: t["full_name"] for t in _ALL_TEAMS}

# ---------- load artifacts & data BEFORE routes ----------
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models", "game_outcome")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
FEAT_PATH  = os.path.join(MODEL_DIR, "features.json")
PROC_PATH  = os.path.join(BASE_DIR, "data", "processed", "game_features.csv")

_model = joblib.load(MODEL_PATH)
_feat  = json.load(open(FEAT_PATH))["features"]
_proc  = pd.read_csv(PROC_PATH)

# Determine whether processed data has TEAM_NAME; if not, add it from TEAM_ID
if "TEAM_ID" in _proc.columns and "TEAM_NAME" not in _proc.columns:
    _proc["TEAM_NAME"] = _proc["TEAM_ID"].map(TEAM_ID_TO_NAME)

TEAM_COL = "TEAM_NAME" if "TEAM_NAME" in _proc.columns else ("TEAM_ID" if "TEAM_ID" in _proc.columns else None)

# Normalize GAME_DATE
if "GAME_DATE" in _proc.columns:
    with pd.option_context("mode.chained_assignment", None):
        try:
            _proc["GAME_DATE"] = pd.to_datetime(_proc["GAME_DATE"])
        except Exception:
            pass

def latest_row_for_team(team_value, use_names=True, before=None):
    """
    Return the latest pre-game row for a team.
    - If use_names=True, team_value is a full team name; we map to TEAM_ID if needed.
    - If use_names=False, team_value is a TEAM_ID int.
    """
    if TEAM_COL is None:
        return None

    df = _proc
    if use_names:
        # Map name -> id if underlying column is TEAM_ID
        if TEAM_COL == "TEAM_ID":
            team_id = TEAM_NAME_TO_ID.get(team_value)
            if team_id is None:
                return None
            df = df[df["TEAM_ID"] == team_id]
        else:
            df = df[df["TEAM_NAME"] == team_value]
    else:
        df = df[df[TEAM_COL] == team_value]

    if "GAME_DATE" in df.columns:
        if before:
            try:
                cutoff = pd.to_datetime(before)
                df = df[df["GAME_DATE"] < cutoff]
            except Exception:
                pass
        df = df.sort_values("GAME_DATE")

    if df.empty:
        return None
    return df.iloc[-1]

# ---------- routes ----------

# Generic player game-log
@app.get("/api/player/games")
def player_games():
    """
    GET /api/player/games?name=LeBron%20James&season=2024-25
    """
    name   = request.args.get("name", "").strip()
    season = request.args.get("season", "2024-25")
    if not name:
        return jsonify({"error": "Missing `name` query param"}), 400

    player_id = full_name_to_id(name)
    if not player_id:
        return jsonify({"error": f"No NBA player found for '{name}'"}), 404

    glog = playergamelog.PlayerGameLog(
        player_id=player_id,
        season=season,
        season_type_all_star="Regular Season",
    )
    df = glog.get_data_frames()[0]
    return jsonify(df.to_dict(orient="records"))

# Teams for UI dropdown (always names)
@app.get("/api/teams")
def list_teams():
    teams = sorted({t["full_name"] for t in _ALL_TEAMS})
    return jsonify({"teams": teams})

# Predict matchup using the CURRENT single-team model (home-only features)
@app.post("/api/predict/matchup")
def predict_matchup():
    """
    JSON body:
    {
      "home_team": "Los Angeles Lakers",
      "away_team": "Boston Celtics",
      "date": "2025-01-15"  # optional
    }
    Uses the home team's latest pre-game feature row, matching _feat order.
    """
    payload = request.get_json(force=True) or {}
    home_team = payload.get("home_team")
    away_team = payload.get("away_team")
    game_date = payload.get("date")  # optional

    if not home_team or not away_team:
        return jsonify({"error": "Provide 'home_team' and 'away_team'"}), 400

    home_row = latest_row_for_team(home_team, use_names=True, before=game_date)
    if home_row is None:
        return jsonify({"error": f"No processed features found for home team '{home_team}'"}), 404

    # Build feature vector in the exact order expected by the model
    row_vals, missing = [], []
    for k in _feat:
        v = home_row.get(k, None)
        if pd.isna(v) or v is None:
            missing.append(k)
        row_vals.append(v)

    if missing:
        return jsonify({"error": f"Home team row is missing features: {missing}"}), 500

    proba = float(_model.predict_proba(np.array([row_vals]))[0, 1])
    return jsonify({
        "home_team": home_team,
        "away_team": away_team,
        "date": game_date,
        "home_win_prob": proba,
        "features_used": _feat
    })

# ---------- run LAST ----------
if __name__ == "__main__":
    print("Registered routes:\n", app.url_map)
    app.run(host="127.0.0.1", port=5000, debug=True)
