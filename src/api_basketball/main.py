# src/api_basketball/main.py
from flask import Flask, jsonify
from flask_cors import CORS
from nba_api.stats.endpoints import playercareerstats, playergamelog

app = Flask(__name__)
CORS(app)  # allow requests from your index.html

# ---------- 1. Career-totals endpoint ----------
@app.route("/api/jokic")
def jokic_career():
    career = playercareerstats.PlayerCareerStats(player_id="203999")
    df = career.get_data_frames()[0]
    return jsonify(df.to_dict(orient="records"))

# ---------- 2. Game-by-game endpoint ----------
@app.route("/api/jokic/games/<season>")
def jokic_games(season):
    glog = playergamelog.PlayerGameLog(
        player_id="203999",
        season=season,                   # e.g. "2024-25"
        season_type_all_star="Regular Season"
    )
    df = glog.get_data_frames()[0]
    return jsonify(df.to_dict(orient="records"))

# ---------- Run the server ----------
if __name__ == "__main__":
    app.run(debug=True)      # prints “Running on http://127.0.0.1:5000”
