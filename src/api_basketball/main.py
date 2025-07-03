# src/api_basketball/main.py
from flask import Flask, jsonify, request
from flask_cors import CORS

from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players as players_static   # NEW

app = Flask(__name__)
CORS(app)   # allow requests from http://localhost:*

# ---------- helper ----------
def full_name_to_id(full_name: str) -> int | None:
    """
    Return the NBA.com player_id that matches `full_name`.
    • First try an exact (case-insensitive) match.
    • Otherwise fall back to the first partial match NBA-API returns.
    """
    matches = players_static.find_players_by_full_name(full_name)
    if not matches:
        return None

    for m in matches:
        if m["full_name"].lower() == full_name.lower():
            return m["id"]
    return matches[0]["id"]        # best-effort fallback


# ---------- generic game-log endpoint ----------
@app.route("/api/player/games")
def player_games():
    """
    GET /api/player/games?name=LeBron%20James&season=2024-25
    • name   (required): Full player name
    • season (optional): "YYYY-YY"; defaults to current NBA season
    """
    name   = request.args.get("name", "").strip()
    season = request.args.get("season", "2024-25")   # default if none supplied

    if not name:
        return jsonify({"error": "Missing `name` query param"}), 400

    player_id = full_name_to_id(name)
    if not player_id:
        return jsonify({"error": f"No NBA player found for '{name}'"}), 404

    glog = playergamelog.PlayerGameLog(
        player_id           = player_id,
        season              = season,
        season_type_all_star= "Regular Season",
    )
    df = glog.get_data_frames()[0]
    return jsonify(df.to_dict(orient="records"))


# ---------- run ----------
if __name__ == "__main__":
    app.run(debug=True)
