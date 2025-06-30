print(">>> main.py actually ran")

from nba_api.stats.endpoints import playercareerstats

# Example: Nikola Jokić's career stats
career = playercareerstats.PlayerCareerStats(player_id='203999')

# Option 1: Show raw dictionary
print(career.get_dict())

# Option 2: Show as DataFrame (requires pandas)
df = career.get_data_frames()[0]
print(df)
