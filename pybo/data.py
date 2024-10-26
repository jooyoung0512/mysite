import pandas as pd
import time
from nba_api.stats.endpoints import playercareerstats
import pandas as pd
from nba_api.stats.static import teams,players

# Nikola Jokić
career = playercareerstats.PlayerCareerStats(player_id='203999')

# pandas data frames (optional: pip install pandas)
df = career.get_data_frames()[0]


# get_teams returns a list of 30 dictionaries, each an NBA team.
nba_teams = teams.get_teams()
print("Number of teams fetched: {}".format(len(nba_teams)))
nba_teams[:30]

# get_players returns a list of dictionaries, each representing a player.
nba_players = players.get_players()
print("Number of players fetched: {}".format(len(nba_players)))
nba_players[:600]
spurs = [team for team in nba_teams if team["full_name"] == "San Antonio Spurs"][0]
big_fundamental = [
    player for player in nba_players if player["full_name"] == "Tim Duncan"
][0]
from nba_api.stats.static import players

# get_players returns a list of dictionaries, each representing a player.
nba_players = players.get_players()
print("Number of players fetched: {}".format(len(nba_players)))
df0 = pd.DataFrame(nba_players)
df0.rename(columns={"id": "PLAYER_ID"}, inplace=True)
df0['PLAYER_ID']
df3 = df

for id in df0['PLAYER_ID']:
  print(id)
  career = playercareerstats.PlayerCareerStats(player_id=id)

  df2 = career.get_data_frames()[0]

  df3 = pd.concat([df3,df2])

  print(id)

  time.sleep(1.5)
  df3.to_csv('player_career.csv', index=True, header=True)