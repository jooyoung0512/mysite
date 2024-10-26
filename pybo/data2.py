import pandas as pd
import time
df = pd.read_csv("/content/drive/MyDrive/player_career.csv")
from nba_api.stats.static import players
# get_players returns a list of dictionaries, each representing a player.
nba_players = players.get_players()
print("Number of players fetched: {}".format(len(nba_players)))
df0 = pd.DataFrame(nba_players)
df = df.rename(columns={'PLAYER_ID':'id'})
pd.merge(df,df0,on='id')
salary_df = pd.read_csv("/content/drive/MyDrive/Nba Player Salaries.csv")
df = df.rename(columns={"full_name": "Player Name"})