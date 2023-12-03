from nba_api.stats.endpoints import *
from nba_api.stats.static import *
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import os

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# get active players from NBA API
nba_players = players.get_active_players()
active_players_id=[]
for dict in nba_players:
    active_players_id.append(dict['id'])
    
# if players_df file does not exist then pull the data from the API and store it. This file stores the player data like ID, Name, Team code, etc
players_df = pd.DataFrame()
if not os.path.isfile('players_df.csv'):
    print("file not found")
    all_players = commonallplayers.CommonAllPlayers().get_data_frames()[0]
    players_df = pd.DataFrame(all_players)
    players_df = players_df[players_df['PERSON_ID'].isin(active_players_id)]
    players_df.to_csv('players_df.csv', index=False)

players_df = pd.read_csv('players_df.csv')

# if career_stats file does not exist then pull the data from the API and store it.
# This file stores the career data of the player like age, gp, gs, fg_pct etc which are used as parameters of the model
career_stats = pd.DataFrame()
error_log=[]
if not os.path.isfile('career_stats.csv'):
    for id in active_players_id:
        try:
            time.sleep(0.1)
            career = playercareerstats.PlayerCareerStats(player_id=id)
            player_career = career.get_data_frames()[0]
            career_stats = pd.concat([career_stats,player_career],axis=0,ignore_index=True)
        except:
            error_log.append(id)
    career_stats.to_csv('career_stats.csv', index=False)

career_stats = pd.read_csv('career_stats.csv')

# if players_awards file does not exist then pull the data from the API and store it. This file stores the awards that players won like Most valuable player, player of the month, etc
players_awards = pd.DataFrame()
error_awards = []
if not os.path.isfile('players_awards.csv'):
    for ID in active_players_id:
        try:
            time.sleep(0.1)
            award = playerawards.PlayerAwards(player_id=ID)
            players_awards = pd.concat([players_awards,award.get_data_frames()[0]],axis=0,ignore_index=True)
        except:
            error_awards.append(ID)
            
    players_awards.to_csv('players_awards.csv',index=False)
players_awards = pd.read_csv('players_awards.csv')


# if team_stats file does not exist then pull the data from the API and store it. This file stores the team stats like wins, loses, win_pct, etc
nba_teams = teams.get_teams()
teams_stats = pd.DataFrame()
error_teams =[]
if not os.path.isfile('team_stats.csv'):
    for i in tqdm(nba_teams):
        try:
            time.sleep(0.1)
            team = teamyearbyyearstats.TeamYearByYearStats(team_id=i['id'])
            team_data = team.get_data_frames()[0]
            teams_stats = pd.concat([teams_stats,team_data],axis=0,ignore_index=True)
        except:
            error_teams.append(i)

    teams_stats.to_csv('team_stats.csv',index=False)

teams_stats = pd.read_csv('team_stats.csv')

# Merge all the above dfs into one df
teams_stats.rename(columns={'YEAR':'SEASON_ID'},inplace=True)
players_teams = career_stats.merge(teams_stats,how='inner',on=['TEAM_ID','SEASON_ID'],suffixes=('_player','_team'))
mvp = players_awards[players_awards['DESCRIPTION']=='NBA Most Valuable Player'].rename(columns={"PERSON_ID":"PLAYER_ID","SEASON":"SEASON_ID","TYPE":"MVP"})
df= players_teams.merge(mvp.loc[:,["PLAYER_ID","SEASON_ID","MVP"]],how="left",on=["PLAYER_ID","SEASON_ID"])
df = pd.merge(df, players_df[['PERSON_ID', 'DISPLAY_FIRST_LAST']], left_on='PLAYER_ID', right_on='PERSON_ID', how='left')

# Clean data
df['SEASON_ID'] = df['SEASON_ID'].map(lambda x: int(x.split("-",1)[0]))
df.drop(columns=["LEAGUE_ID","TEAM_ID","TEAM_ABBREVIATION","TEAM_CITY","TEAM_NAME","NBA_FINALS_APPEARANCE","DISPLAY_FIRST_LAST","PERSON_ID","MVP","DIV_COUNT"],inplace=True)
df.rename(columns={'PTS_player':'MVP'},inplace=True)


num_cols = df.columns.to_list()[2:]
# Reorder columns of the df so that MVP column is at the end
column_order = [col for col in df.columns if col != 'MVP']
column_order.append('MVP')
df = df[column_order]
# Make a copy of original data, might be required later
df_og = df.copy()

# Transform data so that different variables contribute equally to the analysis
scaler = MinMaxScaler()
for year in  df["SEASON_ID"].unique().tolist():
    for col in num_cols:
        df.loc[df['SEASON_ID']==year, col] = scaler.fit_transform(df.loc[df['SEASON_ID']==year, col].to_numpy().reshape(-1,1))
  
# Pick best players
df['MVP'] = df['MVP'].apply(lambda x: 1 if x > 0.75 else 0)

# Train on players from 2000 to 2021 and test on 2021 to 2023
train = df.loc[(df['SEASON_ID']>=2000)&(df['SEASON_ID']<=2023),:]
test = df.loc[(df['SEASON_ID']>=2021)&(df['SEASON_ID']<=2023),:]
y = train['MVP']
X = train.iloc[:,:-1]
y_test = test['MVP']
X_test = test.iloc[:,:-1]

corrmat = df.corr()

# Show corr between parameters
k = 65 # show top k most correlated features
cols = corrmat.nlargest(k, 'MVP')['MVP'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.0)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 8}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

# Drop highly correlted columns
X_corr = X.corr()
corr_names = set()
for i in range(len(X_corr .columns)):
    for j in range(i):
        if abs(X_corr.iloc[i, j]) > 0.8:
            col = X_corr.columns[i]
            corr_names.add(col)

X.drop(columns=corr_names,inplace=True)
X_test.drop(columns=corr_names,inplace=True)

# Classifier model to predict a list of best players to choose from
from sklearn.model_selection import *
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42,class_weight="balanced",max_depth=5,criterion='entropy',n_estimators=500)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)

# print("=====cv score=====")
# print("roc_auc_avg:{:.3f}".format(np.mean(scores)))
# print("roc_auc_std:{:.3f}".format(np.std(scores)))

model.fit(X,y)
# Predict
result = pd.DataFrame(model.predict_proba(X_test),index = X_test.PLAYER_ID)

result.rename(columns={0:'not',1:'mvp'},inplace=True)
result.drop(columns=['not'],inplace=True)
mvp_candidates = result.sort_values(by='mvp',ascending=False)
mvp_candidates.reset_index(inplace=True)

# Get players and merge based on playerID to display names because result df contains IDs
nba_players = players.get_players()
nba_players = pd.DataFrame(nba_players)
nba_players.rename(columns={"id":"PLAYER_ID"},inplace = True)
mvp_candidates = mvp_candidates.merge(nba_players.loc[:,['PLAYER_ID','full_name']],on='PLAYER_ID',how='left')
mvp_candidates = mvp_candidates.groupby(['PLAYER_ID', 'full_name'], as_index=False)['mvp'].mean()
mvp_candidates = mvp_candidates.sort_values(by='mvp',ascending=False)
print("***************************************************************************************************************************************")
print("***************************************************************************************************************************************")
print("***************************************************************************************************************************************")
print(mvp_candidates.to_string())