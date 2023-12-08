from nba_api.stats.endpoints import *
from nba_api.stats.static import *
import pandas as pd
from tqdm import tqdm
import time

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from datetime import datetime

def get_player_data():
    start = datetime.now()
    # get active players from NBA API
    nba_players = players.get_active_players()
    active_players_id=[]
    for dict in nba_players:
        active_players_id.append(dict['id'])
        
    # if players_df file does not exist then pull the data from the API and store it. This file stores the player data like ID, Name, Team code, etc
    players_df = pd.DataFrame()

    all_players = commonallplayers.CommonAllPlayers().get_data_frames()[0]
    players_df = pd.DataFrame(all_players)
    players_df = players_df[players_df['PERSON_ID'].isin(active_players_id)]
    # players_df.to_csv('players_df.csv', index=False, mode='w')

    # players_df = pd.read_csv('players_df.csv')

    # if career_stats file does not exist then pull the data from the API and store it.
    # This file stores the career data of the player like age, gp, gs, fg_pct etc which are used as parameters of the model
    print("Gathering Player Stats...")
    career_stats = pd.DataFrame()
    error_log=[]
    for id in tqdm(active_players_id):
        try:
            time.sleep(0.1)
            career = playercareerstats.PlayerCareerStats(player_id=id)
            player_career = career.get_data_frames()[0]
            career_stats = pd.concat([career_stats,player_career],axis=0,ignore_index=True)
        except:
            error_log.append(id)
    # career_stats.to_csv('career_stats.csv', index=False, mode='w')

    # career_stats = pd.read_csv('career_stats.csv')

    # if players_awards file does not exist then pull the data from the API and store it. This file stores the awards that players won like Most valuable player, player of the month, etc
    print()
    print("Gathering Player Awards...")
    players_awards = pd.DataFrame()
    error_awards = []
    for ID in tqdm(active_players_id):
        try:
            time.sleep(0.1)
            award = playerawards.PlayerAwards(player_id=ID)
            players_awards = pd.concat([players_awards,award.get_data_frames()[0]],axis=0,ignore_index=True)
        except:
            error_awards.append(ID)
            
    # players_awards.to_csv('players_awards.csv',index=False, mode='w')
    # players_awards = pd.read_csv('players_awards.csv')


    # if team_stats file does not exist then pull the data from the API and store it. This file stores the team stats like wins, loses, win_pct, etc
    print()
    print("Gathering Team Stats...")
    nba_teams = teams.get_teams()
    teams_stats = pd.DataFrame()
    error_teams =[]
    for i in tqdm(nba_teams):
        try:
            time.sleep(0.1)
            team = teamyearbyyearstats.TeamYearByYearStats(team_id=i['id'])
            team_data = team.get_data_frames()[0]
            teams_stats = pd.concat([teams_stats,team_data],axis=0,ignore_index=True)
        except:
            error_teams.append(i)

    # teams_stats.to_csv('team_stats.csv',index=False, mode='w')

    # teams_stats = pd.read_csv('team_stats.csv')

    # Merge all the above dfs into one df
    teams_stats.rename(columns={'YEAR':'SEASON_ID'},inplace=True)
    players_teams = career_stats.merge(teams_stats,how='inner',on=['TEAM_ID','SEASON_ID'],suffixes=('_player','_team'))
    mvp = players_awards[players_awards['DESCRIPTION']=='NBA Most Valuable Player'].rename(columns={"PERSON_ID":"PLAYER_ID","SEASON":"SEASON_ID","TYPE":"MVP"})
    df= players_teams.merge(mvp.loc[:,["PLAYER_ID","SEASON_ID","MVP"]],how="left",on=["PLAYER_ID","SEASON_ID"])
    df = pd.merge(df, players_df[['PERSON_ID', 'DISPLAY_FIRST_LAST']], left_on='PLAYER_ID', right_on='PERSON_ID', how='left')

    # Clean data
    df['SEASON_ID'] = df['SEASON_ID'].map(lambda x: int(x.split("-",1)[0]))
    df.drop(columns=["LEAGUE_ID","TEAM_ID","TEAM_ABBREVIATION","TEAM_CITY","TEAM_NAME","NBA_FINALS_APPEARANCE","PERSON_ID","MVP","DIV_COUNT"],inplace=True)
    df.rename(columns={'PTS_player':'MVP'},inplace=True)


    # Reorder columns of the df so that MVP column is at the end
    column_order = [col for col in df.columns if col != 'MVP']
    column_order.append('MVP')
    df = df[column_order]
    df.to_csv('active_players.csv',index=False, mode='w')
    print()
    print("Data download complete.")
    print()
    end = datetime.now()
    # print(start,end,end-start)