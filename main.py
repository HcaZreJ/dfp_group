# Group Member Andrew Ids:
# - Aditya Kolpe: akolpe
# - Zichen Zhu: zichenzh
# - Sophie Golunova: sgolunov
# - Brianna Dincau: bdincau
# - Emily Harvey: eharvey2

# This is the main function of the program.
# You only need to run this file.

print("\nImporting necessary packages for the program (Estimated time: 10 seconds)...")
from source1_nbaAPI.nbaAPI_scrape import get_player_data
from source2_injuryData.injury_scrape import get_injury_data
from source3_espn.espn_scrape import get_headlines
from nba_api.stats.static import players
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import os
import datetime
import warnings
warnings.simplefilter(action='ignore')

################################# Section 1: ESPN Headlines #################################
# Print notification messages and scrape headline data from ESPN
print("\nAttempting to scrape headlines from ESPN\n")
print("A chrome browser window will pop up,\nplease do not do anything to it,\nit will close itself once all javascripts are loaded.\n")
time.sleep(2)
headlines = get_headlines()

# Print the most recent headlines to the user
print("Here are the 5 most recent NBA headlines in case you missed them:", "\n")
for i, headline in zip(range(5), headlines[:5]):
    print(f"Headline {i+1}: {headline}")
print()

################################# Section 2: Player Injury Data #################################
# Specify where the injury data is stored
injury_file = "source2_injuryData/injury_df.csv"

# If there is no readily saved data, then new data must be scraped
if not os.path.isfile(injury_file):
    print("Required player injury data was not found locally. Downloading data (Estimated time: immediate)...")
    print()
    get_injury_data()
# Else we ask the user if they would like to use saved data or download new data
else:
    # Obtain last modified time of file and inquire user input
    timestamp = os.path.getmtime(injury_file)
    dt_object = datetime.datetime.fromtimestamp(timestamp)
    formatted_dt_tm = dt_object.strftime('%d-%b-%Y %H:%M:%S')
    print("Required player injury data was found locally, last modified: "+formatted_dt_tm+". \nWould you like to download fresh data?(Y/N) (Estimated time: immediate)")
    downloadData = input()
    print()
    # Download fresh data
    if downloadData in ('Y','y','yes','Yes','YES'):
        print("Downloading data...")
        print()
        get_injury_data()
    else:
        print("Using existing data...")
        print()

# Read injury data
injury_data = pd.read_csv(injury_file, header=0, index_col=0)

################################# Section 3: Player Career Data #################################
# Specify file path
active_players_file = 'source1_nbaAPI/active_players.csv'

# Refer to section 2 for the downloading logic, this part does the same thing
if not os.path.isfile(active_players_file):
    print("Required player career data was not found locally. Downloading data (Estimated time: 8 mins)...")
    print()
    get_player_data()
else:
    timestamp = os.path.getmtime(active_players_file)
    dt_object = datetime.datetime.fromtimestamp(timestamp)
    formatted_dt_tm = dt_object.strftime('%d-%b-%Y %H:%M:%S')
    print("Required player career data was found locally, last modified: "+formatted_dt_tm+". \nWould you like to download fresh data?(Y/N) (Estimated time: 8 mins)")
    downloadData = input()
    print()
    if downloadData in ('Y','y','yes','Yes','YES'):
        print("Downloading data...")
        print()
        get_player_data()
    else:
        print("Using existing data...")
        print()

# Read career data
df = pd.read_csv(active_players_file, header=0)

# Make a copy so we still have the original data
original_df = df.copy()

################################# Section 4: Data Analysis #################################
# Status message
print("Analysing data...")
print()

# Drop the column of player names because model can only take numeric data
df.drop(columns=['DISPLAY_FIRST_LAST'],inplace=True)
num_cols = df.columns.to_list()[2:]
# Transform data so that different variables contribute equally to the analysis
scaler = MinMaxScaler()
for year in  df["SEASON_ID"].unique().tolist():
    for col in num_cols:
        df.loc[df['SEASON_ID']==year, col] = scaler.fit_transform(df.loc[df['SEASON_ID']==year, col].to_numpy().reshape(-1,1))
  
# Pick best players
df['MVP'] = df['MVP'].apply(lambda x: 1 if x > 0.75 else 0)

# Train on players from 2000 to 2021 and test on 2022 to 2023
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
# plt.show()

print("Picking relevant parameters to train the model...")
print()
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

print("Training the model (Estimated time: 1 min)...")
print()
model = RandomForestClassifier(random_state=42,class_weight="balanced",max_depth=5,criterion='entropy',n_estimators=500)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)

# print("=====cv score=====")
# print("roc_auc_avg:{:.3f}".format(np.mean(scores)))
# print("roc_auc_std:{:.3f}".format(np.std(scores)))

model.fit(X,y)
print("Predicting the best players...")
print()
# Predict
result = pd.DataFrame(model.predict_proba(X_test),index = X_test.PLAYER_ID)
# Clean the output dataframe of the model
result.rename(columns={0:'not',1:'mvp'},inplace=True)
result.drop(columns=['not'],inplace=True)
mvp_candidates = result.sort_values(by='mvp',ascending=False)
mvp_candidates.reset_index(inplace=True)

# Gather a dataframe for age and another dataframe for games injured for the final output
age = original_df[["PLAYER_ID", "PLAYER_AGE"]]
games_injured = injury_data[["Player", "Games"]]

# Get players and merge based on playerID to display names because result df contains IDs
# Get a static file from NBAapi that match Player ID to their names
nba_players = players.get_players()
nba_players = pd.DataFrame(nba_players)
# Rename its id column so it's consistent for the subsequent merge
nba_players.rename(columns={"id":"PLAYER_ID"},inplace = True)

# Merge mvp predictions with player names
mvp_candidates = mvp_candidates.merge(nba_players.loc[:,['PLAYER_ID','full_name']],on='PLAYER_ID',how='left')
# Merge mvp predictions with their age
mvp_candidates = mvp_candidates.merge(age, on='PLAYER_ID',how='left')
# Merge mvp predictions with games they are injured for
mvp_candidates = mvp_candidates.merge(games_injured, left_on='full_name', right_on='Player', how='left')

# Inquire user input for whether they like a younger or older team
print("Would you like to draft a younger team (which has higher potential in the future)")
print("or an older team (which is more experienced)? Input: y for younger, o for older")
# Ensure input is valid
yoinput = input()
while yoinput not in ('y', 'o'):
    print("Your input is not valid. Please try again. Valid Inputs: y for younger, o for older")
    yoinput = input()
print()

# Drop older/younger players depending on user input
if yoinput == 'y':
    mvp_candidates = mvp_candidates[mvp_candidates['PLAYER_AGE'] < 30]
else:
    mvp_candidates = mvp_candidates[mvp_candidates['PLAYER_AGE'] >= 30]

# Group by playerID & name, calculate average mvp chance for players, then order ascending
mvp_candidates = mvp_candidates.groupby(['PLAYER_ID', 'full_name'], as_index=False).agg(
    {'mvp': 'mean',
     'PLAYER_AGE': 'first',
     'Games': 'first'})
mvp_candidates = mvp_candidates.sort_values(by='mvp',ascending=False)

# Format & Print final analysis output to user
# Change column names
mvp_candidates.rename(columns={'full_name':'Player Name',
                               'Games':'Games Injured',
                               'PLAYER_AGE':'Age'},inplace=True)
# Drop PLAYER_ID column
mvp_candidates.drop("PLAYER_ID", axis=1, inplace=True)
# Change Age & Games Injured (0 if not injured) to integer values
mvp_candidates['Age'] = mvp_candidates['Age'].astype(int)
mvp_candidates['Games Injured'] = mvp_candidates['Games Injured'].fillna(0).astype(int)
# Print output to user
print("Based on your selections, here is the dream team that you should draft:")
print()
print(mvp_candidates.head(7).to_string(index=False))
print()
print("Note that some players may be injured for a few games this season.")
print("Keep that in mind when you draft your dream team.")