from nba_api.stats.endpoints import *
from nba_api.stats.static import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from selenium import webdriver
import os
import download_active_players_data
import datetime

import warnings
warnings.simplefilter(action='ignore')

active_players_file = 'active_players.csv'

if not os.path.isfile(active_players_file):
    print("Required data was not found locally. Downloading data (Estimated time: 8 mins)...")
    print()
    download_active_players_data.download_data()
else:
    timestamp = os.path.getmtime(active_players_file)
    dt_object = datetime.datetime.fromtimestamp(timestamp)
    formatted_dt_tm = dt_object.strftime('%d-%b-%Y %H:%M:%S')
    print("Required data was found locally, last modified: "+formatted_dt_tm+". \nWould you like to download fresh data?(Y/N) (Estimated time: 8 mins)")
    downloadData = input()
    print()
    if downloadData in ('Y','y','yes','Yes','YES'):
        print("Downloading data...")
        print()
        download_active_players_data.download_data()
    else:
        print("Using existing data...")
        print()
    
print("Analysing data...")
print()
df = pd.read_csv(active_players_file)
original_df = df.copy()
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
top_10_players = mvp_candidates['full_name'].head(10).tolist()
# for p in top_10_players:
#     print(p)
print(mvp_candidates.head(25))