import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import pickle 

from data_processing.utils import joinnames

##### Data Cleaning #####

#Load Data. Fill in NaNs and eleminate Draws, DQs, and NCs
file_path = "../data/raw_data.json"
alldata = pd.read_json(file_path)
alldata = alldata[alldata['winner'] != 'draw']
alldata = alldata[(alldata['win_method'] != 'Overturned') & (alldata['win_method'] != 'DQ')]
alldata = alldata.drop('time', axis=1)
alldata = alldata.replace(['-', '--', '---'], 0)
fightdata = alldata.sort_values('date').reset_index(drop=True).copy()
fightdata = fightdata.set_index('date', drop=True)


def split_of(data, column):
    """Split of columns with format {landed} of {attempted}"""
    data[column+'_landed'] = data[column].apply(lambda x: int(x.split('of')[0]))
    data[column+'_attempted'] = data[column].apply(lambda x: int(x.split('of')[1]))
    
of_cols = ['f1_total_str', 'f1_td', 'f1_sig_str', 'f1_head', 'f1_body', 'f1_leg', 'f1_distance', 'f1_clinch', 'f1_ground', 'f2_total_str', 'f2_td', 'f2_sig_str', 'f2_head', 'f2_body', 'f2_leg', 'f2_distance', 'f2_clinch', 'f2_ground']
[split_of(fightdata, col) for col in of_cols]
[fightdata.drop(col, axis=1, inplace=True) for col in of_cols]

#Convert column data types to float. Handle percentages and times
for col in fightdata.columns:
    if 'pct' in col:
        fightdata[col] = fightdata[col].apply(lambda x: float(x.strip('%')) if type(x)==str else float(x))

make_seconds = lambda x: int(x.split(':')[0])*60 + int(x.split(':')[1]) if type(x) is str else x
for col in ['f1_ctrl', 'f2_ctrl']:
    fightdata[col] = fightdata[col].apply(make_seconds)

for col in fightdata.columns[8:]:
    fightdata[col] = fightdata[col].astype('float64')

#Check for null values and save
if fightdata.isnull().sum().sum() == 0:
    fightdata.to_pickle("../data/processed_fightdata.pkl")


##### Feature Engineering #####

fightdata = pd.read_pickle("../data/processed_fightdata.pkl")
#Create %landed, %defended, and num absorbed stats for each type of stat 
pct = lambda x, y: (x/y) * 100 
pct_diff = lambda x, y: ((x - y)/ x) * 100 

fs1 = ['sig_str', 'td', 'head', 'body', 'leg']
for ftr in ['f1', 'f2']:
    oftr = 'f2' if ftr == 'f1' else 'f1' 
    for f in fs1:
        fightdata[joinnames(ftr, f, 'pct')] = pct(fightdata[joinnames(ftr, f, 'landed')], fightdata[joinnames(ftr, f, 'attempted')])
        fightdata[joinnames(ftr, f, 'absorbed')] = fightdata[joinnames(oftr, f, 'landed')]
        fightdata[joinnames(ftr, f, 'def')] = pct_diff(fightdata[joinnames(oftr, f, 'attempted')], fightdata[joinnames(oftr, f, 'landed')])

fs2 = ['distance', 'ground', 'clinch']
for ftr in ['f1', 'f2']:
    oftr = 'f2' if ftr == 'f1' else 'f1' 
    for f in fs2:
        fightdata[joinnames(ftr, f, 'pct')] = pct(fightdata[joinnames(ftr, f, 'landed')], fightdata[joinnames(ftr, 'sig_str', 'landed')])

fightdata = fightdata.fillna(0)

#Read features to include in fighterinfo table
with open("../data/fighterinfo_features.txt", 'r') as f:
    lines = f.readlines()
    per_fighter_feats = [l.strip() for l in lines]
    per_fighter_feats = [p for p in per_fighter_feats if p]

#Create series of unque fighters and unique dates
fighters = pd.concat([fightdata['f1'], fightdata['f2']]).unique()
dates = fightdata.index.unique()

#Initialize fighterinfo dataframe
fighterinfo_init = np.empty((len(dates), len(fighters)*len(per_fighter_feats)))
fighterinfo_init[:] = np.NaN
index = dates
columns = pd.MultiIndex.from_product([fighters, per_fighter_feats], names=['fighter', 'features'])
fighterinfo = pd.DataFrame(fighterinfo_init, index=index, columns=columns)

#Populate fighterinfo table by iterating through rows of fightdata 
avg = lambda x, y: (x+y)/2

def update_fight_feats(fighterinfo, row, date, ftrs, starts, ends):
    """Update the fight stats related features of the fighters from 'row' (a row from fightdata df which represents one fight) """
    for ftr in ftrs:
        ftr_idx, ftr_name = ftr[0], ftr[1]
        for start in starts:
            for end in ends:
                fi_end, fd_end = end[0], end[1]
                fi_feat, fd_feat = joinnames(start, fi_end), joinnames(ftr_idx, start, fd_end)
                prev_feat = fighterinfo.loc[date, (ftr_name, fi_feat)]
                prev_feat = 0 if pd.isnull(prev_feat) else prev_feat
                fighterinfo.loc[date, (ftr_name, fi_feat)] = avg(prev_feat, row[fd_feat])

def update_wins_losses_strks(fighterinfo, row, date, winner_name, loser_name):
    """Update the wins, losses, and streak features of the fighters from 'row' """
    #First update win, loss, and streak related feats
    #Update winner wins
    prev_wins = fighterinfo.loc[date, (winner_name, 'wins')] 
    prev_wins = 0 if pd.isnull(prev_wins) else prev_wins
    fighterinfo.loc[date, (winner_name, 'wins')] = prev_wins + 1
    #Update winner losses
    prev_losses = fighterinfo.loc[date, (winner_name, 'losses')] 
    fighterinfo.loc[date, (winner_name, 'losses')] = 0 if pd.isnull(prev_losses) else prev_losses
    #Update winner curr_win_strk
    prev_win_strk = fighterinfo.loc[date, (winner_name, 'curr_win_strk')]
    prev_win_strk = 0 if pd.isnull(prev_win_strk) else prev_win_strk
    fighterinfo.loc[date, (winner_name, 'curr_win_strk')] = prev_win_strk + 1
    #Update winner curr_lose_strk 
    fighterinfo.loc[date, (winner_name, 'curr_loss_strk')] = 0
    #Update loser wins
    prev_wins = fighterinfo.loc[date, (loser_name, 'wins')] 
    fighterinfo.loc[date, (loser_name, 'wins')] = 0 if pd.isnull(prev_wins) else prev_wins
    #Update loser losses
    prev_losses = fighterinfo.loc[date, (loser_name, 'losses')] 
    prev_losses = 0 if pd.isnull(prev_losses) else prev_losses
    fighterinfo.loc[date, (loser_name, 'losses')] = prev_losses + 1
    #Update loser curr_win_strk
    fighterinfo.loc[date, (loser_name, 'curr_win_strk')] = 0
    #Update loser curr_loss_strk
    prev_loss_strk = fighterinfo.loc[date, (loser_name, 'curr_loss_strk')]
    prev_loss_strk = 0 if pd.isnull(prev_loss_strk) else prev_loss_strk
    fighterinfo.loc[date, (loser_name, 'curr_loss_strk')] = prev_loss_strk + 1

    
#Main loop to iterate through fightdata 
prev_date = fighterinfo.index[0]
print("Populating fighterinfo dataframe...")
for date, row in tqdm(fightdata.iterrows()):
    
    winner = row['winner']
    winner_name = row[winner]
    loser = 'f2' if winner == 'f1' else 'f1'
    loser_name = row[loser]
    
    #Make current date stats same as prev date stats so they can be updated
    fighterinfo.loc[date] = fighterinfo.loc[prev_date]
    prev_date = date
    #First update wins, losses, and streaks
    update_wins_losses_strks(fighterinfo, row, date, winner_name, loser_name)
    
    #Create ftrs list for fight features update
    ftrs = [(winner, winner_name), (loser, loser_name)]
    
    #Update 5 end variant feats
    starts = ['sig_str', 'td', 'head', 'body', 'leg']
    ends = [('lnd_pf', 'landed'), ('att_pf', 'attempted'), ('acc_pct', 'pct'), ('abs_pf', 'absorbed'), ('def_pct', 'def')]
    
    update_fight_feats(fighterinfo, row, date, ftrs, starts, ends)
    
    #Update 2 end variant feats
    starts = ['distance', 'ground', 'clinch']
    ends = [('lnd_pf', 'landed'), ('acc_pct', 'pct')]
    
    update_fight_feats(fighterinfo, row, date, ftrs, starts, ends)
    
    #Update 1 end variant feats
    starts = ['sub', 'ctrl', 'rev']
    ends = [('lnd_pf', None)]
    
    update_fight_feats(fighterinfo, row, date, ftrs, starts, ends)  


#Save fighterinfo data
fighterinfo.to_pickle("../data/fighterinfo.pkl")



## Train and Test Set Generation ##

#Read data
fighterinfo = pd.read_pickle("../data/fighterinfo.pkl")
fightdata = pd.read_pickle("../data/processed_fightdata.pkl")

#Set fighterinfo index to datetime
fighterinfo.index = pd.to_datetime(fighterinfo.index)
#Remove first row of fightdata as we dont have data from dates before it to serve as features
fightdata = fightdata[fightdata.index != "1994-03-11"]

#Create dataset with fights, their outcomes, and the stats of fighters before their fight (to be used for training)
dates = fightdata.index
stats = fighterinfo.columns.get_level_values(1).unique()
init_columns = ["winner", "f1", "f2"] + [joinnames(fighter, stat) for fighter in ["f1", "f2"] for stat in stats]
data_init = np.empty((len(dates), len(init_columns)))
data_init[:] = np.NaN
data = pd.DataFrame(data_init, index=dates, columns=init_columns)

#Populate the dataset by iterating through fightdata and using fighterinfo to search for fighter's stats right before their fight 

#Util funcs
other = lambda x: "f1" if x == "f2" else "f2"
date_to_str = lambda date: date.strftime("%Y-%m-%d")
dates = list(date_to_str(fighterinfo.index))
prev_date = lambda date: dates[dates.index(date)-1]


def populate_fighters(data, fightdata, fighterinfo):
    print("Populating training/testing dataset...")
    prev_winner = "f2"
    for idx, (date, fightdata_row) in tqdm(enumerate(fightdata.iterrows())):        
        date = date_to_str(date)
        winner, f1, f2 = fightdata_row["winner"], fightdata_row["f1"], fightdata_row["f2"]
        
        #Switching of winners is done to ensure equal number of winners from f1 and f2 so indexing of features doesn't affect the model
        if winner == prev_winner:
            f1, f2 = f2, f1
            winner = other(winner)        
        prev_winner = winner
        
        loser = other(winner)
                
        data_row = data.iloc[idx].copy()
        
        data_row.loc[["winner", "f1", "f2"]] = [winner, f1, f2]
        
        for fighter in [winner, loser]:
            fighter_name = data_row.loc[fighter]
            fighter_stat_columns = [c for c in data.columns if fighter+"_" in c]
            fighter_stats = fighterinfo.loc[prev_date(date), fighter_name]
            data_row.loc[fighter_stat_columns] = list(fighter_stats) 
        
        data.iloc[idx] = data_row
        
populate_fighters(data, fightdata, fighterinfo)
        
#Remove all debuts from dataset as we do not have stats for fighters making their debut 
debuts_mask = list(data.isnull().any(axis=1))
debuts = data[debuts_mask]
non_debuts_mask = [not x for x in debuts_mask]
non_debuts = data[non_debuts_mask]

#Split the dataset without debuts into train and test data. We test by attempting to predict all fights from 2020.
train = non_debuts.loc[:"2020-01-01", :]
test = non_debuts.loc["2020-01-01":, :]

#Save train and test data
train.to_pickle("../data/train.pkl")
test.to_pickle("../data/test.pkl")

