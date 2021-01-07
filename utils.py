import pandas as pd
import numpy as np


def load_data(season):
    file = f'data/{season}_season_stats_traditional.csv'
    data_espn = pd.read_csv(file)
    data_espn.drop(['SC-EFF','SH-EFF'], axis=1, inplace=True)
    file = f'data/{season}_season_stats_dd.csv'
    data_dd = pd.read_csv(file)
    file = f'data/{season}_season_stats_Hollinger.csv'
    data_hollinger = pd.read_csv(file)
    data_hollinger = data_hollinger[['PLAYER', 'TS%', 'USG']]
    data_hollinger.rename(columns={'PLAYER':'Name'}, inplace=True)
    return data_espn, data_dd, data_hollinger


def preprocess_teams(data_espn):
    data_espn.drop(['GP','GS','MIN','PER'], axis=1,inplace=True)
    data_espn = data_espn[data_espn['Name']=='Total']
    data_espn.reset_index(drop=True, inplace=True)
    data_espn.drop(['Name'], axis=1,inplace=True)
    data_espn.rename(columns={'Pos':'Team'}, inplace=True)
    return data_espn


def preprocess_players(data_espn, data_dd, data_hollinger):
    # keep only player rows
    data_espn.dropna(inplace=True)
    players = data_espn['Name'].unique()
    for p in players:
        # merge traded players
        if data_espn[data_espn['Name']==p].shape[0] > 1:
            temp = data_espn[data_espn['Name']==p]
            info = temp[data_espn.columns[:2]].values[0]
            game_play = np.sum(temp[data_espn.columns[2:4]].values, axis=0)
            weight = temp[data_espn.columns[2]]
            weight = weight / np.sum(weight)
            stats = np.around(np.dot(weight, temp[data_espn.columns[4:]].values), decimals=2)
            new = np.concatenate([info, game_play, stats])
            data_espn[data_espn['Name']==p] = new
    data_espn.drop_duplicates(inplace=True)
    # join double double & hollinger
    data = data_espn.merge(data_dd, how='left', on='Name')
    data = data.merge(data_hollinger, how='left', on='Name')
    data.fillna(0, inplace=True)
    return data


def get_score_table(data, bucket_type):
    columns = ['Name','Pos','GP','PTS','OR','DR','REB','AST','STL','BLK','TO','PF','AST/TO',
               'FGM','FGA','FG%','3PM','3PA','3P%','FTM','FTA','FT%','2PM','2PA','2P%','DD']
    # filter players
    data = data[(data['GP']>=20) & (data['MIN']>=15)]
    # convert to season total stats
    data_tot = data[columns]
    for i in columns:
        if i not in ['Name','Pos','GP','AST/TO','FG%','3P%','FT%','2P%']:
            data_tot[i] = data_tot[i] * data_tot['GP']
    # categories
    category = ['PTS','OR','DR','REB','AST','STL','BLK','3PM','2PM','FGM','FTM','DD','TO','PF']
    data_tot_matrix = data_tot[category]
    # flip negative category
    data_tot_matrix['TO'] *= -1 
    data_tot_matrix['PF'] *= -1 
    # generate buckets for each stats category
    buckets = []
    if bucket_type == 'equal-space': # 10 equally spaced buckets
        for cat in category:
            low = min(data_tot_matrix[cat])
            high = max(data_tot_matrix[cat])
            buckets.append(np.linspace(low,high, num=10, endpoint=False))
    if bucket_type == 'quantile': # based on quantiles
        quantile = [0,.15,.35,.55,.7,.8,.85,.9,.95,.98]
        for cat in category:
            bucket = [data_tot_matrix[cat].quantile(q) for q in quantile]
            buckets.append(bucket)
    # calucate players' score for each stats category
    scores = []
    for i in range(data_tot_matrix.shape[0]):
        player = data_tot_matrix.iloc[i].values.reshape(-1,1)
        scores.append(np.sum(player >= np.array(buckets), axis=1))
    # convert to pandas
    score_table = pd.DataFrame(np.array(scores), columns=category)
    score_table['Name'] = pd.Series(data['Name'].values)
    # Next, calculate players' shooting percentage and AST/TO scores 
    category = {'FG%':['FGA','FGM'], 'FT%':['FTA','FTM'],'3P%':['3PA','3PM'],'2P%':['2PA','2PM'],'AST/TO':['TO','AST']}
    for catg in category:
        # 10 buckets for percentage, where the 5-th bucket is league mean
        made_total = sum(data_tot[category[catg][1]])
        attempt_total = sum(data_tot[category[catg][0]])
        low = min(data[catg])
        league_mean = made_total / attempt_total * 100
        high = max(data[catg])
        below_mean = np.linspace(low, league_mean, num=4, endpoint=False)
        above_mean = np.linspace(league_mean, high, num=6, endpoint=False)
        shoot_per_bucket = np.concatenate((below_mean, above_mean))
        # 5 equally spaced buckets for shooting attempts
        low = min(data_tot[category[catg][0]])
        high = max(data_tot[category[catg][0]])
        shoot_attempt_bucket = np.linspace(low,high, num=5, endpoint=False)
        # divide into 2 groups by shooting percentage league mean
        ind_above = data[catg] >= league_mean
        ind_below = data[catg] < league_mean
        # score based on shooting percentage
        shoot_per_score = np.sum(data[catg].values.reshape(-1,1) >= shoot_per_bucket, axis=1)
        # score based on shooting attempts
        shoot_attempt_score1 = np.sum(data_tot[category[catg][0]].values.reshape(-1,1) >= shoot_attempt_bucket, axis=1) + 5
        shoot_attempt_score2 = 11 - shoot_attempt_score1
        # calculate shooting score incorporate both shooting percentage and shooting attempts
        shoot_score_adjust = np.zeros(data.shape[0])
        shoot_score_adjust[ind_above] = shoot_per_score[ind_above] + shoot_attempt_score1[ind_above]
        shoot_score_adjust[ind_below] = shoot_per_score[ind_below] + shoot_attempt_score2[ind_below]
        shoot_score_adjust /= 2
        # append to score table
        score_table[catg] = shoot_score_adjust
    return score_table




