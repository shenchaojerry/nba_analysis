import pandas as pd
import numpy as np
import requests
import bs4
from difflib import get_close_matches
from collections import defaultdict

######## Web Scrapping ########

def get_ESPN(season):
    teams = ['atl','bkn','bos','cha','chi','cle','dal','den','det','gs','hou','ind',
            'lac','lal','mem','mia','mil','min','no','ny','okc','orl','phi','phx','por',
            'sac','sa','tor','utah','wsh']
    url = "https://www.espn.com/nba/team/stats/_/name/"
    season_type = f"/season/{season}/seasontype/2" # regular season
    class_id = "Table Table--align-right"
    data = []
    for team in teams:
        # web scrapping
        team_info = requests.get(url+team+season_type)
        soup = bs4.BeautifulSoup(team_info.text,"html.parser")
        # get stats tables
        tables = soup.select('table', attrs={'class':class_id})
        tables = pd.read_html(str(tables))
        # process player info
        names = []
        positions = []
        for player in tables[0].values[:-1]:
            info = player[0].replace('*','').split()
            names.append(' '.join(info[:-1]))
            positions.append(info[-1])
        names.append('Total')
        positions.append(team)
        player_info = {'Name': names, 'Pos': positions}
        player_info = pd.DataFrame(player_info)
        # integrate data
        data.append(pd.concat([player_info,tables[1],tables[3]],axis=1))
    Data = pd.concat([d for d in data])
    return Data

def get_dd(season, Names_espn):
    url = "https://www.landofbasketball.com/year_by_year_stats/"
    stats_type = "_double_doubles_rs.htm"
    page = requests.get(url+f'{season-1}_{season}'+stats_type)
    soup = bs4.BeautifulSoup(page.text,"html.parser")
    # get stats tables
    class_id = "color-alt sobre a-center" 
    tables = soup.select('table', attrs={'class':class_id})
    tables = pd.read_html(str(tables))
    data = tables[0]
    data = data.iloc[2:-1].drop(0,axis=1)[[1,2]]
    data.columns = ['Name', 'DD']
    data = data[data['DD'] != "Double-Doubles"]
    # process names to align with ESPN
    names = []
    for player in data['Name']:
        info = player.split(' (')
        name = info[0]
        # name resolution: e.g. C.J. McCollum -> CJ McCollum
        if name not in Names_espn:
            candidates = get_close_matches(name, Names_espn)
            if candidates:
                names.append(candidates[0])
                print(f'change {name} -> {candidates[0]}')
            else:
                print(f'Did not find a match, who is {name} ?!')
        else:
            names.append(name)
    data['Name'] = names
    return data

def get_Hollinger(season, Names_espn):
    url = "http://insider.espn.com/nba/hollinger/statistics/_/sort/usageRate/page/"
    # get number of pages
    page = requests.get(url+f'{1}/year/{season}')
    soup = bs4.BeautifulSoup(page.text,"html.parser")
    class_id = "page-numbers"
    number_of_pages = soup.findAll("div",{'class':class_id})
    number_of_pages = int(str(number_of_pages).split("</")[0][~1:])
    # get stats
    data = []
    class_id = "tablehead"
    for p in range(1,number_of_pages+1):
        page = requests.get(url+f'{p}/year/{season}')
        soup = bs4.BeautifulSoup(page.text,"html.parser")
        # get stats table
        table = soup.select('table', attrs={'class':class_id})
        table = pd.read_html(str(table))
        table = table[0]
        table.drop(0, axis=1, inplace=True)
        # get columns
        if p == 1:
            columns = table.iloc[1].values
        # filter
        table = table[[b.isnumeric() for b in table[2]]]
        table.reset_index(drop=True,inplace=True)
        # process player info
        names = []
        for player in table[1].values:
            name = player.split(',')[0]
            # name resolution: e.g. C.J. McCollum -> CJ McCollum
            if name not in Names_espn:
                candidates = get_close_matches(name, Names_espn)
                if candidates:
                    names.append(candidates[0])
                    print(f'change {name} -> {candidates[0]}')
                else:
                    print(f'Did not find a match, who is {name} ?!')
            else:
                names.append(name)
        data.append(pd.concat([pd.Series(names),table.drop(1,axis=1)],axis=1))
    Data = pd.concat([d for d in data])
    Data.columns = columns
    return Data


######## Proprocessing ########

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

######## Yahoo Fantasy ########

def getDraftResult(path):
    with open(path, 'r') as file:
        page = file.read()

    soup = bs4.BeautifulSoup(page,"html.parser")
    # get stats tables
    class_id = "Table Fz-xxs" 
    tables = soup.select('table', attrs={'class':class_id})
    tables = pd.read_html(str(tables))
    # preprossing
    league = {}
    team_size = 0
    for i in range(len(tables)):
        tb = tables[i]
        team = tb.columns[0]
        draft = tb[tb.columns[-1]].values
        players = [p.split(" (")[0] for p in draft]
        league[team] = players
    return league

def getCurrentStanding(path):
    with open(path, 'r') as file:
        page = file.read()

    soup = bs4.BeautifulSoup(page,"html.parser")
    # get stats tables
    class_id = "Tst-table Table Ta-start Fz-xs Table-mid Table-px-med" 
    tables = soup.select('table', attrs={'class':class_id})
    tables = pd.read_html(str(tables))
    # get ranking table
    Ranking = tables[0]
    columns = [c[1] for c in Ranking.columns]
    columns[-1] = "Total"
    Ranking.columns = columns
    # get statistic table
    Statistic = tables[1]
    columns = [c[1] for c in Statistic.columns]
    Statistic.columns = columns
    return Ranking, Statistic

def trackSeasonStats(season, season_stats):
    # ESPN
    data_espn = get_ESPN(season)
    data_espn.drop(['SC-EFF','SH-EFF'], axis=1, inplace=True)
    # will align all names to ESPN
    Names_espn = data_espn['Name'].values
    # double-double
    data_dd = get_dd(season, Names_espn)
    # Hollinger
    data_hollinger = get_Hollinger(season, Names_espn)
    data_hollinger = data_hollinger[['PLAYER', 'TS%', 'USG']]
    data_hollinger.rename(columns={'PLAYER':'Name'}, inplace=True)
    # integrate
    data_player = preprocess_players(data_espn, data_dd, data_hollinger)
    Names_espn = data_player['Name'].values
    # keep records
    for player in Names_espn:
        temp = data_player[data_player["Name"]==player]
        game_played = temp['GP'].values[0]
        if player not in season_stats:
            season_stats[player] = defaultdict(dict)
            season_stats[player]['GP'] = defaultdict(dict)
        if game_played not in season_stats[player]['GP']:
            season_stats[player]['GP'][game_played] = defaultdict(dict)
            for col in temp.columns[4:]:
                season_stats[player]['GP'][game_played][col] = temp[col].values[0]
    return season_stats

def updateYahooRosters(path, team_size, season_stats, IL=0):
    with open(path, 'r') as file:
        page = file.read()

    soup = bs4.BeautifulSoup(page,"html.parser")
    # get stats tables
    class_id = "Table Table-px-xs Mbot-xl" 
    tables = soup.select('table', attrs={'class':class_id})
    tables = pd.read_html(str(tables))
    # get team names
    class_id = "W-100 Fz-med Ta-c" 
    teams = soup.findAll("p",{'class':class_id})
    teams = [str(t) for t in teams]
    teams = [t.split('</a>')[0].split('>')[-1] for t in teams]
    # preprocessing
    league = {}
    Names_espn = season_stats.keys()
    for i in range(len(tables)):
        tb = tables[i]
        player = tb[tb.columns[1]].values
        # process player names
        player = tb[tb.columns[1]].values
        player = [p.split(" -")[0] for p in player if p != "(Empty)"]
        player = [p.replace("Notes", "Note") for p in player]
        player = [p.split("Note ")[-1] for p in player]
        player = [p.split(" ") for p in player]
        player = [" ".join(p[:-1]) for p in player]
        for j in range(len(player)):
            name = player[j]
            if name not in Names_espn:
                candidates = get_close_matches(name, Names_espn)
                if candidates:
                    if (candidates[0] in name) or (name in candidates[0]): # e.g. Robert Williams -> Robert Williams III
                        player[j] = candidates[0]
                        print(f'change {name} -> {candidates[0]}')
                    else:
                        print(f'pass on {name} -> no records yet')
                else:
                    print(f'pass on {name} -> no records yet')
        if len(player) != team_size+IL: # max team size
            player.extend(["NA"]*(team_size+IL-len(player)))
        league[teams[i]] = player
    return league

def trackYahooLeague(league, key, path, IL, season_stats):  
    # Get draft results
    draft_result = pd.DataFrame(getDraftResult(path+'/draft_result.txt'))
    team_size = draft_result.shape[0]

    # Get current Yahoo league roseters
    league_teams = updateYahooRosters(path+'/Rosters.txt', team_size, season_stats, IL)

    # Get current standings
    Ranking, Statistic = getCurrentStanding(path+'/Standings.txt')

    # Keep records
    league[key] = defaultdict(dict)
    league[key]['Roster'] = league_teams
    league[key]['Ranking'] = Ranking
    league[key]['Statistic'] = Statistic
    return league

def updateInjuryStatus(season_stats):
    url = "https://www.cbssports.com/nba/injuries/"
    page = requests.get(url)
    soup = bs4.BeautifulSoup(page.text,"html.parser")
    # get stats tables
    class_id = "TableBase-table" 
    tables = soup.select('table', attrs={'class':class_id})
    tables = pd.read_html(str(tables))
    injuryList = defaultdict()
    for team in tables:
        players = team["Player"].values
        players = [p.split(" ") for p in players]
        players = [' '.join(p[2:]) for p in players]
        status = team["Injury Status"].values
        for name, s in zip(players, status):
            # name resolution
            if name not in season_stats:
                candidates = get_close_matches(name, season_stats.keys())
                if candidates:
                    if candidates[0] in name: # e.g. Jr. Otto Porter Jr. -> Otto Porter Jr.
                        injuryList[candidates[0]] = s
                        print(f'change {name} -> {candidates[0]}')
                    else:
                        injuryList[name] = s
                else:
                    injuryList[name] = s
            else:
                injuryList[name] = s
    return injuryList

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




