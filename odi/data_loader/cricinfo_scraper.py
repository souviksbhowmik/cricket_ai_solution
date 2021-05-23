import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd
from odi.data_loader import data_loader as dl
from odi.feature_engg import util as cricutil
from odi.preprocessing import rank
import os
import click
from tqdm import tqdm

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import dateutil

include_list = ['Sri Lanka', 'New Zealand', 'England', 'West Indies', 'Bangladesh', 'India', 'Pakistan', 'Australia', 'South Africa']
def get_missing_player_list(href):
    link = 'https://stats.espncricinfo.com'+href
    match_page = requests.get(link)
    match_soup = BeautifulSoup(match_page.content, 'html.parser')
    team_a_list = None
    team_b_list = None
    for idx, table in enumerate(match_soup.find_all("table")):
        # print(table.text)
        if idx == 0 or idx == 2:

            for tr in table.find_all('tr'):
                for td in table.find_all('td'):
                    # print(td.text)
                    # print('------------------')
                    if "Did not bat" in td.text:
                        if idx == 0:
                            info = td.text
                            team_a_list = info.split(':')[1].split(',')
                            # break
                        else:
                            info = td.text
                            team_b_list = info.split(':')[1].split(',')

    #clean the list
    if team_a_list is not None and len(team_a_list)>0:
        temp_list = []
        for player in team_a_list:
            temp_list.append(player.strip())
        team_a_list = temp_list

    if team_b_list is not None and len(team_b_list)>0:
        temp_list = []
        for player in team_b_list:
            temp_list.append(player.strip())
        team_b_list = temp_list

    return team_a_list,team_b_list


def get_multiple_dates(date_str):
    y = date_str.split(' ')

    date_str1 = y[0]+' '+y[1].split('-')[0]+','+' '+y[2]
    date_str2 = y[0]+' '+y[1].split('-')[1]+' '+y[2]
    return date_str1, date_str2

def create_not_batted_list(year_list,mode='a'):
    dict_list = []
    # create a dummy entry
    dummy = {}
    dummy['team_a']='dummy1'
    dummy['team_b'] = 'dummy2'
    dummy['venue']='dummy'
    dummy['date']=datetime.strftime(datetime.today(), '%Y-%m-%d').strip()
    dummy['href']='/dummy/'
    for i in range(11):
        dummy["team_a_batsman_" + str(i + 1)] = 'dummy_player_' + str(i + 1)
        dummy["team_b_batsman_" + str(i + 1)] = 'dummy_player_' + str(i + 1)

    dict_list.append(dummy)
    # end of dummy entry
    for year in tqdm(year_list):
        year = str(year)
        link = 'https://stats.espncricinfo.com/ci/engine/records/team/match_results.html?class=2;id='+year+';type=year'
        #print('link :',link)
        page = requests.get(link)
        soup = BeautifulSoup(page.content, 'html.parser')
        all_tbody = soup.find_all('tbody')
        all_tr = all_tbody[0].find_all('tr')

        for tr in tqdm(all_tr):
            try:
                row_dict = {}
                alt_date = None
                for idx,td in enumerate(tr.find_all('td')):
                    if idx==0:
                        row_dict['team_a']=td.text.strip()
                    elif idx==1:
                        row_dict['team_b'] = td.text.strip()
                    elif idx==4:
                        row_dict['venue'] = td.text.strip()

                    elif idx == 5:
                        if '-' not in td.text.strip():
                            date = datetime.strptime(td.text.strip(), '%b %d, %Y')
                            row_dict['date'] = datetime.strftime(date, '%Y-%m-%d').strip()
                        else:
                            date_str_1,date_str_2 = get_multiple_dates(td.text.strip())
                            date = datetime.strptime(date_str_1, '%b %d, %Y')
                            alt_date = datetime.strptime(date_str_2, '%b %d, %Y')
                            row_dict['date'] = datetime.strftime(date, '%Y-%m-%d').strip()

                    elif idx==6:

                        href = td.find_all('a')[0].get("href")
                        row_dict['href'] =href
                        team_a_list, team_b_list = get_missing_player_list(href)

                        if team_a_list is not None and len(team_a_list)>0:
                            missing_count = len(team_a_list)
                            start_index = 11-missing_count
                            for ind,player in enumerate(team_a_list):
                                player_num = str(start_index+ind+1)
                                row_dict["team_a_batsman_"+player_num] = player

                        if team_b_list is not None and len(team_b_list)>0:
                            missing_count = len(team_b_list)
                            start_index = 11-missing_count
                            for ind,player in enumerate(team_b_list):
                                player_num = str(start_index+ind+1)
                                row_dict["team_b_batsman_"+player_num] = player

                    else:
                        pass

                #print("row_dict ",row_dict)
                dict_list.append(row_dict)
                if alt_date is not None:
                    copy_row_dict = dict(row_dict)
                    copy_row_dict['date']=datetime.strftime(alt_date, '%Y-%m-%d').strip()
                    dict_list.append(copy_row_dict)
                # print(row_dict)
                # print("============")
                #break
            except Exception  as ex:
                print(ex,' skipped ')


    #print(dict_list)
    data_df = pd.DataFrame(dict_list)


    if mode is None or mode!='a':
        #print("mode new ",mode)
        data_df.to_csv(dl.CSV_LOAD_LOCATION+os.sep+'not_batted.csv', index=False)
    else:
        #print("mode append ",mode)
        data_df.to_csv(dl.CSV_LOAD_LOCATION+os.sep+'not_batted.csv', index=False, mode='a',header=False)

def update_match_stats(start_date=None,end_date=None):
    nb_df = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION+os.sep+'not_batted.csv')
    match_list = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION+os.sep+'match_list.csv')
    match_stats = pd.read_csv(dl.CSV_LOAD_LOCATION+os.sep+'match_stats.csv')
    nb_df.rename(columns={"team_a": "first_innings", "team_b": "second_innings"}, inplace=True)

    if start_date is not None:
        start_dt = cricutil.str_to_date_time(start_date)
        match_list = match_list[match_list['date']>=start_date]
    if end_date is not None:
        end_dt = cricutil.str_to_date_time(end_date)
        match_list = match_list[match_list['date'] <= end_date]

    merged_match_list = match_list.merge(nb_df, how="inner", on=["first_innings", "second_innings", "date"])
    merged_match_list = merged_match_list.merge(match_stats, on="match_id", how="inner")
    merged_match_list.fillna('not_available',inplace=True)

    for row_index in tqdm(range(merged_match_list.shape[0])):
        # print(merged_match_list.iloc[row_index])
        team = merged_match_list.iloc[row_index].team_statistics
        first_innings = merged_match_list.iloc[row_index].first_innings
        second_innings = merged_match_list.iloc[row_index].second_innings
        date = merged_match_list.iloc[row_index].date
        match_id = merged_match_list.iloc[row_index].match_id
        # print("========",type(date),date.to_pydatetime())
        ref_date = date.to_pydatetime()

        if team == first_innings:
            team_type = 'team_a'
        else:
            team_type = 'team_b'

        previous_year_batsman_rank_file = rank.get_latest_rank_file('batsman', ref_date=ref_date)
        previous_year_bowler_rank_file = rank.get_latest_rank_file('bowler', ref_date=ref_date)

        a_year = dateutil.relativedelta.relativedelta(years=1)
        ref_date_ahead = ref_date + a_year
        current_year_batsman_rank_file = rank.get_latest_rank_file('batsman', ref_date=ref_date_ahead)
        current_year_bowler_rank_file = rank.get_latest_rank_file('bowler', ref_date=ref_date_ahead)

        #     print(team)
        #     print(ref_date)
        #     print('file 1',previous_year_batsman_rank_file)
        #     print('file 2',previous_year_bowler_rank_file)
        #     print('file 3',current_year_batsman_rank_file)
        #     print('file 4',current_year_bowler_rank_file)

        player_set = set()

        if previous_year_batsman_rank_file is not None:
            previous_year_df = pd.read_csv(previous_year_batsman_rank_file)
            batsman_list = list(previous_year_df[previous_year_df['country'] == team]['batsman'].unique())

            player_set = player_set.union(set(batsman_list))

        if current_year_batsman_rank_file is not None:
            current_year_df = pd.read_csv(current_year_batsman_rank_file)
            batsman_list = list(current_year_df[current_year_df['country'] == team]['batsman'].unique())

            player_set = player_set.union(set(batsman_list))

        if previous_year_bowler_rank_file is not None:
            previous_year_df = pd.read_csv(previous_year_bowler_rank_file)
            bowler_list = list(previous_year_df[previous_year_df['country'] == team]['bowler'].unique())

            player_set = player_set.union(set(bowler_list))

        if current_year_bowler_rank_file is not None:
            current_year_df = pd.read_csv(current_year_bowler_rank_file)
            bowler_list = list(current_year_df[current_year_df['country'] == team]['bowler'].unique())

            player_set = player_set.union(set(bowler_list))

        last_name_list = []
        full_name_list = []
        for player in player_set:
            last_name_list.append(player.split(" ")[-1])
            full_name_list.append(player)

        #     print('=============set=============')
        # print(last_name_list)

        if len(last_name_list) == 0:
            continue

        name_vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3))
        last_name_vecotrs = name_vectorizer.fit_transform(last_name_list)

        for i in range(11):
            batsman_name = merged_match_list.iloc[row_index]['batsman_' + str(i + 1)]
            if batsman_name == 'not_batted':

                reserved_batsman = merged_match_list.iloc[row_index][team_type + '_batsman_' + str(i + 1)]
                if reserved_batsman=='not_available':
                    continue
                #print(date,team,team_type + '_batsman_' + str(i + 1),reserved_batsman)
                #print("getting from not batted list ", reserved_batsman)
                reserved_last_name = reserved_batsman.split(' ')[-1]
                reserved_last_name_vector = name_vectorizer.transform([reserved_last_name])
                cosine_sim_list = []
                for vector in last_name_vecotrs:
                    cosine_sim_list.append(cosine_similarity(reserved_last_name_vector, vector)[0][0])

                max_sim = np.max(np.array(cosine_sim_list))
                if max_sim > 0.9:
                    max_arg = np.argmax(np.array(cosine_sim_list))
                    matching_player = full_name_list[max_arg]
                    # print("\t matched with  ", matching_player)
                    match_stats_idx = match_stats[
                        (match_stats['match_id'] == match_id) & (match_stats['team_statistics'] == team)].index
                    # print('\t at index ', match_stats_idx)
                    match_stats.loc[match_stats_idx, 'batsman_' + str(i + 1)] = matching_player
                else:
                    max_arg = np.argmax(np.array(cosine_sim_list))
                    matching_player = full_name_list[max_arg]
                    print("\t did not find for ",reserved_last_name," - closest match  with  ", matching_player, ' - ', max_sim)

            else:
                #print("batting ", batsman_name)
                pass

    match_stats.to_csv(dl.CSV_LOAD_LOCATION+os.sep+'match_stats.csv',index=False)

def remove_incorrect_matches():
    nb_df = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION+os.sep+'not_batted.csv')
    match_list = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION+os.sep+'match_list.csv')
    nb_df.rename(columns={"team_a": "first_innings", "team_b": "second_innings"}, inplace=True)

    # if start_date is not None:
    #     start_dt = cricutil.str_to_date_time(start_date)
    #     match_list = match_list[match_list['date']>=start_date]
    # if end_date is not None:
    #     end_dt = cricutil.str_to_date_time(end_date)
    #     match_list = match_list[match_list['date'] <= end_date]

    merged_match_list = match_list.merge(nb_df, how="inner", on=["first_innings", "second_innings", "date"])
    keep_list = list(merged_match_list['match_id'])
    match_list = match_list[match_list['match_id'].isin(keep_list)]
    match_list.to_csv(dl.CSV_LOAD_LOCATION+os.sep+'match_list.csv',index=False)


def download_matches(year_list,mode='a'):
    match_list = []
    batting_list = []
    bowling_list = []
    for year in tqdm(year_list):
        year = str(year)
        link = 'https://stats.espncricinfo.com/ci/engine/records/team/match_results.html?class=2;id='+year+';type=year'
        #print('link :',link)
        page = requests.get(link)
        soup = BeautifulSoup(page.content, 'html.parser')
        all_tbody = soup.find_all('tbody')
        all_tr = all_tbody[0].find_all('tr')

        for tr in tqdm(all_tr):
            try:
                row_dict = {}
                alt_date = None
                team_a = None
                team_b = None
                href = None
                for idx,td in enumerate(tr.find_all('td')):
                    if idx==0:
                        team_a=td.text.strip()
                    elif idx==1:
                        team_b = td.text.strip()
                    elif idx==2:
                        row_dict['winner'] = td.text.strip()
                    elif idx==3:
                        if 'runs' in td.text.strip():
                            row_dict['win_by'] = "runs"
                            row_dict["win_by_runs"] = td.text.strip().split(' ')[0]
                            row_dict["win_by_wickets"] = 0
                            row_dict["win_by_count"] = td.text.strip()
                        elif "wickets" in td.text.strip():
                            row_dict['win_by'] = "wickets"
                            row_dict["win_by_runs"] = 0
                            row_dict["win_by_wickets"] = td.text.strip().split(' ')[0]
                            row_dict["win_by_count"] = td.text.strip()
                        else:
                            row_dict["win_by"] = "not_applicable"
                            row_dict["win_by_runs"] = 0
                            row_dict["win_by_wickets"] = 0
                            row_dict["win_by_count"] = 0
                    elif idx==4:
                        row_dict['location'] = td.text.strip()

                    elif idx == 5:
                        if '-' not in td.text.strip():
                            date = datetime.strptime(td.text.strip(), '%b %d, %Y')
                            row_dict['date'] = datetime.strftime(date, '%Y-%m-%d').strip()
                        else:
                            date_str_1,date_str_2 = get_multiple_dates(td.text.strip())
                            date = datetime.strptime(date_str_1, '%b %d, %Y')
                            alt_date = datetime.strptime(date_str_2, '%b %d, %Y')
                            row_dict['date'] = datetime.strftime(date, '%Y-%m-%d').strip()

                    elif idx==6:

                        href = td.find_all('a')[0].get("href")
                        row_dict['href'] =href


                    else:
                        pass

                if team_a not in include_list or team_b not in include_list:
                    continue

                match_soup=get_match_as_soup(href)
                first_innings,second_innings = get_innings_sequence(match_soup,team_a,team_b)
                row_dict['first_innings'] = first_innings
                row_dict['second_innings'] = second_innings
                first_innings_batting, first_innings_bowling, \
                second_innings_batting, second_innings_bowling, toss_winner, \
                total_runs_first,loss_of_wickets_first,extras_first,\
                total_runs_second,loss_of_wickets_second,extras_second=\
               get_match_statistics(match_soup, row_dict['first_innings'], row_dict['second_innings'], row_dict['date'])

                row_dict['toss_winner'] = toss_winner
                row_dict['first_innings_run'] = total_runs_first
                row_dict['first_innings_fow'] = loss_of_wickets_first
                row_dict['first_innings_extras'] = extras_first
                row_dict['second_innings_run'] = total_runs_second
                row_dict['second_innings_fow'] = loss_of_wickets_second
                row_dict['second_innings_extras'] = extras_second
                match_list.append(row_dict)

                batting_list = batting_list + first_innings_batting + second_innings_batting
                bowling_list = bowling_list + first_innings_bowling + second_innings_bowling

                # if alt_date is not None:
                #     copy_row_dict = dict(row_dict)
                #     copy_row_dict['date']=datetime.strftime(alt_date, '%Y-%m-%d').strip()
                #     match_list.append(copy_row_dict)
                # print(row_dict)
                # print("============")
                #break
            except Exception  as ex:
                print(ex,' skipped ')
                #raise ex


    #print(dict_list)
    data_df = pd.DataFrame(match_list)
    batting_df = pd.DataFrame(batting_list)
    bowling_df = pd.DataFrame(bowling_list)


    if mode is None or mode!='a':
        #print("mode new ",mode)
        data_df.to_csv(dl.CSV_LOAD_LOCATION+os.sep+'cricinfo_match_list.csv', index=False)
        batting_df.to_csv(dl.CSV_LOAD_LOCATION + os.sep + 'cricinfo_batting.csv', index=False)
        bowling_df.to_csv(dl.CSV_LOAD_LOCATION + os.sep + 'cricinfo_bowling.csv', index=False)
    else:
        #print("mode append ",mode)
        data_df.to_csv(dl.CSV_LOAD_LOCATION+os.sep+'cricinfo_match_list.csv', index=False, mode='a',header=False)
        batting_df.to_csv(dl.CSV_LOAD_LOCATION + os.sep + 'cricinfo_batting.csv', index=False, mode='a', header=False)
        bowling_df.to_csv(dl.CSV_LOAD_LOCATION + os.sep + 'cricinfo_bowling.csv', index=False, mode='a', header=False)

def get_match_as_soup(href):
    link = 'https://stats.espncricinfo.com' + href
    match_page = requests.get(link)
    match_soup = BeautifulSoup(match_page.content, 'html.parser')

    return match_soup

def get_innings_sequence(match_soup,team_a,team_b):
    h5_list = match_soup.find_all("h5")
    first_inn_h5 = h5_list[0]
    first_innings = None
    second_innings = None

    if team_a in  first_inn_h5.text:
        first_innings = team_a
        second_innings = team_b
    elif team_b in first_inn_h5.text:
        first_innings = team_b
        second_innings = team_a
    else:
        raise Exception("Could not interprete nnings")

    return first_innings,second_innings



def get_match_statistics(match_soup,first_innings,second_innings,date):

    first_innings_batting =[]
    first_innings_bowling = []
    second_innings_batting = []
    second_innings_bowling = []
    toss_winner = None
    total_runs_first = None
    loss_of_wickets_first = None
    extras_first = None
    total_runs_second = None
    loss_of_wickets_second= None
    extras_second = None

    for idx, table in enumerate(match_soup.find_all("table")):
        if idx == 0:
            first_innings_batting,total_runs_first,loss_of_wickets_first,extras_first = get_batting(table,first_innings,second_innings,"first",date)
        elif idx==1:
            first_innings_bowling = get_bowling(table,second_innings,first_innings,"first",date)
        elif idx == 2:
            second_innings_batting,total_runs_second,loss_of_wickets_second,extras_second = get_batting(table,second_innings,first_innings,"second",date)
        elif idx==3:
            second_innings_bowling = get_bowling(table,first_innings,second_innings,"second",date)
        elif idx == 4:
            for tr in table.find_all("tr"):
                if "toss" in tr.text.lower():
                    if first_innings in tr.text:
                        toss_winner = first_innings
                    else:
                        toss_winner = second_innings

                if "player Of the match" in tr.text.lower():
                    if tr.find_all("td")>=2:
                        player_of_the_match = tr.find_all("td")[1].text.strip()
                        if '(c)' in player_of_the_match:
                            player_of_the_match = player_of_the_match.replace('(c)','').strip()
                        if '†' in player_of_the_match:
                            player_of_the_match = player_of_the_match.replace('†','').strip()
                        found = False
                        for entries in first_innings_batting:
                            if entries["name"]==player_of_the_match:
                                entries["player_of_the_match"]=1
                                found= True
                                break

                        if found == False:
                            for entries in second_innings_batting:
                                if entries["name"] == player_of_the_match:
                                    entries["player_of_the_match"] = 1
                                    found = True
                                    break

                        if found == False:
                            for entries in first_innings_bowling:
                                if entries["name"] == player_of_the_match:
                                    entries["player_of_the_match"] = 1
                                    found = True
                                    break

                        if found == False:
                            for entries in second_innings_bowling:
                                if entries["name"] == player_of_the_match:
                                    entries["player_of_the_match"] = 1
                                    found = True
                                    break





        else:
            pass



    return first_innings_batting,first_innings_bowling,second_innings_batting,second_innings_bowling,toss_winner,\
           total_runs_first,loss_of_wickets_first,extras_first,total_runs_second,loss_of_wickets_second,extras_second


def get_batting(table,team,opponent,innings_type,date):
    innings_batting = []
    not_batted_list = None
    batting_pos = 0
    extras = None
    total_runs = None
    loss_of_wickets = None
    for tr in table.find_all("tr"):

        if 'fall of wickets' in tr.text.lower():
            continue
        elif 'did not bat' in tr.text.lower():
            info = tr.find_all('td')[0].text
            not_batted_list = info.split(':')[1].split(',')
            continue
        elif 'extras' in tr.text.lower():
            if len(tr.find_all('td'))>=3:
                extras = tr.find_all('td')[2].text
            continue
        elif 'total' in tr.text.lower():
            if len(tr.find_all('td')) >= 3:
                info = tr.find_all('td')[2].text
                #print("=========",info)
                total_runs = info.split('/')[0]
                if len(info.split('/'))==2:
                    loss_of_wickets = info.split('/')[1]
                else:
                    loss_of_wickets = 10
            continue
        elif tr.text.strip() =='':
            continue
        else:
            pass

        batting_dict = {}
        batting_dict["team"] = team
        batting_dict["opponent"] = opponent
        batting_dict["batting_innings"] = innings_type
        batting_dict["date"] = date
        batting_dict["did_bat"] = 1
        batting_dict["player_of_the_match"] = 0
        batting_dict["is_captain"] = 0
        batting_dict["wc"] = 0

        for td_idx,td in enumerate(tr.find_all('td')):
            if td_idx == 0:
                if td.text.strip() == '':
                    continue
                elif 'fall of wickets' in td.text.lower():
                    pass
                elif 'did not bat' in td.text.lower():
                    info = td.text
                    not_batted_list = info.split(':')[1].split(',')
                elif 'extras' in td.text.lower():
                    info = td.text
                    pass
                elif 'total' in td.text.lower():
                    info = td.text
                    pass
                else:
                    name =td.text.strip()
                    if '(c)' in name:
                        name = name.replace('(c)', '').strip()
                        batting_dict['is_captain'] = 1
                    if '†' in name:
                        name = name.replace('†', '').strip()
                        batting_dict['wc'] = 1

                    batting_dict['name']=name
                    batting_pos = batting_pos+1
                    batting_dict['position'] = batting_pos
            elif td_idx == 1:
                if 'not out' in td.text.lower():
                    batting_dict['is_out']=0
                else:
                    batting_dict['is_out'] = 1
                batting_dict['wicket_type'] = td.text
            elif td_idx == 2:
                batting_dict['runs'] = td.text.strip()
            elif td_idx == 3:
                batting_dict['balls'] = td.text.strip()
            elif td_idx == 4:
                batting_dict['m'] = td.text.strip()
            elif td_idx == 5:
                batting_dict['4s'] = td.text.strip()
            elif td_idx == 6:
                batting_dict['6s'] = td.text.strip()
            elif td_idx == 7:
                batting_dict['sr'] = td.text.strip()
            else:
                pass

        innings_batting.append(batting_dict)

    if not_batted_list is not None and len(not_batted_list)>0:
        for player in not_batted_list:

            batting_dict = {}
            batting_dict["team"] = team
            batting_dict["opponent"] = opponent
            batting_dict["batting_innings"] = innings_type
            batting_dict["date"] = date
            batting_dict["did_bat"] = 0
            batting_dict["player_of_the_match"] = 0
            batting_dict["is_captain"] = 0
            batting_dict["wc"] = 0
            name = td.text.strip()
            if '(c)' in name:
                name = name.replace('(c)', '').strip()
                batting_dict['is_captain'] = 1
            if '†' in name:
                name = name.replace('†', '').strip()
                batting_dict['wc'] = 1

            batting_dict['name'] = name
            batting_pos = batting_pos + 1
            batting_dict['position'] = batting_pos
            batting_dict['runs'] = 0
            batting_dict['balls'] = 0
            batting_dict['m'] = 0
            batting_dict['4s'] = 0
            batting_dict['6s'] = 0
            batting_dict['sr'] = 0

            innings_batting.append(batting_dict)

    return innings_batting,total_runs,loss_of_wickets,extras

def get_bowling(table,team,opponent,innings_type,date):
    innings_bowling = []
    for tr in table.find_all("tr"):

        if tr.text.strip() =='':
            continue

        bowling_dict = {}
        bowling_dict["team"] = team
        bowling_dict["opponent"] = opponent
        bowling_dict["bowling_innings"] = innings_type
        bowling_dict["date"] = date
        bowling_dict["player_of_the_match"] = 0
        bowling_dict["is_captain"] = 0

        for td_idx,td in enumerate(tr.find_all('td')):
            if td_idx == 0:
                if td.text.strip() == '':
                    continue
                else:
                    name = td.text.strip()
                    if '(c)' in name:
                        name = name.replace('(c)', '').strip()
                        bowling_dict['is_captain'] = 1

                    bowling_dict['name']=name
            elif td_idx ==1:
                bowling_dict['overs'] = td.text.strip()
            elif td_idx ==2:
                bowling_dict['maidens'] = td.text.strip()
            elif td_idx ==3:
                bowling_dict['runs'] = td.text.strip()
            elif td_idx ==4:
                bowling_dict['wickets'] = td.text.strip()
            elif td_idx ==5:
                bowling_dict['econ'] = td.text.strip()
            elif td_idx ==6:
                bowling_dict['0s'] = td.text.strip()
            elif td_idx ==7:
                bowling_dict['4s'] = td.text.strip()
            elif td_idx ==8:
                bowling_dict['6s'] = td.text.strip()
            elif td_idx ==9:
                bowling_dict['wd'] = td.text.strip()
            elif td_idx ==9:
                bowling_dict['nb'] = td.text.strip()
            else:
                pass

        innings_bowling.append(bowling_dict)


    return innings_bowling



@click.group()
def scrapper():
    pass

@scrapper.command()
@click.option('--year_list', multiple=True, help='list of years.')
@click.option('--append', default='a', help='a for append,n for refresh.')
def load_not_batted(year_list,append):
    year_list = list(year_list)
    create_not_batted_list(year_list, mode=append)

@scrapper.command()
@click.option('--start_date', help='start date  (YYYY-mm-dd)')
@click.option('--end_date', help='end dat (YYYY-mm-dd)')
def update_stats(start_date, end_date):
    update_match_stats(start_date=start_date, end_date=end_date)

@scrapper.command()
def remove_incorrect():
    remove_incorrect_matches()

@scrapper.command()
@click.option('--year_list', multiple=True, help='list of years.')
@click.option('--append', default='a', help='a for append,n for refresh.')
def load_match_cricinfo(year_list,append):
    year_list = list(year_list)
    download_matches(year_list, mode=append)



if __name__ =='__main__':
    scrapper()
