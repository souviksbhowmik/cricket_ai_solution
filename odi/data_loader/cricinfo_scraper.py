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



if __name__ =='__main__':
    scrapper()
