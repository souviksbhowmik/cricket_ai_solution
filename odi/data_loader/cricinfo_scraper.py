import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd
from odi.data_loader import data_loader as dl
import os
import click
from tqdm import tqdm


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

@click.group()
def scrapper():
    pass

@scrapper.command()
@click.option('--year_list', multiple=True, help='list of years.')
@click.option('--append', default='a', help='a for append,n for refresh.')
def load_not_batted(year_list,append):
    year_list = list(year_list)
    create_not_batted_list(year_list, mode=append)


if __name__ =='__main__':
    scrapper()
