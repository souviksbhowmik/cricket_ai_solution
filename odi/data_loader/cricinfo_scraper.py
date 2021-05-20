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


def create_not_batted_list(year_list,mode='a'):
    dict_list = []
    for year in tqdm(year_list):
        year = str(year)
        link = 'https://stats.espncricinfo.com/ci/engine/records/team/match_results.html?class=2;id='+year+';type=year'
        page = requests.get(link)
        soup = BeautifulSoup(page.content, 'html.parser')
        all_tbody = soup.find_all('tbody')
        all_tr = all_tbody[0].find_all('tr')

        for tr in tqdm(all_tr):
            row_dict = {}
            for idx,td in enumerate(tr.find_all('td')):
                if idx==0:
                    row_dict['team_a']=td.text.strip()
                elif idx==1:
                    row_dict['team_b'] = td.text.strip()
                elif idx==4:
                    row_dict['venue'] = td.text.strip()

                elif idx == 5:
                    date = datetime.strptime(td.text.strip(), '%b %d, %Y')
                    row_dict['date'] = date

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

            dict_list.append(row_dict)

    data_df = pd.DataFrame(dict_list)

    data_df.to_csv(dl.CSV_LOAD_LOCATION + os.sep + 'not_batted.csv',index=False)
    data_df.drop_duplicates(subset=['team_a','team_b','date'],inplace=True)

    if mode is None or mode!='a':
        data_df.to_csv(dl.CSV_LOAD_LOCATION+os.sep+'not_batted.csv', index=False)
    else:
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
