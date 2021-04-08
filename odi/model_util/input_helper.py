import click
import pickle
from odi.model_util import odi_util as outil
from odi.data_loader import data_loader as dl
from odi.preprocessing import rank
from datetime import datetime

import editdistance
import pandas as pd
import os


@click.group()
def creator():
    pass

@creator.command()
@click.option('--location', help='location name you want to verify.')
def find_location(location):
    location_map = pickle.load(open(outil.MODEL_DIR+os.sep+outil.LOC_ENCODING_MAP_FOR_BATSMAN, 'rb'))
    location_list = list(location_map.keys())

    found = False
    location_score_list = []
    for map_location in location_list:

        score = editdistance.eval(location.lower().strip(), map_location.lower().strip())
        if score == 0:
            found = True
            print('found - use this string ', map_location)
            break
        else:
            location_score_list.append({'location': map_location, 'score': score})

    if not found:
        close_match = list(pd.DataFrame(location_score_list).sort_values('score').head(5)['location'])

        print('Exact location not found, if one of the location from the below list matches, use as is')
        print(close_match)

@creator.command()
@click.option('--team', help='team name you want to verify.')
def find_team(team):
    team_list=list(set(pd.read_csv(dl.CSV_LOAD_LOCATION+os.sep+'match_list.csv')['first_innings'].unique()).union(
        set(pd.read_csv(dl.CSV_LOAD_LOCATION + os.sep + 'match_list.csv')['second_innings'].unique())
    ))

    found = False
    team_score_list = []

    for given_team in team_list:
        score = editdistance.eval(team.lower().strip(), given_team.lower().strip())

        if score == 0:
            found = True
            print('found - use this string ', given_team)
            break
        else:
            team_score_list.append({'team': given_team, 'score': score})

    if not found:
        close_match = list(pd.DataFrame(team_score_list).sort_values('score').head(5)['team'])

        print('Exact team not found, if one of the teams from the below list matches, use as is')
        print(close_match)

@creator.command()
@click.option('--team_a', help='name of team A.',required=True)
@click.option('--team_b', help='name of team B.',required=True)
@click.option('--location', help='location of match.',required=True)
@click.option('--out_dir', help='Directory of template.')
def create_input_template(team_a,team_b,location,out_dir):

    custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d")
    match_list_df = pd.read_csv(dl.CSV_LOAD_LOCATION+os.sep+'match_list.csv',
                                parse_dates=['date'],date_parser=custom_date_parser)

    for team,xlsx_name in zip([team_a,team_b],['team_a.xlsx','team_b.xlsx']):

        match_list_team=match_list_df[(match_list_df['first_innings']==team) |\
                                   (match_list_df['second_innings']==team)]

        if match_list_team.shape[0] == 0:
            raise Exception(team+' has not played any match in recent History or name is incorrect')

        match_list_team=match_list_team.sort_values('date',ascending=False)
        match_id = match_list_team.iloc[0]['match_id']
        match_details_df = pd.read_csv(dl.CSV_LOAD_LOCATION+os.sep+str(match_id)+'.csv')

        data_list = list()
        data_list.append({'type': 'location', 'name': location})
        data_list.append({'type': 'team', 'name': team})

        batting_order_list = list(match_details_df[match_details_df['team']==team]['batsman'].unique())
        for player in batting_order_list:
            data_list.append({'type': 'player', 'name': player})

        bowler_list = match_details_df[match_details_df['team']!=team]['bowler'].unique()
        for bowler in bowler_list:
            if bowler not in batting_order_list:
                batting_order_list.append(bowler)
                data_list.append({'type': 'player', 'name': bowler})

        batsman_rank_file = rank.get_latest_rank_file('batsman')
        bowler_rank_file = rank.get_latest_rank_file('bowler')

        batsman_rank_df = pd.read_csv(batsman_rank_file)
        bowler_rank_df = pd.read_csv(bowler_rank_file)

        additional_batsman_list = list(batsman_rank_df[batsman_rank_df['country']==team]['batsman'].unique())
        additional_bowler_list = list(bowler_rank_df[bowler_rank_df['country']==team]['bowler'].unique())

        for add_batsman in additional_batsman_list:
            if add_batsman not in batting_order_list:
                batting_order_list.append(add_batsman)
                data_list.append({'type': 'reserved', 'name': add_batsman})

        for add_bowler in additional_bowler_list:
            if add_bowler not in batting_order_list:
                batting_order_list.append(add_bowler)
                data_list.append({'type': 'reserved', 'name': add_bowler})


        player_df = pd.DataFrame(data_list)

        if out_dir is not None and out_dir.strip() != '':
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)

            player_df.to_excel(out_dir+os.sep+xlsx_name,index=False)
        else:
            player_df.to_excel(xlsx_name, index=False)


if __name__=='__main__':
    creator()
