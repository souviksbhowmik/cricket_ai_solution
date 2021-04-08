from ipl.data_loader import download_controller as dc
import os
from datetime import datetime
from tqdm import tqdm
import yaml
import pandas as pd
import click
import shutil


CSV_LOAD_LOCATION = 'data_ipl' + os.sep + 'csv_load'


def aggregate_matches(downloaded_file_dir,file_list,from_date = None,to_date = None,mode='a'):

    if not os.path.isdir(CSV_LOAD_LOCATION):
        os.makedirs(CSV_LOAD_LOCATION)

    match_list = []
    reference_date = None
    reference_to_date = None
    if from_date is not None:
        reference_date = datetime.strptime(from_date, '%Y-%m-%d')
    if to_date is not None:
        reference_to_date = datetime.strptime(to_date, '%Y-%m-%d')
    for file in tqdm(file_list):
        try:
            if not file.endswith('.yaml'):
                continue
            match_info = yaml.load(open(downloaded_file_dir + os.sep + file), Loader=yaml.FullLoader)
            content_date = match_info['info']['dates'][0]
            if isinstance(content_date, str):
                content_date = datetime.strptime(match_info['info']['dates'][0], '%Y-%m-%d')
            else:
                content_date = datetime.combine(content_date, datetime.min.time())

            if reference_date is not None and content_date < reference_date:
                continue
            if reference_to_date is not None and content_date > reference_to_date:
                continue


            # ignore DL and Draw matches
            if ('result' in match_info['info']['outcome'] and match_info['info']['outcome']['result'] == 'no result') \
                    or ('method' in match_info['info']['outcome'] and match_info['info']['outcome']['method'] == 'D/L'):
                continue
            if match_info['info']['gender'] != 'male':
                continue

            # All checks complete #


            match_dict = {}
            match_stat_list = []

            match_id = file.split('.')[0]
            # print(match_id)
            teams = match_info['info']['teams']

            first_innings_team = None
            second_innings_team = None
            for innings_details in match_info['innings']:
                if '1st innings' in innings_details:
                    first_innings_team = innings_details['1st innings']['team']
                else:
                    second_innings_team = innings_details['2nd innings']['team']

            location = None
            if 'city' in match_info['info']:
                location = match_info['info']['city']
            else:
                location = match_info['info']['venue']

            winner = match_info['info']['outcome']['winner']
            win_by = 'runs' if 'runs' in list(match_info['info']['outcome']['by'].keys()) else 'wickets'
            win_dif = match_info['info']['outcome']['by'][list(match_info['info']['outcome']['by'].keys())[0]]
            # date = content_date
            if 'player_of_match' in match_info['info']:
                player_of_match = match_info['info']['player_of_match'][0]
            else:
                player_of_match = 'NA'

            match_dict['match_id'] = match_id
            match_dict['date'] = content_date
            match_dict['location'] = location
            match_dict['first_innings'] = first_innings_team
            match_dict['second_innings'] = second_innings_team
            match_dict['winner'] = winner
            match_dict['win_by'] = win_by
            match_dict['win_dif'] = win_dif
            match_dict['toss_winner'] = match_info['info']['toss']['winner']
            match_dict['player_of_match'] = player_of_match

            match_list.append(match_dict)

            for innings_details in match_info['innings']:
                innings = list(innings_details.keys())[0]
                innings_info = innings_details[innings]
                team = innings_info['team']

                teams_copy = teams.copy()
                teams_copy.remove(team)
                opponent = teams_copy[-1]
                for delivery in innings_info['deliveries']:
                    row_dict = {}
                    row_dict['match_id'] = match_id
                    row_dict['innings'] = innings
                    # row_dict['date']=content_date
                    # row_dict['location']=location
                    row_dict['team'] = team
                    row_dict['opponent'] = opponent
                    # row_dict['winner']=winner
                    # row_dict['win_by']=win_by
                    # row_dict['win_dif']=win_dif
                    row_dict['ball'] = list(delivery.keys())[0]
                    if 'batsman' in delivery[row_dict['ball']]:
                        row_dict['batsman'] = delivery[row_dict['ball']]['batsman']
                    if 'non_striker' in delivery[row_dict['ball']]:
                        row_dict['non_striker'] = delivery[row_dict['ball']]['non_striker']
                    if 'bowler' in delivery[row_dict['ball']]:
                        row_dict['bowler'] = delivery[row_dict['ball']]['bowler']
                    if 'runs' in delivery[row_dict['ball']]:
                        if 'batsman' in delivery[row_dict['ball']]['runs']:
                            row_dict['scored_runs'] = delivery[row_dict['ball']]['runs']['batsman']
                        if 'extras' in delivery[row_dict['ball']]['runs']:
                            row_dict['extras'] = delivery[row_dict['ball']]['runs']['extras']
                        if 'total' in delivery[row_dict['ball']]['runs']:
                            row_dict['total'] = delivery[row_dict['ball']]['runs']['total']
                    if 'extras' in delivery[row_dict['ball']]:
                        row_dict['extra_type'] = list(delivery[row_dict['ball']]['extras'].keys())[0]
                    else:
                        row_dict['extra_type'] = 'NA'
                    if 'wicket' in delivery[row_dict['ball']]:
                        row_dict['wicket'] = 1
                        row_dict['wicket_type'] = delivery[row_dict['ball']]['wicket']['kind']
                        row_dict['player_out'] = delivery[row_dict['ball']]['wicket']['player_out']
                        if 'fielders' in delivery[row_dict['ball']]['wicket']:
                            row_dict['fielders'] = delivery[row_dict['ball']]['wicket']['fielders'][0]
                        else:
                            row_dict['fielders'] = 'NA'
                    else:
                        row_dict['wicket'] = 0
                        row_dict['wicket_type'] = 'NA'
                        row_dict['player_out'] = 'NA'
                        row_dict['fielders'] = 'NA'
                    if team == winner:
                        row_dict['winner'] = 1
                    else:
                        row_dict['winner'] = 0

                    match_stat_list.append(row_dict)

            match_stat_df = pd.DataFrame(match_stat_list)

            match_stat_df.to_csv(CSV_LOAD_LOCATION + os.sep + match_id + '.csv', index=False)

            match_stat_df.to_csv(CSV_LOAD_LOCATION + os.sep + match_id + '.csv', index=False)

        except Exception as ex:
            print(file, ':', ex)

    match_list_df = pd.DataFrame(match_list)
    match_list_df.sort_values('date', inplace=True)
    if mode is None or mode !='a':
        match_list_df.to_csv(CSV_LOAD_LOCATION + os.sep + 'match_list.csv', index=False)
    else:
        match_list_df.to_csv(CSV_LOAD_LOCATION + os.sep + 'match_list.csv', index=False, mode='a', header=False)

    # remove yamls
    shutil.rmtree(downloaded_file_dir)


def create_extended_match_summary(from_date = None,to_date = None,mode='a'):
    custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d")
    match_summary_df = pd.read_csv(CSV_LOAD_LOCATION + os.sep + 'match_list.csv',
                                   parse_dates=['date'], date_parser=custom_date_parser)

    if from_date is not None:
        cutoff_start_date = datetime.strptime(from_date, '%Y-%m-%d')
        recent_match_summary_df = match_summary_df[match_summary_df['date'] > cutoff_start_date]
    else:
        recent_match_summary_df = pd.DataFrame(match_summary_df)

    if to_date is not None:
        cutoff_end_date = datetime.strptime(to_date, '%Y-%m-%d')
        recent_match_summary_df = recent_match_summary_df[recent_match_summary_df['date'] < cutoff_end_date]

    match_id_list = list(recent_match_summary_df['match_id'].unique())

    match_stat_list = []
    batsman_set = set()
    bowler_set = set()
    for selected_match_id in tqdm(match_id_list):


        match_detail_df = pd.read_csv(CSV_LOAD_LOCATION + os.sep + str(selected_match_id) + '.csv')

        first_innings = \
        recent_match_summary_df[recent_match_summary_df['match_id'] == selected_match_id]['first_innings'].values[0]
        second_innings = \
        recent_match_summary_df[recent_match_summary_df['match_id'] == selected_match_id]['second_innings'].values[0]

        for batting_innings, bowling_innings in zip([first_innings, second_innings], [second_innings, first_innings]):

            team_batsman_list = list(match_detail_df[match_detail_df['team'] == batting_innings]['batsman'].unique())
            team_bowler_list = list(match_detail_df[match_detail_df['team'] == bowling_innings]['bowler'].unique())

            team_stats = {}
            team_stats['match_id'] = selected_match_id
            team_stats['team_statistics'] = batting_innings
            # batsman_set=batsman_set.union(set(team_batsman_list))
            # bowler_set=bowler_set.union(set(team_bowler_list))
            concatenated_batsman_list = []
            concatenated_bowler_list = []
            for bi in range(11):
                if bi < len(team_batsman_list):
                    batsman = team_batsman_list[bi]
                    team_stats['batsman_' + str(bi + 1)] = batsman
                    team_stats['batsman_' + str(bi + 1) + '_runs'] = \
                    match_detail_df[match_detail_df['batsman'] == batsman]['scored_runs'].sum()
                    concatenated_batsman_list.append(batting_innings.strip() + ' ' + batsman.strip())
                else:
                    team_stats['batsman_' + str(bi + 1)] = 'not_batted'
                    team_stats['batsman_' + str(bi + 1) + '_runs'] = 0

            for boi in range(11):
                if boi < len(team_bowler_list):
                    bowler = team_bowler_list[boi]
                    team_stats['bowler_' + str(boi + 1)] = bowler
                    team_stats['bowler_' + str(boi + 1) + '_wickets'] = \
                        match_detail_df[match_detail_df['bowler'] == bowler]['wicket'].sum() - \
                        match_detail_df[(match_detail_df['bowler'] == bowler) & (
                                    match_detail_df['wicket_type'] == 'run out')].shape[0]
                    concatenated_bowler_list.append(batting_innings.strip() + ' ' + bowler.strip())

                else:
                    team_stats['bowler_' + str(boi + 1)] = 'not_bowled'
                    team_stats['bowler_' + str(boi + 1) + '_wickets'] = 0
            batsman_set = batsman_set.union(set(concatenated_batsman_list))
            bowler_set = bowler_set.union(set(concatenated_bowler_list))

            team_stats['total_run'] = match_detail_df[match_detail_df['team'] == batting_innings]['total'].sum()
            team_stats['total_wickets'] = match_detail_df[match_detail_df['team'] == bowling_innings]['wicket'].sum()
            match_stat_list.append(team_stats)

    match_stats_df = pd.DataFrame(match_stat_list)

    if mode is None or mode!='a':
        match_stats_df.to_csv(CSV_LOAD_LOCATION+os.sep+'match_stats.csv', index=False)
    else:
        match_stats_df.to_csv(CSV_LOAD_LOCATION+os.sep+'match_stats.csv', index=False, mode='a',header=False)


@click.group()
def load():
    pass


@load.command()
@click.option('--from_date', help='optional from data in YYYY-mm-dd.')
@click.option('--to_date', help='optional to data in YYYY-mm-dd.')
@click.option('--append', default='a', help='a for append,n for refresh.')
def load_current(from_date, to_date,append):

    if append is None or append !='a':
        # remove exisitng
        if os.path.isdir(CSV_LOAD_LOCATION):
            shutil.rmtree(CSV_LOAD_LOCATION)

    downloaded_file_dir = dc.download_zip('https://cricsheet.org/downloads/ipl.zip')
    print('New zip downloaded ',downloaded_file_dir)
    file_list = dc.get_new_match_list(downloaded_file_dir)
    print('New match list obtained')
    #print(file_list)
    aggregate_matches(downloaded_file_dir, file_list, from_date=from_date, to_date=to_date, mode=append)
    print('matches aggregated')
    create_extended_match_summary(from_date=from_date, to_date=to_date, mode=append)
    print('extended summary created')


@load.command()
@click.option('--from_date', help='from data in YYYY-mm-dd.',required=True)
@click.option('--to_date', help='to data in YYYY-mm-dd.',required=True)
def load_old(from_date, to_date):
    downloaded_file_dir = dc.download_zip('https://cricsheet.org/downloads/ipl.zip')
    print('New zip downloaded ',downloaded_file_dir)
    file_list = dc.get_new_match_list(downloaded_file_dir,return_all=True)
    print('match list obtained')
    #print(file_list)
    aggregate_matches(downloaded_file_dir, file_list, from_date=from_date, to_date=to_date, mode='a')
    print('matches aggregated')
    create_extended_match_summary(from_date=from_date, to_date=to_date, mode='a')
    print('extended summary created')


if __name__ =='__main__':
    load()






