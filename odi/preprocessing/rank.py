import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from datetime import date,datetime
import dateutil
import click
from odi.feature_engg import util as cricutil
from odi.model_util import odi_util as outil
import pickle

from odi.data_loader import data_loader as dl


PREPROCESS_DATA_LOACATION = 'data'+os.sep+'preprocess'

def get_quantile(quantile_df, value):
    q1 = quantile_df.iloc[0][1]
    q2 = quantile_df.iloc[1][1]
    q3 = quantile_df.iloc[2][1]
    iqr = q3-q1

    if value>(q3+1.5*iqr):
        return 5
    elif value>q3:
        return 4
    elif value>q2:
        return 3
    elif value>q1:
        return 2
    else:
        return 1

def get_latest_rank_file(rank_type,ref_date = None):

    rank_type_prefix = {
        'country':'country_rank_',
        'batsman':'batsman_rank_',
        'bowler': 'bowler_rank_',
        'location': 'location_rank_'
    }

    country_list_files = [f for f in os.listdir(PREPROCESS_DATA_LOACATION) if f.startswith(rank_type_prefix[rank_type])]
    if ref_date is None:
        today = date.today()
        ref_date = datetime(year=today.year, month=today.month, day=today.day)
    date_list = list()
    #print('==get_latest_rank_file===',ref_date)
    #print('==get_latest_rank_file===', country_list_files)

    for file in country_list_files:
        try:
            date_str = file.split(rank_type_prefix[rank_type])[1].split('.')[0]
            file_date = datetime.strptime(date_str, '%Y-%m-%d')
            # print('\t===file====', file)
            # print('\t===file date====',file_date)
            if file_date <= ref_date:
                date_list.append(file_date)
        except Exception as ex:
            print(ex)
            #raise ex

    # print('==date_list===', date_list)
    if len(date_list)==0:
        return None
    else:
        date_list.sort()
        latest = date_list[-1].date()
        #print('==latest=',latest)
        return PREPROCESS_DATA_LOACATION+os.sep+rank_type_prefix[rank_type]+str(latest)+'.csv'


def create_country_rank(year_list,no_of_years=1):
    custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d")
    match_list_df = pd.read_csv(dl.CSV_LOAD_LOCATION+os.sep+'match_list.csv',
                                parse_dates=['date'],
                                date_parser=custom_date_parser)

    if year_list is None or len(year_list)==0:
        today = date.today()
        #performance_cutoff_date_start = datetime.strptime(year + '-01-01', '%Y-%m-%d')
        performance_cutoff_date_end = datetime(year=today.year, month=today.month, day=today.day)
        a_year = dateutil.relativedelta.relativedelta(years=no_of_years)
        performance_cutoff_date_start = performance_cutoff_date_end - a_year

        create_country_rank_for_date(performance_cutoff_date_start, performance_cutoff_date_end, match_list_df)

    else:
        for year in tqdm(year_list):
            # performance_cutoff_date_start = datetime.strptime(year + '-01-01', '%Y-%m-%d')
            performance_cutoff_date_end = datetime.strptime(year + '-12-31', '%Y-%m-%d')
            performance_cutoff_date_start = cricutil.add_day_as_datetime(cricutil.substract_year_as_datetime
                                                                         (performance_cutoff_date_end,no_of_years),
                                                                         1)
            create_country_rank_for_date(performance_cutoff_date_start, performance_cutoff_date_end, match_list_df)


def create_country_rank_for_date(performance_cutoff_date_start, performance_cutoff_date_end, match_list_df):
    if not os.path.isdir(PREPROCESS_DATA_LOACATION):
        os.makedirs(PREPROCESS_DATA_LOACATION)

    match_list_year = match_list_df[(match_list_df['date'] >= performance_cutoff_date_start) & (
            match_list_df['date'] <= performance_cutoff_date_end)]
    country_set = set(match_list_year['first_innings'].unique()).union(set(match_list_year['first_innings'].unique()))
    country_list = list(country_set)
    # print(country_list)
    country_rank_list = []
    # print(year,country_rank_list)
    # ab=[]
    for selected_country in tqdm(country_list):
        # print(selected_country)
        win_count = match_list_year[match_list_year['winner'] == selected_country].shape[0]
        matches_played = match_list_year[(match_list_year['first_innings'] == selected_country) | (
                    match_list_year['second_innings'] == selected_country)].shape[0]
        total_win_by_runs = \
        match_list_year[(match_list_year['winner'] == selected_country) & (match_list_year['win_by'] == 'runs')][
            'win_dif'].sum()
        total_win_by_wickets = \
        match_list_year[(match_list_year['winner'] == selected_country) & (match_list_year['win_by'] == 'wickets')][
            'win_dif'].sum()
        win_ratio = win_count / matches_played
        total_loss_by_runs = match_list_year[((match_list_year['first_innings'] == selected_country) | (
                    match_list_year['second_innings'] == selected_country))
                                             & (match_list_year['winner'] != selected_country) & (
                                                         match_list_year['win_by'] == 'runs')]['win_dif'].sum()
        total_loss_by_wickets = match_list_year[((match_list_year['first_innings'] == selected_country) | (
                    match_list_year['second_innings'] == selected_country))
                                                & (match_list_year['winner'] != selected_country) & (
                                                            match_list_year['win_by'] == 'wickets')]['win_dif'].sum()

        rank_dict = {'country': selected_country,
                     'win_ratio': win_ratio,
                     # 'total_win_by_runs':total_win_by_runs,
                     # 'total_loss_by_runs':total_loss_by_runs,
                     # 'total_win_by_wickets':total_win_by_wickets,
                     # 'total_loss_by_wickets':total_loss_by_wickets,
                     'effective_win_by_runs': total_win_by_runs - total_loss_by_runs,
                     'effective_win_by_wickets': total_win_by_wickets - total_loss_by_wickets,
                     'matches_played': matches_played,
                     'win_count': win_count,

                     }
        # print('dict',selected_country,rank_dict)
        country_rank_list.append(rank_dict)
        # print('updated_list',country_rank_list)
        # ab.append(rank_dict)
    # print(country_rank_list)

    # print(country_rank_list[0])
    score_df = pd.DataFrame(country_rank_list).sort_values('win_ratio', ascending=False)
    scaler = MinMaxScaler()

    score_df['score'] = scaler.fit_transform(
        score_df[['win_ratio', 'effective_win_by_runs', 'effective_win_by_wickets', 'matches_played']]).sum(axis=1)

    # score_scaler = MinMaxScaler()
    # score_df['score'] = score_scaler.fit_transform(score_df[['score']])
    score_df = score_df.sort_values('score', ascending=False)
    score_df['rank'] = range(1, score_df.shape[0] + 1)

    score_quantile_df = score_df['score'].quantile([0.25, 0.5, 0.75]).reset_index()
    score_df['country_quantile'] = score_df['score'].apply(
        lambda x: get_quantile(score_quantile_df, x))

    score_df.to_csv(PREPROCESS_DATA_LOACATION+os.sep+'country_rank_' + str(performance_cutoff_date_end.date()) + '.csv', index=False)


# def get_latest_country_rank_file(ref_date = None):
#     country_list_files = [f for f in os.listdir(PREPROCESS_DATA_LOACATION) if f.startswith('country_rank_')]
#     if ref_date is None:
#         today = date.today()
#         ref_date = datetime(year=today.year, month=today.month, day=today.day)
#     date_list = list()
#
#     for file in country_list_files:
#         try:
#             date_str = file.split('country_rank_')[1].split('.')[0]
#             file_date = datetime.strptime(date_str, '%Y-%m-%d')
#             if file_date <= ref_date:
#                 date_list.append(file_date)
#         except Exception as ex:
#             print(ex)
#
#     if len(date_list)==0:
#         return None
#     else:
#         date_list.sort()
#         latest = date_list[-1].date()
#         return PREPROCESS_DATA_LOACATION+os.sep+'country_rank_'+str(latest)+'.csv'


def create_batsman_rank_for_date(performance_cutoff_date_start, performance_cutoff_date_end, match_list_df):
    if not os.path.isdir(PREPROCESS_DATA_LOACATION):
        os.makedirs(PREPROCESS_DATA_LOACATION)

    scaler = MinMaxScaler()

    country_rank_file = get_latest_rank_file('country',ref_date=performance_cutoff_date_end)
    if country_rank_file is None:
        raise Exception('Cannot create batsman rank-'+\
                        'Country rank not available for '+\
                        str(performance_cutoff_date_end.date()))
    country_rank = pd.read_csv(country_rank_file)
    country_list = list(country_rank['country'])

    batsman_performance_list = []
    for selected_country in tqdm(country_list):
        # print(selected_country)
        country_games = match_list_df[(match_list_df['date'] >= performance_cutoff_date_start)
                                      & (match_list_df['date'] <= performance_cutoff_date_end)
                                      & ((match_list_df['first_innings'] == selected_country)
                                         | (match_list_df['second_innings'] == selected_country)
                                         )]
        match_id_list = list(country_games['match_id'])
        match_stat_list = []
        for match_id in match_id_list:

            match_df = pd.read_csv(dl.CSV_LOAD_LOCATION+ os.sep + str(match_id) + '.csv')

            match_stat_list.append(match_df)

        match_stat_df = pd.concat(match_stat_list)
        match_stat_df.fillna('NA', inplace=True)

        match_stat_df = match_stat_df.merge(country_games, how='inner', on='match_id')
        batsman_list = list(match_stat_df[match_stat_df['team'] == selected_country]['batsman'].unique())

        for selected_batsman in tqdm(batsman_list):
            # print(selected_batsman)

            batsman_df = match_stat_df[match_stat_df['batsman'] == selected_batsman]

            total_runs = batsman_df['scored_runs'].sum()
            run_rate = batsman_df['scored_runs'].sum() / \
                       match_stat_df[match_stat_df['batsman'] == selected_batsman].shape[0]
            team_score = country_rank[country_rank['country'] == selected_country]['score'].values[0]
            # opponent_mean

            batsman_df.rename(columns={'opponent': 'country'}, inplace=True)
            batsman_df = batsman_df.merge(country_rank, on='country', how='inner')
            opponent_mean = batsman_df[['match_id', 'country', 'score']].groupby(['match_id']).min().reset_index()[
                'score'].mean()
            # matches_played = len(list(batsman_df['match_id'].unique()))
            player_of_the_match = country_games[country_games['player_of_match'] == selected_batsman].shape[0]

            # winning contribution(effectiveness)-% of winning score
            country_win_list = list(country_games[country_games['winner'] == selected_country]['match_id'])
            winning_match_df = match_stat_df[match_stat_df['match_id'].isin(country_win_list)]
            winning_contribution = winning_match_df[winning_match_df['batsman'] == selected_batsman][
                                       'scored_runs'].sum() / \
                                   winning_match_df[winning_match_df['team'] == selected_country]['scored_runs'].sum()

            # run_rate_effectiveness
            country_run_rate = winning_match_df[winning_match_df['team'] == selected_country]['scored_runs'].sum() / \
                               winning_match_df[winning_match_df['team'] == selected_country].shape[0]
            batsman_run_rate = winning_match_df[winning_match_df['batsman'] == selected_batsman]['scored_runs'].sum() / \
                               winning_match_df[winning_match_df['batsman'] == selected_batsman].shape[0]

            run_rate_effectiveness = batsman_run_rate / country_run_rate

            batting_std = batsman_df.groupby(['match_id'])['scored_runs'].sum().reset_index()['scored_runs'].std()

            consistency = 1 / batting_std if batting_std != 0 else 1
            average_score = batsman_df.groupby(['match_id'])['scored_runs'].sum().reset_index()['scored_runs'].mean()

            batsman_dict = {
                'batsman': selected_batsman,
                'country': selected_country,
                'total_runs': total_runs,
                'run_rate': run_rate,
                'average_score': average_score,
                'team_score': team_score,
                'opponent_mean': opponent_mean,
                # 'matches_played':matches_played,
                'player_of_the_match': player_of_the_match,
                'winning_contribution': winning_contribution,
                'run_rate_effectiveness': run_rate_effectiveness,
                'consistency': consistency
            }

            batsman_performance_list.append(batsman_dict)

    batsman_performance_df = pd.DataFrame(batsman_performance_list)
    batsman_performance_df.fillna(0, inplace=True)
    batsman_performance_df['batsman_score'] = scaler.fit_transform(
        batsman_performance_df.drop(columns=['batsman', 'country', 'consistency'])).sum(axis=1)
    # score_scaler = MinMaxScaler()
    # batsman_performance_df['batsman_score'] = score_scaler.fit_transform(batsman_performance_df[['batsman_score']])
    batsman_performance_df.sort_values('batsman_score', ascending=False, inplace=True)

    batsman_quantile_df = batsman_performance_df['batsman_score'].quantile([0.25,0.5,0.75]).reset_index()

    batsman_performance_df['batsman_quantile'] = batsman_performance_df['batsman_score'].apply(lambda x:get_quantile(batsman_quantile_df,x))
    batsman_performance_df.to_csv(PREPROCESS_DATA_LOACATION+os.sep+'batsman_rank_' + str(performance_cutoff_date_end.date()) + '.csv', index=False)


def create_batsman_rank(year_list,no_of_years=1):
    custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d")
    match_list_df = pd.read_csv(dl.CSV_LOAD_LOCATION+os.sep+'match_list.csv',
                                parse_dates=['date'],
                                date_parser=custom_date_parser)

    if year_list is None or len(year_list)==0:
        today = date.today()
        performance_cutoff_date_end = datetime(year=today.year, month=today.month, day=today.day)
        a_year = dateutil.relativedelta.relativedelta(years=no_of_years)
        performance_cutoff_date_start = performance_cutoff_date_end - a_year

        create_batsman_rank_for_date(performance_cutoff_date_start, performance_cutoff_date_end, match_list_df)

    else:
        for year in tqdm(year_list):
            performance_cutoff_date_start = datetime.strptime(year + '-01-01', '%Y-%m-%d')
            performance_cutoff_date_end = datetime.strptime(year + '-12-31', '%Y-%m-%d')
            create_batsman_rank_for_date(performance_cutoff_date_start, performance_cutoff_date_end, match_list_df)


def create_bowler_rank_for_date(performance_cutoff_date_start, performance_cutoff_date_end, match_list_df):
    if not os.path.isdir(PREPROCESS_DATA_LOACATION):
        os.makedirs(PREPROCESS_DATA_LOACATION)

    scaler = MinMaxScaler()

    country_rank_file = get_latest_rank_file('country', ref_date=performance_cutoff_date_end)
    if country_rank_file is None:
        raise Exception('Cannot create bowler rank-'+\
                        'Country rank not available for '+\
                        str(performance_cutoff_date_end.date()))
    country_rank = pd.read_csv(country_rank_file)
    country_list = list(country_rank['country'])

    bowler_performance_list = []
    for selected_country in tqdm(country_list):
        # print(selected_country)
        country_games = match_list_df[(match_list_df['date'] >= performance_cutoff_date_start)
                                      & (match_list_df['date'] <= performance_cutoff_date_end)
                                      & ((match_list_df['first_innings'] == selected_country)
                                         | (match_list_df['second_innings'] == selected_country)
                                         )]
        match_id_list = list(country_games['match_id'])
        match_stat_list = []
        for match_id in match_id_list:
            match_df = pd.read_csv(dl.CSV_LOAD_LOCATION + os.sep + str(match_id) + '.csv')

            match_stat_list.append(match_df)

        match_stat_df = pd.concat(match_stat_list)
        match_stat_df.fillna('NA', inplace=True)

        match_stat_df = match_stat_df.merge(country_games, how='inner', on='match_id')
        bowler_list = list(match_stat_df[match_stat_df['opponent'] == selected_country]['bowler'].unique())

        for selected_bowler in tqdm(bowler_list):
            # print(selected_batsman)

            bowler_df = match_stat_df[match_stat_df['bowler'] == selected_bowler]
            total_runs = bowler_df['total'].sum()
            run_rate = total_runs / bowler_df.shape[0]
            negative_rate = -run_rate

            # no_of_wickets,wicket_rate,wicket_per_runs
            no_of_wickets = bowler_df['wicket'].sum() - bowler_df[bowler_df['wicket_type'] == 'run out'].shape[0]
            wickets_per_match = no_of_wickets / len(list(bowler_df['match_id'].unique()))
            wickets_per_run = no_of_wickets / total_runs

            team_score = country_rank[country_rank['country'] == selected_country]['score'].values[0]
            # opponent_mean

            bowler_df.rename(columns={'team': 'country'}, inplace=True)
            bowler_df = bowler_df.merge(country_rank, on='country', how='inner')
            opponent_mean = bowler_df[['match_id', 'country', 'score']].groupby(['match_id']).min().reset_index()[
                'score'].mean()
            matches_played = len(list(bowler_df['match_id'].unique()))
            player_of_the_match = country_games[country_games['player_of_match'] == selected_bowler].shape[0]

            # winning contribution(effectiveness)-% of wickets taken in winning matches
            country_win_list = list(country_games[country_games['winner'] == selected_country]['match_id'])
            winning_match_df = match_stat_df[match_stat_df['match_id'].isin(country_win_list)]

            if winning_match_df['wicket'].sum() != 0:
                winning_contribution = winning_match_df[winning_match_df['bowler'] == selected_bowler]['wicket'].sum() / \
                                       winning_match_df['wicket'].sum()
            else:
                winning_contribution = 0

            # winning_wicket_per_run rate contribution
            # winning wicket_per_match contirbution

            team_wickets_per_run = winning_match_df[winning_match_df['opponent'] == selected_country]['wicket'].sum() / \
                                   winning_match_df[winning_match_df['opponent'] == selected_country]['total'].sum()
            bowler_wicket_per_run = winning_match_df[winning_match_df['bowler'] == selected_bowler]['wicket'].sum() / \
                                    winning_match_df[winning_match_df['bowler'] == selected_bowler]['total'].sum()
            winning_wicket_per_run_rate_contribution = bowler_wicket_per_run / team_wickets_per_run

            team_wicket_per_match = winning_match_df[winning_match_df['opponent'] == selected_country]['wicket'].sum() / \
                                    winning_match_df['match_id'].nunique()
            bowler_wicket_per_match = winning_match_df[winning_match_df['bowler'] == selected_bowler]['wicket'].sum() / \
                                      winning_match_df[winning_match_df['bowler'] == selected_bowler][
                                          'match_id'].nunique()
            winning_wicket_per_match_contribution = bowler_wicket_per_match / team_wicket_per_match

            no_of_wins = winning_match_df[winning_match_df['bowler'] == selected_bowler]['match_id'].nunique()
            # consistency
            # consistency = 1/match_stat_df[match_stat_df['bowler']==selected_bowler].groupby(['match_id'])['wicket'].sum().reset_index()['wicket'].std()

            bowler_dict = {
                'bowler': selected_bowler,
                'country': selected_country,
                'negative_rate': negative_rate,
                'no_of_wickets': no_of_wickets,
                'wickets_per_match': wickets_per_match,
                'wickets_per_run': wickets_per_run,
                'no_of_wins': no_of_wins,
                'team_score': team_score,
                'opponent_mean': opponent_mean,
                'winning_contribution': winning_contribution,
                'winning_wicket_rate_contribution': winning_wicket_per_match_contribution,

            }

            bowler_performance_list.append(bowler_dict)

    bowler_performance_df = pd.DataFrame(bowler_performance_list)
    bowler_performance_df.fillna(0, inplace=True)
    bowler_performance_df['bowler_score'] = scaler.fit_transform(
        bowler_performance_df.drop(columns=['bowler', 'country'])).sum(axis=1)
    # score_scaler = MinMaxScaler()
    # bowler_performance_df['bowler_score'] = score_scaler.fit_transform(bowler_performance_df[['bowler_score']])
    bowler_performance_df.sort_values('bowler_score', ascending=False, inplace=True)

    bowler_quantile_df = bowler_performance_df['bowler_score'].quantile([0.25, 0.5, 0.75]).reset_index()
    bowler_performance_df['bowler_quantile'] = bowler_performance_df['bowler_score'].apply(
        lambda x: get_quantile(bowler_quantile_df, x))

    bowler_performance_df.to_csv(PREPROCESS_DATA_LOACATION+os.sep+'bowler_rank_' + str(performance_cutoff_date_end.date()) + '.csv', index=False)


def create_bowler_rank(year_list,no_of_years=1):
    custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d")
    match_list_df = pd.read_csv(dl.CSV_LOAD_LOCATION+os.sep+'match_list.csv',
                                parse_dates=['date'],
                                date_parser=custom_date_parser)

    if year_list is None or len(year_list)==0:
        today = date.today()
        performance_cutoff_date_end = datetime(year=today.year, month=today.month, day=today.day)
        a_year = dateutil.relativedelta.relativedelta(years=no_of_years)
        performance_cutoff_date_start = performance_cutoff_date_end - a_year

        create_bowler_rank_for_date(performance_cutoff_date_start, performance_cutoff_date_end, match_list_df)

    else:
        for year in tqdm(year_list):
            performance_cutoff_date_start = datetime.strptime(year + '-01-01', '%Y-%m-%d')
            performance_cutoff_date_end = datetime.strptime(year + '-12-31', '%Y-%m-%d')
            create_bowler_rank_for_date(performance_cutoff_date_start, performance_cutoff_date_end, match_list_df)


def create_location_rank_for_date(performance_cutoff_date_start,performance_cutoff_date_end, match_list_df):
    if not os.path.isdir(PREPROCESS_DATA_LOACATION):
        os.makedirs(PREPROCESS_DATA_LOACATION)

    selected_games = match_list_df[(match_list_df['date'] >= performance_cutoff_date_start)
                                      & (match_list_df['date'] <= performance_cutoff_date_end)]

    match_stats = pd.read_csv(dl.CSV_LOAD_LOCATION + os.sep + 'match_stats.csv')
    selected_games = selected_games.merge(match_stats, how='inner', on='match_id')

    selected_games_first_innings = selected_games[selected_games['first_innings']==selected_games['team_statistics']][['location','total_run']]
    selected_games_second_innings = selected_games[selected_games['second_innings']==selected_games['team_statistics']][['location','total_run']]

    selected_games_first_innings.dropna(inplace=True)
    selected_games_second_innings.dropna(inplace=True)
    first_innings_mean = selected_games_first_innings.groupby(['location']).mean().reset_index()
    first_innings_default_mean = selected_games_first_innings['total_run'].mean()
    first_innings_mean.loc[len(first_innings_mean.index)] = ['default', first_innings_default_mean]
    first_innings_mean['innings'] = 'first'

    second_innings_mean = selected_games_second_innings.groupby(['location']).mean().reset_index()
    second_innings_default_mean = selected_games_second_innings['total_run'].mean()
    second_innings_mean.loc[len(second_innings_mean.index)] = ['default', second_innings_default_mean]
    second_innings_mean['innings'] = 'second'

    innings_mean = pd.concat([first_innings_mean,second_innings_mean])
    innings_mean.to_csv(
        PREPROCESS_DATA_LOACATION + os.sep + 'location_rank_' + str(performance_cutoff_date_end.date()) + '.csv',
        index=False)


def create_location_rank(year_list,no_of_years=5):
    custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d")
    match_list_df = pd.read_csv(dl.CSV_LOAD_LOCATION+os.sep+'match_list.csv',
                                parse_dates=['date'],
                                date_parser=custom_date_parser)

    if year_list is None or len(year_list)==0:
        today = date.today()
        performance_cutoff_date_end = datetime(year=today.year, month=today.month, day=today.day)
        a_year = dateutil.relativedelta.relativedelta(years=no_of_years)
        performance_cutoff_date_start = performance_cutoff_date_end - a_year

        create_location_rank_for_date(performance_cutoff_date_start, performance_cutoff_date_end, match_list_df)

    else:
        for year in tqdm(year_list):
            performance_cutoff_date_start = datetime.strptime(year + '-01-01', '%Y-%m-%d')
            performance_cutoff_date_end = datetime.strptime(year + '-12-31', '%Y-%m-%d')
            create_location_rank_for_date(performance_cutoff_date_start, performance_cutoff_date_end, match_list_df)



def create_reduciton_factor(start_date,end_date):
    start_date = cricutil.str_to_date_time(start_date)
    end_date = cricutil.str_to_date_time(end_date)
    match_list_df = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION + os.sep + 'match_list.csv')
    match_list_df = match_list_df[(match_list_df['date']>start_date) & (match_list_df['date']<end_date)]
    match_stats_df = pd.read_csv(dl.CSV_LOAD_LOCATION + os.sep + 'match_stats.csv')

    match_info = match_list_df.merge(match_stats_df, how='inner', on='match_id')

    mean_list = []
    match_id_list = match_info['match_id'].unique()
    # for country in list(match_info_2018['first_innings'].unique()):
    for match_id in match_id_list:

        country_matches = match_info[
            (match_info["match_id"] == match_id) &
            (match_info["first_innings"] == match_info["team_statistics"])]
        country = country_matches['first_innings'].values[0]
        date = country_matches['date'].values[0]
        ref_date = cricutil.npdate_to_datetime(date)
        rank_file = get_latest_rank_file('batsman', ref_date)
        rank_df = pd.read_csv(rank_file)
        score_mean = 0
        mean_dict = {"country": country}
        # print('country'country)
        for bi in range(11):
            batsman = country_matches['batsman_' + str(bi + 1)].values[0]
            if batsman == 'not_batted':
                break
            else:
                if rank_df[(rank_df['country'] == country) & (rank_df['batsman'] == batsman)].shape[0] > 0:
                    score = \
                    rank_df[(rank_df['country'] == country) & (rank_df['batsman'] == batsman)]['batsman_score'].values[
                        0]
                    score_mean = (score_mean + score) / (bi + 1)
                    mean_dict['position_' + str(bi + 1)] = score_mean

        mean_list.append(mean_dict)

    mean_df = pd.DataFrame(mean_list)
    mean_df.dropna(inplace=True)

    reduction_factor_df = pd.DataFrame()
    reduction_factor_df['country'] = mean_df['country']
    for cols in range(10):
        reduction_factor_df[str(cols + 1) + '_by_' + str(cols + 2)] = mean_df['position_' + str(cols + 1)] / mean_df[
            'position_' + str(cols + 2)]

    #print(reduction_factor_df)
    reduction_dict = dict(reduction_factor_df.drop(columns="country").mean())

    print(reduction_dict)
    pickle.dump(reduction_dict, open(os.path.join(outil.DEV_DIR, outil.SCORE_MEAN_REDUCTION_FACTOR), 'wb'))

@click.group()
def rank():
    pass

@rank.command()
@click.option('--year_list', multiple=True, help='list of years.')
@click.option('--no_of_years', type=int, default=1,
              help='applicable if year list not provided. How many previous years to consider')
def all(year_list,no_of_years):

    year_list = list(year_list)
    create_country_rank(year_list,no_of_years=no_of_years)
    create_batsman_rank(year_list,no_of_years=no_of_years)
    create_bowler_rank(year_list,no_of_years=no_of_years)

@rank.command()
@click.option('--year_list', multiple=True, help='list of years.')
@click.option('--no_of_years', type=int, default=1,
              help='applicable if year list not provided. How many previous years to consider')
def batsman(year_list,no_of_years):

    year_list = list(year_list)
    create_country_rank(year_list,no_of_years=no_of_years)
    create_batsman_rank(year_list,no_of_years=no_of_years)

@rank.command()
@click.option('--year_list', multiple=True, help='list of years.')
@click.option('--no_of_years', type=int, default=1,
              help='applicable if year list not provided. How many previous years to consider')
def bowler(year_list,no_of_years):

    year_list = list(year_list)
    create_country_rank(year_list,no_of_years=no_of_years)
    create_bowler_rank(year_list,no_of_years=no_of_years)

@rank.command()
@click.option('--year_list', multiple=True, help='list of years.')
@click.option('--no_of_years', type=int, default=1,
              help='applicable if year list not provided. How many previous years to consider')
def batsman_only(year_list,no_of_years):

    year_list = list(year_list)
    create_batsman_rank(year_list,no_of_years=no_of_years)

@rank.command()
@click.option('--year_list', multiple=True, help='list of years.')
@click.option('--no_of_years', type=int, default=1,
              help='applicable if year list not provided. How many previous years to consider')
def bowler_only(year_list,no_of_years):

    year_list = list(year_list)
    create_bowler_rank(year_list,no_of_years=no_of_years)

@rank.command()
@click.option('--year_list', multiple=True, help='list of years.')
@click.option('--no_of_years', type=int, default=1,
              help='applicable if year list not provided. How many previous years to consider')
def country(year_list,no_of_years):

    year_list = list(year_list)
    create_country_rank(year_list,no_of_years)

@rank.command()
@click.option('--year_list', multiple=True, help='list of years.')
@click.option('--no_of_years', type=int, default=5,
              help='applicable if year list not provided. How many previous years to consider')
def location(year_list,no_of_years):

    year_list = list(year_list)
    create_location_rank(year_list,no_of_years)

@rank.command()
@click.option('--start_date', help='start date for player score mean reduction_factor',required=True)
@click.option('--end_date', help='end date for layer score mean reduction factor',required=True)
def reduction_analysis(start_date,end_date):
    create_reduciton_factor(start_date,end_date)


if __name__=='__main__':
    rank()


