import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from datetime import date,datetime
import dateutil
import click
from odi.feature_engg import util as cricutil
from scipy.stats import pearsonr
import numpy as np

from odi.data_loader import data_loader as dl


PREPROCESS_DATA_LOACATION = 'data'+os.sep+'preprocess'
mean_inverse_econ = 0.17
ranking_dates = ['-01-31','-02-28','-03-31','-04-30','-05-31','-06-30','-07-31','-08-31','-09-30','-10-31','-11-30','-12-31']

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
    #print('=====',ref_date)
    #print('=====', country_list_files)

    for file in country_list_files:
        try:
            date_str = file.split(rank_type_prefix[rank_type])[1].split('.')[0]
            file_date = datetime.strptime(date_str, '%Y-%m-%d')
            #print('===file date====',file_date)
            if file_date <= ref_date:
                date_list.append(file_date)
        except Exception as ex:
            print(ex)

    if len(date_list)==0:
        return None
    else:
        date_list.sort()
        latest = date_list[-1].date()
        #print('==latest=',latest)
        return PREPROCESS_DATA_LOACATION+os.sep+rank_type_prefix[rank_type]+str(latest)+'.csv'


def create_country_rank(year_list,no_of_years=1):
    custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d")
    match_list_df = pd.read_csv(dl.CSV_LOAD_LOCATION+os.sep+'cricinfo_match_list.csv',
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
            for quarters in ranking_dates:
                print(" country ranking for ",year + quarters)
                performance_cutoff_date_end = datetime.strptime(year + quarters, '%Y-%m-%d')
                performance_cutoff_date_start = cricutil.add_day_as_datetime(cricutil.substract_year_as_datetime
                                                                             (performance_cutoff_date_end,no_of_years),
                                                                             1)
                create_country_rank_for_date(performance_cutoff_date_start, performance_cutoff_date_end, match_list_df)

def get_win_weightage(previous_country_rank_df,row):
    winner = row['winner']
    first_innings = row['first_innings']
    second_innings = row['second_innings']
    if winner == first_innings:
        loser = second_innings
    elif winner == second_innings:
        loser = first_innings
    else:
        loser = winner

    winner_row = previous_country_rank_df[previous_country_rank_df['country']==winner]
    if winner_row.shape[0]>0:
        winner_score = winner_row.iloc[0].country_quantile
    else:
        winner_score = 1

    loser_row = previous_country_rank_df[previous_country_rank_df['country'] == loser]
    if loser_row.shape[0] > 0:
        loser_score = loser_row.iloc[0].country_quantile
    else:
        loser_score = 1

    win_weightage = winner_score/loser_score
    return win_weightage

def create_country_rank_for_date(performance_cutoff_date_start, performance_cutoff_date_end, match_list_df):
    if not os.path.isdir(PREPROCESS_DATA_LOACATION):
        os.makedirs(PREPROCESS_DATA_LOACATION)



    match_list_year = match_list_df[(match_list_df['date'] >= performance_cutoff_date_start) & (
            match_list_df['date'] <= performance_cutoff_date_end)]

    previous_country_ranking_file = get_latest_rank_file('country', ref_date=performance_cutoff_date_start)
    if previous_country_ranking_file is not None:
        #print("=========calculating win weightage with===",previous_country_ranking_file,' for ',performance_cutoff_date_start)
        previous_country_rank_df = pd.read_csv(previous_country_ranking_file)
        match_list_year['win_weightage']=match_list_year.apply(lambda x:get_win_weightage(previous_country_rank_df,x),axis=1)
    else:
        #print("=========cannot calculate win weightage since file is not there===")
        match_list_year['win_weightage'] = 1

    country_set = set(match_list_year['first_innings'].unique()).union(set(match_list_year['second_innings'].unique()))
    country_list = list(country_set)
    # print(country_list)
    country_rank_list = []
    # print(year,country_rank_list)
    # ab=[]
    for selected_country in tqdm(country_list):
        # print(selected_country)
        win_count = match_list_year[match_list_year['winner'] == selected_country].shape[0]
        weighted_win_count = match_list_year[match_list_year['winner'] == selected_country]['win_weightage'].sum()
        matches_played = match_list_year[(match_list_year['first_innings'] == selected_country) | (
                    match_list_year['second_innings'] == selected_country)].shape[0]
        total_win_by_runs = match_list_year[(match_list_year['winner'] == selected_country)]['win_by_runs'].sum()
        total_win_by_wickets = match_list_year[(match_list_year['winner'] == selected_country)]['win_by_wickets'].sum()
        win_ratio = win_count / matches_played
        total_loss_by_runs = match_list_year[(match_list_year['first_innings'] == selected_country)
                                             & (match_list_year['winner'] != selected_country)]['win_by_runs'].sum()
        total_loss_by_wickets = match_list_year[(match_list_year['second_innings'] == selected_country)
                                                & (match_list_year['winner'] != selected_country)]['win_by_wickets'].sum()

        rank_dict = {'country': selected_country,
                     'win_ratio': win_ratio,
                     # 'total_win_by_runs':total_win_by_runs,
                     # 'total_loss_by_runs':total_loss_by_runs,
                     # 'total_win_by_wickets':total_win_by_wickets,
                     # 'total_loss_by_wickets':total_loss_by_wickets,
                     'effective_win_by_runs': total_win_by_runs - total_loss_by_runs,
                     'effective_win_by_wickets': total_win_by_wickets - total_loss_by_wickets,
                     'matches_played': matches_played,
                     'weighted_win_count': weighted_win_count,

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
        score_df[['win_ratio', 'effective_win_by_runs', 'effective_win_by_wickets', 'matches_played','weighted_win_count']]).sum(axis=1)

    score_scaler = MinMaxScaler(feature_range=(1, 10))
    score_df['score'] = score_scaler.fit_transform(score_df[['score']])
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


def create_batsman_rank_for_date(performance_cutoff_date_start, performance_cutoff_date_end, batsman_df):
    if not os.path.isdir(PREPROCESS_DATA_LOACATION):
        os.makedirs(PREPROCESS_DATA_LOACATION)

    scaler = MinMaxScaler()
    match_list_df = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION + os.sep + 'cricinfo_match_list.csv')
    country_rank_file = get_latest_rank_file('country',ref_date=performance_cutoff_date_end)
    if country_rank_file is None:
        raise Exception('Cannot create batsman rank-'+\
                        'Country rank not available for '+\
                        str(performance_cutoff_date_end.date()))
    country_rank = pd.read_csv(country_rank_file)


    # previous_country_rank_file = get_latest_rank_file('country', ref_date=performance_cutoff_date_start)

    year_batting_df = batsman_df[(batsman_df['date'] >= performance_cutoff_date_start)
                                      & (batsman_df['date'] <= performance_cutoff_date_end)
                                 & (batsman_df['did_bat']==1)]
    year_match_list_df = match_list_df[(match_list_df['date'] >= performance_cutoff_date_start)
                                      & (match_list_df['date'] <= performance_cutoff_date_end)]

    opponent_country_rank = pd.DataFrame(country_rank)
    opponent_country_rank = opponent_country_rank.rename(columns={'country':'opponent','score':'opponent_score'})[['opponent','opponent_score']]
    year_batting_df = year_batting_df.merge(opponent_country_rank,how='inner',on='opponent')

    country_list = list(year_batting_df['team'].unique())

    batsman_performance_list = []
    for selected_country in tqdm(country_list):
        # print(selected_country)
        country_batting_df = year_batting_df[year_batting_df['team']==selected_country]

        batsman_list = list(country_batting_df['name'].unique())

        for selected_batsman in tqdm(batsman_list):
            # print(selected_batsman)

            selected_batsman_df = country_batting_df[country_batting_df['name'] == selected_batsman]
            no_of_batsman_matches = selected_batsman_df['match_id'].nunique()
            total_runs = selected_batsman_df['runs'].sum()
            if selected_batsman_df['balls'].sum() !=0:
                run_rate = selected_batsman_df['runs'].sum() / selected_batsman_df['balls'].sum()
            else:
                run_rate = 0
            team_score = country_rank[country_rank['country'] == selected_country]['score'].values[0]
            # opponent_mean

            opponent_mean = selected_batsman_df['opponent_score'].mean()
            # matches_played = len(list(batsman_df['match_id'].unique()))
            player_of_the_match = selected_batsman_df['player_of_the_match'].sum()

            # winning contribution(effectiveness)-% of winning score
            # country_winning_runs=year_match_list_df[(year_match_list_df['winner']==selected_country)&
            #                    (year_match_list_df['first_innings']==selected_country)]['first_innings_run'].sum()\
            # +\
            #     year_match_list_df[(year_match_list_df['winner'] == selected_country) &
            #                    (year_match_list_df['second_innings'] == selected_country)]['second_innings_run'].sum()

            winning_match_id_list = list(year_match_list_df[(year_match_list_df['winner'] == selected_country)]['match_id'])

            batsman_winning_runs=selected_batsman_df[selected_batsman_df['match_id'].isin(winning_match_id_list)]['runs'].sum()
            country_winning_runs = country_batting_df[(country_batting_df['match_id'].isin(winning_match_id_list))]['runs'].sum()
            winning_contribution = batsman_winning_runs/country_winning_runs

            # run_rate_effectiveness
            # balls_played_by_country_in_winning_matches = year_match_list_df[(year_match_list_df['winner']==selected_country)&
            #                                                                 (year_match_list_df['first_innings']==selected_country)].shape[0]*50 + \
            #                                              year_match_list_df[(year_match_list_df['winner']==selected_country)&
            #                                                                 (year_match_list_df['first_innings']==selected_country)].shape[0]*50

            balls_played_by_country_in_winning_matches = country_batting_df[(country_batting_df['match_id'].isin(winning_match_id_list))]['runs'].sum()
            country_winning_run_rate = country_winning_runs/balls_played_by_country_in_winning_matches
            balls_played_by_batsman_in_winning_matches = selected_batsman_df[selected_batsman_df['match_id'].isin(winning_match_id_list)]['balls'].sum()
            batsman_winning_run_rate = batsman_winning_runs/balls_played_by_batsman_in_winning_matches

            run_rate_effectiveness = batsman_winning_run_rate / country_winning_run_rate

            #batting_std = batsman_df.groupby(['match_id'])['scored_runs'].sum().reset_index()['scored_runs'].std()

            #consistency = 1 / batting_std if batting_std != 0 else 1
            average_score = selected_batsman_df['runs'].mean()

            # correlation
            team_as_first_innings_df = year_match_list_df[year_match_list_df['first_innings']==selected_country]
            team_as_first_innings_df = team_as_first_innings_df.merge(selected_batsman_df,how='inner',on='match_id')
            team_as_first_innings_df = team_as_first_innings_df[["runs","first_innings_run"]]
            team_as_first_innings_df.rename(columns = {"first_innings_run":"innings_run"},inplace=True)

            team_as_second_innings_df = year_match_list_df[year_match_list_df['second_innings'] == selected_country]
            team_as_second_innings_df = team_as_second_innings_df.merge(selected_batsman_df, how='inner', on='match_id')
            team_as_second_innings_df = team_as_second_innings_df[["runs", "second_innings_run"]]
            team_as_second_innings_df.rename(columns={"second_innings_run": "innings_run"}, inplace=True)

            combined_df = pd.concat([team_as_first_innings_df,team_as_second_innings_df])
            player_runs = list(combined_df["runs"])
            team_runs = list(combined_df["innings_run"])
            if combined_df.shape[0]>=2:
                cor,_ =pearsonr(player_runs, team_runs)
            else:
                cor = 0.01



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
                'correlation':cor,
                #'consistency': consistency,
                'no_of_matches_batsman':no_of_batsman_matches
            }

            batsman_performance_list.append(batsman_dict)

    batsman_performance_df = pd.DataFrame(batsman_performance_list)
    batsman_performance_df.fillna(0, inplace=True)
    batsman_performance_df['batsman_score'] = scaler.fit_transform(
        batsman_performance_df.drop(columns=['batsman', 'country','no_of_matches_batsman','correlation'])).sum(axis=1)
    score_scaler = MinMaxScaler(feature_range=(1, 10))
    batsman_performance_df['batsman_score'] = score_scaler.fit_transform(batsman_performance_df[['batsman_score']])
    batsman_performance_df.sort_values('batsman_score', ascending=False, inplace=True)

    batsman_quantile_df = batsman_performance_df['batsman_score'].quantile([0.25, 0.5, 0.75]).reset_index()

    batsman_performance_df['batsman_quantile'] = batsman_performance_df['batsman_score'].apply(
        lambda x: get_quantile(batsman_quantile_df, x))

    batsman_performance_df.to_csv(PREPROCESS_DATA_LOACATION+os.sep+'batsman_rank_' + str(performance_cutoff_date_end.date()) + '.csv', index=False)


def create_batsman_rank(year_list,no_of_years=1):
    custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d")
    batsman_df = pd.read_csv(dl.CSV_LOAD_LOCATION+os.sep+'cricinfo_batting.csv',
                                parse_dates=['date'],
                                date_parser=custom_date_parser)
    batsman_df = batsman_df[~batsman_df['name'].isnull()]
    batsman_df = batsman_df[~batsman_df['runs'].isnull()]

    if year_list is None or len(year_list)==0:
        today = date.today()
        performance_cutoff_date_end = datetime(year=today.year, month=today.month, day=today.day)
        a_year = dateutil.relativedelta.relativedelta(years=no_of_years)
        performance_cutoff_date_start = performance_cutoff_date_end - a_year

        create_batsman_rank_for_date(performance_cutoff_date_start, performance_cutoff_date_end, batsman_df)

    else:
        for year in tqdm(year_list):
            for quarters in ranking_dates:
                print(" Batsman ranking for ",year + quarters)
                performance_cutoff_date_end = datetime.strptime(year + quarters, '%Y-%m-%d')
                performance_cutoff_date_start = cricutil.add_day_as_datetime(cricutil.substract_year_as_datetime
                                                                             (performance_cutoff_date_end, no_of_years),
                                                                             1)
                create_batsman_rank_for_date(performance_cutoff_date_start, performance_cutoff_date_end, batsman_df)


def create_bowler_rank_for_date(performance_cutoff_date_start, performance_cutoff_date_end, bowling_df):
    if not os.path.isdir(PREPROCESS_DATA_LOACATION):
        os.makedirs(PREPROCESS_DATA_LOACATION)

    scaler = MinMaxScaler()

    match_list_df = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION + os.sep + 'cricinfo_match_list.csv')
    country_rank_file = get_latest_rank_file('country', ref_date=performance_cutoff_date_end)
    if country_rank_file is None:
        raise Exception('Cannot create bowler rank-'+\
                        'Country rank not available for '+\
                        str(performance_cutoff_date_end.date()))
    country_rank = pd.read_csv(country_rank_file)

    year_bowling_df = bowling_df[(bowling_df['date'] >= performance_cutoff_date_start)
                                 & (bowling_df['date'] <= performance_cutoff_date_end)
                                 ]
    year_match_list_df = match_list_df[(match_list_df['date'] >= performance_cutoff_date_start)
                                       & (match_list_df['date'] <= performance_cutoff_date_end)]

    opponent_country_rank = pd.DataFrame(country_rank)
    opponent_country_rank = opponent_country_rank.rename(columns={'country': 'opponent', 'score': 'opponent_score'})[['opponent', 'opponent_score']]
    year_bowling_df = year_bowling_df.merge(opponent_country_rank, how='inner', on='opponent')

    country_list = list(year_bowling_df['team'].unique())


    bowler_performance_list = []
    for selected_country in tqdm(country_list):
        # print(selected_country)
        country_bowling_df = year_bowling_df[year_bowling_df['team'] == selected_country]

        bowler_list = list(country_bowling_df['name'].unique())

        for selected_bowler in tqdm(bowler_list):
            # print(selected_batsman)

            selected_bowler_df = country_bowling_df[country_bowling_df['name'] == selected_bowler]
            no_of_bowler_matches = selected_bowler_df['match_id'].nunique()
            total_runs = selected_bowler_df['runs'].sum()
            total_overs = selected_bowler_df['overs'].sum()


            if total_runs != 0:
                inverse_economy = total_overs/total_runs

            elif total_runs ==0 and total_overs>1:
                inverse_economy = 1
            else:
                inverse_economy = mean_inverse_econ



            # no_of_wickets,wicket_rate,wicket_per_runs
            no_of_wickets = selected_bowler_df['wickets'].sum()
            wickets_per_match = no_of_wickets / selected_bowler_df['match_id'].nunique()
            if total_runs ==0:
                total_runs=1
            wickets_per_run = no_of_wickets / total_runs

            team_score = country_rank[country_rank['country'] == selected_country]['score'].values[0]
            # opponent_mean
            opponent_mean = selected_bowler_df['opponent_score'].mean()

            # winning contribution(effectiveness)-% of wickets taken in winning matches
            country_win_list = list(year_match_list_df[(year_match_list_df['winner'] == selected_country)]['match_id'])

            winning_match_df = country_bowling_df[country_bowling_df['match_id'].isin(country_win_list)]

            if winning_match_df['wickets'].sum() != 0:
                winning_contribution = winning_match_df[winning_match_df['name'] == selected_bowler]['wickets'].sum() / \
                                       winning_match_df['wickets'].sum()
            else:
                winning_contribution = 0


            team_wicket_per_match = winning_match_df.groupby(['match_id'])['wickets'].sum().reset_index()['wickets'].mean()

            bowler_wicket_per_match = winning_match_df[winning_match_df['name'] == selected_bowler].groupby(['match_id'])['wickets'].sum().reset_index()['wickets'].mean()
            winning_wicket_per_match_contribution = bowler_wicket_per_match / team_wicket_per_match

            no_of_wins = winning_match_df[winning_match_df['name'] == selected_bowler]['match_id'].nunique()

            bowler_dict = {
                'bowler': selected_bowler,
                'country': selected_country,
                'inverse_economy': inverse_economy,
                'no_of_wickets': no_of_wickets,
                'wickets_per_match': wickets_per_match,
                'wickets_per_run': wickets_per_run,
                'no_of_wins': no_of_wins,
                'team_score': team_score,
                'opponent_mean': opponent_mean,
                'winning_contribution': winning_contribution,
                'winning_wicket_rate_contribution': winning_wicket_per_match_contribution,
                'no_of_matches_bowler': no_of_bowler_matches

            }

            bowler_performance_list.append(bowler_dict)

    bowler_performance_df = pd.DataFrame(bowler_performance_list)
    #print(bowler_performance_df.isin([np.inf, -np.inf]).sum())
    #print(bowler_performance_df.isnull().sum())
    bowler_performance_df.fillna(0, inplace=True)
    bowler_performance_df['bowler_score'] = scaler.fit_transform(
        bowler_performance_df.drop(columns=['bowler', 'country','no_of_matches_bowler'])).sum(axis=1)
    score_scaler = MinMaxScaler(feature_range=(1, 10))
    bowler_performance_df['bowler_score'] = score_scaler.fit_transform(bowler_performance_df[['bowler_score']])
    bowler_performance_df.sort_values('bowler_score', ascending=False, inplace=True)

    bowler_quantile_df = bowler_performance_df['bowler_score'].quantile([0.25, 0.5, 0.75]).reset_index()
    bowler_performance_df['bowler_quantile'] = bowler_performance_df['bowler_score'].apply(
        lambda x: get_quantile(bowler_quantile_df, x))
    bowler_performance_df.to_csv(PREPROCESS_DATA_LOACATION+os.sep+'bowler_rank_' + str(performance_cutoff_date_end.date()) + '.csv', index=False)

def create_bowler_rank(year_list,no_of_years=1):
    custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d")
    bowling_list_df = pd.read_csv(dl.CSV_LOAD_LOCATION+os.sep+'cricinfo_bowling.csv',
                                parse_dates=['date'],
                                date_parser=custom_date_parser)

    if year_list is None or len(year_list)==0:
        today = date.today()
        performance_cutoff_date_end = datetime(year=today.year, month=today.month, day=today.day)
        a_year = dateutil.relativedelta.relativedelta(years=no_of_years)
        performance_cutoff_date_start = performance_cutoff_date_end - a_year

        create_bowler_rank_for_date(performance_cutoff_date_start, performance_cutoff_date_end, bowling_list_df)

    else:
        for year in tqdm(year_list):
            for quarters in ranking_dates:
                print(" Bowler ranking for ", year + quarters)
                performance_cutoff_date_end = datetime.strptime(year + quarters, '%Y-%m-%d')
                performance_cutoff_date_start = cricutil.add_day_as_datetime(cricutil.substract_year_as_datetime
                                                                             (performance_cutoff_date_end, no_of_years),
                                                                             1)
                create_bowler_rank_for_date(performance_cutoff_date_start, performance_cutoff_date_end, bowling_list_df)


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
            for quarters in ranking_dates:
                performance_cutoff_date_end = datetime.strptime(year + quarters, '%Y-%m-%d')
                performance_cutoff_date_start = cricutil.add_day_as_datetime(cricutil.substract_year_as_datetime
                                                                             (performance_cutoff_date_end, no_of_years),
                                                                             1)
                create_location_rank_for_date(performance_cutoff_date_start, performance_cutoff_date_end, match_list_df)



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


if __name__=='__main__':
    rank()


