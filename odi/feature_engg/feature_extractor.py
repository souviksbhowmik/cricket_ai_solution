import pandas as pd
import numpy as np
from odi.data_loader import data_loader as dl
from odi.preprocessing import rank
from odi.inference import prediction
from odi.retrain import create_train_test as ctt
from odi.model_util import odi_util as outil
from odi.feature_engg import util as cricutil
import os
from datetime import datetime,date
import dateutil
import pickle
import numpy as np
import math
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LinearRegression

SELECTED_FIRST_INNINGS_FEATURE_LIST_CACHE = None
SELECTED_SECOND_INNINGS_FEATURE_LIST_CACHE = None
TEAM_OPPONENT_LOCATION_EMBEDDING_MODEL_CACHE = None
TEAM_OPPONENT_LOCATION_EMBEDDING_MODEL_CACHE = None
TEAM_EMBEDDING_MODEL_CACHE = None
COUNTRY_ENC_MAP_CACHE = None
LOC_ENC_MAP_CACHE = None
BATSMAN_ENC_MAP_CACHE = None
BOWLER_ENC_MAP_CACHE = None
LOC_ENC_MAP_FOR_BATSMAN_CACHE = None
BATSMAN_EMBEDDING_MODEL_CACHE = None
BATSMAN_ONLY_EMBEDDING_MODEL_CACHE = None
BATSMAN_EMBEDDING_RUN_MODEL_CACHE = None

ADVERSARIAL_BATSMAN_MODEL_CACHE = None
ADVERSARIAL_BOWLER_MODEL_CACHE = None
ADVERSARIAL_LOCATION_MODEL_CACHE = None
#ADVERSARIAL_LOCATION_MODEL_CACHE = None

# NO_OF_WICKETS = 0


def get_trend(input_df, team_or_opponent, team_name, target_field):
    if input_df.shape[0]==0:
        return None, None,None,None
    input_df.rename(columns={'winner': 'winning_team'}, inplace=True)

    selected_match_id_list = list(input_df['match_id'])
    match_detail_list = []
    for match_id in selected_match_id_list:
        match_info = pd.read_csv(dl.CSV_LOAD_LOCATION+os.sep
                                 + str(match_id) + '.csv')
        match_detail_list.append(match_info)
    match_detail_df = pd.concat(match_detail_list)
    match_detail_df.fillna('NA', inplace=True)

    match_detail_df = input_df.merge(match_detail_df, how='inner', on='match_id')

    sorted_df = match_detail_df[match_detail_df[team_or_opponent].isin(team_name)].groupby('match_id').agg(
        {'date': 'min', target_field: 'sum'}).reset_index()
    sorted_df.sort_values('date', inplace=True)

    y = np.array(sorted_df[target_field])
    x = np.array(range(sorted_df.shape[0])).reshape(-1, 1) + 1
    linear_trend_model = LinearRegression()
    linear_trend_model.fit(x, y)
    next_instance_num = x.shape[0] + 1

    base = linear_trend_model.intercept_
    trend = linear_trend_model.coef_[0]
    y_pred = linear_trend_model.predict(x)
    mean_error = np.mean(y-y_pred)
    #trend_predict = linear_trend_model.predict(np.array([next_instance_num]).reshape(-1, 1))[0]
    trend_predict = linear_trend_model.predict(np.array([next_instance_num]).reshape(-1, 1))[0]+mean_error
    mean = sorted_df[target_field].mean()

    return base, trend, trend_predict, mean


def get_specified_summary_df(match_summary_df=None,ref_date=None,no_of_years=None):

    if match_summary_df is None:
        custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d")
        match_summary_df = pd.read_csv(dl.CSV_LOAD_LOCATION + os.sep + 'match_list.csv',
                                       parse_dates=['date'], date_parser=custom_date_parser)
    if ref_date is None:
        today = date.today()
        ref_date = datetime(year=today.year, month=today.month, day=today.day)

    match_summary_df = match_summary_df[match_summary_df['date'] < ref_date]

    if no_of_years is not None:
        a_year = dateutil.relativedelta.relativedelta(years=no_of_years)
        begin_date = ref_date - a_year
        match_summary_df = match_summary_df[match_summary_df['date'] >= begin_date]

    return match_summary_df


def get_trend_with_opponent(team,opponent,match_summary_df=None,ref_date=None,no_of_years=None):
    match_summary_df = get_specified_summary_df(match_summary_df=match_summary_df,
                                                ref_date=ref_date,
                                                no_of_years=no_of_years)

    last_5_opponent = match_summary_df[(match_summary_df['first_innings'] == team)
                                       & (match_summary_df['second_innings'] == opponent)
                                       ].sort_values('date', ascending=False).head(5)

    return get_trend(last_5_opponent, 'team', [team], 'total')


def get_trend_at_location(team,location,match_summary_df=None,ref_date=None,no_of_years=None):
    match_summary_df = get_specified_summary_df(match_summary_df=match_summary_df,
                                                ref_date=ref_date,
                                                no_of_years=no_of_years)

    last_5_location = match_summary_df[(match_summary_df['first_innings'] == team)
                                       & (match_summary_df['location'] == location)
                                       ].sort_values('date', ascending=False).head(5)

    return get_trend(last_5_location, 'team', [team], 'total')


def get_trend_recent(team,match_summary_df=None,ref_date=None,no_of_years=None):
    match_summary_df = get_specified_summary_df(match_summary_df=match_summary_df,
                                                ref_date=ref_date,
                                                no_of_years=no_of_years)

    last_5_match = match_summary_df[(match_summary_df['first_innings'] == team)
                                    ].sort_values('date', ascending=False).head(5)

    return get_trend(last_5_match, 'team', [team], 'total')


def get_country_score(country, ref_date=None):

    country_rank_file = rank.get_latest_rank_file('country',ref_date=ref_date)
    country_rank_df = pd.read_csv(country_rank_file)
    if country_rank_df[country_rank_df['country'] == country].shape[0]==0:
        raise Exception('Country score not available')
    score = country_rank_df[country_rank_df['country'] == country]['score'].values[0]
    quantile = country_rank_df[country_rank_df['country'] == country]['country_quantile'].values[0]

    return score,quantile


def complete_batting_order(country,batsman_list,bowling_list,ref_date=None,no_of_batsman=7):

    if len(batsman_list)>=no_of_batsman:
        return batsman_list

    batsman_rank_file = rank.get_latest_rank_file('batsman', ref_date=ref_date)
    batsman_rank_df = pd.read_csv(batsman_rank_file)
    batsman_rank_df = batsman_rank_df[batsman_rank_df['country']==country]

    unique_bowling_list = []
    for bowler in bowling_list:
        if bowler not in batsman_list:
            unique_bowling_list.append(bowler)

    if len(batsman_list)+len(unique_bowling_list) ==11:# if all 11 player are available in balling and batting
        # order the batting order of bowlers
        all_rounder_df = batsman_rank_df[batsman_rank_df['batsman'].isin(unique_bowling_list)]
        if all_rounder_df.shape[0]==0:# if none of the bowlers have any batting record
            return batsman_list + unique_bowling_list

        else:# if some bowlers have batting record - sort them by number of matches
            all_rounder_df = all_rounder_df.sort_values(['no_of_matches_batsman'],ascending=False)
            sorted_bowler_list = list(all_rounder_df['batsman'].unique())
            if len(batsman_list) + len(sorted_bowler_list)<no_of_batsman: # if some bowler is left since he did not have batting record
                batsman_list = batsman_list+sorted_bowler_list
                for bowler in unique_bowling_list:
                    if bowler not in sorted_bowler_list:
                        batsman_list.append(bowler)
                return batsman_list
            else:# all bowlers had batting records
                return batsman_list+sorted_bowler_list
    else:# there are players missing - not in batting or bowling
        # fill in the gap
        missing_players = 11 - (len(batsman_list)+len(unique_bowling_list))
        shortage_batsman = no_of_batsman - len(batsman_list)
        all_rounder_df = batsman_rank_df[batsman_rank_df['batsman'].isin(unique_bowling_list)]
        all_non_listed_batsman = batsman_rank_df[~(batsman_rank_df['batsman'].isin(unique_bowling_list))
                                                 & ~(batsman_rank_df['batsman'].isin(batsman_list))]

        if all_rounder_df.shape[0]>0 & all_non_listed_batsman.shape[0]>0: # get the batting_match_count of bowlers who have batting records
            all_rounder_df = all_rounder_df.sort_values(['no_of_matches_batsman'],ascending=False)
            all_non_listed_batsman = all_non_listed_batsman.sort_values(['no_of_matches_batsman'],ascending=False)

            i=0
            j=0
            while i!=shortage_batsman & j!=shortage_batsman & i!=missing_players & i<all_non_listed_batsman.shape[0] & j<all_rounder_df.shape[0]:
                nom_i = all_non_listed_batsman.iloc[i]['no_of_matches_batsman']
                nom_j = all_rounder_df.iloc[j]['no_of_matches_batsman']

                if nom_j>= nom_i:
                    batsman_list.append(all_rounder_df.iloc[j]['batsman'])
                    j=j+1
                else:
                    batsman_list.append(all_non_listed_batsman.iloc[i]['batsman'])
                    i = i + 1

            if (i==missing_players or i==all_non_listed_batsman.shape[0])  & (i+j)<shortage_batsman:
                for z in range(j,shortage_batsman-(i+j)):
                    batsman_list.append(all_rounder_df.iloc[z]['batsman'])
            elif j==all_rounder_df.shape[0] & (i+j)<shortage_batsman:
                for z in range(i,shortage_batsman-(i+j)):
                    batsman_list.append(all_non_listed_batsman.iloc[z]['batsman'])
            else:
                pass

            return batsman_list

        elif all_rounder_df.shape[0] >0 :
            all_rounder_df = all_rounder_df.sort_values(['no_of_matches_batsman'], ascending=False)
            batsman_list = batsman_list + list(all_rounder_df.head(shortage_batsman)['batsman'])
            return batsman_list
        else:
            all_non_listed_batsman = all_non_listed_batsman.sort_values(['no_of_matches_batsman'], ascending=False)
            batsman_list = batsman_list + list(all_non_listed_batsman.head(shortage_batsman)['batsman'])

            return batsman_list


def get_weighted_score(row):
    return row['batsman_score']*math.sqrt(row['no_of_matches_batsman']/row['matches_played'])


# get batsman sum weighted by matches played
def get_batsman_mean_max(country,batsman_list,ref_date=None,no_of_batsman=7):

    batsman_rank_file = rank.get_latest_rank_file('batsman',ref_date=ref_date)
    country_rank_file = rank.get_latest_rank_file('country',ref_date=ref_date)
    batsman_rank_df = pd.read_csv(batsman_rank_file)
    batsman_rank_df = batsman_rank_df[batsman_rank_df['country'] == country]
    country_rank_df = pd.read_csv(country_rank_file)[['country','matches_played']]
    batsman_rank_df = batsman_rank_df.merge(country_rank_df, on='country', how ='inner')
    adjusted_batsman_list = []
    if len(batsman_list)<no_of_batsman:
        raise Exception('Not enough batsman information')

    for i in range(no_of_batsman):
        adjusted_batsman_list.append(batsman_list[i])

    selected_batsman_df = batsman_rank_df[(batsman_rank_df['batsman'].isin(adjusted_batsman_list))]

    if selected_batsman_df.shape[0]==0:
        raise Exception('No batsman score is available for '+country)

    #selected_batsman_df['weighted_score'] = selected_batsman_df.apply(get_weighted_score,axis=1)
    #selected_batsman_df.loc[:,'weighted_score'] = selected_batsman_df.apply(get_weighted_score,axis=1)

    batsman_max = selected_batsman_df['batsman_score'].max()
    batsman_mean = selected_batsman_df['batsman_score'].max()
    batsman_sum = selected_batsman_df['batsman_score'].sum()
    # batsman_max = selected_batsman_df['weighted_score'].max()
    # batsman_mean = selected_batsman_df['weighted_score'].max()
    # batsman_sum = selected_batsman_df['weighted_score'].sum()


    batsman_quantile_mean = selected_batsman_df['batsman_quantile'].mean()
    batsman_quantile_max = selected_batsman_df['batsman_quantile'].max()
    batsman_quantile_sum = selected_batsman_df['batsman_quantile'].sum()
    return batsman_mean,batsman_max,batsman_sum,batsman_quantile_mean,batsman_quantile_max,batsman_quantile_sum


### calculation of weighted sum and mean
# def get_batsman_mean_max(country,batsman_list,ref_date=None,no_of_batsman=7):
#     # batsman_list=get_top_n_batsman(batsman_list, country, n=no_of_batsman, ref_date=ref_date)
#     # global NO_OF_WICKETS
#     # print("setting number of wickets to ", NO_OF_WICKETS)
#     # if NO_OF_WICKETS!=0:
#     #     #print("setting number of wickets to ",NO_OF_WICKETS)
#     #     no_of_batsman=int(NO_OF_WICKETS)
#     batsman_rank_file = rank.get_latest_rank_file('batsman',ref_date=ref_date)
#     batsman_rank_df = pd.read_csv(batsman_rank_file)
#     batsman_rank_df = batsman_rank_df[batsman_rank_df['country'] == country]
#     reduction_dict = pickle.load(open(os.path.join(outil.DEV_DIR, outil.SCORE_MEAN_REDUCTION_FACTOR), 'rb'))
#
#     selected_batsman_df = batsman_rank_df[(batsman_rank_df['batsman'].isin(batsman_list))\
#                                    & (batsman_rank_df['country']==country)]
#     if selected_batsman_df.shape[0]==0:
#         raise Exception('No batsman score is available for '+country)
#     selected_batsman_df = selected_batsman_df.sort_values('batsman_score',ascending=False)
#
#     batsman_max = selected_batsman_df['batsman_score'].max()
#
#
#     #calculating weighted sum and mean
#     batsman_sum = 0
#     weighted_denom = 0
#     unweighted_mean = selected_batsman_df['batsman_score'].mean()
#
#     for idx,batsman in enumerate(batsman_list):
#         wt_idx = 11-idx
#         wt = math.log(math.ceil((wt_idx)/2)+1)
#         if selected_batsman_df[selected_batsman_df['batsman']==batsman]['batsman_score'].shape[0]!=0:
#             score = selected_batsman_df[selected_batsman_df['batsman']==batsman]['batsman_score'].values[0]
#             batsman_sum = batsman_sum + wt * score
#             weighted_denom = weighted_denom + wt
#         if idx==no_of_batsman-1:
#             break
#
#
#     if len(batsman_list)<no_of_batsman:
#         last_available = len(batsman_list)
#         for target in range(no_of_batsman-last_available):
#             current = last_available+target+1
#             previous = last_available+target
#             unweighted_mean = unweighted_mean/reduction_dict[str(previous)+"_by_"+str(current)]
#
#             wt_idx = (11-last_available-target)
#             wt = math.log(math.ceil((wt_idx) / 2) + 1)
#             batsman_sum = batsman_sum+wt*unweighted_mean
#
#     batsman_mean = batsman_sum/11
#
#     batsman_quantile_mean = selected_batsman_df['batsman_quantile'].mean()
#     batsman_quantile_max = selected_batsman_df['batsman_quantile'].max()
#     batsman_quantile_sum = selected_batsman_df['batsman_quantile'].sum()
#     return batsman_mean,batsman_max,batsman_sum,batsman_quantile_mean,batsman_quantile_max,batsman_quantile_sum

## calculating unweighted mean and extrapoliting mean by reduction factor
# def get_batsman_mean_max(country, batsman_list, ref_date=None, no_of_batsman=7):
#     global NO_OF_WICKETS
#     if NO_OF_WICKETS!=0:
#         no_of_batsman=NO_OF_WICKETS
#     batsman_list = get_top_n_batsman(batsman_list, country, n=no_of_batsman, ref_date=ref_date)
#     batsman_rank_file = rank.get_latest_rank_file('batsman', ref_date=ref_date)
#     batsman_rank_df = pd.read_csv(batsman_rank_file)
#     batsman_rank_df = batsman_rank_df[batsman_rank_df['country'] == country]
#     reduction_dict = pickle.load(open(os.path.join(outil.DEV_DIR, outil.SCORE_MEAN_REDUCTION_FACTOR), 'rb'))
#
#     selected_batsman_df = batsman_rank_df[(batsman_rank_df['batsman'].isin(batsman_list)) \
#                                           & (batsman_rank_df['country'] == country)]
#     if selected_batsman_df.shape[0] == 0:
#         raise Exception('No batsman score is available for ' + country)
#     selected_batsman_df = selected_batsman_df.sort_values('batsman_score', ascending=False)
#
#     # batsman_mean = selected_batsman_df.head(7)['batsman_score'].mean()
#     batsman_max = selected_batsman_df['batsman_score'].max()
#     batsman_mean = selected_batsman_df.head(no_of_batsman)['batsman_score'].mean()
#     batsman_sum = selected_batsman_df.head(no_of_batsman)['batsman_score'].sum()
#
#     # calculating weighted sum and mean
#     available_batsman = selected_batsman_df.shape[0]
#
#     if available_batsman < no_of_batsman:
#         last_available = available_batsman
#         for target in range(no_of_batsman - last_available):
#             current = last_available + target + 1
#             previous = last_available + target
#             batsman_mean = batsman_mean / reduction_dict[str(previous) + "_by_" + str(current)]
#             batsman_sum = batsman_sum+batsman_mean
#
#
#
#
#     batsman_quantile_mean = selected_batsman_df['batsman_quantile'].mean()
#     batsman_quantile_max = selected_batsman_df['batsman_quantile'].max()
#     batsman_quantile_sum = selected_batsman_df['batsman_quantile'].sum()
#     return batsman_mean,batsman_max,batsman_sum,batsman_quantile_mean,batsman_quantile_max,batsman_quantile_sum


def get_batsman_vector(country,batsman_list,ref_date=None,no_of_batsman=9):

    batsman_rank_file = rank.get_latest_rank_file('batsman',ref_date=ref_date)
    batsman_rank_df = pd.read_csv(batsman_rank_file)
    batsman_rank_df = batsman_rank_df[batsman_rank_df['country']==country]
    if len(batsman_list) < no_of_batsman:
        # batsman_rank_df.sort_values("batsman_score", ascending=False, inplace=True)
        # all_batsman = list(batsman_rank_df['batsman'])
        # search_index = 0
        # while len(batsman_list) <no_of_batsman and search_index<len(all_batsman):
        #     if all_batsman[search_index] not in batsman_list:
        #         batsman_list.append(all_batsman[search_index])
        #     search_index = search_index+1
        # if len(batsman_list)<no_of_batsman:
        #     raise Exception("not enough batsman")
        raise Exception("not enough batsman information")

    temp_df = pd.DataFrame()
    temp_df['batsman'] = batsman_list
    temp_df['country'] = country

    temp_df = temp_df.merge(batsman_rank_df,on=["batsman","country"],how="inner")
    score_list = list(temp_df['batsman_quantile'])
    if len(score_list)<no_of_batsman:
        #print("adjustement for new batsman")
        no_of_shortage = no_of_batsman-len(score_list)
        for i in range (no_of_shortage):
            score_list.append(np.array(score_list).mean())

    #print("score_list ",score_list)
    return np.array(score_list)[:8]


def get_bowler_mean_max(country,bowler_list,ref_date=None,no_of_bowler=6):
    bowler_list = get_top_n_bowlers(bowler_list, country, n=no_of_bowler, ref_date=ref_date)
    bowler_rank_file = rank.get_latest_rank_file('bowler',ref_date=ref_date)
    bowler_rank_df = pd.read_csv(bowler_rank_file)
    if bowler_rank_df[(bowler_rank_df['bowler'].isin(bowler_list))\
                                  & (bowler_rank_df['country']==country)].shape[0]==0:
        raise Exception('No bowler score available for '+country)
    bowler_mean = bowler_rank_df[(bowler_rank_df['bowler'].isin(bowler_list))\
                                  & (bowler_rank_df['country']==country)].head(no_of_bowler)['bowler_score'].mean()
    bowler_quantile_mean = bowler_rank_df[(bowler_rank_df['bowler'].isin(bowler_list)) \
                                 & (bowler_rank_df['country'] == country)].head(no_of_bowler)['bowler_quantile'].mean()
    bowler_max = bowler_rank_df[(bowler_rank_df['bowler'].isin(bowler_list))\
                                  & (bowler_rank_df['country']==country)].head(no_of_bowler)['bowler_score'].max()
    bowler_quantile_max = bowler_rank_df[(bowler_rank_df['bowler'].isin(bowler_list)) \
                                & (bowler_rank_df['country'] == country)].head(no_of_bowler)['bowler_quantile'].max()
    bowler_sum = bowler_rank_df[(bowler_rank_df['bowler'].isin(bowler_list)) \
                                & (bowler_rank_df['country'] == country)].head(no_of_bowler)['bowler_score'].sum()
    bowler_quantile_sum = bowler_rank_df[(bowler_rank_df['bowler'].isin(bowler_list)) \
                                & (bowler_rank_df['country'] == country)].head(no_of_bowler)['bowler_quantile'].sum()
    return bowler_mean,bowler_max,bowler_sum,bowler_quantile_mean,bowler_quantile_max,bowler_quantile_sum


# def get_recent_batsman_score(country,batsman_list,ref_date=None):
#     match_list_df = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION + os.sep + 'match_list.csv')
#
#     pass
#
#
# def get_recent_bowler_score(country,batsman_list,ref_date=None):
#     pass

def get_location_mean(location,innings,ref_date=None):
    location_rank_file = rank.get_latest_rank_file('location',ref_date=ref_date)
    location_rank_df = pd.read_csv(location_rank_file)
    location_list = list(location_rank_df['location'].unique())
    if location not in location_list:
        location = 'default'
    location_mean = location_rank_df[(location_rank_df['location']==location) & (location_rank_df['innings']==innings)]['total_run'].values[0]
    return location_mean

def get_overall_means(team,location,ref_date,opponent):
    match_list_df = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION + os.sep + 'match_list.csv')
    match_stats_df = pd.read_csv(dl.CSV_LOAD_LOCATION + os.sep + 'match_stats.csv')
    match_df = match_list_df.merge(match_stats_df,on="match_id", how="inner")

    location_mean = match_df[(match_df['date']<ref_date) & (match_df['location']==location) & (match_df['first_innings']==match_df['team_statistics'])]['total_run'].mean()
    team_location_mean = match_df[(match_df['date']<ref_date) & (match_df['location']==location) & (match_df['first_innings']==team)]['total_run'].mean()

    opponent_mean = match_df[(match_df['date']<ref_date) &
                             (match_df['location']==location) &
                             (match_df['first_innings']==match_df['team_statistics']) &
                             (match_df['second_innings']==opponent)]['total_run'].mean()

    team_opponent_mean = match_df[(match_df['date']<ref_date) &
                             (match_df['location']==location) &
                             (match_df['first_innings']==team) &
                             (match_df['second_innings']==opponent)]['total_run'].mean()
    if location_mean == 0 or math.isnan(location_mean):
        location_mean = 250

    if team_location_mean == 0 or math.isnan(team_location_mean):
        team_location_mean = 250

    if opponent_mean == 0 or math.isnan(opponent_mean):
        opponent_mean = 250

    if team_opponent_mean == 0 or math.isnan(team_opponent_mean):
        team_opponent_mean =250

    return location_mean, team_location_mean, opponent_mean, team_opponent_mean

def get_instance_feature_dict(team, opponent, location, team_player_list, opponent_player_list, ref_date=None,no_of_years=None):

    team_score,team_quantile = get_country_score(team, ref_date=ref_date)
    opponent_score,opponent_quantile = get_country_score(opponent, ref_date=ref_date)
    batsman_mean,batsman_max,batsman_sum,batsman_quantile_mean,batsman_quantile_max,batsman_quantile_sum= get_batsman_mean_max(team, team_player_list, ref_date=ref_date)
    bowler_mean, bowler_max, bowler_sum, bowler_quantile_mean, bowler_quantile_max, bowler_quantile_sum= get_bowler_mean_max(opponent, opponent_player_list, ref_date=ref_date)
    #location_overall_mean = get_location_mean(location,"first")
    #batting_score_list = get_batsman_vector(team,team_player_list,ref_date=ref_date)

    current_base, current_trend, current_trend_predict, current_mean =\
        get_trend_recent(team,ref_date=ref_date,no_of_years=no_of_years)

    if current_base is None:
        raise Exception('Team history unavailable')

    location_base, location_trend, location_trend_predict, location_mean =\
        get_trend_at_location(team,location,ref_date=ref_date,no_of_years=no_of_years)

    if location_base is None:
        location_base, location_trend, location_trend_predict, location_mean = \
            (current_base, current_trend, current_trend_predict, current_mean)

    opponent_base, opponent_trend, opponent_trend_predict, opponent_mean = \
        get_trend_with_opponent(team,opponent,ref_date=ref_date,no_of_years=no_of_years)

    if opponent_base is None:
        opponent_base, opponent_trend, opponent_trend_predict, opponent_mean = \
            (current_base, current_trend, current_trend_predict, current_mean)

    #overall_location_mean, overall_team_location_mean, overall_opponent_mean, overall_team_opponent_mean = get_overall_means(team,location,ref_date,opponent)
    #standard_mean = 250

    feature_dict = {
        'team': team,
        'opponent': opponent,
        'location': location,
        'team_score': team_score,
        #'team_quantile':team_quantile,
        'opponent_score': opponent_score,
        #'opponent_quantile':opponent_quantile,
        'opponent_base': opponent_base,
        'opponent_trend': opponent_trend,
        'opponent_trend_predict': opponent_trend_predict,
        'opponent_mean': opponent_mean,
        'location_base': location_base,
        'location_trend': location_trend,
        'location_trend_predict': location_trend_predict,
        'location_mean': location_mean,
        'current_base': current_base,
        'current_trend': current_trend,
        'current_trend_predict': current_trend_predict,
        'current_mean': current_mean,
        'batsman_mean': batsman_mean,
        'batsman_max': batsman_max,
        'batsman_sum':batsman_sum,
        #'batsman_quantile_mean': batsman_quantile_mean,
        #'batsman_quantile_max': batsman_quantile_max,
        #'batsman_quantile_sum': batsman_quantile_sum,
        'bowler_mean': bowler_mean,
        'bowler_max': bowler_max,
        'bowler_sum': bowler_sum
        #'bowler_quantile_mean': bowler_quantile_mean,
        #'bowler_quantile_max': bowler_quantile_max,
        #'bowler_quantile_sum': bowler_quantile_sum
        # 'bat_ball_ratio':batsman_sum/bowler_sum
        # 'location_overall_mean':location_overall_mean
        #'standard_mean':standard_mean,
        #'overall_location_mean':overall_location_mean,
        #'overall_team_location_mean':overall_team_location_mean,
        #'overall_opponent_mean':overall_opponent_mean,
        #'overall_team_opponent_mean':overall_team_opponent_mean
        # 'location_factor':overall_team_location_mean/overall_location_mean,
        # 'overall_location_adder': overall_team_location_mean - overall_location_mean,
        # 'opponent_factor':overall_team_opponent_mean/overall_opponent_mean,
        # 'opponent_adder':overall_team_opponent_mean - overall_opponent_mean


    }

    # batting_score_list = list(batting_score_list)
    # for idx,score in enumerate(batting_score_list):
    #     feature_dict['batsman_'+str(idx)]=score

    #print(feature_dict)
    return feature_dict


def get_first_innings_feature_vector(team, opponent, location, team_player_list, opponent_player_list, ref_date=None, no_of_years=None):
    """ this is not scaled"""
    global SELECTED_FIRST_INNINGS_FEATURE_LIST_CACHE
    feature_dict = get_instance_feature_dict(team, opponent, location,
                                             team_player_list, opponent_player_list, ref_date=ref_date, no_of_years=no_of_years)

    if SELECTED_FIRST_INNINGS_FEATURE_LIST_CACHE is None:
        SELECTED_FIRST_INNINGS_FEATURE_LIST_CACHE = pickle.load(open(outil.MODEL_DIR + os.sep + outil.FIRST_INNINGS_FEATURE_PICKLE, 'rb'))

    selected_feature_list = SELECTED_FIRST_INNINGS_FEATURE_LIST_CACHE

    feature_values = list()
    for feaure_name in selected_feature_list:
        feature_values.append(feature_dict[feaure_name])

    return np.array(feature_values)


def get_second_innings_feature_vector(target, team, opponent, location, team_player_list, opponent_player_list, ref_date=None, no_of_years=None):
    """ this is not scaled"""
    global SELECTED_SECOND_INNINGS_FEATURE_LIST_CACHE
    feature_dict = get_instance_feature_dict(team, opponent, location,
                                             team_player_list, opponent_player_list, ref_date=ref_date, no_of_years=no_of_years)

    feature_dict['target_score'] = target
    if SELECTED_SECOND_INNINGS_FEATURE_LIST_CACHE == None:
        SELECTED_SECOND_INNINGS_FEATURE_LIST_CACHE = pickle.load(open(outil.MODEL_DIR + os.sep + outil.SECOND_INNINGS_FEATURE_PICKLE, 'rb'))
    selected_feature_list = SELECTED_SECOND_INNINGS_FEATURE_LIST_CACHE


    feature_values = list()
    for feaure_name in selected_feature_list:
        feature_values.append(feature_dict[feaure_name])

    return np.array(feature_values)


def get_team_opponent_location_embedding(team,opponent,location):
    global TEAM_OPPONENT_LOCATION_EMBEDDING_MODEL_CACHE
    global COUNTRY_ENC_MAP_CACHE
    global LOC_ENC_MAP_CACHE

    if TEAM_OPPONENT_LOCATION_EMBEDDING_MODEL_CACHE is None:
        TEAM_OPPONENT_LOCATION_EMBEDDING_MODEL_CACHE = outil.load_keras_model(outil.MODEL_DIR \
                                                              + os.sep \
                                                              + outil.TEAM_OPPONENT_LOCATION_EMBEDDING_MODEL)

    team_opponent_location_embedding = TEAM_OPPONENT_LOCATION_EMBEDDING_MODEL_CACHE

    if COUNTRY_ENC_MAP_CACHE is None:
        COUNTRY_ENC_MAP_CACHE = pickle.load(open(outil.MODEL_DIR \
                                   + os.sep \
                                   + outil.COUNTRY_ENCODING_MAP,'rb'))
    country_enc_map = COUNTRY_ENC_MAP_CACHE

    if LOC_ENC_MAP_CACHE is None:
        LOC_ENC_MAP_CACHE = pickle.load(open(outil.MODEL_DIR \
                                             + os.sep \
                                             + outil.LOC_ENCODING_MAP,'rb'))
    loc_enc_map = LOC_ENC_MAP_CACHE

    if team not in country_enc_map or opponent not in country_enc_map:
        raise Exception('Team or opponent not available')
    team_oh_v = np.array(country_enc_map[team]).reshape(1, -1)
    opponent_oh_v = np.array(country_enc_map[opponent]).reshape(1, -1)
    if location not in loc_enc_map:
        location = get_similar_location(location).strip()
    loc_oh_v = np.array(loc_enc_map[location]).reshape(1, -1)

    country_embedding = team_opponent_location_embedding.predict([team_oh_v, opponent_oh_v, loc_oh_v]).reshape(-1)

    return country_embedding


def get_oh_pos(pos):
    vec=np.zeros((11)).astype(int)
    vec[pos-1]=1
    return vec

def get_country_embedding(team):

    global TEAM_EMBEDDING_MODEL_CACHE
    global COUNTRY_ENC_MAP_CACHE

    if TEAM_EMBEDDING_MODEL_CACHE is None:
        TEAM_EMBEDDING_MODEL_CACHE = outil.load_keras_model(outil.MODEL_DIR \
                                                              + os.sep \
                                                              + outil.TEAM_EMBEDDING_MODEL)

    team_embedding = TEAM_EMBEDDING_MODEL_CACHE

    if COUNTRY_ENC_MAP_CACHE is None:
        COUNTRY_ENC_MAP_CACHE = pickle.load(open(outil.MODEL_DIR \
                                   + os.sep \
                                   + outil.COUNTRY_ENCODING_MAP,'rb'))
    country_enc_map = COUNTRY_ENC_MAP_CACHE

    loc_enc_map = LOC_ENC_MAP_CACHE

    if team not in country_enc_map:
        raise Exception('Team not available')
    team_oh_v = np.array(country_enc_map[team]).reshape(1, -1)

    country_embedding = team_embedding.predict([team_oh_v]).reshape(-1)

    return country_embedding


def get_batsman_embedding(batsman_list,team,opponent,location,no_of_batsman=7,ref_date=None):
    global BATSMAN_EMBEDDING_MODEL_CACHE
    global COUNTRY_ENC_MAP_CACHE
    global BATSMAN_ENC_MAP_CACHE
    global LOC_ENC_MAP_FOR_BATSMAN_CACHE

    ## adjust list of batsman
    batsman_rank_file = rank.get_latest_rank_file('batsman', ref_date=ref_date)
    batsman_rank_df = pd.read_csv(batsman_rank_file)
    batsman_rank_df = batsman_rank_df[batsman_rank_df['country'] == team]
    # if len(batsman_list) < no_of_batsman:
    #     batsman_rank_df.sort_values("batsman_score", ascending=False, inplace=True)
    #     all_batsman = list(batsman_rank_df['batsman'])
    #     search_index = 0
    #     while len(batsman_list) < no_of_batsman and search_index < len(all_batsman):
    #         if all_batsman[search_index] not in batsman_list:
    #             batsman_list.append(all_batsman[search_index])
    #         search_index = search_index + 1
    #     if len(batsman_list) < no_of_batsman:
    #         raise Exception("not enough batsman")
    #     raise Exception("not enough batsman information")

    if BATSMAN_EMBEDDING_MODEL_CACHE is None:
        BATSMAN_EMBEDDING_MODEL_CACHE = outil.load_keras_model(outil.MODEL_DIR \
                                                   + os.sep
                                                   + outil.BATSMAN_EMBEDDING_MODEL)
    batsman_embedding = BATSMAN_EMBEDDING_MODEL_CACHE

    if COUNTRY_ENC_MAP_CACHE is None:
        COUNTRY_ENC_MAP_CACHE = pickle.load(open(outil.MODEL_DIR \
                                   + os.sep \
                                   + outil.COUNTRY_ENCODING_MAP,'rb'))
    country_enc_map = COUNTRY_ENC_MAP_CACHE

    if BATSMAN_ENC_MAP_CACHE is None:
        BATSMAN_ENC_MAP_CACHE = pickle.load(open(outil.MODEL_DIR \
                                   + os.sep \
                                   + outil.BATSMAN_ENCODING_MAP, 'rb'))
    batsman_enc_map = BATSMAN_ENC_MAP_CACHE

    if LOC_ENC_MAP_FOR_BATSMAN_CACHE is None:
        LOC_ENC_MAP_FOR_BATSMAN_CACHE = pickle.load(open(outil.MODEL_DIR \
                                   + os.sep \
                                   + outil.LOC_ENCODING_MAP_FOR_BATSMAN, 'rb'))
    loc_enc_map_for_batsman = LOC_ENC_MAP_FOR_BATSMAN_CACHE

    try:
        loc_oh = loc_enc_map_for_batsman[location]
    except:
        location = get_similar_location(location).strip()
        loc_oh = loc_enc_map_for_batsman[location]

    opposition_oh = country_enc_map[opponent]

    batsman_oh_list = []
    position_oh_list = []
    loc_oh_list = []
    opposition_oh_list = []
    # print('getting batsman details')
    if len(batsman_list)< no_of_batsman:
        raise Exception("not enough batsman information")
    for bi,batsman in enumerate(batsman_list):

        if bi == no_of_batsman:
            break
        if team.strip() + ' ' + batsman.strip() not in batsman_enc_map:
            continue
        batsman_oh = batsman_enc_map[team.strip() + ' ' + batsman.strip()]
        position_oh = get_oh_pos(bi + 1)

        batsman_oh_list.append(batsman_oh)
        position_oh_list.append(position_oh)
        loc_oh_list.append(loc_oh)
        opposition_oh_list.append(opposition_oh)

    if len(batsman_oh_list)==0:
        raise Exception('No Batsman embedding available for '+team)

    batsman_mat = np.stack(batsman_oh_list)
    position_mat = np.stack(position_oh_list)
    loc_mat = np.stack(loc_oh_list)
    opposition_mat = np.stack(opposition_oh_list)
    # print('encoding')
    batsman_group_enc_mat = batsman_embedding.predict([batsman_mat, position_mat, loc_mat, opposition_mat])
    batsman_embedding_sum = batsman_group_enc_mat.sum(axis=0)


    if len(batsman_oh_list)<no_of_batsman:
        dif = len(batsman_oh_list) - no_of_batsman
        batsman_embedding_mean = batsman_group_enc_mat.mean(axis=0)
        batsman_embedding_sum = batsman_embedding_sum+dif*batsman_embedding_mean

    return batsman_embedding_sum

def get_batsman_only_embedding(batsman_list,team,no_of_batsman=7,ref_date=None):
    global BATSMAN_ONLY_EMBEDDING_MODEL_CACHE
    global BATSMAN_ENC_MAP_CACHE


    if BATSMAN_ONLY_EMBEDDING_MODEL_CACHE is None:
        BATSMAN_ONLY_EMBEDDING_MODEL_CACHE = outil.load_keras_model(outil.MODEL_DIR \
                                                   + os.sep
                                                   + outil.BATSMAN_ONLY_EMBEDDING_MODEL)
    batsman_embedding = BATSMAN_ONLY_EMBEDDING_MODEL_CACHE


    if BATSMAN_ENC_MAP_CACHE is None:
        BATSMAN_ENC_MAP_CACHE = pickle.load(open(outil.MODEL_DIR \
                                   + os.sep \
                                   + outil.BATSMAN_ENCODING_MAP, 'rb'))
    batsman_enc_map = BATSMAN_ENC_MAP_CACHE

    batsman_oh_list = []
    # print('getting batsman details')
    if len(batsman_list)< no_of_batsman:
        raise Exception("not enough batsman information")
    for bi,batsman in enumerate(batsman_list):

        if bi == no_of_batsman:
            break
        if team.strip() + ' ' + batsman.strip() not in batsman_enc_map:
            continue
        batsman_oh = batsman_enc_map[team.strip() + ' ' + batsman.strip()]

        batsman_oh_list.append(batsman_oh)

    if len(batsman_oh_list)==0:
        raise Exception('No Batsman embedding available for '+team)

    batsman_mat = np.stack(batsman_oh_list)
    # print('encoding')
    batsman_group_enc_mat = batsman_embedding.predict([batsman_mat])
    batsman_embedding_sum = batsman_group_enc_mat.sum(axis=0)


    if len(batsman_oh_list)<no_of_batsman:
        dif = len(batsman_oh_list) - no_of_batsman
        batsman_embedding_mean = batsman_group_enc_mat.mean(axis=0)
        batsman_embedding_sum = batsman_embedding_sum+dif*batsman_embedding_mean

    return batsman_embedding_sum


def get_first_innings_feature_embedding_vector(team, opponent, location, team_player_list, opponent_player_list,
                                               ref_date=None, no_of_years=None):
    feature_vector = get_first_innings_feature_vector(team, opponent, location,
                                                      team_player_list, opponent_player_list,
                                                      ref_date=ref_date, no_of_years=no_of_years)

    country_embedding_vector = get_team_opponent_location_embedding(team,opponent,location)

    #batsman_embedding_vector = get_batsman_embedding(team_player_list, team, opponent, location,ref_date=ref_date)
    #batsman_embedding_vector = get_batsman_only_embedding(team_player_list, team, ref_date=ref_date)

    #final_vector = np.concatenate([batsman_embedding_vector, country_embedding_vector, feature_vector])
    #final_vector = np.concatenate([batsman_embedding_vector, country_embedding_vector])
    final_vector = np.concatenate([country_embedding_vector, feature_vector])

    #
    #return country_embedding_vector
    return final_vector



def get_second_innings_feature_embedding_vector(target, team, opponent, location, team_player_list, opponent_player_list,
                                                ref_date=None, no_of_years=None):
    feature_vector = get_second_innings_feature_vector(target, team, opponent, location,
                                                       team_player_list, opponent_player_list,
                                                       ref_date=ref_date, no_of_years=no_of_years)

    country_embedding_vector = get_team_opponent_location_embedding(team,opponent,location)
    # country_embedding_vector = get_country_embedding(team)

    # batsman_embedding_vector = get_batsman_embedding(team_player_list, team, opponent, location,ref_date=ref_date)
    # batsman_embedding_vector = get_batsman_only_embedding(team_player_list, team, ref_date=ref_date)
    # batsman_embedding_vector = get_batsman_only_embedding(team_player_list, team, ref_date=ref_date)

    #final_vector = np.concatenate([batsman_embedding_vector, country_embedding_vector, feature_vector])
    #final_vector = np.concatenate([batsman_embedding_vector, country_embedding_vector])
    #final_vector = np.concatenate([country_embedding_vector,batsman_embedding_vector, np.array([target])])
    final_vector = np.concatenate([country_embedding_vector, feature_vector])

    return final_vector



### Feature engineering for Batsman run prediction


def get_single_batsman_embedding(batsman,position,team,opponent,location):

    global BATSMAN_EMBEDDING_MODEL_CACHE
    global COUNTRY_ENC_MAP_CACHE
    global BATSMAN_ENC_MAP_CACHE
    global LOC_ENC_MAP_FOR_BATSMAN_CACHE

    if BATSMAN_EMBEDDING_MODEL_CACHE is None:
        BATSMAN_EMBEDDING_MODEL_CACHE = outil.load_keras_model(outil.MODEL_DIR \
                                                   + os.sep
                                                   + outil.BATSMAN_EMBEDDING_MODEL)
    batsman_embedding = BATSMAN_EMBEDDING_MODEL_CACHE

    if COUNTRY_ENC_MAP_CACHE is None:
        COUNTRY_ENC_MAP_CACHE = pickle.load(open(outil.MODEL_DIR \
                                   + os.sep \
                                   + outil.COUNTRY_ENCODING_MAP,'rb'))
    country_enc_map = COUNTRY_ENC_MAP_CACHE

    if BATSMAN_ENC_MAP_CACHE is None:
        BATSMAN_ENC_MAP_CACHE = pickle.load(open(outil.MODEL_DIR \
                                   + os.sep \
                                   + outil.BATSMAN_ENCODING_MAP, 'rb'))
    batsman_enc_map = BATSMAN_ENC_MAP_CACHE

    if LOC_ENC_MAP_FOR_BATSMAN_CACHE is None:
        LOC_ENC_MAP_FOR_BATSMAN_CACHE = pickle.load(open(outil.MODEL_DIR \
                                   + os.sep \
                                   + outil.LOC_ENCODING_MAP_FOR_BATSMAN, 'rb'))
    loc_enc_map_for_batsman = LOC_ENC_MAP_FOR_BATSMAN_CACHE

    loc_oh = loc_enc_map_for_batsman[location].reshape(1,-1)
    opposition_oh = country_enc_map[opponent].reshape(1,-1)
    batsman_oh = batsman_enc_map[team.strip() + ' ' + batsman.strip()].reshape(1,-1)
    position_oh = get_oh_pos(position).reshape(1,-1)

    batsman_group_emb_vec = batsman_embedding.predict([batsman_oh, position_oh, loc_oh, opposition_oh])[0]

    return batsman_group_emb_vec


def get_batsman_score(country,batsman,ref_date=None):
    batsman_rank_file = rank.get_latest_rank_file('batsman',ref_date=ref_date)
    batsman_rank_df = pd.read_csv(batsman_rank_file)
    selected_batsman_df = batsman_rank_df[(batsman_rank_df['batsman']==batsman.strip())\
                                   & (batsman_rank_df['country']==country)]
    if selected_batsman_df.shape[0]==0:
        raise Exception('No batsman score is available for '+country)

    score = selected_batsman_df['batsman_score'].values[0]

    return score


def get_batsman_features_with_embedding(batsman,position,opponent_player_list,team,opponent,location,ref_date=None):
    batsman_group_emb_vec = get_single_batsman_embedding(batsman,position,team,opponent,location)
    batsman_score = get_batsman_score(team,batsman,ref_date=ref_date)
    opponent_score = get_country_score(opponent, ref_date=ref_date)
    bowler_mean, bowler_max, bowler_sum = get_bowler_mean_max(opponent, opponent_player_list, ref_date=ref_date)

    combined_vector = list(batsman_group_emb_vec) + [batsman_score,opponent_score,bowler_mean,bowler_sum]

    return combined_vector


def get_batsman_adversarial_vector(team,team_player_list):
    global BATSMAN_ENC_MAP_CACHE
    global ADVERSARIAL_BATSMAN_MODEL_CACHE

    if BATSMAN_ENC_MAP_CACHE is None:
        BATSMAN_ENC_MAP_CACHE = pickle.load(open(outil.MODEL_DIR \
                                   + os.sep \
                                   + outil.BATSMAN_ENCODING_MAP, 'rb'))
    batsman_enc_map = BATSMAN_ENC_MAP_CACHE

    if ADVERSARIAL_BATSMAN_MODEL_CACHE is None:
        ADVERSARIAL_BATSMAN_MODEL_CACHE = outil.load_keras_model(outil.MODEL_DIR \
                                                               + os.sep
                                                               + outil.ADVERSARIAL_BATSMAN_MODEL)
    batsman_embedding_model = ADVERSARIAL_BATSMAN_MODEL_CACHE
    batsman_oh_list = list()
    for batsman in team_player_list:
        if team.strip() + ' ' + batsman.strip() not in batsman_enc_map:
            continue
        batsman_oh = batsman_enc_map[team.strip() + ' ' + batsman.strip()]
        batsman_oh_list.append(batsman_oh)

    if len(batsman_oh_list)==0:
        raise Exception("Not enough batsman")
    batsman_oh_matrix = np.stack(batsman_oh_list)
    batsman_emb_vec = batsman_embedding_model.predict(batsman_oh_matrix)

    return np.sum(batsman_emb_vec,axis=0)

def get_bowler_adversarial_vector(opponent,opponent_player_list):
    global BOWLER_ENC_MAP_CACHE
    global ADVERSARIAL_BOWLER_MODEL_CACHE

    if BOWLER_ENC_MAP_CACHE is None:
        BOWLER_ENC_MAP_CACHE = pickle.load(open(outil.MODEL_DIR \
                                                 + os.sep \
                                                 + outil.BOWLER_ENCODING_MAP, 'rb'))
    bowler_enc_map = BOWLER_ENC_MAP_CACHE

    if ADVERSARIAL_BOWLER_MODEL_CACHE is None:
        ADVERSARIAL_BOWLER_MODEL_CACHE = outil.load_keras_model(outil.MODEL_DIR \
                                                                 + os.sep
                                                                 + outil.ADVERSARIAL_BOWLER_MODEL)
    bowler_embedding_model = ADVERSARIAL_BOWLER_MODEL_CACHE
    bowler_oh_list = list()
    for bowler in opponent_player_list:
        if opponent.strip() + ' ' + bowler.strip() not in bowler_enc_map:
            continue
        bowler_oh = bowler_enc_map[opponent.strip() + ' ' + bowler.strip()]
        bowler_oh_list.append(bowler_oh)

    if len(bowler_oh_list)==0:
        raise Exception("Not enough bowler")

    bowler_oh_matrix = np.stack(bowler_oh_list)
    bowler_emb_vec = bowler_embedding_model.predict(bowler_oh_matrix)

    return np.sum(bowler_emb_vec, axis=0)


def get_location_adversarial_vector(location):
    global LOC_ENC_MAP_FOR_BATSMAN_CACHE
    global ADVERSARIAL_LOCATION_MODEL_CACHE

    if LOC_ENC_MAP_FOR_BATSMAN_CACHE is None:
        LOC_ENC_MAP_FOR_BATSMAN_CACHE = pickle.load(open(outil.MODEL_DIR \
                                   + os.sep \
                                   + outil.LOC_ENCODING_MAP_FOR_BATSMAN, 'rb'))
    loc_enc_map_for_batsman = LOC_ENC_MAP_FOR_BATSMAN_CACHE

    if ADVERSARIAL_LOCATION_MODEL_CACHE is None:
        ADVERSARIAL_LOCATION_MODEL_CACHE = outil.load_keras_model(outil.MODEL_DIR \
                                                                 + os.sep
                                                                 + outil.ADVERSARIAL_LOCATION_MODEL)
    location_emb_model = ADVERSARIAL_LOCATION_MODEL_CACHE

    loc_oh = loc_enc_map_for_batsman[location].reshape(1, -1)
    location_emb = location_emb_model.predict(loc_oh)[0]
    return location_emb

def get_adversarial_first_innings_feature_vector(team, opponent, location, team_player_list, opponent_player_list,
                                               ref_date=None, no_of_years=None):

    feature_vector = get_first_innings_feature_vector(team, opponent, location,
                                                      team_player_list, opponent_player_list,
                                                      ref_date=ref_date, no_of_years=no_of_years)

    #country_embedding_vector = get_team_opponent_location_embedding(team,opponent,location)

    #batsman_embedding_vector = get_batsman_embedding(team_player_list, team, opponent, location)

    batsman_embedding_vector = get_batsman_adversarial_vector(team,team_player_list)
    bowler_embedding_vector = get_bowler_adversarial_vector(opponent,opponent_player_list)
    location_embedding_vector = get_location_adversarial_vector(location)

    final_vector = np.concatenate([feature_vector,batsman_embedding_vector, bowler_embedding_vector, location_embedding_vector])

    return final_vector


def get_best_batsman_position(batsman,team,opponent,location,played_position):
    global BATSMAN_EMBEDDING_RUN_MODEL_CACHE
    global COUNTRY_ENC_MAP_CACHE
    global BATSMAN_ENC_MAP_CACHE
    global LOC_ENC_MAP_FOR_BATSMAN_CACHE

    if BATSMAN_EMBEDDING_RUN_MODEL_CACHE is None:
        BATSMAN_EMBEDDING_RUN_MODEL_CACHE = outil.load_keras_model(outil.MODEL_DIR \
                                                               + os.sep
                                                               + outil.BATSMAN_EMBEDDING_RUN_MODEL)
    batsman_embedding = BATSMAN_EMBEDDING_RUN_MODEL_CACHE

    if COUNTRY_ENC_MAP_CACHE is None:
        COUNTRY_ENC_MAP_CACHE = pickle.load(open(outil.MODEL_DIR \
                                                 + os.sep \
                                                 + outil.COUNTRY_ENCODING_MAP, 'rb'))
    country_enc_map = COUNTRY_ENC_MAP_CACHE

    if BATSMAN_ENC_MAP_CACHE is None:
        BATSMAN_ENC_MAP_CACHE = pickle.load(open(outil.MODEL_DIR \
                                                 + os.sep \
                                                 + outil.BATSMAN_ENCODING_MAP, 'rb'))
    batsman_enc_map = BATSMAN_ENC_MAP_CACHE

    if LOC_ENC_MAP_FOR_BATSMAN_CACHE is None:
        LOC_ENC_MAP_FOR_BATSMAN_CACHE = pickle.load(open(outil.MODEL_DIR \
                                                         + os.sep \
                                                         + outil.LOC_ENCODING_MAP_FOR_BATSMAN, 'rb'))
    loc_enc_map_for_batsman = LOC_ENC_MAP_FOR_BATSMAN_CACHE

    loc_oh = loc_enc_map_for_batsman[location]
    opposition_oh = country_enc_map[opponent]

    batsman_oh_list = []
    position_oh_list = []
    loc_oh_list = []
    opposition_oh_list = []
    # print('getting batsman details')
    if team.strip() + ' ' + batsman.strip() not in batsman_enc_map:
        raise Exception('No Batsman embedding available for ' + team)

    batsman_oh = batsman_enc_map[team.strip() + ' ' + batsman.strip()]
    for bi in range(11):

        if team.strip() + ' ' + batsman.strip() not in batsman_enc_map:
            continue

        position_oh = get_oh_pos(bi + 1)

        batsman_oh_list.append(batsman_oh)
        position_oh_list.append(position_oh)
        loc_oh_list.append(loc_oh)
        opposition_oh_list.append(opposition_oh)

    batsman_mat = np.stack(batsman_oh_list)
    position_mat = np.stack(position_oh_list)
    loc_mat = np.stack(loc_oh_list)
    opposition_mat = np.stack(opposition_oh_list)
    # print('encoding')
    batsman_encoded_runs = batsman_embedding.predict([batsman_mat, position_mat, loc_mat, opposition_mat])
    suggested_position = batsman_encoded_runs.argmax(axis=0)
    preferred_position_sequence = (-batsman_encoded_runs).argsort(axis=0)

    preferred_position_sequence = np.squeeze(preferred_position_sequence.reshape(1,-1),0)
    run_weightage = np.squeeze(batsman_encoded_runs.reshape(1,-1),0)
    #print("\t\t\tpreferred_position_sequence sqeezed",np.squeeze(preferred_position_sequence.reshape(1,-1),0))
    #print("\t\t\tbatsman_encoded_runs squeezed", np.squeeze(batsman_encoded_runs.reshape(1,-1),0))
    #print('\t\t\t encoded runs',batsman_encoded_runs)
    #print('\t\t\t argmax', suggested_position)
    position_dif = abs(played_position-suggested_position)

    return suggested_position[0],position_dif[0],preferred_position_sequence,run_weightage


def get_batsman_score_by_embedding(batsman,team,opponent,location):

    global BATSMAN_EMBEDDING_RUN_MODEL_CACHE
    global COUNTRY_ENC_MAP_CACHE
    global BATSMAN_ENC_MAP_CACHE
    global LOC_ENC_MAP_FOR_BATSMAN_CACHE

    if BATSMAN_EMBEDDING_RUN_MODEL_CACHE is None:
        BATSMAN_EMBEDDING_RUN_MODEL_CACHE = outil.load_keras_model(outil.MODEL_DIR \
                                                               + os.sep
                                                               + outil.BATSMAN_EMBEDDING_RUN_MODEL)
    batsman_embedding = BATSMAN_EMBEDDING_RUN_MODEL_CACHE

    if COUNTRY_ENC_MAP_CACHE is None:
        COUNTRY_ENC_MAP_CACHE = pickle.load(open(outil.MODEL_DIR \
                                                 + os.sep \
                                                 + outil.COUNTRY_ENCODING_MAP, 'rb'))
    country_enc_map = COUNTRY_ENC_MAP_CACHE

    if BATSMAN_ENC_MAP_CACHE is None:
        BATSMAN_ENC_MAP_CACHE = pickle.load(open(outil.MODEL_DIR \
                                                 + os.sep \
                                                 + outil.BATSMAN_ENCODING_MAP, 'rb'))
    batsman_enc_map = BATSMAN_ENC_MAP_CACHE

    if LOC_ENC_MAP_FOR_BATSMAN_CACHE is None:
        LOC_ENC_MAP_FOR_BATSMAN_CACHE = pickle.load(open(outil.MODEL_DIR \
                                                         + os.sep \
                                                         + outil.LOC_ENCODING_MAP_FOR_BATSMAN, 'rb'))
    loc_enc_map_for_batsman = LOC_ENC_MAP_FOR_BATSMAN_CACHE

    loc_oh = loc_enc_map_for_batsman[location]
    opposition_oh = country_enc_map[opponent]

    batsman_oh_list = []
    position_oh_list = []
    loc_oh_list = []
    opposition_oh_list = []
    # print('getting batsman details')
    if team.strip() + ' ' + batsman.strip() not in batsman_enc_map:
        raise Exception('No Batsman embedding available for ' + team)

    batsman_oh = batsman_enc_map[team.strip() + ' ' + batsman.strip()]
    for bi in range(11):

        if team.strip() + ' ' + batsman.strip() not in batsman_enc_map:
            continue

        position_oh = get_oh_pos(bi + 1)

        batsman_oh_list.append(batsman_oh)
        position_oh_list.append(position_oh)
        loc_oh_list.append(loc_oh)
        opposition_oh_list.append(opposition_oh)

    batsman_mat = np.stack(batsman_oh_list)
    position_mat = np.stack(position_oh_list)
    loc_mat = np.stack(loc_oh_list)
    opposition_mat = np.stack(opposition_oh_list)
    # print('encoding')
    batsman_encoded_runs = batsman_embedding.predict([batsman_mat, position_mat, loc_mat, opposition_mat])
    #print("batsman_encoded_runs ", batsman_encoded_runs)
    squeezed_runs = np.squeeze(batsman_encoded_runs)
    #print("squeezed_runs ", squeezed_runs)
    batsman_score = np.sum(squeezed_runs)+np.max(squeezed_runs)

    return batsman_score


def get_batting_order_matching_metrics(batsman_list,team, opponent, location):
    position_match = 0
    overall_position_dif =0
    overall_position_dif_square = 0
    run_matrix_list = []
    for position,batsman in enumerate(batsman_list):
        try:

            suggested_position, playing_position_dif ,preferred_position_sequence,run_weightage = get_best_batsman_position(batsman, team, opponent, location, position)
            run_matrix_list.append(run_weightage)

            # print("\tfor ", batsman, " at ", position, "suggested at ",suggested_position," difference observed ",position_dif)
            # if suggested_position == position:
            #     print("\t\tmatched")
            #     position_match=position_match+1
            # overall_position_dif = overall_position_dif+position_dif
        except Exception as ex:
            print("ignored ", batsman, " of ", team, " with opponent ", opponent, ' at location',location)
            #raise ex
            #print("ignored ",batsman," of ",team," with opponent ",opponent,' at location')


    #print("========",np.array(run_matrix_list).shape)
    run_matrix = np.array(run_matrix_list)

    for position in range(run_matrix.shape[0]):
        best_batsman_arg = np.argmax(run_matrix[position])
        #print("batsman at ",best_batsman_arg," should have played at ",position)
        if best_batsman_arg == position:
            position_match = position_match+1
        position_dif_square = (best_batsman_arg-position)**2
        overall_position_dif_square = overall_position_dif_square + position_dif_square
        overall_position_dif = overall_position_dif+abs(best_batsman_arg-position)



    #match_percentage = (position_match/len(batsman_list))*100
    #mean_postion_dif = overall_position_dif/len(batsman_list)
    # print("no_of_batsman ",len(batsman_list))
    # print("match_percentage ",match_percentage)
    # print("mean_postion_dif ", mean_postion_dif)

    return position_match,overall_position_dif_square,overall_position_dif


def get_top_n_bowlers(bowler_list,country,n=6,ref_date=None):
    bowler_rank_file=rank.get_latest_rank_file("bowler",ref_date=ref_date)
    current_bowler_df = pd.DataFrame()
    current_bowler_df['bowler']=bowler_list
    current_bowler_df['country']=country

    bowler_rank_df = pd.read_csv(bowler_rank_file)

    current_bowler_df = current_bowler_df.merge(bowler_rank_df,how="inner",on=["bowler","country"])

    current_bowler_df.sort_values("bowler_score", ascending=False, inplace=True)

    return list(current_bowler_df.head(n)['bowler'])


def get_top_n_batsman(batsman_list,country,n=8,ref_date=None):
    batsman_rank_file=rank.get_latest_rank_file("batsman",ref_date=ref_date)
    curren_batsman_df = pd.DataFrame()
    curren_batsman_df['batsman']=batsman_list
    curren_batsman_df['country']=country

    batsman_rank_df = pd.read_csv(batsman_rank_file)

    curren_batsman_df = curren_batsman_df.merge(batsman_rank_df,how="inner",on=["batsman","country"])

    curren_batsman_df.sort_values("batsman_score", ascending=False, inplace=True)

    return list(curren_batsman_df.head(n)['batsman'])

def get_similar_location(current_location):

    global LOC_ENC_MAP_CACHE

    if LOC_ENC_MAP_CACHE is None:
        LOC_ENC_MAP_CACHE = pickle.load(open(outil.MODEL_DIR \
                                             + os.sep \
                                             + outil.LOC_ENCODING_MAP,'rb'))
    loc_enc_map = LOC_ENC_MAP_CACHE


    stop_words = ['Cricket', 'Ground', 'Stadium', 'International', 'bad', 'St']

    locations = list(loc_enc_map.keys())
    adjusted_locations = []

    for loc in locations:
        new_loc = str(loc)
        for word in stop_words:
            new_loc = new_loc.replace(word, '').strip()
        adjusted_locations.append(new_loc)

    vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(1, 3), max_features=500)

    name_vectors = vectorizer.fit_transform(adjusted_locations)

    current_location_copy = str(current_location)
    for word in stop_words:
        current_location_copy = current_location_copy.replace(word, '').strip()

    current_loc_vector = vectorizer.transform([current_location_copy])[0]

    similarity_list = []
    for index,existing_location in enumerate(locations):
        similarity = (np.dot(name_vectors[index, :], current_loc_vector.T))[0, 0] / (
                    np.sqrt(np.square(name_vectors[index, :].toarray()).sum()) * np.sqrt(
                np.square(current_loc_vector.toarray()).sum()))

        similarity_list.append(similarity)

    highest_similarity_argument = np.argmax(np.array(similarity_list))
    highest_similarity = np.array(similarity_list).max()

    if highest_similarity > 0.80:
        print("Found ",locations[highest_similarity_argument]," for ",current_location)
        return locations[highest_similarity_argument]
    else:
        raise Exception(" No similar location could be found")


