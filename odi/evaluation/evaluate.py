from datetime import datetime
from odi.data_loader import data_loader as dl
import pandas as pd
import os
from tqdm import tqdm
from odi.feature_engg import util as cricutil,feature_extractor
from odi.model_util import odi_util as outil
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error,mean_squared_error,accuracy_score
import click
from odi.retrain import create_train_test as ctt
import tensorflow as tf
from odi.feature_engg import util as cricutil
from odi.preprocessing import rank as rank
import math
from odi.inference import prediction as pred



tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def mape(y_true,y_predict):
    return np.sum((np.abs(y_true-y_predict)/y_true)*100)/len(y_true)


def evaluate_expected_threshold(from_date, to_date, environment='production'):
    outil.use_model_from(environment)

    start_date = datetime.strptime(from_date, '%Y-%m-%d')
    end_date = datetime.strptime(to_date, '%Y-%m-%d')

    match_list_df = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION + os.sep + 'match_list.csv')
    match_list_df = match_list_df[(match_list_df['date'] >= start_date) & (match_list_df['date'] <= end_date)]
    match_stats_df = pd.read_csv(dl.CSV_LOAD_LOCATION + os.sep + 'match_stats.csv')

    match_list_df = match_list_df.merge(match_stats_df, on='match_id', how='inner')

    match_id_list = list(match_list_df['match_id'].unique())

    # batsman_rank_file = rank.get_latest_rank_file("batsman", ref_date=start_date)
    # print("=========reading file ",batsman_rank_file)


    dict_list = []
    for match_id in tqdm(match_id_list):

        selected_innings = "first_innings"
        opponent_innings = "second_innings"
        innings_team = match_list_df[(match_list_df['match_id']==match_id) &
                                 (match_list_df[selected_innings]==match_list_df["team_statistics"])]

        team = innings_team["team_statistics"].values[0]
        opponent = match_list_df[(match_list_df['match_id']==match_id)][opponent_innings].values[0]

        opponent_team = match_list_df[(match_list_df['match_id'] == match_id) &
                                      (match_list_df[opponent_innings] == match_list_df["team_statistics"])]

        location = innings_team["location"].values[0]
        win_flag = int(innings_team[selected_innings].values[0] == innings_team["winner"].values[0])
        ref_date = datetime.strptime(innings_team['date'].astype(str).values[0],'%Y-%m-%d')

        first_innings_runs = innings_team['total_run'].values[0]

        team_batsman_list = list()
        for i in range(11):
            player = innings_team['batsman_' + str(i + 1)].values[0]
            if player == 'not_batted':
                break
            else:
                team_batsman_list.append(player)

        team_bowler_list = list()
        for i in range(11):
            player = innings_team['bowler_' + str(i + 1)].values[0]
            if player == 'not_bowled':
                break
            else:
                team_bowler_list.append(player)

        opponent_batsman_list = list()
        for i in range(11):
            player = opponent_team['batsman_' + str(i + 1)].values[0]
            if player == 'not_batted':
                break
            else:
                opponent_batsman_list.append(player)

        opponent_bowler_list = list()
        for i in range(11):
            player = opponent_team['bowler_' + str(i + 1)].values[0]
            if player == 'not_bowled':
                break
            else:
                opponent_bowler_list.append(player)

        try :
            threshold = None
            target_by_a = pred.predict_first_innings_run(team,opponent,location,team_batsman_list,opponent_bowler_list,ref_date=ref_date,no_of_years=None)

            feature_vector = feature_extractor.get_second_innings_feature_embedding_vector(target_by_a, opponent, team, location,
                                                                            opponent_batsman_list, team_bowler_list,
                                                                            ref_date=ref_date)

            second_innings_model = pickle.load(open(outil.MODEL_DIR + os.sep + outil.SECOND_INNINGS_MODEL, 'rb'))
            feature_vector = feature_vector.reshape(1, -1)
            success_by_b = second_innings_model.predict(feature_vector.reshape(1, -1))[0]
            always_win = False
            always_loose = False
            if success_by_b:
                while success_by_b and target_by_a<320:
                    target_by_a = target_by_a + 10
                    feature_vector[:,-1]=target_by_a
                    success_by_b = second_innings_model.predict(feature_vector.reshape(1, -1))[0]
                    if not success_by_b:
                        threshold = target_by_a -5
                if threshold is None:
                    always_loose = True
            else:
                while not success_by_b and target_by_a>180:
                    target_by_a = target_by_a - 10
                    feature_vector[:,-1]=target_by_a
                    success_by_b = second_innings_model.predict(feature_vector.reshape(1, -1))[0]
                    if success_by_b:
                        threshold = target_by_a +5
                if threshold is None:
                    always_win = True

            if threshold is None:
                continue
            else:
                row = {"threshold":threshold,
                       "runs": first_innings_runs,
                       "crossed_threshold": int(first_innings_runs > threshold),
                       "win": win_flag}
                dict_list.append(row)

            # print("=====predicted ",target_by_a," actual ",first_innings_runs,
            #       "threshold =",threshold,"=== crossed ",int(first_innings_runs > threshold),"did win ", win_flag,always_win,always_loose)

        except Exception as ex:
            print(ex," : ignored ", team ," vs ",opponent, " on ",ref_date," at ",location )

    results = pd.DataFrame(dict_list)

    threshold_prediction_matched = results[results['crossed_threshold']==results['win']].shape[0]/results.shape[0]
    threshold_prediction_mismatched = results[results['crossed_threshold']!=results['win']].shape[0]/results.shape[0]

    print("prediction matched ",threshold_prediction_matched)
    print("prediction mismatched ", threshold_prediction_mismatched)
    outil.create_model_meta_info_entry('threshold_recommendation_validation',
                                       (),
                                       (threshold_prediction_matched,threshold_prediction_mismatched),
                                       info="metrics is prediction match and prediction mismatch bassed on threshold",
                                       )



def evaluate_batsman_recommendation(from_date, to_date, environment='production'):
    outil.use_model_from(environment)

    start_date = datetime.strptime(from_date, '%Y-%m-%d')
    end_date = datetime.strptime(to_date, '%Y-%m-%d')

    match_list_df = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION + os.sep + 'match_list.csv')
    match_list_df = match_list_df[(match_list_df['date'] >= start_date) & (match_list_df['date'] <= end_date)]
    match_stats_df = pd.read_csv(dl.CSV_LOAD_LOCATION + os.sep + 'match_stats.csv')

    match_list_df = match_list_df.merge(match_stats_df, on='match_id', how='inner')

    match_id_list = list(match_list_df['match_id'].unique())

    batsman_rank_file = rank.get_latest_rank_file("batsman", ref_date=start_date)
    # print("=========reading file ",batsman_rank_file)
    batsman_rank_df = pd.read_csv(batsman_rank_file)

    no_of_suggestion = 6
    dict_list = []
    for match_id in tqdm(match_id_list):
        for combination in [('first_innings','second_innings'),('second_innings','first_innings')]:
            selected_innings = combination[0]
            opponent_innings = combination[1]
            innings_team = match_list_df[(match_list_df['match_id']==match_id) &
                                     (match_list_df[selected_innings]==match_list_df["team_statistics"])]

            team = innings_team["team_statistics"].values[0]
            opponent = match_list_df[(match_list_df['match_id']==match_id)][opponent_innings].values[0]

            location = innings_team["location"].values[0]
            win_flag = int(innings_team[selected_innings].values[0] == innings_team["winner"].values[0])
            ref_date = datetime.strptime(innings_team['date'].astype(str).values[0],'%Y-%m-%d')

            batsman_list = list()
            for i in range(11):
                player = innings_team['batsman_'+str(i+1)].values[0]
                if player == 'not_batted':
                    break
                else:
                    batsman_list.append(player)

            # bowler_list = list()
            # for i in range(11):
            #     player = innings_team['bowler_' + str(i + 1)].values[0]
            #     if player == 'not_bowled':
            #         break
            #     else:
            #         bowler_list.append(player)

            ## get_available_batsman_list
            #print("==========",ref_date," type",type(ref_date))


            available_batsman_list = list(batsman_rank_df[batsman_rank_df['country']==team]['batsman'])

            # bowler_rank_file = rank.get_latest_rank_file("bowler", ref_date)
            # bowler_rank_df = pd.read_csv(bowler_rank_file)
            #
            # available_bowlerlist = list(bowler_rank_df[bowler_rank_df['country'] == team]['bowler'])

            score_list = []
            for batsman in available_batsman_list:
                try:
                    score = feature_extractor.get_batsman_score_by_embedding(batsman,team,opponent,location)
                    score_list.append(score)
                except Exception as ex:
                    score_list.append(0)
                    print(" Ignoring ",batsman," of ",team," on ",ref_date)

            sorted_args = np.argsort(-np.array(score_list))
            match_score = 0
            suggested_batsman_list =  []
            for count in range(no_of_suggestion):
                arg = sorted_args[count]
                suggested_batsman_list.append(available_batsman_list[arg])
                if available_batsman_list[arg] in batsman_list:
                    match_score = match_score + 1 + math.log(no_of_suggestion-count)
                    #match_score = match_score + 1

            # print("selected Innings " ,selected_innings)
            # print("team ",team)
            # print("opponent ", opponent)
            # print("date ", ref_date)
            # print(" file ",batsman_rank_file)
            # print("playing batsman list ",batsman_list)
            # print("suggested batsman list ",suggested_batsman_list)
            # print("match_score",match_score)
            # print("win",win_flag)
            # print("======================================")



            dict_list.append({"match_score":match_score,"win_flag":win_flag})

        #break

    score_df = pd.DataFrame(dict_list)

    winning_mean = score_df[score_df["win_flag"]==1]['match_score'].mean()
    loosing_mean = score_df[score_df["win_flag"]==0]['match_score'].mean()

    print("winning_mean ",winning_mean)
    print("loosing_mean ", loosing_mean)

    sigma_1 = score_df[score_df["win_flag"]==1]['match_score'].std()
    sigma_2 = score_df[score_df["win_flag"]==0]['match_score'].std()

    n1 = score_df[score_df["win_flag"] == 1].shape[0]
    n2 = score_df[score_df["win_flag"] == 0].shape[0]

    z_score = (winning_mean - loosing_mean) / math.sqrt((sigma_1 ** 2 / n1) + (sigma_2 ** 2 / n2))

    print("z score ",z_score)

    outil.create_model_meta_info_entry('batsman_recommendation_validation',
                                       (),
                                       (winning_mean, loosing_mean, z_score),
                                       info="metrics is mean match in winnning teams, mean match in loosing teams, z statistics of 2 means",
                                       )


def evaluate_batting_position(from_date, to_date, environment='production'):
    outil.use_model_from(environment)

    start_date = datetime.strptime(from_date, '%Y-%m-%d')
    end_date = datetime.strptime(to_date, '%Y-%m-%d')

    # custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d")
    # match_list_df = pd.read_csv(dl.CSV_LOAD_LOCATION + os.sep + 'match_list.csv', parse_dates=['date'],
    #                             date_parser=custom_date_parser)

    match_list_df = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION + os.sep + 'match_list.csv')
    match_list_df = match_list_df[(match_list_df['date'] >= start_date) & (match_list_df['date'] <= end_date)]
    match_stats_df = pd.read_csv(dl.CSV_LOAD_LOCATION + os.sep + 'match_stats.csv')

    match_list_df = match_list_df.merge(match_stats_df, on='match_id', how='inner')
    #match_list_df = match_list_df[match_list_df['first_innings']==match_list_df['team_statistics']]


    match_id_list = list(match_list_df['match_id'].unique())

    dict_list = []
    for match_id in tqdm(match_id_list):
        for combination in [('first_innings','second_innings'),('second_innings','first_innings')]:
            selected_innings = combination[0]
            opponent_innings = combination[1]
            innings_team = match_list_df[(match_list_df['match_id']==match_id) &
                                     (match_list_df[selected_innings]==match_list_df["team_statistics"])]

            team = match_list_df["team_statistics"].values[0]

            opponent = match_list_df[(match_list_df['match_id'] == match_id)][opponent_innings].values[0]

            location = innings_team["location"].values[0]
            win_flag = int(innings_team[selected_innings].values[0] == innings_team["winner"].values[0])

            batsman_list = list()
            for i in range(11):
                player = innings_team['batsman_'+str(i+1)].values[0]
                if player == 'not_batted':
                    break
                else:
                    batsman_list.append(player)

            position_match, overall_position_dif_square, overall_position_dif = feature_extractor.get_batting_order_matching_metrics(batsman_list,team, opponent, location)
            row_dit = {
                "innings":selected_innings,
                "win_flag":win_flag,
                "position_match":position_match,
                "overall_position_dif":overall_position_dif,
                "overall_position_dif_square":overall_position_dif_square,
                "no_of_batsman":len(batsman_list)

            }

            dict_list.append(row_dit)

        # break

    result_df = pd.DataFrame(dict_list)

    winning_team_match = result_df[result_df["win_flag"]==1]["position_match"].mean()
    loosing_team_match = result_df[result_df["win_flag"] == 0]["position_match"].mean()

    #winning_team_position_dif = result_df[result_df["win_flag"] == 1]["overall_position_dif"].sum()/result_df[result_df["win_flag"] == 1]["no_of_batsman"].sum()
    #loosing_team_position_dif = result_df[result_df["win_flag"] == 0]["overall_position_dif"].sum()/result_df[result_df["win_flag"] == 0]["no_of_batsman"].sum()

    winning_team_position_dif = result_df[result_df["win_flag"] == 1]["overall_position_dif"].mean()
    loosing_team_position_dif = result_df[result_df["win_flag"] == 0]["overall_position_dif"].mean()

    sigma_1 = result_df[result_df["win_flag"] == 1]["position_match"].std()
    sigma_2 = result_df[result_df["win_flag"] == 0]["position_match"].std()

    n1 = result_df[result_df["win_flag"] == 1].shape[0]
    n2 = result_df[result_df["win_flag"] == 0].shape[0]

    # winning_team_position_rmse = math.sqrt(result_df[result_df["win_flag"] == 1]["overall_position_dif_square"].sum() / \
    #                            result_df[result_df["win_flag"] == 1]["no_of_batsman"].sum())
    # loosing_team_position_rmse = math.sqrt(result_df[result_df["win_flag"] == 0]["overall_position_dif_square"].sum() / \
    #                             result_df[result_df["win_flag"] == 0]["no_of_batsman"].sum())

    winning_team_position_rmse = math.sqrt(result_df[result_df["win_flag"] == 1]["overall_position_dif_square"].mean())
    loosing_team_position_rmse = math.sqrt(result_df[result_df["win_flag"] == 0]["overall_position_dif_square"].mean())

    z_score = (winning_team_match-loosing_team_match)/math.sqrt((sigma_1**2/n1)+(sigma_2**2/n2))

    # first_inn_winning_team_match_percentage = result_df[(result_df["win_flag"] == 1) & (result_df["innings"] == "first_innings")]["match_percentage"].mean()
    # first_inn_loosing_team_match_percentage = result_df[(result_df["win_flag"] == 0) & (result_df["innings"] == "first_innings")]["match_percentage"].mean()
    #
    # #first_inn_winning_team_position_dif = result_df[(result_df["win_flag"] == 1) & (result_df["innings"] == "first_innings")]["overall_position_dif"].sum()/result_df[(result_df["win_flag"] == 1) & (result_df["innings"] == "first_innings")]["no_of_batsman"].sum()
    # #first_inn_loosing_team_position_dif = result_df[(result_df["win_flag"] == 0) & (result_df["innings"] == "first_innings")]["overall_position_dif"].sum()/result_df[(result_df["win_flag"] == 0) & (result_df["innings"] == "first_innings")]["no_of_batsman"].sum()
    #
    # first_inn_winning_team_position_dif = \
    # result_df[(result_df["win_flag"] == 1) & (result_df["innings"] == "first_innings")]["overall_position_dif"].mean()
    # first_inn_loosing_team_position_dif = \
    # result_df[(result_df["win_flag"] == 0) & (result_df["innings"] == "first_innings")]["overall_position_dif"].mean()
    #
    # # first_inn_winning_team_position_rmse = \
    # # math.sqrt(result_df[(result_df["win_flag"] == 1) & (result_df["innings"] == "first_innings")]["overall_position_dif_square"].sum() / \
    # # result_df[(result_df["win_flag"] == 1) & (result_df["innings"] == "first_innings")]["no_of_batsman"].sum())
    # # first_inn_loosing_team_position_rmse = \
    # # math.sqrt(result_df[(result_df["win_flag"] == 0) & (result_df["innings"] == "first_innings")]["overall_position_dif_square"].sum() / \
    # # result_df[(result_df["win_flag"] == 0) & (result_df["innings"] == "first_innings")]["overall_position_dif_square"].sum())
    #
    # first_inn_winning_team_position_rmse = \
    #     math.sqrt(result_df[(result_df["win_flag"] == 1) & (result_df["innings"] == "first_innings")][
    #                   "overall_position_dif_square"].mean())
    # first_inn_loosing_team_position_rmse = \
    #     math.sqrt(result_df[(result_df["win_flag"] == 0) & (result_df["innings"] == "first_innings")][
    #                   "overall_position_dif_square"].mean())

    print("winning_team_match-",winning_team_match)
    print("loosing_team_match-", loosing_team_match)

    print("winning_team_positon_dif-", winning_team_position_dif)
    print("loosing_team_position_dif-", loosing_team_position_dif)

    print("winning_team_positon_rmse-", winning_team_position_rmse)
    print("loosing_team_positon_rmse-", loosing_team_position_rmse)

    print("z-score",z_score)

    outil.create_model_meta_info_entry('batting_order_validation',
                                       (),
                                       (winning_team_match,loosing_team_match,z_score),
                                       info="metrics is mean match in winnning teams, mean match in loosing teams, z statistics of 2 means",
                                       )




def evaluate_first_innings(from_date, to_date, environment='production',use_emb=True,model='team'):
    outil.use_model_from(environment)

    predictor = None
    if not use_emb:
        try:

            predictor = pickle.load(
                open(os.path.join(outil.MODEL_DIR, outil.FIRST_INNINGS_MODEL_BASE), 'rb'))
        except:
            raise Exception("use_emb: " + str(use_emb) + " option related files not available")
    else:
        try:
            if model == 'team':
                predictor = pickle.load(open(outil.MODEL_DIR + os.sep + outil.FIRST_INNINGS_MODEL, 'rb'))
            else:
                predictor = pickle.load(open(outil.MODEL_DIR + os.sep + outil.ADVERSARIAL_FIRST_INNINGS, 'rb'))
        except:
            print(' Provided First innings model in option not available')


    start_date = datetime.strptime(from_date, '%Y-%m-%d')
    end_date = datetime.strptime(to_date, '%Y-%m-%d')

    custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d")
    match_list_df = pd.read_csv(dl.CSV_LOAD_LOCATION+os.sep+'match_list.csv',parse_dates=['date'],date_parser=custom_date_parser)
    match_list_df = match_list_df[(match_list_df['date']>=start_date) & (match_list_df['date']<=end_date)]
    match_stats_df = pd.read_csv(dl.CSV_LOAD_LOCATION+os.sep+'match_stats.csv')

    match_list_df=match_list_df.merge(match_stats_df,on='match_id',how='inner')

    match_id_list = list(match_list_df['match_id'].unique())
    feature_vector_list =[]
    score_list = []

    for match_id in tqdm(match_id_list):

        first_innings_team = match_list_df[(match_list_df['match_id']==match_id) & \
                                      (match_list_df['first_innings']==match_list_df['team_statistics'])]
        second_innings_team = match_list_df[(match_list_df['match_id'] == match_id) & \
                                      (match_list_df['second_innings'] == match_list_df['team_statistics'])]
        team = first_innings_team['team_statistics'].values[0]
        opponent = second_innings_team['team_statistics'].values[0]
        location = first_innings_team['location'].values[0]
        ref_date = cricutil.npdate_to_datetime(first_innings_team['date'].values[0])
        try:
            team_player_list = []
            for i in range(11):
                player = first_innings_team['batsman_'+str(i+1)].values[0]
                if player == 'not_batted':
                    break
                else:
                    team_player_list.append(player)
            opponent_player_list = []
            for i in range(11):
                bowler = second_innings_team['bowler_'+str(i+1)].values[0]
                if bowler == 'not_bowled':
                    break
                else:
                    opponent_player_list.append(bowler)
            #print('============',opponent_player_list)
            if use_emb:
                if model == 'team':
                    feature_vector = feature_extractor.get_first_innings_feature_embedding_vector(team, opponent, location,\
                                                                                                 team_player_list,\
                                                                                                 opponent_player_list,\
                                                                                                 ref_date=ref_date)
                else:
                    feature_vector = feature_extractor.get_adversarial_first_innings_feature_vector(team, opponent,
                                                                                                    location, \
                                                                                                    team_player_list, \
                                                                                                    opponent_player_list, \
                                                                                                    ref_date=ref_date)
            else:

                feature_vector = feature_extractor.get_first_innings_feature_vector(team, opponent, location,\
                                                                                    team_player_list, opponent_player_list,\
                                                                                    ref_date=ref_date)


            feature_vector_list.append(feature_vector)
            score_list.append(first_innings_team['total_run'].values[0])
        except Exception as ex:
            print(match_id,': Exception for match between  ',team,' and ',opponent,' on ',ref_date)
            print(ex)


    feature_matrix = np.stack(feature_vector_list)
    actual_runs = np.stack(score_list)

    predicted_runs = predictor.predict(feature_matrix)

    mape_val=mape(actual_runs, predicted_runs)
    mae=mean_absolute_error(actual_runs, predicted_runs)
    mse=mean_squared_error(actual_runs, predicted_runs)

    print('mape :',mape_val)
    print('mae :',mae)
    print('mse :',mse)
    print('total data ',len(actual_runs))

    return mape_val,mae,mse,len(actual_runs)


def evaluate_second_innings(from_date, to_date, environment='production',use_emb=True):
    outil.use_model_from(environment)

    if not use_emb:
        try:
            predictor = pickle.load(
                open(os.path.join(outil.MODEL_DIR, outil.SECOND_INNINGS_MODEL_BASE), 'rb'))
        except:
            raise Exception("use_emb: " + str(use_emb) + " option related files not available")
    else:
        predictor = pickle.load(open(outil.MODEL_DIR + os.sep + outil.SECOND_INNINGS_MODEL, 'rb'))

    start_date = datetime.strptime(from_date, '%Y-%m-%d')
    end_date = datetime.strptime(to_date, '%Y-%m-%d')

    custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d")
    match_list_df = pd.read_csv(dl.CSV_LOAD_LOCATION + os.sep + 'match_list.csv', parse_dates=['date'],
                                date_parser=custom_date_parser)
    match_list_df = match_list_df[(match_list_df['date'] >= start_date) & (match_list_df['date'] <= end_date)]
    match_stats_df = pd.read_csv(dl.CSV_LOAD_LOCATION + os.sep + 'match_stats.csv')

    match_list_df = match_list_df.merge(match_stats_df, on='match_id', how='inner')

    match_id_list = list(match_list_df['match_id'].unique())
    feature_vector_list = []
    result_list = []

    for match_id in tqdm(match_id_list):
        first_innings_team = match_list_df[(match_list_df['match_id'] == match_id) & \
                                           (match_list_df['first_innings'] == match_list_df['team_statistics'])]
        second_innings_team = match_list_df[(match_list_df['match_id'] == match_id) & \
                                            (match_list_df['second_innings'] == match_list_df['team_statistics'])]
        team = second_innings_team['team_statistics'].values[0]
        opponent = first_innings_team['team_statistics'].values[0]
        location = first_innings_team['location'].values[0]
        ref_date = cricutil.npdate_to_datetime(first_innings_team['date'].values[0])

        try:
            team_player_list = []
            for i in range(11):
                player = second_innings_team['batsman_'+str(i+1)].values[0]
                if player == 'not_batted':
                    break
                else:
                    team_player_list.append(player)
            opponent_player_list = []
            for i in range(11):
                bowler = first_innings_team['bowler_'+str(i+1)].values[0]
                if bowler == 'not_bowled':
                    break
                else:
                    opponent_player_list.append(bowler)
            target = first_innings_team['total_run'].values[0]

            if use_emb:
                feature_vector = feature_extractor.get_second_innings_feature_embedding_vector(target,team, opponent, location,\
                                                                                               team_player_list,\
                                                                                             opponent_player_list,\
                                                                                             ref_date=ref_date)
            else:
                feature_vector = feature_extractor.get_second_innings_feature_vector(target, team, opponent,\
                                                                                    location, team_player_list,\
                                                                                    opponent_player_list, ref_date=ref_date)
            feature_vector_list.append(feature_vector)
            if second_innings_team['second_innings'].values[0]==second_innings_team['winner'].values[0]:
                result_list.append(1)
            else:
                result_list.append(0)

        except Exception as ex:
            print(match_id,': Exception for match between  ',team,' and ',opponent,' on ',ref_date)
            print(ex)

    feature_matrix = np.stack(feature_vector_list)
    actual_results = np.stack(result_list)
    predicted_results = predictor.predict(feature_matrix)

    accuracy = accuracy_score(actual_results, predicted_results)

    print('accuracy ',accuracy)
    print('data size ', len(result_list))
    return accuracy,len(result_list)


def evaluate_combined_innings(from_date, to_date, environment='production',
                              first_innings_emb = True,second_innings_emb=True,
                              first_emb_model='team', second_emb_model='team'):
    outil.use_model_from(environment)

    predictor_second_innings = None
    if not second_innings_emb:
        try:
            predictor_second_innings = pickle.load(
                open(os.path.join(outil.MODEL_DIR, outil.SECOND_INNINGS_MODEL_BASE), 'rb'))
        except:
            raise Exception("second_innings_emb: " + str(second_innings_emb) + " option related files not available")
    else:
        if second_emb_model == 'team':
            predictor_second_innings = pickle.load(open(outil.MODEL_DIR + os.sep + outil.SECOND_INNINGS_MODEL, 'rb'))
        else:
            raise Exception("Provided option for second innings not available")

    predictor_first_innings = None
    if not first_innings_emb:
        try:

            predictor_first_innings = pickle.load(
                open(os.path.join(outil.MODEL_DIR, outil.FIRST_INNINGS_MODEL_BASE), 'rb'))
        except:
            raise Exception("first_innings_emb: " + str(first_innings_emb) + " option related files not available")
    else:
        try:
            if first_emb_model == 'team':
                predictor_first_innings = pickle.load(open(outil.MODEL_DIR + os.sep + outil.FIRST_INNINGS_MODEL, 'rb'))
            else:
                predictor_first_innings = pickle.load(open(outil.MODEL_DIR + os.sep + outil.ADVERSARIAL_FIRST_INNINGS, 'rb'))
        except:
            print(' Provided First innings model in option not available')

    start_date = datetime.strptime(from_date, '%Y-%m-%d')
    end_date = datetime.strptime(to_date, '%Y-%m-%d')

    custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d")
    match_list_df = pd.read_csv(dl.CSV_LOAD_LOCATION+os.sep+'match_list.csv',parse_dates=['date'],date_parser=custom_date_parser)
    match_list_df = match_list_df[(match_list_df['date']>=start_date) & (match_list_df['date']<=end_date)]
    match_stats_df = pd.read_csv(dl.CSV_LOAD_LOCATION+os.sep+'match_stats.csv')

    match_list_df=match_list_df.merge(match_stats_df,on='match_id',how='inner')

    match_id_list = list(match_list_df['match_id'].unique())
    feature_vector_list =[]
    score_list = []

    first_innings_match_id_list = list()
    for match_id in tqdm(match_id_list):

        first_innings_team = match_list_df[(match_list_df['match_id']==match_id) & \
                                      (match_list_df['first_innings']==match_list_df['team_statistics'])]
        second_innings_team = match_list_df[(match_list_df['match_id'] == match_id) & \
                                      (match_list_df['second_innings'] == match_list_df['team_statistics'])]
        team = first_innings_team['team_statistics'].values[0]
        opponent = second_innings_team['team_statistics'].values[0]
        location = first_innings_team['location'].values[0]
        ref_date = cricutil.npdate_to_datetime(first_innings_team['date'].values[0])
        try:
            team_player_list = []
            for i in range(11):
                player = first_innings_team['batsman_'+str(i+1)].values[0]
                if player == 'not_batted':
                    break
                else:
                    team_player_list.append(player)
            opponent_player_list = []
            for i in range(11):
                bowler = second_innings_team['bowler_'+str(i+1)].values[0]
                if bowler == 'not_bowled':
                    break
                else:
                    opponent_player_list.append(bowler)

            if first_innings_emb:
                if first_emb_model == 'team':
                    feature_vector = feature_extractor.get_first_innings_feature_embedding_vector(team, opponent, location,\
                                                                                                 team_player_list,\
                                                                                                 opponent_player_list,\
                                                                                                ref_date=ref_date)
                else:
                    feature_vector = feature_extractor.get_adversarial_first_innings_feature_vector(team, opponent,
                                                                                                    location, \
                                                                                                    team_player_list, \
                                                                                                    opponent_player_list, \
                                                                                                    ref_date=ref_date)
            else:


                    feature_vector = feature_extractor.get_first_innings_feature_vector(team, opponent, location,\
                                                                                        team_player_list, opponent_player_list,\
                                                                                        ref_date=ref_date)

            feature_vector_list.append(feature_vector)
            score_list.append(first_innings_team['total_run'].values[0])
            first_innings_match_id_list.append(match_id)
        except Exception as ex:
            print(match_id,': Exception for match between  ',team,' and ',opponent,' on ',ref_date,' at ',location)
            print(ex)


    feature_matrix = np.stack(feature_vector_list)
    actual_runs = np.stack(score_list)
    predicted_runs = predictor_first_innings.predict(feature_matrix)

    predicted_run_list = list(predicted_runs)

    mape_val = mape(actual_runs, predicted_runs)
    mae = mean_absolute_error(actual_runs, predicted_runs)
    mse = mean_squared_error(actual_runs, predicted_runs)
    print("=====FIRST INNNINGS METRICS=====")
    print('mape :', mape_val)
    print('mae :', mae)
    print('mse :', mse)
    print('total data ', len(actual_runs))

    # Predict second Innings
    result_list = list()
    second_innings_feature_vector_list = list()
    for match_id,predicted_first_innings_run in tqdm(zip(first_innings_match_id_list,predicted_run_list)):

        first_innings_team = match_list_df[(match_list_df['match_id'] == match_id) & \
                                           (match_list_df['first_innings'] == match_list_df['team_statistics'])]
        second_innings_team = match_list_df[(match_list_df['match_id'] == match_id) & \
                                            (match_list_df['second_innings'] == match_list_df['team_statistics'])]
        team = second_innings_team['team_statistics'].values[0]
        opponent = first_innings_team['team_statistics'].values[0]
        location = first_innings_team['location'].values[0]
        ref_date = cricutil.npdate_to_datetime(second_innings_team['date'].values[0])

        try:
            team_player_list = []
            for i in range(11):
                player = second_innings_team['batsman_'+str(i+1)].values[0]
                if player == 'not_batted':
                    break
                else:
                    team_player_list.append(player)
            opponent_player_list = []
            for i in range(11):
                bowler = first_innings_team['bowler_'+str(i+1)].values[0]
                if bowler == 'not_bowled':
                    break
                else:
                    opponent_player_list.append(bowler)
            # target = first_innings_team['total_run'].values[0]

            if second_innings_emb:
                feature_vector = feature_extractor.get_second_innings_feature_embedding_vector(predicted_first_innings_run,team, opponent, location,
                                                                                             team_player_list,
                                                                                             opponent_player_list,
                                                                                             ref_date=ref_date)
            else:

                feature_vector = feature_extractor.get_second_innings_feature_vector(predicted_first_innings_run, team, opponent,\
                                                                                    location, team_player_list,\
                                                                                    opponent_player_list, ref_date=ref_date)



            second_innings_feature_vector_list.append(feature_vector)
            if second_innings_team['second_innings'].values[0]==second_innings_team['winner'].values[0]:
                result_list.append(1)
            else:
                result_list.append(0)

        except Exception as ex:
            print(match_id,': 2nd innings Exception for match between  ',team,' and ',opponent,' on ',ref_date,' at ',location)
            print(ex)

    second_innings_feature_matrix = np.stack(second_innings_feature_vector_list)
    actual_results = np.stack(result_list)

    predicted_results = predictor_second_innings.predict(second_innings_feature_matrix)

    accuracy = accuracy_score(actual_results, predicted_results)

    print("=====First Innings====== ")
    print('mape :', mape_val)
    print('mae :', mae)
    print('mse :', mse)
    print('total data ', len(actual_runs))
    print("=====Combined Innings====== ")
    print('accuracy ',accuracy)
    print('data size ', len(result_list))

    if environment != "production":
        outil.create_model_meta_info_entry("combined_validation_fie_"+
                                           str(first_innings_emb)+"_sie_"+
                                           str(second_innings_emb)+"_fim_"+
                                           first_emb_model,
                                           0,accuracy,info=from_date+"_to_"+to_date+
                                           ": first innings embedding - "+str(first_innings_emb)+
                                           ": second innings embedding - " + str(second_innings_emb))

    # print("predictions ",predicted_results)
    # print("actuals ", actual_results)
    return mape_val,mae,mse,accuracy,len(result_list)


@click.group()
def evaluate():
    pass

@evaluate.command()
@click.option('--from_date', help='start date in YYYY-mm-dd',required=True)
@click.option('--to_date', help='end date in YYYY-mm-dd',required=True)
@click.option('--env', help='end date in YYYY-mm-dd',default='production')
@click.option('--use_emb', help='set False to use base model',default=True,type=bool)
@click.option('--model', help='use team for team with batsman/advesarial for player wise',default='team')
def first(from_date,to_date,env,use_emb,model):
    evaluate_first_innings(from_date, to_date, environment=env,use_emb=use_emb,model=model)

@evaluate.command()
@click.option('--from_date', help='start date in YYYY-mm-dd',required=True)
@click.option('--to_date', help='end date in YYYY-mm-dd',required=True)
@click.option('--env', help='end date in YYYY-mm-dd',default='production')
@click.option('--use_emb', help='set False to use base model',default=True,type=bool)
def second(from_date,to_date,env,use_emb):
    evaluate_second_innings(from_date, to_date, environment=env,use_emb=use_emb)

@evaluate.command()
@click.option('--from_date', help='start date in YYYY-mm-dd',required=True)
@click.option('--to_date', help='end date in YYYY-mm-dd',required=True)
@click.option('--env', help='end date in YYYY-mm-dd',default='production')
@click.option('--first_innings_emb', help='set False to use base model',default=True,type=bool)
@click.option('--second_innings_emb', help='set False to use base model',default=True,type=bool)
@click.option('--first_emb_model', help='use team for team with batsman/advesarial for player wise',default='team')
@click.option('--second_emb_model', help='use team for team with batsman/advesarial for player wise',default='team')
def combined(from_date,to_date,env,first_innings_emb,second_innings_emb,first_emb_model,second_emb_model):
    evaluate_combined_innings(from_date, to_date, environment=env,
                              first_innings_emb = first_innings_emb,second_innings_emb=second_innings_emb,
                              first_emb_model = first_emb_model, second_emb_model = second_emb_model)


@evaluate.command()
@click.option('--from_date', help='start date in YYYY-mm-dd',required=True)
@click.option('--to_date', help='end date in YYYY-mm-dd',required=True)
@click.option('--env', help='end date in YYYY-mm-dd',default='production')
def batting_order(from_date, to_date, env):
    evaluate_batting_position(from_date, to_date, environment=env)


@evaluate.command()
@click.option('--from_date', help='start date in YYYY-mm-dd',required=True)
@click.option('--to_date', help='end date in YYYY-mm-dd',required=True)
@click.option('--env', help='end date in YYYY-mm-dd',default='production')
def batting_recommendation(from_date, to_date, env):
    evaluate_batsman_recommendation(from_date, to_date, environment=env)

@evaluate.command()
@click.option('--from_date', help='start date in YYYY-mm-dd',required=True)
@click.option('--to_date', help='end date in YYYY-mm-dd',required=True)
@click.option('--env', help='end date in YYYY-mm-dd',default='production')
def expected_threshold(from_date, to_date, env):
    evaluate_expected_threshold(from_date, to_date, environment=env)


if __name__=='__main__':
    evaluate()


