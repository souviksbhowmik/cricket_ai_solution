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
from odi.retrain import create_train_test_ci as ctt
from odi.inference import prediction_ci as predci
import math
from odi.inference import prediction as pred



tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def mape(y_true,y_predict):
    return np.sum((np.abs(y_true-y_predict)/y_true)*100)/len(y_true)


def evaluate_threshold(test_start,test_end=None,env='dev'):
    if not os.path.isdir(ctt.TRAIN_TEST_DIR):
        os.makedirs(ctt.TRAIN_TEST_DIR)

    outil.use_model_from(env)
    test_start_dt = cricutil.str_to_date_time(test_start)
    if test_end is None:
        test_end_dt = cricutil.today_as_date_time()
    else:
        test_end_dt = cricutil.str_to_date_time(test_end)

    overall_start = test_start_dt
    overall_end = test_end_dt
    match_list_df = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION + os.sep + 'cricinfo_match_list.csv')
    match_list_df = match_list_df[(match_list_df['date'] >= overall_start) & \
                                  (match_list_df['date'] <= overall_end)]
    #match_list_df = match_list_df[match_list_df['is_dl'] == 0]
    batting_list_df = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION + os.sep + 'cricinfo_batting.csv')
    batting_list_df = batting_list_df[(batting_list_df['date'] >= overall_start) & \
                                      (batting_list_df['date'] <= overall_end)]
    bowling_list_df = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION + os.sep + 'cricinfo_bowling.csv')
    bowling_list_df = bowling_list_df[(bowling_list_df['date'] >= overall_start) & \
                                      (bowling_list_df['date'] <= overall_end)]


    match_id_list = list(match_list_df['match_id'].unique())
    prediction_list = []
    win_list = []
    for index,match_id in tqdm(enumerate(match_id_list)):
        team = match_list_df[match_list_df['match_id'] == match_id].iloc[0]["second_innings"]
        opponent = match_list_df[match_list_df['match_id'] == match_id].iloc[0]["first_innings"]
        winner = match_list_df[match_list_df['match_id'] == match_id].iloc[0]["winner"]
        location = match_list_df[match_list_df['match_id'] == match_id].iloc[0]["location"]
        ref_dt_np = match_list_df[match_list_df['match_id'] == match_id].iloc[0]["date"]
        ref_date = cricutil.pandas_timestamp_to_datetime(ref_dt_np)
        target = match_list_df[match_list_df['match_id'] == match_id].iloc[0]['first_innings_run']
        runs_achieved = match_list_df[match_list_df['match_id'] == match_id].iloc[0]['second_innings_run']

        win = 1*(team==winner)

        team_player_list_df = batting_list_df[
            (batting_list_df['match_id'] == match_id) & (batting_list_df['team'] == team)]

        team_player_list_df = team_player_list_df[['team', 'name', 'position']]


        opponent_bowler_list_df = bowling_list_df[
            (bowling_list_df['match_id'] == match_id) & (bowling_list_df['team'] == opponent)]

        opponent_bowler_list_df = opponent_bowler_list_df[['team', 'name']]


        try:


            threshold = predci.get_optimum_run(team,opponent,location,team_player_list_df,opponent_bowler_list_df,ref_date=ref_date)
            #print(threshold)
            if threshold is None:
                raise Exception ("cannot determine threshold")
            if target > threshold:
                predicted_win = 0
            else:
                predicted_win = 1

            prediction_list.append(predicted_win)
            win_list.append(win)
        except Exception as ex:
            print(ex, ' for ',team, opponent, location, ' on ',ref_date.date() )
            #raise ex



    actual = np.stack(win_list)
    predicted = np.stack(prediction_list)
    accuracy = accuracy_score(actual,predicted)
    print("Threshold accuracy is ",accuracy)
    print("No of matches ", len(prediction_list))

    outil.create_model_meta_info_entry('threshold_validation',
                                       (accuracy),
                                       (),
                                       info=" measured from "+str(test_start_dt)+" to "+str(test_end_dt),
                                       file_list=[ ])


def evaluate_multi_output_neural(env='production'):
    outil.use_model_from(env)

    combined_model = outil.load_keras_model(os.path.join(outil.DEV_DIR, outil.ONE_SHOT_MULTI_NEURAL))
    x1_scaler = pickle.load(open(os.path.join(outil.DEV_DIR, outil.ONE_SHOT_MULTI_SCALER_X1), "rb"))
    x2_scaler = pickle.load(open(os.path.join(outil.DEV_DIR, outil.ONE_SHOT_MULTI_SCALER_X2), "rb"))

    train_x_1 = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.one_shot_multi_train_x_1), 'rb'))
    train_x_2 = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.one_shot_multi_train_x_2), 'rb'))
    train_x_1 = x1_scaler.fit_transform(train_x_1)
    train_x_2 = x2_scaler.fit_transform(train_x_2)

    train_y_1 = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.one_shot_multi_train_y_1), 'rb'))
    train_y_2 = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.one_shot_multi_train_y_2), 'rb'))
    train_y_3 = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.one_shot_multi_train_y_3), 'rb'))

    test_x_1 = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.one_shot_multi_test_x_1), 'rb'))
    test_x_2 = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.one_shot_multi_test_x_2), 'rb'))
    test_x_1 = x1_scaler.transform(test_x_1)
    test_x_2 = x2_scaler.transform(test_x_2)

    test_y_1 = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.one_shot_multi_test_y_1), 'rb'))
    test_y_2 = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.one_shot_multi_test_y_2), 'rb'))
    test_y_3 = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.one_shot_multi_test_y_3), 'rb'))

    train_metrics = combined_model.evaluate([train_x_1, train_x_2], [train_y_1, train_y_3, train_y_2])
    test_metrics = combined_model.evaluate([test_x_1, test_x_2], [test_y_1, test_y_3, test_y_2])

    print("Metrics ( overall loss, mse first innings runs, mse second innings runs, binary cross entropy loss,"+
          "mape first innings, mae second innings, mape second innings, mae second innings, accuracy team A win )")
    print(" Train ")
    print(train_metrics)
    print(" Test ")
    print(test_metrics)


def evaluate_combined_neural(env='production'):
    outil.use_model_from(env)

    train_x = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.second_level_any_inst_train_x), 'rb'))
    train_y = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.second_level_any_inst_train_y), 'rb'))

    test_x = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.second_level_any_inst_test_x), 'rb'))
    test_y = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.second_level_any_inst_test_y), 'rb'))

    train_pipe = pickle.load(open(os.path.join(outil.DEV_DIR,outil.COMBINED_MODEL_NON_NEURAL),'rb'))

    train_predict = train_pipe.predict(train_x)
    test_predict = train_pipe.predict(test_x)

    train_accuracy = accuracy_score(train_y, train_predict)
    test_accuracy = accuracy_score(test_y, test_predict)

    print(" train accuracy ",train_accuracy)
    print(" test accuracy ", test_accuracy)



@click.group()
def evaluate():
    pass


@evaluate.command()
@click.option('--test_start', help='start date in YYYY-mm-dd',required=True)
@click.option('--test_end', help='end date in YYYY-mm-dd',required=True)
@click.option('--env', help='environment dev/production',default='production')
def expected_threshold(test_start,test_end,env):
    evaluate_threshold(test_start=test_start, test_end=test_end,env=env)


@evaluate.command()
@click.option('--env', help='environment dev/production',default='production')
def multi_output_neural(env):
    evaluate_multi_output_neural(env=env)

@evaluate.command()
@click.option('--env', help='environment dev/production',default='production')
def multi_output_neural(env):
    evaluate_multi_output_neural(env=env)

@evaluate.command()
@click.option('--env', help='environment dev/production',default='production')
def combined_neural(env):
    evaluate_combined_neural(env=env)


if __name__=='__main__':
    evaluate()


