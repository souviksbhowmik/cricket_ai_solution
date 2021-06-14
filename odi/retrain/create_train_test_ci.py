import pandas as pd
from odi.feature_engg import util as cricutil
from odi.data_loader import  data_loader as dl
from odi.model_util import odi_util as outil
from odi.feature_engg import feature_extractor_ci as fec
from odi.evaluation import evaluate as cric_eval
from odi.inference import prediction_ci as predc
from datetime import datetime
from odi.inference import prediction_ci as predci
from sklearn.metrics import accuracy_score

import os
import numpy as np
import pickle
from tqdm import tqdm
import click
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

TRAIN_TEST_DIR = 'retrain_xy'
country_emb_feature_team_oh_train_x = 'country_emb_team_oh_train_x.pkl'
country_emb_feature_opponent_oh_train_x = 'country_emb_opponent_oh_train_x.pkl'
country_emb_feature_location_oh_train_x = 'country_emb_location_oh_train_x.pkl'
country_emb_feature_runs_scored_train_y = 'country_emb_runs_scored_train_y.pkl'

country_emb_feature_team_oh_test_x = 'country_emb_team_oh_test_x.pkl'
country_emb_feature_opponent_oh_test_x = 'country_emb_opponent_oh_test_x.pkl'
country_emb_feature_location_oh_test_x = 'country_emb_location_oh_test_x.pkl'
country_emb_feature_runs_scored_test_y = 'country_emb_runs_scored_test_y'

country_emb_feature_2nd_team_oh_train_x = 'country_emb_feature_2nd_team_oh_train_x.pkl'
country_emb_feature_2nd_opponent_oh_train_x = 'country_emb_feature_2nd_opponent_oh_train_x.pkl'
country_emb_feature_2nd_location_oh_train_x = 'country_emb_feature_2nd_location_oh_train_x.pkl'
country_emb_feature_2nd_target_oh_train_x = 'country_emb_feature_2nd_target_oh_train_x.pkl'
country_emb_feature_2nd_win_train_y = 'country_emb_feature_2nd_win_train_y'

country_emb_feature_2nd_team_oh_test_x = 'country_emb_feature_2nd_team_oh_test_x.pkl'
country_emb_feature_2nd_opponent_oh_test_x = 'country_emb_feature_2nd_opponent_oh_test_x.pkl'
country_emb_feature_2nd_location_oh_test_x = 'country_emb_feature_2nd_location_oh_test_x.pkl'
country_emb_feature_2nd_target_oh_test_x = 'country_emb_feature_2nd_target_oh_test_x.pkl'
country_emb_feature_2nd_win_test_y = 'country_emb_feature_2nd_win_test_y.pkl'


batsman_emb_feature_batsman_oh_train_x = 'batsman_emb_batsman_oh_train_x.pkl'
batsman_emb_feature_position_oh_train_x = 'batsman_emb_position_oh_train_x.pkl'
batsman_emb_feature_location_oh_train_x = 'batsman_emb_location_oh_train_x.pkl'
batsman_emb_feature_opponent_oh_train_x = 'batsman_emb_opponent_oh_train_x.pkl'
batsman_emb_feature_runs_scored_train_y = 'batsman_emb_runs_scored_train_y.pkl'

batsman_emb_feature_batsman_oh_test_x = 'batsman_emb_batsman_oh_test_x.pkl'
batsman_emb_feature_position_oh_test_x = 'batsman_emb_position_oh_test_x.pkl'
batsman_emb_feature_location_oh_test_x = 'batsman_emb_location_oh_test_x.pkl'
batsman_emb_feature_opponent_oh_test_x = 'batsman_emb_opponent_oh_test_x.pkl'
batsman_emb_feature_runs_scored_test_y = 'batsman_emb_runs_scored_test_y.pkl'

first_innings_base_train_x = 'first_innings_base_train_x.pkl'
first_innings_base_train_y = 'first_innings_base_train_y.pkl'
first_innings_base_test_x = 'first_innings_base_test_x.pkl'
first_innings_base_test_y = 'first_innings_base_test_y.pkl'
# first_innings_base_scaler = 'first_innings_base_scaler.pkl'
first_innings_base_columns = 'first_innings_base_columns.pkl'

second_innings_base_train_x = 'second_innings_base_train_x.pkl'
second_innings_base_train_y = 'second_innings_base_train_y.pkl'
second_innings_base_train_y_2 = 'second_innings_base_train_y_2.pkl'
second_innings_base_test_x = 'second_innings_base_test_x.pkl'
second_innings_base_test_y = 'second_innings_base_test_y.pkl'
second_innings_base_test_y_2 = 'second_innings_base_test_y_2.pkl'
# second_innings_base_scaler = 'second_innings_base_scaler.pkl'
second_innings_base_columns = 'second_innings_base_columns.pkl'

one_shot_train_x = 'one_shot_train_x.pkl'
one_shot_train_y = 'one_shot_train_y.pkl'
one_shot_test_x = 'one_shot_test_x.pkl'
one_shot_test_y = 'one_shot_test_y.pkl'
# first_innings_base_scaler = 'first_innings_base_scaler.pkl'
one_shot_columns = 'one_shot_columns.pkl'

one_shot_multi_train_x_1 = 'one_shot_multi_train_x_1.pkl'
one_shot_multi_train_y_1 = 'one_shot_multi_train_y_1.pkl'
one_shot_multi_test_x_1 = 'one_shot_multi_test_x_1.pkl'
one_shot_multi_test_y_1 = 'one_shot_multi_test_y_1.pkl'
one_shot_multi_columns_1 = 'one_shot_multi_columns_1.pkl'


one_shot_multi_train_x_2 = 'one_shot_multi_train_x_2.pkl'
one_shot_multi_train_y_2 = 'one_shot_multi_train_y_2.pkl'
one_shot_multi_train_y_3 = 'one_shot_multi_train_y_3.pkl'
one_shot_multi_test_x_2 = 'one_shot_multi_test_x_2.pkl'
one_shot_multi_test_y_2 = 'one_shot_multi_test_y_2.pkl'
one_shot_multi_test_y_3 = 'one_shot_multi_test_y_3.pkl'
one_shot_multi_columns_2 = 'one_shot_multi_columns_2.pkl'


mg_train_x = 'mg_train_x.pkl'
mg_test_x = 'mg_test_x.pkl'

mg_train_y = 'mg_train_y.pkl'
mg_test_y = 'mg_test_y.pkl'

second_level_any_inst_train_x = 'second_level_any_inst_train_x.pkl'
second_level_any_inst_train_y = 'second_level_any_inst_train_y.pkl'
second_level_any_inst_test_x = 'second_level_any_inst_test_x.pkl'
second_level_any_inst_test_y = 'second_level_any_inst_test_y.pkl'

second_level_non_neural_train_x = 'second_level_non_neural_train_x.pkl'
second_level_non_neural_train_y = 'second_level_non_neural_train_y.pkl'
second_level_non_neural_test_x = 'second_level_non_neural_test_x.pkl'
second_level_non_neural_test_y = 'second_level_non_neural_test_y.pkl'

first_innings_train_x = 'first_innings_train_x.pkl'
first_innings_train_y = 'first_innings_train_y.pkl'
first_innings_test_x = 'first_innings_test_x.pkl'
first_innings_test_y = 'first_innings_test_y.pkl'

second_innings_train_x = 'second_innings_train_x.pkl'
second_innings_train_y = 'second_innings_train_y.pkl'
second_innings_test_x = 'second_innings_test_x.pkl'
second_innings_test_y = 'second_innings_test_y.pkl'

batsman_runs_train_x = 'batsman_runs_train_x.pkl'
batsman_runs_train_y = 'batsman_runs_train_y.pkl'
batsman_runs_test_x = 'batsman_runs_test_x.pkl'
batsman_runs_test_y = 'batsman_runs_test_y.pkl'

adversarial_feature_batsman_oh_train_x = 'adversarial_feature_batsman_oh_train_x.pkl'
adversarial_feature_position_oh_train_x = 'adversarial_feature_position_oh_train_x.pkl'
adversarial_feature_location_oh_train_x = 'adversarial_feature_location_oh_train_x.pkl'
adversarial_feature_bowler_oh_train_x = 'adversarial_feature_bowler_oh_train_x.pkl'
adversarial_feature_runs_scored_train_y = 'adversarial_feature_runs_scored_train_y.pkl'

adversarial_feature_batsman_oh_test_x = 'adversarial_feature_batsman_oh_test_x.pkl'
adversarial_feature_position_oh_test_x = 'adversarial_feature_position_oh_test_x.pkl'
adversarial_feature_location_oh_test_x = 'adversarial_feature_location_oh_test_x.pkl'
adversarial_feature_bowler_oh_test_x = 'adversarial_feature_bowler_oh_test_x.pkl'
adversarial_feature_runs_scored_test_y = 'adversarial_feature_runs_scored_test_y.pkl'

adversarial_first_innings_train_x = 'adversarial_first_innings_train_x.pkl'
adversarial_first_innings_train_y = 'adversarial_first_innings_train_y.pkl'
adversarial_first_innings_test_x = 'adversarial_first_innings_test_x'
adversarial_first_innings_test_y = 'adversarial_first_innings_test_y'

combined_train_x = 'combined_train_x.pkl'
combined_train_y = 'combined_train_y.pkl'
combined_test_x = 'combined_test_x.pkl'
combined_test_y = 'combined_test_y.pkl'

default_location = 'The Oval'



def create_country_embedding_train_test(train_start,test_start,test_end=None,encoding_source='dev'):

    if not os.path.isdir(TRAIN_TEST_DIR):
        os.makedirs(TRAIN_TEST_DIR)

    if encoding_source == 'production':
        encoding_dir = outil.PROD_DIR
    else:
        encoding_dir = outil.DEV_DIR

    train_start_dt = cricutil.str_to_date_time(train_start)
    test_start_dt = cricutil.str_to_date_time(test_start)
    if test_end is None:
        test_end_dt = cricutil.today_as_date_time()
    else:
        test_end_dt = cricutil.str_to_date_time(test_end)

    overall_start = train_start_dt
    overall_end = test_end_dt
    match_list_df = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION + os.sep + 'cricinfo_match_list.csv')
    match_list_df = match_list_df[(match_list_df['date'] >= overall_start) & \
                                  (match_list_df['date'] <= overall_end)]
    # match_stats_df = pd.read_csv(dl.CSV_LOAD_LOCATION + os.sep + 'match_stats.csv')
    # match_list_df = match_list_df.merge(match_stats_df,how='inner',on='match_id')


    no_of_matches = match_list_df.shape[0]
    country_enc_map = pickle.load(open(os.path.join(encoding_dir,outil.COUNTRY_ENCODING_MAP),'rb'))
    location_enc_map = pickle.load(open(os.path.join(encoding_dir, outil.LOC_ENCODING_MAP), 'rb'))

    team_oh_list_train = []
    opponent_oh_list_train = []
    location_oh_list_train =[]
    runs_scored_list_train =[]

    team_oh_list_test = []
    opponent_oh_list_test = []
    location_oh_list_test = []
    runs_scored_list_test = []

    for index in tqdm(range(no_of_matches)):

        # if match_list_df.iloc[index]['first_innings'].strip() == match_list_df.iloc[index]['team_statistics'].strip():
        #     batting_innings = 'first_innings'
        #     bowling_innings = 'second_innings'
        # else:
        #     batting_innings = 'second_innings'
        #     bowling_innings = 'first_innings'
        #     continue


        team = match_list_df.iloc[index]['first_innings'].strip()
        opponent = match_list_df.iloc[index]['second_innings'].strip()
        location = match_list_df.iloc[index]['location'].strip()
        runs_scored = match_list_df.iloc[index]['first_innings_run']
        date = match_list_df.iloc[index]['date']
        try:
            team_oh = np.array(country_enc_map[team])
            opponent_oh = np.array(country_enc_map[opponent])
            location_oh = np.array(location_enc_map[location])

            if date<test_start_dt:
                team_oh_list_train.append(team_oh)
                opponent_oh_list_train.append(opponent_oh)
                location_oh_list_train.append(location_oh)
                runs_scored_list_train.append(runs_scored)
            else:
                team_oh_list_test.append(team_oh)
                opponent_oh_list_test.append(opponent_oh)
                location_oh_list_test.append(location_oh)
                runs_scored_list_test.append(runs_scored)


        except Exception as ex:
            print(ex,' for ',team,opponent,location,' on ',date)
            #raise ex

    team_oh_train_x = np.stack(team_oh_list_train)
    opponent_oh_train_x = np.stack(opponent_oh_list_train)
    location_oh_train_x = np.stack(location_oh_list_train)
    runs_scored_train_y = np.stack(runs_scored_list_train)

    team_oh_test_x = np.stack(team_oh_list_test)
    opponent_oh_test_x = np.stack(opponent_oh_list_test)
    location_oh_test_x = np.stack(location_oh_list_test)
    runs_scored_test_y = np.stack(runs_scored_list_test)

    pickle.dump(team_oh_train_x, open(os.path.join(TRAIN_TEST_DIR, country_emb_feature_team_oh_train_x), 'wb'))
    pickle.dump(opponent_oh_train_x, open(os.path.join(TRAIN_TEST_DIR, country_emb_feature_opponent_oh_train_x), 'wb'))
    pickle.dump(location_oh_train_x, open(os.path.join(TRAIN_TEST_DIR, country_emb_feature_location_oh_train_x), 'wb'))
    pickle.dump(runs_scored_train_y, open(os.path.join(TRAIN_TEST_DIR, country_emb_feature_runs_scored_train_y), 'wb'))

    outil.create_meta_info_entry('country_embedding_train_xy', train_start,
                                 str(cricutil.substract_day_as_datetime(test_start_dt,1).date()),
                                 file_list=[country_emb_feature_team_oh_train_x,
                                            country_emb_feature_opponent_oh_train_x,
                                            country_emb_feature_location_oh_train_x,
                                            country_emb_feature_runs_scored_train_y])

    pickle.dump(team_oh_test_x, open(os.path.join(TRAIN_TEST_DIR, country_emb_feature_team_oh_test_x), 'wb'))
    pickle.dump(opponent_oh_test_x, open(os.path.join(TRAIN_TEST_DIR, country_emb_feature_opponent_oh_test_x), 'wb'))
    pickle.dump(location_oh_test_x, open(os.path.join(TRAIN_TEST_DIR, country_emb_feature_location_oh_test_x), 'wb'))
    pickle.dump(runs_scored_test_y, open(os.path.join(TRAIN_TEST_DIR, country_emb_feature_runs_scored_test_y), 'wb'))

    outil.create_meta_info_entry('country_embedding_test_xy', test_start, str(test_end_dt.date()),
                                 file_list=[country_emb_feature_team_oh_test_x,
                                            country_emb_feature_opponent_oh_test_x,
                                            country_emb_feature_location_oh_test_x,
                                            country_emb_feature_runs_scored_test_y])

def set_target_quantile(q1,q2,q3,val):
    if val < q3:
        return 4
    elif val > q2:
        return 3
    elif val > q1:
        return 2
    else:
        return 1

def get_quantile_vector(q):
    vec = np.array([0,0,0,0])
    vec[q-1]=1
    return vec

def create_country_embedding_second_innings_train_test(train_start,test_start,test_end=None,encoding_source='dev'):

    if not os.path.isdir(TRAIN_TEST_DIR):
        os.makedirs(TRAIN_TEST_DIR)

    if encoding_source == 'production':
        encoding_dir = outil.PROD_DIR
    else:
        encoding_dir = outil.DEV_DIR

    train_start_dt = cricutil.str_to_date_time(train_start)
    test_start_dt = cricutil.str_to_date_time(test_start)
    if test_end is None:
        test_end_dt = cricutil.today_as_date_time()
    else:
        test_end_dt = cricutil.str_to_date_time(test_end)

    overall_start = train_start_dt
    overall_end = test_end_dt
    match_list_df = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION + os.sep + 'cricinfo_match_list.csv')
    match_list_df = match_list_df[(match_list_df['date'] >= overall_start) & \
                                  (match_list_df['date'] <= overall_end)]
    # match_stats_df = pd.read_csv(dl.CSV_LOAD_LOCATION + os.sep + 'match_stats.csv')
    # match_list_df = match_list_df.merge(match_stats_df,how='inner',on='match_id')
    target_run_list = list(match_list_df['first_innings_run'])
    q1 = np.quantile(target_run_list, 0.25)
    q2 = np.quantile(target_run_list, 0.50)
    q3 = np.quantile(target_run_list, 0.75)
    match_list_df['target_q'] = match_list_df['first_innings_run'].apply(lambda x:set_target_quantile(q1,q2,q3,x))




    no_of_matches = match_list_df.shape[0]
    country_enc_map = pickle.load(open(os.path.join(encoding_dir,outil.COUNTRY_ENCODING_MAP),'rb'))
    location_enc_map = pickle.load(open(os.path.join(encoding_dir, outil.LOC_ENCODING_MAP), 'rb'))

    team_oh_list_train = []
    opponent_oh_list_train = []
    location_oh_list_train =[]
    target_oh_list_train = []
    runs_scored_list_train =[]

    team_oh_list_test = []
    opponent_oh_list_test = []
    location_oh_list_test = []
    target_oh_list_test = []
    runs_scored_list_test = []

    for index in tqdm(range(no_of_matches)):

        # if match_list_df.iloc[index]['first_innings'].strip() == match_list_df.iloc[index]['team_statistics'].strip():
        #     batting_innings = 'first_innings'
        #     bowling_innings = 'second_innings'
        # else:
        #     batting_innings = 'second_innings'
        #     bowling_innings = 'first_innings'
        #     continue


        team = match_list_df.iloc[index]['second_innings'].strip()
        opponent = match_list_df.iloc[index]['first_innings'].strip()
        location = match_list_df.iloc[index]['location'].strip()
        winner =  match_list_df.iloc[index]['winner'].strip()
        win = 1*(team == winner)
        target_q = match_list_df.iloc[index]['target_q']
        date = match_list_df.iloc[index]['date']
        try:
            team_oh = np.array(country_enc_map[team])
            opponent_oh = np.array(country_enc_map[opponent])
            location_oh = np.array(location_enc_map[location])
            target_oh = get_quantile_vector(target_q)

            if date<test_start_dt:
                team_oh_list_train.append(team_oh)
                opponent_oh_list_train.append(opponent_oh)
                location_oh_list_train.append(location_oh)
                target_oh_list_train.append(target_oh)
                runs_scored_list_train.append(win)
            else:
                team_oh_list_test.append(team_oh)
                opponent_oh_list_test.append(opponent_oh)
                location_oh_list_test.append(location_oh)
                target_oh_list_test.append(target_oh)
                runs_scored_list_test.append(win)


        except Exception as ex:
            print(ex,' for ',team,opponent,location,' on ',date)
            #raise ex

    team_oh_train_x = np.stack(team_oh_list_train)
    opponent_oh_train_x = np.stack(opponent_oh_list_train)
    location_oh_train_x = np.stack(location_oh_list_train)
    target_oh_train_x = np.stack(target_oh_list_train)
    runs_scored_train_y = np.stack(runs_scored_list_train)

    team_oh_test_x = np.stack(team_oh_list_test)
    opponent_oh_test_x = np.stack(opponent_oh_list_test)
    location_oh_test_x = np.stack(location_oh_list_test)
    target_oh_test_x = np.stack(target_oh_list_test)
    runs_scored_test_y = np.stack(runs_scored_list_test)

    pickle.dump(team_oh_train_x, open(os.path.join(TRAIN_TEST_DIR, country_emb_feature_2nd_team_oh_train_x), 'wb'))
    pickle.dump(opponent_oh_train_x, open(os.path.join(TRAIN_TEST_DIR, country_emb_feature_2nd_opponent_oh_train_x), 'wb'))
    pickle.dump(location_oh_train_x, open(os.path.join(TRAIN_TEST_DIR, country_emb_feature_2nd_location_oh_train_x), 'wb'))
    pickle.dump(location_oh_train_x,
                open(os.path.join(TRAIN_TEST_DIR, country_emb_feature_2nd_target_oh_train_x), 'wb'))
    pickle.dump(runs_scored_train_y, open(os.path.join(TRAIN_TEST_DIR, country_emb_feature_2nd_win_train_y), 'wb'))

    outil.create_meta_info_entry('country_embedding_2nd_train_xy', train_start,
                                 str(cricutil.substract_day_as_datetime(test_start_dt,1).date()),
                                 file_list=[country_emb_feature_2nd_team_oh_train_x,
                                            country_emb_feature_2nd_opponent_oh_train_x,
                                            country_emb_feature_2nd_location_oh_train_x,
                                            country_emb_feature_2nd_target_oh_train_x,
                                            country_emb_feature_2nd_win_train_y])

    pickle.dump(team_oh_test_x, open(os.path.join(TRAIN_TEST_DIR, country_emb_feature_2nd_team_oh_test_x), 'wb'))
    pickle.dump(opponent_oh_test_x, open(os.path.join(TRAIN_TEST_DIR, country_emb_feature_2nd_opponent_oh_test_x), 'wb'))
    pickle.dump(location_oh_test_x, open(os.path.join(TRAIN_TEST_DIR, country_emb_feature_2nd_location_oh_test_x), 'wb'))
    pickle.dump(location_oh_train_x,
                open(os.path.join(TRAIN_TEST_DIR, country_emb_feature_2nd_target_oh_test_x), 'wb'))
    pickle.dump(runs_scored_test_y, open(os.path.join(TRAIN_TEST_DIR, country_emb_feature_2nd_win_test_y), 'wb'))

    outil.create_meta_info_entry('country_embedding_2nd_test_xy', test_start, str(test_end_dt.date()),
                                 file_list=[country_emb_feature_2nd_team_oh_test_x,
                                            country_emb_feature_2nd_opponent_oh_test_x,
                                            country_emb_feature_2nd_location_oh_test_x,
                                            country_emb_feature_2nd_target_oh_test_x,
                                            country_emb_feature_2nd_win_test_y])


def create_batsman_embedding_train_test(train_start,test_start,test_end=None,encoding_source='dev',include_not_batted=False):
    if not os.path.isdir(TRAIN_TEST_DIR):
        os.makedirs(TRAIN_TEST_DIR)

    if encoding_source == 'production':
        encoding_dir = outil.PROD_DIR
    else:
        encoding_dir = outil.DEV_DIR

    outil.use_model_from('dev')

    train_start_dt = cricutil.str_to_date_time(train_start)
    test_start_dt = cricutil.str_to_date_time(test_start)
    if test_end is None:
        test_end_dt = cricutil.today_as_date_time()
    else:
        test_end_dt = cricutil.str_to_date_time(test_end)

    overall_start = train_start_dt
    overall_end = test_end_dt
    match_list_df = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION + os.sep + 'match_list.csv')
    match_list_df = match_list_df[(match_list_df['date'] >= overall_start) & \
                                  (match_list_df['date'] <= overall_end)]
    match_stats_df = pd.read_csv(dl.CSV_LOAD_LOCATION + os.sep + 'match_stats.csv')
    match_list_df = match_list_df.merge(match_stats_df,how='inner',on='match_id')

    no_of_matches = match_list_df.shape[0]
    country_enc_map = pickle.load(open(os.path.join(encoding_dir, outil.COUNTRY_ENCODING_MAP), 'rb'))
    location_enc_map_for_batsman = pickle.load(open(os.path.join(encoding_dir, outil.LOC_ENCODING_MAP_FOR_BATSMAN), 'rb'))
    batsman_enc_map = pickle.load(open(os.path.join(encoding_dir, outil.BATSMAN_ENCODING_MAP), 'rb'))

    batsman_oh_list_train = []
    position_oh_list_train = []
    location_oh_list_train = []
    opponent_oh_list_train = []
    runs_scored_list_train = []

    batsman_oh_list_test = []
    position_oh_list_test = []
    location_oh_list_test = []
    opponent_oh_list_test = []
    runs_scored_list_test = []

    for index in tqdm(range(no_of_matches)):
        if match_list_df.iloc[index]['first_innings']==match_list_df.iloc[index]['team_statistics']:
            opponent = match_list_df.iloc[index]['second_innings'].strip()
            country = match_list_df.iloc[index]['first_innings'].strip()
        else:
            opponent = match_list_df.iloc[index]['first_innings'].strip()
            country = match_list_df.iloc[index]['second_innings'].strip()
        location = match_list_df.iloc[index]['location'].strip()
        match_id = match_list_df.iloc[index]['match_id']

        date = match_list_df.iloc[index]['date']

        if location not in location_enc_map_for_batsman:
            try:
                location = fe.get_similar_location(location).strip()
            except:
                print('location ',location,' not encoded for ',country,opponent,' on ',date)
                continue

        if opponent not in country_enc_map:
            print('opponent ', opponent, ' not encoded for ', country, opponent, ' on ', date, ' at ',location)
            continue

        location_oh = np.array(location_enc_map_for_batsman[location])
        opponent_oh = country_enc_map[opponent]

        for bi in range(11):
            batsman = match_list_df.iloc[index]['batsman_'+str(bi+1)].strip()
            try:
                if batsman == 'not_batted':
                    if not include_not_batted:
                        break
                    else:
                        batsman_oh = np.array(batsman_enc_map[batsman])
                else:

                    batsman_oh = np.array(batsman_enc_map[country+' '+batsman.strip()])
                position_oh = fe.get_oh_pos(bi+1)
                runs_scored = match_list_df.iloc[index]['batsman_'+str(bi+1)+'_runs']

                if date<test_start_dt:
                    batsman_oh_list_train.append(batsman_oh)
                    position_oh_list_train.append(position_oh)
                    location_oh_list_train.append(location_oh)
                    opponent_oh_list_train.append(opponent_oh)
                    runs_scored_list_train.append(runs_scored)
                else:
                    batsman_oh_list_test.append(batsman_oh)
                    position_oh_list_test.append(position_oh)
                    location_oh_list_test.append(location_oh)
                    opponent_oh_list_test.append(opponent_oh)
                    runs_scored_list_test.append(runs_scored)
            except Exception as ex:
                print(ex,' for ',country,batsman,opponent,date,' -match_id: ',match_id)

    batsman_oh_train_x = np.stack(batsman_oh_list_train)
    position_oh_train_x = np.stack(position_oh_list_train)
    location_oh_train_x = np.stack(location_oh_list_train)
    opponent_oh_train_x = np.stack(opponent_oh_list_train)
    runs_scored_train_y = np.stack(runs_scored_list_train)

    batsman_oh_test_x = np.stack(batsman_oh_list_test)
    position_oh_test_x = np.stack(position_oh_list_test)
    location_oh_test_x = np.stack(location_oh_list_test)
    opponent_oh_test_x = np.stack(opponent_oh_list_test)
    runs_scored_test_y = np.stack(runs_scored_list_test)

    pickle.dump(batsman_oh_train_x, open(os.path.join(TRAIN_TEST_DIR, batsman_emb_feature_batsman_oh_train_x), 'wb'))
    pickle.dump(position_oh_train_x, open(os.path.join(TRAIN_TEST_DIR, batsman_emb_feature_position_oh_train_x), 'wb'))
    pickle.dump(location_oh_train_x, open(os.path.join(TRAIN_TEST_DIR, batsman_emb_feature_location_oh_train_x), 'wb'))
    pickle.dump(opponent_oh_train_x, open(os.path.join(TRAIN_TEST_DIR, batsman_emb_feature_opponent_oh_train_x), 'wb'))
    pickle.dump(runs_scored_train_y, open(os.path.join(TRAIN_TEST_DIR, batsman_emb_feature_runs_scored_train_y), 'wb'))

    outil.create_meta_info_entry('batsman_embedding_train_xy', train_start,
                                 str(cricutil.substract_day_as_datetime(test_start_dt, 1).date()),
                                 file_list=[batsman_emb_feature_batsman_oh_train_x,
                                            batsman_emb_feature_position_oh_train_x,
                                            batsman_emb_feature_location_oh_train_x,
                                            batsman_emb_feature_opponent_oh_train_x,
                                            batsman_emb_feature_runs_scored_train_y])

    pickle.dump(batsman_oh_test_x, open(os.path.join(TRAIN_TEST_DIR, batsman_emb_feature_batsman_oh_test_x), 'wb'))
    pickle.dump(position_oh_test_x, open(os.path.join(TRAIN_TEST_DIR, batsman_emb_feature_position_oh_test_x), 'wb'))
    pickle.dump(location_oh_test_x, open(os.path.join(TRAIN_TEST_DIR, batsman_emb_feature_location_oh_test_x), 'wb'))
    pickle.dump(opponent_oh_test_x, open(os.path.join(TRAIN_TEST_DIR, batsman_emb_feature_opponent_oh_test_x), 'wb'))
    pickle.dump(runs_scored_test_y, open(os.path.join(TRAIN_TEST_DIR, batsman_emb_feature_runs_scored_test_y), 'wb'))

    outil.create_meta_info_entry('batsman_embedding_test_xy', test_start, str(test_end_dt.date()),
                                 file_list=[batsman_emb_feature_batsman_oh_test_x,
                                            batsman_emb_feature_position_oh_test_x,
                                            batsman_emb_feature_location_oh_test_x,
                                            batsman_emb_feature_opponent_oh_test_x,
                                            batsman_emb_feature_runs_scored_test_y])


def create_one_shot_prediction_train_test(train_start,test_start,test_end=None):
    if not os.path.isdir(TRAIN_TEST_DIR):
        os.makedirs(TRAIN_TEST_DIR)

    outil.use_model_from('dev')
    train_start_dt = cricutil.str_to_date_time(train_start)
    test_start_dt = cricutil.str_to_date_time(test_start)
    if test_end is None:
        test_end_dt = cricutil.today_as_date_time()
    else:
        test_end_dt = cricutil.str_to_date_time(test_end)

    overall_start = train_start_dt
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
    feature_list_train = []
    target_list_train = []

    feature_list_test =[]
    target_list_test = []
    #no_of_basman = 0
    for index,match_id in tqdm(enumerate(match_id_list)):


        team_a = match_list_df[match_list_df['match_id']==match_id].iloc[0]["first_innings"]
        team_b = match_list_df[match_list_df['match_id']==match_id].iloc[0]["second_innings"]
        location = match_list_df[match_list_df['match_id']==match_id].iloc[0]["location"]
        ref_dt_np = match_list_df[match_list_df['match_id']==match_id].iloc[0]["date"]
        ref_date = cricutil.pandas_timestamp_to_datetime(ref_dt_np)
        runs_scored = match_list_df[match_list_df['match_id']==match_id].iloc[0]['first_innings_run']
        winner = match_list_df[match_list_df['match_id']==match_id].iloc[0]["winner"]
        team_a_win = (team_a==winner)*1

        team_a_player_list_df = batting_list_df[(batting_list_df['match_id']==match_id) & (batting_list_df['team']==team_a)]

        team_a_player_list_df = team_a_player_list_df[['team', 'name', 'position']]

        team_b_player_list_df = batting_list_df[
            (batting_list_df['match_id'] == match_id) & (batting_list_df['team'] == team_b)]

        team_b_player_list_df = team_b_player_list_df[['team', 'name', 'position']]

        team_a_bowler_list = bowling_list_df[
            (bowling_list_df['match_id'] == match_id) & (bowling_list_df['team'] == team_a)]

        team_a_bowler_list = team_a_bowler_list[['team', 'name']]


        team_b_bowler_list = bowling_list_df[(bowling_list_df['match_id'] == match_id) & (bowling_list_df['team'] == team_b)]

        team_b_bowler_list = team_b_bowler_list[['team', 'name']]

        try:

            feature_dict = fec.get_one_shot_feature_dict(team_a, team_b, location, team_a_player_list_df,team_b_player_list_df, team_a_bowler_list,team_b_bowler_list, ref_date=ref_date,no_of_years=None)
            #print(feature_dict)
            #no_of_basman = no_of_basman+len(team_player_list)

            if ref_date<test_start_dt:
                feature_list_train.append(feature_dict)
                target_list_train.append(team_a_win)
            else:
                feature_list_test.append(feature_dict)
                target_list_test.append(team_a_win)
        except Exception as ex:
            print(ex, ' for ',team_a, team_b, location, ' on ',ref_date.date() )
            #raise ex

    train_y = np.stack(target_list_train)
    test_y = np.stack(target_list_test)

    #print(pd.DataFrame(feature_list_train))
    train_x = np.array(pd.DataFrame(feature_list_train).drop(columns=['team_a','team_b','location']))
    #print(train_x)
    test_x  = np.array(pd.DataFrame(feature_list_test).drop(columns=['team_a','team_b','location']))
    cols = list(pd.DataFrame(feature_list_train).drop(columns=['team_a','team_b','location']).columns)

    pickle.dump(train_x,open(os.path.join(TRAIN_TEST_DIR,one_shot_train_x),'wb'))
    pickle.dump(train_y, open(os.path.join(TRAIN_TEST_DIR,one_shot_train_y), 'wb'))
    pickle.dump(cols, open(os.path.join(outil.DEV_DIR, one_shot_columns), 'wb'))

    outil.create_meta_info_entry('one_shot_train_xy', train_start,
                                 str(cricutil.substract_day_as_datetime(test_start_dt, 1).date()),
                                 file_list=[one_shot_train_x,
                                            one_shot_train_y,
                                            one_shot_columns])

    pickle.dump(test_x, open(os.path.join(TRAIN_TEST_DIR, one_shot_test_x), 'wb'))
    pickle.dump(test_y, open(os.path.join(TRAIN_TEST_DIR, one_shot_test_y), 'wb'))

    outil.create_meta_info_entry('one_shot_test_xy', str(test_start_dt.date()),
                                 str(test_end_dt.date()),
                                 file_list=[one_shot_test_x,
                                            one_shot_test_y])

    print("train size ",train_x.shape)
    print("test size ", test_x.shape)


def create_one_shot_multi_output_train_test(train_start,test_start,test_end=None,embedding=False):
    if not os.path.isdir(TRAIN_TEST_DIR):
        os.makedirs(TRAIN_TEST_DIR)

    outil.use_model_from('dev')
    train_start_dt = cricutil.str_to_date_time(train_start)
    test_start_dt = cricutil.str_to_date_time(test_start)
    if test_end is None:
        test_end_dt = cricutil.today_as_date_time()
    else:
        test_end_dt = cricutil.str_to_date_time(test_end)

    overall_start = train_start_dt
    overall_end = test_end_dt
    match_list_df = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION + os.sep + 'cricinfo_match_list.csv')
    match_list_df = match_list_df[(match_list_df['date'] >= overall_start) & \
                                  (match_list_df['date'] <= overall_end)]

    match_list_df = match_list_df[match_list_df['is_dl']==0]
    batting_list_df = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION + os.sep + 'cricinfo_batting.csv')
    batting_list_df = batting_list_df[(batting_list_df['date'] >= overall_start) & \
                                  (batting_list_df['date'] <= overall_end)]
    bowling_list_df = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION + os.sep + 'cricinfo_bowling.csv')
    bowling_list_df = bowling_list_df[(bowling_list_df['date'] >= overall_start) & \
                                      (bowling_list_df['date'] <= overall_end)]

    match_id_list = list(match_list_df['match_id'].unique())
    feature_list_team_a_train = []
    feature_list_team_b_train = []
    target_list_train = []
    achieved_list_train = []
    win_list_train = []

    feature_list_team_a_test =[]
    feature_list_team_b_test = []
    target_list_test = []
    achieved_list_test = []
    win_list_test = []
    #no_of_basman = 0
    print("iterations needed ",len(match_id_list))
    for index,match_id in tqdm(enumerate(match_id_list)):


        team_a = match_list_df[match_list_df['match_id']==match_id].iloc[0]["first_innings"]
        team_b = match_list_df[match_list_df['match_id']==match_id].iloc[0]["second_innings"]
        location = match_list_df[match_list_df['match_id']==match_id].iloc[0]["location"]
        ref_dt_np = match_list_df[match_list_df['match_id']==match_id].iloc[0]["date"]
        ref_date = cricutil.pandas_timestamp_to_datetime(ref_dt_np)
        runs_scored = match_list_df[match_list_df['match_id']==match_id].iloc[0]['first_innings_run']

        winner = match_list_df[match_list_df['match_id']==match_id].iloc[0]["winner"]
        runs_achieved = match_list_df[match_list_df['match_id'] == match_id].iloc[0]['second_innings_run']
        if winner == team_b:
            #runs_achieved = runs_achieved+15
            #adjust runcs achieved
            chasing_overs = round(match_list_df[match_list_df['match_id']==match_id].iloc[0]['chasing_overs'])
            if chasing_overs == 0:
                runs_achieved = runs_achieved + 40
            elif chasing_overs >= 49:
                runs_achieved = runs_achieved + 15
            else:
                runs_achieved=(runs_achieved/chasing_overs)*50
        # else:
        #     runs_achieved = runs_achieved - 10
        team_a_win = (team_a==winner)*1

        team_a_player_list_df = batting_list_df[(batting_list_df['match_id']==match_id) & (batting_list_df['team']==team_a)]

        team_a_player_list_df = team_a_player_list_df[['team', 'name', 'position']]

        team_b_player_list_df = batting_list_df[
            (batting_list_df['match_id'] == match_id) & (batting_list_df['team'] == team_b)]

        team_b_player_list_df = team_b_player_list_df[['team', 'name', 'position']]

        team_a_bowler_list = bowling_list_df[
            (bowling_list_df['match_id'] == match_id) & (bowling_list_df['team'] == team_a)]

        team_a_bowler_list = team_a_bowler_list[['team', 'name']]


        team_b_bowler_list = bowling_list_df[(bowling_list_df['match_id'] == match_id) & (bowling_list_df['team'] == team_b)]

        team_b_bowler_list = team_b_bowler_list[['team', 'name']]

        try:

            feature_dict_team_a,feature_dict_team_b = fec.get_one_shot_multi_output_feature_dict(team_a, team_b, location,
                                                                                                 team_a_player_list_df,team_b_player_list_df,
                                                                                                 team_a_bowler_list,team_b_bowler_list,
                                                                                                 ref_date=ref_date,no_of_years=None,embedding=embedding)
            #print(feature_dict)
            #no_of_basman = no_of_basman+len(team_player_list)

            if ref_date<test_start_dt:
                feature_list_team_a_train.append(feature_dict_team_a)
                feature_list_team_b_train.append(feature_dict_team_b)
                target_list_train.append(runs_scored)
                achieved_list_train.append(runs_achieved)
                win_list_train.append(team_a_win)

            else:
                feature_list_team_a_test.append(feature_dict_team_a)
                feature_list_team_b_test.append(feature_dict_team_b)
                target_list_test.append(runs_scored)
                achieved_list_test.append(runs_achieved)
                win_list_test.append(team_a_win)
        except Exception as ex:
            print(ex, ' for ',team_a, team_b, location, ' on ',ref_date.date() )
            #raise ex

    if embedding:
        print("updating with location score")

    first_innings_train_df = pd.DataFrame(feature_list_team_a_train)
    if embedding :
        first_innings_train_df = update_first_innings_location_score(first_innings_train_df)
    second_innings_train_df =  pd.DataFrame(feature_list_team_b_train)
    if embedding:
        second_innings_train_df =  update_second_innings_location_score(second_innings_train_df)
    first_innings_train_df['target'] = target_list_train
    second_innings_train_df['win'] = win_list_train
    second_innings_train_df['achieved'] = achieved_list_train

    first_innings_test_df = pd.DataFrame(feature_list_team_a_test)
    if embedding:
        first_innings_test_df = update_first_innings_location_score(first_innings_test_df)
    second_innings_test_df = pd.DataFrame(feature_list_team_b_test)
    if embedding:
        second_innings_test_df = update_second_innings_location_score(second_innings_test_df)
    first_innings_test_df['target'] = target_list_test
    second_innings_test_df['win'] = win_list_test
    second_innings_test_df['achieved'] = achieved_list_test

    concatenated_train_df = pd.concat([first_innings_train_df,second_innings_train_df],axis=1)
    concatenated_test_df = pd.concat([first_innings_test_df, second_innings_test_df], axis=1)

    concatenated_train_df.dropna(inplace=True)
    concatenated_test_df.dropna(inplace=True)

    first_innings_train_df = concatenated_train_df[list(first_innings_train_df.columns)]
    second_innings_train_df = concatenated_train_df[list(second_innings_train_df.columns)]
    target_list_train = list(concatenated_train_df['target'])
    win_list_train = list(concatenated_train_df['win'])
    achieved_list_train = list(concatenated_train_df['achieved'])

    first_innings_test_df = concatenated_test_df[list(first_innings_test_df.columns)]
    second_innings_test_df = concatenated_test_df[list(second_innings_test_df.columns)]
    target_list_test = list(concatenated_test_df['target'])
    win_list_test = list(concatenated_test_df['win'])
    achieved_list_est = list(concatenated_test_df['achieved'])

    train_x_1 = np.array(first_innings_train_df.drop(columns=['first_team_a', 'first_team_b', 'first_location','target']))
    train_x_2 = np.array(second_innings_train_df.drop(columns=['second_team_a', 'second_team_b', 'second_location','win','achieved']))

    train_y_1 =  np.array(concatenated_train_df['target'])
    train_y_2 = np.array(concatenated_train_df['win'])
    train_y_3 = np.array(concatenated_train_df['achieved'])

    test_x_1 = np.array(first_innings_test_df.drop(columns=['first_team_a', 'first_team_b', 'first_location','target']))
    test_x_2 = np.array(second_innings_test_df.drop(columns=['second_team_a', 'second_team_b', 'second_location','win','achieved']))

    test_y_1 = np.array(concatenated_test_df['target'])
    test_y_2 = np.array(concatenated_test_df['win'])
    test_y_3 = np.array(concatenated_test_df['achieved'])

    cols_1 = first_innings_train_df.drop(columns=['first_team_a', 'first_team_b', 'first_location','target']).columns
    cols_2 = second_innings_train_df.drop(columns=['second_team_a', 'second_team_b', 'second_location','win','achieved']).columns

    print("first innings columns-",len(cols_1) ,"\n",cols_1)
    print("second innings columns",len(cols_2) ,"\n",cols_2)


    pickle.dump(train_x_1,open(os.path.join(TRAIN_TEST_DIR,one_shot_multi_train_x_1),'wb'))
    pickle.dump(train_y_1, open(os.path.join(TRAIN_TEST_DIR,one_shot_multi_train_y_1), 'wb'))
    pickle.dump(train_x_2, open(os.path.join(TRAIN_TEST_DIR, one_shot_multi_train_x_2), 'wb'))
    pickle.dump(train_y_2, open(os.path.join(TRAIN_TEST_DIR, one_shot_multi_train_y_2), 'wb'))
    pickle.dump(train_y_3, open(os.path.join(TRAIN_TEST_DIR, one_shot_multi_train_y_3), 'wb'))
    pickle.dump(cols_1, open(os.path.join(outil.DEV_DIR, one_shot_multi_columns_1), 'wb'))
    pickle.dump(cols_2, open(os.path.join(outil.DEV_DIR, one_shot_multi_columns_2), 'wb'))

    pickle.dump(test_x_1, open(os.path.join(TRAIN_TEST_DIR, one_shot_multi_test_x_1), 'wb'))
    pickle.dump(test_y_1, open(os.path.join(TRAIN_TEST_DIR, one_shot_multi_test_y_1), 'wb'))
    pickle.dump(test_x_2, open(os.path.join(TRAIN_TEST_DIR, one_shot_multi_test_x_2), 'wb'))
    pickle.dump(test_y_2, open(os.path.join(TRAIN_TEST_DIR, one_shot_multi_test_y_2), 'wb'))
    pickle.dump(test_y_3, open(os.path.join(TRAIN_TEST_DIR, one_shot_multi_test_y_3), 'wb'))



    outil.create_meta_info_entry('one_shot_multi_train_xy', train_start,
                                 str(cricutil.substract_day_as_datetime(test_start_dt, 1).date()),
                                 file_list=[one_shot_multi_train_x_1,
                                            one_shot_multi_train_x_2,
                                            one_shot_multi_train_y_1,
                                            one_shot_multi_train_y_2,
                                            one_shot_multi_train_y_3,
                                            one_shot_multi_columns_1,
                                            one_shot_multi_columns_2])



    outil.create_meta_info_entry('one_shot_Multi test_xy', str(test_start_dt.date()),
                                 str(test_end_dt.date()),
                                 file_list=[one_shot_multi_test_x_1,
                                            one_shot_multi_test_x_2,
                                            one_shot_multi_test_y_1,
                                            one_shot_multi_test_y_2,
                                            one_shot_multi_test_y_3,
                                            one_shot_multi_columns_1,
                                            one_shot_multi_columns_2])

    print("train size ",train_x_1.shape,train_x_2.shape)
    print("test size ", test_x_1.shape,test_x_2.shape)



def create_mg_classification_train_test(train_start,test_start,test_end=None):
    if not os.path.isdir(TRAIN_TEST_DIR):
        os.makedirs(TRAIN_TEST_DIR)

    outil.use_model_from('dev')
    train_start_dt = cricutil.str_to_date_time(train_start)
    test_start_dt = cricutil.str_to_date_time(test_start)
    if test_end is None:
        test_end_dt = cricutil.today_as_date_time()
    else:
        test_end_dt = cricutil.str_to_date_time(test_end)

    overall_start = train_start_dt
    overall_end = test_end_dt
    match_list_df = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION + os.sep + 'cricinfo_match_list.csv')
    match_list_df = match_list_df[(match_list_df['date'] >= overall_start) & \
                                  (match_list_df['date'] <= overall_end)]
    #match_list_df = match_list_df[match_list_df['is_dl'] == 0]
    batting_list_df = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION + os.sep + 'cricinfo_batting.csv')
    # batting_list_df = batting_list_df[(batting_list_df['date'] >= overall_start) & \
    #                               (batting_list_df['date'] <= overall_end)]
    bowling_list_df = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION + os.sep + 'cricinfo_bowling.csv')
    # bowling_list_df = bowling_list_df[(bowling_list_df['date'] >= overall_start) & \
    #                                   (bowling_list_df['date'] <= overall_end)]

    loc_df = pd.read_excel(dl.CSV_LOAD_LOCATION + os.sep + 'location.xlsx')



    match_id_list = list(match_list_df['match_id'].unique())
    feature_list_train = []
    win_list_train = []

    feature_list_test =[]
    win_list_test = []
    #no_of_basman = 0
    matches_skipped = 0
    print("no of iters - ",len(match_id_list))
    for index,match_id in tqdm(enumerate(match_id_list)):


        team_a = match_list_df[match_list_df['match_id']==match_id].iloc[0]["first_innings"]
        team_b = match_list_df[match_list_df['match_id']==match_id].iloc[0]["second_innings"]
        location = match_list_df[match_list_df['match_id']==match_id].iloc[0]["location"]
        ref_dt_np = match_list_df[match_list_df['match_id']==match_id].iloc[0]["date"]
        ref_date = cricutil.pandas_timestamp_to_datetime(ref_dt_np)
        #ref_date = cricutil.npdate_to_datetime(ref_dt_np)
        #runs_scored = match_list_df[match_list_df['match_id']==match_id].iloc[0]['first_innings_run']
        loc_country = loc_df[loc_df['locations'] == location]['Country'].values[0].strip()
        toss_winner = match_list_df[match_list_df['match_id']==match_id].iloc[0]['toss_winner']

        team_a_loc = 0
        team_b_loc = 0
        other_loc = 0

        if loc_country ==team_a:
            team_a_loc = 1
        elif loc_country ==team_b:
            team_b_loc = 1
        else:
            other_loc = 1

        if toss_winner == team_a:
            toss = 1
        else:
            toss = 0



        winner = match_list_df[match_list_df['match_id']==match_id].iloc[0]["winner"]
        runs_achieved = match_list_df[match_list_df['match_id'] == match_id].iloc[0]['second_innings_run']
        if winner == team_b:
            runs_achieved = runs_achieved+15

        team_a_win = (team_a==winner)*1

        team_a_player_list_df = batting_list_df[(batting_list_df['match_id']==match_id) & (batting_list_df['team']==team_a)]

        team_a_player_list_df = team_a_player_list_df[['team', 'name', 'position']]

        team_b_player_list_df = batting_list_df[
            (batting_list_df['match_id'] == match_id) & (batting_list_df['team'] == team_b)]

        team_b_player_list_df = team_b_player_list_df[['team', 'name', 'position']]

        # team_a_bowler_list = bowling_list_df[
        #     (bowling_list_df['match_id'] == match_id) & (bowling_list_df['team'] == team_a)]
        #
        # team_a_bowler_list = team_a_bowler_list[['team', 'name']]


        # team_b_bowler_list = bowling_list_df[(bowling_list_df['match_id'] == match_id) & (bowling_list_df['team'] == team_b)]
        #
        # team_b_bowler_list = team_b_bowler_list[['team', 'name']]

        try:


            strength_a_b = fec.get_strength_ab_mg(team_a_player_list_df, team_b_player_list_df,
                               batsman_master_df=batting_list_df, bowler_master_df=bowling_list_df, ref_date=ref_date)


            feature_dict = {
                "strength":strength_a_b,
                "toss":toss,
                "team_a_loc":team_a_loc,
                "team_b_loc":team_b_loc,
                "other_loc":other_loc

            }
            if ref_date<test_start_dt:
                feature_list_train.append(feature_dict)
                win_list_train.append(team_a_win)

            else:
                feature_list_test.append(feature_dict)
                win_list_test.append(team_a_win)
        except Exception as ex:
            matches_skipped = matches_skipped + 1
            print(ex, ' for ',team_a, team_b, location, ' on ',ref_date.date(),' skipped so far ',matches_skipped)

            #raise ex


    train_df = pd.DataFrame(feature_list_train)
    test_df = pd.DataFrame(feature_list_test)

    train_x = np.array(train_df)
    test_x = np.array(test_df)

    train_y =  np.array(win_list_train)
    test_y = np.array(win_list_test)



    pickle.dump(train_x,open(os.path.join(TRAIN_TEST_DIR,mg_train_x),'wb'))
    pickle.dump(train_y, open(os.path.join(TRAIN_TEST_DIR,mg_train_y), 'wb'))

    pickle.dump(test_x, open(os.path.join(TRAIN_TEST_DIR, mg_test_x), 'wb'))
    pickle.dump(test_y, open(os.path.join(TRAIN_TEST_DIR, mg_test_y), 'wb'))


    outil.create_meta_info_entry('mg_classification_train_xy', train_start,
                                 str(cricutil.substract_day_as_datetime(test_start_dt, 1).date()),
                                 file_list=[mg_train_x,mg_train_y])



    outil.create_meta_info_entry('mg_classification test_xy', str(test_start_dt.date()),
                                 str(test_end_dt.date()),
                                 file_list=[mg_test_x,mg_test_y])

    print("train size ",train_x.shape)
    print("test size ", test_x.shape)


def update_first_innings_location_score(first_innings_df):

    country_map = pickle.load(open(outil.DEV_DIR + os.sep + outil.COUNTRY_ENCODING_MAP, 'rb'))
    loc_map = pickle.load(open(outil.DEV_DIR + os.sep + outil.LOC_ENCODING_MAP, 'rb'))

    runs_model = outil.load_keras_model(os.path.join(outil.DEV_DIR, outil.TEAM_OPPONENT_LOCATION_EMBEDDING_RUN_MODEL))

    team_oh_list = []
    opponent_oh_list = []
    location_oh_list = []
    default_loc_oh = loc_map[default_location]
    for index in range(first_innings_df.shape[0]):
        team = first_innings_df.iloc[index]['first_team_a']
        opponent = first_innings_df.iloc[index]['first_team_b']
        location = first_innings_df.iloc[index]['first_location']

        team_oh = country_map[team]
        opponent_oh = country_map[opponent]
        if location in loc_map:
            location_oh = loc_map[location]
        else:
            location_oh = default_loc_oh

        team_oh_list.append(team_oh)
        opponent_oh_list.append(opponent_oh)
        location_oh_list.append(location_oh)

    team_oh_mat = np.stack(team_oh_list)
    opponent_oh_mat = np.stack(opponent_oh_list)
    location_oh_mat = np.stack(location_oh_list)

    prediction = runs_model.predict([team_oh_mat,opponent_oh_mat,location_oh_mat])
    first_innings_df["location_embedding_prediction"]=prediction

    return first_innings_df

def update_second_innings_location_score(second_innings_df):

    country_map = pickle.load(open(outil.DEV_DIR + os.sep + outil.COUNTRY_ENCODING_MAP, 'rb'))
    loc_map = pickle.load(open(outil.DEV_DIR + os.sep + outil.LOC_ENCODING_MAP, 'rb'))

    runs_model = outil.load_keras_model(os.path.join(outil.DEV_DIR, outil.TEAM_OPPONENT_LOCATION_EMBEDDING_RUN_MODEL_2ND))

    team_oh_list = []
    opponent_oh_list = []
    location_oh_list = []
    default_loc_oh = loc_map[default_location]
    for index in range(second_innings_df.shape[0]):
        team = second_innings_df.iloc[index]['second_team_b']
        opponent = second_innings_df.iloc[index]['second_team_a']
        location = second_innings_df.iloc[index]['second_location']

        team_oh = country_map[team]
        opponent_oh = country_map[opponent]
        if location in loc_map:
            location_oh = loc_map[location]
        else:
            location_oh = default_loc_oh

        team_oh_list.append(team_oh)
        opponent_oh_list.append(opponent_oh)
        location_oh_list.append(location_oh)

    team_oh_mat = np.stack(team_oh_list)
    opponent_oh_mat = np.stack(opponent_oh_list)
    location_oh_mat = np.stack(location_oh_list)

    prediction = runs_model.predict([team_oh_mat, opponent_oh_mat, location_oh_mat])
    second_innings_df["location_embedding_prediction"] = prediction

    return second_innings_df


def add_prefix_to_dict(source,prefix):
    modified_source = {}
    for key in source.keys():
        modified_source[prefix+key]=source[key]

    return modified_source

def create_second_level_any_innings_non_neural_train_test(train_start,test_start,test_end=None,embedding=False):

    if not os.path.isdir(TRAIN_TEST_DIR):
        os.makedirs(TRAIN_TEST_DIR)

    outil.use_model_from('dev')
    train_start_dt = cricutil.str_to_date_time(train_start)
    test_start_dt = cricutil.str_to_date_time(test_start)
    if test_end is None:
        test_end_dt = cricutil.today_as_date_time()
    else:
        test_end_dt = cricutil.str_to_date_time(test_end)

    overall_start = train_start_dt
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



    first_innings_model = pickle.load(open(os.path.join(outil.DEV_DIR, outil.FIRST_INNINGS_MODEL_BASE), 'rb'))
    second_innings_model = pickle.load(open(os.path.join(outil.DEV_DIR, outil.SECOND_INNINGS_MODEL_BASE), 'rb'))

    match_id_list = list(match_list_df['match_id'].unique())
    feature_list_train = []
    win_list_train = []

    feature_list_test =[]
    win_list_test = []
    #no_of_basman = 0
    for index,match_id in tqdm(enumerate(match_id_list)):


        team_a = match_list_df[match_list_df['match_id']==match_id].iloc[0]["first_innings"]
        team_b = match_list_df[match_list_df['match_id']==match_id].iloc[0]["second_innings"]
        location = match_list_df[match_list_df['match_id']==match_id].iloc[0]["location"]
        ref_dt_np = match_list_df[match_list_df['match_id']==match_id].iloc[0]["date"]
        ref_date = cricutil.pandas_timestamp_to_datetime(ref_dt_np)

        winner = match_list_df[match_list_df['match_id']==match_id].iloc[0]["winner"]

        team_a_win = (team_a==winner)*1

        team_a_player_list_df = batting_list_df[(batting_list_df['match_id']==match_id) & (batting_list_df['team']==team_a)]

        team_a_player_list_df = team_a_player_list_df[['team', 'name', 'position']]

        team_b_player_list_df = batting_list_df[
            (batting_list_df['match_id'] == match_id) & (batting_list_df['team'] == team_b)]

        team_b_player_list_df = team_b_player_list_df[['team', 'name', 'position']]

        team_a_bowler_list = bowling_list_df[
            (bowling_list_df['match_id'] == match_id) & (bowling_list_df['team'] == team_a)]

        team_a_bowler_list = team_a_bowler_list[['team', 'name']]


        team_b_bowler_list = bowling_list_df[(bowling_list_df['match_id'] == match_id) & (bowling_list_df['team'] == team_b)]

        team_b_bowler_list = team_b_bowler_list[['team', 'name']]

        try:

            feature_dict_team_a_batting = fec.get_instance_feature_dict(team_a, team_b, location, team_a_player_list_df,
                                                                      team_b_bowler_list, ref_date=ref_date,
                                                                      innings_type='first')

            feature_vec_team_a_first_batting = np.array(pd.DataFrame([feature_dict_team_a_batting]).drop(columns=['team','opponent','location']))
            team_a_first_target = first_innings_model.predict(feature_vec_team_a_first_batting)[0]
            feature_dict_team_b_batting = fec.get_instance_feature_dict(team_b, team_a, location, team_b_player_list_df,
                                                                       team_a_bowler_list, ref_date=ref_date,
                                                                       innings_type='second')
            feature_vec_team_b_first_batting = np.array(pd.DataFrame([feature_dict_team_b_batting]).drop(columns=['team','opponent','location']))
            team_b_first_target = first_innings_model.predict(feature_vec_team_b_first_batting)[0]

            #print("=====",team_a_first_target,team_b_first_target)
            feature_dict_team_b_batting['target_score'] = team_a_first_target
            feature_dict_team_a_batting['target_score'] = team_b_first_target

            feature_vector_team_a_chasing = np.array(pd.DataFrame([feature_dict_team_a_batting]).drop(columns=['team','opponent','location']))
            feature_vector_team_b_chasing = np.array(pd.DataFrame([feature_dict_team_b_batting]).drop(columns=['team', 'opponent', 'location']))

            team_a_chasing_success = second_innings_model.predict_proba(feature_vector_team_a_chasing)[0][0]
            team_b_chasing_success = second_innings_model.predict_proba(feature_vector_team_b_chasing)[0][0]

            #print("=====", team_a_chasing_success, team_b_chasing_success)
            combined_feature_vector = [team_a_first_target,team_b_chasing_success,team_b_first_target,team_a_chasing_success]

            if ref_date<test_start_dt:
                feature_list_train.append(combined_feature_vector)
                win_list_train.append(team_a_win)
            else:
                feature_list_test.append(combined_feature_vector)
                win_list_test.append(team_a_win)

        except Exception as ex:
            print(ex, ' for ',team_a, team_b, location, ' on ',ref_date.date() )

            #raise ex

    train_x = np.array(feature_list_train)
    train_y = np.array(win_list_train)

    test_x = np.array(feature_list_test)
    test_y = np.array(win_list_test)


    pickle.dump(train_x,open(os.path.join(TRAIN_TEST_DIR,second_level_non_neural_train_x),'wb'))
    pickle.dump(train_y, open(os.path.join(TRAIN_TEST_DIR, second_level_non_neural_train_y), 'wb'))
    pickle.dump(test_x, open(os.path.join(TRAIN_TEST_DIR, second_level_non_neural_test_x), 'wb'))
    pickle.dump(test_y, open(os.path.join(TRAIN_TEST_DIR, second_level_non_neural_test_y), 'wb'))


    outil.create_meta_info_entry('second_level_non_neural_train_xy', train_start,
                                 str(cricutil.substract_day_as_datetime(test_start_dt, 1).date()),
                                 file_list=[second_level_non_neural_train_x,
                                            second_level_non_neural_train_y])



    outil.create_meta_info_entry('second_level_non_neural_test_xy', str(test_start_dt.date()),
                                 str(test_end_dt.date()),
                                 file_list=[second_level_non_neural_test_x,
                                            second_level_non_neural_test_y])

    print("train size ",train_x.shape)
    print("test size ", test_x.shape)

def create_second_level_any_innings_train_test(train_start,test_start,test_end=None,embedding=False):

    if not os.path.isdir(TRAIN_TEST_DIR):
        os.makedirs(TRAIN_TEST_DIR)

    outil.use_model_from('dev')
    train_start_dt = cricutil.str_to_date_time(train_start)
    test_start_dt = cricutil.str_to_date_time(test_start)
    if test_end is None:
        test_end_dt = cricutil.today_as_date_time()
    else:
        test_end_dt = cricutil.str_to_date_time(test_end)

    overall_start = train_start_dt
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
    feature_list_team_a_first_train = []
    feature_list_team_b_second_train = []
    feature_list_team_b_first_train = []
    feature_list_team_a_second_train = []
    # target_list_train = []
    # achieved_list_train = []
    win_list_train = []

    feature_list_team_a_first_test =[]
    feature_list_team_b_second_test = []
    feature_list_team_b_first_test = []
    feature_list_team_a_second_test = []
    # target_list_test = []
    # achieved_list_test = []
    win_list_test = []
    #no_of_basman = 0
    for index,match_id in tqdm(enumerate(match_id_list)):


        team_a = match_list_df[match_list_df['match_id']==match_id].iloc[0]["first_innings"]
        team_b = match_list_df[match_list_df['match_id']==match_id].iloc[0]["second_innings"]
        location = match_list_df[match_list_df['match_id']==match_id].iloc[0]["location"]
        ref_dt_np = match_list_df[match_list_df['match_id']==match_id].iloc[0]["date"]
        ref_date = cricutil.pandas_timestamp_to_datetime(ref_dt_np)
        runs_scored = match_list_df[match_list_df['match_id']==match_id].iloc[0]['first_innings_run']

        winner = match_list_df[match_list_df['match_id']==match_id].iloc[0]["winner"]
        runs_achieved = match_list_df[match_list_df['match_id'] == match_id].iloc[0]['second_innings_run']
        if winner == team_b:
            runs_achieved = runs_achieved+15

        team_a_win = (team_a==winner)*1

        team_a_player_list_df = batting_list_df[(batting_list_df['match_id']==match_id) & (batting_list_df['team']==team_a)]

        team_a_player_list_df = team_a_player_list_df[['team', 'name', 'position']]

        team_b_player_list_df = batting_list_df[
            (batting_list_df['match_id'] == match_id) & (batting_list_df['team'] == team_b)]

        team_b_player_list_df = team_b_player_list_df[['team', 'name', 'position']]

        team_a_bowler_list = bowling_list_df[
            (bowling_list_df['match_id'] == match_id) & (bowling_list_df['team'] == team_a)]

        team_a_bowler_list = team_a_bowler_list[['team', 'name']]


        team_b_bowler_list = bowling_list_df[(bowling_list_df['match_id'] == match_id) & (bowling_list_df['team'] == team_b)]

        team_b_bowler_list = team_b_bowler_list[['team', 'name']]

        try:

            feature_dict_first_innings_team_a,feature_dict_second_innings_team_b = fec.get_one_shot_multi_output_feature_dict(team_a, team_b, location,
                                                                                                 team_a_player_list_df,team_b_player_list_df,
                                                                                                 team_a_bowler_list,team_b_bowler_list,
                                                                                                 ref_date=ref_date,no_of_years=None,embedding=embedding)

            feature_dict_first_innings_team_b, feature_dict_second_innings_team_a = fec.get_one_shot_multi_output_feature_dict(team_b, team_a, location,
                                                                                                team_b_player_list_df, team_a_player_list_df,
                                                                                                team_b_bowler_list, team_a_bowler_list,
                                                                                                ref_date=ref_date, no_of_years=None, embedding=embedding)

            feature_dict_first_innings_team_b= add_prefix_to_dict(feature_dict_first_innings_team_b,'alt_')
            feature_dict_second_innings_team_a = add_prefix_to_dict(feature_dict_second_innings_team_a,'alt_')
            #print(feature_dict)
            #no_of_basman = no_of_basman+len(team_player_list)

            if ref_date<test_start_dt:
                feature_list_team_a_first_train.append(feature_dict_first_innings_team_a)
                feature_list_team_b_second_train.append(feature_dict_second_innings_team_b)

                feature_list_team_b_first_train.append(feature_dict_first_innings_team_b)
                feature_list_team_a_second_train.append(feature_dict_second_innings_team_a)
                win_list_train.append(team_a_win)


            else:
                feature_list_team_a_first_test.append(feature_dict_first_innings_team_a)
                feature_list_team_b_second_test.append(feature_dict_second_innings_team_b)

                feature_list_team_b_first_test.append(feature_dict_first_innings_team_b)
                feature_list_team_a_second_test.append(feature_dict_second_innings_team_a)

                win_list_test.append(team_a_win)

        except Exception as ex:
            print(ex, ' for ',team_a, team_b, location, ' on ',ref_date.date() )

            #raise ex


    first_innings_team_a_train_df = pd.DataFrame(feature_list_team_a_first_train)
    second_innings_team_b_train_df =  pd.DataFrame(feature_list_team_b_second_train)
    first_innings_team_b_train_df = pd.DataFrame(feature_list_team_b_first_train)
    second_innings_team_a_train_df = pd.DataFrame(feature_list_team_a_second_train)


    first_innings_team_a_test_df = pd.DataFrame(feature_list_team_a_first_test)
    second_innings_team_b_test_df = pd.DataFrame(feature_list_team_b_second_test)
    first_innings_team_b_test_df = pd.DataFrame(feature_list_team_b_first_test)
    second_innings_team_a_test_df = pd.DataFrame(feature_list_team_a_second_test)


    concatenated_train_df = pd.concat([first_innings_team_a_train_df,second_innings_team_b_train_df,
                                       first_innings_team_b_train_df,second_innings_team_a_train_df],
                                      axis=1)
    concatenated_train_df['win']=win_list_train
    concatenated_test_df = pd.concat([first_innings_team_a_test_df, second_innings_team_b_test_df,
                                      first_innings_team_b_test_df,second_innings_team_a_test_df],
                                     axis=1)
    concatenated_test_df['win']=win_list_test

    concatenated_train_df.dropna(inplace=True)
    concatenated_test_df.dropna(inplace=True)

    #******* to be removed*******#
    #pickle.dump(concatenated_train_df,open('concatenated_train_df.pkl','wb'))
    #pickle.dump(concatenated_test_df, open('concatenated_test_df.pkl', 'wb'))
    #******************#

    cols_1 = pickle.load(open(os.path.join(outil.DEV_DIR, one_shot_multi_columns_1), 'rb'))
    cols_2 = pickle.load(open(os.path.join(outil.DEV_DIR, one_shot_multi_columns_2), 'rb'))

    alt_cols_1 = []
    for col in cols_1:
        alt_cols_1.append('alt_' + col)

    alt_cols_2 = []
    for col in cols_2:
        alt_cols_2.append('alt_' + col)

    first_innings_team_a_train_df = concatenated_train_df[list(cols_1)]
    second_innings_team_b_train_df = concatenated_train_df[list(cols_2)]
    first_innings_team_b_train_df = concatenated_train_df[alt_cols_1]
    second_innings_team_a_train_df = concatenated_train_df[alt_cols_2]

    win_list_train = list(concatenated_train_df['win'])

    first_innings_team_a_test_df = concatenated_test_df[list(cols_1)]
    second_innings_team_b_test_df = concatenated_test_df[list(cols_2)]
    first_innings_team_b_test_df = concatenated_test_df[alt_cols_1]
    second_innings_team_a_test_df = concatenated_test_df[alt_cols_2]

    win_list_test = list(concatenated_test_df['win'])

    train_x_1 = np.array(first_innings_team_a_train_df)
    train_x_2 = np.array(second_innings_team_b_train_df)

    train_x_1_alt = np.array(first_innings_team_b_train_df)
    train_x_2_alt = np.array(second_innings_team_a_train_df)

    train_y = np.array(concatenated_train_df['win'])

    test_x_1 = np.array(first_innings_team_a_test_df)
    test_x_2 = np.array(second_innings_team_b_test_df)
    test_x_1_alt = np.array(first_innings_team_b_test_df)
    test_x_2_alt = np.array(second_innings_team_a_test_df)

    test_y = np.array(concatenated_test_df['win'])

    x1_scaler = pickle.load(open(os.path.join(outil.DEV_DIR, outil.ONE_SHOT_MULTI_SCALER_X1), "rb"))
    x2_scaler = pickle.load(open(os.path.join(outil.DEV_DIR, outil.ONE_SHOT_MULTI_SCALER_X2), "rb"))

    train_x_1 = x1_scaler.transform(train_x_1)
    train_x_2 = x2_scaler.transform(train_x_2)
    train_x_1_alt = x1_scaler.transform(train_x_1_alt)
    train_x_2_alt = x2_scaler.transform(train_x_2_alt)

    test_x_1 = x1_scaler.transform(test_x_1)
    test_x_2 = x2_scaler.transform(test_x_2)
    test_x_1_alt = x1_scaler.transform(test_x_1_alt)
    test_x_2_alt = x2_scaler.transform(test_x_2_alt)

    combined_model = outil.load_keras_model(os.path.join(outil.DEV_DIR, outil.ONE_SHOT_MULTI_NEURAL))

    predict_train_1, predict_train_2, predict_train_3 = combined_model.predict([train_x_1, train_x_2])
    predict_train_alt_1, predict_train_alt_2, predict_train_alt_3 = combined_model.predict([train_x_1_alt, train_x_2_alt])

    predict_test_1, predict_test_2, predict_test_3 = combined_model.predict([test_x_1, test_x_2])
    predict_test_alt_1, predict_test_alt_2, predict_test_alt_3 = combined_model.predict([test_x_1_alt, test_x_2_alt])

    train_x = np.concatenate([predict_train_1, predict_train_2, predict_train_3, predict_train_alt_1, predict_train_alt_2, predict_train_alt_3], axis=1)
    test_x = np.concatenate([predict_test_1, predict_test_2, predict_test_3,predict_test_alt_1, predict_test_alt_2, predict_test_alt_3], axis=1)



    pickle.dump(train_x,open(os.path.join(TRAIN_TEST_DIR,second_level_any_inst_train_x),'wb'))
    pickle.dump(train_y, open(os.path.join(TRAIN_TEST_DIR, second_level_any_inst_train_y), 'wb'))
    pickle.dump(test_x, open(os.path.join(TRAIN_TEST_DIR, second_level_any_inst_test_x), 'wb'))
    pickle.dump(test_y, open(os.path.join(TRAIN_TEST_DIR, second_level_any_inst_test_y), 'wb'))


    outil.create_meta_info_entry('second_level_any_inst_train_xy', train_start,
                                 str(cricutil.substract_day_as_datetime(test_start_dt, 1).date()),
                                 file_list=[second_level_any_inst_train_x,
                                            second_level_any_inst_train_y])



    outil.create_meta_info_entry('second_level_any_inst_test_xy', str(test_start_dt.date()),
                                 str(test_end_dt.date()),
                                 file_list=[second_level_any_inst_test_x,
                                            second_level_any_inst_test_y])

    print("train size ",train_x.shape)
    print("test size ", test_x.shape)





def create_first_innings_base_train_test(train_start,test_start,test_end=None):
    if not os.path.isdir(TRAIN_TEST_DIR):
        os.makedirs(TRAIN_TEST_DIR)

    outil.use_model_from('dev')
    train_start_dt = cricutil.str_to_date_time(train_start)
    test_start_dt = cricutil.str_to_date_time(test_start)
    if test_end is None:
        test_end_dt = cricutil.today_as_date_time()
    else:
        test_end_dt = cricutil.str_to_date_time(test_end)

    overall_start = train_start_dt
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
    feature_list_train = []
    target_list_train = []

    feature_list_test =[]
    target_list_test = []
    #no_of_basman = 0
    for index,match_id in tqdm(enumerate(match_id_list)):


        team = match_list_df[match_list_df['match_id']==match_id].iloc[0]["first_innings"]
        opponent = match_list_df[match_list_df['match_id']==match_id].iloc[0]["second_innings"]
        location = match_list_df[match_list_df['match_id']==match_id].iloc[0]["location"]
        ref_dt_np = match_list_df[match_list_df['match_id']==match_id].iloc[0]["date"]
        ref_date = cricutil.pandas_timestamp_to_datetime(ref_dt_np)
        runs_scored = match_list_df[match_list_df['match_id']==match_id].iloc[0]['first_innings_run']

        team_player_list_df = batting_list_df[(batting_list_df['match_id']==match_id) & (batting_list_df['team']==team)]

        team_player_list_df = team_player_list_df[['team', 'name', 'position']]

        opponent_bowler_list_df = bowling_list_df[(bowling_list_df['match_id'] == match_id) & (bowling_list_df['team'] == opponent)]

        opponent_bowler_list_df = opponent_bowler_list_df[['team', 'name']]

        try:

            feature_dict = fec.get_instance_feature_dict(team, opponent, location,team_player_list_df,
                                                                opponent_bowler_list_df,ref_date=ref_date,innings_type='first')
            #print(feature_dict)
            #no_of_basman = no_of_basman+len(team_player_list)

            if ref_date<test_start_dt:
                feature_list_train.append(feature_dict)
                target_list_train.append(runs_scored)
            else:
                feature_list_test.append(feature_dict)
                target_list_test.append(runs_scored)
        except Exception as ex:
            print(ex, ' for ',team, opponent, location, ' on ',ref_date.date() )
            #raise ex

    train_y = np.stack(target_list_train)
    test_y = np.stack(target_list_test)

    #print(pd.DataFrame(feature_list_train))
    train_x = np.array(pd.DataFrame(feature_list_train).drop(columns=['team','opponent','location']))
    #print(train_x)
    test_x  = np.array(pd.DataFrame(feature_list_test).drop(columns=['team','opponent','location']))
    cols = list(pd.DataFrame(feature_list_train).drop(columns=['team','opponent','location']).columns)

    pickle.dump(train_x,open(os.path.join(TRAIN_TEST_DIR,first_innings_base_train_x),'wb'))
    pickle.dump(train_y, open(os.path.join(TRAIN_TEST_DIR,first_innings_base_train_y), 'wb'))
    pickle.dump(cols, open(os.path.join(outil.DEV_DIR, first_innings_base_columns), 'wb'))

    outil.create_meta_info_entry('first_innings_base_train_xy', train_start,
                                 str(cricutil.substract_day_as_datetime(test_start_dt, 1).date()),
                                 file_list=[first_innings_base_train_x,
                                            first_innings_base_train_y,
                                            first_innings_base_columns])

    pickle.dump(test_x, open(os.path.join(TRAIN_TEST_DIR, first_innings_base_test_x), 'wb'))
    pickle.dump(test_y, open(os.path.join(TRAIN_TEST_DIR, first_innings_base_test_y), 'wb'))

    outil.create_meta_info_entry('first_innings_base_test_xy', str(test_start_dt.date()),
                                 str(test_end_dt.date()),
                                 file_list=[first_innings_base_test_x,
                                            first_innings_base_test_y])

    # trend prediction test data
    trend_data_df = pd.DataFrame(feature_list_test)
    trend_data_df['runs_scored'] = target_list_test
    trend_data_df[['team','opponent','location','opponent_trend_predict','location_trend_predict','current_trend_predict','runs_scored']].to_csv(os.path.join(outil.DEV_DIR, "trend_predict.csv"), index=False)
    # trend_data_df = trend_data_df[['opponent_trend_predict','location_trend_predict','current_trend_predict','runs_scored']]

    mape_opponent_trend = cric_eval.mape(np.array(trend_data_df['runs_scored']),
                                         np.array(trend_data_df['opponent_trend_predict']))

    mape_location_trend = cric_eval.mape(np.array(trend_data_df['runs_scored']),
                                    np.array(trend_data_df['location_trend_predict']))

    mape_current_trend = cric_eval.mape(np.array(trend_data_df['runs_scored']),
                                         np.array(trend_data_df['current_trend_predict']))

    mape_opponent_mean = cric_eval.mape(np.array(trend_data_df['runs_scored']),
                                         np.array(trend_data_df['opponent_mean']))

    mape_location_mean = cric_eval.mape(np.array(trend_data_df['runs_scored']),
                                         np.array(trend_data_df['location_mean']))

    mape_current_mean = cric_eval.mape(np.array(trend_data_df['runs_scored']),
                                        np.array(trend_data_df['current_mean']))


    # trend prediction train data

    trend_data_train_df = pd.DataFrame(feature_list_train)
    trend_data_train_df['runs_scored'] = target_list_train
    trend_data_train_df[
        ['team', 'opponent', 'location', 'opponent_trend_predict', 'location_trend_predict', 'current_trend_predict',
         'runs_scored']].to_csv(os.path.join(outil.DEV_DIR, "trend_predict_train.csv"), index=False)
    # trend_data_train_df = trend_data_train_df[['opponent_trend_predict','location_trend_predict','current_trend_predict','runs_scored']]

    mape_train_opponent_trend = cric_eval.mape(np.array(trend_data_train_df['runs_scored']),
                                         np.array(trend_data_train_df['opponent_trend_predict']))

    mape_train_location_trend = cric_eval.mape(np.array(trend_data_train_df['runs_scored']),
                                         np.array(trend_data_train_df['location_trend_predict']))

    mape_train_current_trend = cric_eval.mape(np.array(trend_data_train_df['runs_scored']),
                                        np.array(trend_data_train_df['current_trend_predict']))

    mape_train_opponent_mean = cric_eval.mape(np.array(trend_data_train_df['runs_scored']),
                                        np.array(trend_data_train_df['opponent_mean']))

    mape_train_location_mean = cric_eval.mape(np.array(trend_data_train_df['runs_scored']),
                                        np.array(trend_data_train_df['location_mean']))

    mape_train_current_mean = cric_eval.mape(np.array(trend_data_train_df['runs_scored']),
                                       np.array(trend_data_train_df['current_mean']))

    outil.create_model_meta_info_entry('first_innings_trend_prediction_metrics',
                                       (mape_train_opponent_trend, mape_train_location_trend, mape_train_current_trend,
                                        mape_train_opponent_mean, mape_train_location_mean, mape_train_current_mean),
                                       (mape_opponent_trend, mape_location_trend, mape_current_trend,
                                        mape_opponent_mean, mape_location_mean, mape_current_mean),
                                       info="metrics is mape_opponent_trend, mape_location_trend, mape_current_trend,"+
                                            "mape_opponent_mean, mape_location_mean, mape_current_mean ",
                                       file_list=[
                                           "tred_predict.csv",
                                       ]
                                       )
    print("mape_train_opponent_trend", mape_train_opponent_trend)
    print("mape_train_location_trend", mape_train_location_trend)
    print("mape_train_current_trend", mape_train_current_trend)

    print("mape_train_opponent_mean", mape_train_opponent_mean)
    print("mape_train_location_mean", mape_train_location_mean)
    print("mape_train_current_mean", mape_train_current_mean)

    print("mape_opponent_trend",mape_opponent_trend)
    print("mape_location_trend", mape_location_trend)
    print("mape_current_trend", mape_current_trend)

    print("mape_opponent_mean", mape_opponent_mean)
    print("mape_location_mean", mape_location_mean)
    print("mape_current_mean", mape_current_mean)


def create_second_innings_base_train_test(train_start,test_start,test_end=None):
    if not os.path.isdir(TRAIN_TEST_DIR):
        os.makedirs(TRAIN_TEST_DIR)

    outil.use_model_from('dev')
    train_start_dt = cricutil.str_to_date_time(train_start)
    test_start_dt = cricutil.str_to_date_time(test_start)
    if test_end is None:
        test_end_dt = cricutil.today_as_date_time()
    else:
        test_end_dt = cricutil.str_to_date_time(test_end)

    overall_start = train_start_dt
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

    first_innings_model = pickle.load(open(os.path.join(outil.MODEL_DIR, outil.FIRST_INNINGS_MODEL_BASE), 'rb'))

    match_id_list = list(match_list_df['match_id'].unique())
    feature_list_train = []
    result_list_train = []
    runs_achieved_list_train = []

    feature_list_test = []
    result_list_test = []
    runs_achieved_list_test = []
    for index,match_id in tqdm(enumerate(match_id_list)):
        team = match_list_df[match_list_df['match_id'] == match_id].iloc[0]["second_innings"]
        opponent = match_list_df[match_list_df['match_id'] == match_id].iloc[0]["first_innings"]
        winner = match_list_df[match_list_df['match_id'] == match_id].iloc[0]["winner"]
        location = match_list_df[match_list_df['match_id'] == match_id].iloc[0]["location"]
        ref_dt_np = match_list_df[match_list_df['match_id'] == match_id].iloc[0]["date"]
        ref_date = cricutil.pandas_timestamp_to_datetime(ref_dt_np)
        target = match_list_df[match_list_df['match_id'] == match_id].iloc[0]['first_innings_run']
        runs_achieved = match_list_df[match_list_df['match_id'] == match_id].iloc[0]['second_innings_run']
        if winner == team:
            #runs_achieved = runs_achieved+15
            #adjust runcs achieved
            chasing_overs = round(match_list_df[match_list_df['match_id']==match_id].iloc[0]['chasing_overs'])
            if chasing_overs == 0:
                runs_achieved = runs_achieved + 40
            elif chasing_overs >= 49:
                runs_achieved = runs_achieved + 15
            else:
                runs_achieved=(runs_achieved/chasing_overs)*50

        win = 1*(team==winner)

        team_player_list_df = batting_list_df[
            (batting_list_df['match_id'] == match_id) & (batting_list_df['team'] == team)]

        team_player_list_df = team_player_list_df[['team', 'name', 'position']]

        opponent_player_list_df = batting_list_df[
            (batting_list_df['match_id'] == match_id) & (batting_list_df['team'] == opponent)]

        opponent_player_list_df = opponent_player_list_df[['team', 'name', 'position']]

        team_bowler_list_df = bowling_list_df[
            (bowling_list_df['match_id'] == match_id) & (bowling_list_df['team'] == team)]

        team_bowler_list_df = team_bowler_list_df[['team', 'name']]

        opponent_bowler_list_df = bowling_list_df[
            (bowling_list_df['match_id'] == match_id) & (bowling_list_df['team'] == opponent)]

        opponent_bowler_list_df = opponent_bowler_list_df[['team', 'name']]


        try:

            # get predicted first innings score



            feature_dict = fec.get_instance_feature_dict(team, opponent, location,team_player_list_df,
                                                                opponent_bowler_list_df,ref_date=ref_date,innings_type='second',target=target)

            feature_dict['target_score'] = target

            if ref_date<test_start_dt:
                feature_list_train.append(feature_dict)
                result_list_train.append(win)
                runs_achieved_list_train.append(runs_achieved)
            else:
                feature_dict_first = fec.get_first_innings_feature_vector(opponent, team, location,
                                                                          opponent_player_list_df, team_bowler_list_df,
                                                                          ref_date=ref_date)

                predicted_target = first_innings_model.predict(feature_dict_first.reshape(1, -1))[0]

                feature_dict['target_score'] = predicted_target
                feature_list_test.append(feature_dict)
                result_list_test.append(win)
                runs_achieved_list_test.append(runs_achieved)
        except Exception as ex:
            print(ex, ' for ',team, opponent, location, ' on ',ref_date.date() )
            #raise ex

    train_y = np.stack(result_list_train)
    train_y_2 = np.stack(runs_achieved_list_train)
    test_y = np.stack(result_list_test)
    test_y_2 = np.stack(runs_achieved_list_test)

    train_x = np.array(pd.DataFrame(feature_list_train).drop(columns=['team','opponent','location']))
    test_x  = np.array(pd.DataFrame(feature_list_test).drop(columns=['team','opponent','location']))
    cols = list(pd.DataFrame(feature_list_train).drop(columns=['team','opponent','location']).columns)

    pickle.dump(train_x,open(os.path.join(TRAIN_TEST_DIR,second_innings_base_train_x),'wb'))
    pickle.dump(train_y, open(os.path.join(TRAIN_TEST_DIR,second_innings_base_train_y), 'wb'))
    pickle.dump(train_y_2, open(os.path.join(TRAIN_TEST_DIR, second_innings_base_train_y_2), 'wb'))
    pickle.dump(cols, open(os.path.join(outil.DEV_DIR, second_innings_base_columns), 'wb'))

    outil.create_meta_info_entry('second_innings_base_train_xy', train_start,
                                 str(cricutil.substract_day_as_datetime(test_start_dt, 1).date()),
                                 file_list=[second_innings_base_train_x,
                                            second_innings_base_train_y,
                                            second_innings_base_train_y_2,
                                            second_innings_base_columns])

    pickle.dump(test_x, open(os.path.join(TRAIN_TEST_DIR, second_innings_base_test_x), 'wb'))
    pickle.dump(test_y, open(os.path.join(TRAIN_TEST_DIR, second_innings_base_test_y), 'wb'))
    pickle.dump(test_y_2, open(os.path.join(TRAIN_TEST_DIR, second_innings_base_test_y_2), 'wb'))

    outil.create_meta_info_entry('second_innings_base_test_xy', str(test_start_dt.date()),
                                 str(test_end_dt.date()),
                                 file_list=[second_innings_base_test_x,
                                            second_innings_base_test_y,
                                            second_innings_base_test_y_2])

    # print(pd.DataFrame(feature_list_train))

def verify_threshod(test_start,test_end=None):
    if not os.path.isdir(TRAIN_TEST_DIR):
        os.makedirs(TRAIN_TEST_DIR)

    outil.use_model_from('dev')
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


def create_first_innings_train_test(train_start,test_start,test_end=None):
    if not os.path.isdir(TRAIN_TEST_DIR):
        os.makedirs(TRAIN_TEST_DIR)

    outil.use_model_from('dev')
    train_start_dt = cricutil.str_to_date_time(train_start)
    test_start_dt = cricutil.str_to_date_time(test_start)
    if test_end is None:
        test_end_dt = cricutil.today_as_date_time()
    else:
        test_end_dt = cricutil.str_to_date_time(test_end)

    overall_start = train_start_dt
    overall_end = test_end_dt
    match_list_df = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION + os.sep + 'match_list.csv')
    match_list_df = match_list_df[(match_list_df['date'] >= overall_start) & \
                                  (match_list_df['date'] <= overall_end)]
    match_stats_df = pd.read_csv(dl.CSV_LOAD_LOCATION + os.sep + 'match_stats.csv')
    match_list_df = match_list_df.merge(match_stats_df, how='inner', on='match_id')
    match_id_list = list(match_list_df['match_id'].unique())

    feature_list_train = []
    target_list_train = []

    feature_list_test =[]
    target_list_test = []

    for index,match_id in tqdm(enumerate(match_id_list)):
        match_info = match_list_df[match_list_df['match_id']==match_id]
        team_info = match_info[match_info['first_innings']==match_info['team_statistics']]
        opponent_info = match_info[match_info['second_innings']==match_info['team_statistics']]

        team = team_info['team_statistics'].values[0].strip()
        opponent = opponent_info['team_statistics'].values[0].strip()
        location = team_info['location'].values[0].strip()
        ref_dt_np = team_info['date'].values[0]
        ref_date = cricutil.npdate_to_datetime(ref_dt_np)
        runs_scored = team_info['total_run'].values[0]

        team_player_list = list()
        for bi in range(11):
            batsman = team_info['batsman_'+str(bi+1)].values[0].strip()
            if batsman == 'not_batted':
                break
            else:
                team_player_list.append(batsman)

        team_bowler_list = []
        for tbo in range(11):
            t_bowler = team_info['bowler_'+str(tbo+1)].values[0].strip()
            if t_bowler == 'not_bowled':
                break
            elif t_bowler not in team_player_list:
                team_bowler_list.append(t_bowler)
            else:
                pass

        # if len(team_player_list) + len(team_bowler_list) ==11:
        #     team_player_list = team_player_list+team_bowler_list
        #team_player_list = fe.complete_batting_order(team, team_player_list, team_bowler_list, ref_date=ref_date,no_of_batsman=7)

        opponent_player_list = list()
        for boi in range(11):
            bowler = opponent_info['bowler_' + str(boi + 1)].values[0].strip()
            if bowler == 'not_bowled':
                break
            else:
                opponent_player_list.append(bowler)

        try:
            feature_vector = fe.get_first_innings_feature_embedding_vector(team, opponent, location,
                                                                            team_player_list, opponent_player_list,
                                                                            ref_date=ref_date)

            if ref_date<test_start_dt:
                feature_list_train.append(feature_vector)
                target_list_train.append(runs_scored)
            else:
                feature_list_test.append(feature_vector)
                target_list_test.append(runs_scored)
        except Exception as ex:
            print(ex, ' for ',team, opponent, location, ' on ',ref_date.date() )

    #print("pre-scaled values \n",np.stack(feature_list_train))
    train_x = np.stack(feature_list_train)
    train_y = np.stack(target_list_train)

    test_x = np.stack(feature_list_test)
    test_y = np.stack(target_list_test)

    # pickle train_x, train_y,test_x,test_y,scaler
    pickle.dump(train_x,open(os.path.join(TRAIN_TEST_DIR,first_innings_train_x),'wb'))
    pickle.dump(train_y, open(os.path.join(TRAIN_TEST_DIR, first_innings_train_y), 'wb'))

    pickle.dump(test_x, open(os.path.join(TRAIN_TEST_DIR, first_innings_test_x), 'wb'))
    pickle.dump(test_y, open(os.path.join(TRAIN_TEST_DIR, first_innings_test_y), 'wb'))


    outil.create_meta_info_entry('first_innings_train_xy', train_start,
                                 str(cricutil.substract_day_as_datetime(test_start_dt, 1).date()),
                                 file_list=[first_innings_train_x,
                                            first_innings_train_y])

    outil.create_meta_info_entry('first_innings_test_xy', str(test_start_dt.date()),
                                 str(test_end_dt.date()),
                                 file_list=[first_innings_test_x,
                                            first_innings_test_y])


def create_second_innings_train_test(train_start,test_start,test_end=None):
    if not os.path.isdir(TRAIN_TEST_DIR):
        os.makedirs(TRAIN_TEST_DIR)

    outil.use_model_from('dev')
    train_start_dt = cricutil.str_to_date_time(train_start)
    test_start_dt = cricutil.str_to_date_time(test_start)
    if test_end is None:
        test_end_dt = cricutil.today_as_date_time()
    else:
        test_end_dt = cricutil.str_to_date_time(test_end)

    overall_start = train_start_dt
    overall_end = test_end_dt
    match_list_df = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION + os.sep + 'match_list.csv')
    match_list_df = match_list_df[(match_list_df['date'] >= overall_start) & \
                                  (match_list_df['date'] <= overall_end)]
    match_stats_df = pd.read_csv(dl.CSV_LOAD_LOCATION + os.sep + 'match_stats.csv')
    match_list_df = match_list_df.merge(match_stats_df, how='inner', on='match_id')
    match_id_list = list(match_list_df['match_id'].unique())

    feature_list_train = []
    result_list_train = []

    feature_list_test =[]
    result_list_test = []

    for index,match_id in tqdm(enumerate(match_id_list)):
        match_info = match_list_df[match_list_df['match_id']==match_id]
        team_info = match_info[match_info['second_innings']==match_info['team_statistics']]
        opponent_info = match_info[match_info['first_innings']==match_info['team_statistics']]

        team = team_info['team_statistics'].values[0].strip()
        opponent = opponent_info['team_statistics'].values[0].strip()
        location = team_info['location'].values[0].strip()
        ref_dt_np = team_info['date'].values[0]
        ref_date = cricutil.npdate_to_datetime(ref_dt_np)
        target = opponent_info['total_run'].values[0]
        win=0
        if team_info['winner'].values[0]==team:
            win=1


        team_player_list = list()
        for bi in range(11):
            batsman = team_info['batsman_'+str(bi+1)].values[0].strip()
            if batsman == 'not_batted':
                break
            else:
                team_player_list.append(batsman)

        team_bowler_list = []
        for tbo in range(11):
            t_bowler = team_info['bowler_' + str(tbo + 1)].values[0].strip()
            if t_bowler == 'not_bowled':
                break
            elif t_bowler not in team_player_list:
                team_bowler_list.append(t_bowler)
            else:
                pass

        # if len(team_player_list) + len(team_bowler_list) ==11:
        #     team_player_list = team_player_list+team_bowler_list
        # team_player_list = fe.complete_batting_order(team, team_player_list, team_bowler_list,
        #                                              ref_date=ref_date,
        #                                              no_of_batsman=7)

        opponent_player_list = list()
        for boi in range(11):
            bowler = opponent_info['bowler_' + str(boi + 1)].values[0].strip()
            if bowler == 'not_bowled':
                break
            else:
                opponent_player_list.append(bowler)


        try:
            feature_vec = fe.get_second_innings_feature_embedding_vector(target, team, opponent, location,
                                                                         team_player_list, opponent_player_list,
                                                                         ref_date=ref_date)

            if ref_date<test_start_dt:
                feature_list_train.append(feature_vec)
                result_list_train.append(win)
            else:
                feature_list_test.append(feature_vec)
                result_list_test.append(win)
        except Exception as ex:
            #raise ex
            print(ex, ' for ',team, opponent, location, ' on ',ref_date.date() )

    train_x = np.stack(feature_list_train)
    train_y = np.stack(result_list_train)

    test_x = np.stack(feature_list_test)
    test_y = np.stack(result_list_test)

    # pickle train_x, train_y,test_x,test_y,scaler
    pickle.dump(train_x, open(os.path.join(TRAIN_TEST_DIR, second_innings_train_x), 'wb'))
    pickle.dump(train_y, open(os.path.join(TRAIN_TEST_DIR, second_innings_train_y), 'wb'))

    pickle.dump(test_x, open(os.path.join(TRAIN_TEST_DIR, second_innings_test_x), 'wb'))
    pickle.dump(test_y, open(os.path.join(TRAIN_TEST_DIR, second_innings_test_y), 'wb'))

    outil.create_meta_info_entry('second_innings_train_xy', train_start,
                                 str(cricutil.substract_day_as_datetime(test_start_dt, 1).date()),
                                 file_list=[second_innings_train_x,
                                            second_innings_train_y])

    outil.create_meta_info_entry('second_innings_test_xy', str(test_start_dt.date()),
                                 str(test_end_dt.date()),
                                 file_list=[second_innings_test_x,
                                            second_innings_test_y])


def create_batsman_runs_train_test(train_start,test_start,test_end=None):
    if not os.path.isdir(TRAIN_TEST_DIR):
        os.makedirs(TRAIN_TEST_DIR)

    outil.use_model_from('dev')
    train_start_dt = cricutil.str_to_date_time(train_start)
    test_start_dt = cricutil.str_to_date_time(test_start)
    if test_end is None:
        test_end_dt = cricutil.today_as_date_time()
    else:
        test_end_dt = cricutil.str_to_date_time(test_end)

    overall_start = train_start_dt
    overall_end = test_end_dt
    print('match_list ')
    match_list_df = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION + os.sep + 'match_list.csv')
    match_list_df = match_list_df[(match_list_df['date'] >= overall_start) & \
                                  (match_list_df['date'] <= overall_end)]
    match_stats_df = pd.read_csv(dl.CSV_LOAD_LOCATION + os.sep + 'match_stats.csv')
    match_list_df = match_list_df.merge(match_stats_df, how='inner', on='match_id')
    match_id_list = list(match_list_df['match_id'].unique())

    feature_list_train = []
    target_list_train = []

    feature_list_test =[]
    target_list_test = []

    for index,match_id in tqdm(enumerate(match_id_list)):

        for team_innings,opponent_innings in zip(['first_innings','second_innings'],['second_innings','first_innings']):
            match_info = match_list_df[match_list_df['match_id']==match_id]
            team_info = match_info[match_info[team_innings]==match_info['team_statistics']]
            opponent_info = match_info[match_info[opponent_innings]==match_info['team_statistics']]

            team = team_info['team_statistics'].values[0].strip()
            opponent = opponent_info['team_statistics'].values[0].strip()
            location = team_info['location'].values[0].strip()
            ref_dt_np = team_info['date'].values[0]
            ref_date = cricutil.npdate_to_datetime(ref_dt_np)
            opponent_player_list = list()
            for boi in range(11):
                bowler = opponent_info['bowler_' + str(boi + 1)].values[0].strip()
                if bowler == 'not_bowled':
                    break
                else:
                    opponent_player_list.append(bowler)

            for bi in range(11):

                batsman = team_info['batsman_'+str(bi+1)].values[0].strip()
                if batsman == 'not_batted':
                    break
                else:
                    try:
                        feature_vector = fe.get_batsman_features_with_embedding(batsman,bi+1,opponent_player_list,team,opponent,location,ref_date=ref_date)
                        runs_scored = team_info['batsman_'+str(bi+1)+'_runs'].values[0]
                        if ref_date<test_start_dt:
                            feature_list_train.append(feature_vector)
                            target_list_train.append(runs_scored)
                        else:
                            feature_list_test.append(feature_vector)
                            target_list_test.append(runs_scored)
                    except Exception as ex:
                        print(ex, ' for ',batsman, team, opponent, location, ' on ',ref_date.date() )
                        #raise ex

    #print("pre-scaled values \n",np.stack(feature_list_train))
    train_x = np.stack(feature_list_train)
    train_y = np.stack(target_list_train)

    test_x = np.stack(feature_list_test)
    test_y = np.stack(target_list_test)

    # pickle train_x, train_y,test_x,test_y,scaler
    pickle.dump(train_x,open(os.path.join(TRAIN_TEST_DIR, batsman_runs_train_x),'wb'))
    pickle.dump(train_y, open(os.path.join(TRAIN_TEST_DIR, batsman_runs_train_y), 'wb'))

    pickle.dump(test_x, open(os.path.join(TRAIN_TEST_DIR, batsman_runs_test_x), 'wb'))
    pickle.dump(test_y, open(os.path.join(TRAIN_TEST_DIR, batsman_runs_test_y), 'wb'))

    outil.create_meta_info_entry('batsman_runs_train_xy', train_start,
                                 str(cricutil.substract_day_as_datetime(test_start_dt, 1).date()),
                                 file_list=[batsman_runs_train_x,
                                            batsman_runs_train_y])

    outil.create_meta_info_entry('batsman_runs_test_xy', str(test_start_dt.date()),
                                 str(test_end_dt.date()),
                                 file_list=[batsman_runs_test_x,
                                            batsman_runs_test_y])


def create_adversarial_train_test(train_start,test_start,test_end=None,encoding_source='dev',include_not_batted=False):
    if not os.path.isdir(TRAIN_TEST_DIR):
        os.makedirs(TRAIN_TEST_DIR)

    if encoding_source == 'production':
        encoding_dir = outil.PROD_DIR
    else:
        encoding_dir = outil.DEV_DIR

    train_start_dt = cricutil.str_to_date_time(train_start)
    test_start_dt = cricutil.str_to_date_time(test_start)
    if test_end is None:
        test_end_dt = cricutil.today_as_date_time()
    else:
        test_end_dt = cricutil.str_to_date_time(test_end)

    overall_start = train_start_dt
    overall_end = test_end_dt
    match_list_df = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION + os.sep + 'match_list.csv')
    match_list_df = match_list_df[(match_list_df['date'] >= overall_start) & \
                                  (match_list_df['date'] <= overall_end)]
    match_stats_df = pd.read_csv(dl.CSV_LOAD_LOCATION + os.sep + 'match_stats.csv')
    match_list_df = match_list_df.merge(match_stats_df,how='inner',on='match_id')

    no_of_matches = match_list_df.shape[0]
    country_enc_map = pickle.load(open(os.path.join(encoding_dir, outil.COUNTRY_ENCODING_MAP), 'rb'))
    location_enc_map_for_batsman = pickle.load(open(os.path.join(encoding_dir, outil.LOC_ENCODING_MAP_FOR_BATSMAN), 'rb'))
    batsman_enc_map = pickle.load(open(os.path.join(encoding_dir, outil.BATSMAN_ENCODING_MAP), 'rb'))
    bowler_enc_map = pickle.load(open(os.path.join(encoding_dir, outil.BOWLER_ENCODING_MAP), 'rb'))

    batsman_oh_list_train = []
    position_oh_list_train = []
    location_oh_list_train = []
    bowler_oh_list_train = []
    runs_scored_list_train = []

    batsman_oh_list_test = []
    position_oh_list_test = []
    location_oh_list_test = []
    bowler_oh_list_test = []
    runs_scored_list_test = []

    for index in tqdm(range(no_of_matches)):
        if match_list_df.iloc[index]['first_innings']==match_list_df.iloc[index]['team_statistics']:
            opponent = match_list_df.iloc[index]['second_innings'].strip()
            country = match_list_df.iloc[index]['first_innings'].strip()
        else:
            opponent = match_list_df.iloc[index]['first_innings'].strip()
            country = match_list_df.iloc[index]['second_innings'].strip()
        location = match_list_df.iloc[index]['location'].strip()
        match_id = match_list_df.iloc[index]['match_id']
        match_details = pd.read_csv(dl.CSV_LOAD_LOCATION + os.sep + str(match_id)+'.csv')

        date = match_list_df.iloc[index]['date']

        if location not in location_enc_map_for_batsman:
            print('location ',location,' not encoded for ',country,opponent,' on ',date)
            continue
        if opponent not in country_enc_map:
            print('opponent ', opponent, ' not encoded for ', country, opponent, ' on ', date, ' at ',location)
            continue

        location_oh = location_enc_map_for_batsman[location]

        team_batsman_list = list(match_details[(match_details['team']==country)]['batsman'].unique())
        opponent_bowler_list = list(match_details[(match_details['team']==country)]['bowler'].unique())

        for batsman in team_batsman_list:
            for bowler in opponent_bowler_list:
                try:
                    batsman_oh = batsman_enc_map[country.strip()+' '+batsman.strip()]
                    bowler_oh = bowler_enc_map[opponent.strip()+' '+bowler.strip()]
                    position_oh = fe.get_oh_pos(team_batsman_list.index(batsman)+1)

                    runs_scored = match_details[(match_details['team']==country) &\
                                                (match_details['batsman']==batsman) &\
                                                (match_details['bowler']==bowler)]['scored_runs'].sum()

                    if date<test_start_dt:
                        batsman_oh_list_train.append(batsman_oh)
                        position_oh_list_train.append(position_oh)
                        location_oh_list_train.append(location_oh)
                        bowler_oh_list_train.append(bowler_oh)
                        runs_scored_list_train.append(runs_scored)
                    else:
                        batsman_oh_list_test.append(batsman_oh)
                        position_oh_list_test.append(position_oh)
                        location_oh_list_test.append(location_oh)
                        bowler_oh_list_test.append(bowler_oh)
                        runs_scored_list_test.append(runs_scored)
                except Exception as ex:
                    print(ex,' for ',batsman,bowler,country,opponent,' at ',location,' on ',date)


    batsman_oh_train_x = np.stack(batsman_oh_list_train)
    position_oh_train_x = np.stack(position_oh_list_train)
    location_oh_train_x = np.stack(location_oh_list_train)
    bowler_oh_train_x = np.stack(bowler_oh_list_train)
    runs_scored_train_y = np.stack(runs_scored_list_train)

    batsman_oh_test_x = np.stack(batsman_oh_list_test)
    position_oh_test_x = np.stack(position_oh_list_test)
    location_oh_test_x = np.stack(location_oh_list_test)
    bowler_oh_test_x = np.stack(bowler_oh_list_test)
    runs_scored_test_y = np.stack(runs_scored_list_test)

    pickle.dump(batsman_oh_train_x, open(os.path.join(TRAIN_TEST_DIR, adversarial_feature_batsman_oh_train_x), 'wb'))
    pickle.dump(position_oh_train_x, open(os.path.join(TRAIN_TEST_DIR, adversarial_feature_position_oh_train_x), 'wb'))
    pickle.dump(location_oh_train_x, open(os.path.join(TRAIN_TEST_DIR, adversarial_feature_location_oh_train_x), 'wb'))
    pickle.dump(bowler_oh_train_x, open(os.path.join(TRAIN_TEST_DIR, adversarial_feature_bowler_oh_train_x), 'wb'))
    pickle.dump(runs_scored_train_y, open(os.path.join(TRAIN_TEST_DIR, adversarial_feature_runs_scored_train_y), 'wb'))

    outil.create_meta_info_entry('adversarial_train_xy', train_start,
                                 str(cricutil.substract_day_as_datetime(test_start_dt, 1).date()),
                                 file_list=[adversarial_feature_batsman_oh_train_x,
                                            adversarial_feature_position_oh_train_x,
                                            adversarial_feature_location_oh_train_x,
                                            adversarial_feature_bowler_oh_train_x,
                                            adversarial_feature_runs_scored_train_y])

    pickle.dump(batsman_oh_test_x, open(os.path.join(TRAIN_TEST_DIR, adversarial_feature_batsman_oh_test_x), 'wb'))
    pickle.dump(position_oh_test_x, open(os.path.join(TRAIN_TEST_DIR, adversarial_feature_position_oh_test_x), 'wb'))
    pickle.dump(location_oh_test_x, open(os.path.join(TRAIN_TEST_DIR, adversarial_feature_location_oh_test_x), 'wb'))
    pickle.dump(bowler_oh_test_x, open(os.path.join(TRAIN_TEST_DIR, adversarial_feature_bowler_oh_test_x), 'wb'))
    pickle.dump(runs_scored_test_y, open(os.path.join(TRAIN_TEST_DIR, adversarial_feature_runs_scored_test_y), 'wb'))

    outil.create_meta_info_entry('adversarial_test_xy', test_start, str(test_end_dt.date()),
                                 file_list=[adversarial_feature_batsman_oh_test_x,
                                            adversarial_feature_position_oh_test_x,
                                            adversarial_feature_location_oh_test_x,
                                            adversarial_feature_bowler_oh_test_x,
                                            adversarial_feature_runs_scored_test_y])


def create_adversarial_first_innings_train_test(train_start,test_start,test_end=None):
    if not os.path.isdir(TRAIN_TEST_DIR):
        os.makedirs(TRAIN_TEST_DIR)

    outil.use_model_from('dev')
    train_start_dt = cricutil.str_to_date_time(train_start)
    test_start_dt = cricutil.str_to_date_time(test_start)
    if test_end is None:
        test_end_dt = cricutil.today_as_date_time()
    else:
        test_end_dt = cricutil.str_to_date_time(test_end)

    overall_start = train_start_dt
    overall_end = test_end_dt
    match_list_df = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION + os.sep + 'match_list.csv')
    match_list_df = match_list_df[(match_list_df['date'] >= overall_start) & \
                                  (match_list_df['date'] <= overall_end)]
    match_stats_df = pd.read_csv(dl.CSV_LOAD_LOCATION + os.sep + 'match_stats.csv')
    match_list_df = match_list_df.merge(match_stats_df, how='inner', on='match_id')
    match_id_list = list(match_list_df['match_id'].unique())

    feature_list_train = []
    target_list_train = []

    feature_list_test =[]
    target_list_test = []

    for index,match_id in tqdm(enumerate(match_id_list)):
        match_info = match_list_df[match_list_df['match_id']==match_id]
        team_info = match_info[match_info['first_innings']==match_info['team_statistics']]
        opponent_info = match_info[match_info['second_innings']==match_info['team_statistics']]

        team = team_info['team_statistics'].values[0].strip()
        opponent = opponent_info['team_statistics'].values[0].strip()
        location = team_info['location'].values[0].strip()
        ref_dt_np = team_info['date'].values[0]
        ref_date = cricutil.npdate_to_datetime(ref_dt_np)
        runs_scored = team_info['total_run'].values[0]

        team_player_list = list()
        for bi in range(11):
            batsman = team_info['batsman_'+str(bi+1)].values[0].strip()
            if batsman == 'not_batted':
                break
            else:
                team_player_list.append(batsman)
        opponent_player_list = list()
        for boi in range(11):
            bowler = opponent_info['bowler_' + str(boi + 1)].values[0].strip()
            if bowler == 'not_bowled':
                break
            else:
                opponent_player_list.append(bowler)

        try:
            feature_vector = fe.get_adversarial_first_innings_feature_vector(team, opponent, location,
                                                                            team_player_list, opponent_player_list,
                                                                            ref_date=ref_date)

            if ref_date<test_start_dt:
                feature_list_train.append(feature_vector)
                target_list_train.append(runs_scored)
            else:
                feature_list_test.append(feature_vector)
                target_list_test.append(runs_scored)
        except Exception as ex:
            print(ex, ' for ',team, opponent, location, ' on ',ref_date.date() )

    #print("pre-scaled values \n",np.stack(feature_list_train))
    train_x = np.stack(feature_list_train)
    train_y = np.stack(target_list_train)

    test_x = np.stack(feature_list_test)
    test_y = np.stack(target_list_test)

    # pickle train_x, train_y,test_x,test_y,scaler
    pickle.dump(train_x,open(os.path.join(TRAIN_TEST_DIR,adversarial_first_innings_train_x),'wb'))
    pickle.dump(train_y, open(os.path.join(TRAIN_TEST_DIR, adversarial_first_innings_train_y), 'wb'))

    pickle.dump(test_x, open(os.path.join(TRAIN_TEST_DIR, adversarial_first_innings_test_x), 'wb'))
    pickle.dump(test_y, open(os.path.join(TRAIN_TEST_DIR, adversarial_first_innings_test_y), 'wb'))


    outil.create_meta_info_entry('adversarial_first_innings_train_xy', train_start,
                                 str(cricutil.substract_day_as_datetime(test_start_dt, 1).date()),
                                 file_list=[adversarial_first_innings_train_x,
                                            adversarial_first_innings_train_y])

    outil.create_meta_info_entry('adversarial_first_innings_test_xy', str(test_start_dt.date()),
                                 str(test_end_dt.date()),
                                 file_list=[adversarial_first_innings_test_x,
                                            adversarial_first_innings_test_y])

def create_combined_prediction_train_test(train_start,test_start,test_end=None, first_innings_emb=True, second_innings_emb=True):

    if not os.path.isdir(TRAIN_TEST_DIR):
        os.makedirs(TRAIN_TEST_DIR)

    train_start_dt = cricutil.str_to_date_time(train_start)
    test_start_dt = cricutil.str_to_date_time(test_start)
    if test_end is None:
        test_end_dt = cricutil.today_as_date_time()
    else:
        test_end_dt = cricutil.str_to_date_time(test_end)

    overall_start = train_start_dt
    overall_end = test_end_dt
    match_list_df = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION + os.sep + 'cricinfo_match_list.csv')
    match_list_df = match_list_df[(match_list_df['date'] >= overall_start) & \
                                  (match_list_df['date'] <= overall_end)]
    batting_list_df = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION + os.sep + 'cricinfo_batting.csv')
    batting_list_df = batting_list_df[(batting_list_df['date'] >= overall_start) & \
                                      (batting_list_df['date'] <= overall_end)]
    bowling_list_df = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION + os.sep + 'cricinfo_bowling.csv')
    bowling_list_df = bowling_list_df[(bowling_list_df['date'] >= overall_start) & \
                                      (bowling_list_df['date'] <= overall_end)]

    match_id_list = list(match_list_df['match_id'].unique())

    feature_list_train = []
    target_list_train = []

    feature_list_test = []
    target_list_test = []

    for match_id in tqdm(match_id_list):

        team_a = match_list_df[match_list_df['match_id'] == match_id].iloc[0]["first_innings"]
        team_b = match_list_df[match_list_df['match_id'] == match_id].iloc[0]["second_innings"]
        location = match_list_df[match_list_df['match_id'] == match_id].iloc[0]["location"]
        ref_dt_np = match_list_df[match_list_df['match_id'] == match_id].iloc[0]["date"]
        ref_date = cricutil.pandas_timestamp_to_datetime(ref_dt_np)
        runs_scored = match_list_df[match_list_df['match_id'] == match_id].iloc[0]['first_innings_run']

        team_a_player_list_df = batting_list_df[
            (batting_list_df['match_id'] == match_id) & (batting_list_df['team'] == team_a)]

        team_a_player_list_df = team_a_player_list_df[['team', 'name', 'position']]

        team_a_bowler_list_df = bowling_list_df[
            (bowling_list_df['match_id'] == match_id) & (bowling_list_df['team'] == team_a)]

        team_a_bowler_list_df = team_a_bowler_list_df[['team', 'name']]

        try :
            predc.set_first_innings_emb(first_innings_emb)
            predc.set_second_innings_emb(second_innings_emb)

            outil.use_model_from("dev")
            target_by_a = predc.predict_first_innings_run(team,opponent,location,team_batsman_list,opponent_bowler_list,ref_date=ref_date,no_of_years=None,mode="train")
            #target_by_a = current_mean

            success_by_b, probability_by_b = pred.predict_second_innings_success(target_by_a, opponent, team, location,opponent_batsman_list, team_bowler_list,ref_date=ref_date, no_of_years=None,mode="train")

            target_by_b = pred.predict_first_innings_run(opponent,team,location,opponent_batsman_list,team_bowler_list,ref_date=ref_date,no_of_years=None,mode="train")
            #target_by_b = current_mean_b

            success_by_a, probability_by_a = pred.predict_second_innings_success(target_by_b, team, opponent, location,team_batsman_list, opponent_bowler_list,ref_date=ref_date, no_of_years=None,mode="train")


            # print("----------summary------------")
            # print(team,opponent,location,ref_date,np.array([target_by_a,probability_by_b,target_by_b,probability_by_a]),win_flag,innings_team["winner"].values[0])
            # print(team_batsman_list)
            # print(team_bowler_list)
            # print(opponent_batsman_list)
            # print(opponent_bowler_list)
            # print("===============================")
            # print(np.array([target_by_a,probability_by_b,target_by_b,probability_by_a]))
            # print(win_flag)
            if ref_date<test_start_dt:
                feature_list_train.append(np.array([target_by_a,probability_by_b,target_by_b,probability_by_a]))
                target_list_train.append(win_flag)
            else:
                feature_list_test.append(np.array([target_by_a,probability_by_b,target_by_b,probability_by_a]))
                target_list_test.append(win_flag)
                #print("added to test...")

        except Exception as ex:
            print(ex," : ignored ", team ," vs ",opponent, " on ",ref_date," at ",location )
            #raise ex

    print("train size ",len(feature_list_train))
    train_x = np.stack(feature_list_train)
    train_y = np.stack(target_list_train)
    print("test size ", len(feature_list_test))
    test_x = np.stack(feature_list_test)
    test_y = np.stack(target_list_test)

    # pickle train_x, train_y,test_x,test_y,
    pickle.dump(train_x, open(os.path.join(TRAIN_TEST_DIR, "fi_" + str(first_innings_emb) + "_si_" + str(second_innings_emb) + "_" + combined_train_x), 'wb'))
    pickle.dump(train_y, open(os.path.join(TRAIN_TEST_DIR, "fi_" + str(first_innings_emb) + "_si_" + str(second_innings_emb) + "_" + combined_train_y), 'wb'))

    pickle.dump(test_x, open(os.path.join(TRAIN_TEST_DIR, "fi_" + str(first_innings_emb) + "_si_" + str(second_innings_emb) + "_" + combined_test_x), 'wb'))
    pickle.dump(test_y, open(os.path.join(TRAIN_TEST_DIR, "fi_" + str(first_innings_emb) + "_si_" + str(second_innings_emb) + "_" + combined_test_y), 'wb'))

    outil.create_meta_info_entry(
        'combined_train_xy_' + "fi_" + str(first_innings_emb) + "_si_" + str(second_innings_emb), train_start,
        str(cricutil.substract_day_as_datetime(test_start_dt, 1).date()),
        file_list=[combined_train_x,
                   combined_train_y])

    outil.create_meta_info_entry(
        'combined_test_xy_' + "fi_" + str(first_innings_emb) + "_si_" + str(second_innings_emb),
        str(test_start_dt.date()),
        str(test_end_dt.date()),
        file_list=[combined_test_x,
                   combined_test_y])


def create_first_fow_train_test(train_start,test_start,test_end=None):
    if not os.path.isdir(TRAIN_TEST_DIR):
        os.makedirs(TRAIN_TEST_DIR)

    outil.use_model_from('dev')
    train_start_dt = cricutil.str_to_date_time(train_start)
    test_start_dt = cricutil.str_to_date_time(test_start)
    if test_end is None:
        test_end_dt = cricutil.today_as_date_time()
    else:
        test_end_dt = cricutil.str_to_date_time(test_end)

    overall_start = train_start_dt
    overall_end = test_end_dt
    match_list_df = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION + os.sep + 'match_list.csv')
    match_list_df = match_list_df[(match_list_df['date'] >= overall_start) & \
                                  (match_list_df['date'] <= overall_end)]
    match_stats_df = pd.read_csv(dl.CSV_LOAD_LOCATION + os.sep + 'match_stats.csv')
    match_list_df = match_list_df.merge(match_stats_df, how='inner', on='match_id')
    match_id_list = list(match_list_df['match_id'].unique())

    feature_list_train = []
    target_list_train = []

    feature_list_test =[]
    target_list_test = []
    no_of_basman = 0
    for index,match_id in tqdm(enumerate(match_id_list)):
        match_info = match_list_df[match_list_df['match_id']==match_id]
        team_info = match_info[match_info['first_innings']==match_info['team_statistics']]
        opponent_info = match_info[match_info['second_innings']==match_info['team_statistics']]

        team = team_info['team_statistics'].values[0].strip()
        opponent = opponent_info['team_statistics'].values[0].strip()
        location = team_info['location'].values[0].strip()
        ref_dt_np = team_info['date'].values[0]
        ref_date = cricutil.npdate_to_datetime(ref_dt_np)
        runs_scored = team_info['total_run'].values[0]

        team_player_list = list()
        for bi in range(11):
            batsman = team_info['batsman_'+str(bi+1)].values[0].strip()
            if batsman == 'not_batted':
                break
            else:
                team_player_list.append(batsman)


        opponent_player_list = list()
        for boi in range(11):
            bowler = opponent_info['bowler_' + str(boi + 1)].values[0].strip()
            if bowler == 'not_bowled':
                break
            else:
                opponent_player_list.append(bowler)


        try:
            feature_dict = fe.get_instance_feature_dict(team, opponent, location,
                                                        team_player_list, opponent_player_list,
                                                        ref_date)
            #print(feature_dict)
            no_of_basman = no_of_basman+len(team_player_list)

            if ref_date<test_start_dt:
                feature_list_train.append(feature_dict)
                target_list_train.append(runs_scored)
            else:
                feature_list_test.append(feature_dict)
                target_list_test.append(runs_scored)
        except Exception as ex:
            print(ex, ' for ',team, opponent, location, ' on ',ref_date.date() )
            #raise ex

    print('mean no of batsman - ',no_of_basman/index)
    train_y = np.stack(target_list_train)
    test_y = np.stack(target_list_test)

    #print(pd.DataFrame(feature_list_train))
    train_x = np.array(pd.DataFrame(feature_list_train).drop(columns=['team','opponent','location']))
    #print(train_x)
    test_x  = np.array(pd.DataFrame(feature_list_test).drop(columns=['team','opponent','location']))
    cols = list(pd.DataFrame(feature_list_train).drop(columns=['team','opponent','location']).columns)

    pickle.dump(train_x,open(os.path.join(TRAIN_TEST_DIR,first_innings_base_train_x),'wb'))
    pickle.dump(train_y, open(os.path.join(TRAIN_TEST_DIR,first_innings_base_train_y), 'wb'))
    pickle.dump(cols, open(os.path.join(outil.DEV_DIR, first_innings_base_columns), 'wb'))

    outil.create_meta_info_entry('first_innings_base_train_xy', train_start,
                                 str(cricutil.substract_day_as_datetime(test_start_dt, 1).date()),
                                 file_list=[first_innings_base_train_x,
                                            first_innings_base_train_y,
                                            first_innings_base_columns])

    pickle.dump(test_x, open(os.path.join(TRAIN_TEST_DIR, first_innings_base_test_x), 'wb'))
    pickle.dump(test_y, open(os.path.join(TRAIN_TEST_DIR, first_innings_base_test_y), 'wb'))

    outil.create_meta_info_entry('first_innings_base_test_xy', str(test_start_dt.date()),
                                 str(test_end_dt.date()),
                                 file_list=[first_innings_base_test_x,
                                            first_innings_base_test_y])

    # trend prediction test data
    trend_data_df = pd.DataFrame(feature_list_test)
    trend_data_df['runs_scored'] = target_list_test
    trend_data_df[['team','opponent','location','opponent_trend_predict','location_trend_predict','current_trend_predict','runs_scored']].to_csv(os.path.join(outil.DEV_DIR, "trend_predict.csv"), index=False)
    # trend_data_df = trend_data_df[['opponent_trend_predict','location_trend_predict','current_trend_predict','runs_scored']]

    mape_opponent_trend = cric_eval.mape(np.array(trend_data_df['runs_scored']),
                                         np.array(trend_data_df['opponent_trend_predict']))

    mape_location_trend = cric_eval.mape(np.array(trend_data_df['runs_scored']),
                                    np.array(trend_data_df['location_trend_predict']))

    mape_current_trend = cric_eval.mape(np.array(trend_data_df['runs_scored']),
                                         np.array(trend_data_df['current_trend_predict']))

    mape_opponent_mean = cric_eval.mape(np.array(trend_data_df['runs_scored']),
                                         np.array(trend_data_df['opponent_mean']))

    mape_location_mean = cric_eval.mape(np.array(trend_data_df['runs_scored']),
                                         np.array(trend_data_df['location_mean']))

    mape_current_mean = cric_eval.mape(np.array(trend_data_df['runs_scored']),
                                        np.array(trend_data_df['current_mean']))


    # trend prediction train data

    trend_data_train_df = pd.DataFrame(feature_list_train)
    trend_data_train_df['runs_scored'] = target_list_train
    trend_data_train_df[
        ['team', 'opponent', 'location', 'opponent_trend_predict', 'location_trend_predict', 'current_trend_predict',
         'runs_scored']].to_csv(os.path.join(outil.DEV_DIR, "trend_predict_train.csv"), index=False)
    # trend_data_train_df = trend_data_train_df[['opponent_trend_predict','location_trend_predict','current_trend_predict','runs_scored']]

    mape_train_opponent_trend = cric_eval.mape(np.array(trend_data_train_df['runs_scored']),
                                         np.array(trend_data_train_df['opponent_trend_predict']))

    mape_train_location_trend = cric_eval.mape(np.array(trend_data_train_df['runs_scored']),
                                         np.array(trend_data_train_df['location_trend_predict']))

    mape_train_current_trend = cric_eval.mape(np.array(trend_data_train_df['runs_scored']),
                                        np.array(trend_data_train_df['current_trend_predict']))

    mape_train_opponent_mean = cric_eval.mape(np.array(trend_data_train_df['runs_scored']),
                                        np.array(trend_data_train_df['opponent_mean']))

    mape_train_location_mean = cric_eval.mape(np.array(trend_data_train_df['runs_scored']),
                                        np.array(trend_data_train_df['location_mean']))

    mape_train_current_mean = cric_eval.mape(np.array(trend_data_train_df['runs_scored']),
                                       np.array(trend_data_train_df['current_mean']))

    outil.create_model_meta_info_entry('first_innings_trend_prediction_metrics',
                                       (mape_train_opponent_trend, mape_train_location_trend, mape_train_current_trend,
                                        mape_train_opponent_mean, mape_train_location_mean, mape_train_current_mean),
                                       (mape_opponent_trend, mape_location_trend, mape_current_trend,
                                        mape_opponent_mean, mape_location_mean, mape_current_mean),
                                       info="metrics is mape_opponent_trend, mape_location_trend, mape_current_trend,"+
                                            "mape_opponent_mean, mape_location_mean, mape_current_mean ",
                                       file_list=[
                                           "tred_predict.csv",
                                       ]
                                       )
    print("mape_train_opponent_trend", mape_train_opponent_trend)
    print("mape_train_location_trend", mape_train_location_trend)
    print("mape_train_current_trend", mape_train_current_trend)

    print("mape_train_opponent_mean", mape_train_opponent_mean)
    print("mape_train_location_mean", mape_train_location_mean)
    print("mape_train_current_mean", mape_train_current_mean)

    print("mape_opponent_trend",mape_opponent_trend)
    print("mape_location_trend", mape_location_trend)
    print("mape_current_trend", mape_current_trend)

    print("mape_opponent_mean", mape_opponent_mean)
    print("mape_location_mean", mape_location_mean)
    print("mape_current_mean", mape_current_mean)



@click.group()
def traintest():
    pass

@traintest.command()
@click.option('--train_start', help='start date for train data (YYYY-mm-dd)',required=True)
@click.option('--test_start', help='start date for test data (YYYY-mm-dd)',required=True)
@click.option('--test_end', help='end date for test (YYYY-mm-dd)')
@click.option('--encoding_source', help='which enviornment to read from for one hot encoding(dev/production)',
              default='dev')
def country_embedding(train_start,test_start,test_end=None,encoding_source='dev'):
    create_country_embedding_train_test(train_start,
                                        test_start,
                                        test_end=test_end,
                                        encoding_source=encoding_source)

@traintest.command()
@click.option('--train_start', help='start date for train data (YYYY-mm-dd)',required=True)
@click.option('--test_start', help='start date for test data (YYYY-mm-dd)',required=True)
@click.option('--test_end', help='end date for test (YYYY-mm-dd)')
@click.option('--encoding_source', help='which enviornment to read from for one hot encoding(dev/production)',
              default='dev')
def country_embedding_2nd(train_start,test_start,test_end=None,encoding_source='dev'):
    create_country_embedding_second_innings_train_test(train_start,
                                        test_start,
                                        test_end=test_end,
                                        encoding_source=encoding_source)

@traintest.command()
@click.option('--train_start', help='start date for train data (YYYY-mm-dd)',required=True)
@click.option('--test_start', help='start date for test data (YYYY-mm-dd)',required=True)
@click.option('--test_end', help='end date for test (YYYY-mm-dd)')
@click.option('--encoding_source', help='which environment to read from for one hot encoding(dev/production)',
              default='dev')
@click.option('--include_not_batted', help='whether to create an encoding for not_batted)',
              default=False,type=bool)
def batsman_embedding(train_start,test_start,test_end,encoding_source,include_not_batted):
    create_batsman_embedding_train_test(train_start,
                                        test_start,
                                        test_end=test_end,
                                        encoding_source=encoding_source,
                                        include_not_batted=include_not_batted)

@traintest.command()
@click.option('--train_start', help='start date for train data (YYYY-mm-dd)',required=True)
@click.option('--test_start', help='start date for test data (YYYY-mm-dd)',required=True)
@click.option('--test_end', help='end date for test (YYYY-mm-dd)')
@click.option('--encoding_source', help='which environment to read from for one hot encoding(dev/production)',
              default='dev')
@click.option('--include_not_batted', help='whether to create an encoding for not_batted)',
              default=False)
def adversarial(train_start,test_start,test_end,encoding_source,include_not_batted):
    create_adversarial_train_test(train_start,
                                  test_start,
                                  test_end=test_end,
                                  encoding_source=encoding_source,
                                  include_not_batted=include_not_batted)


@traintest.command()
@click.option('--train_start', help='start date for train data (YYYY-mm-dd)',required=True)
@click.option('--test_start', help='start date for test data (YYYY-mm-dd)',required=True)
@click.option('--test_end', help='end date for test (YYYY-mm-dd)')
def first_innings_base(train_start, test_start, test_end):
    create_first_innings_base_train_test(train_start, test_start, test_end=test_end)


@traintest.command()
@click.option('--train_start', help='start date for train data (YYYY-mm-dd)',required=True)
@click.option('--test_start', help='start date for test data (YYYY-mm-dd)',required=True)
@click.option('--test_end', help='end date for test (YYYY-mm-dd)')
def second_innings_base(train_start, test_start, test_end):
    create_second_innings_base_train_test(train_start, test_start, test_end=test_end)

@traintest.command()
@click.option('--train_start', help='start date for train data (YYYY-mm-dd)',required=True)
@click.option('--test_start', help='start date for test data (YYYY-mm-dd)',required=True)
@click.option('--test_end', help='end date for test (YYYY-mm-dd)')
def one_shot(train_start, test_start, test_end):
    create_one_shot_prediction_train_test(train_start, test_start, test_end=test_end)

@traintest.command()
@click.option('--train_start', help='start date for train data (YYYY-mm-dd)',required=True)
@click.option('--test_start', help='start date for test data (YYYY-mm-dd)',required=True)
@click.option('--test_end', help='end date for test (YYYY-mm-dd)')
@click.option('--embedding', help='whether to use embedding',type=bool, default=False)
def one_shot_multi(train_start, test_start, test_end ,embedding):
    create_one_shot_multi_output_train_test(train_start, test_start, test_end=test_end,embedding=embedding)

@traintest.command()
@click.option('--train_start', help='start date for train data (YYYY-mm-dd)',required=True)
@click.option('--test_start', help='start date for test data (YYYY-mm-dd)',required=True)
@click.option('--test_end', help='end date for test (YYYY-mm-dd)')
def mg(train_start, test_start, test_end):
    create_mg_classification_train_test(train_start, test_start, test_end=test_end)


@traintest.command()
@click.option('--train_start', help='start date for train data (YYYY-mm-dd)',required=True)
@click.option('--test_start', help='start date for test data (YYYY-mm-dd)',required=True)
@click.option('--test_end', help='end date for test (YYYY-mm-dd)')
@click.option('--embedding', help='whether to use embedding',type=bool, default=False)
def second_level_any(train_start, test_start, test_end ,embedding):
    create_second_level_any_innings_train_test(train_start, test_start, test_end=test_end,embedding=embedding)

@traintest.command()
@click.option('--train_start', help='start date for train data (YYYY-mm-dd)',required=True)
@click.option('--test_start', help='start date for test data (YYYY-mm-dd)',required=True)
@click.option('--test_end', help='end date for test (YYYY-mm-dd)')
@click.option('--embedding', help='whether to use embedding',type=bool, default=False)
def second_level_non_neural(train_start, test_start, test_end ,embedding):
    create_second_level_any_innings_non_neural_train_test(train_start, test_start, test_end=test_end,embedding=embedding)


@traintest.command()
@click.option('--train_start', help='start date for train data (YYYY-mm-dd)',required=True)
@click.option('--test_start', help='start date for test data (YYYY-mm-dd)',required=True)
@click.option('--test_end', help='end date for test (YYYY-mm-dd)')
def first_innings(train_start, test_start, test_end):
    create_first_innings_train_test(train_start, test_start, test_end=test_end)

@traintest.command()
@click.option('--train_start', help='start date for train data (YYYY-mm-dd)',required=True)
@click.option('--test_start', help='start date for test data (YYYY-mm-dd)',required=True)
@click.option('--test_end', help='end date for test (YYYY-mm-dd)')
def second_innings(train_start, test_start, test_end):
    create_second_innings_train_test(train_start, test_start, test_end=test_end)

@traintest.command()
@click.option('--train_start', help='start date for train data (YYYY-mm-dd)',required=True)
@click.option('--test_start', help='start date for test data (YYYY-mm-dd)',required=True)
@click.option('--test_end', help='end date for test (YYYY-mm-dd)')
def batsman_runs(train_start, test_start, test_end):
    create_batsman_runs_train_test(train_start, test_start, test_end=test_end)

@traintest.command()
@click.option('--train_start', help='start date for train data (YYYY-mm-dd)',required=True)
@click.option('--test_start', help='start date for test data (YYYY-mm-dd)',required=True)
@click.option('--test_end', help='end date for test (YYYY-mm-dd)')
def adversarial_first_innings(train_start, test_start, test_end):
    create_adversarial_first_innings_train_test(train_start, test_start, test_end=test_end)


@traintest.command()
@click.option('--train_start', help='start date for train data (YYYY-mm-dd)',required=True)
@click.option('--test_start', help='start date for test data (YYYY-mm-dd)',required=True)
@click.option('--test_end', help='end date for test (YYYY-mm-dd)')
@click.option('--first_innings_emb', help='whether to use embedding in first innnings',default=True,type=bool)
@click.option('--second_innings_emb', help='whether to use embedding in first innnings',default=True,type=bool)
def combined_prediction(train_start, test_start, test_end, first_innings_emb,second_innings_emb):
    create_combined_prediction_train_test(train_start, test_start, test_end=test_end,first_innings_emb=first_innings_emb,second_innings_emb=second_innings_emb)

@traintest.command()
@click.option('--test_start', help='start date for test data (YYYY-mm-dd)',required=True)
@click.option('--test_end', help='end date for test (YYYY-mm-dd)')
def test_threshold(test_start, test_end):
    verify_threshod(test_start,test_end=test_end)

if __name__=="__main__":
    traintest()
