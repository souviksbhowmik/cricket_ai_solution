from odi.model_util import odi_util as outil
from odi.feature_engg import feature_extractor_ci as fec
from odi.feature_engg import util as cricutil
from odi.preprocessing import rank
from odi.retrain import  create_train_test as ctt
import pickle
import os
import click
import pandas as pd
import warnings
import json
import numpy as np

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)




def get_optimum_run(team_b,team_a,location,team_b_batsman_df,team_a_bowler_df,ref_date=None):
    feature_dict = fec.get_instance_feature_dict(team_b, team_a, location, team_b_batsman_df, team_a_bowler_df, ref_date=ref_date,innings_type="second")
    starting_run = 260
    feature_dict['target_score'] = starting_run

    second_innings_model = pickle.load(open(os.path.join(outil.DEV_DIR, outil.SECOND_INNINGS_MODEL_BASE), 'rb'))
    second_innings_selected_column = pickle.load(open(os.path.join(outil.DEV_DIR, outil.SECOND_INNINGS_FEATURE_PICKLE), 'rb'))

    x = np.array(pd.DataFrame([feature_dict])[second_innings_selected_column])

    win =  second_innings_model.predict(x)[0]
    initial_val = win

    thresh = None
    if win==1:
        while win==1 and starting_run <350:
            starting_run=starting_run+5
            x_new = np.array(x)
            x_new[0,-1]=starting_run
            win = second_innings_model.predict(x_new)[0]
            if win ==0:

                thresh = starting_run
    else:
        while win==0 and starting_run >150:
            starting_run=starting_run-5
            x_new = np.array(x)
            x_new[0,-1]=starting_run
            win = second_innings_model.predict(x_new)[0]
            if win ==1:

                thresh = starting_run


    if thresh is None and initial_val==1:
        thresh = 500
    elif thresh is None and initial_val==0:
        thresh  =0
    else:
        pass
    return thresh




@click.group()
def predict():
    warnings.filterwarnings('ignore')
    pass

@predict.command()
@click.option('--team_a_xlsx', help='team_a template excel',default='team_a.xlsx')
@click.option('--team_b_xlsx', help='team_b template excel',default='team_b.xlsx')
@click.option('--ref_date', help='date of the match (by default current)')
@click.option('--no_of_years', help='no of years for considering trend')
@click.option('--non_neural', help='whether to use non neural approach',default=True,type=bool)
@click.option('--neural', help='whether to use neural network',default=True,type=bool)
@click.option('--any_sequence', help='whether to use any sequence of innings, by default team_a is first innings',default=True,type=bool)
@click.option('--second_only', help='whether to use any sequence',default=False,type=bool)
@click.option('--target', help='target for second innings(applicable for second_only)')
@click.option('--simple_one_shot', help='using one shot classification (no first innings prediciton)',default=True,type=bool)
@click.option('--mg', help='using prior thesis',default=False,type=bool)
@click.option('--env', help='which models to use for prediction',default='production')
def match(team_a_xlsx,team_b_xlsx,ref_date,no_of_years,non_neural,neural,any_sequence,second_only,target,simple_one_shot,mg,env):

    outil.use_model_from(env)

    if ref_date is not None:
        ref_date = cricutil.str_to_date_time(ref_date)

    team_a_df = pd.read_excel(team_a_xlsx)
    team_b_df = pd.read_excel(team_b_xlsx)

    team_a = team_a_df['team'].values[0]
    team_b = team_b_df['team'].values[0]

    location = team_a_df['location'].values[0]
    location_b = team_b_df['location'].values[0]
    team_a_player_df = team_a_df[team_a_df['playing']=='Y']
    team_b_player_df = team_b_df[team_b_df['playing'] == 'Y']
    team_a_player_df = team_a_player_df[['team', 'name', 'position']]
    team_b_player_df = team_b_player_df[['team', 'name', 'position']]


    team_a_bowler_df = team_a_df[(team_a_df['playing'] == 'Y')&(team_a_df['bowler'] == 'Y')]
    team_b_bowler_df = team_b_df[(team_b_df['playing'] == 'Y')&(team_b_df['bowler'] == 'Y')]
    team_a_bowler_df = team_a_bowler_df[['team', 'name']]
    team_b_bowler_df = team_b_bowler_df[['team', 'name']]

    if location != location_b:
        raise Exception('location of team A does not match team B')

    if team_a_player_df.shape[0]>11:
        raise Exception('More than 11 players for team A')

    if team_b_player_df.shape[0]>11:
        raise Exception('More than 11 players for team B')

    print(team_a,' vs ',team_b,' at ',location,'\n\n')

    print(team_a,' players : ')
    print(list(team_a_player_df['name']),'\n\n')

    print(team_b, ' players : ')
    print(list(team_b_player_df['name']), '\n\n')

    print('==============RESULTS==============')
    if second_only:
        any_sequence = False

    if mg:
        print("Inference option with MG currently not available")
        exit()

    if simple_one_shot:
        feature_dict_one_shot = fec.get_one_shot_feature_dict(team_a, team_b, location, team_a_player_df,
                                                     team_b_player_df, team_a_bowler_df, team_b_bowler_df,
                                                     ref_date=ref_date, no_of_years=no_of_years)

        one_shot_columns = pickle.load(open(os.path.join(outil.DEV_DIR, outil.ONE_SHOT_CLASSIFICATION_FEATURE_PICKLE), 'rb'))
        simple_one_shot_model = pickle.load(open(os.path.join(outil.DEV_DIR, outil.ONE_SHOT_CLASSIFICATION_MODEL), 'rb'))

        feature_vector_one_shot = np.array(pd.DataFrame([feature_dict_one_shot])[one_shot_columns])

        one_shot_team_a_win = simple_one_shot_model.predict(feature_vector_one_shot)[0]
        one_shot_team_a_win_probability = simple_one_shot_model.predict_proba(feature_vector_one_shot)[0][1]

        print(" Through simple one shot classification ")
        print(team_a," will win ? ",np.array([int(one_shot_team_a_win)]).astype(bool)[0])
        print(" with probability ",one_shot_team_a_win_probability)
        print("===================Break up prediction=========================")


    if non_neural:
        first_innings_model = pickle.load(open(os.path.join(outil.DEV_DIR, outil.FIRST_INNINGS_MODEL_BASE), 'rb'))
        second_innings_model = pickle.load(open(os.path.join(outil.DEV_DIR, outil.SECOND_INNINGS_MODEL_BASE), 'rb'))
        first_innings_selected_column = pickle.load(open(os.path.join(outil.DEV_DIR, outil.FIRST_INNINGS_FEATURE_PICKLE), 'rb'))
        second_innings_selected_column = pickle.load(open(os.path.join(outil.DEV_DIR, outil.SECOND_INNINGS_FEATURE_PICKLE), 'rb'))

        combined_model = pickle.load(open(os.path.join(outil.DEV_DIR, outil.COMBINED_MODEL_NON_NEURAL), 'rb'))
        if not second_only:
            feature_dict_team_a_batting_first = fec.get_instance_feature_dict(team_a, team_b, location,
                                                                              team_a_player_df,
                                                                              team_b_bowler_df, ref_date=ref_date,
                                                                              innings_type="first",no_of_years=no_of_years)
            feature_vec_team_a_first_batting = np.array(pd.DataFrame([feature_dict_team_a_batting_first])[first_innings_selected_column])

            team_a_first_target = first_innings_model.predict(feature_vec_team_a_first_batting)[0]
            target = team_a_first_target
            print(" For ", team_a, " batting first ")
            print(team_a, " scores ", target)



        if target is None:
            raise Exception("Provide target for chase prediction")
        feature_dict_team_b_batting_second = fec.get_instance_feature_dict(team_b, team_a, location, team_b_player_df,
                                                                           team_a_bowler_df, ref_date=ref_date,
                                                                           innings_type='second',no_of_years=no_of_years)
        feature_dict_team_b_batting_second['target_score'] = target
        feature_vector_team_b_chasing = np.array(pd.DataFrame([feature_dict_team_b_batting_second])[second_innings_selected_column])
        team_a_defend_success_probability = second_innings_model.predict_proba(feature_vector_team_b_chasing)[0][0]
        team_b_chasing_success = second_innings_model.predict(feature_vector_team_b_chasing)[0]


        print(team_b, " will be able to chase ? ",np.array([int(team_b_chasing_success)]).astype(bool)[0])
        print(team_a, " win probability ", team_a_defend_success_probability)
        if any_sequence:
            feature_dict_team_b_batting_first = fec.get_instance_feature_dict(team_b, team_a, location,
                                                                               team_b_player_df,
                                                                               team_a_bowler_df, ref_date=ref_date,
                                                                               innings_type='first',no_of_years=no_of_years)
            feature_dict_team_a_batting_second = fec.get_instance_feature_dict(team_a, team_b, location,
                                                                              team_a_player_df,
                                                                              team_b_bowler_df, ref_date=ref_date,
                                                                              innings_type="second",no_of_years=no_of_years)

            feature_vec_team_b_first_batting = np.array(
                pd.DataFrame([feature_dict_team_b_batting_first])[first_innings_selected_column])

            team_b_first_target = first_innings_model.predict(feature_vec_team_b_first_batting)[0]

            feature_dict_team_a_batting_second['target_score'] = team_b_first_target
            feature_vector_team_a_chasing = np.array(pd.DataFrame([feature_dict_team_a_batting_second])[second_innings_selected_column])

            team_b_defend_success_probability = second_innings_model.predict_proba(feature_vector_team_a_chasing)[0][0]
            team_a_chasing_success = second_innings_model.predict(feature_vector_team_a_chasing)[0]

            combined_feature_vector = np.array([target, team_a_defend_success_probability, team_b_first_target,
                                       team_b_defend_success_probability]).reshape(1,-1)

            predict_final_outcome_team_a = combined_model.predict(combined_feature_vector)
            pridict_final_outcome_probability_team_a = combined_model.predict_proba(combined_feature_vector)[0][1]
            pridict_final_outcome_probability_team_b = combined_model.predict_proba(combined_feature_vector)[0][0]

            print("==============================================================")

            print(" For ", team_b, " batting first ")
            print(team_b, " scores ", team_b_first_target)
            print(team_a, " will be able to chase ? ",np.array([int(team_a_chasing_success)]).astype(bool)[0])
            print(team_b, " win probability ", team_b_defend_success_probability)

            print("==============================================================")
            print(" Overall result ")
            print(team_a, " will win ", np.array([int(predict_final_outcome_team_a)]).astype(bool)[0])
            print(team_a, " win probability ", pridict_final_outcome_probability_team_a)
            print(team_b, " win probability ", pridict_final_outcome_probability_team_b)



    if neural and not second_only:
        print("======Neural prediction=============")
        print("Neural option currently not available")


@predict.command()
@click.option('--team_a_xlsx', help='team_a template excel',default='team_a.xlsx')
@click.option('--team_b_xlsx', help='team_b template excel',default='team_b.xlsx')
@click.option('--ref_date', help='date of the match (by default current)')
@click.option('--env', help='which models to use for prediction',default='production')
def optimize(team_a_xlsx,team_b_xlsx,ref_date,env):
    outil.use_model_from(env)
    team_a_df = pd.read_excel(team_a_xlsx)
    team_b_df = pd.read_excel(team_b_xlsx)

    team_a = team_a_df['team'].values[0]
    team_b = team_b_df['team'].values[0]

    location = team_a_df['location'].values[0]
    location_b = team_b_df['location'].values[0]

    team_b_player_df = team_b_df[team_b_df['playing'] == 'Y']
    team_b_player_df = team_b_player_df[['team', 'name', 'position']]

    team_a_bowler_df = team_a_df[(team_a_df['playing'] == 'Y') & (team_a_df['bowler'] == 'Y')]
    team_a_bowler_df = team_a_bowler_df[['team', 'name']]

    if location != location_b:
        raise Exception('location of team A does not match team B')

    if team_b_player_df.shape[0] > 11:
        raise Exception('More than 11 players for team A')


    optimum_runs = get_optimum_run(team_b,team_a,location,team_b_player_df,team_a_bowler_df,ref_date=ref_date)
    print(" the threshold run is ",optimum_runs)
    print("anything below that is chaseable")

if __name__=='__main__':
    predict()
