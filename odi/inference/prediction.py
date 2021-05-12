from odi.model_util import odi_util as outil
from odi.feature_engg import feature_extractor as fe
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

first_innings_emb = True
second_innings_emb = True
first_emb_model = 'team'
second_emb_model = 'team'


def set_embedding_usage():
    global first_innings_emb
    global second_innings_emb
    global first_emb_model
    global second_emb_model

    if os.path.isfile(os.path.join(outil.MODEL_DIR,'inference_config.json')):
        conf = json.load(open(os.path.join(outil.MODEL_DIR,'inference_config.json'),'r'))
        first_innings_emb = conf['first_innings_emb']
        second_innings_emb = conf['second_innings_emb']
        if "first_emb_model" in conf:
            first_emb_model = conf['first_emb_model']

        if "second_emb_model" in conf:
            second_emb_model = conf['second_emb_model']



def predict_first_innings_run(team,opponent,location,
                              team_player_list,opponent_player_list,
                              ref_date=None,no_of_years=None):

    # print(' embedding in first Innings ',first_innings_emb)
    # print(' first_emb_model ', first_emb_model)
    if first_innings_emb:
        if first_emb_model == 'team':
            # print('======Predicting with team========')
            first_innings_model = pickle.load(open(outil.MODEL_DIR+os.sep+outil.FIRST_INNINGS_MODEL,'rb'))
            feature_vector = fe.get_first_innings_feature_embedding_vector(team,opponent,location,
                                                                           team_player_list,opponent_player_list,
                                                                           ref_date=ref_date,no_of_years=no_of_years)
        else:
            # print('======Predicting with adversarial========')
            first_innings_model = pickle.load(open(outil.MODEL_DIR + os.sep + outil.ADVERSARIAL_FIRST_INNINGS, 'rb'))
            feature_vector = fe.get_adversarial_first_innings_feature_vector(team, opponent, location,
                                                                           team_player_list, opponent_player_list,
                                                                           ref_date=ref_date, no_of_years=no_of_years)

    else:

        first_innings_model = pickle.load(open(os.path.join(outil.MODEL_DIR, outil.FIRST_INNINGS_MODEL_BASE), 'rb'))
        feature_vector = fe.get_first_innings_feature_vector(team, opponent, location,\
                                                                            team_player_list, opponent_player_list, \
                                                                            ref_date=ref_date,no_of_years=no_of_years)


    predicted_runs = first_innings_model.predict(feature_vector.reshape(1, -1))[0]

    return predicted_runs


def predict_second_innings_success(target,team,opponent,location,
                              team_player_list,opponent_player_list,
                              ref_date=None,no_of_years=None):

    # print(' using embedding in second innings ',second_innings_emb)
    if second_innings_emb:
        if second_emb_model == 'team':
            second_innings_model = pickle.load(open(outil.MODEL_DIR+os.sep+outil.SECOND_INNINGS_MODEL,'rb'))
            feature_vector = fe.get_second_innings_feature_embedding_vector(target,team,opponent,location,
                                                                           team_player_list,opponent_player_list,
                                                                           ref_date=ref_date,no_of_years=no_of_years)
        else:
            raise Exception("Second innings adversarial model not available")
    else:
        second_innings_model = pickle.load(open(os.path.join(outil.MODEL_DIR, outil.SECOND_INNINGS_MODEL_BASE), 'rb'))
        feature_vector = fe.get_second_innings_feature_vector(target, team, opponent,\
                                                                             location, team_player_list,\
                                                                             opponent_player_list, ref_date=ref_date,no_of_years=no_of_years)

    success = second_innings_model.predict(feature_vector.reshape(1, -1))[0]
    probability = second_innings_model.predict_proba(feature_vector.reshape(1, -1))[0, 1]


    if success == 1:
        return True,probability
    else:
        return False,probability

def predict_match_outcome(team_a,team_b,location,
                          team_a_player_list, team_b_player_list,
                          ref_date=None, no_of_years=None
                          ):

    print('For ',team_a,' as first innings ')

    team = team_a
    opponent = team_b
    team_player_list = team_a_player_list.copy()
    opponent_player_list = team_b_player_list.copy()

    target_by_a = predict_first_innings_run(team,opponent,location,
                                              team_player_list,opponent_player_list,
                                              ref_date=ref_date,no_of_years=no_of_years)

    team = team_b
    opponent = team_a
    team_player_list = team_b_player_list.copy()
    opponent_player_list = team_a_player_list.copy()
    successfully_chase,probability_by_b = predict_second_innings_success(target_by_a,team,opponent,location,
                              team_player_list,opponent_player_list,
                              ref_date=ref_date,no_of_years=no_of_years)


    chase = 'NO'
    if successfully_chase:
        chase = 'YES'
    print('\t',team_a,' will score ',target_by_a)
    print('\t',team_b,' will be able to chase ? ',chase)
    print('\t', team_a, ' win probability ? ', 1-probability_by_b)
    print('\t', team_b, ' win probability ? ', probability_by_b)

    print('For ', team_b, ' as first innings ')

    team = team_b
    opponent = team_a
    team_player_list = team_b_player_list.copy()
    opponent_player_list = team_a_player_list.copy()

    target_by_b = predict_first_innings_run(team, opponent, location,
                                              team_player_list, opponent_player_list,
                                              ref_date=ref_date, no_of_years=no_of_years)

    team = team_a
    opponent = team_b
    team_player_list = team_a_player_list.copy()
    opponent_player_list = team_b_player_list.copy()
    successfully_chase, probability_by_a = predict_second_innings_success(target_by_b, team, opponent, location,
                                                                     team_player_list, opponent_player_list,
                                                                     ref_date=ref_date, no_of_years=no_of_years)

    chase = 'NO'
    if successfully_chase:
        chase = 'YES'
    print('\t', team_b, ' will score ', round(target_by_b))
    print('\t', team_a, ' will be able to chase ? ', chase)
    print('\t', team_b, ' win probability ? ', 1 - probability_by_a)
    print('\t', team_a, ' win probability ? ', probability_by_a)


    print("===Predicting with combined model=====")
    feature_combined = np.array([target_by_a, probability_by_b, target_by_b, probability_by_a]).reshape(1,-1)
    combined_model = pickle.load(open(os.path.join(outil.MODEL_DIR,outil.COMBINED_MODEL),"wb"))

    overall_prediction = combined_model.predict(feature_combined)
    print(overall_prediction)


def get_optimum_run(team,opponent,location,
                    team_player_list,opponent_player_list,
                    ref_date=None, no_of_years=None):

    predicted_run = predict_first_innings_run(team, opponent, location,
                                              team_player_list, opponent_player_list,
                                              ref_date=ref_date, no_of_years=no_of_years)

    successfully_chase, probability = predict_second_innings_success(predicted_run, team, opponent, location,
                                                                     team_player_list, opponent_player_list,
                                                                     ref_date=ref_date, no_of_years=no_of_years)

    if not successfully_chase:
        print(team, 'will be able to score ',predicted_run)
        print(opponent, 'will not be able to chase ')

        print('finding lowest defendable score')

        loop = True
        predicted_run = predicted_run - 5
        while(loop):
            successfully_chase, probability = predict_second_innings_success(predicted_run, team, opponent, location,
                                                                             team_player_list, opponent_player_list,
                                                                             ref_date=ref_date, no_of_years=no_of_years)
            if successfully_chase:
                print('scoring ',predicted_run, ' or below will be chaseable by opponent ')
                break
            elif predicted_run<150:
                print(' Score as low as 150 will also be defendable')
                break
            else:
                predicted_run = predicted_run-5
    else:
        print(team, 'will be able to score ', predicted_run)
        print(opponent, 'will be able to chase ')

        print('finding defendable score')

        loop = True
        predicted_run = predicted_run + 5
        while (loop):
            successfully_chase, probability = predict_second_innings_success(predicted_run, team, opponent, location,
                                                                             team_player_list, opponent_player_list,
                                                                             ref_date=ref_date, no_of_years=no_of_years)
            if not successfully_chase:
                print('scoring ', predicted_run, ' or above will be not be chaseable by opponent ')
                break
            elif predicted_run > 350:
                print(' Score as high as 350 will also be chaseable')
                break
            else:
                predicted_run = predicted_run + 5


def predict_individual_runs(team,opponent,location,
                            team_player_list,opponent_player_list,
                            ref_date=None,no_of_years=None):

    prediction_dict = dict()
    batsman_runs_model = pickle.load(open(os.path.join(outil.DEV_DIR,outil.BATSMAN_RUNS_MODELS),'rb'))
    for bi,batsman in enumerate(team_player_list):
        try:
            feature_vector = fe.get_batsman_features_with_embedding(batsman, bi + 1, opponent_player_list,
                                                   team, opponent, location,ref_date=ref_date)
            predicted_runs = batsman_runs_model.predict(np.array(feature_vector).reshape(1,-1))
            prediction_dict[batsman]=round(predicted_runs[0])
        except Exception as ex:
            print(ex, ' batsman ',batsman,' of ',team)

    return prediction_dict




@click.group()
def predict():
    warnings.filterwarnings('ignore')
    pass

@predict.command()
@click.option('--team_a_xlsx', help='team_a template excel')
@click.option('--team_b_xlsx', help='team_b template excel')
@click.option('--ref_date', help='date of the match (by default current)')
@click.option('--no_of_years', help='no of years for considering trend')
@click.option('--env', help='which models to use for predictio',default='production')
def match(team_a_xlsx,team_b_xlsx,ref_date,no_of_years,env):
    outil.use_model_from(env)
    set_embedding_usage()
    if team_a_xlsx is None:
        team_a_xlsx = 'team_a.xlsx'

    if team_b_xlsx is None:
        team_b_xlsx = 'team_b.xlsx'

    team_a_df = pd.read_excel(team_a_xlsx)
    team_b_df = pd.read_excel(team_b_xlsx)

    team_a = team_a_df[team_a_df['type']=='team']['name'].values[0]
    team_b = team_b_df[team_b_df['type']=='team']['name'].values[0]

    location = team_a_df[team_a_df['type']=='location']['name'].values[0]
    location_b = team_b_df[team_b_df['type']=='location']['name'].values[0]
    team_a_player_list = list(team_a_df[team_a_df['type']=='player']['name'].unique())
    team_b_player_list = list(team_b_df[team_b_df['type'] == 'player']['name'].unique())

    if location != location_b:
        raise Exception('location of team A does not match team B')

    if len(team_a_player_list)>11:
        raise Exception('More than 11 players for team A')

    if len(team_b_player_list)>11:
        raise Exception('More than 11 players for team B')

    print(team_a,' vs ',team_b,' at ',location,'\n\n')

    print(team_a,' players : ')
    print(team_a_player_list,'\n\n')

    print(team_b, ' players : ')
    print(team_b_player_list, '\n\n')

    print('==============RESULTS==============')

    predict_match_outcome(team_a, team_b, location,
                          team_a_player_list, team_b_player_list,
                          ref_date=ref_date, no_of_years=no_of_years
                          )


@predict.command()
@click.option('--innings',required=True,help='innings to predict(first\second\optimize)')
@click.option('--team_xlsx', help='team_a template excel')
@click.option('--opponent_xlsx', help='team_b template excel')
@click.option('--target',type=int)
@click.option('--ref_date', help='team_b template excel')
@click.option('--no_of_years', help='no of years for considering trend')
@click.option('--env', help='which models to use for predictio',default='production')
def team(innings,team_xlsx, opponent_xlsx,target, ref_date, no_of_years,env):
    outil.use_model_from(env)
    set_embedding_usage()
    if innings == 'second' and target is None:
        raise Exception('--target is mandatory for second innings')

    if team_xlsx is None:
        team_xlsx = 'team_a.xlsx'

    if opponent_xlsx is None:
        opponent_xlsx = 'team_b.xlsx'

    team_df = pd.read_excel(team_xlsx)
    opponent_df = pd.read_excel(opponent_xlsx)

    team = team_df[team_df['type']=='team']['name'].values[0]
    opponent = opponent_df[opponent_df['type']=='team']['name'].values[0]

    location = team_df[team_df['type']=='location']['name'].values[0]
    location_b = opponent_df[opponent_df['type']=='location']['name'].values[0]
    team_player_list = list(team_df[team_df['type']=='player']['name'].unique())
    opponent_player_list = list(opponent_df[opponent_df['type'] == 'player']['name'].unique())

    if location != location_b:
        raise Exception('location of team does not match opponent')

    if len(team_player_list)>11:
        raise Exception('More than 11 players for team')

    if len(opponent_player_list)>11:
        raise Exception('More than 11 players for opponent')

    print(team,' vs ',opponent,' at ',location,'\n\n')

    print(team,' players : ')
    print(team_player_list,'\n\n')

    print(opponent, ' players : ')
    print(opponent_player_list, '\n\n')

    print('==============INNINGS PREDICTION==============')

    if innings=='first':
        print('FIRST')
        predicted_run = predict_first_innings_run(team, opponent, location,
                                  team_player_list, opponent_player_list,
                                  ref_date=ref_date, no_of_years=no_of_years)

        print('predicted run ',predicted_run)

    if innings=='second':
        print('SECOND ....chasing ',target)
        success,probability=predict_second_innings_success(target,team, opponent, location,
                                  team_player_list, opponent_player_list,
                                  ref_date=ref_date, no_of_years=no_of_years)

        print('predicted successful chase ', success)
        print('probability of successful chase ', success)

    if innings=='optimize':
        print('OPTIMUM RUN')
        get_optimum_run(team, opponent, location,
                              team_player_list, opponent_player_list,
                              ref_date=ref_date, no_of_years=no_of_years
                              )


@predict.command()
@click.option('--team_a_xlsx', help='team_a template excel')
@click.option('--team_b_xlsx', help='team_b template excel')
@click.option('--ref_date', help='date of the match (by default current)')
@click.option('--no_of_years', help='no of years for considering trend')
@click.option('--env', help='which models to use for predictio',default='production')
def individual_runs(team_a_xlsx,team_b_xlsx,ref_date,no_of_years,env):
    outil.use_model_from(env)
    set_embedding_usage()
    if team_a_xlsx is None:
        team_a_xlsx = 'team_a.xlsx'

    if team_b_xlsx is None:
        team_b_xlsx = 'team_b.xlsx'

    team_a_df = pd.read_excel(team_a_xlsx)
    team_b_df = pd.read_excel(team_b_xlsx)

    team_a = team_a_df[team_a_df['type']=='team']['name'].values[0]
    team_b = team_b_df[team_b_df['type']=='team']['name'].values[0]

    location = team_a_df[team_a_df['type']=='location']['name'].values[0]
    location_b = team_b_df[team_b_df['type']=='location']['name'].values[0]
    team_a_player_list = list(team_a_df[team_a_df['type']=='player']['name'].unique())
    team_b_player_list = list(team_b_df[team_b_df['type'] == 'player']['name'].unique())

    if location != location_b:
        raise Exception('location of team A does not match team B')

    if len(team_a_player_list)>11:
        raise Exception('More than 11 players for team A')

    if len(team_b_player_list)>11:
        raise Exception('More than 11 players for team B')

    print(team_a,' vs ',team_b,' at ',location,'\n\n')

    print(team_a,' players : ')
    print(team_a_player_list,'\n\n')

    print(team_b, ' players : ')
    print(team_b_player_list, '\n\n')

    print('==============RESULTS==============')
    team = team_a
    opponent = team_b
    team_player_list = team_a_player_list.copy()
    opponent_player_list = team_b_player_list.copy()
    print('Individual runs for ', team)

    predicted_runs_dict = predict_individual_runs(team,opponent,location,\
                                                  team_player_list,opponent_player_list,\
                                                  ref_date=ref_date,no_of_years=no_of_years)

    print(json.dumps(predicted_runs_dict,indent=2))

    team = team_b
    opponent = team_a
    team_player_list = team_b_player_list.copy()
    opponent_player_list = team_a_player_list.copy()
    print('Individual runs for ', team)

    predicted_runs_dict = predict_individual_runs(team, opponent, location,\
                                                  team_player_list, opponent_player_list,\
                                                  ref_date=ref_date, no_of_years=no_of_years)

    print(json.dumps(predicted_runs_dict, indent=2))


if __name__=='__main__':
    predict()
