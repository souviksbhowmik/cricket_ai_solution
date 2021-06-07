
from odi.retrain import base_model_architecture as bma
from odi.retrain import create_train_test_ci as ctt
from odi.model_util import odi_util as outil
from odi.evaluation import evaluate as cric_eval
from odi.feature_engg import util as cricutil
from odi.feature_engg import feature_extractor_ci as fec

from tqdm import tqdm
import pandas as pd
from scipy.stats import pearsonr


from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_absolute_error,mean_squared_error,accuracy_score,precision_recall_fscore_support,mean_absolute_percentage_error

from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


import statsmodels.api as sm

import pickle
import os
import numpy as np

import click


def retrain_country_embedding(learning_rate=0.001,epoch = 150,batch_size=10,monitor="mape",mode="train"):
    metrics_map={
        "mape":"val_mean_absolute_percentage_error",
        "mae":"val_mean_absolute_error"
    }

    if not os.path.isdir(outil.CHECKPOINT_DIR):
        os.makedirs(outil.CHECKPOINT_DIR)

    checkpoint_file_name = os.path.join(outil.CHECKPOINT_DIR,outil.TEAM_OPPONENT_LOCATION_EMBEDDING_RUN_MODEL+'_chk.h5')
    team_oh_train_x = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.country_emb_feature_team_oh_train_x), 'rb'))
    opponent_oh_train_x = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.country_emb_feature_opponent_oh_train_x), 'rb'))
    location_oh_train_x = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.country_emb_feature_location_oh_train_x), 'rb'))
    runs_scored_train_y = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.country_emb_feature_runs_scored_train_y), 'rb'))

    team_oh_test_x = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.country_emb_feature_team_oh_test_x), 'rb'))
    opponent_oh_test_x = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.country_emb_feature_opponent_oh_test_x), 'rb'))
    location_oh_test_x = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.country_emb_feature_location_oh_test_x), 'rb'))
    runs_scored_test_y = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.country_emb_feature_runs_scored_test_y), 'rb'))

    team_model, opponent_model, location_model, group_encode_model, runs_model = \
        bma.create_country_embedding_model_2nd(team_oh_train_x.shape[1],\
                                           opponent_oh_train_x.shape[1],\
                                           location_oh_train_x.shape[1])

    runs_model.compile(loss="mean_squared_error", metrics=["mean_absolute_percentage_error", "mean_absolute_error"],
                       optimizer=Adam(learning_rate))

    # load exisitng wiights for tuning
    pretune_train_metrics = None
    pretune_test_metrics = None
    if mode=="tune":
        runs_model = outil.load_keras_model_weights(runs_model,
                                                    os.path.join(outil.DEV_DIR,
                                                                 outil.TEAM_OPPONENT_LOCATION_EMBEDDING_RUN_MODEL)
                                                    )
        pretune_train_metrics = runs_model.evaluate([team_oh_train_x, opponent_oh_train_x, location_oh_train_x], runs_scored_train_y)
        pretune_test_metrics = runs_model.evaluate([team_oh_test_x, opponent_oh_test_x, location_oh_test_x], runs_scored_test_y)

    checkpoint = ModelCheckpoint(checkpoint_file_name, monitor=metrics_map[monitor],
                                 verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    runs_model.fit([team_oh_train_x, opponent_oh_train_x, location_oh_train_x], runs_scored_train_y,
                   validation_data=([team_oh_test_x, opponent_oh_test_x, location_oh_test_x], runs_scored_test_y),
                   epochs=epoch, batch_size=batch_size,
                   callbacks=callbacks_list)

    train_metrics = runs_model.evaluate([team_oh_train_x, opponent_oh_train_x, location_oh_train_x], runs_scored_train_y)
    test_metrics = runs_model.evaluate([team_oh_test_x, opponent_oh_test_x, location_oh_test_x], runs_scored_test_y)

    print('\n\nFINAL METRICS:')
    print(train_metrics)
    print(test_metrics)

    print('\n\nCHECKPOINT METRICS:')
    runs_model = outil.load_keras_model_weights(runs_model,checkpoint_file_name)
    train_metrics = runs_model.evaluate([team_oh_train_x, opponent_oh_train_x, location_oh_train_x],
                                        runs_scored_train_y)
    test_metrics = runs_model.evaluate([team_oh_test_x, opponent_oh_test_x, location_oh_test_x], runs_scored_test_y)
    print(train_metrics)
    print(test_metrics)

    print('\n\nPRETUNED METRICS:')
    print(pretune_train_metrics)
    print(pretune_test_metrics)

    metrics_index = list(metrics_map.keys()).index(monitor) + 1
    if (mode == "train") or \
            (mode == "tune" and test_metrics[metrics_index] < pretune_test_metrics[metrics_index]):

        print("Saving models - (in case of tuning - metrics improved) ")
        outil.store_keras_model(runs_model,os.path.join(outil.DEV_DIR,outil.TEAM_OPPONENT_LOCATION_EMBEDDING_RUN_MODEL))
        outil.store_keras_model(group_encode_model,
                                os.path.join(outil.DEV_DIR, outil.TEAM_OPPONENT_LOCATION_EMBEDDING_MODEL))
        outil.store_keras_model(team_model,
                                os.path.join(outil.DEV_DIR, outil.TEAM_EMBEDDING_MODEL))
        outil.create_model_meta_info_entry('team_opponent_location_embedding',
                                           train_metrics,
                                           test_metrics,
                                           info="metrics is mse, mape, mae(best mape)",
                                           file_list=[
                                               outil.TEAM_OPPONENT_LOCATION_EMBEDDING_RUN_MODEL+'.json',
                                               outil.TEAM_OPPONENT_LOCATION_EMBEDDING_RUN_MODEL + '.h5',
                                               outil.TEAM_OPPONENT_LOCATION_EMBEDDING_MODEL + '.json',
                                               outil.TEAM_OPPONENT_LOCATION_EMBEDDING_MODEL + '.h5'

                                           ])


    else:
        print("Metrics not better than Pre-tune")


def retrain_country_embedding_second(learning_rate=0.001,epoch = 150,batch_size=10,monitor="accuracy",mode="train"):
    metrics_map={
        "accuracy":"val_accuracy",
    }

    if not os.path.isdir(outil.CHECKPOINT_DIR):
        os.makedirs(outil.CHECKPOINT_DIR)

    checkpoint_file_name = os.path.join(outil.CHECKPOINT_DIR,outil.TEAM_OPPONENT_LOCATION_EMBEDDING_RUN_MODEL_2ND+'_chk.h5')
    team_oh_train_x = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.country_emb_feature_2nd_team_oh_train_x), 'rb'))
    opponent_oh_train_x = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.country_emb_feature_2nd_opponent_oh_train_x), 'rb'))
    location_oh_train_x = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.country_emb_feature_2nd_location_oh_train_x), 'rb'))
    # target_oh_train_x = pickle.load(
    #     open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.country_emb_feature_2nd_target_oh_train_x), 'rb'))
    win_train_y = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.country_emb_feature_2nd_win_train_y), 'rb'))

    team_oh_test_x = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.country_emb_feature_2nd_team_oh_test_x), 'rb'))
    opponent_oh_test_x = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.country_emb_feature_2nd_opponent_oh_test_x), 'rb'))
    location_oh_test_x = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.country_emb_feature_2nd_location_oh_test_x), 'rb'))
    # target_oh_test_x = pickle.load(
    #     open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.country_emb_feature_2nd_target_oh_test_x), 'rb'))
    win_test_y = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.country_emb_feature_2nd_win_test_y), 'rb'))

    team_model, opponent_model, location_model, group_encode_model, runs_model = \
        bma.create_country_embedding_model_2nd(team_oh_train_x.shape[1],\
                                           opponent_oh_train_x.shape[1],\
                                           location_oh_train_x.shape[1])

    runs_model.compile(loss="binary_crossentropy", metrics=["accuracy"],
                       optimizer=Adam(learning_rate))

    # load exisitng wiights for tuning
    pretune_train_metrics = None
    pretune_test_metrics = None
    if mode=="tune":
        runs_model = outil.load_keras_model_weights(runs_model,
                                                    os.path.join(outil.DEV_DIR,
                                                                 outil.TEAM_OPPONENT_LOCATION_EMBEDDING_RUN_MODEL)
                                                    )
        pretune_train_metrics = runs_model.evaluate([team_oh_train_x, opponent_oh_train_x, location_oh_train_x], win_train_y)
        pretune_test_metrics = runs_model.evaluate([team_oh_test_x, opponent_oh_test_x, location_oh_test_x], win_test_y)

    checkpoint = ModelCheckpoint(checkpoint_file_name, monitor=metrics_map[monitor],
                                 verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    runs_model.fit([team_oh_train_x, opponent_oh_train_x, location_oh_train_x], win_train_y,
                   validation_data=([team_oh_test_x, opponent_oh_test_x, location_oh_test_x], win_test_y),
                   epochs=epoch, batch_size=batch_size,
                   callbacks=callbacks_list)

    train_metrics = runs_model.evaluate([team_oh_train_x, opponent_oh_train_x, location_oh_train_x], win_train_y)
    test_metrics = runs_model.evaluate([team_oh_test_x, opponent_oh_test_x, location_oh_test_x], win_test_y)

    print('\n\nFINAL METRICS:')
    print(train_metrics)
    print(test_metrics)

    print('\n\nCHECKPOINT METRICS:')
    runs_model = outil.load_keras_model_weights(runs_model,checkpoint_file_name)
    train_metrics = runs_model.evaluate([team_oh_train_x, opponent_oh_train_x, location_oh_train_x],
                                        win_train_y)
    test_metrics = runs_model.evaluate([team_oh_test_x, opponent_oh_test_x, location_oh_test_x], win_test_y)
    print(train_metrics)
    print(test_metrics)

    print('\n\nPRETUNED METRICS:')
    print(pretune_train_metrics)
    print(pretune_test_metrics)

    metrics_index = list(metrics_map.keys()).index(monitor) + 1
    if (mode == "train") or \
            (mode == "tune" and test_metrics[metrics_index] > pretune_test_metrics[metrics_index]):

        print("Saving models - (in case of tuning - metrics improved) ")
        outil.store_keras_model(runs_model,os.path.join(outil.DEV_DIR,outil.TEAM_OPPONENT_LOCATION_EMBEDDING_RUN_MODEL_2ND))
        outil.store_keras_model(group_encode_model,
                                os.path.join(outil.DEV_DIR, outil.TEAM_OPPONENT_LOCATION_EMBEDDING_MODEL_2ND))
        outil.store_keras_model(team_model,
                                os.path.join(outil.DEV_DIR, outil.TEAM_EMBEDDING_MODEL_2ND))
        outil.create_model_meta_info_entry('team_opponent_location_embedding',
                                           train_metrics,
                                           test_metrics,
                                           info="metrics is mse, mape, mae(best mape)",
                                           file_list=[
                                               outil.TEAM_OPPONENT_LOCATION_EMBEDDING_RUN_MODEL_2ND+'.json',
                                               outil.TEAM_OPPONENT_LOCATION_EMBEDDING_RUN_MODEL_2ND + '.h5',
                                               outil.TEAM_OPPONENT_LOCATION_EMBEDDING_MODEL_2ND + '.json',
                                               outil.TEAM_OPPONENT_LOCATION_EMBEDDING_MODEL_2ND + '.h5'

                                           ])


    else:
        print("Metrics not better than Pre-tune")


def retrain_one_shot_multi(learning_rate=0.001,epoch = 150,batch_size=10,monitor="accuracy",mode="train"):
    metrics_map={
        "mape":"val_final_score_mean_absolute_percentage_error",
        "mae":"val_final_score_mean_absolute_error",
        "accuracy":"val_is_win_accuracy:"
    }

    if not os.path.isdir(outil.CHECKPOINT_DIR):
        os.makedirs(outil.CHECKPOINT_DIR)

    checkpoint_file_name = os.path.join(outil.CHECKPOINT_DIR,outil.ONE_SHOT_MULTI_NEURAL+'_chk.h5')
    x1_scaler = StandardScaler()
    x2_scaler = StandardScaler()

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

    cols_1 = pickle.load(open(os.path.join(outil.DEV_DIR, ctt.one_shot_multi_columns_1), 'rb'))
    cols_2 = pickle.load(open(os.path.join(outil.DEV_DIR, ctt.one_shot_multi_columns_2), 'rb'))
    combined_model = bma.one_shot_multi_output_neural(train_x_1.shape[1],train_x_2.shape[1])

    loss = {
               'final_score': 'mean_squared_error',
               'achieved_score': 'mean_squared_error',
               'is_win': 'binary_crossentropy'

            }
    metrics = {
        'final_score': ["mean_absolute_percentage_error", "mean_absolute_error"],
        'achieved_score': ["mean_absolute_percentage_error", "mean_absolute_error"],
        'is_win': 'accuracy'

    }
    loss_weights = {
                       'final_score':4,
                        'achieved_score':4,
                       'is_win': 100


                        #is_win:2000
                    }



    combined_model.compile(loss=loss, metrics=metrics,loss_weights=loss_weights,optimizer=Adam(learning_rate))


    # load exisitng wiights for tuning
    pretune_train_metrics = None
    pretune_test_metrics = None
    if mode=="tune":
        combined_model = outil.load_keras_model_weights(combined_model,
                                                    os.path.join(outil.DEV_DIR,
                                                                 outil.ONE_SHOT_MULTI_NEURAL)
                                                    )
        pretune_train_metrics = combined_model.evaluate([train_x_1, train_x_2], [train_y_1,train_y_3,train_y_2])
        pretune_test_metrics = combined_model.evaluate([test_x_1, test_x_2], [test_y_1,test_y_3,test_y_2])

    checkpoint = ModelCheckpoint(filepath=checkpoint_file_name,
                                 monitor='val_is_win_accuracy',
                                 verbose=1,
                                 save_freq="epoch",
                                 save_best_only=True,
                                 mode='max')
    callbacks_list = [checkpoint]

    combined_model.fit([train_x_1, train_x_2], [train_y_1,train_y_3,train_y_2],
                   validation_data=([test_x_1, test_x_2], [test_y_1,test_y_3,test_y_2]),
                   epochs=epoch, batch_size=batch_size,
                   callbacks=callbacks_list)

    train_metrics = combined_model.evaluate([train_x_1, train_x_2], [train_y_1,train_y_3,train_y_2])
    test_metrics = combined_model.evaluate([test_x_1, test_x_2], [test_y_1,test_y_3,test_y_2])

    print('\n\nFINAL METRICS:')
    print(train_metrics)
    print(test_metrics)

    print('\n\nCHECKPOINT METRICS:')
    combined_model = outil.load_keras_model_weights(combined_model,checkpoint_file_name)
    train_metrics = combined_model.evaluate([train_x_1, train_x_2], [train_y_1,train_y_3,train_y_2])
    test_metrics = combined_model.evaluate([test_x_1, test_x_2], [test_y_1,test_y_3,test_y_2])
    print(train_metrics)
    print(test_metrics)

    print('\n\nPRETUNED METRICS:')
    print(pretune_train_metrics)
    print(pretune_test_metrics)

    metrics_index = 5
    if (mode == "train") or \
            (mode == "tune" and test_metrics[metrics_index] > pretune_test_metrics[metrics_index]):

        print("Saving models - (in case of tuning - metrics improved) ")
        outil.store_keras_model(combined_model,os.path.join(outil.DEV_DIR,outil.ONE_SHOT_MULTI_NEURAL))
        pickle.dump(x1_scaler,open(os.path.join(outil.DEV_DIR,outil.ONE_SHOT_MULTI_SCALER_X1),"wb"))
        pickle.dump(x2_scaler, open(os.path.join(outil.DEV_DIR, outil.ONE_SHOT_MULTI_SCALER_X2), "wb"))
        outil.create_model_meta_info_entry('combined_multi_output',
                                           train_metrics,
                                           test_metrics,
                                           info="metrics is mape, mae, accuracy(best accuracy)",
                                           file_list=[
                                               outil.ONE_SHOT_MULTI_NEURAL+'.json',
                                               outil.ONE_SHOT_MULTI_NEURAL + '.h5',
                                               outil.ONE_SHOT_MULTI_SCALER_X1,
                                               outil.ONE_SHOT_MULTI_SCALER_X2

                                           ])


    else:
        print("Metrics not better than Pre-tune")


def retrain_batsman_embedding(learning_rate=0.001,epoch = 150,batch_size=10,monitor="mape",mode="train"):
    metrics_map={
        "mae":"val_mean_absolute_error"
    }

    if not os.path.isdir(outil.CHECKPOINT_DIR):
        os.makedirs(outil.CHECKPOINT_DIR)

    checkpoint_file_name = os.path.join(outil.CHECKPOINT_DIR,outil.BATSMAN_EMBEDDING_RUN_MODEL+'_chk.h5')
    batsman_oh_train_x = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.batsman_emb_feature_batsman_oh_train_x), 'rb'))
    position_oh_train_x = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.batsman_emb_feature_position_oh_train_x), 'rb'))
    location_oh_train_x = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.batsman_emb_feature_location_oh_train_x), 'rb'))
    opponent_oh_train_x = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.batsman_emb_feature_opponent_oh_train_x), 'rb'))
    runs_scored_train_y = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.batsman_emb_feature_runs_scored_train_y), 'rb'))

    batsman_oh_test_x = pickle.load(
        open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.batsman_emb_feature_batsman_oh_test_x), 'rb'))
    position_oh_test_x = pickle.load(
        open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.batsman_emb_feature_position_oh_test_x), 'rb'))
    location_oh_test_x = pickle.load(
        open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.batsman_emb_feature_location_oh_test_x), 'rb'))
    opponent_oh_test_x = pickle.load(
        open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.batsman_emb_feature_opponent_oh_test_x), 'rb'))
    runs_scored_test_y = pickle.load(
        open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.batsman_emb_feature_runs_scored_test_y), 'rb'))

    batsman_model, position_model, location_model, opposition_model, group_encode_model, runs_model = \
        bma.create_batsman_embedding_model(batsman_oh_train_x.shape[1],\
                                           position_oh_train_x.shape[1],\
                                           location_oh_train_x.shape[1],\
                                           opponent_oh_train_x.shape[1])

    runs_model.compile(loss="mean_squared_error", metrics=["mean_absolute_error"],
                       optimizer=Adam(learning_rate))

    # load exisitng wiights for tuning
    pretune_train_metrics = None
    pretune_test_metrics = None
    if mode=="tune":
        runs_model = outil.load_keras_model_weights(runs_model,
                                                    os.path.join(outil.DEV_DIR,
                                                                 outil.BATSMAN_EMBEDDING_RUN_MODEL)
                                                    )
        pretune_train_metrics = runs_model.evaluate([batsman_oh_train_x, position_oh_train_x, location_oh_train_x, opponent_oh_train_x], runs_scored_train_y)
        pretune_test_metrics = runs_model.evaluate([batsman_oh_test_x, position_oh_test_x, location_oh_test_x, opponent_oh_test_x], runs_scored_test_y)

    checkpoint = ModelCheckpoint(checkpoint_file_name, monitor=metrics_map[monitor],
                                 verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    runs_model.fit([batsman_oh_train_x, position_oh_train_x, location_oh_train_x,opponent_oh_train_x], runs_scored_train_y,
                   validation_data=([batsman_oh_test_x, position_oh_test_x, location_oh_test_x, opponent_oh_test_x], runs_scored_test_y),
                   epochs=epoch, batch_size=batch_size,
                   callbacks=callbacks_list)

    train_metrics = runs_model.evaluate([batsman_oh_train_x, position_oh_train_x, location_oh_train_x, opponent_oh_train_x], runs_scored_train_y)
    test_metrics = runs_model.evaluate([batsman_oh_test_x, position_oh_test_x, location_oh_test_x, opponent_oh_test_x], runs_scored_test_y)

    print('\n\nFINAL METRICS:')
    print(train_metrics)
    print(test_metrics)

    print('\n\nCHECKPOINT METRICS:')
    runs_model = outil.load_keras_model_weights(runs_model,checkpoint_file_name)
    train_metrics = runs_model.evaluate([batsman_oh_train_x, position_oh_train_x, location_oh_train_x, opponent_oh_train_x],
                                        runs_scored_train_y)
    test_metrics = runs_model.evaluate([batsman_oh_test_x, position_oh_test_x, location_oh_test_x, opponent_oh_test_x], runs_scored_test_y)
    print(train_metrics)
    print(test_metrics)

    print('\n\nPRETUNED METRICS:')
    print(pretune_train_metrics)
    print(pretune_test_metrics)

    metrics_index = list(metrics_map.keys()).index(monitor) + 1
    if (mode == "train") or \
            (mode == "tune" and test_metrics[metrics_index] < pretune_test_metrics[metrics_index]):

        print("Saving models - (in case of tuning - metrics improved) ")
        outil.store_keras_model(runs_model,os.path.join(outil.DEV_DIR,outil.BATSMAN_EMBEDDING_RUN_MODEL))
        outil.store_keras_model(group_encode_model, os.path.join(outil.DEV_DIR, outil.BATSMAN_EMBEDDING_MODEL))
        outil.store_keras_model(batsman_model, os.path.join(outil.DEV_DIR, outil.BATSMAN_ONLY_EMBEDDING_MODEL))
        outil.create_model_meta_info_entry('batsman_position_opponent_location_embedding',
                                           train_metrics,
                                           test_metrics,
                                           info="metrics is mse,mae(best mae)",
                                           file_list=[
                                               outil.BATSMAN_EMBEDDING_RUN_MODEL+'.json',
                                               outil.BATSMAN_EMBEDDING_RUN_MODEL + '.h5',
                                               outil.BATSMAN_EMBEDDING_MODEL + '.json',
                                               outil.BATSMAN_EMBEDDING_MODEL + '.h5'

                                           ])

    else:
        print("Metrics not better than Pre-tune")



def retrain_first_innings_base(create_output=True):
    train_x =pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.first_innings_base_train_x), 'rb'))
    train_y =pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.first_innings_base_train_y), 'rb'))

    test_x = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.first_innings_base_test_x), 'rb'))
    test_y = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.first_innings_base_test_y), 'rb'))

    column_list = pickle.load(open(os.path.join(outil.DEV_DIR, ctt.first_innings_base_columns), 'rb'))

    train_df = pd.DataFrame(train_x)
    train_df.columns = column_list
    train_df['runs'] = train_y

    test_df = pd.DataFrame(test_x)
    test_df.columns = column_list
    test_df['runs'] = test_y

    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    pipe = Pipeline([('scaler', StandardScaler()), ('regression', LinearRegression())])
    sfs = SequentialFeatureSelector(pipe, n_features_to_select=10)
    sfs.fit(train_df.drop(columns='runs'), train_df['runs'])

    selected_cols = []
    print("selected columns")
    for idx in np.where(sfs.get_support())[0]:
        print(column_list[idx])
        selected_cols.append(column_list[idx])

    train_x_selected = np.array(train_df[selected_cols])
    test_x_selected = np.array(test_df[selected_cols])

    train_y_selected = np.array(train_df['runs'])
    test_y_selected = np.array(test_df['runs'])

    train_pipe = Pipeline(
        [('scaler', StandardScaler()), ('regression', LinearRegression())])

    train_pipe.fit(train_x_selected, train_y_selected)

    train_predict = train_pipe.predict(train_x_selected)
    test_predict = train_pipe.predict(test_x_selected)



    mape_train = mean_absolute_percentage_error(train_y_selected,train_predict)
    mape_test = mean_absolute_percentage_error(test_y_selected,test_predict)

    mae_train = mean_absolute_error(train_y_selected,train_predict)
    mae_test = mean_absolute_error(test_y_selected,test_predict)

    print("from scikit learn")
    print('metrics train ', mape_train, mae_train)
    print('metrics test ', mape_test, mae_test)


    if create_output:
        pickle.dump(selected_cols,open(os.path.join(outil.DEV_DIR,outil.FIRST_INNINGS_FEATURE_PICKLE),'wb'))
        pickle.dump(train_pipe, open(os.path.join(outil.DEV_DIR, outil.FIRST_INNINGS_MODEL_BASE), 'wb'))
        pickle.dump(list(np.where(sfs.get_support())),
                    open(os.path.join(outil.DEV_DIR, outil.FIRST_INNINGS_SELECTED_COLUMN_INDEX), 'wb'))

        outil.create_model_meta_info_entry('selected_first_innings_features',
                                           (mape_train, mae_train),
                                           (mape_test, mae_test),
                                           info="metrics is mape,mae - selected :"+str(selected_cols),
                                           file_list=[
                                               outil.FIRST_INNINGS_FEATURE_PICKLE,
                                               outil.FIRST_INNINGS_MODEL_BASE,
                                               outil.FIRST_INNINGS_SELECTED_COLUMN_INDEX
                                               ])


def retrain_first_innings():
    train_x =pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.first_innings_train_x), 'rb'))
    train_y =pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.first_innings_train_y), 'rb'))

    test_x = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.first_innings_test_x), 'rb'))
    test_y = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.first_innings_test_y), 'rb'))

    adj_r2=None
    try:
        statsmodel_scaler = StandardScaler()
        train_x_scaled = statsmodel_scaler.fit_transform((train_x))
        model = sm.OLS(train_y, sm.add_constant(train_x_scaled)).fit()

        train_y_predict = model.predict(sm.add_constant(train_x_scaled))
        test_y_predict = model.predict(sm.add_constant(statsmodel_scaler.transform(test_x)))

        mape_train = cric_eval.mape(train_y,train_y_predict)
        mape_test = cric_eval.mape(test_y, test_y_predict)

        mae_train = mean_absolute_error(train_y,train_y_predict)
        mae_test = mean_absolute_error(test_y,test_y_predict)
        adj_r2=model.rsquared_adj

        print(model.summary())
        print("Using stats model")
        print('metrics train ', mape_train, mae_train)
        print('metrics test ', mape_test, mae_test)
        print('adjusted r2 ', adj_r2)
    except Exception as ex:
        print("could not use statsmodel")

    pipe = Pipeline([('scaler', StandardScaler()), ('regression', LinearRegression())])
    pipe.fit(train_x,train_y)

    train_y_predict_lr = pipe.predict(train_x)
    test_y_predict_lr = pipe.predict(test_x)

    mape_train_lr = cric_eval.mape(train_y, train_y_predict_lr)
    mape_test_lr = cric_eval.mape(test_y, test_y_predict_lr)

    mae_train_lr = mean_absolute_error(train_y, train_y_predict_lr)
    mae_test_lr = mean_absolute_error(test_y, test_y_predict_lr)

    mse_train_lr = mean_squared_error(train_y, train_y_predict_lr)
    mse_test_lr = mean_squared_error(test_y, test_y_predict_lr)

    print("from scikit learn")
    print('metrics train ', mape_train_lr, mae_train_lr)
    print('metrics test ', mape_test_lr, mae_test_lr)

    pickle.dump(pipe,open(os.path.join(outil.DEV_DIR,outil.FIRST_INNINGS_MODEL),'wb'))

    outil.create_model_meta_info_entry('first_innings_model',
                                       (mape_train_lr, mae_train_lr,mse_train_lr,adj_r2),
                                       (mape_test_lr, mae_test_lr,mse_test_lr,adj_r2),
                                       info="metrics is mape,mae,mse,adjusted r squared - selected ",
                                       file_list=[
                                           outil.FIRST_INNINGS_MODEL,
                                           ])


def retrain_second_innings_base(create_output=True):
    train_x =pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.second_innings_base_train_x), 'rb'))
    train_y =pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.second_innings_base_train_y), 'rb'))

    test_x = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.second_innings_base_test_x), 'rb'))
    test_y = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.second_innings_base_test_y), 'rb'))

    try:
        statsmodel_scaler = StandardScaler()
        train_x_scaled = statsmodel_scaler.fit_transform((train_x))
        model = sm.Logit(train_y, sm.add_constant(train_x_scaled)).fit()

        train_y_predict = np.round(model.predict(sm.add_constant(train_x_scaled)))
        test_y_predict = np.round(model.predict(sm.add_constant(statsmodel_scaler.transform(test_x))))

        accuracy_train = accuracy_score(train_y,train_y_predict)
        accuracy_test = accuracy_score(test_y, test_y_predict)


        print(model.summary())
        print('metrics train ', accuracy_train)
        print('metrics test ', accuracy_test)
    except:
        print("could nt do statsmodel")

    pipe = Pipeline([('scaler', StandardScaler()), ('logistic_regression', LogisticRegression())])
    #pipe = Pipeline([('scaler', StandardScaler()), ('svm', SVC(gamma='auto',probability=True))])

    pipe.fit(train_x,train_y)

    train_y_predict_lr = pipe.predict(train_x)
    test_y_predict_lr = pipe.predict(test_x)

    accuracy_train_lr = accuracy_score(train_y, train_y_predict_lr)
    accuracy_test_lr = accuracy_score(test_y, test_y_predict_lr)


    print("from scikit learn")
    print('metrics train ', accuracy_train_lr)
    print('metrics test ', accuracy_test_lr)

    #print(np.where(np.array(model.pvalues) < 0.05))
    #selected_feature_index = list(np.where(np.array(model.pvalues) < 0.05)[0])
    selected_feature_index=list(range(train_x.shape[1]+1))
    print("selected indices including bias ",selected_feature_index)
    # need to substract 1 to exclude bias index consideration:
    column_list = pickle.load(open(os.path.join(outil.DEV_DIR, ctt.second_innings_base_columns), 'rb'))

    selected_column = list()
    selected_index = list()
    for index in selected_feature_index:
        if index == 0:
            continue
        selection_index = index-1
        selected_column.append(column_list[selection_index])
        selected_index.append(selection_index)

    print('Selected columns ',selected_column)

    new_train_x = train_x[:,np.array(selected_index)]
    new_test_x = test_x[:,np.array(selected_index)]
    print("selected_index ",selected_index)
    print("new train_x ",new_train_x.shape)
    print("new test x ", new_test_x.shape)
    pipe_new = Pipeline([('scaler', StandardScaler()), ('logistic_regression', LogisticRegression())])
    #pipe_new = Pipeline([('scaler', StandardScaler()), ('svm', SVC(gamma='auto', probability=True,kernel='poly'))])
    pipe_new.fit(new_train_x,train_y)

    new_train_y_predict = pipe_new.predict(new_train_x)
    new_test_y_predict = pipe_new.predict(new_test_x)

    accuracy_train_new = accuracy_score(train_y, new_train_y_predict)
    accuracy_test_new = accuracy_score(test_y, new_test_y_predict)

    print("After selecting columns by p-value")
    print("metrics train ",accuracy_train_new)
    print("metrics test ", accuracy_test_new)

    if create_output:
        pickle.dump(selected_column,open(os.path.join(outil.DEV_DIR,outil.SECOND_INNINGS_FEATURE_PICKLE),'wb'))
        pickle.dump(pipe_new, open(os.path.join(outil.DEV_DIR, outil.SECOND_INNINGS_MODEL_BASE), 'wb'))
        pickle.dump(selected_index,
                    open(os.path.join(outil.DEV_DIR, outil.SECOND_INNINGS_SELECTED_COLUMN_INDEX), 'wb'))

        outil.create_model_meta_info_entry('selected_second_innings_features',
                                           accuracy_train_new,
                                           accuracy_test_new,
                                           info="metrics is accuracy - selected :"+str(selected_column),
                                           file_list=[
                                               outil.SECOND_INNINGS_FEATURE_PICKLE,
                                               outil.SECOND_INNINGS_MODEL_BASE,
                                               outil.SECOND_INNINGS_SELECTED_COLUMN_INDEX
                                               ])

def retrain_second_innings():
    train_x =pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.second_innings_train_x), 'rb'))
    train_y =pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.second_innings_train_y), 'rb'))

    test_x = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.second_innings_test_x), 'rb'))
    test_y = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.second_innings_test_y), 'rb'))

    statsmodel_scaler = StandardScaler()
    train_x_scaled = statsmodel_scaler.fit_transform((train_x))
    try:
        model = sm.Logit(train_y, sm.add_constant(train_x_scaled)).fit()

        train_y_predict = np.round(model.predict(sm.add_constant(train_x_scaled)))
        test_y_predict = np.round(model.predict(sm.add_constant(statsmodel_scaler.transform(test_x))))

        accuracy_train = accuracy_score(train_y,train_y_predict)
        accuracy_test = accuracy_score(test_y, test_y_predict)


        print(model.summary())
        print('Using stats model')

        print('metrics train ', accuracy_train)
        print('metrics test ', accuracy_test)

    except Exception as ex:
        print(ex)
        print("Statsmodel could not be evaluated")

    pipe = Pipeline([('scaler', StandardScaler()), ('logistic_regression', LogisticRegression())])
    pipe.fit(train_x,train_y)

    train_y_predict_lr = pipe.predict(train_x)
    test_y_predict_lr = pipe.predict(test_x)

    accuracy_train_lr = accuracy_score(train_y, train_y_predict_lr)
    accuracy_test_lr = accuracy_score(test_y, test_y_predict_lr)


    print("from scikit learn")
    print('metrics train ', accuracy_train_lr)
    print('metrics test ', accuracy_test_lr)


    pickle.dump(pipe,open(os.path.join(outil.DEV_DIR,outil.SECOND_INNINGS_MODEL),'wb'))

    outil.create_model_meta_info_entry('second_innings_model',
                                       accuracy_train_lr,
                                       accuracy_test_lr,
                                       info="metrics is accuracy",
                                       file_list=[
                                           outil.SECOND_INNINGS_MODEL,
                                           ])


def select_all_columns(innings):
    if innings=='first':
        column_list = pickle.load(open(os.path.join(outil.DEV_DIR, ctt.first_innings_base_columns), 'rb'))
        pickle.dump(column_list, open(os.path.join(outil.DEV_DIR, outil.FIRST_INNINGS_FEATURE_PICKLE), 'wb'))
        outil.create_model_meta_info_entry('selected_first_innings_features',
                                           (0,0.0),
                                           (0,0,0),
                                           info="all columns",
                                           file_list=[
                                               outil.FIRST_INNINGS_FEATURE_PICKLE,
                                           ])
    else:
        column_list = pickle.load(open(os.path.join(outil.DEV_DIR, ctt.second_innings_base_columns), 'rb'))
        pickle.dump(column_list, open(os.path.join(outil.DEV_DIR, outil.SECOND_INNINGS_FEATURE_PICKLE), 'wb'))
        outil.create_model_meta_info_entry('selected_second_innings_features',
                                           (0, 0.0),
                                           (0,0,0),
                                           info="all columns",
                                           file_list=[
                                               outil.SECOND_INNINGS_FEATURE_PICKLE,
                                           ])


def retrain_batsman_runs():
    train_x =pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.batsman_runs_train_x), 'rb'))
    train_y =pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.batsman_runs_train_y), 'rb'))

    test_x = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.batsman_runs_test_x), 'rb'))
    test_y = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.batsman_runs_test_y), 'rb'))

    statsmodel_scaler = StandardScaler()
    train_x_scaled = statsmodel_scaler.fit_transform((train_x))
    model = sm.OLS(train_y, sm.add_constant(train_x_scaled)).fit()

    train_y_predict = model.predict(sm.add_constant(train_x_scaled))
    test_y_predict = model.predict(sm.add_constant(statsmodel_scaler.transform(test_x)))

    mape_train = cric_eval.mape(train_y,train_y_predict)
    mape_test = cric_eval.mape(test_y, test_y_predict)

    mae_train = mean_absolute_error(train_y,train_y_predict)
    mae_test = mean_absolute_error(test_y,test_y_predict)

    print(model.summary())
    print("Using stats model")
    print('metrics train ', mape_train, mae_train)
    print('metrics test ', mape_test, mae_test)
    print('adjusted r square ',model.rsquared_adj)

    pipe = Pipeline([('scaler', StandardScaler()), ('regression', LinearRegression())])
    pipe.fit(train_x,train_y)

    train_y_predict_lr = pipe.predict(train_x)
    test_y_predict_lr = pipe.predict(test_x)

    mape_train_lr = cric_eval.mape(train_y, train_y_predict_lr)
    mape_test_lr = cric_eval.mape(test_y, test_y_predict_lr)

    mae_train_lr = mean_absolute_error(train_y, train_y_predict_lr)
    mae_test_lr = mean_absolute_error(test_y, test_y_predict_lr)

    mse_train_lr = mean_squared_error(train_y, train_y_predict_lr)
    mse_test_lr = mean_squared_error(test_y, test_y_predict_lr)

    print("from scikit learn")
    print('metrics train ',  mae_train_lr)
    print('metrics test ',  mae_test_lr)

    pickle.dump(pipe,open(os.path.join(outil.DEV_DIR,outil.BATSMAN_RUNS_MODELS),'wb'))

    outil.create_model_meta_info_entry('batsman_runs',
                                       (mae_train_lr,mse_train_lr,model.rsquared_adj),
                                       ( mae_test_lr,mse_test_lr,model.rsquared_adj),
                                       info="metrics is mae, mse, adjusted r square- selected ",
                                       file_list=[
                                           outil.BATSMAN_RUNS_MODELS,
                                           ])


def retrain_adversarial(learning_rate=0.001,epoch = 150,batch_size=10,monitor="loss",mode="train"):
    metrics_map={
        "loss": "val_loss",
        "mae":"val_mean_absolute_error"
    }

    if not os.path.isdir(outil.CHECKPOINT_DIR):
        os.makedirs(outil.CHECKPOINT_DIR)

    checkpoint_file_name = os.path.join(outil.CHECKPOINT_DIR,outil.ADVERSARIAL_RUN_MODEL+'_chk.h5')
    batsman_oh_train_x = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.adversarial_feature_batsman_oh_train_x), 'rb'))
    position_oh_train_x = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.adversarial_feature_position_oh_train_x), 'rb'))
    location_oh_train_x = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.adversarial_feature_location_oh_train_x), 'rb'))
    bowler_oh_train_x = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.adversarial_feature_bowler_oh_train_x), 'rb'))
    runs_scored_train_y = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.adversarial_feature_runs_scored_train_y), 'rb'))

    batsman_oh_test_x = pickle.load(
        open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.adversarial_feature_batsman_oh_test_x), 'rb'))
    position_oh_test_x = pickle.load(
        open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.adversarial_feature_position_oh_test_x), 'rb'))
    location_oh_test_x = pickle.load(
        open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.adversarial_feature_location_oh_test_x), 'rb'))
    bowler_oh_test_x = pickle.load(
        open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.adversarial_feature_bowler_oh_test_x), 'rb'))
    runs_scored_test_y = pickle.load(
        open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.adversarial_feature_runs_scored_test_y), 'rb'))

    batsman_model, position_model, location_model, bowler_model, group_encode_model, runs_model = \
        bma.create_batsman_embedding_model(batsman_oh_train_x.shape[1],\
                                           position_oh_train_x.shape[1],\
                                           location_oh_train_x.shape[1],\
                                           bowler_oh_train_x.shape[1])

    runs_model.compile(loss="mean_squared_error", metrics=["mean_absolute_error"],
                       optimizer=Adam(learning_rate))

    # load exisitng wiights for tuning
    pretune_train_metrics = None
    pretune_test_metrics = None
    if mode=="tune":
        runs_model = outil.load_keras_model_weights(runs_model,
                                                    os.path.join(outil.DEV_DIR,
                                                                 outil.ADVERSARIAL_RUN_MODEL)
                                                    )
        pretune_train_metrics = runs_model.evaluate([batsman_oh_train_x, position_oh_train_x, location_oh_train_x, bowler_oh_train_x], runs_scored_train_y)
        pretune_test_metrics = runs_model.evaluate([batsman_oh_test_x, position_oh_test_x, location_oh_test_x, bowler_oh_test_x], runs_scored_test_y)

    checkpoint = ModelCheckpoint(checkpoint_file_name, monitor=metrics_map[monitor],
                                 verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    runs_model.fit([batsman_oh_train_x, position_oh_train_x, location_oh_train_x,bowler_oh_train_x], runs_scored_train_y,
                   validation_data=([batsman_oh_test_x, position_oh_test_x, location_oh_test_x, bowler_oh_test_x], runs_scored_test_y),
                   epochs=epoch, batch_size=batch_size,
                   callbacks=callbacks_list)

    train_metrics = runs_model.evaluate([batsman_oh_train_x, position_oh_train_x, location_oh_train_x, bowler_oh_train_x], runs_scored_train_y)
    test_metrics = runs_model.evaluate([batsman_oh_test_x, position_oh_test_x, location_oh_test_x, bowler_oh_test_x], runs_scored_test_y)

    print('\n\nFINAL METRICS:')
    print(train_metrics)
    print(test_metrics)

    print('\n\nCHECKPOINT METRICS:')
    runs_model = outil.load_keras_model_weights(runs_model,checkpoint_file_name)
    train_metrics = runs_model.evaluate([batsman_oh_train_x, position_oh_train_x, location_oh_train_x, bowler_oh_train_x],
                                        runs_scored_train_y)
    test_metrics = runs_model.evaluate([batsman_oh_test_x, position_oh_test_x, location_oh_test_x, bowler_oh_test_x], runs_scored_test_y)
    print(train_metrics)
    print(test_metrics)

    print('\n\nPRETUNED METRICS:')
    print(pretune_train_metrics)
    print(pretune_test_metrics)

    metrics_index = list(metrics_map.keys()).index(monitor)
    if (mode == "train") or \
            (mode == "tune" and test_metrics[metrics_index] < pretune_test_metrics[metrics_index]):

        print("Saving models - (in case of tuning - metrics improved) ")
        outil.store_keras_model(runs_model,os.path.join(outil.DEV_DIR,outil.ADVERSARIAL_RUN_MODEL))
        outil.store_keras_model(group_encode_model, os.path.join(outil.DEV_DIR, outil.ADVERSARIAL_MODEL))
        outil.store_keras_model(batsman_model, os.path.join(outil.DEV_DIR, outil.ADVERSARIAL_BATSMAN_MODEL))
        outil.store_keras_model(bowler_model, os.path.join(outil.DEV_DIR, outil.ADVERSARIAL_BOWLER_MODEL))
        outil.store_keras_model(location_model, os.path.join(outil.DEV_DIR, outil.ADVERSARIAL_LOCATION_MODEL))
        outil.store_keras_model(position_model, os.path.join(outil.DEV_DIR, outil.ADVERSARIAL_POSITION_MODEL))
        outil.create_model_meta_info_entry('batsman_position_opponent_location_embedding',
                                           train_metrics,
                                           test_metrics,
                                           info="metrics is mse,mae(best mae)",
                                           file_list=[
                                               outil.ADVERSARIAL_RUN_MODEL+'.json',
                                               outil.ADVERSARIAL_RUN_MODEL + '.h5',
                                               outil.ADVERSARIAL_MODEL + '.json',
                                               outil.ADVERSARIAL_MODEL + '.h5',
                                               outil.ADVERSARIAL_BATSMAN_MODEL + '.json',
                                               outil.ADVERSARIAL_BATSMAN_MODEL + '.h5',
                                               outil.ADVERSARIAL_BOWLER_MODEL + '.json',
                                               outil.ADVERSARIAL_BOWLER_MODEL + '.h5',
                                               outil.ADVERSARIAL_LOCATION_MODEL + '.json',
                                               outil.ADVERSARIAL_LOCATION_MODEL + '.h5',
                                               outil.ADVERSARIAL_POSITION_MODEL + '.json',
                                               outil.ADVERSARIAL_POSITION_MODEL + '.h5'

                                           ])

    else:
        print("Metrics not better than Pre-tune")


## still at experiment level
def adversarial_first_innings_runs():
    train_x =pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.adversarial_first_innings_train_x), 'rb'))
    train_y =pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.adversarial_first_innings_train_y), 'rb'))

    test_x = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.adversarial_first_innings_test_x), 'rb'))
    test_y = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.adversarial_first_innings_test_y), 'rb'))

    statsmodel_scaler = StandardScaler()
    train_x_scaled = statsmodel_scaler.fit_transform((train_x))
    model = sm.OLS(train_y, sm.add_constant(train_x_scaled)).fit()

    train_y_predict = model.predict(sm.add_constant(train_x_scaled))
    test_y_predict = model.predict(sm.add_constant(statsmodel_scaler.transform(test_x)))

    mape_train = cric_eval.mape(train_y,train_y_predict)
    mape_test = cric_eval.mape(test_y, test_y_predict)

    mae_train = mean_absolute_error(train_y,train_y_predict)
    mae_test = mean_absolute_error(test_y,test_y_predict)
    adj_r2 = model.rsquared_adj

    print(model.summary())
    print("Using stats model")
    print('metrics train ', mape_train, mae_train)
    print('metrics test ', mape_test, mae_test)
    print('adjusted R square ', adj_r2)

    pipe = Pipeline([('scaler', StandardScaler()), ('regression', LinearRegression())])
    pipe.fit(train_x,train_y)

    train_y_predict_lr = pipe.predict(train_x)
    test_y_predict_lr = pipe.predict(test_x)

    mape_train_lr = cric_eval.mape(train_y, train_y_predict_lr)
    mape_test_lr = cric_eval.mape(test_y, test_y_predict_lr)

    mae_train_lr = mean_absolute_error(train_y, train_y_predict_lr)
    mae_test_lr = mean_absolute_error(test_y, test_y_predict_lr)

    mse_train_lr = mean_squared_error(train_y, train_y_predict_lr)
    mse_test_lr = mean_squared_error(test_y, test_y_predict_lr)

    print("from scikit learn")
    print('metrics train ',  mape_train_lr,mae_train_lr)
    print('metrics test ',  mape_test_lr,mae_test_lr)

    pickle.dump(pipe,open(os.path.join(outil.DEV_DIR,outil.ADVERSARIAL_FIRST_INNINGS),'wb'))

    outil.create_model_meta_info_entry('adversarial_first_innings_runs',
                                       (mape_train_lr,mae_train_lr,mse_train_lr,adj_r2),
                                       (mape_test_lr,mae_test_lr,mse_test_lr,adj_r2),
                                       info="metrics is mape,mae, mse- selected,adjusted r square ",
                                       file_list=[
                                           outil.ADVERSARIAL_FIRST_INNINGS,
                                           ])


def retrain_first_innings_base_neural(learning_rate=0.001,epoch = 150,batch_size=10,monitor="mape",mode="train"):
    metrics_map = {
        "mape": "val_mean_absolute_percentage_error",
        "mae": "val_mean_absolute_error"
    }

    if not os.path.isdir(outil.CHECKPOINT_DIR):
        os.makedirs(outil.CHECKPOINT_DIR)

    checkpoint_file_name = os.path.join(outil.CHECKPOINT_DIR,
                                        outil.FIRST_INNINGS_REGRESSION_NEURAL + '_chk.h5')
    train_x =pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.first_innings_base_train_x), 'rb'))
    train_y =pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.first_innings_base_train_y), 'rb'))

    test_x = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.first_innings_base_test_x), 'rb'))
    test_y = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.first_innings_base_test_y), 'rb'))
    column_list = pickle.load(open(os.path.join(outil.DEV_DIR, ctt.first_innings_base_columns), 'rb'))

    train_df = pd.DataFrame(train_x)
    train_df.columns = column_list
    train_df['runs'] = train_y

    test_df = pd.DataFrame(test_x)
    test_df.columns = column_list
    test_df['runs'] = test_y

    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    train_x = np.array(train_df[column_list])
    train_y = np.array(train_df['runs'])

    test_x = np.array(test_df[column_list])
    test_y = np.array(test_df['runs'])

    neural_sclaer = StandardScaler()
    train_x_scaled=neural_sclaer.fit_transform((train_x))
    test_x_scaled = neural_sclaer.transform(test_x)

    runs_model = bma.create_dense_regression_model(train_x_scaled.shape[1])

    runs_model.compile(loss="mean_squared_error", metrics=["mean_absolute_percentage_error", "mean_absolute_error"],
                       optimizer=Adam(learning_rate))

    # load exisitng wiights for tuning
    pretune_train_metrics = None
    pretune_test_metrics = None
    if mode == "tune":
        runs_model = outil.load_keras_model_weights(runs_model,
                                                    os.path.join(outil.DEV_DIR,
                                                                 outil.FIRST_INNINGS_REGRESSION_NEURAL)
                                                    )
        pretune_train_metrics = runs_model.evaluate([train_x_scaled],train_y)
        pretune_test_metrics = runs_model.evaluate([test_x_scaled],test_y)

    checkpoint = ModelCheckpoint(checkpoint_file_name, monitor=metrics_map[monitor],
                                 verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    runs_model.fit([train_x_scaled], train_y,
                   validation_data=([test_x_scaled], test_y),
                   epochs=epoch, batch_size=batch_size,
                   callbacks=callbacks_list)

    train_metrics = runs_model.evaluate([train_x_scaled],train_y)
    test_metrics = runs_model.evaluate([test_x_scaled],test_y)

    print('\n\nFINAL METRICS:')
    print(train_metrics)
    print(test_metrics)

    print('\n\nCHECKPOINT METRICS:')
    runs_model = outil.load_keras_model_weights(runs_model, checkpoint_file_name)
    train_metrics = runs_model.evaluate([train_x_scaled],train_y)
    test_metrics = runs_model.evaluate([test_x_scaled],test_y)
    print(train_metrics)
    print(test_metrics)

    print('\n\nPRETUNED METRICS:')
    print(pretune_train_metrics)
    print(pretune_test_metrics)

    metrics_index = list(metrics_map.keys()).index(monitor) + 1
    if (mode == "train") or \
            (mode == "tune" and test_metrics[metrics_index] < pretune_test_metrics[metrics_index]):

        print("Saving models - (in case of tuning - metrics improved) ")
        outil.store_keras_model(runs_model,
                                os.path.join(outil.DEV_DIR, outil.FIRST_INNINGS_REGRESSION_NEURAL))
        outil.create_model_meta_info_entry('first_innings_regression_neural',
                                           train_metrics,
                                           test_metrics,
                                           info="metrics is mse, mape, mae(best mape)",
                                           file_list=[
                                               outil.FIRST_INNINGS_REGRESSION_NEURAL + '.json',
                                               outil.FIRST_INNINGS_REGRESSION_NEURAL + '.h5'
                                           ])


    else:
        print("Metrics not better than Pre-tune")


def retrain_one_shot_neural(learning_rate=0.001,epoch = 150,batch_size=10,monitor="accuracy",mode="train"):
    metrics_map = {
        "accuracy": "val_accuracy:",
    }

    if not os.path.isdir(outil.CHECKPOINT_DIR):
        os.makedirs(outil.CHECKPOINT_DIR)

    checkpoint_file_name = os.path.join(outil.CHECKPOINT_DIR,
                                        outil.ONE_SHOT_NEURAL + '_chk.h5')
    train_x =pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.one_shot_train_x), 'rb'))
    train_y =pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.one_shot_train_y), 'rb'))

    test_x = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.one_shot_test_x), 'rb'))
    test_y = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.one_shot_test_y), 'rb'))
    column_list = pickle.load(open(os.path.join(outil.DEV_DIR, ctt.one_shot_columns), 'rb'))

    train_df = pd.DataFrame(train_x)
    train_df.columns = column_list
    train_df['team_a_win'] = train_y

    test_df = pd.DataFrame(test_x)
    test_df.columns = column_list
    test_df['team_a_win'] = test_y

    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    train_x = np.array(train_df[column_list])
    train_y = np.array(train_df['team_a_win'])

    test_x = np.array(test_df[column_list])
    test_y = np.array(test_df['team_a_win'])

    neural_sclaer = StandardScaler()
    train_x_scaled=neural_sclaer.fit_transform((train_x))
    test_x_scaled = neural_sclaer.transform(test_x)

    win_model = bma.create_dense_classification_model(train_x_scaled.shape[1])

    win_model.compile(loss="binary_crossentropy", metrics=['accuracy'],
                       optimizer=Adam(learning_rate))

    # load exisitng wiights for tuning
    pretune_train_metrics = None
    pretune_test_metrics = None
    if mode == "tune":
        win_model = outil.load_keras_model_weights(win_model,
                                                    os.path.join(outil.DEV_DIR,
                                                                 outil.ONE_SHOT_NEURAL)
                                                    )
        pretune_train_metrics = win_model.evaluate([train_x_scaled],train_y)
        pretune_test_metrics = win_model.evaluate([test_x_scaled],test_y)

    checkpoint = ModelCheckpoint(checkpoint_file_name, monitor=metrics_map[monitor],
                                 verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    win_model.fit([train_x_scaled], train_y,
                   validation_data=([test_x_scaled], test_y),
                   epochs=epoch, batch_size=batch_size,
                   callbacks=callbacks_list)

    train_metrics = win_model.evaluate([train_x_scaled],train_y)
    test_metrics = win_model.evaluate([test_x_scaled],test_y)

    print('\n\nFINAL METRICS:')
    print(train_metrics)
    print(test_metrics)

    print('\n\nCHECKPOINT METRICS:')
    win_model = outil.load_keras_model_weights(win_model, checkpoint_file_name)
    train_metrics = win_model.evaluate([train_x_scaled],train_y)
    test_metrics = win_model.evaluate([test_x_scaled],test_y)
    print(train_metrics)
    print(test_metrics)

    print('\n\nPRETUNED METRICS:')
    print(pretune_train_metrics)
    print(pretune_test_metrics)

    metrics_index = list(metrics_map.keys()).index(monitor) + 1
    if (mode == "train") or \
            (mode == "tune" and test_metrics[metrics_index] > pretune_test_metrics[metrics_index]):

        print("Saving models - (in case of tuning - metrics improved) ")
        outil.store_keras_model(win_model,
                                os.path.join(outil.DEV_DIR, outil.ONE_SHOT_NEURAL))
        outil.create_model_meta_info_entry('first_innings_one_shot_neural',
                                           train_metrics,
                                           test_metrics,
                                           info="metrics is mse, mape, mae(best mape)",
                                           file_list=[
                                               outil.ONE_SHOT_NEURAL + '.json',
                                               outil.ONE_SHOT_NEURAL + '.h5'
                                           ])


    else:
        print("Metrics not better than Pre-tune")


def retrain_combined_innings(first_innings_emb=True,second_innings_emb=True):
    train_x = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, "fi_" + str(first_innings_emb) + "_si_" + str(
        second_innings_emb) + "_" + ctt.combined_train_x), 'rb'))
    train_y = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, "fi_" + str(first_innings_emb) + "_si_" + str(
        second_innings_emb) + "_" + ctt.combined_train_y), 'rb'))

    test_x = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, "fi_" + str(first_innings_emb) + "_si_" + str(
        second_innings_emb) + "_" + ctt.combined_test_x), 'rb'))
    test_y = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, "fi_" + str(first_innings_emb) + "_si_" + str(
        second_innings_emb) + "_" + ctt.combined_test_y), 'rb'))

    statsmodel_scaler = StandardScaler()
    train_x_scaled = statsmodel_scaler.fit_transform((train_x))
    try:
        model = sm.Logit(train_y, sm.add_constant(train_x_scaled)).fit()

        train_y_predict = np.round(model.predict(sm.add_constant(train_x_scaled)))
        test_y_predict = np.round(model.predict(sm.add_constant(statsmodel_scaler.transform(test_x))))

        accuracy_train = accuracy_score(train_y,train_y_predict)
        accuracy_test = accuracy_score(test_y, test_y_predict)


        print(model.summary())
        print('Using stats model')

        print('metrics train ', accuracy_train)
        print('metrics test ', accuracy_test)

    except Exception as ex:
        print(ex)
        print("Statsmodel could not be evaluated")

    pipe = Pipeline([('scaler', StandardScaler()), ('logistic_regression', LogisticRegression())])
    pipe.fit(train_x,train_y)

    train_y_predict_lr = pipe.predict(train_x)
    test_y_predict_lr = pipe.predict(test_x)

    accuracy_train_lr = accuracy_score(train_y, train_y_predict_lr)
    precision_train,recall_train,fscore_train,_ = precision_recall_fscore_support(train_y, train_y_predict_lr,average="binary")
    accuracy_test_lr = accuracy_score(test_y, test_y_predict_lr)
    precision_test, recall_test, fscore_test, _ = precision_recall_fscore_support(test_y, test_y_predict_lr,average="binary")

    inermediate_predict = (np.round(test_x[:,1])!=1)*1
    accuracy_test_intermediate = accuracy_score(test_y, inermediate_predict)
    precision_test_intermediate, recall_test_intermediate, fscore_test_intermediate, _ = precision_recall_fscore_support(test_y, inermediate_predict,
                                                                                  average="binary")
    print("from scikit learn")
    print('metrics train ', accuracy_train_lr,precision_train,recall_train,fscore_train)
    print('metrics test ', accuracy_test_lr,precision_test, recall_test, fscore_test)
    print('metrics test intermediate ', accuracy_test_intermediate, precision_test_intermediate, recall_test_intermediate, fscore_test_intermediate)
    print("data volume train ",train_x.shape[0])
    print("data volume test ", test_x.shape[0])

    pickle.dump(pipe,open(os.path.join(outil.DEV_DIR,outil.COMBINED_MODEL),'wb'))

    pickle.dump(pipe, open(os.path.join(outil.DEV_DIR, "fi_" + str(first_innings_emb) + "_si_" + str(
        second_innings_emb) + "_" + outil.COMBINED_MODEL), 'wb'))

    outil.create_model_meta_info_entry(
        'combined_model_' + "fi_" + str(first_innings_emb) + "_si_" + str(second_innings_emb),
        (accuracy_train_lr, precision_train, recall_train, fscore_train),
        (accuracy_test_lr, precision_test, recall_test, fscore_test,accuracy_test_intermediate, precision_test_intermediate, recall_test_intermediate, fscore_test_intermediate),
        info="metrics is accuracy,precision, recall,fscore and intemediate accuracy,precision, recall,fscore",
        file_list=[
            outil.COMBINED_MODEL,
        ])



def score_correlation(start_date,end_date,first_innings_select_count,second_innings_select_count):

    start_dt = cricutil.str_to_date_time(start_date)
    end_dt = cricutil.str_to_date_time(end_date)

    match_list_df = cricutil.read_csv_with_date('data/csv_load/cricinfo_match_list.csv')
    batting_df = cricutil.read_csv_with_date('data/csv_load/cricinfo_batting.csv')
    match_list_df = match_list_df[(match_list_df['date'] >= start_dt) & (match_list_df['date'] < end_dt)]

    match_id_list = list(match_list_df['match_id'].unique())

    dict_list = []
    for match_id in tqdm(match_id_list):
        for innings_type in ['first', 'second']:
            runs = match_list_df[match_list_df["match_id"] == match_id].iloc[0][innings_type + "_innings_run"]
            team = match_list_df[match_list_df["match_id"] == match_id].iloc[0][innings_type + "_innings"]
            winner = match_list_df[match_list_df["match_id"] == match_id].iloc[0]['winner']
            is_win = (team == winner) * 1
            date = match_list_df['date'].iloc[0]
            ref_date = date.to_pydatetime()
            player_list_df = batting_df[
                (batting_df['match_id'] == match_id) & (batting_df['batting_innings'] == innings_type)]

            player_list_df = player_list_df[['team', 'name', 'position']]
            try:
                entry = fec.get_batsman_score_features(player_list_df,ref_date=ref_date)
                entry["innings_type"] = innings_type
                entry["runs"] = runs
                entry["is_win"] = is_win
                dict_list.append(entry)
            except Exception as ex:
                #print("skipped due to  ",ex)
                pass

    score_df = pd.DataFrame(dict_list)
    score_df.dropna(inplace=True)
    print("no of instances = ",score_df.shape[0])

    all_cols = list(score_df.columns)

    score_df_first = score_df[score_df['innings_type'] == 'first']
    target = list(score_df_first['runs'])
    displaye_list_first = []
    for col_idx in range(len(all_cols) - 3):
        l = list(score_df_first[all_cols[col_idx]])
        corr, p_value = pearsonr(l, target)
        # print(all_cols[col_idx],'\t\t\t\t',corr,'\t',p_value)
        display_dict = {
            'col': all_cols[col_idx],
            'cor': corr,
            'p_value': p_value
        }

        displaye_list_first.append(display_dict)

    score_importance_first = pd.DataFrame(displaye_list_first).sort_values(['cor'], ascending=False)
    print("First innings correlations")
    print(score_importance_first)

    score_df_second = score_df[score_df['innings_type'] == 'second']
    target = list(score_df_second['runs'])
    displaye_list_second = []
    for col_idx in range(len(all_cols) - 3):
        l = list(score_df_second[all_cols[col_idx]])
        corr, p_value = pearsonr(l, target)
        # print(all_cols[col_idx],'\t\t\t\t',corr,'\t',p_value)
        display_dict = {
            'col': all_cols[col_idx],
            'cor': corr,
            'p_value': p_value
        }

        displaye_list_second.append(display_dict)

    score_importance_second = pd.DataFrame(displaye_list_second).sort_values(['cor'], ascending=False)
    print("Second innings correlations")
    print(score_importance_second)

    pipe = Pipeline([('scaler', StandardScaler()), ('regression', LinearRegression())])
    sfs_first = SequentialFeatureSelector(pipe, n_features_to_select=5)
    sfs_first.fit(score_df_first.drop(columns=['innings_type', 'runs', 'is_win']), score_df_first['runs'])

    print("=======first innnings improtance=========")
    for idx in np.where(sfs_first.get_support())[0]:
        print(all_cols[idx])

    pipe = Pipeline([('scaler', StandardScaler()), ('regression', LinearRegression())])
    sfs_second = SequentialFeatureSelector(pipe, n_features_to_select=5)
    sfs_second.fit(score_df_second.drop(columns=['innings_type', 'runs', 'is_win']), score_df_second['runs'])

    print("=======second innnings improtance for runs=========")
    for idx in np.where(sfs_second.get_support())[0]:
        print(all_cols[idx])

    pipe = Pipeline([('scaler', StandardScaler()), ('regression', LogisticRegression())])
    sfs_second_log = SequentialFeatureSelector(pipe, n_features_to_select=5)
    sfs_second_log.fit(score_df_second.drop(columns=['innings_type', 'runs', 'is_win']), score_df_second['is_win'])

    print("=======second innnings improtance for win=========")
    for idx in np.where(sfs_second_log.get_support())[0]:
        print(all_cols[idx])

    # selected_first_innings_score_features = list(score_importance_first['col'][:first_innings_select_count])
    # selected_second_innings_score_features = list(score_importance_second['col'][:second_innings_select_count])
    #
    # pickle.dump(selected_first_innings_score_features,open(os.path.join(outil.DEV_DIR,outil.FIRST_INN_SCORE_COLS),'wb'))
    # pickle.dump(selected_second_innings_score_features, open(os.path.join(outil.DEV_DIR, outil.SECOND_INN_SCORE_COLS), 'wb'))
    #
    #





@click.group()
def retrain():
    pass

@retrain.command()
@click.option('--learning_rate', help='learning rate',default=0.001,type=float)
@click.option('--epoch', help='no of epochs',default=150,type=int)
@click.option('--batch_size', help='batch_size',default=10,type=int)
@click.option('--monitor', help='mae or mape',default='mape')
@click.option('--mode', help='train or tune',default='train')
def train_country_embedding(learning_rate,epoch,batch_size,monitor,mode):
    retrain_country_embedding(learning_rate=learning_rate, epoch=epoch, batch_size=batch_size,monitor=monitor,mode=mode)

@retrain.command()
@click.option('--learning_rate', help='learning rate',default=0.001,type=float)
@click.option('--epoch', help='no of epochs',default=150,type=int)
@click.option('--batch_size', help='batch_size',default=10,type=int)
@click.option('--monitor', help='mae or mape',default='accuracy')
@click.option('--mode', help='train or tune',default='train')
def train_country_embedding_2nd(learning_rate,epoch,batch_size,monitor,mode):
    retrain_country_embedding_second(learning_rate=learning_rate, epoch=epoch, batch_size=batch_size,monitor=monitor,mode=mode)


@retrain.command()
@click.option('--learning_rate', help='learning rate',default=0.001,type=float)
@click.option('--epoch', help='no of epochs',default=150,type=int)
@click.option('--batch_size', help='batch_size',default=10,type=int)
@click.option('--monitor', help='mae or mape',default='mae')
@click.option('--mode', help='train or tune',default='train')
def train_batsman_embedding(learning_rate,epoch,batch_size,monitor,mode):
    retrain_batsman_embedding(learning_rate=learning_rate, epoch=epoch, batch_size=batch_size,monitor=monitor,mode=mode)


@retrain.command()
@click.option('--learning_rate', help='learning rate',default=0.001,type=float)
@click.option('--epoch', help='no of epochs',default=150,type=int)
@click.option('--batch_size', help='batch_size',default=10,type=int)
@click.option('--monitor', help='mae or mape',default='accuracy')
@click.option('--mode', help='train or tune',default='train')
def train_multi_output_neural(learning_rate,epoch,batch_size,monitor,mode):
    retrain_one_shot_multi(learning_rate=learning_rate, epoch=epoch, batch_size=batch_size,monitor=monitor,mode=mode)


@retrain.command()
@click.option('--learning_rate', help='learning rate',default=0.001,type=float)
@click.option('--epoch', help='no of epochs',default=150,type=int)
@click.option('--batch_size', help='batch_size',default=10,type=int)
@click.option('--monitor', help='mae or loss',default='loss')
@click.option('--mode', help='train or tune',default='train')
def train_adversarial(learning_rate,epoch,batch_size,monitor,mode):
    retrain_adversarial(learning_rate=learning_rate, epoch=epoch, batch_size=batch_size,monitor=monitor,mode=mode)

@retrain.command()
@click.option('--learning_rate', help='learning rate',default=0.001,type=float)
@click.option('--epoch', help='no of epochs',default=150,type=int)
@click.option('--batch_size', help='batch_size',default=10,type=int)
@click.option('--monitor', help='mae or mape',default='mape')
@click.option('--mode', help='train or tune',default='train')
def train_first_innings_base_neural(learning_rate,epoch,batch_size,monitor,mode):
    retrain_first_innings_base_neural(learning_rate=learning_rate, epoch=epoch, batch_size=batch_size,monitor=monitor,mode=mode)


@retrain.command()
@click.option('--learning_rate', help='learning rate',default=0.001,type=float)
@click.option('--epoch', help='no of epochs',default=150,type=int)
@click.option('--batch_size', help='batch_size',default=10,type=int)
@click.option('--monitor', help='acuracy',default='accuracy')
@click.option('--mode', help='train or tune',default='train')
def train_one_shot_neural(learning_rate,epoch,batch_size,monitor,mode):
    retrain_one_shot_neural(learning_rate=learning_rate, epoch=epoch, batch_size=batch_size,monitor=monitor,mode=mode)


@retrain.command()
@click.option('--create_output', help='whether to create output or not True\False',default=True,type=bool)
@click.option('--select_all', help='Select all features if true/otherwise use p-value',default=False,type=bool)
def select_first_innings_feature_columns(create_output,select_all):
    if not select_all:
        retrain_first_innings_base(create_output=create_output)
    else:
        select_all_columns('first')


@retrain.command()
@click.option('--create_output', help='whether to create output or not True\False',default=True,type=bool)
@click.option('--select_all', help='Select all features if true/otherwise use p-value',default=False,type=bool)
def select_second_innings_feature_columns(create_output,select_all):
    if not select_all:
        retrain_second_innings_base(create_output=create_output)
    else:
        select_all_columns('second')

@retrain.command()
def first_innings():
    retrain_first_innings()

@retrain.command()
def second_innings():
    retrain_second_innings()



@retrain.command()
def check_country_embedding():
    team_oh_test_x = pickle.load(open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.country_emb_feature_team_oh_test_x), 'rb'))
    opponent_oh_test_x = pickle.load(
        open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.country_emb_feature_opponent_oh_test_x), 'rb'))
    location_oh_test_x = pickle.load(
        open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.country_emb_feature_location_oh_test_x), 'rb'))


    runs_model = outil.load_keras_model(os.path.join(outil.DEV_DIR,
                                                             outil.TEAM_OPPONENT_LOCATION_EMBEDDING_RUN_MODEL)
                                                )
    print(runs_model.predict([team_oh_test_x,opponent_oh_test_x,location_oh_test_x]))

@retrain.command()
def check_batsman_embedding():
    batsman_oh_test_x = pickle.load(
        open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.batsman_emb_feature_batsman_oh_test_x), 'rb'))
    position_oh_test_x = pickle.load(
        open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.batsman_emb_feature_position_oh_test_x), 'rb'))
    location_oh_test_x = pickle.load(
        open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.batsman_emb_feature_location_oh_test_x), 'rb'))
    opponent_oh_test_x = pickle.load(
        open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.batsman_emb_feature_opponent_oh_test_x), 'rb'))

    runs_model = outil.load_keras_model(os.path.join(outil.DEV_DIR,
                                                             outil.BATSMAN_EMBEDDING_RUN_MODEL)
                                                )
    print(runs_model.predict([batsman_oh_test_x,position_oh_test_x,location_oh_test_x,opponent_oh_test_x]))

@retrain.command()
def check_adversarial():
    batsman_oh_test_x = pickle.load(
        open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.adversarial_feature_batsman_oh_test_x), 'rb'))
    position_oh_test_x = pickle.load(
        open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.adversarial_feature_position_oh_test_x), 'rb'))
    location_oh_test_x = pickle.load(
        open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.adversarial_feature_location_oh_test_x), 'rb'))
    bowler_oh_test_x = pickle.load(
        open(os.path.join(ctt.TRAIN_TEST_DIR, ctt.adversarial_feature_bowler_oh_test_x), 'rb'))

    runs_model = outil.load_keras_model(os.path.join(outil.DEV_DIR,
                                                             outil.ADVERSARIAL_RUN_MODEL)
                                                )
    print(runs_model.predict([batsman_oh_test_x,position_oh_test_x,location_oh_test_x,bowler_oh_test_x]))

@retrain.command()
def batsman_runs():
    retrain_batsman_runs()

@retrain.command()
def adversarial_first_innings():
    adversarial_first_innings_runs()

@retrain.command()
@click.option('--first_innings_emb', help='whether to use embedding in first innnings',required=True,type=bool)
@click.option('--second_innings_emb', help='whether to use embedding in first innnings',required=True,type=bool)
def combined(first_innings_emb,second_innings_emb):
    retrain_combined_innings(first_innings_emb=first_innings_emb,second_innings_emb=second_innings_emb)

@retrain.command()
@click.option('--start_date', help='start date for train data (YYYY-mm-dd)',required=True)
@click.option('--end_date', help='start date for test data (YYYY-mm-dd)',required=True)
@click.option('--first_innings_select_count',help='no of score features to select',default=5)
@click.option('--second_innings_select_count',help='no of score fetures to select for second innings',default=6)
def select_score_cols(start_date,end_date,first_innings_select_count,second_innings_select_count):
    score_correlation(start_date,end_date,first_innings_select_count=first_innings_select_count,
                      second_innings_select_count=second_innings_select_count)




if __name__=="__main__":
    retrain()
