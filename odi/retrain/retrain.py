
from odi.retrain import base_model_architecture as bma
from odi.retrain import create_train_test as ctt
from odi.model_util import odi_util as outil
from odi.evaluation import evaluate as cric_eval


from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_absolute_error,mean_squared_error,accuracy_score,precision_recall_fscore_support

from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.svm import SVC
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
        bma.create_country_embedding_model(team_oh_train_x.shape[1],\
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

    statsmodel_scaler = StandardScaler()
    train_x_scaled=statsmodel_scaler.fit_transform((train_x))
    model = sm.OLS(train_y, sm.add_constant(train_x_scaled)).fit()

    train_y_predict = model.predict(sm.add_constant(train_x_scaled))
    test_y_predict = model.predict(sm.add_constant(statsmodel_scaler.transform(test_x)))

    mape_train = cric_eval.mape(train_y,train_y_predict)
    mape_test = cric_eval.mape(test_y, test_y_predict)

    mae_train = mean_absolute_error(train_y,train_y_predict)
    mae_test = mean_absolute_error(test_y,test_y_predict)

    mse_train = mean_squared_error(train_y,train_y_predict)
    mse_test = mean_squared_error(train_y,train_y_predict)

    print(model.summary())
    print('metrics train ', mape_train, mae_train)
    print('metrics test ', mape_test, mae_test)

    pipe = Pipeline([('scaler', StandardScaler()), ('regression', LinearRegression())])
    pipe.fit(train_x,train_y)

    train_y_predict_lr = pipe.predict(train_x)
    test_y_predict_lr = pipe.predict(test_x)

    mape_train_lr = cric_eval.mape(train_y, train_y_predict_lr)
    mape_test_lr = cric_eval.mape(test_y, test_y_predict_lr)

    mae_train_lr = mean_absolute_error(train_y, train_y_predict_lr)
    mae_test_lr = mean_absolute_error(test_y, test_y_predict_lr)

    print("from scikit learn")
    print('metrics train ', mape_train_lr, mae_train_lr)
    print('metrics test ', mape_test_lr, mae_test_lr)

    #print(np.where(np.array(model.pvalues) < 0.05))
    selected_feature_index = list(np.where(np.array(model.pvalues) < 0.05)[0])
    print("selected indices including bias ",selected_feature_index)
    # need to substract 1 to exclude bias index consideration:
    column_list = pickle.load(open(os.path.join(outil.DEV_DIR, ctt.first_innings_base_columns), 'rb'))

    selected_column = list()
    selected_index = list()
    for index in selected_feature_index:
        if index == 0:
            continue
        selection_index = index-1
        selected_column.append(column_list[selection_index])
        selected_index.append(index-1)

    print('Selected columns ',selected_column)

    new_train_x = train_x[:,np.array(selected_index)]
    new_test_x = test_x[:,np.array(selected_index)]
    print("selected_index ",selected_index)
    print("new train_x ",new_train_x.shape)
    print("new test x ", new_test_x.shape)
    pipe_new = Pipeline([('scaler', StandardScaler()), ('regression', LinearRegression())])
    pipe_new.fit(new_train_x,train_y)

    new_train_y_predict = pipe_new.predict(new_train_x)
    new_test_y_predict = pipe_new.predict(new_test_x)

    mape_train_new = cric_eval.mape(train_y, new_train_y_predict)
    mape_test_new = cric_eval.mape(test_y, new_test_y_predict)

    mae_train_new = mean_absolute_error(train_y, new_train_y_predict)
    mae_test_new = mean_absolute_error(test_y, new_test_y_predict)

    mse_train_new = mean_squared_error(train_y, new_train_y_predict)
    mse_test_new = mean_squared_error(test_y, new_test_y_predict)

    print("After selecting columns by p-value")
    print("metrics train ",mape_train_new,mae_train_new)
    print("metrics test ", mape_test_new, mae_test_new)


    if create_output:
        pickle.dump(selected_column,open(os.path.join(outil.DEV_DIR,outil.FIRST_INNINGS_FEATURE_PICKLE),'wb'))
        pickle.dump(pipe_new, open(os.path.join(outil.DEV_DIR, outil.FIRST_INNINGS_MODEL_BASE), 'wb'))
        pickle.dump(selected_index,
                    open(os.path.join(outil.DEV_DIR, outil.FIRST_INNINGS_SELECTED_COLUMN_INDEX), 'wb'))

        outil.create_model_meta_info_entry('selected_first_innings_features',
                                           (mape_train_new, mae_train_new,mse_train_new,model.rsquared_adj),
                                           (mape_test_new, mae_test_new,mse_test_new,model.rsquared_adj),
                                           info="metrics is mape,mae,mse,adjusted r quared - selected :"+str(selected_column),
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


    print("from scikit learn")
    print('metrics train ', accuracy_train_lr,precision_train,recall_train,fscore_train)
    print('metrics test ', accuracy_test_lr,precision_test, recall_test, fscore_test)


    pickle.dump(pipe,open(os.path.join(outil.DEV_DIR,outil.COMBINED_MODEL),'wb'))

    pickle.dump(pipe, open(os.path.join(outil.DEV_DIR, "fi_" + str(first_innings_emb) + "_si_" + str(
        second_innings_emb) + "_" + outil.COMBINED_MODEL), 'wb'))

    outil.create_model_meta_info_entry(
        'combined_model_' + "fi_" + str(first_innings_emb) + "_si_" + str(second_innings_emb),
        (accuracy_train_lr, precision_train, recall_train, fscore_train),
        (accuracy_test_lr, precision_test, recall_test, fscore_test),
        info="metrics is accuracy,precision, recall,fscore",
        file_list=[
            outil.COMBINED_MODEL,
        ])


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
@click.option('--monitor', help='mae or mape',default='mae')
@click.option('--mode', help='train or tune',default='train')
def train_batsman_embedding(learning_rate,epoch,batch_size,monitor,mode):
    retrain_batsman_embedding(learning_rate=learning_rate, epoch=epoch, batch_size=batch_size,monitor=monitor,mode=mode)


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




if __name__=="__main__":
    retrain()
