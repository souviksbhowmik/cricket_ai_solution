from keras.models import model_from_json
import os
import json

PROD_DIR = 'model'
DEV_DIR = 'model_dev'
CHECKPOINT_DIR = DEV_DIR+os.sep+'checkpoint'
MODEL_DIR = PROD_DIR
FIRST_INNINGS_FEATURE_PICKLE = 'first_innings_selected_features.pkl'
SECOND_INNINGS_FEATURE_PICKLE = 'second_innings_selected_features.pkl'
FIRST_INNINGS_SELECTED_COLUMN_INDEX = 'first_innings_selected_colindex.pkl'
SECOND_INNINGS_SELECTED_COLUMN_INDEX = 'second_innings_selected_colindex.pkl'
COUNTRY_ENCODING_MAP = 'country_enc_map.pkl'
LOC_ENCODING_MAP = 'loc_enc_map.pkl'
LOC_ENCODING_MAP_FOR_BATSMAN = 'location_enc_map_for_batsman.pkl'
BATSMAN_ENCODING_MAP = 'batsman_enc_map.pkl'
BOWLER_ENCODING_MAP = 'bowler_enc_map.pkl'
# FIRST_INNINGS_SCALER = 'first_innings_scaler.pkl'# scaler_combined_embedding_first_innings_regression.pkl
# SECOND_INNINGS_SCALER = 'second_innings_scaler.pkl'# second_innings_scaler_with_embedding.pkl

FIRST_INNINGS_MODEL = 'first_innings_model.pkl' # combined_embedding_first_innings_regression.pkl
SECOND_INNINGS_MODEL = 'second_innings_model.pkl' # second_innings_model_with_embedding_lrg.pkl

FIRST_INNINGS_MODEL_BASE = 'first_innings_model_base.pkl' # combined_embedding_first_innings_regression.pkl
SECOND_INNINGS_MODEL_BASE = 'second_innings_model_base.pkl' # second_innings_model_with_embedding_lrg.pkl

TEAM_OPPONENT_LOCATION_EMBEDDING_MODEL = 'group_encode_model'# V2 renamed
TEAM_OPPONENT_LOCATION_EMBEDDING_RUN_MODEL = 'group_encode_run_model'
BATSMAN_EMBEDDING_MODEL = 'batsman_group_encode_model'
BATSMAN_EMBEDDING_RUN_MODEL = 'batsman_group_encode_run_model'

FIRST_INNINGS_REGRESSION_NEURAL = 'first_innings_regression_neural'

ADVERSARIAL_RUN_MODEL = 'adversarial_run_model'
ADVERSARIAL_MODEL = 'adversarial_model'
ADVERSARIAL_BATSMAN_MODEL = 'adversarial_batsman_model'
ADVERSARIAL_BOWLER_MODEL = 'adversarial_bowler_model'
ADVERSARIAL_POSITION_MODEL = 'adversarial_position_model'
ADVERSARIAL_LOCATION_MODEL = 'adversarial_location_model'
ADVERSARIAL_FIRST_INNINGS = 'advesarial_first_innings.pkl'

BATSMAN_RUNS_MODELS = 'batsman_runs_models.pkl'


def store_keras_model(model,model_name):
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_name + ".h5")
    print("Saved model to disk")


def save_keras_model_architecture(model,model_name):
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_name + ".json", "w") as json_file:
        json_file.write(model_json)


def save_keras_model_weights(model,model_name):
    if not model_name.endswith('.h5') and not model_name.endswith('.hdf5'):
        model_name = model_name + ".h5"
    # serialize weights to HDF5
    model.save_weights(model_name)
    print("Saved model to disk")


def load_keras_model(model_name):
    # load json and create model
    json_file = open(model_name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_name+".h5")
    return loaded_model


def load_keras_model_architecture(model_name):
    # load json and create model
    json_file = open(model_name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    return loaded_model


def load_keras_model_weights(model_architecture,model_name):
    # load weights into new model
    if not model_name.endswith('.h5') and not model_name.endswith('.hdf5'):
        model_name=model_name + ".h5"

    model_architecture.load_weights(model_name)
    return model_architecture


def use_model_from(env_name):

    global MODEL_DIR
    if env_name == 'production':
        MODEL_DIR = PROD_DIR
    else:
        MODEL_DIR = DEV_DIR


def create_meta_info_entry(meta_type,start_date,end_date,file_list=[]):

    if os.path.isfile(DEV_DIR + os.sep +'meta_info.json'):
        meta_info = json.load(open(DEV_DIR + os.sep +'meta_info.json','r'))
    else:
        meta_info= dict()
    meta_info[meta_type]={
        'start_date':start_date,
        'end_date':end_date,
        'files':file_list
    }
    json.dump(meta_info,open(DEV_DIR + os.sep +'meta_info.json','w'),indent=2)


def create_model_meta_info_entry(meta_type,train_metrics,test_metrics,info=None,file_list=[]):

    if os.path.isfile(DEV_DIR + os.sep +'meta_info.json'):
        meta_info = json.load(open(DEV_DIR + os.sep +'meta_info.json','r'))
    else:
        meta_info= dict()
    meta_info[meta_type]={
        'train_metrics':train_metrics,
        'test_metrics':test_metrics,
        'info':info,
        'files':file_list
    }
    json.dump(meta_info,open(DEV_DIR + os.sep +'meta_info.json','w'),indent=2)
