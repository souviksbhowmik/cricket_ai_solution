import pandas as pd
from odi.feature_engg import util as cricutil
from odi.data_loader import  data_loader as dl
from odi.model_util import odi_util as outil
import os
import numpy as np
import pickle
import click
from tqdm import tqdm


def create_country_encoding(start_date,end_date):

    if not os.path.isdir(outil.DEV_DIR):
        os.makedirs(outil.DEV_DIR)

    start_dt = cricutil.str_to_date_time(start_date)
    end_dt = cricutil.str_to_date_time(end_date)

    match_list_df = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION + os.sep + 'cricinfo_match_list.csv')
    match_list_df = match_list_df[(match_list_df['date'] >= start_dt) & \
                                  (match_list_df['date'] <= end_dt)]

    country_list = list(set(match_list_df['first_innings'].unique()).union(set(match_list_df['second_innings'].unique())))
    vec_len = len(country_list)
    country_map = dict()
    for index,country in tqdm(enumerate(country_list)):
        vec = np.zeros((vec_len)).astype(int)
        vec[index] = 1
        country_map[country.strip()] = vec

    pickle.dump(country_map,open(outil.DEV_DIR+os.sep+outil.COUNTRY_ENCODING_MAP,'wb'))
    outil.create_meta_info_entry('country_encoding',start_date,end_date,
                                 file_list=[outil.COUNTRY_ENCODING_MAP])



def create_location_encoding(start_date,end_date):

    if not os.path.isdir(outil.DEV_DIR):
        os.makedirs(outil.DEV_DIR)

    start_dt = cricutil.str_to_date_time(start_date)
    end_dt = cricutil.str_to_date_time(end_date)

    match_list_df = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION + os.sep + 'cricinfo_match_list.csv')
    match_list_df = match_list_df[(match_list_df['date'] >= start_dt) & \
                                  (match_list_df['date'] <= end_dt)]

    location_list = list(match_list_df['location'].unique())
    vec_len = len(location_list)
    loc_map = dict()
    for index,location in tqdm(enumerate(location_list)):
        vec = np.zeros((vec_len)).astype(int)
        vec[index] = 1
        loc_map[location.strip()] = vec

    pickle.dump(loc_map,open(outil.DEV_DIR + os.sep + outil.LOC_ENCODING_MAP,'wb'))
    pickle.dump(loc_map, open(outil.DEV_DIR + os.sep + outil.LOC_ENCODING_MAP_FOR_BATSMAN, 'wb'))
    outil.create_meta_info_entry('location_encoding', start_date, end_date,
                                 file_list=[outil.LOC_ENCODING_MAP,
                                            outil.LOC_ENCODING_MAP_FOR_BATSMAN])


# def create_batsman_encoding(start_date,end_date):
#     if not os.path.isdir(outil.DEV_DIR):
#         os.makedirs(outil.DEV_DIR)
#
#     start_dt = cricutil.str_to_date_time(start_date)
#     end_dt = cricutil.str_to_date_time(end_date)
#
#     batting_list_df = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION + os.sep + 'cricinfo_batting.csv')
#
#     batting_list_df = batting_list_df[(batting_list_df['date'] >= start_dt) & \
#                                   (batting_list_df['date'] <= end_dt)]
#
#
#     no_of_rows = batting_list_df.shape[0]
#     batsman_set = set()
#
#     for index in tqdm(range(no_of_rows)):
#         country = batting_list_df.iloc[index]['team_statistics'].strip()
#         for bi in range(11):
#             batsman = batting_list_df.iloc[index]['batsman_'+str(bi+1)].strip()
#             if batsman != 'not_batted':
#                 batsman_set.add(country+' '+batsman)
#             else:
#                 break
#
#     batsman_list = list(batsman_set)
#
#     vec_len = len(batsman_list)
#     batsman_map = dict()
#     batsman_map['not_batted'] = np.zeros((vec_len)).astype(int)
#     for index, batsman in tqdm(enumerate(batsman_list)):
#         vec = np.zeros((vec_len)).astype(int)
#         vec[index] = 1
#         batsman_map[batsman] = vec
#
#     pickle.dump(batsman_map, open(outil.DEV_DIR + os.sep + outil.BATSMAN_ENCODING_MAP, 'wb'))
#     outil.create_meta_info_entry('batsman_encoding', start_date, end_date,
#                                  file_list=[outil.BATSMAN_ENCODING_MAP])

# def create_bowler_encoding(start_date,end_date):
#     if not os.path.isdir(outil.DEV_DIR):
#         os.makedirs(outil.DEV_DIR)
#
#     start_dt = cricutil.str_to_date_time(start_date)
#     end_dt = cricutil.str_to_date_time(end_date)
#
#     match_list_df = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION + os.sep + 'match_list.csv')
#     match_list_df = match_list_df[(match_list_df['date'] >= start_dt) & \
#                                   (match_list_df['date'] <= end_dt)]
#
#     match_stats_df = pd.read_csv(dl.CSV_LOAD_LOCATION + os.sep + 'match_stats.csv')
#     match_list_df = match_list_df.merge(match_stats_df,on='match_id',how='inner')
#     no_of_rows = match_list_df.shape[0]
#     bowler_set = set()
#
#     for index in tqdm(range(no_of_rows)):
#         country = match_list_df.iloc[index]['team_statistics'].strip()
#         for bi in range(11):
#             bowler = match_list_df.iloc[index]['bowler_'+str(bi+1)].strip()
#             if bowler != 'not_bowled':
#                 bowler_set.add(country+' '+bowler)
#             else:
#                 break
#
#     bowler_list = list(bowler_set)
#
#     vec_len = len(bowler_list)
#     bowler_map = dict()
#     bowler_map['not_bowled'] = np.zeros((vec_len)).astype(int)
#     for index, bowler in tqdm(enumerate(bowler_list)):
#         vec = np.zeros((vec_len)).astype(int)
#         vec[index] = 1
#         bowler_map[bowler] = vec
#
#     pickle.dump(bowler_map, open(outil.DEV_DIR + os.sep + outil.BOWLER_ENCODING_MAP, 'wb'))
#     outil.create_meta_info_entry('bowler_encoding', start_date, end_date,
#                                  file_list=[outil.BOWLER_ENCODING_MAP])
#
# def copy_encoding(enc_type,new_value,existing_value):
#     if enc_type=="loc":
#         loc_map = pickle.load(open(outil.DEV_DIR + os.sep + outil.LOC_ENCODING_MAP, 'rb'))
#         loc_for_batsman_map =pickle.load(open(outil.DEV_DIR + os.sep + outil.LOC_ENCODING_MAP_FOR_BATSMAN, 'rb'))
#
#         vec_1 = np.array(loc_map[existing_value])
#         vec_2 = np.array(loc_for_batsman_map[existing_value])
#
#         loc_map[new_value.strip()]=vec_1
#         loc_for_batsman_map[new_value.strip()]=vec_2
#
#         pickle.dump(loc_map, open(outil.DEV_DIR + os.sep + outil.LOC_ENCODING_MAP, 'wb'))
#         pickle.dump(loc_for_batsman_map, open(outil.DEV_DIR + os.sep + outil.LOC_ENCODING_MAP_FOR_BATSMAN, 'wb'))
#

@click.group()
def encode():
    pass


@encode.command()
@click.option('--start_date', help='start date in YYYY-mm-dd .')
@click.option('--end_date', help='start date in YYYY-mm-dd .')
def country(start_date,end_date):
    create_country_encoding(start_date, end_date)


@encode.command()
@click.option('--start_date', help='start date in YYYY-mm-dd .')
@click.option('--end_date', help='start date in YYYY-mm-dd .')
def location(start_date,end_date):
    create_location_encoding(start_date, end_date)


# @encode.command()
# @click.option('--start_date', help='start date in YYYY-mm-dd .')
# @click.option('--end_date', help='start date in YYYY-mm-dd .')
# def batsman(start_date,end_date):
#     create_batsman_encoding(start_date, end_date)
#
# @encode.command()
# @click.option('--start_date', help='start date in YYYY-mm-dd .')
# @click.option('--end_date', help='start date in YYYY-mm-dd .')
# def bowler(start_date,end_date):
#     create_bowler_encoding(start_date, end_date)
#
# @encode.command()
# @click.option('--enc_type', help='loc or batsman or country .',default='loc')
# @click.option('--new_value', help='new value to be inserted',required=True)
# @click.option('--existing_value', help='existing value to be copied from',required=True)
# def copy(enc_type,new_value,existing_value):
#     copy_encoding(enc_type, new_value, existing_value)


if __name__=="__main__":
    encode()
