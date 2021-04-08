import requests
#from datetime import date
import datetime
import os
import zipfile
import json
import pandas as pd
from ipl.data_loader import data_loader as dl


DATA_DIR = 'data_ipl'
DOWNLOAD_DIR = DATA_DIR+os.sep+'downloads'


def download_zip(url):

    today_str = str(datetime.date.today())
    if not os.path.isdir(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)

    r = requests.get(url, allow_redirects=True)
    open(DOWNLOAD_DIR + os.sep + today_str + '.zip', 'wb').write(r.content)
    print('downloaded...')
    print('unzippig....')
    with zipfile.ZipFile(DOWNLOAD_DIR + os.sep + today_str + '.zip', 'r') as zip_ref:
        zip_ref.extractall(DOWNLOAD_DIR+ os.sep + today_str)

    print('Removing zip file....')
    os.remove(DOWNLOAD_DIR + os.sep + today_str + '.zip')

    return DOWNLOAD_DIR+ os.sep + today_str


def get_new_match_list(match_dir,return_all=False):

    current_content = os.listdir(match_dir)

    new_content = []
    if os.path.isfile(os.path.join(dl.CSV_LOAD_LOCATION,'match_lst.csv')) and not return_all:
        match_list_df = pd.read_csv(os.path.join(dl.CSV_LOAD_LOCATION,'match_lst.csv'))
        match_id_list = list(match_list_df['match_id'].unique().astype(str))
        for file in current_content:
            if (file.endswith('yaml')) and (file.strip('.yaml') not in match_id_list):
                new_content.append(file)
        return new_content
    else:
        return current_content

    # if not os.path.isfile(DOWNLOAD_DIR+os.sep+'visited_match_list.json') or return_all:
    #     json.dump(current_content, open(DOWNLOAD_DIR + os.sep + 'visited_match_list.json', 'w'))
    #     return current_content
    # else:
    #     visited_list = json.load(open(DOWNLOAD_DIR + os.sep + 'visited_match_list.json', 'r'))
    #     for file in current_content:
    #         if file not in visited_list:
    #             final_file_list.append(file)
    #
    #     json.dump(visited_list + final_file_list, open(DOWNLOAD_DIR + os.sep + 'visited_match_list.json', 'w'))
    #     return final_file_list


# def get_new_match_list(match_dir):
#
#     dir_list = os.listdir(DOWNLOAD_DIR)
#     current_dir = match_dir.split(os.sep)[-1]
#     print('curent dir ',current_dir)
#     # getting previous latest dir
#     print(' finding latest previous downloaded directory ')
#     latest = None
#     for dir in dir_list:
#         if dir != current_dir:
#             try:
#                 date_val = datetime.datetime.strptime(dir, '%Y-%m-%d').date()
#             except:
#                 continue
#             if latest is None or latest < date_val:
#                 latest = date_val
#
#     if latest is not None:
#         print('previous latest directory  ',str(latest))
#         previous_dir = str(latest)
#         previous_content = os.listdir(DOWNLOAD_DIR+os.sep+previous_dir)
#         current_content = os.listdir(match_dir)
#         final_file_list = list()
#
#         for file in current_content:
#             if file not in previous_content:
#                 final_file_list.append(file)
#
#
#         return final_file_list
#
#     else:
#         print('no previous directory ')
#         return os.listdir(match_dir)


if __name__=="__main__":
    # print(str(date.today()))
    url = 'https://cricsheet.org/downloads/ipl.zip'
    directory = download_zip(url)
    # date_str = str(datetime.date.today())
    # date_time_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
    # print(date_time_obj)
    # print(file_list)
    # print(len(file_list))
    print(directory)
