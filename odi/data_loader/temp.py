from odi.data_loader import data_loader as dl
import os
from datetime import datetime,date
import dateutil
#import datetime
import click

import pandas as pd

# custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d")
# match_summary_df = pd.read_csv(dl.CSV_LOAD_LOCATION + os.sep + 'match_list.csv',
#                                parse_dates=['date'], date_parser=custom_date_parser)
#
# print(match_summary_df['date'].min())
# print(match_summary_df['date'].max())
# print(match_summary_df[match_summary_df['date']==match_summary_df['date'].max()])

# today = date.today()
# today_with_time = datetime(
#     year=today.year,
#     month=today.month,
#     day=today.day,
# )
#
# a_year = dateutil.relativedelta.relativedelta(years=1)

# test_list = ['2019-06-08','2018-05-06','2019-03-05','2017-02-03']
# date_list = list()
# for t in test_list:
#     try:
#         date_list.append(datetime.strptime(t, '%Y-%m-%d'))
#     except Exception as ex:
#         print(ex)
#
# date_list.sort()
# print(date_list[-1].date())

# @click.command()
# #@click.option('--year_list', multiple=True,help='list of years.')
# @click.option('--something', help='anythong',type=int)
# def test(something):
#     #print(list(year_list))
#     print('something ',something)
#     # converted_list = list(year_list)
#     # converted_list.sort()
#     # print(converted_list)
#     # if year_list is not None:
#     #     print(type(year_list))
#
# if __name__=='__main__':
#     test()

# import pickle
#
# map=pickle.load(open('../../model_dev/loc_enc_map.pkl','rb'))
# print(map['Kolkata'])


import click

@click.group()
@click.option('--config', help='Path to config file.', type=click.File('r'))
@click.pass_context
def test(ctx, config):
    print('=====',type(ctx))
    pass

@test.command()
@click.pass_context
def test2(ctx):
    print('===========', type(ctx))
    pass


if __name__=="__main__":
    print("invoked")
    test()