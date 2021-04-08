import pandas as pd
from datetime import datetime
from odi.data_loader import data_loader as dl
import os
import click

@click.command()
@click.option('--purge_date', help='purge data in YYYY-mm-dd.',required=True)
def purge_before(purge_date):
    reference_date = datetime.strptime(purge_date, '%Y-%m-%d')

    custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d")
    match_summary_df = pd.read_csv(dl.CSV_LOAD_LOCATION + os.sep + 'match_list.csv',
                                   parse_dates=['date'], date_parser=custom_date_parser)
    match_stats_df = pd.read_csv(dl.CSV_LOAD_LOCATION + os.sep + 'match_stats.csv')

    match_summary_df = match_summary_df[match_summary_df['date'] <= reference_date]
    deletion_id_list  = match_summary_df['match_id'].unique()
    for id in deletion_id_list:
        os.remove(dl.CSV_LOAD_LOCATION + os.sep + str(id)+'.csv')

    match_stats_df = match_stats_df[~match_stats_df['match_id'].isin(deletion_id_list)]

    match_summary_df.to_csv(dl.CSV_LOAD_LOCATION + os.sep + 'match_list.csv', index=False)
    match_stats_df.to_csv(dl.CSV_LOAD_LOCATION + os.sep + 'match_stats.csv', index=False)


if __name__=="__main__":
    purge_before()
