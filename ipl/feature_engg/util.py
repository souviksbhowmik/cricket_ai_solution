from datetime import datetime,date
import dateutil
import pandas as pd


def str_to_date_time(strvalue):
    return datetime.strptime(strvalue, '%Y-%m-%d')


def today_as_date():
    return date.today()


def today_as_date_time():
    today = date.today()
    return datetime(year=today.year, month=today.month, day=today.day)


def substract_year_as_datetime(date_in_datetime,no_of_years):

    a_year = dateutil.relativedelta.relativedelta(years=no_of_years)
    return date_in_datetime-a_year


def substract_day_as_datetime(date_in_datetime,no_of_days):
    a_day = dateutil.relativedelta.relativedelta(days=no_of_days)
    return date_in_datetime-a_day


def add_day_as_datetime(date_in_datetime,no_of_days):
    a_day = dateutil.relativedelta.relativedelta(days=no_of_days)
    return date_in_datetime+a_day


def npdate_to_datetime(npDate):
    return datetime.utcfromtimestamp(npDate.tolist() / 1e9)


def read_csv_with_date(file_loc,date_fied="date"):
    custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d")
    df= pd.read_csv(file_loc,parse_dates=[date_fied],date_parser=custom_date_parser)
    return df


