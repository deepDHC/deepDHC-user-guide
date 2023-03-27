import logging
import math
import typing
import collections
import pandas as pd
import numpy as np
from workalendar.europe.germany import Bavaria

import holidays_ulm

WEEK_DAY = 1
SATURDAY = 2
SUNDAY = 3
HOLIDAY = 4


def add_season_sin_and_day_of_the_year(data_frame: pd.DataFrame):
    day_of_year = []
    season_sin = []

    for date in data_frame['MESS_DATUM'].values:
        date_time = pd.to_datetime(str(date))
        date_str = date_time.strftime('%Y-%m-%d')
        period = pd.Period(date_str, freq='H')
        day = period.dayofyear

        sin = math.sin((2 * math.pi / 365) * (265 - day))
        day_of_year.append(day)
        season_sin.append(sin)

    data_frame['DayoftheYear'] = day_of_year
    data_frame['Season_Sin'] = season_sin
    return data_frame


def remove_invalid_data(data_frame: pd.DataFrame, net: str):
    logging.info('Starting removal of invalid data ...')

    min_load = 0.000001 if net != 'F3_15bar' else 0

    range_tuple = collections.namedtuple('Range', 'index min max')
    valid_ranges = [
        range_tuple('Temperature', -50, 60),
        range_tuple('Dewpoint', -50, 60),
        range_tuple('Windspeed', 0, 25),
        range_tuple('Winddirection', 0, 360),
        range_tuple('Pressure_NN', 900, 1200),
        range_tuple('Load_MW', min_load, 100)
    ]

    removed_indices = 0

    for index, row in data_frame.iterrows():
        for valid_range in valid_ranges:

            value = float(row[valid_range.index])

            if value < valid_range.min or value > valid_range.max:
                data_frame = data_frame.replace(row[valid_range.index], np.nan)

                log = "{0:>3}: Throwing out line {1:>5}. Column={2:<12}," \
                      " Value={3}, min={4}, max={5}\n".format(removed_indices,
                                                              index,
                                                              valid_range.index,
                                                              row[valid_range.index],
                                                              valid_range.min,
                                                              valid_range.max)
                logging.debug(log)
                removed_indices += 1

    logging.debug(f'Removed {removed_indices} indices')
    logging.info('Finished removing invalid data')
    return data_frame


def categorize_days(data_frame: pd.DataFrame, federal_state: str) -> pd.DataFrame:
    logging.info('Start day categorization ...')

    for index in data_frame.index:

        date = data_frame['MESS_DATUM'].loc[index]

        if federal_state == 'BW':
            ulm_holidays = holidays_ulm.UlmHolidays()
            holidays = ulm_holidays.get_all_holidays(date.year)
        else:
            holidays = Bavaria().get_calendar_holidays(date.year)

        weekday = date.dayofweek
        if weekday == 5:
            day_type = SATURDAY
        elif weekday == 6:
            day_type = SUNDAY
        else:
            day_type = WEEK_DAY

        for holiday in holidays:
            if holiday[0] == date.date():
                day_type = HOLIDAY
                break

        data_frame.loc[index, 'Weekday'] = weekday

        # NOTE(Fabian): Make the four values of day type to four 0-1 binary flags
        data_frame.loc[index, 'Holiday'] = 0
        data_frame.loc[index, 'Weekdays'] = 0
        data_frame.loc[index, 'Saturday'] = 0
        data_frame.loc[index, 'Sunday'] = 0

        if day_type == HOLIDAY:
            data_frame.loc[index, 'Holiday'] = 1
        elif day_type == WEEK_DAY:
            data_frame.loc[index, 'Weekdays'] = 1
        elif day_type == SATURDAY:
            data_frame.loc[index, 'Saturday'] = 1
        elif day_type == SUNDAY:
            data_frame.loc[index, 'Sunday'] = 1
        else:
            assert False and 'No day type was specified'

    logging.info('Categorization of days finished')
    return data_frame


def remove_duplicate_rows(data_frame: pd.DataFrame):
    data_frame['IS_DUPLICATE'] = data_frame.duplicated(subset=['MESS_DATUM'])
    duplicates_indices = data_frame[data_frame['IS_DUPLICATE'] == True].index
    data_frame = data_frame.drop(duplicates_indices)
    data_frame = data_frame.drop('IS_DUPLICATE', axis=1)
    logging.debug('{0} duplicate timestamps were removed'.format(len(duplicates_indices)))
    logging.debug(duplicates_indices)

    return data_frame, duplicates_indices


def calculate_average_load_and_temperature(data: pd.DataFrame, time_slice: int) -> typing.Tuple[float, float]:

    load = 0
    temperature = 0

    for i in range(time_slice):
        load += data['Load_MW'][23 - i]
        temperature += data['Temperature'][23 - i]

    load = load / time_slice
    temperature = temperature / time_slice

    logging.debug(f'Average load: {load}')
    logging.debug(f'Average temperature: {temperature}')
    return load, temperature
