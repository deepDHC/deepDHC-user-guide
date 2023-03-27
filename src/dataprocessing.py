import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import timedelta

import data_transformation as transform
import merge_purge

def add_previous_feature(feature_data_frame: pd.DataFrame, number_of_previous_features: int):
    """
    :param feature_data_frame: DataFrame needs to consist of MESS_DATUM as first column and the feature as second column
    :param number_of_previous_features: Number of features that are added from previous rows
    :return: Returns DataFrame with additional previous features for each row
    """
    data_frame = feature_data_frame[['MESS_DATUM']]
    feature_name = feature_data_frame.columns[1]

    for n in range(number_of_previous_features):
        shifted_load_key = f'{feature_name}-{n}'
        data_frame[shifted_load_key] = feature_data_frame[feature_name].shift(n)

    data_frame = data_frame.iloc[number_of_previous_features:]

    created_features = [f'{feature_name}-{i}' for i in range(number_of_previous_features)]

    return data_frame, created_features


def shift_forecast(data_frame_forecast: pd.DataFrame, shift_size: int) -> pd.DataFrame:
    df_with_shifted_forecast = data_frame_forecast.copy()
    for n in range(shift_size):
        shifted_load_key = f'Temperature_Forecast+{n + 1}'
        df_with_shifted_forecast[shifted_load_key] = data_frame_forecast['Temperature_Forecast'].shift(-(n + 1))
        df_with_shifted_forecast = df_with_shifted_forecast.copy()
    for n in range(shift_size):
        shifted_load_key = f'Dewpoint_Forecast+{n + 1}'
        df_with_shifted_forecast[shifted_load_key] = data_frame_forecast['Dewpoint_Forecast'].shift(-(n + 1))
        df_with_shifted_forecast = df_with_shifted_forecast.copy()

    df_with_shifted_forecast = df_with_shifted_forecast.iloc[:df_with_shifted_forecast.shape[0] - shift_size]

    return df_with_shifted_forecast

def create_data_and_labels_multi_feature(dataset, labels_data, start, end,
                                         history_size: int,
                                         target_size: int,
                                         step_size: int,
                                         single_step=False):
    """
    Creates a set of input data and a set of corresponding labels for a dataset
    that contains multiple features.

    dataset: The dataset used for creating data.

    labels_data: A list of values that get used as labels.

    start: Start index of the data that will be used as input data

    end: End index of the data that will be used as input data. If None end
         will be chosen so that the largest possible amount of input data
         gets created.

    history_size: The number of prior entries in the dataset that will be used
                  as the history of the current entry.

    target_size: The number of data entries that will be predicted by the model.

    step_size: The step size used while creating the data slices

    single_step: Whether the data is used in a single step or a multistep model.

    """

    data = []
    labels = []

    """ 
    NOTE(Fabian): Calculate the actual start index in the list. This is
                  start + history_size because one set of input data consists
                  of the current value and the last history_size values before
                  it. So the first entry can only have history_size values
                  before it the start
                  gets has the same offset.
    """
    start = start + history_size

    """
    NOTE(Fabian): If end is none we assign end the maximal possible value. This
                  is the size of the dataset minus the target_size, because
                  these last values will be predicted with the model, so they
                  are no input data.
    """
    if end is None:
        end = len(dataset) - target_size

    for i in range(start, end):
        """
        NOTE(Fabian): Get the indices of the current entry and the complete
                      history of the entry.
        """
        indices = range(i - history_size, i, step_size)
        data.append(dataset[indices])

        """
        NOTE(Fabian): In a single step model we only need to add one label
                      while a multi step model requires multiple labels to
                      be added.
        """
        if single_step:
            labels.append(labels_data[i + target_size])
        else:
            labels.append(labels_data[i: i + target_size])

    # NOTE(Fabian): Makes the data and labels numpy arrays before returning them
    return np.array(data), np.array(labels)


def split_train_and_val_data(feature_dataset, target_dataset, train_split, history_size, future_target, step_length,
                             buffer_size, batch_size, single_step):
    data_train, label_train = create_data_and_labels_multi_feature(feature_dataset,
                                                                   target_dataset,
                                                                   0,
                                                                   train_split, history_size,
                                                                   future_target, step_length,
                                                                   single_step=single_step)

    data_val, label_val = create_data_and_labels_multi_feature(feature_dataset,
                                                               target_dataset,
                                                               train_split, None, history_size,
                                                               future_target, step_length,
                                                               single_step=single_step)

    # NOTE(Stefan): set and shuffle training dataset
    tf_train_data = tf.data.Dataset.from_tensor_slices((data_train, label_train))
    tf_train_data = tf_train_data.cache().shuffle(buffer_size).batch(batch_size).repeat()

    # NOTE(Stefan): set and batch validation dataset
    tf_val_data = tf.data.Dataset.from_tensor_slices((data_val, label_val))
    tf_val_data = tf_val_data.batch(batch_size).repeat()

    return tf_train_data, tf_val_data, data_train


def create_multi_feature_standardized_dataset(dataset,
                                              features_considered,
                                              index: str,
                                              evaluation_start_index: int,
                                              train_split: float,
                                              power_transform: bool,
                                              standardize: bool,
                                              normalize: bool):

    features = dataset[features_considered]
    # NOTE(Stefan): Sets the date of the entries as the key used for indexing the dataset
    features.index = dataset[index]
    data = features.values

    # NOTE(Stefan): Not ideal, but works with the current train module
    main_data = data[:evaluation_start_index]
    train_data = main_data[:int(len(main_data) * train_split)]

    test_data = data[evaluation_start_index:]

    power_transformer = False
    standardizer = False
    normalizer = False
    if power_transform:
        main_data, train_data, test_data, power_transformer = transform.power_transform(main_data, train_data, test_data)
    if standardize:
        main_data, train_data, test_data, standardizer = transform.standardization(main_data, train_data, test_data)
    if normalize:
        main_data, train_data, test_data, normalizer = transform.normalization(main_data, train_data, test_data)

    return main_data, test_data, power_transformer, standardizer, normalizer


def create_standardized_input_dataset(dataframe,
                                      index: 'str',
                                      features_considered,
                                      power_transformer,
                                      standardizer,
                                      normalizer):
    """

    Parameters
    ----------
    dataframe
    index
    features_considered
    power_transformer
    standardizer
    normalizer

    Returns
    -------

    """

    features = dataframe[features_considered]
    # NOTE(Stefan): Sets the date of the entries as the key used for indexing the dataset
    features.index = dataframe[index]

    data = features.values
    data = transform.transform_data(data, power_transformer, standardizer, normalizer)

    return data


def denormalize_data(prediction, power_transformer, standardizer, normalizer):
    if (power_transformer == False) and (standardizer == False) and (normalizer == False):
        return prediction
    prediction = np.array(prediction)
    prediction = prediction.reshape(-1, 1)
    # NOTE(Stefan): If it crashes here, it´s likely there is no standardizer
    empty_array = np.zeros(shape=(len(prediction), len(standardizer.scale_) - 1))  # 36 len(standardizer.scale_)
    prediction = np.column_stack((prediction, empty_array))
    if normalizer:
        prediction = transform.reverse_normalization(prediction, normalizer)
    if standardizer:
        prediction = transform.reverse_standardization(prediction, standardizer)
    if power_transformer:
        prediction = transform.reverse_power_transform(prediction, power_transformer)
    prediction = prediction[:, 0]
    prediction = list(prediction)
    return prediction


def generate_feature_list(n_loads, forecast_shift):
    """
    List of all features
    ['_id_x', 'MESS_DATUM', 'Load', 'Windspeed', 'Winddirection', 'Cloudcover', 'Pressure_NN', 'Temperature', 'Dewpoint',
    'Last-Load', 'avgLoad24', 'avgLoad12', 'avgLoad6', 'avgTemp24', 'avgTemp12', 'avgTemp6', 'Year', 'Month', 'Day', 'Hour',
     'DayoftheYear', 'Season_Sin', 'Weekday', 'Weekdays', 'Saturday', 'Sunday', 'Holiday', '_id_y', 'Windspeed_Forecast',
      'Winddirection_Forecast', 'Pressure_NN_Forecast', 'Temperature_Forecast', 'Dewpoint_Forecast']
     """
    # NOTE(Stefan): Features considered for training
    load_features_considered = ['Load_MW', 'Temperature', 'Season_Sin', 'avgTemp24', 'Hour']
    forecast_features_considered = ['Temperature_Forecast', 'Dewpoint_Forecast']

    load_features_used = load_features_considered.copy()
    shifted_features = generate_time_shifted_features(n_loads, forecast_shift)

    features_used = load_features_used + shifted_features
    features_considered_tuple = (load_features_considered, forecast_features_considered)

    return features_used, features_considered_tuple

def generate_time_shifted_features(n_loads: int, forecast_shift: int) -> list:

    load_features = [f'Load_MW-{i}' for i in range(n_loads)]

    temperature_features = [f'Temperature_Forecast+{forecast_shift}']
    dewpoint_features = [f'Dewpoint_Forecast+{forecast_shift}']

    return load_features + temperature_features + dewpoint_features


def add_derived_features(data: pd.DataFrame) -> pd.DataFrame:

    data['avgLoad6_MW'] = data['Load_MW'].rolling(window=6).mean()
    data['avgLoad12_MW'] = data['Load_MW'].rolling(window=12).mean()
    data['avgLoad24_MW'] = data['Load_MW'].rolling(window=24).mean()

    data['avgTemp6'] = data['Temperature'].rolling(window=6).mean()
    data['avgTemp12'] = data['Temperature'].rolling(window=12).mean()
    data['avgTemp24'] = data['Temperature'].rolling(window=24).mean()

    data = data[24:]

    converted_dates = pd.to_datetime(data['MESS_DATUM'])
    data = data.drop(columns=['MESS_DATUM'])
    data.insert(0, 'MESS_DATUM', converted_dates)
    data.insert(0, 'Year', data['MESS_DATUM'].dt.year)
    data.insert(0, 'Month', data['MESS_DATUM'].dt.month)
    data.insert(0, 'Day', data['MESS_DATUM'].dt.day)
    data.insert(0, 'Hour', data['MESS_DATUM'].dt.hour)
    data = merge_purge.add_season_sin_and_day_of_the_year(data)
    data = merge_purge.categorize_days(data, 'BW')

    return data


def shift_n_loads(load_data, n_loads: int):
    n_loads_array = []
    only_loads_data = load_data[['MESS_DATUM', 'Load_MW']]

    for n in range(n_loads):
        only_loads_copy = only_loads_data.copy()
        only_loads_copy = only_loads_copy.add_suffix('-' + str(n))
        only_loads_copy.rename(columns={('MESS_DATUM-' + str(n)): 'MESS_DATUM'}, inplace=True)
        for x in range(len(only_loads_copy)):
            only_loads_copy.iloc[x, 0] = only_loads_copy.iloc[x, 0] + timedelta(hours=n)
        n_loads_array.append(only_loads_copy.copy())

    for x in range(len(n_loads_array)):
        load_data = pd.merge(load_data, n_loads_array[x], on='MESS_DATUM', how='inner')

    return load_data
