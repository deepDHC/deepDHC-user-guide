import pandas as pd
import numpy as np
import dataprocessing as dp

from datetime import datetime, timedelta
from lstm_model import LstmModel


def predict_time_series(data: pd.DataFrame, model: LstmModel):

    history_size = model.history_size
    time_shift = model.time_shift

    features = model.db_features_used[0].copy()
    shifted_features = dp.generate_time_shifted_features(model.n_loads, time_shift)
    features += shifted_features

    data_frame = dp.create_standardized_input_dataset(dataframe=data, index='MESS_DATUM',
                                                      features_considered=features,
                                                      power_transformer=model.power_transformer,
                                                      standardizer=model.standardizer,
                                                      normalizer=model.normalizer)

    input_data = data_frame.reshape(1, history_size, len(features))

    predictions_from_model = np.array(model.keras_model(input_data))
    predictions_from_model = predictions_from_model.flatten()
    predictions_from_model = dp.denormalize_data(prediction=predictions_from_model,
                                                 power_transformer=model.power_transformer,
                                                 standardizer=model.standardizer,
                                                 normalizer=model.normalizer)

    return predictions_from_model


def predict_time_frame_for_existing_model(data: pd.DataFrame, start: datetime,
                                          prediction_length: int,
                                          model: LstmModel):

    all_predictions = []
    for x in range(int(prediction_length / model.future_size)):

        start_date = pd.Timestamp(start).tz_localize('utc')
        end_date = pd.Timestamp(start + timedelta(hours=model.future_size - 1)).tz_localize('utc')
        sliced_data = data[(data['MESS_DATUM'] >= start_date) & (data['MESS_DATUM'] <= end_date)]

        predictions_from_model = predict_time_series(data=sliced_data,
                                                     model=model)

        all_predictions.append(predictions_from_model)
        start += timedelta(hours=model.future_size)

    all_predictions_flat = [y for x in all_predictions for y in x]

    return all_predictions_flat


def get_comparison_data_for_time_frame(start: datetime, end: datetime,
                                       data: pd.DataFrame,
                                       model: LstmModel):
    """
    Returns both the actual loads and predictions made by the specified model for a given time frame
    in order to use them for comparisons etc.
    """

    data['MESS_DATUM'] = pd.to_datetime(data['MESS_DATUM'])
    start_date = pd.Timestamp(start).tz_localize('utc')
    end_date = pd.Timestamp(end).tz_localize('utc')
    data = data[(data['MESS_DATUM'] >= start_date) & (data['MESS_DATUM'] <= end_date)]
    true_future_loads = data['Load_MW']

    predictions = predict_time_frame_for_existing_model(data=data,
                                                        start=start,
                                                        prediction_length=len(true_future_loads),
                                                        model=model)
    # NOTE(Fabian): Pop first item of true_future_loads because predictions start at t+1 and
    #               cut away the last true values for which not predictions can be made
    true_future_loads = list(true_future_loads[1 : len(predictions) + 1])

    return predictions, true_future_loads
