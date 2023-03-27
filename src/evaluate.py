import typing
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calc_all_errors(true_values: list, predicted_values: list) \
        -> typing.Tuple[float, float, float, float]:
    assert len(true_values) == len(predicted_values)

    nmae, _, _ = nmae_series(true_values, predicted_values)
    mape = mape_series(true_values, predicted_values)
    rmse = rmse_series(true_values, predicted_values)
    mae = mean_absolute_error(true_values, predicted_values)

    return nmae, mape, rmse, mae


def nmae_series(y_true: list, y_pred: list):
    first_part = []
    second_part = []

    max_range = min(len(y_true), len(y_pred))
    for x in range(max_range):
        first_part.append(abs(y_true[x] - y_pred[x]))
        second_part.append(y_true[x])
    numerator = 0
    denominator = 0
    error_per_time_step = []
    absolute_error_per_time_step = []
    for x in range(max_range):
        numerator = numerator + first_part[x]
        denominator = denominator + second_part[x]
        error_per_time_step.append((abs(y_true[x] - y_pred[x])/y_true[x]))
        absolute_error_per_time_step.append((abs(y_true[x] - y_pred[x])))
    normalized_mean_absolute_error = numerator/denominator

    return float(normalized_mean_absolute_error), error_per_time_step, absolute_error_per_time_step


def mae_series(actual_values: typing.List[float], forecast_values: typing.List[float]) \
        -> typing.Optional[float]:
    if len(actual_values) != len(forecast_values):
        return None

    accumulated_error = 0
    skipped_entries = 0
    for i in range(0, len(actual_values)):
        if abs(actual_values[i]) < 0.1:
            skipped_entries += 1
            continue

        accumulated_error += abs(actual_values[i] - forecast_values[i])

    return float(accumulated_error / (len(actual_values) - skipped_entries))


def mape_series(actual_values: typing.List[float], forecast_values: typing.List[float]) \
        -> typing.Optional[float]:
    if len(actual_values) != len(forecast_values):
        return None

    accumulated_error = 0
    skipped_entries = 0
    for i in range(0, len(actual_values)):
        if abs(actual_values[i]) < 0.1:
            skipped_entries += 1
            continue

        accumulated_error += abs((actual_values[i] - forecast_values[i]) / actual_values[i])

    return float(accumulated_error / (len(actual_values) - skipped_entries))


def rmse_series(true_values: typing.List[float], predicted_values: typing.List[float]) -> float:
    mse = mean_squared_error(true_values, predicted_values)
    return np.sqrt(mse)


def max_ape(error_array: list):
    error_array.sort(key=float, reverse=True)
    return error_array[0]


def min_ape(error_array: list):
    error_array.sort(key=float)
    return error_array[0]
