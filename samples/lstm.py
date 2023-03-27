import pandas as pd
import tensorflow as tf
import dataprocessing as dp

from datetime import datetime
from time_frame import TimeFrame
from train import train
from serialization import save_model, load_model
from prediction import get_comparison_data_for_time_frame
from evaluate import calc_all_errors

# NOTE(Stefan): Workaround for memory issues with Pycharm where the first process allocates all gpu memory
physical_devices = tf.config.list_physical_devices('GPU')
for x in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[x], True)

print("Loading data")
data_path = '../data/data.csv'
data = pd.read_csv(data_path)

print("Adding features")
data = dp.add_derived_features(data)
print("Shift loads")
data = dp.shift_n_loads(data, 6)
print("Shift forecasts")
data = dp.shift_forecast(data, 72)

time_frame = [TimeFrame(datetime(year=2014, month=9, day=15, hour=0),
                        datetime(year=2021, month=12, day=31, hour=0))]
prediction_start = datetime(year=2022, month=1, day=1, hour=0)
prediction_end = datetime(year=2022, month=9, day=30, hour=0)

print("Start training")
cut_off_date = pd.Timestamp(prediction_start).tz_localize('utc')
train_data = data[data['MESS_DATUM'] < cut_off_date]
model, result = train(train_data, plotting=True)
print(f"Training finished with a mae of: {result.mae}")

print("Save model")
save_model('lstm.model', model, 'multi')
print("Load model")
model = load_model('lstm.model')

print("Predict time frame")
pred, true = get_comparison_data_for_time_frame(prediction_start, prediction_end, data, model)
nmae, mape, rmse, mae = calc_all_errors(true_values=true, predicted_values=pred)
print(f'nmae: {nmae}, mape : {mape}, rmse: {rmse}, mae: {mae}')