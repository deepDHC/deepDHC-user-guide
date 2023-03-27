# A user guide for implementing machine learning methods for thermal load prediction in district heating networks

## Introduction
Time series forecasting can be a powerful tool for predicting the thermal load demand of a district heating network.
By analyzing historical data on energy usage and other relevant factors like weather, a machine learning algorithm can identify patterns and trends that help it make accurate predictions about future demand.
This allows the network operator to optimize the use of resources, cut costs and reduce emissions, ensuring that heat is delivered efficiently and cost-effectively to customers.
Additionally, accurate forecasting can help the operator plan for maintenance and repairs, ensuring that the network remains reliable and functional even during periods of high demand.
Overall, time series forecasting can help district heating networks operate more efficiently and save money, while providing a good service to customers.

One possible the deep learning method that can be used to do time series forecasting is the LSTM network.
A Long Short-Term Memory (LSTM) network is a type of Recurrent Neural Network (RNN) that is particularly useful for processing and predicting sequential data.
It uses a complex architecture of cells that can remember values for long periods of time, allowing it to capture dependencies and patterns in sequential data.
This short guide will go through an example scenario that will use thermal load and weather data to train a LSTM model.
This model will then be saved and loaded from disc to do load predictions for another timeframe.
Moreover, it shows how to calculate errors of the predictions of the real load demand is known, thereby providing a brief overview about all basic components needed to create model for district heating demand predictions. 

This guide was created as part of the research project "deepDHC" at University of Applied Sciences Kempten.
You can find more information about the project at [deepDHC.de](deepDHC.de).
DeepDHC was funded by the Federal Ministry of Economics and Energy of Germany under the funding code 03EN3017, with additional financial and in-kind support by the project partners AGFW, Fernwärme Ulm GmbH and ZAK Energie GmbH.
The responsibility for the content of this publication lies with the authors.

## Requirements
In order to run the sample you will need any version of Python 3.7.
You can install all dependencies with your global Python environment, but using a virtual env is recommended.
To install all requirements run:

`pip install -r requirements.txt`

## Example

The following example can be found in `samples/lstm.py`

First, the code reads a CSV file containing the whole dataset needed for the example.
It only loads the basic data.
There are many features that you can extract from this base dataset, like for example you can extract the day of the year from the date column in the dataset.
This is done by using `add_derived_features`, `shfit_n_loads` and `shift_forecast`, respectively.
By shifting the loads and the forecast you can provide the model information about the load demand prior to the point of prediction or how the weather will be for that point.

```python

data_path = '../data/data-clean.csv'
data = pd.read_csv(data_path)

data = dp.add_derived_features(data)
data = dp.shift_n_loads(data, 6)
data = dp.shift_forecast(data, 72)

```

After that the code defines a time frame used for training the model and to more variables that define which time frame will be predicted later on in the example.

```python

time_frame = [TimeFrame(datetime(year=2014, month=9, day=15, hour=0),
                        datetime(year=2021, month=12, day=31, hour=0))]
prediction_start = datetime(year=2022, month=1, day=1, hour=0)
prediction_end = datetime(year=2022, month=9, day=30, hour=0)

```

The code then trains an LSTM model on the training data with a cut-off based on the start of the predictions done later in the example.
This is important to prevent data leakage.
The result are only applicable when the model has not seen the data it will predict during the training process.
When the training is finished the mean absolute error (MAE) of the trained model, indicating its accuracy is printed for the user.

```python

cut_off_date = pd.Timestamp(prediction_start).tz_localize('utc')
train_data = data[data['MESS_DATUM'] < cut_off_date]
model, result = train(train_data, plotting=True)
print(f"training finished with a mae of: {result.mae}")

```

The code then saves the trained LSTM model in a file it again. This example only supports saving to file for simplicity reasons, but you can also save your model to a database or a cloud.

```python

save_model('lstm.model', model, 'multi')
model = load_model('lstm.model')

```

Using the loaded model, then the formerly defined evaluation time frame is predicted.
As the real loads for that time frame are known the code does not simply return the predicted values.
It compares then with the real load demand of that time frame and calculates various errors to indciate the accuracy of the prediction.
Finally, those errors are printed out for the user.

```python

pred, true = get_comparison_data_for_time_frame(prediction_start, prediction_end, data, model)
nmae, mape, rmse, mae = calc_all_errors(true_values=true, predicted_values=pred)
print(f'nmae: {nmae}, mape : {mape}, rmse: {rmse}, mae: {mae}')

```