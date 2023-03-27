import typing
import tensorflow as tf
import pandas as pd
import math
import time

from topology import Topology
from hyperparameter_configuration import HyperparameterConfiguration
from keras.callbacks import EarlyStopping
from lstm_model import LstmModel
import dataprocessing as dp
import evaluate
import plot
from train_result import TrainResult

def do_multi_feature_multi_step_training(feature_dataset,
                                         target_dataset,
                                         topology: Topology,
                                         train_split,
                                         history_size: int,
                                         future_target: int,
                                         step_length: int,
                                         buffer_size: int,
                                         batch_size: int,
                                         epochs: int,
                                         steps_per_epoch: int,
                                         validation_steps: int,
                                         loss_function: str,
                                         dropout: float,
                                         patience: int,
                                         plot_training: bool = False,
                                         validation_samples: int = 1):

    tf_train_data, tf_val_data, data_train = dp.split_train_and_val_data(feature_dataset=feature_dataset,
                                                                         target_dataset=target_dataset,
                                                                         train_split=train_split,
                                                                         history_size=history_size,
                                                                         future_target=future_target,
                                                                         step_length=step_length,
                                                                         buffer_size=buffer_size,
                                                                         batch_size=batch_size,
                                                                         single_step=False)

    model = tf.keras.models.Sequential()
    # NOTE(Stefan): If there is only one hidden layer, return_sequences must be False
    return_sequences = len(topology.hidden_layers) != 1

    model.add(tf.keras.layers.LSTM(topology.hidden_layers[0], return_sequences=return_sequences,
                                   dropout=dropout,
                                   input_shape=data_train.shape[-2:]))

    for i in range(1, len(topology.hidden_layers)):
        return_sequences = False if i == (len(topology.hidden_layers) - 1) else True
        model.add(tf.keras.layers.LSTM(topology.hidden_layers[i], return_sequences=return_sequences))

    model.add(tf.keras.layers.Dense(future_target))

    # NOTE(Stefan): Configures the model for training
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss_function)

    # NOTE(Reni): Implements early stopping
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=patience, restore_best_weights=True)

    # NOTE(Stefan): Trains the model for a fixed number of epochs (iterations on a dataset)
    history = model.fit(tf_train_data, epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=tf_val_data,
                        validation_steps=validation_steps,
                        callbacks=[early_stopping])

    if plot_training:
        plot.plot_training_and_validation_loss(history, 'Training and validation loss')

        # NOTE(Stefan): Plot Multi Step Multi Feature Multi Layer Model
        for x, y in tf_val_data.take(validation_samples):
            plot.multi_step_plot(x[0], y[0], model.predict(x)[0], step_length)

    return model


def train(data: pd.DataFrame, topology: Topology = Topology([72, 36]),
          plotting: bool = False,
          hyperparameter: HyperparameterConfiguration = HyperparameterConfiguration(),
          data_frame_cut: float = 0,
          evaluation_size: float = 0) -> typing.Tuple[typing.Optional[LstmModel],
                                                      typing.Optional[TrainResult]]:

    tf.random.set_seed(42)
    tf.random.set_global_generator(tf.random.Generator.from_seed(42))

    # NOTE(Stefan): Fixed parameters
    step_length = 1
    batch_size = 256
    buffer_size = 30_000
    
    # NOTE(Stefan): Select a percentage of the dataset
    start = int(len(data) * data_frame_cut)
    end = int(len(data))
    data_frame = data[start:end]
    loads = pd.DataFrame(data[['Load_MW', 'MESS_DATUM']][start:end])

    features_used, features_considered_tuple = \
        dp.generate_feature_list(n_loads=hyperparameter.n_loads,
                                 forecast_shift=hyperparameter.prediction_size)

    if evaluation_size == 0:
        evaluation_start_index = int(len(loads) * 0.9)
    else:
        evaluation_start_index = -evaluation_size


    train_split = 0.7

    dataset_train, dataset_test, power_transformer, standardizer, normalizer = \
        dp.create_multi_feature_standardized_dataset(dataset=data_frame,
                                                     features_considered=features_used,
                                                     index='MESS_DATUM',
                                                     evaluation_start_index=evaluation_start_index,
                                                     train_split=train_split,
                                                     power_transform=True,
                                                     standardize=True,
                                                     normalize=False)

    true_prediction = loads.iloc[evaluation_start_index:]
    true_prediction = list(true_prediction['Load_MW'])

    # NOTE(Stefan): Drop the first few entries because they don´t get predicted
    for x in range(hyperparameter.history_size):
        true_prediction.pop(0)

    train_split_index = int(len(dataset_train) * train_split)

    # NOTE(Stefan): Extract labels
    index = data.columns.get_loc('Load_MW')
    target_dataset = dataset_train[:, 0]

    start_timer = time.perf_counter()

    model = do_multi_feature_multi_step_training(feature_dataset=dataset_train,
                                                 target_dataset=target_dataset,
                                                 topology=topology,
                                                 train_split=train_split_index,
                                                 history_size=hyperparameter.history_size,
                                                 future_target=hyperparameter.prediction_size,
                                                 step_length=step_length,
                                                 buffer_size=buffer_size,
                                                 batch_size=batch_size,
                                                 epochs=hyperparameter.epochs,
                                                 steps_per_epoch=hyperparameter.steps_per_epoch,
                                                 validation_steps=hyperparameter.validation_steps,
                                                 dropout=hyperparameter.dropout,
                                                 patience=hyperparameter.patience,
                                                 loss_function=hyperparameter.loss,
                                                 plot_training=plotting,
                                                 validation_samples=3)

    end_timer = time.perf_counter()
    print('Needed Time: ', end_timer - start_timer)
    print('Finished training!')

    model_predictions = []
    # NOTE(Fabian): The scrap is the amount data points at the end which can not be predicted because they don't
    #               fit in an even amount of predictions steps. Cutting these values is only required in the LSTM to
    #               prevent it from crashing. Do not copy this logic into another KNN model.
    # NOTE(Stefan): Evaluation loop

    scrap = (abs(hyperparameter.history_size - hyperparameter.prediction_size) / hyperparameter.prediction_size) + 1
    for prediction in range(math.floor((len(dataset_test) / hyperparameter.prediction_size) - scrap)):
        input_data = dataset_test[prediction * hyperparameter.prediction_size:
                                  (prediction * hyperparameter.prediction_size) + hyperparameter.history_size]
        input_data = input_data.reshape(1, hyperparameter.history_size, len(features_used))
        predictions_from_model = model.predict(input_data)
        predictions_from_model = predictions_from_model.flatten()
        model_predictions.append(predictions_from_model)

    # ---------- Calculating errors ---------- #
    model_predictions_flat = [y for x in model_predictions for y in x]
    model_predictions_flat = dp.denormalize_data(prediction=model_predictions_flat,
                                                 power_transformer=power_transformer,
                                                 standardizer=standardizer,
                                                 normalizer=normalizer)

    true_prediction = true_prediction[:len(model_predictions_flat)]
    nmae, mape, rmse, mae = evaluate.calc_all_errors(true_prediction, model_predictions_flat)
    result = TrainResult(nmae, mape, rmse, mae)
    # ---------------------------------------- #

    print("Calculated error.", mape)
    if plotting:
        plot.plot_predict_fit_line(y_test=true_prediction, y_pred=model_predictions_flat, start=0, end=144)
    model = LstmModel('LSTM_F6', power_transformer, standardizer, normalizer, hyperparameter.history_size,
                      hyperparameter.prediction_size, hyperparameter.prediction_size, hyperparameter.n_loads,
                      features_considered_tuple, model, 'multi', topology)

    return model, result
