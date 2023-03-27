import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def create_time_steps(length):
    """
    Creates a list of time steps ranging from -length to 0.

    length: length of the list of time steps
    """
    return list(range(-length, 0))


def multi_step_plot(history, true_future, prediction, step):
    """
    Plotting a sample data point

    history: real past data

    true future: real future data

    prediction: predicted future date

    """
    # NOTE(Stefan): figsize is the scaling of the shown chart
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)
    # NOTE(Stefan): 4 is Load_MW-0
    plt.plot(num_in, np.array(history[:, 0]), label='History')
    plt.plot(np.arange(num_out) / step, np.array(true_future), 'bo',
             label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out) / step, np.array(prediction), 'ro',
                 label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()


def plot_training_and_validation_loss(gradient, title):
    """
    Creates a plot that shows the loss and the validation loss gradient.
    
    gradient: The gradient used to extract the loss functions

    title: The title of the plot. 
    """

    # NOTE(Fabian): Retrieve the loss and validation loss gradient  
    loss = gradient.history['loss']
    validation_loss = gradient.history['val_loss']

    """ 
    NOTE(Fabian): Each epoch generates one value so the length of the list
                  of loss values equals the amount of points used for plotting
                  the gradient. 
    """
    samples_count = range(len(loss))

    """
    NOTE(Fabian): Start a new figure so that the plotting does not affect the
                  former figures.
    """
    plt.figure()

    # NOTE(Fabian): Set up and draw the graph
    plt.plot(samples_count, loss, 'b', label='Training loss')
    plt.plot(samples_count, validation_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()
    plt.show()



def build_moving_average(dataframe_input: pd.DataFrame, parameter: str, window: int):
    dataframe_input[parameter + ' Rolling Average'] = dataframe_input.loc[:, parameter].rolling(window=window).mean()
    return parameter + ' Rolling Average'


def plot_predict_fit_line(y_test, y_pred, start=0, end=72, save_location: str = None):
    # blue = real, orange = predicted
    plt.plot(y_test[start:end], label="real")
    plt.plot(y_pred[start:end], label="predicted")
    plt.xlabel("Stunden")
    plt.ylabel("Last")

    if save_location is not None:
        plt.savefig(fname=save_location)

    plt.show()