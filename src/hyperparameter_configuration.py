import random


class HyperparameterConfiguration:
    def __init__(self):

        # NOTE(Fabian): This configuration can be trained quite fast, so it fits the purpose of
        #               validating the implementation of new features. However, you should not
        #               expect good training results with those

        # NOTE(Stefan): Random parameters that are different for each data preparation
        self.history_size = 72  # random.choice([24, 48, 72])
        self.prediction_size = 72  # random.randint(1, 72)
        self.n_loads = 6  # random.randint(2, 12)

        # NOTE(Stefan): Random parameters for training process
        self.validation_steps = 200  # random.randint(50, 500) # 200
        self.steps_per_epoch = 900  # random.randint(100, 700) # 800
        self.epochs = 20  # random.randint(8, 80) # 8
        self.loss = 'mae'  # random.choice(['mae', 'mse', 'mape'])
        self.dropout = 0  # random.choice([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
        self.patience = 5   # number of epochs after which the training is stopped in case of increase of self.loss,
        # usually between 0 and 100

    @staticmethod
    def random():
        result = HyperparameterConfiguration()
        result.history_size = random.choice([24, 48, 72])
        result.prediction_size = random.randint(1, 16)
        result.n_loads = random.randint(2, 12)

        result.validation_steps = random.randint(50, 500)
        result.steps_per_epoch = random.randint(100, 1200)
        result.epochs = random.randint(6, 80)
        result.loss = random.choice(['mae', 'mse', 'mape'])
        result.dropout = random.choice([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])

        return result
