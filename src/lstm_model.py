from typing import List
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import PowerTransformer, StandardScaler, MinMaxScaler
from topology import Topology


class LstmModel:
    """
    Structure that bundles all relevant information of a LSTM model
    """
    def __init__(self, name: str, power_transformer: PowerTransformer,
                 standardizer: StandardScaler,
                 normalizer: MinMaxScaler,
                 history_size: int,
                 future_size: int,
                 time_shift: int,
                 n_loads: int,
                 db_features_used,
                 keras_model: Sequential,
                 mode: str,
                 topology: Topology):

        self.name = name
        self.power_transformer = power_transformer
        self.standardizer = standardizer
        self.normalizer = normalizer
        self.history_size = history_size
        self.future_size = future_size
        self.time_shift = time_shift
        self.n_loads = n_loads
        self.db_features_used = db_features_used
        self.keras_model = keras_model
        self.mode = mode
        self.topology = topology
