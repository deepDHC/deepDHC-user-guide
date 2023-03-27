import pickle
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils

from lstm_model import LstmModel
from topology import Topology

def unpack_keras_model(model, training_config, weights):
    """
     Unpacks keras model without the need
     to save in a temp file

    Parameters
    ----------
    model : keras model
        DESCRIPTION.
    training_config : TYPE
        DESCRIPTION.
    weights : TYPE
       .

    Returns
    -------
    restored_model : pick able model
        DESCRIPTION.

    """
    restored_model = deserialize(model)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(
                training_config
            )
        )
    restored_model.set_weights(weights)
    return restored_model


def make_keras_picklable():
    """
     Makes keras models picklable.
    -------
       

    """

    def __reduce__(self):
        model_metadata = saving_utils.model_metadata(self)
        training_config = model_metadata.get('training_config', None)
        model = serialize(self)
        weights = self.get_weights()
        return unpack_keras_model, (model, training_config, weights)

    cls = Model
    cls.__reduce__ = __reduce__


def save_model(path: str, model: LstmModel, mode: str):

    # NOTE(Fabian): This must be done to make the whole model with metadata serializable
    make_keras_picklable()

    data = {'model_name': model.name, 'mode': mode, 'history_size': model.history_size,
            'prediction_size': model.future_size, 'time_shift': model.time_shift,
            'n_loads': model.n_loads, 'power_transformer': model.power_transformer,
            'standardizer': model.standardizer, 'normalizer': model.normalizer,
            'features': model.db_features_used, 'topology' : model.topology.hidden_layers,
            'model': model.keras_model}

    outfile = open(path, 'wb')
    pickle.dump(data, outfile)
    outfile.close()


def load_model(path: str) -> LstmModel:

    infile = open(path, 'rb')
    data = pickle.load(infile)
    infile.close()

    return LstmModel(data['model_name'], data['power_transformer'], data['standardizer'],
                     data['normalizer'],
                     data['history_size'],
                     data['prediction_size'],
                     data['time_shift'],
                     data['n_loads'],
                     data['features'],
                     data['model'],
                     data['mode'],
                     Topology(data['topology']))
