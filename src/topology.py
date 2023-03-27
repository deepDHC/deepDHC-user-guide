import random
import typing


class Topology:

    def __init__(self, hidden_layers: typing.List[int]):
        self.hidden_layers = hidden_layers

    @classmethod
    def random(cls, min_hidden_layers: int, max_hidden_layers: int, min_neurons_per_layer: int,
               max_neurons_per_layer: int):
        hidden_layers = []
        for i in range(random.randint(min_hidden_layers, max_hidden_layers)):
            hidden_layers.append(random.randint(min_neurons_per_layer, max_neurons_per_layer))

        return Topology(hidden_layers)


