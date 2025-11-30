from typing import Dict
from GeneralHelpers import Singleton
from Graphs.Factories.Networks.neural_net import Neural_Net


class Base_Network_Factory(metaclass=Singleton):
    """ **Factory** is a _singleton_ object that generates neural networks.
        Base class for network factory (where Factory is a design
        pattern: https://realpython.com/factory-method-python/).
        Subclasses decide which class to instantiate.
    """
    def __init__(self):
        # Protected storage of networks
        self._neural_nets: Dict[str, Neural_Net] = {}
