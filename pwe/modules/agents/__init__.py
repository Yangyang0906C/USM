REGISTRY = {}

from .rnn_agent import RNNAgent
from .nn_agent import RNNAgent as NNAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["nn"] = NNAgent