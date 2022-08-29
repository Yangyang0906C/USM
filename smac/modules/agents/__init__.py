REGISTRY = {}

from .rnn_agent import RNNAgent
from .noise_rnn_agent import RNNAgent as NoiseRNNAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["noise_rnn"] = NoiseRNNAgent