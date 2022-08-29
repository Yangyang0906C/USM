from .episode_buffer import ReplayBuffer
from .episode_buffer_per import ReplayBufferPER


REGISTRY = {}
REGISTRY["episode_buffer"] = ReplayBuffer
REGISTRY["episode_buffer_per"] = ReplayBufferPER

