from .episode_buffer import ReplayBuffer
from .episode_buffer_per import ReplayBufferPER
from .random_buffer import ReplayBuffer as RandomBuffer

REGISTRY = {}
REGISTRY["episode_buffer"] = ReplayBuffer
REGISTRY["episode_buffer_per"] = ReplayBufferPER
REGISTRY["random_buffer"] = RandomBuffer
