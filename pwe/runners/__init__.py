REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner


from .episode_uesm import EpisodeRunner as EpisodeRunnerUesm
REGISTRY["episode_uesm"] = EpisodeRunnerUesm
