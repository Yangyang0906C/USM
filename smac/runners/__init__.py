REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .maven_runner import ParallelRunner as MavenRunner
REGISTRY["maven_runner"] = MavenRunner

from .random_runner import EpisodeRunner as RandomRunner
REGISTRY["random"] = RandomRunner

