from functools import partial
from .multiagentenv import MultiAgentEnv
from .starcraft2 import StarCraft2Env

from .push_box import PushBox
from .rooms import Rooms
from .hh_island import Island as HHIsland

# from .sh_island import Island as SHIsland

import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)


REGISTRY['pushbox'] = PushBox
REGISTRY['rooms'] = Rooms
REGISTRY['hh_island'] = HHIsland
# REGISTRY['sh_island'] = SHIsland


if sys.platform == "linux":
    # os.environ.setdefault("SC2PATH",
    #                       os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
    # the path of SC2
    os.environ.setdefault("SC2PATH",
                          '/code/pymarl/3rdparty/StarCraftII')
