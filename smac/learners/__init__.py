from .q_learner import QLearner
from .q_uesm_l_learner import QLearner as QUesmLLearner


REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["q_uesm_l_learner"] = QUesmLLearner
