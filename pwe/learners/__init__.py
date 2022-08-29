from .q_learner import QLearner

from .q_lambda_learner import QLearner as QLambdaLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner

REGISTRY["q_lambda_learner"] = QLambdaLearner
