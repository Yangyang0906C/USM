from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.buffer.episode_buffer import EpisodeBatch
import numpy as np
from data_gather import Gather
import copy
from types import SimpleNamespace as SN
from utils.rl_utils import obs_to_ind

class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1
        env_args = SN(**self.args.env_args)
        self.env = env_REGISTRY[self.args.env](env_args)
        self.episode_limit = self.env.args.episode_length
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}
        self.info_list = []
        self.done_list = []
        self.rew_ext = []
        self.rew_int = []
        self.rew_counter = []
        self.rew_mask = []
        self.kill_list = []
        self.landmark_list = []
        self.time_length_list = [[] for i in range(env_args.n_agent)]
        self.death_list = [[] for i in range(env_args.n_agent)]
        self.data_gather = Gather(env_args)

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, counter,state_rms, args, scheme, groups, preprocess, mac):
        self.counter = counter
        self.state_rms = state_rms
        self.base = args.env_args["grid_size"]
        self.raw_dim = args.state_mini_shape
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac


    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0
        self.pre_state = None



    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        rwd_int_l = self.args.rwd_int_lambda

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "mini_state": [self.env.get_state_mini()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.obs_n()]
            }

            ind = obs_to_ind(pre_transition_data["mini_state"], self.base, self.raw_dim)


            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            reward, terminated, env_info = self.env.step(actions[0])
            # if reward >0:
            #     print('here')
            episode_return += reward

            if not test_mode:
                self.counter.update(ind)
                self.rew_ext.append(reward)
                self.info_list.append(env_info)
                self.done_list.append(terminated)
                if self.args.env == 'hh_island':
                    info_r = env_info['rew']
                    self.kill_list.append(info_r['kill'])
                    self.landmark_list.append(info_r['landmark'])
                    for j, death in enumerate(info_r['death']):
                        self.death_list[j].append(int(death))
                        self.time_length_list[j].append(info_r['time_length'][j])
                if self.t != 0:
                    reward_counter = self.counter.output(ind)
                    state_mask = (pre_transition_data["mini_state"][0] != self.pre_state) * self.state_rms.mask
                    reward_mask = np.sum(state_mask) / np.sum(self.state_rms.mask)
                    reward_int = rwd_int_l * reward_counter * reward_mask
                    self.rew_int.append(reward_int)
                    self.rew_counter.append(reward_counter)
                    self.rew_mask.append(reward_mask)


            self.pre_state = pre_transition_data["mini_state"][0]
            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "mini_state": [self.env.get_state_mini()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.obs_n()]
        }
        ind = obs_to_ind(last_data["mini_state"], self.base, self.raw_dim)
        self.batch.update(last_data, ts=self.t)
        if not test_mode:
            self.counter.update(ind)
            reward_counter = self.counter.output(ind)
            state_mask = (last_data["mini_state"][0] != self.pre_state) * self.state_rms.mask
            reward_mask = np.sum(state_mask) / np.sum(self.state_rms.mask)

            reward_int = rwd_int_l * reward_counter * reward_mask
            self.rew_int.append(reward_int)
            self.rew_counter.append(reward_counter)
            self.rew_mask.append(reward_mask)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        # cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env
            t_data = (self.rew_ext, self.rew_int, self.rew_counter, self.rew_mask, self.done_list, self.info_list, self.kill_list, \
                      self.landmark_list, self.death_list, self.time_length_list)
            t_data = copy.deepcopy(t_data)
            self.data_gather.update(t_data)
            self.info_list = []
            self.done_list = []
            self.rew_ext = []
            self.rew_int = []
            self.rew_mask = []
            self.rew_counter = []
            self.kill_list = []
            self.landmark_list = []
            self.time_length_list = [[] for i in range(self.args.env_args["n_agent"])]
            self.death_list = [[] for i in range(self.args.env_args["n_agent"])]


        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
