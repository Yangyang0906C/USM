import gym
import numpy as np
import copy

TOP = 0
BOT = 1
LEFT = 2
RIGHT = 3


class Entity(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Door(Entity):
    def __init__(self, x, y):
        super(Door, self).__init__(x, y)
        self.open = 0


class Rooms(gym.Env):
    def __init__(self,args, checkpoint=False):
        # (x1, y1, x2, y2, door_opened)
        self.args = args
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=[self.args.grid_size * 4 + 2])
        self.obs_group_sizes = [2, 2, 1]
        # each agent can choose one branch at each timestep
        self.action_space = gym.spaces.Discrete(self.args.n_actions)
        grid_size = self.args.grid_size
        self.init_agents = [Entity(1 + grid_size // 10, 1 + grid_size // 10), Entity(grid_size // 10, grid_size // 10)]
        self.init_door = Door(grid_size // 2, grid_size // 2)
        self.switches = [Entity(grid_size // 10, int(grid_size * 0.8)), Entity(int(grid_size * 0.8), grid_size // 10)]
        self.n_agents = self.args.n_agents
        self.n_actions =  self.args.n_actions
        self.grid_size = grid_size
        self.init_wall_map = np.zeros((grid_size, grid_size))
        self.init_wall_map[:, self.grid_size // 2] = 1
        self.H = self.args.H
        self.step_count = 0
        self.done = False
        self.agents, self.wall_map, self.door = None, None, None
        self.checkpoint = checkpoint
        self.n_ckpt_bins = 10
        self.ckpt_bins = {'left_switch': np.linspace(1.5 * (self.grid_size // 10) + 1, self._dist(self.init_agents[0], self.switches[0]) - 1, self.n_ckpt_bins),
                          'door': np.linspace(4, self._dist(self.init_agents[0], self.init_door) - 1, self.n_ckpt_bins),
                          'right_switch': np.linspace(1.5 * (self.grid_size // 10) + 1, self._dist(self.init_door, self.switches[1]) - 1, self.n_ckpt_bins)}
        self.ckpts = {'left_switch': [[False, bin] for bin in self.ckpt_bins['left_switch']],
                      'door': [[False, bin] for bin in self.ckpt_bins['door']],
                      'right_switch': [[False, bin] for bin in self.ckpt_bins['right_switch']]}
        self.success_rew = 3 if self.checkpoint else 100
        self.eye = np.eye(self.grid_size)
        self.flag = np.eye(2)
        self.epsilon = 0.01

    def reset(self):
        self.agents = copy.deepcopy(self.init_agents)
        self.wall_map = copy.deepcopy(self.init_wall_map)
        self.door = copy.deepcopy(self.init_door)
        self.step_count = 0
        self.done = False
        self.ckpts = {'left_switch': [[False, bin] for bin in self.ckpt_bins['left_switch']],
                      'door': [[False, bin] for bin in self.ckpt_bins['door']],
                      'right_switch': [[False, bin] for bin in self.ckpt_bins['right_switch']]}
        return np.array([self.agents[0].x, self.agents[0].y, self.agents[1].x, self.agents[1].y, self.door.open])

    def step(self, action):
        assert not self.done, "error: Trying to call step() after an episode is done"
        obs = []
        for agent_id, agent in enumerate(self.agents):
            self._update_agent_location(agent_id, action[agent_id])
            obs.extend([agent.x, agent.y])
        self._update_door_status()
        obs.append(self.door.open)
        self.step_count += 1
        rew = self._reward()
        self.done = True if self.step_count == self.H or rew >= self.success_rew else False
        state_n = [np.array([self.agents[i].x, self.agents[i].y]) for i in range(self.n_agents)]
        info = {'door': self.door.open, 'state': copy.deepcopy(state_n)}

        return rew, self.done, info

    def _update_agent_location(self, agent_id, action):
        x, y = self.agents[agent_id].x, self.agents[agent_id].y
        if action == TOP and y > 0 and self.wall_map[y - 1, x] == 0:
            self.agents[agent_id].y -= 1
        elif action == BOT and y < self.grid_size - 1 and self.wall_map[y + 1, x] == 0:
            self.agents[agent_id].y += 1
        elif action == LEFT and x > 0 and self.wall_map[y, x - 1] == 0:
            self.agents[agent_id].x -= 1
        elif action == RIGHT and x < self.grid_size - 1 and self.wall_map[y, x + 1] == 0:
            self.agents[agent_id].x += 1

    def _update_door_status(self):
        door_radius = self.grid_size // 10
        for switch in self.switches:
            for agent in self.agents:
                if self._dist(agent, switch) <= 1.5 * door_radius:
                    self.door.open = 1
                    self.wall_map[self.door.y, self.door.x] = 0
                    self.wall_map[self.door.y - door_radius: self.door.y + door_radius + 1, self.door.x] = 0
                    return
        self.door.open = 0
        self.wall_map[self.door.x, self.door.y] = 1
        self.wall_map[self.door.y - door_radius: self.door.y + door_radius + 1, self.door.x] = 1


    def _dist(self, e1, e2):
        return np.sqrt((e1.x - e2.x) ** 2 + (e1.y - e2.y) ** 2)

    def _reward(self):
        rew = 0
        if self.checkpoint:
            rew += self._checkpoint_rew()
        for agent in self.agents:
            if agent.x < (self.grid_size / 2 + 1):
                return rew
        rew += self.success_rew
        return rew

    def _checkpoint_rew(self):
        rew = 0

        # door
        if self.door.open:
            for ckpt in self.ckpts['door']:
                for agent in self.agents:
                    if ckpt[0]:
                        continue
                    if agent.x < (self.grid_size / 2 + 1):
                        if self._dist(agent, self.door) <= ckpt[1]:
                            rew += 0.1
                            ckpt[0] = True

        # Right switch
        for ckpt in self.ckpts['right_switch']:
            for agent in self.agents:
                if ckpt[0]:
                    continue
                if agent.x >= (self.grid_size / 2 + 1):
                    if self._dist(agent, self.switches[1]) <= ckpt[1]:
                        rew += 0.1
                        ckpt[0] = True

        return rew

    def get_state(self):
        return np.concatenate([self.eye[self.agents[0].x], self.eye[self.agents[0].y],
                               self.eye[self.agents[1].x], self.eye[self.agents[1].y],
                               self.flag[self.door.open]]).copy()

    def get_state_mini(self):
        return np.array([self.agents[0].x, self.agents[0].y, self.agents[1].x, self.agents[1].y, self.door.open])

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = [1 for i in range(self.n_actions)]
            avail_actions.append(avail_agent)
        return avail_actions

    def get_env_info(self):
        env_info = {"state_shape": self.grid_size*4 + 2,
                    "mini_state_shape": 5,
                    "obs_shape": self.grid_size*4 + 2,
                    "n_actions": self.n_actions,
                    "n_agents": self.n_agents,
                    "episode_limit": self.H,
                    "nvec": list([self.grid_size,self.grid_size,self.grid_size,self.grid_size,2])}
        return env_info

    def obs_n(self):
        return [self.obs(i) for i in range(self.n_agents)]

    # def observations_c(self):
    #     return [self.observation_c(i) for i in range(self.n_agent)]

    def obs(self, i):
        return np.concatenate([self.eye[self.agents[i].x], self.eye[self.agents[i].y],
                               self.eye[self.agents[1-i].x], self.eye[self.agents[1-i].y],
                               self.flag[self.door.open]]).copy()

