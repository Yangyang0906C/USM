import numpy as np
import copy
from .rooms import Entity
import gym


TOP = 0
BOT = 1
LEFT = 2
RIGHT = 3


class Box(Entity):
    def __init__(self, x, y):
        super(Box, self).__init__(x, y)
        self.radius = 1
        self.force = np.zeros(4)


class PushBox(object):
    def __init__(self, args, checkpoint=False):
        # (x1, y1, x2, y2, door1_opened, door2_opened, door3_opened)
        self.args = args
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=[self.args.grid_size*2*self.args.n_agents + self.args.grid_size*2])
        # each agent can choose one branch at each timestep
        self.action_space = gym.spaces.Discrete(self.args.n_actions)
        self.init_agents = [Entity(4 + self.args.grid_size // 2, 4 + self.args.grid_size // 2), Entity(2 + self.args.grid_size // 2, 2 + self.args.grid_size // 2)]
        self.init_box = Box(self.args.grid_size // 2, self.args.grid_size // 2)
        self.wall_map = np.zeros((self.args.grid_size, self.args.grid_size))
        self.n_agents = self.args.n_agents
        self.n_actions = self.args.n_actions
        self.grid_size = self.args.grid_size
        self.H = self.args.H
        self.step_count = 0
        self.done = False
        self.agents, self.box = None, None
        self.checkpoint = checkpoint
        if self.checkpoint:
            self.cur_checkpoint = {'dir':-1, 'dist':0}
        self.success_rew = 100
        self.eye = np.eye(self.args.grid_size)

    def reset(self):
        self.agents = copy.deepcopy(self.init_agents)
        self.box = copy.deepcopy(self.init_box)
        self.ball_n = np.array([self.box.x, self.box.y])
        self.state_n = [np.array([agent.x, agent.y]) for agent_id, agent in enumerate(self.agents)]
        self._update_wall()
        self.step_count = 0
        self.done = False
        if self.checkpoint:
            self.cur_checkpoint = {'dir':-1, 'dist':0}
        return np.array([self.agents[0].x, self.agents[0].y, self.agents[1].x, self.agents[1].y, self.box.x, self.box.y])

    def step(self, action):
        assert not self.done, "error: Trying to call step() after an episode is done"
        obs = []
        self._compute_force(action)
        self._update_box_location()
        for agent_id, agent in enumerate(self.agents):
            self._update_agent_location(agent_id, action[agent_id])
            obs.extend([agent.x, agent.y])
        obs.extend([self.box.x, self.box.y])
        self.step_count += 1
        rew = self._reward()
        self.done = True if self.step_count == self.H or rew >= self.success_rew else False
        self.ball_n = np.array([self.box.x, self.box.y])
        self.state_n = [np.array([agent.x, agent.y]) for agent_id, agent in enumerate(self.agents)]
        info_state_n = []
        for i, state in enumerate(self.state_n):
            full_state = np.concatenate([state, self.ball_n])
            info_state_n.append(full_state)

        info = {'ball': self.ball_n, 'state': copy.deepcopy(info_state_n)}

        return rew, self.done, info

    def _compute_force(self, actions):
        # compute force on the box
        self.box.force[:] = 0
        box_top = self.box.y - self.box.radius
        box_bot = self.box.y + self.box.radius
        box_left = self.box.x - self.box.radius
        box_right = self.box.x + self.box.radius
        for i, agent in enumerate(self.agents):
            if box_top <= agent.y <= box_bot:
                if agent.x == box_right + 1 and actions[i] == LEFT:
                    self.box.force[LEFT] += 1
                elif agent.x == box_left - 1 and actions[i] == RIGHT:
                    self.box.force[RIGHT] += 1
            if box_left <= agent.x <= box_right:
                if agent.y == box_top - 1 and actions[i] == BOT:
                    self.box.force[BOT] += 1
                elif agent.y == box_bot + 1 and actions[i] == TOP:
                    self.box.force[TOP] += 1

    def _update_box_location(self):
        idx = np.where(self.box.force >= 2)[0]
        if idx.size > 0:
            idx = idx.item()
            if idx == TOP and self.box.y - self.box.radius > 0:
                self.box.y -= 1
            elif idx == BOT and self.box.y + self.box.radius < self.grid_size - 1:
                self.box.y += 1
            elif idx == LEFT and self.box.x - self.box.radius > 0:
                self.box.x -= 1
            elif idx == RIGHT and self.box.x + self.box.radius < self.grid_size - 1:
                self.box.x += 1
            self._update_wall()

    def _update_wall(self):
        self.wall_map[:] = 0
        self.wall_map[max(0, self.box.y - self.box.radius) : min(self.grid_size, self.box.y + self.box.radius + 1),
            max(0, self.box.x - self.box.radius) : min(self.grid_size, self.box.x + self.box.radius + 1)] = 1

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

    def _dist(self, e1, e2):
        return np.sqrt((e1.x - e2.x) ** 2 + (e1.y - e2.y) ** 2)

    def _reward(self):
        rew = 0
        # self.box.y + self.box.radius == self.grid_size - 1 ---one direction
        if self.box.x - self.box.radius == 0 or self.box.x + self.box.radius == self.grid_size - 1 \
                or self.box.y - self.box.radius == 0 or self.box.y + self.box.radius == self.grid_size - 1:
            rew += self.success_rew
        if self.checkpoint:
            rew += self._check_point_rew()
        return rew

    def _check_point_rew(self):
        rew = 0
        if self.cur_checkpoint['dir'] == -1:
            if self.box.y > self.init_box.y:
                self.cur_checkpoint['dir'] = 0
            if self.box.x > self.init_box.x:
                self.cur_checkpoint['dir'] = 1
            if self.box.y < self.init_box.y:
                self.cur_checkpoint['dir'] = 2
            if self.box.x < self.init_box.x:
                self.cur_checkpoint['dir'] = 3
        elif self.cur_checkpoint['dir'] == 0:
            if self.box.y - self.init_box.y > self.cur_checkpoint['dist']:
                rew += 0.1
                self.cur_checkpoint['dist'] = self.box.y - self.init_box.y
        elif self.cur_checkpoint['dir'] == 1:
            if self.box.x - self.init_box.x > self.cur_checkpoint['dist']:
                rew += 0.1
                self.cur_checkpoint['dist'] = self.box.x - self.init_box.x
        elif self.cur_checkpoint['dir'] == 2:
            if self.init_box.y - self.box.y > self.cur_checkpoint['dist']:
                rew += 0.1
                self.cur_checkpoint['dist'] = self.init_box.y - self.box.y
        elif self.cur_checkpoint['dir'] == 3:
            if self.init_box.x - self.box.x > self.cur_checkpoint['dist']:
                rew += 0.1
                self.cur_checkpoint['dist'] = self.init_box.x - self.box.x
        return rew



    def get_state(self):
        ball = np.concatenate([self.eye[self.ball_n[0]], self.eye[self.ball_n[1]]], axis=0)
        return np.concatenate([self.eye[self.state_n[0][0]], self.eye[self.state_n[0][1]],
                               self.eye[self.state_n[1][0]], self.eye[self.state_n[1][1]],
                               ball]).copy()
    def get_state_mini(self):
        return np.concatenate([self.state_n[0],self.state_n[1], self.ball_n],axis=0).copy()

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = [1 for i in range(self.n_actions)]
            avail_actions.append(avail_action)
        return avail_actions

    def get_env_info(self):
        env_info = {"state_shape": self.grid_size * 6,
                    "mini_state_shape": 6,
                    "obs_shape": self.grid_size * 6,
                    "n_actions": self.n_actions,
                    "n_agents": self.n_agents,
                    "episode_limit": self.H,
                    "nvec": [self.grid_size for i in range(6)]}
        return env_info

    def obs_n(self):
        return [self.obs(i) for i in range(self.n_agents)]

    def obs(self, i):
        ball = np.concatenate([self.eye[self.ball_n[0]], self.eye[self.ball_n[1]]], axis=0)
        return np.concatenate([self.eye[self.state_n[i][0]], self.eye[self.state_n[i][1]],
                               self.eye[self.state_n[1-i][0]], self.eye[self.state_n[1-i][1]],
                               ball]).copy()












