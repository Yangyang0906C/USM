import os
import math
import pickle
import shutil
import numpy as np
# import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import copy


def mkdir(path):
	if os.path.exists(path):
		shutil.rmtree(path)
	os.makedirs(path)

INTERVAL = 50


class Gather_lc_3:

	def __init__(self, save_path, args, name):
		self.args = args
		self.name = name
		self.save_path = save_path + name + '/'
		self.p_step = 0
		self.buffer = []

		mkdir(self.save_path + '/figure/')
		mkdir(self.save_path + '/data/')

	def smooth(self, data):
		smoothed = []
		for i in range(len(data)):
			smoothed.append(np.mean(data[max(0, i + 1 - 100): i + 1]))
		return smoothed

	def save_img(self):
		data_s = self.smooth(self.buffer)
		m_figure = plt.figure(figsize=(16, 10))
		m_ax1 = m_figure.add_subplot(1, 2, 1)
		m_ax2 = m_figure.add_subplot(1, 2, 2)
		m_ax1.plot(self.buffer)
		m_ax2.plot(data_s)
		m_figure.savefig('%s/figure/%d.png' % (self.save_path, self.p_step))
		plt.close(m_figure)

	def save_data(self):
		filename = '%s/data/%d.pkl' % (self.save_path, self.p_step)
		with open(filename, 'wb') as file:
			pickle.dump(self.buffer, file)

	def update(self, data):
		self.buffer.append(data)
		self.p_step += 1
		if self.p_step % INTERVAL == 0:
			self.save_data()
			self.save_img()


class Gather_hm_2:

	def __init__(self, save_path, args, name):
		self.args = args
		self.name = name
		self.save_path = save_path + name + '/'
		self.p_step = 0

		mkdir(self.save_path + '/figure/')
		mkdir(self.save_path + '/data/')

		self.visited = [np.zeros([2, args.size, args.size]) for _ in range(self.args.n_agent)]
		self.visited_old = [np.zeros([2, args.size, args.size]) for _ in range(self.args.n_agent)]
		self.value = [np.zeros([2, args.size, args.size]) for _ in range(self.args.n_agent)]
		self.value_old = [np.zeros([2, args.size, args.size]) for _ in range(self.args.n_agent)]

	def normal(self, data, total):
		t_data = data.copy()
		t_total = total.copy()
		zero = total == 0
		t_total[zero] = 1
		t_data = t_data / t_total
		t_min = np.min(t_data)
		t_max = np.max(t_data)
		t_data[zero] = t_min
		t_data = (t_data - t_min) / (t_max - t_min + 1)
		return np.log(t_data + 1)

	def save_img(self):

		figure = plt.figure(figsize=(16, 10))

		ax1 = figure.add_subplot(2, 4, 1)
		ax2 = figure.add_subplot(2, 4, 2)
		ax3 = figure.add_subplot(2, 4, 3)
		ax4 = figure.add_subplot(2, 4, 4)
		ax5 = figure.add_subplot(2, 4, 5)
		ax6 = figure.add_subplot(2, 4, 6)
		ax7 = figure.add_subplot(2, 4, 7)
		ax8 = figure.add_subplot(2, 4, 8)

		ax1.imshow(self.normal(self.value[0][0], self.visited[0][0]))
		ax2.imshow(self.normal(self.value[0][0] - self.value_old[0][0],
		                       self.visited[0][0] - self.visited_old[0][0]))
		ax3.imshow(self.normal(self.value[0][1], self.visited[0][1]))
		ax4.imshow(self.normal(self.value[0][1] - self.value_old[0][1],
		                       self.visited[0][1] - self.visited_old[0][1]))

		ax5.imshow(self.normal(self.value[1][0], self.visited[1][0]))
		ax6.imshow(self.normal(self.value[1][0] - self.value_old[1][0],
		                       self.visited[1][0] - self.visited_old[1][0]))
		ax7.imshow(self.normal(self.value[1][1], self.visited[1][1]))
		ax8.imshow(self.normal(self.value[1][1] - self.value_old[1][1],
		                       self.visited[1][1] - self.visited_old[1][1]))

		figure.savefig('%s/figure/%d.png' % (self.save_path, self.p_step))
		plt.close(figure)

	def save_data(self):
		filename = '%s/data/visited_%d.pkl' % (self.save_path, self.p_step)
		with open(filename, 'wb') as file:
			pickle.dump(self.visited, file)
		filename = '%s/data/visited_old_%d.pkl' % (self.save_path, self.p_step)
		with open(filename, 'wb') as file:
			pickle.dump(self.visited_old, file)
		filename = '%s/data/value_%d.pkl' % (self.save_path, self.p_step)
		with open(filename, 'wb') as file:
			pickle.dump(self.value, file)
		filename = '%s/data/value_old_%d.pkl' % (self.save_path, self.p_step)
		with open(filename, 'wb') as file:
			pickle.dump(self.value_old, file)

	def save_trajectory(self):
		figure = plt.figure(figsize=(16, 10))

		ax1 = figure.add_subplot(2, 2, 1)
		ax2 = figure.add_subplot(2, 2, 2)
		ax3 = figure.add_subplot(2, 2, 3)
		ax4 = figure.add_subplot(2, 2, 4)

		ax1.imshow(np.log(self.visited[0][0] + 1))
		ax2.imshow(np.log(self.visited[0][0] - self.visited_old[0][0] + 1))
		ax3.imshow(np.log(self.visited[1][0] + 1))
		ax4.imshow(np.log(self.visited[1][0] - self.visited_old[1][0] + 1))

		figure.savefig('%s/figure/%d.png' % (self.save_path, self.p_step))
		plt.close(figure)

	def update_trajectory(self, infos):
		for item_id, item in enumerate(infos):
			state = item['state']
			for j in range(self.args.n_agent):
				self.visited[j][0][state[j][1]][state[j][0]] += 1
		self.p_step += 1
		if self.p_step % INTERVAL == 0:
			self.save_data()
			self.save_trajectory()
			self.visited_old = [v.copy() for v in self.visited]

	def update_rew(self, infos, rew):
		for item_id, item in enumerate(infos):
			pre_state = item['state']
			if pre_state != None:
				for j in range(self.args.n_agent):
					self.value[j][0][pre_state[j][1]][pre_state[j][0]] += rew[item_id]
					self.visited[j][0][pre_state[j][1]][pre_state[j][0]] += 1
					self.value[1 - j][1][pre_state[1 - j][1]][pre_state[1 - j][0]] += rew[item_id]
					self.visited[1 - j][1][pre_state[1 - j][1]][pre_state[1 - j][0]] += 1
		self.p_step += 1
		if self.p_step % INTERVAL == 0:
			self.save_data()
			self.save_img()
			self.visited_old = [v.copy() for v in self.visited]
			self.value_old = [v.copy() for v in self.value]

class Gather_hm_2_ball:

	def __init__(self, save_path, args, name):
		self.args = args
		self.name = name
		self.save_path = save_path + name + '/'
		self.p_step = 0

		mkdir(self.save_path + '/figure/')
		mkdir(self.save_path + '/data/')

		self.visited = np.zeros([args.size, args.size])
		self.visited_old = np.zeros([args.size, args.size])


	def normal(self, data, total):
		t_data = data.copy()
		t_total = total.copy()
		zero = total == 0
		t_total[zero] = 1
		t_data = t_data / t_total
		t_min = np.min(t_data)
		t_max = np.max(t_data)
		t_data[zero] = t_min
		t_data = (t_data - t_min) / (t_max - t_min + 1)
		return np.log(t_data + 1)


	def save_data(self):
		filename = '%s/data/visited_%d.pkl' % (self.save_path, self.p_step)
		with open(filename, 'wb') as file:
			pickle.dump(self.visited, file)
		filename = '%s/data/visited_old_%d.pkl' % (self.save_path, self.p_step)
		with open(filename, 'wb') as file:
			pickle.dump(self.visited_old, file)

	def save_trajectory(self):
		figure = plt.figure(figsize=(16, 10))

		ax1 = figure.add_subplot(2, 2, 1)
		ax2 = figure.add_subplot(2, 2, 2)
		ax3 = figure.add_subplot(2, 2, 3)
		ax4 = figure.add_subplot(2, 2, 4)

		ax1.imshow(np.log(self.visited + 1))
		ax2.imshow(np.log(self.visited - self.visited_old + 1))
		ax3.imshow(np.log(self.visited + 1))
		ax4.imshow(np.log(self.visited - self.visited_old + 1))

		figure.savefig('%s/figure/%d.png' % (self.save_path, self.p_step))
		plt.close(figure)

	def update_trajectory(self, infos):
		for item_id, item in enumerate(infos):
			state = item['ball']
			self.visited[state[1]][state[0]] += 1
		self.p_step += 1
		if self.p_step % INTERVAL == 0:
			self.save_data()
			self.save_trajectory()
			self.visited_old = copy.deepcopy(self.visited)



class Gather_lc_2:

	def __init__(self, save_path, args, name):
		self.args = args
		self.name = name
		self.save_path = save_path + name + '/'
		self.gather = [Gather_lc_3(self.save_path, self.args, 'agent_%d' % i) for i in range(self.args.n_agent)]

	def update(self, data):
		for i in range(self.args.n_agent):
			self.gather[i].update(data[i])


class Gather_hm:

	def __init__(self, save_path, args, name):
		self.args = args
		self.name = name
		self.save_path = save_path + name + '/'
		self.trajectory = Gather_hm_2(self.save_path, self.args, 'trajectory')
		self.ball_trajectory = Gather_hm_2_ball(self.save_path, self.args, 'ball_trajectory')

		self.rew_int = Gather_hm_2(self.save_path, self.args, 'rew_int')
		if self.args.alg_name == "qmix_usem":
			self.rew_counter = Gather_hm_2(self.save_path, self.args, 'rew_counter')
			self.rew_mask = Gather_hm_2(self.save_path, self.args, 'rew_mask')



	def update(self, data):

		rew_int, rew_counter, rew_mask, infos_list = data

		self.trajectory.update_trajectory(infos_list)

		self.rew_int.update_rew(infos_list, rew_int)
		if self.args.alg_name == "qmix_usem":
			self.rew_counter.update_rew(infos_list, rew_counter)
			self.rew_mask.update_rew(infos_list, rew_mask)


class Gather_lc:

	def __init__(self, save_path, args, name):
		self.args = args
		self.name = name
		self.save_path = save_path + name + '/'
		self.rew_ext = Gather_lc_3(self.save_path, self.args, 'rew_ext')
		self.rew_int = Gather_lc_3(self.save_path, self.args, 'rew_int')
		if self.args.alg_name == "qmix_usem":
			self.rew_counter = Gather_lc_3(self.save_path, self.args, 'rew_counter')
			self.rew_mask = Gather_lc_3(self.save_path, self.args, 'rew_mask')
		if self.args.map_name == 'hh_island':
			self.time_length = Gather_lc_2(self.save_path, self.args, 'time_length')
			self.death = Gather_lc_2(self.save_path, self.args, 'death')
			self.kill = Gather_lc_3(self.save_path, self.args, 'kill')
			self.landmark = Gather_lc_3(self.save_path, self.args, 'landmark')


	def update(self, data, total_data):

		rew_ext, rew_int, rew_counter, rew_mask, time_length_list, death_list = data
		kill_list, landmark_list = total_data
		self.rew_ext.update(rew_ext)
		self.rew_int.update(rew_int)
		if self.args.map_name == 'hh_island':
			self.time_length.update(time_length_list)
			self.death.update(death_list)
			self.kill.update(kill_list)
			self.landmark.update(landmark_list)

		if self.args.alg_name == "qmix_usem":
			self.rew_counter.update(rew_counter)
			self.rew_mask.update(rew_mask)






class Gather_update:

	def __init__(self, save_path, args, name):
		self.args = args
		self.name = name
		self.save_path = save_path + name + '/'+'try_%d' % self.args.s_try_num + '/'
		self.learning_curve_gather = Gather_lc(self.save_path, self.args, 'learning_curve')
		self.heatmap_gather = Gather_hm(self.save_path, self.args, 'heatmap')

	def merge_reward(self, rew):
		rew = np.sum(np.array(rew))
		return rew

	def merge_data(self, data):

		rew_ext, rew_int, rew_counter, rew_mask, done_list, infos_list, kill_list, lm_list, death_list, time_list = data

		hm_list = [copy.deepcopy(rew_int),copy.deepcopy(rew_counter),copy.deepcopy(rew_mask),infos_list]

		num_episode = np.sum(done_list) #+ self.args.num_env



		rew_data = []
		rew = (rew_ext, rew_int, rew_counter, rew_mask)
		for item in rew:
			rew_data.append(1. * self.merge_reward(item) / num_episode)

		single_rew = (time_list, death_list)
		for item in single_rew:
			rew_data.append(1. * np.sum(item, axis=1) / num_episode)

		total_rew = (kill_list, lm_list)
		total_rew_data = []
		for item in total_rew:
			total_rew_data.append(1. * np.sum(item) / num_episode)


		return rew_data, total_rew_data, hm_list

	def update(self, data):
		t_rew, t_total_rew, t_hm = self.merge_data(data)
		self.learning_curve_gather.update(t_rew, t_total_rew)
		self.heatmap_gather.update(t_hm)


class Gather:

	def __init__(self, args):
		self.args = args
		self.save_path = self.args.s_data_path + self.args.map_name + '/'
		self.update_gather = Gather_update(self.save_path, self.args, 'update')

	def update(self, data):
		self.update_gather.update(data)
