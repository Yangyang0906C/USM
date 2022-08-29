import torch as th
import numpy as np
import pickle
from .xmeans import XMeans
from torch.nn import functional as F
from torch.distributions import Categorical
from sklearn.cluster import KMeans
from scipy.stats import entropy


def build_td_lambda_targets(rewards, terminated, mask, target_qs, n_agents, gamma, td_lambda):
    # Assumes  <target_qs > in B*T*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1
    # Initialise  last  lambda -return  for  not  terminated  episodes
    ret = target_qs.new_zeros(*target_qs.shape)
    ret[:, -1] = target_qs[:, -1] * (1 - th.sum(terminated, dim=1))
    # Backwards  recursive  update  of the "forward  view"
    for t in range(ret.shape[1] - 2, -1,  -1):
        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] \
                    * (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
    # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
    return ret[:, 0:-1]

def obs_to_ind(obs, base, raw_dim):
    if th.is_tensor(obs) ==  False:
        obs = np.array(obs)
    else:
        obs = obs.double()
    if raw_dim == 5:
        return obs[:,4] * base**4 + obs[:,0] * base**3 + obs[:,1] * base**2 + obs[:,2] * base + obs[:,3]
    elif raw_dim == 6:
        return obs[:,0] * base**5 + obs[:,1] * base**4 + obs[:,2] * base**3 + obs[:,3] * base**2 + obs[:,4] * base + obs[:,5]
    elif raw_dim == 9:
        ans = obs[:,8] * base**8 + obs[:,0] * base**7 + obs[:,1] * base**6 + obs[:,3] * base**5 + \
              obs[:, 4] * base **4 + obs[:, 6] * base ** 3 + obs[:, 7] * base ** 2 + obs[:, 2] * base +obs[:,5]
        return ans
    elif raw_dim == 10:
        ans = obs[:,9] * base**9 + obs[:,8] * base**8 + obs[:,0] * base**7 + obs[:,1] * base**6 + obs[:,3] * base**5 + \
              obs[:, 4] * base **4 + obs[:, 6] * base ** 3 + obs[:, 7] * base ** 2 + obs[:, 2] * base +obs[:,5]
        return ans




class Cen_counter(object):
    def __init__(self, nstates):
        self.counter = np.zeros(nstates)  # shape,
    def update(self,state):
        self.counter[state] += 1
    def output(self, state):
        return 1./ np.sqrt(self.counter[state])


class Ball_counter(object):
    def __init__(self, dim):
        self.counter = np.zeros((dim,dim))  # shape,
    def update(self,state):
        self.counter[state[0],state[1]] += 1
    def output(self, state):
        if len(state.shape) == 1:
            return 1./ np.sqrt(self.counter[state[0],state[1]])
        else:
            return 1./ np.sqrt(self.counter[state[:,0],state[:,1]])



class RunningMeanStdPWE(object):
    def __init__(self, epsilon=1e-4, shape=(),counter_limit=0):
        self.shape = shape

        self.mask = np.ones(shape, 'float64')
        self.ent = np.zeros(shape,'float64')
        self.save_count = 0

        self.ent_buffer = []
        self.mask_buffer = []
        self.counter = np.zeros([shape, counter_limit],'int')


    def update(self, x):
        x = x[:,:self.shape]
        ind_x = np.array(list(range(self.shape)))
        for i in range(x.shape[0]):
            self.counter[ind_x, x[i,:].astype(np.int64)] += 1


    def save_data(self, path):
        self.mask_buffer.append(self.mask.copy())
        self.ent_buffer.append(self.ent.copy())
        self.save_count += 1
        if self.save_count % 100 == 0:
            file_name = '%s/ent-%d.pkl' % (path, self.save_count)
            with open(file_name, 'wb') as file:
                pickle.dump(self.ent_buffer, file)
            file_name = '%s/mask-%d.pkl' % (path, self.save_count)
            with open(file_name, 'wb') as file:
                pickle.dump(self.mask_buffer, file)
            file_name = '%s/counter-%d.pkl' % (path, self.save_count)
            with open(file_name, 'wb') as file:
                pickle.dump(self.counter, file)

    def get_mask(self):
        self.mask = np.zeros(self.shape, 'float64')
        self.ent = entropy(self.counter.T)
        num = np.log(np.count_nonzero(self.counter,1))
        num[num == 0] = 1
        self.ent = self.ent / num
        ent = self.ent.copy()

        cluster = KMeans(2)
        f = ent.reshape(-1,1)

        cluster.fit(f)
        min_f = np.argmin(ent)
        ind = cluster.labels_[min_f]
        indd = [i for i,v in enumerate(cluster.labels_) if v == ind]
        self.mask[indd] = 1

        return self.mask





