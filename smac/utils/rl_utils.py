import torch as th
import numpy as np
import pickle
from .xmeans import XMeans
from torch.nn import functional as F
from torch.distributions import  Categorical
from sklearn.cluster import KMeans

def build_td_lambda_targets(rewards, terminated, mask, target_qs, gamma, td_lambda):
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

class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=(),mask_shape=()):
        self.shape = shape
        # no_action
        self.mask_shape = mask_shape
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.f_min = np.ones(shape, 'float64')
        self.f_max = -1 * np.ones(shape, 'float64')
        self.mask = np.zeros(mask_shape, 'float64')
        self.count = epsilon
        self.epsilon = epsilon
        self.save_count = 0
        self.var_buffer = []
        self.mean_buffer = []
        self.f_min_buffer = []
        self.f_max_buffer = []
        self.mask_buffer = []


    def update(self, x):
        x = x[:,:self.shape]
        tmp_min = np.vstack((x, self.f_min))
        self.f_min = np.min(tmp_min, axis=0)
        tmp_max = np.vstack((x, self.f_max))
        self.f_max = np.max(tmp_max, axis=0)
        tmp_delta = self.f_max - self.f_min
        ind = [i for i,v in enumerate(tmp_delta) if v==0]
        tmp_delta[ind] = 1
        x = (x -self.f_min) / tmp_delta
        batch_mean, batch_std, batch_count = np.mean(x, axis=0), np.std(x, axis=0), x.shape[0]
        batch_var = np.square(batch_std)
        self.update_from_moments(batch_mean, batch_var, batch_count)


    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def save_data(self, path):
        self.var_buffer.append(self.var)
        self.mask_buffer.append(self.mask.copy())
        self.save_count += 1
        if self.save_count % 100 == 0:
            file_name = '%s/var-%d.pkl' % (path, self.save_count)
            with open(file_name, 'wb') as file:
                pickle.dump(self.var_buffer, file)
            file_name = '%s/mask-%d.pkl' % (path, self.save_count)
            with open(file_name, 'wb') as file:
                pickle.dump(self.mask_buffer, file)

    def get_mask(self):
        self.mask = np.zeros(self.mask_shape, 'float64')
        var_c = self.var
        max_var = np.max(var_c)
        var_c[var_c==0] = max_var
        # 1
        cluster = KMeans(2)
        # 3
        # cluster = XMeans(kmax=len(var_c))
        f = var_c.reshape(-1,1)
        cluster.fit(f)
        min_f = np.argmin(var_c)
        ind = cluster.labels_[min_f]
        indd = [i for i,v in enumerate(cluster.labels_) if v == ind]
        self.mask[indd] = 1
        return self.mask

def flatten(tensor):
    return tensor.view(tensor.size(0), -1)








