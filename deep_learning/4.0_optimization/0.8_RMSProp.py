'''
RMSProp
'''

import math
import torch
from dl_utils.plot._backend import pyplot as plt
from dl_utils.d2l.optim import (
    get_data_ch11,
    train_2d,
    train_ch11,
    train_concise_ch11,
)
from dl_utils.plot.figures import set_figsize, trace2d

# set_figsize()
# gammas = [0.95, 0.9, 0.8, 0.7]
# for gamma in gammas:
#     x = torch.arange(40).detach().numpy()
#     plt.plot(x, (1-gamma) * gamma ** x, label=f'gamma = {gamma:.2f}')
# plt.title('RMSProp smoothing parameter: different gamma')
# plt.xlabel('time')
# plt.ylabel('weight coefficient')
# plt.legend()

def rmsprop_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
    s1 = gamma * s1 + (1 - gamma) * g1 ** 2
    s2 = gamma * s2 + (1 - gamma) * g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta, gamma = 0.4, 0.9
# trace2d(f_2d, train_2d(rmsprop_2d))

# scripted implementation
def init_rmsprop_states(feature_dim):
    s_w = torch.zeros((feature_dim, 1))
    s_b = torch.zeros(1)
    return (s_w, s_b)

def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] = gamma * s + (1 - gamma) * torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()

data_iter, feature_dim = get_data_ch11(batch_size=10)
train_ch11(rmsprop, init_rmsprop_states(feature_dim),
               {'lr': 0.01, 'gamma': 0.9}, data_iter, feature_dim)

# concise implementation
trainer = torch.optim.RMSprop
train_concise_ch11(trainer, {'lr': 0.01, 'alpha': 0.9},
                       data_iter)

# plt.ioff()
# plt.show()
