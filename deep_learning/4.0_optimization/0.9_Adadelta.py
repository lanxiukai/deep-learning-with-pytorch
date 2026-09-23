'''
Adadelta
'''

import torch
from dl_utils.plot._backend import pyplot as plt
from dl_utils.d2l.optim import get_data_ch11, train_ch11, train_concise_ch11

def init_adadelta_states(feature_dim):
    s_w, s_b = torch.zeros((feature_dim, 1)), torch.zeros(1)
    delta_w, delta_b = torch.zeros((feature_dim, 1)), torch.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        with torch.no_grad():
            # In-place updates via [:]
            s[:] = rho * s + (1 - rho) * torch.square(p.grad)
            g = (torch.sqrt(delta + eps) / torch.sqrt(s + eps)) * p.grad
            p[:] -= g
            delta[:] = rho * delta + (1 - rho) * torch.square(g)
        p.grad.data.zero_()

# Adadelta training
data_iter, feature_dim = get_data_ch11(batch_size=10)
train_ch11(adadelta, init_adadelta_states(feature_dim),
               {'rho': 0.9}, data_iter, feature_dim);

# Adadelta concise implementation
trainer = torch.optim.Adadelta
train_concise_ch11(trainer, {'rho': 0.9}, data_iter)

# plt.ioff()
# plt.show()
