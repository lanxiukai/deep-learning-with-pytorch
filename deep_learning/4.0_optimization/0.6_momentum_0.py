'''
Minibatch Stochastic Gradient Descent with Momentum
'''

import torch
from dl_utils.plot._backend import pyplot as plt
from dl_utils.d2l.optim import get_data_ch11, train_ch11
from dl_utils.plot.figures import set_figsize

def init_momentum_states(feature_dim):
    # Velocity variables accumulate gradient history; 
    # initialized to zero with same shape as parameters (w and b)
    v_w = torch.zeros((feature_dim, 1))
    v_b = torch.zeros(1)
    return (v_w, v_b)

def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        with torch.no_grad():
            v[:] = hyperparams['momentum'] * v + p.grad
            p[:] -= hyperparams['lr'] * v
        p.grad.data.zero_()

def train_momentum(lr, momentum, num_epochs=2):
    train_ch11(sgd_momentum, init_momentum_states(feature_dim),
                   {'lr': lr, 'momentum': momentum}, data_iter,
                   feature_dim, num_epochs)

data_iter, feature_dim = get_data_ch11(batch_size=10)
# train_momentum(0.02, 0.5)

# train_momentum(0.01, 0.9)

# train_momentum(0.005, 0.9)

# trainer = torch.optim.SGD
# train_concise_ch11(trainer, {'lr': 0.005, 'momentum': 0.9}, data_iter)

lambdas = [0.1, 1, 10, 19]
eta = 0.1
set_figsize((6, 4))
for lam in lambdas:
    t = torch.arange(20).detach().numpy()
plt.plot(t, (1 - eta * lam) ** t, label=f'lambda = {lam:.2f}')
plt.xlabel('time')
plt.legend()

# plt.ioff()
# plt.show()
