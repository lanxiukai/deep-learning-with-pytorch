'''
Momentum Demo
'''

import torch
from dl_utils.plot._backend import pyplot as plt
from dl_utils.d2l.optim import train_2d
from dl_utils.plot.figures import set_figsize, trace2d

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

# trace2d(f_2d, train_2d(gd_2d))

# eta = 0.6
# trace2d(f_2d, train_2d(gd_2d))

def momentum_2d(x1, x2, v1, v2):
    v1 = beta * v1 + 0.2 * x1
    v2 = beta * v2 + 4 * x2
    return x1 - eta * v1, x2 - eta * v2, v1, v2

# eta, beta = 0.6, 0.5
# trace2d(f_2d, train_2d(momentum_2d))

# eta, beta = 0.6, 0.25
# trace2d(f_2d, train_2d(momentum_2d))

set_figsize()
betas = [0.95, 0.9, 0.6, 0]
for beta in betas:
    x = torch.arange(40).detach().numpy()
plt.plot(x, beta ** x, label=f'beta = {beta:.2f}')
plt.xlabel('time')
plt.legend()

# plt.ioff()
# plt.show()
