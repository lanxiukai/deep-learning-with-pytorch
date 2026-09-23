'''
Minibatch Stochastic Gradient Descent (Scratch)
'''

import torch

from dl_utils.d2l.optim import get_data_ch11, train_ch11
from dl_utils.plot._backend import pyplot as plt
from dl_utils.plot.figures import plot, set_figsize
from dl_utils.runtime.timing import Timer

timer = Timer()
A = torch.zeros(256, 256)
B = torch.randn(256, 256)
C = torch.randn(256, 256)

# compute A=BC element by element
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j] = torch.dot(B[i, :], C[:, j])
timer.stop()

# compute A=BC column by column
timer.start()
for j in range(256):
    A[:, j] = torch.mv(B, C[:, j])
timer.stop()

# compute A=BC all at once
timer.start()
A = torch.mm(B, C)
timer.stop()

# multiplication and addition as separate operations (fused in practice)
gigaflops = [2/i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')

timer.start()
for j in range(0, 256, 64):  # divide by 4 blocks
    A[:, j:j+64] = torch.mm(B, C[:, j:j+64])
timer.stop()
print(f'performance in Gigaflops: block {2 / timer.times[3]:.3f}')

def sgd(params, states, hyperparams):
    for p in params:
        p.data.sub_(hyperparams['lr'] * p.grad)  # subtract the gradient from the parameter
        p.grad.data.zero_()                      # reset the gradient

def train_sgd(lr, batch_size, num_epochs=2, prefix=''):
    data_iter, feature_dim = get_data_ch11(batch_size)
    return train_ch11(
        sgd, None, {'lr': lr}, data_iter, feature_dim, num_epochs, prefix)

gd_res = train_sgd(1, 1500, 10, prefix='gd')

sgd_res = train_sgd(0.005, 1, prefix='sgd')

mini1_res = train_sgd(.4, 100, prefix='batch size=100')

mini2_res = train_sgd(.05, 10, prefix='batch size=10')

set_figsize([6, 3])
plot(*list(map(list, zip(gd_res, sgd_res, mini1_res, mini2_res))),
         'time (sec)', 'loss', xlim=[1e-2, 10],
         legend=['gd', 'sgd', 'batch size=100', 'batch size=10'])
plt.gca().set_xscale('log')

# plt.ioff()
# plt.show()
