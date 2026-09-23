'''
Minibatch Stochastic Gradient Descent (Concise Implementation)
'''

import torch
from dl_utils.plot._backend import pyplot as plt
from dl_utils.d2l.optim import get_data_ch11, train_concise_ch11

batch_size = 10
data_iter, _ = get_data_ch11(batch_size)
trainer = torch.optim.SGD
train_concise_ch11(
    trainer, {'lr': 0.01}, data_iter, prefix=f'batch size={batch_size}')

# plt.ioff()
# plt.show()
