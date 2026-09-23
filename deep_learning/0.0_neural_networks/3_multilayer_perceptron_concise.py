'''
Multilayer Perceptron (Concise Implementation)
'''

import torch
from torch import nn

from dl_utils.d2l.data_fashion import (
    load_data_fashion_mnist,
    predict_ch3,
    train_ch3,
)
from dl_utils.runtime.timing import Timer


def init_weights(m):
    '''Initialize the weights of the linear layer.'''
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

if __name__ == '__main__':
    num_inputs, num_outputs, num_hiddens = 784, 10, 256
    batch_size, lr, num_epochs = 256, 0.1, 10

    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    net = nn.Sequential(nn.Flatten(), 
                        nn.Linear(num_inputs, num_hiddens), 
                        nn.ReLU(), 
                        nn.Linear(num_hiddens, num_outputs))
    net.apply(init_weights)

    loss = nn.CrossEntropyLoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    timer = Timer()
    train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
    predict_ch3(net, test_iter)
    print(f'{timer.stop():.3f} sec')
    # plt.ioff()
    # plt.show()
