'''
LeNet-5 using GPU
'''

import torch
from torch import nn
from dl_utils.d2l.cnn import train_ch6
from dl_utils.d2l.data_fashion import load_data_fashion_mnist
from dl_utils.runtime.devices import try_gpu

def print_net(net):
    input_tensor = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
    for layer in net:
        input_tensor = layer(input_tensor)
        print(layer.__class__.__name__,'output shape:\t', input_tensor.shape)

if __name__ == '__main__':
    net = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
        nn.Linear(120, 84), nn.Sigmoid(),
        nn.Linear(84, 10))

    # print_net(net)
    lr, num_epochs, batch_size = 0.9, 10, 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    train_ch6(net, train_iter, test_iter, num_epochs, lr,
              device=try_gpu(), net_name='LeNet-5')

    # plt.ioff()
    # plt.show()
