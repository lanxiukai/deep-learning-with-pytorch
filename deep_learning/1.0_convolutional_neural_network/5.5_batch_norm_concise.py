'''
Batch Normalization (Concise Implementation)
'''

from torch import nn
from dl_utils.d2l.cnn import train_ch6
from dl_utils.d2l.data_fashion import load_data_fashion_mnist
from dl_utils.runtime.devices import try_gpu

if __name__ == '__main__':
    net = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
        nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
        nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
        nn.Linear(84, 10))

    lr, num_epochs, batch_size = 1.0, 10, 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    train_ch6(net, train_iter, test_iter, num_epochs, lr,
                       device=try_gpu(), net_name='LeNet-5 with BatchNorm')
    
    # plt.ioff()
    # plt.show()
