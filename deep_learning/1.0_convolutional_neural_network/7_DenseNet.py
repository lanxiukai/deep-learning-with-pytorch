'''
DenseNet
'''

import torch
from torch import nn
from dl_utils.d2l.cnn import train_ch6
from dl_utils.d2l.data_fashion import load_data_fashion_mnist
from dl_utils.runtime.devices import try_gpu

def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))

class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # Concatenate the input and output of each block along the channel dimension
            X = torch.cat((X, Y), dim=1)
        return X

# Add a transition layer between dense blocks to decrease the number of channels
def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))  # Reduce the spatial dimension by half

def print_dense_block_shape():
    blk = DenseBlock(2, 3, 10)
    X = torch.randn(4, 3, 8, 8)
    Y = blk(X)
    print(Y.shape)

    blk = transition_block(23, 10)
    print(blk(Y).shape)

if __name__ == '__main__':
    # print_dense_block_shape()

    b1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    # num_channels is the number of channels in the current block
    num_channels, growth_rate = 64, 32  # Initialize the num_channels and growth_rate
    num_convs_in_dense_blocks = [4, 4, 4, 4]
    blks = []

    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        blks.append(DenseBlock(num_convs, num_channels, growth_rate))
        # The number of output channels of the previous dense block
        num_channels += num_convs * growth_rate
        # Add a transition layer between dense blocks to halve the number of channels
        if i != len(num_convs_in_dense_blocks) - 1:
            blks.append(transition_block(num_channels, num_channels // 2))
            num_channels = num_channels // 2

    net = nn.Sequential(
        b1, *blks,
        nn.BatchNorm2d(num_channels), nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(num_channels, 10))

    lr, num_epochs, batch_size = 0.1, 10, 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)
    train_ch6(net, train_iter, test_iter, num_epochs, lr,
                       device=try_gpu(), net_name='DenseNet')
    
    # plt.ioff()
    # plt.show()
