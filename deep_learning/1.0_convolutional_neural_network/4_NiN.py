'''
NiN
'''

import torch
from torch import nn
from dl_utils.d2l.cnn import train_ch6
from dl_utils.d2l.data_fashion import load_data_fashion_mnist
from dl_utils.runtime.devices import try_gpu

def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU()
    )

def print_net(net):
    X = torch.rand(size=(1, 1, 224, 224), dtype=torch.float32)
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__,'output shape:\t', X.shape)

if __name__ == '__main__':
    net = nn.Sequential(
        nin_block(1, 96, kernel_size=11, stride=4, padding=0),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nin_block(96, 256, kernel_size=5, stride=1, padding=2),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nin_block(256, 384, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Dropout(p=0.5),
        nin_block(384, 10, kernel_size=3, stride=1, padding=1),
        # Global average pooling
        nn.AdaptiveAvgPool2d((1, 1)), # The output shape (H_out, W_out) = (1, 1)
        # Convert the 4D output to a 2D output with shape (batch_size, 10)
        nn.Flatten()
    )

    # print_net(net)

    lr, num_epochs, batch_size = 0.1, 10, 128
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)
    train_ch6(net, train_iter, test_iter, num_epochs, lr,
                       device=try_gpu(), net_name='NiN')
    
    # plt.ioff()
    # plt.show()
