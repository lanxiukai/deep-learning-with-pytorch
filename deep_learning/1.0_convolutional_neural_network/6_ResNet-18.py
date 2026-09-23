'''
ResNet-18
'''

import torch
from torch import nn
from dl_utils.d2l.cnn import Residual, train_ch6
from dl_utils.d2l.data_fashion import load_data_fashion_mnist
from dl_utils.runtime.devices import try_gpu

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                    use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

def print_residual_shape():
    blk = Residual(3,3)
    input_tensor = torch.rand(4, 3, 6, 6)
    output_tensor = blk(input_tensor)
    print(output_tensor.shape)
    blk = Residual(3,6, use_1x1conv=True, strides=2)
    print(blk(input_tensor).shape)

def print_net(net):
    input_tensor = torch.rand(size=(1, 1, 224, 224))
    for layer in net:
        input_tensor = layer(input_tensor)
        print(layer.__class__.__name__,'output shape:\t', input_tensor.shape)

if __name__ == '__main__':
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(64), nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))

    net = nn.Sequential(b1, b2, b3, b4, b5,
                        nn.AdaptiveAvgPool2d((1,1)),
                        nn.Flatten(), nn.Linear(512, 10))

    # print_residual_shape()
    # print_net(net)

    lr, num_epochs, batch_size = 0.05, 10, 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)
    train_ch6(net, train_iter, test_iter, num_epochs, lr,
                       device=try_gpu(), net_name='ResNet-18')
    
    # plt.ioff()
    # plt.show()
