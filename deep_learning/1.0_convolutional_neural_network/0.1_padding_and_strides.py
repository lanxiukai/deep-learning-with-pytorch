'''
Padding and Strides
'''

import torch
from torch import nn

# We define a function to compute the convolution for convenience. This function initializes the 
# convolution layer weights and expands or reduces the corresponding dimensions of the input and output.
def comp_conv2d(conv2d, X):
    # Here, (1, 1) indicates that both the batch size and the number of channels are 1.
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # Omit the first two dimensions: batch size and channels.
    return Y.reshape(Y.shape[2:])

# Note that one row or column is padded on each side, adding a total of two rows or columns.
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))
print(comp_conv2d(conv2d, X).shape)

conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
print(comp_conv2d(conv2d, X).shape)

conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
print(comp_conv2d(conv2d, X).shape)

conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
print(comp_conv2d(conv2d, X).shape)
