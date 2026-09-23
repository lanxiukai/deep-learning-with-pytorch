'''
Convolution Layer
'''

import torch
from torch import nn
from dl_utils.d2l.cnn import corr2d

class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, X):
        return corr2d(X, self.weight) + self.bias

print('0--------------------------------')
input_tensor = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
initial_kernel = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print(corr2d(input_tensor, initial_kernel))

input_tensor = torch.ones((6, 8))
input_tensor[:, 2:6] = 0
print(input_tensor)

edge_kernel = torch.tensor([[1.0, -1.0]])
correlation_output = corr2d(input_tensor, edge_kernel)
print(correlation_output)
print(corr2d(input_tensor.t(), edge_kernel))
print('1--------------------------------')

# Construct a 2D convolutional layer with 1 output channel and a convolution kernel of shape (1, 2)
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)

# This 2D convolutional layer uses a four-dimensional input and output format 
# (batch size, channels, height, width)
# where both the batch size and the number of channels are 1
convolution_input = input_tensor.reshape((1, 1, 6, 8))
convolution_output = correlation_output.reshape((1, 1, 6, 7))
lr = 3e-2

for i in range(10):
    prediction = conv2d(convolution_input)
    l = (prediction - convolution_output) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # Iterate over the convolution kernel
    gradient = conv2d.weight.grad
    assert gradient is not None
    conv2d.weight.data[:] -= lr * gradient
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {l.sum():.3f}')

print(conv2d.weight.data.reshape(1, 2))
