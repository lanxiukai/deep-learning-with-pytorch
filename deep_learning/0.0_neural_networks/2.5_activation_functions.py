'''
Activation Functions
'''

import torch
from dl_utils.plot._backend import pyplot as plt
from dl_utils.plot.figures import plot

# ReLU activation function
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
# plt.show()

# Gradient of ReLU
y.backward(torch.ones_like(x), retain_graph=True)
plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
# plt.show()

# Sigmoid activation function
y = torch.sigmoid(x)
plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
# plt.show()

# Gradient of Sigmoid
x.grad.data.zero_()
y.backward(torch.ones_like(x), retain_graph=True)
plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
# plt.show()

# Tanh activation function
y = torch.tanh(x)
plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
# plt.show()

# Gradient of Tanh
x.grad.data.zero_()
y.backward(torch.ones_like(x), retain_graph=True)
plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
# plt.show()
