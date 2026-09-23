"""
Attention Pooling (nonparametric)
"""
import torch
from dl_utils.plot._backend import pyplot as plt
from torch import nn
from dl_utils.plot.figures import heatmap, plot

n_train = 50  # Number of training samples
x_train, _ = torch.sort(torch.rand(n_train) * 5)   # Sorted training samples from minimum to maximum

def f(x):
    return 2 * torch.sin(x) + x**0.8

y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))  # Outputs of training samples
x_test = torch.arange(0, 5, 0.1)  # Test samples
y_truth = f(x_test)  # Ground-truth outputs of test samples
n_test = len(x_test)  # Number of test samples
print(n_test)

def plot_kernel_reg(y_hat):
    plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
             xlim=[0, 5], ylim=[-1, 5])
    plt.plot(x_train, y_train, 'o', alpha=0.5);

y_hat = torch.repeat_interleave(y_train.mean(), n_test)  # Repeat the mean of y_train for n_test times
plot_kernel_reg(y_hat)

# Shape of X_repeat: (n_test, n_train)
# Each row contains the same test input (e.g., the same query)
X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
# x_train contains the keys. Shape of attention_weights: (n_test, n_train)
# Each row contains attention weights assigned over values (y_train) for a query
attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / 2, dim=1)
# softmax: normalize the attention weights to sum to 1 by dividing by the sum of the weights
# Each element of y_hat is a weighted average of values, using attention weights
y_hat = torch.matmul(attention_weights, y_train)
plot_kernel_reg(y_hat)

heatmap(attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')

# plt.ioff()
# plt.show()
