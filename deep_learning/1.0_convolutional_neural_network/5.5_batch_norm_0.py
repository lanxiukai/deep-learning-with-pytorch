'''
Batch Normalization
'''

import torch
from torch import nn
from dl_utils.d2l.cnn import train_ch6
from dl_utils.d2l.data_fashion import load_data_fashion_mnist
from dl_utils.runtime.devices import try_gpu

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use is_grad_enabled to determine whether we are in training or inference mode
    if not torch.is_grad_enabled():  # Inference mode
        # In inference mode, use the moving mean and variance provided
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:                            # Training mode
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2: # Fully connected layers
            # For fully connected layers, compute mean and variance over feature dimension
            mean = X.mean(dim=0)                 # (F,)
            var = ((X - mean) ** 2).mean(dim=0)  # (F,)
        else:                 # Convolutional layers
            # For 2D convolutional layers, compute mean and variance over channel dimension (axis=1).
            # Keep X's shape for later broadcasting
            mean = X.mean(dim=(0, 2, 3), keepdim=True)                 # (1, C, 1, 1)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)  # (1, C, 1, 1)
        # In training, normalize using the current batch mean and variance
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Update moving mean and variance of the entire dataset
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean.data, moving_var.data  # .data: detach the tensor from the computational graph

class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        '''
        Initialize the BatchNorm layer.
        Args:
            num_features: number of outputs for a fully connected layer or number of output channels for a convolutional layer.
            num_dims: 2 for fully connected layers, 4 for convolutional layers
        '''
        super().__init__()
        if num_dims == 2:  # Fully connected layers
            shape = (1, num_features)
        else:              # Convolutional layers
            shape = (1, num_features, 1, 1)
        # Learnable scale and shift parameters (gamma and beta), initialized to 1 and 0
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # Non-parameter buffers for moving mean and variance, initialized to 0 and 1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # Move moving_mean and moving_var to the same device as X
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # Save updated moving_mean and moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y

if __name__ == '__main__':
    net = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
                        nn.AvgPool2d(kernel_size=2, stride=2),
                        nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
                        nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
                        nn.Linear(16 * 4 * 4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
                        nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
                        nn.Linear(84, 10))

    lr, num_epochs, batch_size = 1.0, 10, 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    train_ch6(net, train_iter, test_iter, num_epochs, lr,
                       device=try_gpu(), net_name='LeNet-5 with BatchNorm')
    print(net[1].gamma.reshape((-1,)), net[1].beta.reshape((-1,)))
    
    # plt.ioff()
    # plt.show()
