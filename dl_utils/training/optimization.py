"""Generic optimization primitives."""

import torch
from torch import nn


def sgd(params, lr, batch_size):
    """
    Batch stochastic gradient descent.
    
    Args:
        params: the parameters
        lr: the learning rate
    """
    with torch.no_grad():  # no need to track gradient, just update the parameters
        for param in params:  # traverse the parameters
            param -= lr * param.grad / batch_size  # update the parameter
            param.grad.zero_()                     # reset the gradient


def grad_clipping(net, theta):
    """
    Clip gradients (global norm clipping).
    
    Args:
        net: the network
        theta: the threshold of the gradient
    """
    if isinstance(net, nn.Module):
        params = [parameter for parameter in net.parameters() if parameter.requires_grad]
    else:
        params = net.params
    grads = [parameter.grad for parameter in params if parameter.grad is not None]
    norm = torch.sqrt(sum(
        (torch.sum(grad ** 2) for grad in grads),
        torch.zeros_like(grads[0].sum()),
    ))
    if norm > theta:
        for grad in grads:
            grad[:] *= theta / norm
