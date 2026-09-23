import numpy as np
import torch
import torch.nn as nn

from dl_utils.d2l.linear import linreg, squared_loss
from dl_utils.data.downloads import download
from dl_utils.data.vision import load_array
from dl_utils.plot.figures import Animator
from dl_utils.training.metrics import evaluate_loss
from dl_utils.training.timing import Timer

def train_2d(trainer, steps=20, f_grad=None):
    """Optimize a 2D objective function with a customized trainer"""
    # s1 and s2 are internal state variables to be used later (Momentum or Adam ...)
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]  # record the results of x1 and x2 for plotting
    for i in range(steps):  # iterate for steps times
        if f_grad:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad)
        else:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    # print the results of x1 and x2 for the last iteration
    print(f'epoch {steps}, x1: {float(x1):f}, x2: {float(x2):f}')
    # return the results of x1 and x2 for plotting
    return results


def get_data_ch11(batch_size=10, n=1500):
    # load data from airfoil_self_noise.dat, columns are divided by '\t'
    data = np.genfromtxt(download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    # normalize data by subtracting the mean and dividing by the standard deviation of each column
    data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0))
    # create data iterator using the first n rows, the last column of each row is label,
    # the rest are features, shuffle is True
    data_iter = load_array((data[:n, :-1], data[:n, -1]),
                           batch_size, is_train=True)
    # return data iterator and the number of features
    return data_iter, data.shape[1]-1


def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=4, prefix=''):
    '''
    A general training function for linear regression.

    Args:
        trainer_fn: the trainer function (optimizer)
        states: the states (e.g. momentum states, Adam states)
        hyperparams: the hyperparams
        data_iter: the data iterator (X, y)
        feature_dim: the feature dimension
        num_epochs: the number of epochs (default: 4)

    Returns:
        the cumulative time and the loss (list)
    '''
    # initialize model
    w = torch.normal(mean=0.0, std=0.01, size=(feature_dim, 1),
                     requires_grad=True)
    b = torch.zeros((1), requires_grad=True)
    net, loss = lambda X: linreg(X, w, b), squared_loss
    # train model
    animator = Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, Timer()
    for _ in range(num_epochs):
        for batch_idx, (X, y) in enumerate(data_iter):
            l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0 or batch_idx == len(data_iter) - 1:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (evaluate_loss(net, data_iter, loss),))
                timer.start()
    assert animator.Y is not None, 'no checkpoints fired'
    print(f'{prefix} loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/log')
    return timer.cumsum(), animator.Y[0]


def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=4, prefix=''):
    '''
    A general training function for linear regression (concise implementation).

    Args:
        trainer_fn: the trainer function (optimizer)
        hyperparams: the hyperparams
        data_iter: the data iterator (X, y)
        num_epochs: the number of epochs (default: 4)
        prefix: the prefix of the loss (default: '')

    Returns:
        None
    '''
    # initialize model
    net = nn.Sequential(nn.Linear(5, 1))

    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.normal_(m.weight, std=0.01)

    net.apply(init_weights)

    # **hyperparams: unpack the hyperparams dictionary into keyword arguments
    optimizer = trainer_fn(net.parameters(), **hyperparams)
    loss = nn.MSELoss(reduction='none')
    animator = Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, Timer()
    for _ in range(num_epochs):
        for batch_idx, (X, y) in enumerate(data_iter):
            optimizer.zero_grad()
            out = net(X)              # out shape: (batch_size, 1)
            y = y.reshape(out.shape)  # y shape: (batch_size,) -> (batch_size, 1)
            l = loss(out, y)
            l.mean().backward()
            optimizer.step()
            n += X.shape[0]
            if n % 200 == 0 or batch_idx == len(data_iter) - 1:
                timer.stop()
                # MSELoss computes squared error without the 1/2 coefficient
                animator.add(n/X.shape[0]/len(data_iter),
                             (evaluate_loss(net, data_iter, loss) / 2,))
                timer.start()
    assert animator.Y is not None, 'no checkpoints fired'
    print(f'{prefix} loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/log')
