'''
Softmax Regression
'''

import torch

from dl_utils.d2l.data_fashion import get_fashion_mnist_labels, load_data_fashion_mnist
from dl_utils.plot._backend import pyplot as plt
from dl_utils.plot.figures import set_axes
from dl_utils.plot.images import show_images
from dl_utils.runtime.timing import Timer
from dl_utils.training.optimization import sgd


def softmax(X):
    '''Compute the softmax of the input.

    Args:
        X: Wx + b (B, num_outputs)
    Returns:
        the softmax of the input (B, num_outputs)
    '''
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)  # partition: (B, 1)
    # keepdim=True means the dimension of the output is the same as the input
    return X_exp / partition

def net(X):
    '''Compute the predicted value of the network.
    
    Args:
        X: the input features of the network 
            (X.reshape(-1, W.shape[0])) Any -> (B, num_inputs)
        W: the weight of the network (num_inputs, num_outputs)
        b: the bias of the network (num_outputs,)
    Returns:
        the predicted value of the network (B, num_outputs)
    '''
    return softmax(torch.matmul(X.reshape(-1, W.shape[0]), W) + b)

def cross_entropy(y_hat, y):
    '''Cross Entropy Loss
    
    Args:
        y_hat: the predicted value (B, num_outputs)
        y: the true value (B,)
    Returns:
        the cross entropy loss (B,)
    '''
    return - torch.log(y_hat[range(len(y_hat)), y])  # -log(y_hat[i, y[i]])

def accuracy(y_hat, y):
    '''Compute the number of correct predictions.
    
    Args:
        y_hat: the predicted value (B, num_outputs) or (B,)
        y: the true value (B,)
    Returns:
        the number of correct predictions
    '''
    # len(y_hat.shape) > 1 and y_hat.shape[1] > 1: means y_hat is a matrix
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat_idx = y_hat.argmax(axis=1)  # y_hat_idx: (B,), obtain the index of the maximum probability
    cmp = y_hat_idx.type(y.dtype) == y  # cmp: (B,), True or False
    return float(cmp.type(y.dtype).sum())  # sum the True values, convert to float

def evaluate_accuracy(net, data_iter):
    '''Compute the accuracy for a model on a dataset.
    
    Args:
        net: the network
        data_iter: the data iterator
    Returns:
        the accuracy of the model
    '''
    if isinstance(net, torch.nn.Module): # Determine whether net is an instance of torch.nn.Module
        net.eval()  # set the model to evaluation mode
    metric = Accumulator(2)  # correct predictions, total predictions
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())  # metric.add(correct predictions, total predictions)
    return metric[0] / metric[1]  # return the accuracy

class Accumulator:
    '''Accumulate sum over n variables.
    
    Args:
        n: the number of variables, initialize the data with n zeros
    '''
    def __init__(self, n):
        self.data = [0.0] * n  # initialize the data with n zeros

    def add(self, *args):
        '''Add the arguments to the data.
        
        Args:
            args: the arguments to add (can be a tuple)
        '''
        self.data = [a + float(b) for a, b in zip(self.data, args)]  # add the arguments to the data

    def reset(self):
        '''Reset the data.'''
        self.data = [0.0] * len(self.data)  # reset the data to n zeros

    def __getitem__(self, idx):  # double underscores getitem: get the data at the index
        '''Get the data at the index.'''
        return self.data[idx]  # return the data at the index

def train_epoch_ch3(net, train_iter, loss, updater): 
    '''Train the model for one epoch.
    
    Args:
        net: the network
        train_iter: the training data iterator
        loss: the loss function
        updater: the optimizer
    Returns:
        the average loss and accuracy
    '''
    if isinstance(net, torch.nn.Module): # Determine whether net is an instance of torch.nn.Module
        net.train()  # set the model to training mode
    metric = Accumulator(3)  # total loss, total accuracy, total number of samples
    for X, y in train_iter:
        # compute the gradient and update the parameters
        y_hat = net(X)      # y_hat: the predicted value (B, num_outputs) or (B,)
        l = loss(y_hat, y)  # l: the loss (B,); y: the true value (B,)
        if isinstance(updater, torch.optim.Optimizer):  # Determine whether updater is an instance of torch.optim.Optimizer
            # use the built-in optimizer and loss function in PyTorch
            updater.zero_grad()  # reset the gradient
            l.mean().backward()  # compute the gradient
            updater.step()       # update the parameters
        else:
            # use the custom optimizer and loss function
            l.sum().backward()  # compute the gradient
            updater(X.shape[0])  # update the parameters
        metric.add(l.detach().sum().item(), accuracy(y_hat, y), y.numel())  # total loss, total accuracy, total number of samples
    return metric[0] / metric[2], metric[1] / metric[2]

class Animator:
    """Animate data curves using Matplotlib interactive mode."""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
        plt.ion()
        self.fig.show()
        self._closed = False

    def add(self, x, y):
        '''Add the data to the animator.
        
        Args:
            x: the x-axis data
            y: the y-axis data
        '''
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x_vals, y_vals, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x_vals, y_vals, fmt)
        self.config_axes()
        self.fig.canvas.draw_idle()
        if hasattr(self.fig.canvas, "flush_events"):
            self.fig.canvas.flush_events()
        plt.pause(0.001)

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """Train a model with multiple epochs.
    
    Args:
        net: the network
        train_iter: the training data iterator
        test_iter: the test data iterator
        loss: the loss function
        num_epochs: the number of epochs
        updater: the optimizer
        is_hold: whether to hold the animator (Default: False)
    """
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)  # train_metrics: (train_loss, train_acc)
        test_acc = evaluate_accuracy(net, test_iter)                     # test_acc: the accuracy of the test data
        animator.add(epoch + 1, train_metrics + (test_acc,))             # add the train_metrics and test_acc to the animator
    train_loss, train_acc = train_metrics
    # Check whether the data is out of bounds
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

def updater(batch_size):
    return sgd([W, b], lr, batch_size)

def predict_ch3(net, test_iter, n=6):
    """Predict labels.
    
    Args:
        net: the network
        test_iter: the test data iterator
        n: the number of images to predict (Default: 6)
    """
    for X, y in test_iter:
        break
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

if __name__ == '__main__':
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    num_inputs = 784
    num_outputs = 10

    # Initialize the weights and bias
    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)

    lr = 0.1
    num_epochs = 10

    timer = Timer()
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
    predict_ch3(net, test_iter)
    print(f'{timer.stop():.3f} sec')
    # plt.ioff()
    # plt.show()
