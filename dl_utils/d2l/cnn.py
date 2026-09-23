import torch
from torch import nn
from torch.nn import functional as F

from dl_utils.plot.figures import Animator
from dl_utils.training.metrics import (
    Accumulator,
    accuracy,
    evaluate_accuracy_gpu,
)
from dl_utils.training.timing import Timer

def corr2d(X, K):
    """
    Calculate 2D cross-correlation.

    Args:
        X: the input tensor
        K: the kernel tensor
    Returns:
        the output tensor
    """
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


def train_ch6(net, train_iter, test_iter, num_epochs,
              lr, device, net_name=None):
    """
    Train the model using GPU.

    Args:
        net: the network
        train_iter: the training data iterator
        test_iter: the testing data iterator
        num_epochs: the number of epochs
        lr: the learning rate
        device: the device to use
        net_name: the name of the network (Default: None)
    """
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)

    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = Timer(), len(train_iter)
    report_interval = max(1, num_batches // 5)
    metric = train_l = train_acc = test_acc = None

    if net_name is not None:
        print(f'\n{net_name} is training on {device} ...')
    else:
        print(f'\nTraining on {device} ...')

    for epoch in range(num_epochs):
        # (training) loss_sum, total number of correct predictions, total number of samples
        metric = Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X_device, y_device = X.to(device), y.to(device)
            y_hat = net(X_device)
            l = loss(y_hat, y_device)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X_device.shape[0], accuracy(y_hat, y_device), X_device.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % report_interval == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    if metric is None or train_l is None or train_acc is None or test_acc is None:
        return
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec, '
          f'total time: {timer.format_time()}')


class Residual(nn.Module):
    """
    Residual block.

    Args:
        input_channels: the number of input channels
        num_channels: the number of output channels
        use_1x1conv: whether to use 1x1 convolution (Default: False)
        strides: the stride of the convolution (Default: 1)
    """
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        """
        Forward pass.

        Args:
            X: the input tensor (batch_size, input_channels, height, width)
        Returns:
            The output tensor (batch_size, num_channels, height, width)
        """
        output = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(X)))))
        if self.conv3:
            residual = self.conv3(X)
        else:
            residual = X
        output += residual
        return F.relu(output)
