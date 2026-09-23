'''
Model Selection, Underfitting and Overfitting (Polynomial Regression)
'''

import math

import numpy as np
import torch
from torch import nn

from dl_utils.d2l.data_fashion import train_epoch_ch3
from dl_utils.data.loading import load_array
from dl_utils.evaluation.supervised import evaluate_loss
from dl_utils.plot.figures import Animator


def train(train_features, test_features, train_labels, test_labels, num_epochs=400):
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1]  # the number of features
    # Set bias=False, because we have already considered the bias in the polynomial features
    linear = nn.Linear(input_shape, 1, bias=False)
    net = nn.Sequential(linear)
    batch_size = min(10, train_labels.shape[0])
    train_iter = load_array((train_features, train_labels.reshape(-1, 1)), batch_size)
    test_iter = load_array((test_features, test_labels.reshape(-1, 1)), batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = Animator(xlabel='epoch', ylabel='loss', yscale='log',
                                 xlim=[1, num_epochs], ylim=[1e-3, 1e2], legend=['train', 'test'])
    for epoch in range(num_epochs):
        train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))
    print(f'weight: {linear.weight.data.numpy()}')

if __name__ == '__main__':
    max_degree = 20  # maximum order of the polynomial
    n_train, n_test = 100, 100  # number of training and testing samples
    true_w = np.zeros(max_degree)
    true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

    features = np.random.normal(size=(n_train + n_test, 1))
    np.random.shuffle(features)
    poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))  # Sample-wise power operation
    for i in range(max_degree):
        poly_features[:, i] /= math.gamma(i + 1)  # gamma(n) = (n-1)!
    labels = np.matmul(poly_features, true_w)  # (n_train + n_test,)
    labels += np.random.normal(scale=0.1, size=labels.shape)

    # convert to tensor
    true_w, features, poly_features, labels = [
        torch.tensor(x, dtype=torch.float32)
        for x in (true_w, features, poly_features, labels)
    ]

    train(poly_features[:n_train, :4], poly_features[n_train:, :4],
          labels[:n_train], labels[n_train:])

    train(poly_features[:n_train, :2], poly_features[n_train:, :2],
          labels[:n_train], labels[n_train:], num_epochs=1500)

    train(poly_features[:n_train, :20], poly_features[n_train:, :20],
          labels[:n_train], labels[n_train:], num_epochs=1500)
