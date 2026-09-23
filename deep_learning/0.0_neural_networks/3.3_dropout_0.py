'''
Dropout
'''

import torch
from torch import nn

from dl_utils.d2l.data_fashion import load_data_fashion_mnist, train_ch3
from dl_utils.runtime.timing import Timer


def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # In this case, all elements are dropped
    if dropout == 1:
        return torch.zeros_like(X)
    # In this case, all elements are retained
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, is_training=True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # Dropout is only applied in training mode
        if self.training == True:
            # Use dropout only when training the model
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            # Add a dropout layer after the second fully connected layer
            H2 = dropout_layer(H2, dropout2)
        return self.lin3(H2)

if __name__ == '__main__':
    num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
    dropout1, dropout2 = 0.2, 0.5
    num_epochs, lr, batch_size = 10, 0.5, 256
    net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)

    loss = nn.CrossEntropyLoss(reduction='none')
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    timer = Timer()
    train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
    print(f'{timer.stop():.3f} sec')
    # plt.ioff()
    # plt.show()
