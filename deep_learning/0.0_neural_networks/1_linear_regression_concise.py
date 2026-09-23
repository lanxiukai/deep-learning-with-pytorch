'''
Linear Regression (Concise Implementation)
'''

import torch
from torch import nn

from dl_utils.d2l.linear import synthetic_data
from dl_utils.data.loading import load_array

if __name__ == '__main__':
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)

    batch_size = 10
    data_iter = load_array((features, labels), batch_size)

    # Define the neural network model (2 * 1 Linear network)
    # weight(out_features, in_features), bias(out_features,)
    linear = nn.Linear(2, 1)
    net = nn.Sequential(linear)
    '''
    The net has only one submodule with a length of 1; net[0] is the only linear layer. 
    The input layer does not correspond to an indexable module.
    '''

    # Initialize the weights (normal distribution) and bias (0)
    linear.weight.data.normal_(0, 0.01)
    linear.bias.data.fill_(0)

    # Define the loss function (MSE: Mean Squared Error)
    loss = nn.MSELoss()
    # Define the optimizers (SGD: Stochastic Gradient Descent)
    trainer = torch.optim.SGD(net.parameters(), lr=0.03)
    # net.parameters() returns a generator that yields the parameters of the net

    num_epochs = 3
    # Iterate over the data for num_epochs times
    for epoch in range(num_epochs):
        for X, y in data_iter:  # iterate over the data
            l = loss(net(X), y)    # calculate the MSE loss (forward propagation)
            trainer.zero_grad()    # reset the gradient (PyTorch will accumulate the gradient by default)
            l.backward()           # calculate the gradient (backward propagation)
            trainer.step()         # update the parameters
        l = loss(net(features), labels) # calculate the MSE loss after each epoch
        print(f'epoch {epoch + 1}, loss {l:f}')

    w = linear.weight.data  # The final weight after training
    b = linear.bias.data    # The final bias after training

    print(f'The estimated error in w: {true_w - w.reshape(true_w.shape)}')
    print(f'The estimated error in b: {true_b - b}')
