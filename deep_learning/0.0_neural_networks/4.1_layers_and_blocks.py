'''
Layers and Blocks
'''

import torch
from torch import nn
from torch.nn import functional as F

X = torch.randn(2, 20)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256) # Hidden layer: 256 neurons
        self.out = nn.Linear(256, 10)    # Output layer: 10 neurons

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))

class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # Here, module is an instance of a subclass of Module, 
            # and we store it in the Module class's member variable _modules. 
            # The type of _modules is OrderedDict
            self._modules[str(idx)] = module
    
    def forward(self, X):
        # OrderedDict ensures that its members are iterated in the order they were added
        for block in self._modules.values():  
            X = block(X)
        return X

class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # A randomly initialized weight parameter that does not compute gradients, so it remains unchanged during training
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # Use the created constant parameters along with the `relu` and `mm` functions
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # Reuse the fully connected layer -- this is equivalent to two fully connected layers sharing parameters
        X = self.linear(X)
        # Control flow
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()

class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

if __name__ == '__main__':
    net = MLP()
    print(net(X))

    net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
    print(net(X))

    net = FixedHiddenMLP()
    print(net(X))

    chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
    print(chimera(X))
