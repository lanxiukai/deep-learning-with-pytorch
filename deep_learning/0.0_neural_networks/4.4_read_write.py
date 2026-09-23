'''
Read and Write Tensor
'''

import torch
from torch import nn
from torch.nn import functional as F

from dl_utils.filesystem.project_root import infer_project_root


_output_dir = infer_project_root() / 'output' / 'read_write'
_output_dir.mkdir(parents=True, exist_ok=True)

print('0--------------------------------')

x = torch.arange(4)
torch.save(x, _output_dir / 'x-file')

x2 = torch.load(_output_dir / 'x-file')
print(x2)

y = torch.zeros(4)
torch.save([x, y], _output_dir / 'x-file')

x2, y2 = torch.load(_output_dir / 'x-file')
print(x2, y2)

mydict = {'x': x, 'y': y}
torch.save(mydict, _output_dir / 'mydict')
mydict2 = torch.load(_output_dir / 'mydict')
print(mydict2)
print('1--------------------------------')

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, X):
        return self.output(F.relu(self.hidden(X)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)

torch.save(net.state_dict(), _output_dir / 'mlp.params')

clone = MLP()
clone.load_state_dict(torch.load(_output_dir / 'mlp.params'))
print(clone.eval())

Y_clone = clone(X)
print(Y_clone == Y)
