'''
Asynchronous Computation
'''

import numpy
import torch
from dl_utils.d2l.benchmark import Benchmark
from dl_utils.runtime.devices import try_gpu

# GPU computation warm-up
device = try_gpu()
a = torch.randn(size=(1000, 1000), device=device)
b = torch.mm(a, a)

with Benchmark('numpy'):
    for _ in range(10):
        a = numpy.random.normal(size=(1000, 1000))
        b = numpy.dot(a, a)

with Benchmark('torch'):
    for _ in range(10):
        a = torch.randn(size=(1000, 1000), device=device)
        b = torch.mm(a, a)
    # GPU ops are dispatched asynchronously, so CPU returns without waiting for GPU to finish
    # thus the measured time only reflects dispatch overhead, far less than actual GPU compute time

with Benchmark():
    for _ in range(10):
        a = torch.randn(size=(1000, 1000), device=device)
        b = torch.mm(a, a)
    torch.cuda.synchronize(device) # wait for all GPU operations to complete

x = torch.ones((1, 2), device=device)
y = torch.ones((1, 2), device=device)
z = x * y + 2
print(z)
