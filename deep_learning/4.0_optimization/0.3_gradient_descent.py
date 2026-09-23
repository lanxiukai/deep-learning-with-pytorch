'''
Gradient Descent
'''

import numpy as np
import torch
from dl_utils.plot._backend import pyplot as plt
from dl_utils.d2l.optim import train_2d
from dl_utils.plot.figures import plot, set_figsize, trace2d

def f(x):  # objective function
    return x ** 2

def f_grad(x):  # gradient (derivative) of the objective function
    return 2 * x

def gd(eta, f_grad):
    x = 10.0  # initial value
    results = [x]  # record the results of x for plotting
    for i in range(10):  # iterate for 10 times
        x -= eta * f_grad(x)      # update x using the gradient
        results.append(float(x))  # record the results of x for plotting
    print(f'epoch 10, x: {x:f}')
    return results

results = gd(0.2, f_grad)

def show_trace(results, f):
    n = max(abs(min(results)), abs(max(results)))
    f_line = torch.arange(-n, n, 0.01)
    set_figsize()
    plot([f_line, results], [[f(x) for x in f_line], [
        f(x) for x in results]], 'x', 'f(x)', fmts=['-', '-o'])

show_trace(results, f)

show_trace(gd(0.05, f_grad), f)

show_trace(gd(1.1, f_grad), f)

c = torch.tensor(0.15 * np.pi)

def f(x):  # objective function
    return x * torch.cos(c * x)

def f_grad(x):  # gradient of the objective function
    return torch.cos(c * x) - c * x * torch.sin(c * x)

show_trace(gd(2, f_grad), f)

def f_2d(x1, x2):  # objective function
    return x1 ** 2 + 2 * x2 ** 2

def f_2d_grad(x1, x2):  # gradient of the objective function
    return (2 * x1, 4 * x2)

def gd_2d(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    return (x1 - eta * g1, x2 - eta * g2, 0, 0)

eta = 0.1
trace2d(f_2d, train_2d(gd_2d, f_grad=f_2d_grad))

c = torch.tensor(0.5)

def f(x):  # objective function
    return torch.cosh(c * x)

def f_grad(x):  # gradient of the objective function
    return c * torch.sinh(c * x)

def f_hess(x):  # Hessian of the objective function
    return c**2 * torch.cosh(c * x)

def newton(eta=1):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x) / f_hess(x)
        results.append(float(x))
    print('epoch 10, x:', x)
    return results

show_trace(newton(), f)

c = torch.tensor(0.15 * np.pi)

def f(x):  # objective function
    return x * torch.cos(c * x)

def f_grad(x):  # gradient of the objective function
    return torch.cos(c * x) - c * x * torch.sin(c * x)

def f_hess(x):  # Hessian of the objective function
    return - 2 * c * torch.sin(c * x) - x * c**2 * torch.cos(c * x)

show_trace(newton(), f)

show_trace(newton(0.5), f)

# plt.ioff()
# plt.show()
