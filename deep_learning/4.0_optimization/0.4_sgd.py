'''
Stochastic Gradient Descent
'''

import math
import torch
from dl_utils.d2l.optim import train_2d
from dl_utils.plot.figures import trace2d

def f(x1, x2):  # objective function
    return x1 ** 2 + 2 * x2 ** 2

def f_grad(x1, x2):  # gradient of the objective function
    return 2 * x1, 4 * x2

def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # simulate noisy gradient
    g1 += torch.normal(0.0, 1, (1,)).item()
    g2 += torch.normal(0.0, 1, (1,)).item()
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)

def constant_lr():
    return 1

eta = 0.1
lr = constant_lr  # constant learning rate
trace2d(f, train_2d(sgd, steps=50, f_grad=f_grad))

def exponential_lr():
    # global variable defined outside the function and updated inside
    global t
    t += 1
    return math.exp(-0.1 * t)

t = 1
lr = exponential_lr
trace2d(f, train_2d(sgd, steps=1000, f_grad=f_grad))

def polynomial_lr():
    # global variable defined outside the function and updated inside
    global t
    t += 1
    return (1 + 0.1 * t) ** (-0.5)

t = 1
lr = polynomial_lr
trace2d(f, train_2d(sgd, steps=50, f_grad=f_grad))
