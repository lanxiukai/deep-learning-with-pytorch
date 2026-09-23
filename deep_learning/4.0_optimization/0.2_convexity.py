'''
Convexity
'''

import numpy as np
import torch
from dl_utils.plot._backend import pyplot as plt
from mpl_toolkits import mplot3d
from dl_utils.plot.figures import plot, set_figsize, use_svg_display

f = lambda x: 0.5 * x**2  # Convex function
g = lambda x: torch.cos(np.pi * x)  # Nonconvex function
h = lambda x: torch.exp(0.5 * x)  # Convex function

x, segment = torch.arange(-2, 2, 0.01), torch.tensor([-1.5, 1])
use_svg_display()
_, axes = plt.subplots(1, 3, figsize=(9, 3))
for ax, func in zip(axes, [f, g, h]):
    plot([x, segment], [func(x), func(segment)], axes=ax)

f = lambda x: (x - 1) ** 2
set_figsize()
plot([x, segment], [f(x), f(segment)], 'x', 'f(x)')

# plt.show()
