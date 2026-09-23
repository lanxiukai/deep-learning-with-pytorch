"""
Stochastic Gradient Descent (SGD) 3D Visualization Animation

Objective: f(x1, x2) = 0.15*(x1^2 + x2^2) + 3*sin(1.3*x1)*cos(1.3*x2)
  -- quadratic bowl + bidirectional sine ripples, creating dozens of local minima
Random noise simulates the stochasticity of mini-batch gradients
"""

import numpy as np
from dl_utils.plot._backend import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from dl_utils.filesystem.project_root import infer_project_root

# ---- Objective (multi-modal landscape) ----------------------------------------
# Quadratic bowl keeps the function bounded; sine terms create complex local minima terrain
FREQ = 1.3
AMP  = 3.0
REG  = 0.15

def f(x, y):
    return REG * (x**2 + y**2) + AMP * np.sin(FREQ * x) * np.cos(FREQ * y)

def f_grad(x, y):
    gx = 2 * REG * x + AMP * FREQ * np.cos(FREQ * x) * np.cos(FREQ * y)
    gy = 2 * REG * y - AMP * FREQ * np.sin(FREQ * x) * np.sin(FREQ * y)
    return gx, gy

# ---- SGD (with random noise simulating mini-batch gradients) ------------------
def run_sgd(eta=0.50, noise_scale=0.7, steps=140, seed=7):
    np.random.seed(seed)
    x, y = -5.5, 4.8
    traj = [(x, y, f(x, y))]
    for _ in range(steps):
        gx, gy = f_grad(x, y)
        gx += np.random.randn() * noise_scale
        gy += np.random.randn() * noise_scale
        x -= eta * gx
        y -= eta * gy
        traj.append((x, y, f(x, y)))
    return traj

trajectory = run_sgd()
traj_x = np.array([p[0] for p in trajectory])
traj_y = np.array([p[1] for p in trajectory])
traj_z = np.array([p[2] for p in trajectory])
n_steps = len(trajectory)

# ---- Surface grid ------------------------------------------------------------
x_grid = np.linspace(-7, 7, 120)
y_grid = np.linspace(-7, 7, 120)
X, Y = np.meshgrid(x_grid, y_grid)
Z = f(X, Y)

# ---- Figure ------------------------------------------------------------------
BG = '#0d0d1f'
fig = plt.figure(figsize=(12, 8), facecolor=BG)
ax  = fig.add_subplot(111, projection='3d', facecolor=BG)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

# Surface (turbo colormap for strong contrast between peaks and valleys)
ax.plot_surface(X, Y, Z, alpha=0.55, cmap='turbo', edgecolor='none', zorder=1)

# Axis styling
ax.set_xlabel('$x_1$', color='#ccccee', labelpad=8, fontsize=12)
ax.set_ylabel('$x_2$', color='#ccccee', labelpad=8, fontsize=12)
ax.set_zlabel('$f(x_1, x_2)$', color='#ccccee', labelpad=8, fontsize=12)
ax.tick_params(colors='#888899', labelsize=8)
for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
    pane.fill = False
    pane.set_edgecolor('#22224a')
ax.grid(False)

ax.set_title(
    'SGD on Multi-Modal Landscape\n'
    r'$f(x_1,x_2)=0.15(x_1^2+x_2^2)+3\sin(1.3x_1)\cos(1.3x_2)$',
    color='white', pad=14, fontsize=13, fontweight='bold'
)
ax.view_init(elev=35, azim=-55)

# Color gradient: plasma colormap for each path segment (purple -> orange -> yellow)
seg_colors = plt.cm.plasma(np.linspace(0.05, 0.95, n_steps))

# Updatable objects
line,  = ax.plot([], [], [], '-',  linewidth=2.0, color='#ff6f3c', zorder=5)
head,  = ax.plot([], [], [], 'o',  markersize=11,
                 color='#ffee00', markeredgecolor='white',
                 markeredgewidth=1.5, zorder=6)
# Start point marker
ax.scatter([traj_x[0]], [traj_y[0]], [traj_z[0]],
           s=80, color='#00ffcc', marker='*', zorder=7, label='Start')

info = ax.text2D(0.02, 0.93, '', transform=ax.transAxes,
                 color='white', fontsize=11,
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a3a',
                           edgecolor='#5555aa', alpha=0.8))

# ---- Animation functions -----------------------------------------------------
def init():
    line.set_data([], [])
    line.set_3d_properties([])
    head.set_data([], [])
    head.set_3d_properties([])
    info.set_text('')
    return line, head, info

def update(frame):
    n = frame + 1
    # Draw trajectory segment by segment, color changes over time
    line.set_data(traj_x[:n], traj_y[:n])
    line.set_3d_properties(traj_z[:n])
    line.set_color(seg_colors[frame])

    head.set_data([traj_x[frame]], [traj_y[frame]])
    head.set_3d_properties([traj_z[frame]])

    info.set_text(
        f'Step: {frame:>3d} / {n_steps - 1}\n'
        f'f   : {traj_z[frame]:.5f}\n'
        f'x1  : {traj_x[frame]:+.3f}\n'
        f'x2  : {traj_y[frame]:+.3f}'
    )
    # Slowly rotate the view to enhance depth perception
    ax.view_init(elev=35, azim=-55 + frame * 0.45)
    return line, head, info

# ---- Render and save animation -----------------------------------------------
anim = FuncAnimation(
    fig, update,
    frames=n_steps,
    init_func=init,
    interval=100,       # frame interval in ms
    blit=False
)

output = infer_project_root() / 'output' / 'sgd_animation.gif'
output.parent.mkdir(parents=True, exist_ok=True)
writer = PillowWriter(fps=12)
print('Saving animation, please wait...')
anim.save(str(output), writer=writer, dpi=110)
print(f'Animation saved to: {output}')
plt.close()
