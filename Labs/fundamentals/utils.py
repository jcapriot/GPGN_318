import matplotlib.pyplot as plt
import numpy as np

def make_lab_grid(ax=None):
    if ax is None:
        ax = plt.gca()
    xy = np.mgrid[0:20:5j,0:20:5j]
    ps = ax.scatter(*xy, marker='x')
    ax.set_aspect(1)
    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    return ps