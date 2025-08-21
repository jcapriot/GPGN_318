import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

def draw_tri_figure(centers, radii, angles, xlim=(0, 1000), ylim=(0, 1000), ax=None):
    if ax is None:
        ax = plt.gca()
    
    for center, radius, angle in zip(centers, radii, angles):
        circ = patches.Circle(center, radius, fill=False, edgecolor="red", linewidth=2)
        circ_center = patches.Circle(center, 10, facecolor="#EEEEEE", edgecolor="black")
    
        ax.add_patch(circ)
        ax.add_patch(circ_center)
    
        orient = np.r_[np.cos(angle), -np.sin(angle)]
        start = (center + 10 * orient)
        end = (center + radius * orient)
        arrow = patches.Arrow(*start,*(end - start), width=50, color="grey", capstyle="round")
        ax.add_patch(arrow)
    
    plt.axis('off')
    ax.set_aspect(1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return ax

def draw_smiley(center, size, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.set_aspect(1)
    
    eye_offset = (3 * size / 8, size / 4)
    face = patches.Circle(center, size, fill=True, color='#FFCC00')
    eye_r = patches.Circle(center + eye_offset, size/8, fill=True, color="black")
    eye_offset *= np.r_[-1, 1]
    eye_l = patches.Circle(center + eye_offset, size/8, fill=True, color="black")

    smile = patches.Arc(center, size, size, angle=-90, theta1=-45, theta2=45, color="black")

    ax.add_patch(face)
    ax.add_patch(eye_r)
    ax.add_patch(eye_l)
    ax.add_patch(smile)
    return ax