# Compute the gravity response of a polygon
# Originally version of gpoly adapted from Blakely (1995) by Yaoguo Li for GPGN-303 Gravity and Magnetics
# Expanded for multiple geologic sources and tunnel detection by Richard Krahenbuhl
# for GPGN-303 (Gravity, Magnetics, Electrical) GPGN-304 (Gravity & Magnetics) & GPGN-314 (Applied Geophysics)
# Updated in Python by Andy McAliley for GPGN-304 (Gravity & Magnetics)
# Continued updates by Gavin Wilson during GPGN-314 (Applied Geophysics)
# Added docstring by Joe Capriotti
# fix for true limits (no need to adjust small numbers) by Joe Capriotti
# vectorized call over observations by Joe Capriotti

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.constants import mu_0


def _is_cw(points):
    points = np.asarray(points)
    if points.shape[1] != 2:
        raise ValueError("points must be an Nx2 array")

    x = points[:, 0]
    y = points[:, 1]
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)

    area2 = np.sum((x_next - x) * (y_next + y))

    return area2 > 0

def mpoly(obs, nodes, magnetization, proj_dir=None):
    """Calculate the magnetic flux density of an infinite 2D polygonal prism.

    This function computes the magnetic flux density at observation points
    due to a polygonal body defined by a set of nodes using the method
    described in Blakely, 1996. It optionally will also project the anomaly
    along a specific direction.

    Parameters
    ----------
    obs : (n_loc, 2) numpy.ndarray
        Observation locations in meters (x, z) (z positive down).
    nodes : (n_nodes, 2) numpy.ndarray
        Polygon node locations in meters (x, z), must be listed in clockwise order.
    magnetization : (2, ) numpy.ndarray
        Magnetization of the the prism (nA/m).
    proj_dir : (2, ) numpy.ndarray, optional
        Direction to project the anomaly onto.

    Returns
    -------
    B : (n_loc, 3) numpy.ndarray,
        The magnetic flux density, and total field anomaly in units of nT
        (ordered as [Bx, Bz, TFA]).

    Notes:
    ------
    The function uses the formula described in Blakely, 1996, to compute the magnetic flux density
    attraction.

    References:
    -----------
    1. Blakely, R. J. (1996). Potential theory in gravity and magnetic applications.
       Cambridge University Press.
    """
    #Blakely, 1996
    obs = np.asarray(obs)
    nodes = np.asarray(nodes)
    
    # append a value to obs to do a cursory test for CW vs CCW
    # basically add a point outside and at a lower z-value than the body.
    # This point is above the center of the object.
    
    numobs = len(obs)
    numnodes = len(nodes)
    b = np.zeros((numobs, 2))
    
    for i1 in range(numnodes):
        i2 = (i1 + 1) % numnodes

        gzx_gzz = fz_segment_dxz(obs, nodes[i1], nodes[i2])
        gxx_gxz = -fz_segment_dxz(obs[:, ::-1], nodes[i1][::-1], nodes[i2][::-1])[:, ::-1]

        bx = gxx_gxz.dot(magnetization)
        bz = gzx_gzz.dot(magnetization)
        b += np.c_[bx, bz]

    if _is_cw(nodes): b *= -1
    b *= mu_0 / (4 * np.pi)
    if proj_dir is None:
        proj_dir = magnetization
    proj_dir = proj_dir / np.linalg.norm(proj_dir)
    return np.c_[b, b.dot(proj_dir)]
        

def fz_segment_dxz(obs, node_a, node_b):
    dr1 = node_a - obs
    dr2 = node_b - obs
    segment = node_a - node_b
    a = dr1[:, 1]
    b = dr1[:, 0]
    c = dr2[:, 1]
    d = dr2[:, 0]

    dtheta = _dtheta(a, b, c, d)

    flat_line = segment[1] == 0
    if flat_line:
        g_dtheta  = dr1[:, 1]
        g_dx1 = np.zeros_like(dtheta)
        g_dz1 = dtheta
        g_a = np.zeros_like(a)
        g_b = np.zeros_like(b)
        g_c = np.zeros_like(c)
        g_d = np.zeros_like(d)
    else:
        r1sq = a * a + b * b
        r2sq = c * c + d * d

        off_nodes = (r1sq != 0) & (r2sq != 0)
        r1sq = r1sq[off_nodes]
        r2sq = r2sq[off_nodes]
        
        alpha = segment[0]/segment[1]
        beta = dr1[:, 0] - alpha * dr1[:, 1]
        
        term1 = np.zeros_like(beta)
        term1[off_nodes] = 0.5 * np.log(r2sq / r1sq)
        term2 = alpha * dtheta
        factor = beta / (1 + alpha ** 2)

        factor[~off_nodes] = 0.0

        #contribution = factor * (term1 - term2)
        g_factor = (term1 - term2)
        g_term1 = factor
        g_term2 = -factor
        #factor = beta / (1 + alpha ** 2)
        g_beta = 1/(1 + alpha ** 2) * g_factor
        #term2 = alpha * dtheta
        g_dtheta = alpha * g_term2
        #term1[off_nodes] = 0.5 * np.log(r2sq / r1sq)
        g_r2sq = np.zeros_like(beta)
        g_r1sq = np.zeros_like(beta)

        g_r2sq[off_nodes] = 1 / (2 * r2sq) * g_term1[off_nodes]
        g_r1sq[off_nodes] = -1 / (2 * r1sq) * g_term1[off_nodes]
        #beta = dr1[:, 0] - alpha * dr1[:, 1]
        g_dx1 = g_beta
        g_dz1 = - alpha * g_beta
        # r1sq = a * a + b * b
        g_a = 2 * a * g_r1sq
        g_b = 2 * b * g_r1sq
        # r2sq = c * c + d * d
        g_c = 2 * c * g_r2sq
        g_d = 2 * d * g_r2sq

    #dtheta = _dtheta(a, b, c, d)
    inds = b != 0
    g_a[inds] += _dtheta_da(a[inds], b[inds], c[inds], d[inds]) * g_dtheta[inds]
    inds = a != 0
    g_b[inds] += _dtheta_db(a[inds], b[inds], c[inds], d[inds]) * g_dtheta[inds]
    inds = d != 0
    g_c[inds] += _dtheta_dc(a[inds], b[inds], c[inds], d[inds]) * g_dtheta[inds]
    inds = c != 0
    g_d[inds] += _dtheta_dd(a[inds], b[inds], c[inds], d[inds]) * g_dtheta[inds]

    g_dx2 = g_d
    g_dz2 = g_c
    g_dx1 += g_b
    g_dz1 += g_a

    g_obs_x = - (g_dx2 + g_dx1)
    g_obs_z = - (g_dz2 + g_dz1)

    return np.c_[g_obs_x, g_obs_z]
    

def _dtheta(a, b, c, d):
    return np.arctan2(b * c - a * d, b * d + a * c)

def _dtheta_da(a, b, c, d):
    return -b/(a * a + b * b)

def _dtheta_db(a, b, c, d):
    return a/(a * a + b * b)

def _dtheta_dc(a, b, c, d):
    return d/(c * c + d * d)

def _dtheta_dd(a, b, c, d):
    return -c/(c * c + d * d)

def plot_model(data, obs_locs, *polygons, show_locations=True, loc_size=0.2):
    """
    Plot geophysical data and geological models with dikes.

    Parameters
    ----------
    data : array_like
        Array of geophysical data values.
    obs_locs : array_like
        Array of observation locations.
    *polygons : variable number of arrays
        Arrays representing the dikes as polygons.
    show_locations : bool
        Whether to show the observation locations on the model plot
    loc_size : float
        Size of the observation points (if show_locations is True).

    Returns
    -------
    ax_data, ax_model : matplotlib.axes.Axes
        axes for the data and model plots.

    Notes
    -----
    This function creates a two-panel plot with the top panel showing
    geophysical data and the bottom panel displaying geological models
    with dikes represented as polygons.

    The top panel includes the following elements:
    - x-axis labeled as 'Position (m)'
    - y-axis labeled as 'g_D (mGal)'
    - Plot of geophysical data at observation locations.

    The bottom panel includes the following elements:
    - x-axis labeled as 'Position (m)'
    - y-axis labeled as 'Depth (m)'
    - Polygons represented as patches.
    - Y-axis inverted to display depth increasing downwards.

    Examples
    --------
    >>> data = [10, 20, 30, 40, 50]
    >>> obs_locs = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    >>> dike1 = np.array([[1, 2], [2, 3], [3, 4]])
    >>> dike2 = np.array([[4, 5], [5, 6], [6, 7]])
    >>> plot_model(data, obs_locs, dike1, dike2)
    """

    fig = plt.figure()
    ax_data = fig.add_subplot(2,1,1)
    plt.xlabel('Position (m)')
    ax_data.set_ylabel(r'$\vec{B}$ (nT)')

    ax_model = fig.add_subplot(2,1,2,sharex=ax_data)
    ax_model.invert_yaxis()
    ax_model.set_ylabel('Depth (m)')
    ax_model.set_xlabel('Position (m)')
    
    if show_locations:
        ax_model.scatter(obs_locs[:, 0], obs_locs[:, 1], s=loc_size)

    ndim = data.ndim
    if ndim > 1:
        labels = ['Bx', 'Bz', 'TFA']
        for i in range(data.shape[1]):
            ax_data.plot(obs_locs[:, 0], data[:, i], label=labels[i])
        ax_data.legend()
    else:
        ax_data.plot(obs_locs[:, 0], data)

    for poly in polygons:
        polygon = patches.Polygon(poly)
        ax_model.add_patch(polygon)
    ax_model.use_sticky_edges = False
    ax_model.set_ymargin(0.1)
    ax_model.autoscale()
    return ax_data, ax_model
