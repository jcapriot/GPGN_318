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

def gpoly(obs,nodes,density):
    """Calculate the gravitational attraction of an infinite 2D polygonal prism.

    This function computes the gravitational attraction at observation points
    due to a polygonal body defined by a set of nodes using the method
    described in Blakely, 1996.

    Parameters
    ----------
    obs : (n_loc, 2) np.ndarray
        Observation locations in meters (x, z) (z positive down).
    nodes : (n_nodes, 2) np.ndarray
        Polygon node locations in meters (x, z), must be listed in clockwise order.
    density : float
        Density of the polygon in in g/cc

    Returns
    -------
    g_d : (n_loc, ) np.ndarray of float
        The vertical component of gravity, positive down, due to the polygon
        in units of mGal.

    Notes:
    ------
    The function uses the formula described in Blakely, 1996, to compute the gravitational
    attraction. It assumes a constant gravitational constant (gamma) of 6.672E-03 mGal.

    References:
    -----------
    1. Blakely, R. J. (1996). Potential theory in gravity and magnetic applications.
       Cambridge University Press.

    Example:
    --------
    >>> obs = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    >>> density = 2.5  # grams per cubic centimeter (g/cm³)
    >>> grav = gpoly(obs, nodes, density)
    """
    #Blakely, 1996
    gamma = 6.672E-03 # mGal
    obs = np.asarray(obs)
    nodes = np.asarray(nodes)
    
    # append a value to obs to do a cursory test for CW vs CCW
    # basically add a point outside and at a lower z-value than the body.
    # it's sign should match the sign of density.
    # This point is above the center of the object.
    
    obs = np.r_[obs, [[np.median(nodes[:, 0]), np.min(nodes[:, 1]) - 10]]]
    
    numobs = len(obs)
    numnodes = len(nodes)
    grav = np.zeros(numobs)
    
    for i1 in range(numnodes):
        i2 = (i1 + 1) % numnodes
        dr1 = nodes[i1] - obs
        dr2 = nodes[i2] - obs
        segment = nodes[i2] - nodes[i1]
        a = dr1[:, 1]
        b = dr1[:, 0]
        c = dr2[:, 1]
        d = dr2[:, 0]
    
        # arctan2(dr2[:, 1], dr2[:, 0]) - arctan2(dr1[:, 1], dr1[:, 0])
        dtheta = np.arctan2(b * c - a * d, b * d + a * c)

        flat_line = segment[1] == 0
        if flat_line:
            grav += dr1[:, 1] * dtheta
        else:
            r1 = np.linalg.norm(dr1, axis=-1)
            r2 = np.linalg.norm(dr2, axis=-1)

            off_nodes = (r1 != 0) & (r2 != 0)
            r1 = r1[off_nodes]
            r2 = r2[off_nodes]
            
            alpha = segment[0]/segment[1]
            beta = (dr1[:, 0] - alpha * dr1[:, 1])
            
            term1 = np.zeros_like(beta)
            term1[off_nodes] = np.log(r2 / r1)
            term2 = alpha * dtheta
            factor = beta / (1 + alpha ** 2)

            factor[~off_nodes] = 0.0
            
            grav += factor * (term1 - term2)
    grav *= 2*gamma*density
    
    if np.sign(grav[-1]) != np.sign(density):
        grav *= -1
    return grav[:-1]


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
    ax_data.set_ylabel(r'$g_D$ (mGal)')

    ax_model = fig.add_subplot(2,1,2,sharex=ax_data)
    ax_model.invert_yaxis()
    ax_model.set_ylabel('Depth (m)')
    ax_model.set_xlabel('Position (m)')
    
    if show_locations:
        ax_model.scatter(obs_locs[:, 0], obs_locs[:, 1], s=loc_size)

    ax_data.plot(obs_locs[:, 0], data)

    for poly in polygons:
        polygon = patches.Polygon(poly)
        ax_model.add_patch(polygon)
    ax_model.use_sticky_edges = False
    ax_model.set_ymargin(0.1)
    ax_model.autoscale()
    return ax_data, ax_model


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    xmin = 0.
    xmax = 10.
    nx = 201

    obs = np.zeros((nx,2))
    obs[:,0] = np.linspace(xmin,xmax,nx)

    nodes = np.zeros((4,2))
    nodes[0] = [2.,1.5]
    nodes[1] = [3.,1.5]
    nodes[2] = [3.,2.5]
    nodes[3] = [2.,2.5]

    density = 1

    grav = gpoly(obs,nodes,density)

    plt.plot(obs[:,0],grav)
    plt.show()
