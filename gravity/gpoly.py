# Compute the gravity response of a polygon
# Originally version of gpoly adapted from Blakely (1995) by Yaoguo Li for GPGN-303 Gravity and Magnetics
# Expanded for multiple geologic sources and tunnel detection by Richard Krahenbuhl
# for GPGN-303 (Gravity, Magnetics, Electrical) GPGN-304 (Gravity & Magnetics) & GPGN-314 (Applied Geophysics)
# Updated in Python by Andy McAliley for GPGN-304 (Gravity & Magnetics)
# Continued updates by Gavin Wilson during GPGN-314 (Applied Geophysics)
# Added docsrting by Joe Capriotti

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
    numobs = len(obs)
    numnodes = len(nodes)
    grav = np.zeros(numobs)
    for iobs in range(numobs):
        shiftNodes = nodes - obs[iobs]
        sum = 0
        # loop over segments
        for i1 in range(numnodes):
            i2 = i1+1
            # last node must wrap around to first node
            i2 = np.mod(i2,numnodes)
            x1 = shiftNodes[i1,0]
            x2 = shiftNodes[i2,0]
            z1 = shiftNodes[i1,1]
            z2 = shiftNodes[i2,1]

            dx = x2 - x1
            dz = z2 - z1
            # avoid zero division
            if abs(dz) < 1E-8:
                # move on if points are identical
                if abs(dx) < 1E-8:
                    continue
                dz = dz - dx*(1E-7)
            alpha = dx/dz
            beta = x1-alpha*z1
            r1 = np.sqrt(x1**2+z1**2)
            r2 = np.sqrt(x2**2+z2**2)
            theta1 = np.arctan2(z1,x1)
            theta2 = np.arctan2(z2,x2)

            term1 = np.log(r2/r1)
            term2 = alpha*(theta2-theta1)
            factor = beta/(1+alpha**2)
            sum = sum + factor*(term1-term2)
        grav[iobs] = 2*gamma*density*sum
    return grav


def plot_model(data, obs_locs, *polygons):
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
