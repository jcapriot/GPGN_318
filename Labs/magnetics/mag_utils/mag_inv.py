import numpy as np
import matplotlib.pyplot as plt
import discretize
from simpeg.potential_fields import magnetics as mag
from simpeg import (
    maps,
    data,
    data_misfit,
    inverse_problem,
    regularization,
    optimization,
    directives,
    inversion,
)
import pandas as pd
import re


def setup_data(file_name):
    with open(file_name) as f:
        dec = float(f.readline()[:-1].split(":")[-1].split()[0])
        inc = float(f.readline()[:-1].split(":")[-1].split()[0])
        b0 = float(f.readline()[:-1].split(":")[-1].split()[0].strip())

        data_file = pd.read_csv(f)
        rx = mag.Point(
            np.c_[
                data_file['Easting (m)'],
                data_file['Northing (m)'],
                data_file['Elevation (m)']
            ],
            components=['tmi']
        )
        src = mag.UniformBackgroundField(rx, amplitude=b0, inclination=inc, declination=dec)
        srvy = mag.Survey(src)

        data_mag = data.Data(
            srvy,
            dobs=data_file['Total Magnetic Anomaly (nT)'],
            standard_deviation=data_file['Standard Deviation (nT)']
        )
    return data_mag

# setup the mesh
def setup_mesh(survey, dh, padding=1.5, dz=None):
    xyz = survey.receiver_locations
    # survey extent (bounding box)
    loc_min = xyz.min(axis=0)[:-1]
    loc_max = xyz.max(axis=0)[:-1]
    width = loc_max - loc_min
    middle = np.median(xyz, axis=0)[:-1]

    width = 2 * padding * width + width

    nx = int(width[0]//dh) + 1
    ny = int(width[1]//dh) + 1

    if dz is None:
        dz = dh/2

    nz = int((np.max(width)/4)//dz) + 1
    # get to nearest power of 2 for treemesh
    nx = int(2**np.ceil(np.log2(nx)))
    ny = int(2**np.ceil(np.log2(ny)))
    nz = int(2**np.ceil(np.log2(nz)))

    hx = np.full(nx, dh)
    hy = np.full(ny, dh)
    hz = np.full(nz, dz)

    mesh = discretize.TreeMesh([hx, hy, hz], origin='CCN')

    mesh_middle = np.r_[
        mesh.origin[0] + np.sum(mesh.h[0])/2,
        mesh.origin[1] + np.sum(mesh.h[1])/2,
    ]
    shift = np.r_[middle - mesh_middle, 0]
    mesh.origin += shift

    mesh.refine_ball(xyz, dh * 4, -1, finalize=False)
    mesh.refine_ball(xyz, dh * 8, -2, finalize=False)
    mesh.refine_ball(xyz, dh * 16, -3, finalize=True)
    return mesh

# setup simulation
def get_simulation(mesh, survey, n_processes=4):
     return mag.Simulation3DIntegral(mesh, survey=survey, chiMap=maps.IdentityMap(mesh), n_processes=n_processes)


def run_inversion(
        data,
        simulation,
        alpha_s=1.0,
        length_scale_x=None,
        length_scale_y=None,
        length_scale_z=None,
        reference_model=None,
        max_iter=20,
        lower_bound=0,
        upper_bound=1,
):
    dmis = data_misfit.L2DataMisfit(data=data, simulation=simulation)

    # Define the regularization (model objective function).
    mesh = simulation.mesh
    if reference_model is None:
        reference_model = np.zeros(mesh.n_cells)
    reg = regularization.WeightedLeastSquares(
        mesh,
        alpha_s=alpha_s,
        length_scale_x=length_scale_x,
        length_scale_y=length_scale_y,
        length_scale_z=length_scale_z,
        reference_model=reference_model,
    )

    # Define how the optimization problem is solved. Here we will use a projected
    # Gauss-Newton approach that employs the conjugate gradient solver.
    opt = optimization.ProjectedGNCG(
        maxIter=max_iter, lower=lower_bound, upper=upper_bound, maxIterLS=20, maxIterCG=100, tolCG=1e-5
    )

    # Here we define the inverse problem that is to be solved
    inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)
    starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=1e3)
    beta_schedule = directives.BetaSchedule(coolingFactor=5, coolingRate=1)
    save_iteration = directives.SaveOutputEveryIteration(save_txt=False)
    update_jacobi = directives.UpdatePreconditioner()
    target_misfit = directives.TargetMisfit(chifact=1)
    sensitivity_weights = directives.UpdateSensitivityWeights(every_iteration=False)

    # The directives are defined as a list.
    directives_list = [
        sensitivity_weights,
        starting_beta,
        beta_schedule,
        save_iteration,
        update_jacobi,
        target_misfit,
    ]

    # Here we combine the inverse problem and the set of directives
    inv = inversion.BaseInversion(inv_prob, directives_list)

    m_0 = np.zeros(mesh.n_cells) + (upper_bound - lower_bound) / 2
    # Run inversion
    recovered_model = inv.run(m_0)
    return recovered_model