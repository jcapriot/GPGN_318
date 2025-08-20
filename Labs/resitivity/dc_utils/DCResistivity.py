import numpy as np

from simpeg import maps, data
from simpeg.electromagnetics.static import resistivity as DC
import pandas as pd
from simpeg.utils.solver_utils import get_default_solver

DSolver = get_default_solver()

from simpeg import (data_misfit, regularization,
    optimization, inverse_problem, inversion, directives, utils,
    Data
)

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import matplotlib
import warnings
warnings.filterwarnings("ignore")
matplotlib.rcParams['font.size'] = 14

from ipywidgets import GridspecLayout, widgets, fixed
import os
from scipy.interpolate import griddata
from discretize import TensorMesh

class DCRInversionApp(object):
    """docstring for DCRInversionApp"""

    uncertainty = None
    mesh = None
    mesh_core = None
    actind = None
    IO = None
    survey = None
    phi_d = None
    phi_m = None
    dpred = None
    m = None
    sigma_air = 1e-8
    topo = None
    _Jmatrix = None
    _JtJ = None
    _doi_index = None
    doi = False
    use_iterative = False


    def __init__(self):
        super(DCRInversionApp, self).__init__()
        self.IO = DC.IO()

    @property
    def Jmatrix(self):
        if self._Jmatrix is None:
            self._Jmatrix = self.problem.getJ(self.m[-1])
        return self._Jmatrix

    @property
    def JtJ(self):
        if self._JtJ is None:
            self._JtJ = np.sqrt((self.Jmatrix ** 2).sum(axis=0)) / self.mesh.cell_volumes
        return self._JtJ

    @property
    def P(self):
        if self._P is None:
            self._P = self.mesh.getInterpolationMat(self.mesh_core.gridCC, locType='CC')
        return self._P

    def set_mesh(self, dx=None, dz=None, corezlength=None, show_core=None, xpad=None, zpad=None, mesh_type='TensorMesh'):

        sort_ind = np.argsort(self.IO.electrode_locations[:,0])
        if self.topo is None:
            topo_tmp = self.IO.electrode_locations[sort_ind,:]
        else:
            topo_tmp = self.topo.copy()
        tmp_x = np.r_[-1e10, topo_tmp[:,0], 1e10]
        tmp_z = np.r_[topo_tmp[0,1], topo_tmp[:,1], topo_tmp[-1,1]]
        topo = np.c_[tmp_x, tmp_z]

        npad_x = 0
        val = 0.
        while val < xpad/self.IO.dx:
            npad_x = npad_x + 1
            val = val + 1.3**npad_x

        npad_z = 0
        val = 0.
        while val < zpad/self.IO.dz:
            npad_z = npad_z + 1
            val = val + 1.3**npad_z

        # if dx == 'None':
        #     dx = None

        # if dz == 'None':
        #     dz = None

        # if corezlength == 'None':
        #     corezlength = None

        # if npad_x == 'None':
        #     npad_x = 10

        # if npad_z == 'None':
        #     npad_z = 10

        self.mesh, self.actind = self.IO.set_mesh(
            topo=topo,
            method='linear',
            dx=dx,
            dz=dz,
            npad_x=npad_x,
            npad_z=npad_z,
            corezlength=corezlength,
            mesh_type=mesh_type
        )

        if dx is not None:
            print(f'Total cells in mesh : {self.mesh.nC}')
            print(f'Active cells in mesh : {np.sum(self.actind)}')
            fig, ax = plt.subplots(1,1, figsize=(10, 5))
            self.mesh.plot_image(
                self.actind, grid=True, ax=ax,
                grid_opts={'color':'white', 'alpha':0.5}
            )

            ax.plot(self.IO.electrode_locations[:,0], self.IO.electrode_locations[:,1], 'k.')
            # src = self.survey.srcList[i_src]
            # rx = src.rxList[0]

            # src_type = self.IO.survey_type .split('-')[0]
            # rx_type = self.IO.survey_type .split('-')[1]

            # if src_type == 'dipole':
            #     ax.plot(src.loc[0][0], src.loc[0][1], 'ro')
            #     ax.plot(src.loc[1][0], src.loc[1][1], 'bo')
            # elif src_type == 'pole':
            #     ax.plot(src.loc[0], src.loc[1], 'ro')
            # if rx_type == 'dipole':
            #     m, n = rx.locs[0], rx.locs[1]
            #     ax.plot(m[:,0],m[:,1],'yo', ms=8)
            #     ax.plot(n[:,0],n[:,1],'go', ms=4)
            # elif rx_type == 'pole':
            #     m = rx.locs
            #     ax.plot(m[:,0],m[:,1],'yo', ms=8)

            ax.set_aspect(1)
            ax.set_xlabel("x (m")
            ax.set_ylabel("z (m")
            if show_core:
                ax.set_xlim(self.IO.xyzlim[0,:])
                ax.set_ylim(self.IO.xyzlim[1,:])
            else:
                ax.set_ylim(self.mesh.nodes_y.min(), self.mesh.nodes_y.max() + 10)

    def load_obs(self, A, B, M, N, rho_a):

        self.survey = self.IO.from_abmn_locations_to_survey(
            A, B, M, N, survey_type='dipole-dipole', data_dc=rho_a, data_dc_type="apparent_resistivity"
        )
        print (">> survey type: {}".format(self.IO.survey_type))
        print ("   # of data: {0}".format(self.survey.nD))
        rho_0 = self.get_initial_resistivity()
        print ((">> suggested initial resistivity: %1.f ohm-m")%(rho_0))

    def get_problem(self):
        store_J = True
        solver_type = DSolver
        actmap = maps.InjectActiveCells(
            self.mesh, indActive=self.actind, valInactive=np.log(self.sigma_air),
        )
        mapping = maps.ExpMap(self.mesh) * actmap
        sim = DC.Simulation2DNodal(
            self.mesh,
            sigmaMap=mapping,
            storeJ=store_J,
            solver=solver_type,
            survey=self.survey,
            miniaturize=True
        )
        return sim

    def get_initial_resistivity(self):
        out = np.histogram(np.log10(abs(self.IO.voltages/self.IO.G)))
        return 10**out[1][np.argmax(out[0])]


    def set_uncertainty(self, percentage, floor, choice, set_value=True):
        self.percentage = percentage
        self.floor = floor
        dobs = self.IO.voltages

        if set_value:
            self.uncertainty = abs(dobs) * percentage / 100.+ floor
            print ((">> percent error: %.1f and floor error: %.2e are set") % (percentage, floor))
        else:
            self.uncertainty = self.survey.std.copy()
            print (">> uncertainty in the observation file is used")
        if np.any(self.uncertainty==0.):
            print ("warning: uncertainty includse zero values!")

        fig, ax = plt.subplots(1, 1, figsize=(12, 7))

        dobs_sorted = np.sort(np.abs(dobs))
        k = np.argsort(np.abs(dobs))
        dunc_sorted = self.uncertainty[k]
        x = np.linspace(1, len(dobs_sorted), len(dobs_sorted))
        if choice == 'linear':
            ax.plot(x, dobs_sorted-dunc_sorted, 'k:', x, dobs_sorted+dunc_sorted, 'k:')
            ax.fill_between(x, dobs_sorted-dunc_sorted, dobs_sorted+dunc_sorted, facecolor=[0.7, 0.7, 0.7], interpolate=True)
            ax.plot(x, dobs_sorted, 'k', lw=2)
        else:
            ax.semilogx(x, dobs_sorted-dunc_sorted, 'k:', x, dobs_sorted+dunc_sorted, 'k:')
            ax.fill_between(x, dobs_sorted-dunc_sorted, dobs_sorted+dunc_sorted, facecolor=[0.7, 0.7, 0.7], interpolate=True)
            ax.semilogy(x, dobs_sorted, 'k', lw=2)

        ax.set_xlabel("Datum")
        ax.set_ylabel("Volts")
        ax.set_title("Data and uncertainties sorted by absolute value")


    def run_inversion(
        self,
        rho_0,
        rho_ref=None,
        alpha_s=1e-3,
        alpha_x=1,
        alpha_z=1,
        maxIter=20,
        chifact=1.,
        beta0_ratio=1.,
        coolingFactor=5,
        coolingRate=2,
        rho_upper=np.inf,
        rho_lower=-np.inf,
        run=True,
    ):
        # self.use_iterative=use_iterative
        if run:
            maxIterCG=20
            self.problem = self.get_problem()
            m0 = np.ones(self.actind.sum()) * np.log(1./rho_0)
            if rho_ref is None:
                rho_ref = rho_0
            mref = np.ones(self.actind.sum()) * np.log(1./rho_ref)

            dc_data = data.Data(self.survey, dobs=self.IO.voltages)
            dmis = data_misfit.L2DataMisfit(
                data=dc_data, simulation=self.problem
            )
            dmis.W = 1./self.uncertainty
            reg = regularization.WeightedLeastSquares(
                mesh=self.mesh,
                active_cells=self.actind,
                alpha_s=alpha_s,
                alpha_x=alpha_x,
                alpha_y=alpha_z,
                mapping=maps.IdentityMap(nP=int(self.actind.sum())),
                reference_model=mref
            )
            # Personal preference for this solver with a Jacobi preconditioner
            opt = optimization.ProjectedGNCG(
                maxIter=maxIter, maxIterCG=maxIterCG, print_type='ubc'
            )
            opt.factor = 1
            opt.remember('xc')
            invProb = inverse_problem.BaseInvProblem(dmis, reg, opt)
            beta = directives.BetaEstimate_ByEig(beta0_ratio=beta0_ratio)
            target = directives.TargetMisfit(chifact=chifact)
            beta_schedule = directives.BetaSchedule(
                coolingFactor=coolingFactor,
                coolingRate=coolingRate
            )
            save = directives.SaveOutputEveryIteration()
            save_outputs = directives.SaveOutputDictEveryIteration()
            sense_weight = directives.UpdateSensitivityWeights()
            inv = inversion.BaseInversion(
                invProb,
                directiveList=[beta, target, beta_schedule, save_outputs]
            )

            minv = inv.run(m0)

            # Store all inversion parameters

            if self.doi:
                self.m_doi = minv.copy()
            else:
                self.alpha_s = alpha_s
                self.alpha_x = alpha_x
                self.alpha_z = alpha_z
                self.rho_0 = rho_0
                self.rho_ref = rho_ref
                self.beta0_ratio = beta0_ratio
                self.chifact = chifact
                self.maxIter = maxIter
                self.coolingFactor = coolingFactor
                self.coolingRate = coolingRate

                self.phi_d = []
                self.phi_m = []
                self.m = []
                self.dpred = []

                for key in save_outputs.outDict.keys():
                    self.phi_d.append(save_outputs.outDict[key]["phi_d"].copy() * 2.0)
                    self.phi_m.append(save_outputs.outDict[key]["phi_m"].copy() * 2.0)
                    self.m.append(save_outputs.outDict[key]["m"].copy())
                    self.dpred.append(save_outputs.outDict[key]["dpred"].copy())

        else:
            pass

    def interact_set_mesh(self):
        self.IO.set_mesh()
        dx = widgets.FloatText(value=self.IO.dx, min=0.1, max=100.)
        dz = widgets.FloatText(value=self.IO.dz, min=0.1, max=100.)
        corezlength = widgets.FloatText(value=self.IO.corezlength*5, min=0.1)
        xpad = widgets.FloatText(value=2*self.IO.corezlength, min=0., max=10000.)
        zpad = widgets.FloatText(value=2*self.IO.corezlength, min=0., max=10000.)

        # npad_x = widgets.IntSlider(value=self.IO.npad_x, min=1, max=30)
        # npad_z = widgets.IntSlider(value=self.IO.npad_z, min=1, max=30)

        npad_x = 1
        val = 1.3
        while val < 2*self.IO.corezlength/self.IO.dx:
            npad_x = npad_x + 1
            val = val + 1.3**npad_x

        npad_z = 1
        val = 1.3
        while val < 2*self.IO.corezlength/self.IO.dz:
            npad_z = npad_z + 1
            val = val + 1.3**npad_z

        # i_src = widgets.IntSlider(value=0, min=0, max=self.survey.nSrc-1, step=1)
        show_core = widgets.Checkbox(
                value=True, description="show core region only", disabled=False
        )

        mesh_type = widgets.ToggleButtons(
            value='TensorMesh', options=['TensorMesh', 'TREE']
        )

        print(">> suggested dx: {} m".format(self.IO.dx))
        print(">> suggested dz: {} m".format(self.IO.dz))
        print(">> suggested x padding: {} m".format(2*self.IO.corezlength))
        print(">> suggested z padding: {} m".format(2*self.IO.corezlength))
        print(">> suggested corezlength: {} m".format(self.IO.corezlength))

        widgets.interact(
            self.set_mesh,
                dx=dx,
                dz=dz,
                corezlength=corezlength,
                show_core=show_core,
                xpad=xpad,
                zpad=zpad,
                mesh_type=mesh_type,
                # i_src=i_src
        )

    def plot_obs_data(self, data_type, plot_type, aspect_ratio):
        fig, ax = plt.subplots(1,1, figsize=(10, 5))
        if plot_type == "pseudo-section":
            self.IO.plotPseudoSection(aspect_ratio=1, cmap='viridis', data_type=data_type, ax=ax)
            ax.set_aspect(aspect_ratio)
        elif plot_type == "histogram":
            if data_type == "apparent_resistivity":
                out = ax.hist(np.log10(self.IO.apparent_resistivity), edgecolor='k')
                xlabel = r'App. Res ($\Omega$m)'
                xticks = ax.get_xticks()
                ax.set_xticklabels([ ("%.1f")%(10**xtick)for xtick in xticks])

            elif data_type == "volt":
                out = ax.hist(np.log10(abs(self.IO.voltages)), edgecolor='k')
                xlabel = 'Voltage (V)'
                xticks = ax.get_xticks()
                ax.set_xticklabels([ ("%.1e")%(10**xtick)for xtick in xticks])
            ax.set_ylabel('Count')
            ax.set_xlabel(xlabel)

    def plot_misfit_curve(self, iteration, scale='linear', curve_type='misfit'):
        fig, ax = plt.subplots(1,1, figsize=(10, 5))
        if curve_type == "misfit":
            ax_1 = ax.twinx()
            ax.plot(np.arange(len(self.phi_m))+1, self.phi_d, 'k.-')
            ax_1.plot(np.arange(len(self.phi_d))+1, self.phi_m, 'r.-')
            ax.plot(iteration, self.phi_d[iteration-1], 'ko', ms=10)
            ax_1.plot(iteration, self.phi_m[iteration-1], 'ro', ms=10)

            xlim = plt.xlim()
            ax.plot(xlim, np.ones(2)*self.survey.nD, 'k--')
            ax.set_xlim(xlim)
            ax.set_xlabel("Iteration")
            ax.set_ylabel(r"$\phi_d$", fontsize=16)
            ax_1.set_ylabel(r"$\phi_m$", fontsize=16)
            ax.set_yscale(scale)
            ax_1.set_yscale(scale)
            ax.set_title(("Misfit / Target misfit: %1.f / %.1f")%(self.phi_d[iteration-1], self.survey.nD))
        elif curve_type == "tikhonov":
            ax.plot(self.phi_m, self.phi_d, 'k.-')
            ax.plot(self.phi_m[iteration-1], self.phi_d[iteration-1], 'ko', ms=10)
            ax.set_ylabel(r"$\phi_d$", fontsize=16)
            ax.set_xlabel(r"$\phi_m$", fontsize=16)
            ax.set_xscale(scale)
            ax.set_yscale(scale)

    def plot_data_misfit(self, iteration, aspect_ratio=1):
        dobs = self.IO.voltages
        appres = self.IO.apparent_resistivity

        vmin, vmax = appres.min(), appres.max()
        dpred = self.dpred[iteration-1]
        appres_pred = dpred / self.IO.G
        fig, axs = plt.subplots(3,1, figsize = (10, 9))
        self.IO.plotPseudoSection(data=appres, clim=(vmin, vmax), aspect_ratio=1, ax=axs[0], cmap='viridis', scale='log')
        self.IO.plotPseudoSection(data=appres_pred, clim=(vmin, vmax), aspect_ratio=1, ax=axs[1], cmap='viridis', scale='log')
        misfit = (dpred-dobs) / self.uncertainty
        self.IO.plotPseudoSection(
            data=misfit, data_type='volt', scale='linear', aspect_ratio=1, ax=axs[2], clim=(-3, 3),
            label='Normalized Misfit', cmap='seismic'
        )
        titles = ["Observed", "Predicted", "Normalized misfit"]
        for i_ax, ax in enumerate(axs):
            ax.set_title(titles[i_ax])
            ax.set_aspect(aspect_ratio)

    def plot_model(
        self, iteration, 
        vmin=None, 
        vmax=None, 
        aspect_ratio=1, 
        scale="log", 
        show_core=True, 
        show_grid=False, 
        reverse_color=False
    ):
        clim = (vmin, vmax)
        # inds_core, self. = Utils.ExtractCoreMesh(self.IO.xyzlim, self.mesh)
        fig, ax = plt.subplots(1,1, figsize=(10, 5))
        tmp = 1./(self.problem.sigmaMap*self.m[iteration-1])
        tmp[~self.actind] = np.nan
        if clim is None:
            vmin, vmax = tmp[self.actind].min(), tmp[self.actind].max()
        else:
            vmin, vmax = clim

        if reverse_color:
            cmap_type = 'viridis_r'
        else:
            cmap_type = 'viridis'

        if scale == "log":
            norm = LogNorm(vmin, vmax)
            ticks = np.logspace(np.log10(vmin), np.log10(vmax), 4)
        else:
            norm = Normalize(vmin, vmax)
            ticks = np.linspace(vmin, vmax, 4)

        if show_grid:
            grid_opts = {"color": "white", "alpha": 0.5}
        else:
            grid_opts = {}

        out = self.mesh.plot_image(
            tmp, grid=show_grid, pcolor_opts={'cmap':cmap_type, 'norm':norm}, ax=ax,
            grid_opts=grid_opts
        )
        cb = plt.colorbar(out[0], orientation='horizontal', format="%.1f", fraction=0.06, ax=ax, ticks=ticks)
        cb.ax.minorticks_off()
        cb.ax.set_xlabel(r'Resistivity ($\Omega m$)')

        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
        ax.set_aspect(aspect_ratio)
        if show_core:
            ymin, ymax = self.IO.xyzlim[1,:]
            xmin, xmax = self.IO.xyzlim[0,:]
            dy = (ymax-ymin)/10.
            ax.set_ylim(ymin, ymax+dy)
            ax.set_xlim(xmin, xmax)
        else:
            ymin, ymax = self.mesh.nodes_y.min(), self.mesh.nodes_y.max()
            xmin, xmax = self.mesh.nodes_x.min(), self.mesh.nodes_x.max()
            dy = (ymax-ymin)/10.
            ax.set_ylim(ymin, ymax+dy)
            ax.set_xlim(xmin, xmax+dy)

        plt.tight_layout()

    def plot_sensitivity(self, show_core, show_grid, scale, aspect_ratio):
        # inds_core, self. = Utils.ExtractCoreMesh(self.IO.xyzlim, self.mesh)
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        tmp = self.JtJ
        tmp[~self.actind] = np.nan
        vmin, vmax = np.nanmin(tmp), np.nanmax(tmp)

        if scale == "log":
            norm = LogNorm(vmin, vmax)
            ticks = np.logspace(np.log10(vmin), np.log10(vmax), 4)
        else:
            norm = Normalize(vmin, vmax)
            ticks = np.linspace(vmin, vmax, 4)

        if show_grid:
            grid_opts = {"color": "white", "alpha": 0.5}
        else:
            grid_opts = {}

        out = self.mesh.plot_image(
            tmp,
            grid=show_grid,
            pcolor_opts={"cmap": "viridis",'norm':norm},
            ax=ax,
            grid_opts=grid_opts,
        )
        cb = plt.colorbar(
            out[0], orientation="horizontal", fraction=0.06, ticks=ticks, format="%.1e", ax=ax
        )
        cb.ax.minorticks_off()

        ax.plot(
            self.IO.electrode_locations[:, 0],
            self.IO.electrode_locations[:, 1],
            "wo",
            markeredgecolor="k",
        )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
        ax.set_aspect(aspect_ratio)
        if show_core:
            ymin, ymax = self.IO.xyzlim[1, :]
            xmin, xmax = self.IO.xyzlim[0, :]
            dy = (ymax - ymin) / 10.0
            ax.set_ylim(ymin, ymax + dy)
            ax.set_xlim(xmin, xmax)
        else:
            ymin, ymax = self.mesh.nodes_y.min(), self.mesh.nodes_y.max()
            xmin, xmax = self.mesh.nodes_x.min(), self.mesh.nodes_x.max()
            dy = (ymax - ymin) / 10.0
            ax.set_ylim(ymin, ymax + dy)
            ax.set_xlim(xmin, xmax + dy)

        plt.tight_layout()

    def plot_inversion_results(
        self,
        iteration=1,
        curve_type='misfit',
        scale='log',
        plot_type='misfit_curve',
        rho_min=100,
        rho_max=1000,
        aspect_ratio=1,
        show_grid=False,
        show_core=True,
        reverse_color=False
    ):
        if plot_type == "misfit_curve":
            self.plot_misfit_curve(
                iteration, curve_type=curve_type,
                scale=scale
            )
        elif plot_type == "model":
            self.plot_model(
                iteration,
                vmin=rho_min,
                vmax=rho_max,
                aspect_ratio=aspect_ratio,
                show_core=show_core,
                show_grid=show_grid,
                scale=scale,
                reverse_color=reverse_color
            )
        elif plot_type == "data_misfit":
            self.plot_data_misfit(
                iteration,
                aspect_ratio=aspect_ratio
            )
        elif plot_type == "sensitivity":
            self.plot_sensitivity(
                scale=scale,
                show_core=show_core,
                show_grid=show_grid,
                aspect_ratio=aspect_ratio,
            )
        else:
            raise NotImplementedError()

    def plot_model_doi(
        self,
        vmin=None,
        vmax=None,
        show_core=True,
        show_grid=False,
        scale="log",
        aspect_ratio=1,
        reverse_color=False,
    ):
        problem = self.get_problem()
        rho1 = 1.0 / (problem.sigmaMap * self.m[self.m_index])
        rho2 = 1.0 / (problem.sigmaMap * self.m_doi)
        rho1[~self.actind] = np.nan
        rho2[~self.actind] = np.nan

        if reverse_color:
            cmap = "viridis_r"
        else:
            cmap = "viridis"

        if scale == "log":
            norm = LogNorm(vmin, vmax)
            ticks = np.logspace(np.log10(vmin), np.log10(vmax), 4)
        else:
            norm = Normalize(vmin, vmax)
            ticks = np.linspace(vmin, vmax, 4)

        fig, axs = plt.subplots(2, 1, figsize=(10, 7))
        ax1 = axs[0]
        ax2 = axs[1]

        if show_grid:
            grid_opts = {"color": "white", "alpha": 0.5}
        else:
            grid_opts = {}

        out = self.mesh.plot_image(
            rho1,
            grid=show_grid,
            pcolor_opts={"cmap": cmap, 'norm':norm},
            ax=ax1,
            grid_opts=grid_opts,
        )
        self.mesh.plot_image(
            rho2,
            grid=show_grid,
            pcolor_opts={"cmap": cmap, 'norm':norm},
            ax=ax2,
            grid_opts=grid_opts,
        )

        for ii, ax in enumerate(axs):

            ax.plot(
                self.IO.electrode_locations[:, 0],
                self.IO.electrode_locations[:, 1],
                "wo",
                markeredgecolor="k",
            )
            if ii!=0:
                ax.set_xlabel("x (m)")
            else:
                ax.set_xlabel(" ")
            ax.set_ylabel("z (m)")
            ax.set_aspect(aspect_ratio)
            if show_core:
                ymin, ymax = self.IO.xyzlim[1, :]
                xmin, xmax = self.IO.xyzlim[0, :]
                dy = (ymax - ymin) / 10.0
                ax.set_ylim(ymin, ymax + dy)
                ax.set_xlim(xmin, xmax)
            else:
                ymin, ymax = self.mesh.nodes_y.min(), self.mesh.nodes_y.max()
                xmin, xmax = self.mesh.nodes_x.min(), self.mesh.nodes_x.max()
                dy = (ymax - ymin) / 10.0
                ax.set_ylim(ymin, ymax + dy)
                ax.set_xlim(xmin, xmax + dy)

        cb = fig.colorbar(out[0], ax=axs[:], format="%.1f", fraction=0.02, ticks=ticks, location='right')
        cb.ax.set_ylabel(r'Resistivity ($\Omega m$)')
        cb.ax.minorticks_off()
    #     cb = plt.colorbar(out[0], orientation='horizontal', format="%.1f", fraction=0.02, ax=ax2, ticks=ticks)
    #     plt.tight_layout()

    def plot_doi_index(
        self,
        show_core=True,
        show_grid=False,
        vmin=0,
        vmax=2,
        level=0.3,
        k=100,
        power=2,
        aspect_ratio=1,
        reverse_color=False
    ):

        m1 = self.m[self.m_index]
        m2 = self.m_doi

        mref_1 = np.log(1.0 / self.rho_ref)
        mref_2 = np.log(1.0 / (self.rho_ref * self.factor))

        def compute_doi_index(m1, m2, mref_1, mref_2):
            doi_index = np.abs((m1 - m2) / (mref_1 - mref_2))
            return doi_index

        doi_index = compute_doi_index(m1, m2, mref_1, mref_2)
        tmp = np.ones(self.mesh.nC) * np.nan
        tmp[self.actind] = doi_index

        if reverse_color:
            cmap = "viridis_r"
        else:
            cmap = "viridis"

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        if show_grid:
            grid_opts = {"color": "white", "alpha": 0.5}
        else:
            grid_opts = {}

        out = self.mesh.plot_image(
            tmp,
            grid=show_grid,
            pcolor_opts={"cmap": cmap},
            ax=ax,
            grid_opts=grid_opts,
        )

        tmp_contour = np.ones(self.mesh.n_cells) * np.nan
        tmp_contour[self.actind] = doi_index
        tmp_contour = np.ma.masked_array(tmp_contour, ~self.actind)

        if self.mesh._meshType == 'TREE':
            def up_search_tree(mesh, cell_ind, doi_index, level, in_doi, inds):
                cell = mesh[cell_ind]
                neighbors = cell.neighbors[3]  # +y inds
                indoi = in_doi or (doi_index[cell_ind] < level)
                if not indoi:
                    inds.append(cell_ind)
                if neighbors == -1:
                    return inds
                elif isinstance(neighbors, list):  # loop through neighbors
                    for neighbor in neighbors:
                        inds = up_search_tree(mesh, neighbor, doi_index, level, indoi, inds)
                else:
                    inds = up_search_tree(mesh, neighbors, doi_index, level, indoi, inds)
                return inds
            # get the cells at the bottom of the mesh
            bottom_cells = self.mesh.cell_boundary_indices[2]
            inds = []
            for cell_ind in bottom_cells:
                inds = up_search_tree(self.mesh, cell_ind, tmp, level, False, inds)
        else:
            tmp_meshed = tmp.reshape(self.mesh.shape_cells, order='F')
            bot_surf_inds = np.zeros(self.mesh.shape_cells[0])
            for ix in range(self.mesh.shape_cells[0]):
                ind = 0
                while ind < self.mesh.shape_cells[1] and tmp_meshed[ix, ind] >= level:
                    ind += 1
                bot_surf_inds[ix] = ind
            inds = bot_surf_inds[:, None] >= np.arange(self.mesh.shape_cells[1])
            inds = inds.reshape(-1, order='F')

        self.doi_inds = inds
        self.doi_index = doi_index

        if self.mesh._meshType == 'TREE':
            contour_mesh = TensorMesh(self.mesh.h, self.mesh.x0)
            tmp_contour = griddata(self.mesh.cell_centers, tmp_contour, contour_mesh.cell_centers)
        else:
            contour_mesh = self.mesh

        cs = ax.contour(
            contour_mesh.cell_centers_x,
            contour_mesh.cell_centers_y,
            tmp_contour.reshape(contour_mesh.shape_cells, order="F").T,
            levels=[level],
            colors="k",
        )
        ax.clabel(cs, fmt="%.1f", colors="k", fontsize=12)  # contour line labels

        ticks = np.linspace(vmin, vmax, 3)
        cb = plt.colorbar(out[0], orientation='horizontal', format="%.1f", fraction=0.06, ax=ax, ticks=ticks)
        cb.ax.minorticks_off()


        ax.plot(
            self.IO.electrode_locations[:, 0],
            self.IO.electrode_locations[:, 1],
            "wo",
            markeredgecolor="k",
        )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
        ax.set_aspect(aspect_ratio)
        if show_core:
            ymin, ymax = self.IO.xyzlim[1, :]
            xmin, xmax = self.IO.xyzlim[0, :]
            dy = (ymax - ymin) / 10.0
            ax.set_ylim(ymin, ymax + dy)
            ax.set_xlim(xmin, xmax)
        else:
            ymin, ymax = self.mesh.nodes_y.min(), self.mesh.nodes_y.max()
            xmin, xmax = self.mesh.nodes_x.min(), self.mesh.nodes_x.max()
            dy = (ymax - ymin) / 10.0
            ax.set_ylim(ymin, ymax + dy)
            ax.set_xlim(xmin, xmax + dy)

        plt.tight_layout()

    def plot_model_with_doi(
        self,
        vmin=None,
        vmax=None,
        show_core=True,
        show_grid=False,
        scale="log",
        aspect_ratio=1,
        reverse_color=False,
    ):
        problem = self.get_problem()
        rho1 = 1.0 / (problem.sigmaMap * self.m[self.m_index])

        rho1[~self.actind] = np.nan
        rho1[self.doi_inds] = np.nan

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        if reverse_color:
            cmap = "viridis_r"
        else:
            cmap = "viridis"

        if scale == "log":
            norm = LogNorm(vmin, vmax)
            ticks = np.logspace(np.log10(vmin), np.log10(vmax), 4)
        else:
            norm = Normalize(vmin, vmax)
            ticks = np.linspace(vmin, vmax, 4)

        if show_grid:
            grid_opts = {"color": "white", "alpha": 0.5}
        else:
            grid_opts = {}

        out = self.mesh.plot_image(
            rho1,
            grid=show_grid,
            pcolor_opts={"cmap": cmap, 'norm':norm},
            ax=ax,
            grid_opts=grid_opts
        )
        cb = plt.colorbar(out[0], orientation='horizontal', format="%.1f", fraction=0.06, ax=ax, ticks=ticks)
        cb.ax.minorticks_off()
        cb.ax.set_xlabel(r'Resistivity ($\Omega m$)')

        ax.plot(
            self.IO.electrode_locations[:, 0],
            self.IO.electrode_locations[:, 1],
            "wo",
            markeredgecolor="k",
        )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
        ax.set_aspect(aspect_ratio)
        if show_core:
            ymin, ymax = self.IO.xyzlim[1, :]
            xmin, xmax = self.IO.xyzlim[0, :]
            dy = (ymax - ymin) / 10.0
            ax.set_ylim(ymin, ymax + dy)
            ax.set_xlim(xmin, xmax)
        else:
            ymin, ymax = self.mesh.nodes_y.min(), self.mesh.nodes_y.max()
            xmin, xmax = self.mesh.nodes_x.min(), self.mesh.nodes_x.max()
            dy = (ymax - ymin) / 10.0
            ax.set_ylim(ymin, ymax + dy)
            ax.set_xlim(xmin, xmax + dy)

        plt.tight_layout()

    def plot_doi_results(
        self,
        plot_type="models",
        scale="log",
        rho_min=100,
        rho_max=1000,
        doi_level=0.3,
        aspect_ratio=1,
        show_grid=False,
        show_core=True,
        reverse_color=False,
    ):
        try:
            if plot_type == "models":
                self.plot_model_doi(
                    vmin=rho_min,
                    vmax=rho_max,
                    show_core=show_core,
                    show_grid=show_grid,
                    scale=scale,
                    aspect_ratio=aspect_ratio,
                    reverse_color=reverse_color,
                )
            elif plot_type == "doi":
                self.plot_doi_index(
                    show_core=show_core,
                    show_grid=show_grid,
                    vmin=0,
                    vmax=1,
                    level=doi_level,
                    aspect_ratio=aspect_ratio,
                    reverse_color=reverse_color,
                )
            elif plot_type == "final":
                self.plot_model_with_doi(
                    vmin=rho_min,
                    vmax=rho_max,
                    show_core=show_core,
                    show_grid=show_grid,
                    scale=scale,
                    aspect_ratio=aspect_ratio,
                    reverse_color=reverse_color,
                )
            else:
                raise NotImplementedError()
        except:
            print (">> an inversion for doi calculation is needed to be run")

    def run_doi(self, factor, doi_iter, run=False):
        self.factor = factor
        self.doi = True
        self.m_index = int(doi_iter - 1)
        if run:
            self.run_inversion(
                self.rho_0 * factor,
                self.rho_ref * factor,
                alpha_s=self.alpha_s,
                alpha_x=self.alpha_x,
                alpha_z=self.alpha_z,
                maxIter=doi_iter,
                chifact=self.chifact,
                beta0_ratio=self.beta0_ratio,
                coolingFactor=self.coolingFactor,
                coolingRate=self.coolingRate,
                run=True,
            )
        self.doi = False

    def interact_run_doi(self):

        try:
            N = len(self.m)
        except TypeError:
            N = 1
        widgets.interact(
            self.run_doi,
            factor=widgets.FloatText(0.1),
            doi_iter=widgets.IntSlider(value=N, min=1, max=N)
        )

    def interact_plot_doi_results(self):
        try:
            plot_type = widgets.ToggleButtons(
                options=["models", "doi", "final"], value="doi", description="plot type"
            )
            scale = widgets.ToggleButtons(
                options=["log", "linear"], value="log", description="scale"
            )

            rho = 1.0 / np.exp(self.m[self.m_index])
            rho_min = widgets.FloatText(
                value=np.ceil(rho.min()),
                continuous_update=False,
                description="$\\rho_{min}$",
            )
            rho_max = widgets.FloatText(
                value=np.ceil(rho.max()),
                continuous_update=False,
                description="$\\rho_{max}$",
            )

            show_grid = widgets.Checkbox(
                value=False, description="show grid?", disabled=False
            )
            show_core = widgets.Checkbox(
                value=True, description="show core?", disabled=False
            )
            reverse_color = widgets.Checkbox(
                value=False, description="reverse colormap on model plot?", disabled=False
            )

            doi_level = widgets.FloatText(value=0.3)

            aspect_ratio = widgets.FloatText(value=1)

            widgets.interact(
                self.plot_doi_results,
                plot_type=plot_type,
                rho_min=rho_min,
                rho_max=rho_max,
                doi_level=doi_level,
                show_grid=show_grid,
                show_core=show_core,
                reverse_color=reverse_color,
                scale=scale,
                aspect_ratio=aspect_ratio,
            )
        except Exception as err:
            print(err)
            print (">> an inversion for doi calculation is needed to be run")

    def interact_plot_obs_data(self):
        data_type = widgets.ToggleButtons(
            options=["apparent_resistivity", "volt"],
            value="apparent_resistivity",
            description="data type"
        )
        plot_type = widgets.ToggleButtons(
            options=["pseudo-section", "histogram"],
            value="pseudo-section",
            description="plot type"
        )
        aspect_ratio = widgets.FloatText(value=1)

        widgets.interact(
            self.plot_obs_data,
            data_type=data_type,
            plot_type=plot_type,
            aspect_ratio=aspect_ratio
        )

    def interact_set_uncertainty(self):
        percentage = widgets.FloatText(value=5.)
        floor = widgets.FloatText(value=0.)
        choice = widgets.RadioButtons(
            options=['linear', 'log'],
            value='linear',
            description='Plotting Scale:',
            disabled=False
        )
        widgets.interact(
            self.set_uncertainty,
            percentage=percentage,
            floor=floor,
            choice=choice
        )

    def interact_run_inversion(self):
        run = widgets.Checkbox(
            value=False, description="run", disabled=False
        )

        rho_initial = np.ceil(self.get_initial_resistivity())
        maxIter = widgets.IntText(value=10, continuous_update=False)
        rho_0 = widgets.FloatText(
            value=rho_initial, continuous_update=False,
            description=r"$\rho_0$"
        )
        rho_ref = widgets.FloatText(
            value=rho_initial, continuous_update=False,
            description=r"$\rho_{ref}$"
            )
        percentage = widgets.FloatText(value=self.percentage, continuous_update=False,
            description="noise percent")
        floor = widgets.FloatText(value=self.floor, continuous_update=False,
            description="noise floor")
        chifact = widgets.FloatText(value=1.0, continuous_update=False)
        beta0_ratio = widgets.FloatText(value=10., continuous_update=False)
    
        coolingFactor = widgets.FloatSlider(
           min=0.1, max=10, step=1, value=2, continuous_update=False,
           description='cooling factor',
        )
        coolingRate = widgets.IntSlider(
            min=1, max=10, step=1, value=1, continuous_update=False,
            description='n_iter / beta'
        )
        alpha_s = widgets.FloatText(
            value=1e-2, continuous_update=False,
            description=r"$\alpha_{s}$"
        )
        alpha_x = widgets.FloatText(
            value=1, continuous_update=False,
            description=r"$\alpha_{x}$"
        )
        alpha_z = widgets.FloatText(
            value=1, continuous_update=False,
            description=r"$\alpha_{z}$"
        )
        # use_iterative = widgets.Checkbox(
        #     value=False, continuous_update=False, description="use iterative solver"
        # )

        widgets.interact(
            self.run_inversion,
            run=run,
            rho_initial=rho_initial,
            maxIter=maxIter,
            rho_0=rho_0,
            rho_ref=rho_ref,
            chifact=chifact,
            beta0_ratio=beta0_ratio,
            coolingFactor=coolingFactor,
            coolingRate=coolingRate,
            alpha_s=alpha_s,
            alpha_x=alpha_x,
            alpha_z=alpha_z,
            rho_upper=fixed(np.inf),
            rho_lower=fixed(-np.inf),
        )

    def interact_plot_inversion_results(self):
        try:
            iteration = widgets.IntSlider(
                min=1, max=len(self.m), step=1, value=1, continuous_update=False
            )
            curve_type = widgets.ToggleButtons(
                options=["misfit", "tikhonov"],
                value="misfit",
                description="curve type"
            )
            scale=widgets.ToggleButtons(
                options=["linear", "log"],
                value="log",
                description="scale"
            )
            plot_type = widgets.ToggleButtons(
                options=["misfit_curve", "model", "data_misfit", "sensitivity"],
                value="misfit_curve",
                description="plot type"
            )
            rho = 1./np.exp(self.m[-1])
            rho_min=widgets.FloatText(
                value=np.ceil(rho.min()), continuous_update=False,
                description="$\\rho_{min}$"
            )
            rho_max=widgets.FloatText(
                value=np.ceil(rho.max()), continuous_update=False,
                description="$\\rho_{max}$"
            )
            aspect_ratio=widgets.FloatText(
                value=1, continuous_update=False,
            )
            show_grid = widgets.Checkbox(
                value=False, description="show grid on model plot?", disabled=False
            )
            show_core = widgets.Checkbox(
                value=True, description="show core on model plot?", disabled=False
            )
            reverse_color = widgets.Checkbox(
                value=False, description="reverse color map on model plot?", disabled=False
            )

            widgets.interact(
                self.plot_inversion_results,
                iteration=iteration,
                curve_type=curve_type,
                scale=scale,
                plot_type=plot_type,
                rho_min=rho_min,
                rho_max=rho_max,
                aspect_ratio=aspect_ratio,
                show_grid=show_grid,
                show_core=show_core,
                reverse_color=reverse_color
            )
        except:
            print (">> no inversion results yet")
