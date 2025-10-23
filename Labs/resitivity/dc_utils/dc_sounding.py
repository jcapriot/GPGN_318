import sys, pathlib
current_file_path = pathlib.Path(__file__).parent
sys.path.append(str(current_file_path.parent.parent.parent))

from gpgn_utilities.interact_1d import ConductivityModelCanvas
import plotly.graph_objects as go

import numpy as np
from ipywidgets import VBox, HBox, Layout, Output, Button, SelectionSlider, Box

from simpeg.electromagnetics.static import resistivity as dc
from simpeg import (
    maps,
    data,
    data_misfit,
    regularization,
    optimization,
    inverse_problem,
    inversion,
    directives,
)
from simpeg.electromagnetics.static import utils as static_utils
import discretize

class DCSoundingInteract(VBox):
    
    def __init__(self, A, B, M, N, observed_voltage=None, standard_deviation=None, sigma_0=None, n_layer=1):

        def check_dims(locs):
            locs = np.atleast_1d(locs)

            if locs.ndim == 1:
                locs = locs[:, None]
            if locs.ndim == 2 and locs.shape[1] < 3:
                locs = np.pad(locs, [(0, 0), (0, 3-locs.shape[1])])
            return locs

        A = check_dims(A)
        B = check_dims(B)
        M = check_dims(M)
        N = check_dims(N)

        self._output = Output(layout={'border': '1px solid black'})

        # volt/apparent resistivity display
        self._dtype_toggle = SelectionSlider(
            options=['Volts', 'Apparent Conductivity', 'Apparent Resistivity'],
            description='x-axis',
            orientation='horizontal',
            disabled=False,
            readout=False,
        )

        
        def dtype_toggle_call(change):
            dtype = self._dtype_toggle.value
            if dtype == 'Apparent Conductivity':
                self.fig.update_layout(xaxis_title="Apparent Conductivity (S/m)")
            elif dtype == 'Apparent Resistivity':
                self.fig.update_layout(xaxis_title="Apparent Resistivity (Ohm m)")
            else:
                self.fig.update_layout(xaxis_title='Voltage (V)')
            self._update_fig()

        # make the simpeg survey
        
        src_list = []
        for a, b, m, n in zip(A, B, M, N):
            if n is None:
                rx = dc.receivers.Pole(m, data_type='volt')
            else:
                rx = dc.receivers.Dipole(m, n, data_type='volt')
            if b is None:
                src = dc.sources.Pole(rx, a)
            else:
                src = dc.sources.Dipole(rx, a, b)
            src_list.append(src)
        survey = dc.Survey(src_list)

        if observed_voltage is not None:
            if standard_deviation is None:
                standard_deviation = 0.05 * np.abs(observed_voltage)
            dobs = data.Data(survey, dobs=observed_voltage, standard_deviation=standard_deviation)
            self._dobs = dobs
        else:
            self._dobs = None
        
        self._sim = dc.Simulation1DLayers(survey=survey)

        if sigma_0 is None:
            if observed_voltage is not None:
                sigma_aps = static_utils.geometric_factor(survey)/observed_voltage
                sigma_0 = np.median(sigma_aps)
            else:
                sigma_0 = 1.0
                
        self._ab_half = np.linalg.norm(survey.locations_a - survey.locations_b, axis=-1) * 0.5
        sigma = np.repeat(sigma_0, n_layer)

        if n_layer > 1:
            min_depth_guess = np.log10(self._ab_half.min() * 10)
            max_depth_guess = np.log10(self._ab_half.max() * 10)
            depths = np.r_[0, np.logspace(min_depth_guess, max_depth_guess, n_layer - 1)]
            thick = depths[1:] - depths[:-1]
        else:
            thick = []

        self.editor = ConductivityModelCanvas(
            thick, sigma, width=1200, height=1800,
            layout=Layout(justify_content="center", max_width="100%")
        )
        self.editor.monitor = self._output

        self._calc_volts(*self.editor.get_model())

        self._ab_half = np.linalg.norm(survey.locations_a - survey.locations_b, axis=-1) * 0.5

        self.pre_plot = go.Scatter(
            x=np.abs(self._dpred_volts),
            y=self._ab_half,
            mode="markers",
            marker=dict(color=np.where(self._dpred_volts > 0, 'blue', 'red')),
            name='Observed'
        )
        plotly_plots = [self.pre_plot]

        annotations = []
        if self._dobs is not None:
            self.obs_plot = go.Scatter(
                x=np.abs(self._dobs.dobs),
                y=self._ab_half,
                mode='markers',
                marker=dict(color=np.where(self._dobs.dobs > 0, 'green', 'yellow'), symbol='hourglass'),
                error_x=dict(type="data", array=self._dobs.standard_deviation, visible=True),
                name='Predicted'
            )
            plotly_plots.append(self.obs_plot)
            annotations.append(
                dict(
                    text=f"Misfit = {self.data_misfit:.3f}",
                    ax = 1,
                    ay = 1,
                    x = 1,
                    y = 1,
                    showarrow = False,
                    align = 'right',
                    xanchor = 'right',
                    yanchor = 'top',
                    xref = 'paper',
                    yref = 'paper',
                )
            )
        margin = dict(b=0, l=0, r=0, t=20)
        modebar = dict(remove=["lasso", "select"])
        self.fig = go.FigureWidget(
            data=plotly_plots,
            layout=dict(
                annotations=annotations, showlegend=(self._dobs is not None), margin=margin, modebar=modebar, autosize=True, width=None, height=500,
                xaxis_title="Voltage (V)",
                yaxis_title="AB / 2 (m)",
                xaxis_type='log',
                yaxis_type='log'
            ),
        )
        self.fig.update_yaxes(autorange='reversed')

        self.editor.on_model_update(self._update_data_plot)
        
        # setup function for clicking a line:
        # Define a function for handling button press events

        self._invert_button = Button(
            description='Run an inversion',
            disabled=observed_voltage is None,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Invert',
            icon='square-left', # (FontAwesome names without the `fa-` prefix)
        )

        def inv_button_call(change):
           self._run_inversion()


        self._invert_button.on_click(inv_button_call)
        self._dtype_toggle.observe(dtype_toggle_call)

        right_box = VBox([self.fig, self._dtype_toggle, self._invert_button], layout=Layout(height="900", max_width="45%"))
        edit_box = Box([self.editor], layout=Layout(max_width="45%"))
        top = HBox([edit_box, right_box],
            layout=Layout(justify_content="center", max_height="900")
        )
        super().__init__([top, self._output],
            layout=Layout(justify_content="center")
        )

        self._dtype_toggle.value = "Apparent Conductivity"

    @property
    def n_data(self):
        return self._sim.survey.nD

    def _calc_volts(self, thicknesses, conductivities):
        sim = self._sim
        sim.sigma = conductivities
        sim.thicknesses = thicknesses
        dpred = sim.dpred(None)
        self._dpred_volts = dpred
    
    @property
    def data_misfit(self):
        d_diff = self._dobs.dobs - self._dpred_volts
        d_diff /= self._dobs.standard_deviation
        return np.dot(d_diff, d_diff) / self.n_data
    
    def _update_data_plot(self, thicknesses, conductivities):
        self._calc_volts(thicknesses, conductivities)
        self._update_fig()

    def _update_fig(self):
        with self.fig.batch_update():  # more efficient batch updates
            dpred = self._dpred_volts
            if self._dobs is not None:
                self.fig.layout.annotations[0].text = f"Misfit = {self.data_misfit:.3f}"
            
            if self._dtype_toggle.value != "Volts":
                g = static_utils.geometric_factor(self._sim.survey)
                if self._dtype_toggle.value == "Apparent Resistivity":
                    dpred = dpred / g
                else:
                    dpred = g / dpred
        
            da_min, da_max = [np.abs(dpred).min(), np.abs(dpred).max()]
            if self._dobs is not None:
                d_obs = self._dobs.dobs
                rel_std =  np.abs(self._dobs.standard_deviation / d_obs)
                if self._dtype_toggle.value != "Volts":
                    g = static_utils.geometric_factor(self._sim.survey)
                    if self._dtype_toggle.value == "Apparent Resistivity":
                        d_obs = d_obs / g
                    else:
                        d_obs = g / d_obs
                std = rel_std * d_obs
                self.fig.data[1].x = np.abs(d_obs)
                self.fig.data[1].error_x = dict(type="data", array=std, visible=True)
                self.fig.data[1].marker = dict(color=np.where(d_obs > 0, 'green', 'yellow'), symbol='hourglass')
                da_min = min(da_min, np.abs(d_obs).min())
                da_max = max(da_max, np.abs(d_obs).max())
    
            self.fig.data[0].x = np.abs(dpred)
            self.fig.data[0].marker = dict(color=np.where(dpred > 0, 'blue', 'red'))

            self.fig.update_layout(xaxis_range=[np.log10(da_min)-0.25, np.log10(da_max)+0.25])

    def get_data(self):
        """Get the data associated with the current model.

        Returns
        -------
        numpy.ndarray
            The forward modeled data in volts.
        """
        return self._dpred_volts

    def get_model(self):
        """

        Returns
        -------
        resistivity : (n_layer,) numpy.ndarray
            resistivity of each layer, from the surface downward, in ohm*m.
        thicknesses : (n_layer-1,) numpy.ndarray
            thicknesses of each layer, from the surface downward, in meters.
        """
        return self.editor.get_model()

    def set_model(self, thicknesses, conductivities):
        """Sets the current model to have the given resistivity and thicknesses.

        Parameters
        ----------
        resistivity : (n_layer, ) array_like
            resistivity of each layer, from the surface downward, in ohm*m.
        thicknesses : (n_layer-1, ) array_like
            thicknesses of each layer, from the surface downward, in meters.
            if n_layer == 1, this is optional.
        """
        self.editor.set_model(thicknesses, conductivities)
        self._calc_volts(thicknesses, conductivities)
        self._update_fig()

    def _run_inversion(self):
        # get the current initial model
        if self._dobs is None:
            return
        dobs = self._dobs
        thick, sigma = self.get_model()

        # determine if the number of unknowns is less than the number of data points
        underdetermined = (len(sigma) + len(thick)) > dobs.nD

        # if underdetermined need to set up a regularization
        # set the mappings for the simulation
        n_layers = len(sigma)
        if n_layers > 1:
            mapping = maps.Wires(('sigma', n_layers), ('thick', n_layers-1))
            self._sim.sigma = None
            self._sim.thicknesses = None
            self._sim.sigmaMap = maps.ExpMap() * mapping.sigma
            self._sim.thicknessesMap = maps.ExpMap() * mapping.thick

            init_model = np.log(np.r_[sigma, thick])
        else:
            self._sim.sigma = None
            self._sim.thicknesses = []
            self._sim.sigmaMap = maps.ExpMap(nP=1)

            init_model = np.log(sigma)
        mesh = discretize.TensorMesh([len(init_model)])

        lower_sigma = np.log(1E-8)
        upper_sigma = np.log(1E8)
        lower_thick = np.log(1E-3)
        upper_thick = np.log(1E7)
        upper = np.r_[np.full_like(sigma, upper_sigma), np.full_like(thick, upper_thick)]
        lower = np.r_[np.full_like(sigma, lower_sigma), np.full_like(thick, lower_thick)]


        # Define the data misfit. Here the data misfit is the L2 norm of the weighted
        # residual between the observed data and the data predicted for a given model.
        # Within the data misfit, the residual between predicted and observed data are
        # normalized by the data's standard deviation.
        dmis = data_misfit.L2DataMisfit(simulation=self._sim, data=dobs)

        # Define the regularization (model objective function)
        reg = regularization.WeightedLeastSquares(
            mesh, alpha_s=1.0, alpha_x=0.0, reference_model=init_model
        )

        # Define how the optimization problem is solved. Here we will use an inexact
        # Gauss-Newton approach that employs the conjugate gradient solver.
        opt = optimization.ProjectedGNCG(maxIter=30, maxIterCG=20, lower=lower, upper=upper)

        # Define the inverse problem
        inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

        # Setting a stopping criteria for the inversion.
        target_misfit = directives.TargetMisfit(chifact=1)

        if underdetermined:
            # Apply and update sensitivity weighting as the model updates
            update_sensitivity_weights = directives.UpdateSensitivityWeights()

            # Defining a starting value for the trade-off parameter (beta) between the data
            # misfit and the regularization.
            starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=1e1)

            # Set the rate of reduction in trade-off parameter (beta) each time the
            # the inverse problem is solved. And set the number of Gauss-Newton iterations
            # for each trade-off parameter value.
            beta_schedule = directives.BetaSchedule(coolingFactor=5.0, coolingRate=3.0)

            # The directives are defined as a list.
            directives_list = [
                update_sensitivity_weights,
                starting_beta,
                beta_schedule,
                target_misfit,
            ]
        else:
            inv_prob.beta = 0.0

            # The directives are defined as a list.
            directives_list = [
            ]

        self._output.clear_output()
        with self._output:
            # Here we combine the inverse problem and the set of directives
            inv = inversion.BaseInversion(inv_prob, directives_list)

            # Run the inversion
            recovered_model = inv.run(init_model)
        self._sim.model = recovered_model
        sigma = self._sim.sigma
        thick = self._sim.thicknesses

        self._sim.rhoMap = None
        self._sim.thicknessesMap = None
        self._sim.model = None

        self.set_model(thick, sigma)



        