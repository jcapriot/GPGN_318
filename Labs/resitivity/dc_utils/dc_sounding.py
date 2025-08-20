import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import HTML, VBox, Label, Widget, FloatSlider, HBox, IntSlider, Layout, ToggleButtons, Output, Button

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
    utils,
)
from simpeg.electromagnetics.static import utils as static_utils
import discretize

class DCSoundingInteract():
    
    def __init__(self, A, B, M, N, observed_voltage=None, standard_deviation=None, rho_0=1):

        n_layer = 1
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

        rho = np.repeat(rho_0, n_layer)
        thick = 10**(np.repeat(2, n_layer-1))

        self._model = self._rho_thick_to_model(rho, thick)

        # volt/apparent resistivity display
        self._dtype_toggle = ToggleButtons(
            options=['Apparent Resistivity', 'Volts'],
            description='Data View',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltips=['Display the data in voltage', 'Display the data as apparent resistivity',],
        )
        
        # make the simpeg survey
        
        src_list = []
        for a, b, m, n in zip(A, B, M, N):
            if n is None:
                rx = dc.receivers.Pole(m, data_type='volt')
            else:
                rx = dc.receivers.Dipole(m, n)
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
        
        # Create a figure and axis
        with plt.ioff():
            fig, [ax_model, ax_data] = plt.subplots(1, 2)
        self._fig = fig
        
        toolbar = plt.get_current_fig_manager().toolbar
        ax_model.set_xscale('log')
        ax_model.set_yscale('log')
        ax_model.invert_yaxis()

        ax_model.set_ylim([1.05 * self._model[-1, 1], 0.5 * self._model[1, 1]])
        ax_model.grid(True)

        ax_model.set_xlabel(r'resistivity ($\Omega$ m)')
        ax_model.set_ylabel(r'Depth (m)')

        ax_data.set_yscale('log')
        ax_data.set_xscale('log')
        ax_data.yaxis.set_label_position("right")
        ax_data.yaxis.tick_right()
        ax_data.invert_yaxis()
        ax_data.set_ylabel(r'Array halfwidth (m)')
        ax_data.grid(True)
        self._ax_data = ax_data
        self._ax_model = ax_model
        
        # Initialize the line with some initial data
        self._line_model, = ax_model.plot(self._model[:, 0], self._model[:, 1], color='C0')
        self._markers_model, = ax_model.plot(self._model[:-1, 0], self._model[:-1, 1], color='C0', marker='o', linewidth=0)
        self._end_mark_model, = ax_model.plot(self._model[[-1], 0], self._model[[-1], 1], color='C0', marker='v', linewidth=0)

        ab_half = np.linalg.norm(survey.locations_a - survey.locations_b, axis=-1) * 0.5

        dpred = self._dpred()
        self._line_data, = ax_data.plot(dpred, ab_half, marker='.', linewidth=0)
        da_min, da_max = [0.95 * dpred.min(), 1.05 * dpred.max()]
        ax_data.set_xlabel(r'$\rho_a$ ($\Omega m$)')

        if observed_voltage is not None:
            g = static_utils.geometric_factor(survey)
            rho_obs = observed_voltage / g
            self._data_line, = ax_data.plot(rho_obs, ab_half, marker='x', color='C1', linewidth=0)
            da_min = min(da_min, 0.95*rho_obs.min())
            da_max = max(da_max, 1.05*rho_obs.max())
        ax_data.set_xlim([da_min, da_max])

        def toggle_obs(change):
            if self._dtype_toggle.value == 'Apparent Resistivity':
                ax_data.set_xlabel(r'$\rho_a$ ($\Omega m$)')
            else:
                ax_data.set_xlabel('Potential (Volts)')
            self._update_dpred_plot()

        self._dtype_toggle.observe(toggle_obs)
        
        self.__dragging = False
        self.__selection_point = None
        self.__segment_ind = None
        self.__segment_start = None
        
        # setup function for clicking a line:
        # Define a function for handling button press events
        def on_press(event):
            if event.inaxes == ax_model and toolbar.mode != 'pan/zoom':
                contains, attrd = self._line_model.contains(event)
                if contains:
                    selection_point = [event.xdata, event.ydata]
                    close_inds = attrd['ind']
                    if len(close_inds) == 0:
                        close_inds = np.array([-2])
                    closest = np.argmin(np.linalg.norm(selection_point - self._model[close_inds]))
                    segment_ind = close_inds[closest]
                    if event.key == 'shift':
                        # need to split segment and trigger redraw of data
                        nodes_before = self._model[:segment_ind+1]
                        nodes_after = self._model[segment_ind+1:]

                        if segment_ind % 2 == 0:
                            # horizontal use event.xdata
                            new_point = np.array([[nodes_before[-1, 0], selection_point[1]]])
                        else:
                            new_point = np.array([[selection_point[0], nodes_before[-1, 1]]])

                        # find closest point on line to selection
                        new_nodes = np.r_[nodes_before, new_point, new_point, nodes_after]
                        self._model = new_nodes
                        self._update_model_plot()
                        self._update_dpred_plot()
                    else:
                        self.__selection_point = selection_point
                        self.__segment_ind = segment_ind
                        self.__segment_start = self._model[segment_ind:segment_ind+2].copy()
                        self.__dragging = True

        # Define a function for handling mouse motion events
        def on_motion(event):
            if self.__dragging and event.inaxes == ax_model and toolbar.mode != 'pan/zoom':
                segment_ind = self.__segment_ind
                selection_point = self.__selection_point
                segment_start = self.__segment_start
                x, y = event.xdata, event.ydata
                # vertical line
                if segment_ind % 2 == 0:
                    dx = x - selection_point[0]
                    dy = 0
                # horizontal line
                else:
                    dx = 0
                    dy = y - selection_point[1]
                new_points = segment_start + [dx, dy]
                # if horizontal line, check for bounds on heights
                if segment_ind % 2 == 1:
                    new_y = new_points[0, 1]
                    # need to check if it is valid
                    if segment_ind + 2 < self._model.shape[0]:
                        next_y = self._model[segment_ind + 2, 1]
                        new_y = min(new_y, next_y)
                    new_y = max(new_y, self._model[segment_ind - 1, 1])
                    new_points[:, 1] = new_y

                # put some reasonable guardrails on interacting.
                # new_points = np.maximum(1E-15, new_points)
                # new_points = np.minimum(1E15, new_points)

                self._model[segment_ind:segment_ind + 2] = new_points
                self._update_model_plot()
                self._update_dpred_plot()

        # Define a function for handling button release events
        def on_release(event):
            self.__dragging = False

        # Connect the event handlers
        fig.canvas.mpl_connect('button_press_event', on_press)
        fig.canvas.mpl_connect('motion_notify_event', on_motion)
        fig.canvas.mpl_connect('button_release_event', on_release)

        self._output = Output(layout={'border': '1px solid black'})

        self._invert_button = Button(
            description='Run an inversion',
            disabled=observed_voltage is None,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Invert',
            icon='square-left', # (FontAwesome names without the `fa-` prefix)
        )

        def button_call(change):
           self._run_inversion()

        self._invert_button.on_click(button_call)

        delete_button = Button(
           description='Remove Last Layer',
           disabled=False,
           button_style='',  # 'success', 'info', 'warning', 'danger' or ''
           tooltip='remove the last layer of the model',
           icon='x',  # (FontAwesome names without the `fa-` prefix)
        )
        def delete_call(change):
            if len(self._model) > 2:
                self._model = self._model[:-2]
                self._update_model_plot()
                self._update_dpred_plot()

        delete_button.on_click(delete_call)

        button_box = HBox([self._invert_button, delete_button, self._dtype_toggle])

        self._box = VBox([button_box, fig.canvas, self._output])

    def _dpred(self, volts=False):
        rho = self._model[::2, 0]
        thick = np.diff(self._model[::2, 1])
        sim = self._sim
        sim.rhoMap = None
        sim.thicknessesMap = None
        sim.rho = rho
        sim.thicknesses = thick
        dpred = sim.dpred(None)
        # optionally do this?
        if self._dtype_toggle.value == 'Apparent Resistivity' and not volts:
            g = static_utils.geometric_factor(self._sim.survey)
            dpred = dpred/g
        return dpred
    
    def _update_model_plot(self):
        self._line_model.set_data(np.atleast_1d(self._model[:, 0]), np.atleast_1d(self._model[:, 1]))
        self._markers_model.set_data(np.atleast_1d(self._model[:-1, 0]), np.atleast_1d(self._model[:-1, 1]))
        self._end_mark_model.set_data(np.atleast_1d(self._model[-1, 0]), np.atleast_1d(self._model[-1, 1]))
        
    def _update_dpred_plot(self):
        dpred = self._dpred()
        self._line_data.set_xdata(dpred)
        da_min, da_max = [dpred.min(), dpred.max()]

        if self._dobs is not None:
            d_obs = self._dobs.dobs
            if self._dtype_toggle.value == 'Apparent Resistivity':
                g = static_utils.geometric_factor(self._sim.survey)
                d_obs = d_obs/g
            self._data_line.set_xdata(np.atleast_1d(d_obs))
            da_min = min(da_min, d_obs.min())
            da_max = max(da_max, d_obs.max())

        self._ax_data.set_xlim([0.95*da_min, 1.05*da_max])

    def display(self):
        return self._box

    def get_data(self):
        """Get the data associated with the current model.

        Returns
        -------
        numpy.ndarray
            The forward modeled data in volts.
        """
        return self._dpred(volts=True)

    def get_model(self):
        """

        Returns
        -------
        resistivity : (n_layer,) numpy.ndarray
            resistivity of each layer, from the surface downward, in ohm*m.
        thicknesses : (n_layer-1,) numpy.ndarray
            thicknesses of each layer, from the surface downward, in meters.
        """
        rho = self._model[::2, 0]
        thick = np.diff(self._model[::2, 1])

        return rho, thick

    def _rho_thick_to_model(self, resistivity, thicknesses):
        rhos = np.c_[resistivity, resistivity].reshape(-1)
        if len(thicknesses) == 0:
            depths = np.r_[0.0, 100.0]
        else:
            thicks = np.r_[thicknesses, thicknesses[-1]*10]
            depths = np.cumsum(thicks)
            depths = np.r_[0, np.c_[depths, depths].reshape(-1)[:-1]]
        return np.c_[rhos, depths]

    def set_model(self, resistivity, thicknesses=None):
        """Sets the current model to have the given resistivity and thicknesses.

        Parameters
        ----------
        resistivity : (n_layer, ) array_like
            resistivity of each layer, from the surface downward, in ohm*m.
        thicknesses : (n_layer-1, ) array_like
            thicknesses of each layer, from the surface downward, in meters.
            if n_layer == 1, this is optional.
        """
        resistivity = np.atleast_1d(resistivity)
        n_layer = resistivity.shape[0]

        if n_layer == 1 and thicknesses is None:
            thicknesses = np.array([])
        thicknesses = np.atleast_1d(thicknesses)

        if thicknesses.shape[0] != n_layer - 1:
            raise ValueError(
                'Incompatible number of resistivities and thicknesses. '
                f'There were {thicknesses.shape[0]}, but I expected {n_layer - 1}.'
            )

        self._model = self._rho_thick_to_model(resistivity, thicknesses)
        self._update_model_plot()

        # update the axis limits
        xlim = [0.95 * resistivity.min(), 1.05 * resistivity.max()]
        ylim = [1.05 * self._model[-1, 1], 0.5 * self._model[1, 1]]

        self._ax_model.set_xlim(xlim)
        self._ax_model.set_ylim(ylim)
        self._update_dpred_plot()

    def _run_inversion(self):
        # get the current initial model
        if self._dobs is None:
            return
        dobs = self._dobs
        rho = self._model[::2, 0]
        thick = np.diff(self._model[::2, 1])
        thick = np.maximum(thick, 0.1)

        # determine if the number of unknowns is less than the number of data points
        underdetermined = (len(rho) + len(thick)) > dobs.nD

        # if underdetermined need to set up a regularization
        # set the mappings for the simulation
        n_layers = len(rho)
        if n_layers > 1:
            mapping = maps.Wires(('rho', n_layers), ('thick', n_layers-1))
            self._sim.rho = None
            self._sim.thicknesses = None
            self._sim.rhoMap = maps.ExpMap() * mapping.rho
            self._sim.thicknessesMap = maps.ExpMap() * mapping.thick

            init_model = np.log(np.r_[rho, thick])
        else:
            self._sim.rho = None
            self._sim.thicknesses = []
            self._sim.rhoMap = maps.ExpMap(nP=1)

            init_model = np.log(rho)
        mesh = discretize.TensorMesh([len(init_model)])

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
        opt = optimization.InexactGaussNewton(maxIter=30, maxIterCG=20)

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
        rho = self._sim.rho
        thick = self._sim.thicknesses

        self._sim.rhoMap = None
        self._sim.thicknessesMap = None
        self._sim.model = None

        self.set_model(rho, thick)



        