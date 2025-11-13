import sys, pathlib
current_file_path = pathlib.Path(__file__).parent
sys.path.append(str(current_file_path.parent.parent.parent))

from gpgn_utilities.poly_interact import PolygonEditor

import plotly.graph_objects as go
import numpy as np
import ipywidgets as widgets
from scipy.constants import mu_0

from Labs.magnetics.mag_utils.mpoly import mpoly

class MagPolyInteract(widgets.VBox):

    def __init__(self, locations):

        locations = np.atleast_2d(locations)

        arg_sort = np.argsort(locations[:, 0])  # sort on x-position
        self.locations = locations[arg_sort]

        self.editor = PolygonEditor(
            width=1000, height=400, points=locations, layout=widgets.Layout(justify_content="center", max_width="100%")
        )

        x = self.locations[:, 0]
        self.mag = np.zeros((self.locations.shape[0], 3))
        self.mx_plot = go.Scatter(x=x, y=self.mag[:, 0], mode='lines', name=r'B_a,x')
        self.mz_plot = go.Scatter(x=x, y=self.mag[:, 1], mode='lines', name=r'B_a,down')
        self.tmi_plot = go.Scatter(x=x, y=self.mag[:, 2], mode='lines', name='TMI')
        plotly_plots = [self.mx_plot, self.mz_plot, self.tmi_plot]

        self.susc = widgets.FloatLogSlider(
            value=0.05, min=-4, max=1, step=0.01, description="Susceptibility",
            continuous_update=True, readout_format=".2f", orientation='horizontal', readout=True,
            layout=widgets.Layout(width="80%")
        )

        self.inclination = widgets.FloatSlider(
            value = 45, min=-90, max=90, step=0.01, description="Inclination (deg)", readout=True
        )
        self.inducing_strength = widgets.FloatSlider(
            value=40_000, min=0, max=70_000, steps=1, description="Inducing field strength (nT)", readout=True
        )

        margin = dict(b=0, l=0, r=0, t=20)
        modebar = dict(remove=["lasso", "select"])
        self.fig = go.FigureWidget(
            data=plotly_plots,
            layout=dict(
                showlegend=True, margin=margin, modebar=modebar, autosize=True, width=None, height=200,
                xaxis_title="Profile Distance (m)",
                yaxis_title="B (nT)",
            ),
        )

        self.editor.on_poly_update(self._calc_mv_data)

        self.susc.observe(self._update_fig)
        self.inducing_strength.observe(self._update_fig)
        def inc_update(change=None):
            polys = self.get_polygons()
            self._calc_mv_data(polys)
        self.inclination.observe(inc_update)
        
        pieces = [self.fig, self.susc, self.inclination, self.inducing_strength]
        pieces.append(self.editor)

        super().__init__(pieces, layout=widgets.Layout(justify_content="center", max_width="70%"))

    @property
    def _mag_dir(self):
        inc = np.deg2rad(self.inclination.value)
        mx = np.cos(inc)
        mz = np.sin(inc)
        return np.asarray([mx, mz])

    def set_susceptibility(self, value):
        self.susc.value = value

    def get_susceptibility(self):
        return self.susc.value

    def set_field_strength(self, value):
        self.inducing_strength.value = value

    def get_field_strength(self):
        return self.inducing_strength.value

    def set_inclination(self, value):
        self.inclination.value = value

    def get_inclination(self):
        return self.inclination.value
    
    def _calc_mv_data(self, polygons):
        self.mag = np.zeros((len(self.locations), 3))
        locs = self.locations * [1, -1]
        for poly in polygons:
            poly = np.asarray(poly) * [1, -1]
            self.mag += mpoly(locs, poly, self._mag_dir)
        self._update_fig()

    def _update_fig(self, change=None):
        with self.fig.batch_update():  # more efficient batch updates
            mag = self.get_data()
            self.fig.data[0].y = mag[:, 0]
            self.fig.data[1].y = mag[:, 1]
            self.fig.data[2].y = mag[:, 2]

    def get_polygons(self):
        return [np.asarray(poly, copy=True) for poly in self.editor.polygons]
    
    def set_polygons(self, polys):
        self.editor.add_polygons(polys)

    def get_data(self):
        return self.mag * self.susc.value * self.inducing_strength.value / mu_0

