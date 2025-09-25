import sys, pathlib
current_file_path = pathlib.Path(__file__).parent
sys.path.append(str(current_file_path.parent.parent.parent))

from gpgn_utilities.poly_interact import PolygonEditor

import plotly.graph_objects as go
import numpy as np
import ipywidgets as widgets
from ipyevents import Event

from Labs.gravity.grav_utils.gpoly import gpoly

class GravInteract(widgets.VBox):

    def __init__(self, locations, data=None, std=None):

        locations = np.atleast_2d(locations)

        arg_sort = np.argsort(locations[:, 0])  # sort on x-position
        self.locations = locations[arg_sort]

        self.editor = PolygonEditor(
            width=1000, height=400, points=locations, layout=widgets.Layout(justify_content="center", max_width="100%")
        )

        x = self.locations[:, 0]
        self.gz = np.zeros(self.n_data)
        self.pre_plot = go.Scatter(x=x, y=self.gz, mode='lines')
        plotly_plots = [self.pre_plot]

        self.data = data
        self.std = std

        self.density = widgets.FloatSlider(
            value=1.0, min=-10, max=10, step=0.01, description="Δρ (g/cc)",
            continuous_update=True, readout_format=".2f", orientation='horizontal', readout=True,
            layout=widgets.Layout(width="80%")
        )

        annotations = []
        if self.data is not None:
            self.tie_position = widgets.IntSlider(
                value=0, min=0, max=len(self.data)-1, layout=widgets.Layout(width="80%"),
                description = "Data tie index", orientation='horizontal',
            )
            self.tie_position.observe(self._update_fig)
            self.data = np.asarray(self.data)[arg_sort]
            scatter_args = dict(x=x, y=data, mode="markers")
            if self.std is not None:
                self.std = np.asarray(self.std)[arg_sort]
                scatter_args["error_y"] = dict(type="data", array=std, visible=True)
            else:
                self.std = np.ones_like(self.data)
            plotly_plots.append(go.Scatter(**scatter_args))
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
                annotations=annotations, showlegend=False, margin=margin, modebar=modebar, autosize=True, width=None, height=200,
                xaxis_title="Profile Distance (m)",
                yaxis_title="Gz (mGal)",
            ),
        )

        self.editor.on_poly_update(self._calc_grav_data)

        self.density.observe(self._update_fig)
        
        pieces = [self.fig, self.density]
        if self.data is not None:
            pieces.append(self.tie_position)
        pieces.append(self.editor)

        super().__init__(pieces, layout=widgets.Layout(justify_content="center", max_width="70%"))


    @property
    def n_data(self):
        return len(self.locations)
    
    @property
    def data_misfit(self):
        d_diff = self.density.value * self.gz - self.data
        tie_ind = self.tie_position.value
        d_diff -= d_diff[tie_ind]
        d_diff /= self.std
        return np.dot(d_diff, d_diff) / self.n_data
    
    def _calc_grav_data(self, polygons):
        self.gz = np.zeros(len(self.locations))
        for poly in polygons:
            self.gz -= gpoly(self.locations, poly, 1.0)
        self._update_fig()

    def _update_fig(self, change=None):
        with self.fig.batch_update():  # more efficient batch updates
            gz = self.gz * self.density.value
            if self.data is not None:
                tie_ind = self.tie_position.value
                shift = self.data[tie_ind] - gz[tie_ind]
                gz += shift
                self.fig.layout.annotations[0].text = f"Misfit = {self.data_misfit:.3f}"
            self.fig.data[0].y = gz

    def get_polygons(self):
        return [np.asarray(poly, copy=True) for poly in self.editor.polygons]
    
    def set_polygons(self, polys):
        self.editor.add_polygons(polys)

    def get_data(self):
        return self.gz.copy()

