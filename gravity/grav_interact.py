from ipywidgets import HTML, VBox, Label, Widget, FloatSlider, HBox, IntSlider, Layout
from ipyleaflet import Map, DrawControl, GeoJSON, Polyline, WidgetControl, projections, Marker, DivIcon, AwesomeIcon
import matplotlib.pyplot as plt

from gpoly import gpoly
import numpy as np

class GravInteract():
    
    def __init__(self, obs, data, std=None, xmarks=None, zmarks=None):
        """Create a widget to interactively model and fit gravity data

        The coordinate system is x positive to the right, and z positive
        down.

        Parameters
        ----------
        obs : (n_data, 2) numpy.ndarray
            The observation point locations.
        data : (n_data, ) numpy.ndarray
            The observed gravity values in mGal. (positive down).
        std : (n_data, ) numpy.ndarray, optional
            Standard deviations of the observed gravity values.
        
        """

        obs = np.asarray(obs)
        data = np.asarray(data)
        if std is not None:
            std = np.asarray(std)
        self._obs = obs
        self._data = data
        self._std = std
        
        m = Map(layers=[], center=(0, 0), crs=projections.Simple, zoom=1)
        
        grid = []
        labels = []
        xmin, zmin = obs.min(axis=0)
        xmax, zmax = obs.max(axis=0)

        if xmarks is None or zmarks is None:
            dx = xmax - xmin
            dz = zmax - zmin
            max_d = max(dx, dz)
            d_mark = 10 ** (np.floor(np.log10(max_d)) - 1)
            n_d = int(2 * (max_d // d_mark))
            if xmarks is None:
                center = np.round((xmin + xmax) / d_mark) * d_mark
                start = (center - d_mark * n_d // 2)
                end = (center + d_mark * n_d // 2)
                xmarks = np.linspace(start, end, n_d+1)
            if zmarks is None:
                center = np.round((zmin + zmax) / d_mark) * d_mark
                start = (center - d_mark * n_d // 2)
                end = (center + d_mark * n_d // 2)
                zmarks = np.linspace(start, end, n_d + 1)

        x_center = xmarks[len(xmarks)//2]
        zmin = zmarks.min()
        zmax = zmarks.max()

        z_center = zmarks[len(zmarks)//2]
        xmin = xmarks.min()
        xmax = xmarks.max()

        self._xscale = (xmax - xmin) / 1000
        self._zscale = (zmax - zmin) / 1000
        self._xshift = x_center
        self._zshift = z_center

        for mark in xmarks:
            local_x, _ = self._global_to_local(mark, 0)
            grid.append(
                Polyline(
                    locations=[(-500, local_x), (500, local_x)],
                    color="black",
                    weight=3 if mark == x_center else 1,
                )
            )
            icon = DivIcon(html=str(mark), icon_size=[0, 0])
            labels.append(
                Marker(icon=icon, location=(0, local_x), draggable=False)
            )
        for mark in zmarks:
            _, local_z = self._global_to_local(0, -mark)
            grid.append(
                Polyline(
                    locations=[(local_z, -500), (local_z, 500)],
                    color="black",
                    weight=3 if mark == z_center else 1,
                )
            )
            if mark != z_center:
                icon = DivIcon(html=str(mark), icon_size=[0, 0])
                labels.append(
                    Marker(icon=icon, location=(local_z, 0), draggable=False)
                )

        for line in grid:
            m.add_layer(line)

        for label in labels:
            m.add_layer(label)
            
        # observation Markers
        for x, z in self._obs:
            icon = AwesomeIcon(name='close') #, icon_size=[20, 20])
            local_x, local_z = self._global_to_local(x, -z)
            marker = Marker(icon=icon, location=(local_z, local_x), draggable=False)
            m.add_layer(marker)
            
        draw_control = DrawControl(
            polyline = {},
            circlemarker = {},
            polygon = {"shapeOptions": {"color": "#f06eaa"}, "allowIntersection": False}
        )

        with plt.ioff():
            fig = plt.figure()
        ax = fig.gca()
        if std is None:
            self._data_plot = ax.scatter(obs[:, 0], data, zorder=0)
        else:
            self._data_plot = ax.errorbar(obs[:, 0], data, yerr=std, fmt="o", zorder=0)
        self._for_dat = np.zeros(obs.shape[0])
        self._line = ax.plot(obs[:, 0], self._for_dat, zorder=1, color='C1', linewidth=0, marker='s')[0]
        self._tie = ax.plot(obs[0, 0], data[0], marker='s', color='C2', linewidth=0, zorder=2)[0]
        
        ax.set_ylabel(r'$g_d$ (mGal)')
        ax.set_xlabel(r'Profile (m)')
        
        def handle_trait(change):
            if change['name'] == '_property_lock':
                if isinstance(change['old'], dict):
                    json_data = change['old'].get("data", None)
                    self._update_raw_data(json_data)
                    self._update_plot()

        draw_control.observe(handle_trait)

        m.add_control(draw_control)
        
        lat_lon_display = Label()

        def handle_interaction(**kwargs):
            if kwargs.get('type') == 'mousemove':
                y, x = kwargs.get('coordinates')
                x, y = self._local_to_global(x, y)
                lat_lon_display.value = f"x={x} , depth={-y}"

        m.on_interaction(handle_interaction)
        
        self._density_slider = FloatSlider(
            value=1.0,
            min=-5,
            max=5,
            step=0.01,
            description=r'$\Delta \rho$ (g/cc):',
            disabled=False,
            continuous_update=True,
            readout=True,
            readout_format='.2f',
            layout = Layout(width='50%')
        )
                
        self._tie_slider = IntSlider(
            value=0,
            min=0,
            max=obs.shape[0]-1,
            description=r'Tie Index:',
            disabled=False,
            continuous_update=True,
            readout=True,
            layout = Layout(width='50%')
        )
        
        def slider_handle(change):
            if change['name'] == 'value':
                self._update_plot()
        
        self._density_slider.observe(slider_handle)
        self._tie_slider.observe(slider_handle)
        
        self._update_plot()
        
        self._m = m
        self._draw_control = draw_control
        fig.canvas.layout.width = '100%'

        sliders = HBox(children = [self._density_slider, self._tie_slider], layout=Layout(width='100%'))
        self._box = VBox(children=[fig.canvas, sliders, m, lat_lon_display])

    def _local_to_global(self, x, z):
        return x * self._xscale + self._xshift, z * self._zscale + self._zshift

    def _global_to_local(self, x, z):
        return (x - self._xshift) / self._xscale , (z - self._zshift) / self._zscale
        
    def _update_raw_data(self, json_data):
        gz = np.zeros(self._obs.shape[0])
        if json_data is not None:
            for item in json_data:
                nodes = np.asarray(item['geometry']['coordinates']).squeeze()
                nodes = np.c_[self._local_to_global(nodes[:, 0], nodes[:, 1])]
                nodes[:, 1] *= -1
                gz += gpoly(self._obs, nodes, 1.0)
        self._for_dat = gz
        return gz
    
    def _update_plot(self):
        tie_index = self._tie_slider.value
        density = self._density_slider.value
        
        gz = density * self._for_dat
        
        dz = self._data[tie_index] - gz[tie_index]
        
        gz += dz
        
        self._line.set_data(self._obs[:, 0], gz)
        self._tie.set_data([self._obs[tie_index, 0]], [gz[tie_index]])
        
    def display(self):
        display(self._box)
        
    def _repr_html_(self):
        return self._box
    
    def get_polygons(self):
        """Get a list of polygons in the current modeler widget

        Returns
        -------
        polys : (n_poly, ) list of (n_node, 2) numpy.ndarray
            The nodes of the polygons.
        """
        dc = self._draw_control
        data = dc.data
        polys = []
        for item in data:
            xy = np.asarray(item['geometry']['coordinates'])[0]
            xy = np.c_[self._local_to_global(xy[:, 0], xy[:, 1])]
            xy[:, 1] *= -1
            polys.append(xy[:-1])
        return polys
    
    def add_polygon(self, nodes):
        """Add a polygon to the modeler.

        Parameters
        ----------
        nodes : (n_nodes, 2) numpy.ndarray
            The nodes of the polygons, listed in order (either CW or CCW).
        """
        dc = self._draw_control
        
        state = dc.get_state()
        
        nodes = np.asarray(nodes).copy()
        nodes = np.c_[self._global_to_local(nodes[:, 0], -nodes[:, 1])]
        nodes = np.r_[nodes, [nodes[0]]]

        new_poly = {
            'type': 'Feature',
            'properties': {'style': {'stroke': True,
                'color': '#f06eaa',
                'weight': 4,
                'opacity': 0.5,
                'fill': True,
                'fillColor': None,
                'fillOpacity': 0.2,
                'clickable': True}},
            'geometry': {'type': 'Polygon',
                'coordinates': [nodes]}
        }
        state['data'].append(new_poly)
        
        dc.send_state(state)
        
        self._update_raw_data(state['data'])
        self._update_plot()
        