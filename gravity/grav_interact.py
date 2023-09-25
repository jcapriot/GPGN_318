from ipywidgets import HTML, VBox, Label, Widget, FloatSlider, HBox, IntSlider, Layout
from ipyleaflet import Map, DrawControl, GeoJSON, Polyline, WidgetControl, projections, Marker, DivIcon, AwesomeIcon
import matplotlib.pyplot as plt

from gpoly import gpoly
import numpy as np

class GravInteract():
    
    def __init__(self, obs, data, std=None):
        
        self._obs = obs
        self._data = data
        self._std = None
        
        m = Map(layers=[], center=(0, 0), crs=projections.Simple, zoom=6)
        
        grid = []
        labels = []
        for i in np.linspace(-50, 50, 101):
            grid.append(
                Polyline(
                    locations=[(i, -720), (i, 720)],
                    color="black",
                    weight=3 if i==0 else 1,
                )
            )
            icon = DivIcon(html=str(-i), icon_size=[0, 0])
            labels.append(
                Marker(icon=icon, location=(i, 0), draggable=False)
            )

        for i in np.linspace(-50, 50, 101):
            grid.append(
                Polyline(
                    locations=[(-360, i), (360, i)],
                    color="black",
                    weight=3 if i==0 else 1,
                )
            )
            if i != 0:
                icon = DivIcon(html=str(i), icon_size=[0, 0])
                labels.append(
                    Marker(icon=icon, location=(0, i), draggable=False)
                )

        for line in grid:
            m.add_layer(line)

        for label in labels:
            m.add_layer(label)
            
        # observation Markers
        for x, y in self._obs:
            icon = AwesomeIcon(name='close')#, icon_size=[20, 20])
            y = -y
            marker = Marker(icon=icon, location=(y, x), draggable=False)
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
        
    def _update_raw_data(self, json_data):
        gz = np.zeros(self._obs.shape[0])
        if json_data is not None:
            for item in json_data:
                nodes = np.asarray(item['geometry']['coordinates']).squeeze()
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
        self._tie.set_data(self._obs[tie_index, 0], gz[tie_index])
        
    def display(self):
        display(self._box)
        
    def _repr_html_(self):
        return self._box
    
    def get_polygons(self):
        dc = self._draw_control
        data = dc.data
        polys = []
        for item in data:
            xy = np.asarray(item['geometry']['coordinates'])[0]
            xy[:, 1] *= -1
            polys.append(xy[:-1])
        return polys
    
    def add_polygon(self, nodes):
        dc = self._draw_control
        
        state = dc.get_state()
        
        nodes = np.asarray(nodes).copy()
        nodes[:, -1] *= -1
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
        