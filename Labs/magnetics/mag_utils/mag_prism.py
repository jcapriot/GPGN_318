import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import HTML, VBox, Label, Widget, FloatSlider, HBox, IntSlider, Layout, ToggleButtons, Output, Button

from geoana.em.static import MagneticPrism
from scipy.constants import mu_0
from simpeg.potential_fields.magnetics.analytics import IDTtoxyz

class MagneticPrismInteract():

    def __init__(self, grid_east=None, grid_north=None, grid_height=None, prism=None):

        if grid_east is None:
            grid_east = np.linspace(-1000,1000,101)
        if grid_north is None:
            grid_north = np.linspace(-1000,1000,101)
        if grid_height is None:
            grid_height = 0
        if prism is None:
            prism = [[-100, -100, -300],[100, 100, -100]]
        self._grid_x = np.asarray(grid_north)
        self._grid_y = np.asarray(grid_east)
        self._grid_z = -grid_height
        x, y, z = np.meshgrid(self._grid_x, self._grid_y, [self._grid_z])
        self._obs = np.stack([x[..., 0], y[..., 0], z[..., 0]], axis=-1)

        self.prism = MagneticPrism(*prism)
        

        # toggle hemisphere
        self._hemi_toggle = ToggleButtons(
            options=['North', 'South'],
            description='Hemisphere',
            disabled=False,
            button_style='',
            tooltips=['Which hemisphere is your prism in?',],
            layout=Layout(width='200px')
        )

        # toggle component
        self._comp_toggle = ToggleButtons(
            options=['TFA', 'East', 'North', 'Down'],
            description='Component',
            disabled=False,
            button_style='',
            tooltips=['Which component would you like to plot?',],
            layout=Layout(width='200px')
        )

        with plt.ioff():
            #fig, ax_polar = plt.subplots(subplot_kw={'projection': 'polar'})
            #fig, [ax_polar, ax_image] = plt.subplots(1, 2, subplot_kw={'projection': 'polar'})
            fig = plt.figure(figsize=(5,2))
            ax_polar = plt.subplot(121, projection='polar')
            ax_image = plt.subplot(122)

                
        toolbar = plt.get_current_fig_manager().toolbar
        scatter = ax_polar.scatter([0],[90])
        ax_polar.set_rlim(bottom=90, top=0)
        ax_polar.grid(True)
        ax_polar.set_theta_zero_location("N")
        ax_polar.set_theta_direction(-1)
        ax_polar.set_xlabel('Declination')

        self._scatter = scatter

        self._magnitude_slider = FloatSlider(
            value=50000,
            min=25000,
            max=65000,
            step=1,
            description='|B| (nT)',
            disabled=False,
            continuous_update=True,
            orientation='vertical',
            readout=True,
            readout_format='.1f',
        )

        self._sus_slider = FloatSlider(
            value=0.1,
            min=0,
            max=1,
            step=0.01,
            description='Susceptibility (SI)',
            disabled=False,
            continuous_update=True,
            orientation='vertical',
            readout=True,
            readout_format='.2f',
        )

        B, ba = self._do_forward()
        # setup data plot
        # with plt.ioff():
        #     fig2, ax_image = plt.subplots()
        im_ba = ax_image.pcolormesh(grid_east, grid_north, ba)
        ax_image.set_aspect('equal', 'box')
        ax_image.set_xlabel('Easting (m)')
        ax_image.set_ylabel('Northing (m)')
        cb = plt.colorbar(im_ba)
        cb.set_label('Total Field Anomaly (nT)')
        self._im_ba = im_ba
        self._ax_image = ax_image
        self._cb_image = cb

        def toggle_hemi(event):
            if self._hemi_toggle.value == 'North':
                data = scatter.get_offsets()
                data[:, 1] = np.abs(data[:, 1])
                ax_polar.set_rlim(bottom=90, top=0)
                scatter.set_offsets(data)
            else:
                data = scatter.get_offsets()
                data[:, 1] = -np.abs(data[:, 1])
                ax_polar.set_rlim(bottom=-90, top=0)
                scatter.set_offsets(data)
            self._update_image()

        def slider_update(event):
            self._update_image()
                
        self._hemi_toggle.on_trait_change(toggle_hemi)
        self._magnitude_slider.on_trait_change(slider_update)
        self._sus_slider.on_trait_change(slider_update)
        self._comp_toggle.on_trait_change(slider_update)
        
        self.__dragging = False
        # # setup function for clicking a line:
        # # Define a function for handling button press events
        def on_press(event):
            if event.inaxes == ax_polar and toolbar.mode != 'pan/zoom':
                contains, attrd = scatter.contains(event)
                if contains:
                    self.__dragging = True
        
        # Define a function for handling mouse motion events
        def on_motion(event):
            if self.__dragging and event.inaxes == ax_polar and toolbar.mode != 'pan/zoom':
                x, y = event.xdata, event.ydata
                if self._hemi_toggle.value == 'North': 
                    y = 180 + y
                scatter.set_offsets([x, y])
                self._update_image()
                # update data plot
        
        # Define a function for handling button release events
        def on_release(event):
            self.__dragging = False
        
        # # Connect the event handlers
        fig.canvas.mpl_connect('button_press_event', on_press)
        fig.canvas.mpl_connect('motion_notify_event', on_motion)
        fig.canvas.mpl_connect('button_release_event', on_release)
        slider_box = HBox([self._magnitude_slider, self._sus_slider])

        vbox = VBox([self._hemi_toggle, slider_box])
        box2 = HBox([vbox, self._comp_toggle])

        box = HBox([box2, fig.canvas])

        #big_box = VBox([box, fig2.canvas])

        self._box = box

    def _do_forward(self):
        
        data = self._scatter.get_offsets()
        dec = data[0,0] * 180/np.pi
        inc = data[0,1]

        strength = self._magnitude_slider.value
        sus = self._sus_slider.value
        b0 = IDTtoxyz(inc, dec, strength)
        b0 = np.r_[b0[0], b0[1], -b0[2]]
        M = b0 / mu_0 * sus
        
        self.prism.magnetization = M
        mag_vec = self.prism.magnetic_flux_density(self._obs)
        ba = mag_vec.dot(b0) / strength
        return mag_vec, ba

    def _update_image(self):
        B, ba = self._do_forward()
        if self._comp_toggle.value == 'East':
            b = B[..., 0]
            label = r"$B_{east}$ (nT)"
        elif self._comp_toggle.value == 'North':
            b = B[..., 1]
            label = r"$B_{north}$ (nT)"
        elif self._comp_toggle.value == 'Down':
            b = -B[..., 2]
            label = r"$B_{down}$ (nT)"
        else:
            b = ba
            label = r"Total Field Anomaly (nT)"
            
        self._im_ba.update({
            'array': b.ravel(),
            'clim':[b.min(), b.max()],
        })
        self._cb_image.set_label(label)

    def display(self):
        return self._box