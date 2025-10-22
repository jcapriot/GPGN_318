import numpy as np
from ipycanvas import Canvas, hold_canvas, MultiCanvas
from ipywidgets import VBox, HBox, Button, Label, Output, Layout

class ConductivityModelCanvas(VBox):

    monitor = Output()

    def __init__(self, thicknesses, conductivities, width=600, height=400, layout=None):

        if layout is None:
            layout = {}
        super().__init__(layout=layout)

        self.thicknesses = list(thicknesses)
        self.conductivities = list(conductivities)
        if len(self.thicknesses) != len(self.conductivities) - 1:
            raise ValueError(
                "Thicknesses array must have one less element than the conductivity array"
            )
        self.width = width
        self.height = height
        self.zoom_factor = 1.0
        self.selected_line = None
        self.drag_start = None

        self.canvas = MultiCanvas(
            2,
            width=width,
            height=height,
            layout=Layout(
                border='1px solid #ccc'
            )
        )
        self.zoom_in_btn = Button(icon='search-plus', tooltip="Zoom In")
        self.zoom_out_btn = Button(icon='search-minus', tooltip="Zoom Out")
        self.autoscale_btn = Button(icon='arrows-alt', tooltip='Autoscale')
        self.add_btn = Button(description="+", tooltip="Add Layer")
        self.remove_btn = Button(description="-", tooltip="Remove Layer")

        self.children = [
            HBox([self.zoom_in_btn, self.zoom_out_btn,self.autoscale_btn,
                  self.add_btn, self.remove_btn]),
            self.canvas,
            self.monitor,
        ]

        # Hook up handlers
        self.zoom_in_btn.on_click(lambda _: self._zoom(0.8))
        self.zoom_out_btn.on_click(lambda _: self._zoom(1.25))
        self.add_btn.on_click(lambda _: self.add_layer())
        self.remove_btn.on_click(lambda _: self.remove_layer())
        self.autoscale_btn.on_click(lambda _: self.autoscale())

        self.canvas.on_mouse_down(self._handle_mousedown)
        self.canvas.on_mouse_move(self._handle_mousemove)
        self.canvas.on_mouse_up(self._handle_mouseup)

        self._model_update_handles = set()

        self._min_depth = 1E-2
        self._zoom(self.zoom_factor)

        # --- Coordinate transforms ---
    def _depths(self):
        return np.cumsum(self.thicknesses)

    def _depth_to_y(self, depth):
        log_min, log_max = np.log10(self._min_depth), self._max_log_depth + np.log10(self.zoom_factor)
        log_y = np.log10(depth + self._min_depth)
        return (log_y - log_min) / (log_max - log_min) * self.height

    def _y_to_depth(self, y):
        log_min, log_max = np.log10(self._min_depth), self._max_log_depth + np.log10(self.zoom_factor)
        log_y = (y  / self.height) * (log_max - log_min) + log_min
        return 10**log_y - self._min_depth

    def _cond_to_x(self, cond):
        log_min, log_max = self._log_cond_bounds
        log_max += np.log10(self.zoom_factor)
        return (np.log10(cond) - log_min) / (log_max - log_min) * self.width

    def _x_to_cond(self, x):
        log_min, log_max = self._log_cond_bounds
        log_max += np.log10(self.zoom_factor)
        log_c = (x / self.width) * (log_max - log_min) + log_min
        return 10 ** log_c
    
    @property
    def _text_size(self):
        return int(self.height / 400 * 9)
    
    @property
    def _line_width(self):
        return int(self.height / 400 * 2)
    
    @monitor.capture()
    def _draw_grid(self):
        ctx = self.canvas[0]
        with hold_canvas(ctx):
            ctx.clear()

            # Grid (log x, linear y)
            ctx.font = f"{self._text_size}px sans-serif"
            ctx.line_width = self._line_width
            log_c_min, log_c_max = self._log_cond_bounds
            for lx in range(int(np.floor(log_c_min)), int(np.ceil(log_c_max))):
                x = self._cond_to_x(10**lx)
                ctx.stroke_style = "#ccc"
                ctx.begin_path()
                ctx.move_to(x, 0)
                ctx.line_to(x, self.height)
                ctx.stroke()
                ctx.fill_text(f"1e{lx} S/m", x + 2, self._text_size)

            log_d_min = np.log10(self._min_depth)
            log_d_max = self._max_log_depth
            #for ly in np.linspace(0, self._max_depth, 6):
            for ly in range(int(np.floor(log_d_min)), int(np.ceil(log_d_max))):
                y = self._depth_to_y(10**ly)
                ctx.stroke_style = "#ccc"
                ctx.begin_path()
                ctx.move_to(0, y)
                ctx.line_to(self.width, y)
                ctx.stroke()
                ctx.fill_text(f"1e{ly} m", 4, y - 2)

    # --- Drawing ---
    @monitor.capture()
    def _draw_model(self):
        ctx = self.canvas[1]
        with hold_canvas(self.canvas):
            ctx.clear()
            ctx.line_width = self._line_width

            # Horizontal (layer) boundaries
            top = 0.0
            y0 = self._depth_to_y(0)
            x0 = self._cond_to_x(self.conductivities[0])
            
            ctx.stroke_style = "#000"
            ctx.begin_path()
            ctx.move_to(x0, y0)
            for thk, cond in zip(self.thicknesses, self.conductivities[1:]):
                # take step down in thickness
                top += thk
                y1 = self._depth_to_y(top)
                ctx.line_to(x0, y1)
                
                # take step over in conductivity
                x1 = self._cond_to_x(cond)
                ctx.line_to(x1, y1)

                # cycle them.
                x0, y0 = x1, y1
            # last step goes all the way down!
            ctx.line_to(x0, self.height)
            ctx.stroke()
        self._call_model_update_handles()

    def _redraw(self):
        self._draw_grid()
        self._draw_model()

    def on_model_update(self, handle, remove=False):
        if not remove:
            self._model_update_handles.add(handle)
        else:
            self._model_update_handles.discard(handle)

    def _call_model_update_handles(self):
        for handle in self._model_update_handles:
            handle(self.thicknesses, self.conductivities)

    # --- Interaction ---
    @monitor.capture()
    def _handle_mousedown(self, x, y):
        tol = (self.height / 400 ) * 5
        xs = self._cond_to_x(self.conductivities)
        ys = self._depth_to_y(self._depths())
        y0 = 0
        x0 = xs[0]
        for i, (x1, y1) in enumerate(zip(xs[1:], ys)):
            x_span = [x0, x1]
            x_span.sort()
            within_y = y0 < y < y1
            within_x = x_span[0] < x < x_span[1]
            if within_x and abs(y - y1) < tol:
                self.selected_line = ('h', i)
                return
            if within_y and abs(x - x0) < tol:
                self.selected_line = ('v', i)
                return
            x0, y0 = x1, y1
        within_y = y0 < y < np.inf
        if within_y and abs(x - x0) < tol:
            self.selected_line = ('v', -1)
            return
            
    @monitor.capture()
    def _handle_mousemove(self, x, y):
        if not self.selected_line:
            return
        kind, idx = self.selected_line
        if kind == 'h':
            depths = self._depths()
            new_depth = self._y_to_depth(y)
            if idx - 1 < 0:
                depth_above = 0.0
            else:
                depth_above = depths[idx - 1]
            if idx + 1 < len(depths):
                depth_below = depths[idx + 1]
            else:
                depth_below = np.inf
            if depth_above <= new_depth <= depth_below:
                self.thicknesses[idx] = new_depth - depth_above
                if idx + 1 < len(self.thicknesses):
                    self.thicknesses[idx+1] = depth_below - new_depth
        elif kind == 'v':
            new_cond = self._x_to_cond(x)
            log_c_min, log_c_max = self._log_cond_bounds

            self.conductivities[idx] = np.clip(new_cond, 10**log_c_min, 10**log_c_max)
        self._draw_model()

    def _handle_mouseup(self, x, y):
        self.selected_line = None

    def autoscale(self):
        self.zoom_factor = 1.0
        self._zoom(1.0)

    # --- Zooming ---
    @monitor.capture()
    def _zoom(self, factor):
        self.zoom_factor = np.clip(self.zoom_factor * factor, 0.5, 5.0)
        depths = self._depths()
        if len(depths) > 0:
            self._max_depth = self._depths()[-1] * 10
        else:
            self._max_depth = 100
        
        self._max_log_depth = np.log10(self._max_depth)
        log_min_c = np.log10(np.min(self.conductivities))
        log_max_c = np.log10(np.max(self.conductivities))
        self._log_cond_bounds = (log_min_c-1, log_max_c + 1)
        self._redraw()

    # --- Layer management ---
    def add_layer(self, thickness=10, conductivity=1):
        self.thicknesses.append(thickness)
        self.conductivities.append(conductivity)
        self._draw_model()
        self.autoscale()

    def remove_layer(self):
        if len(self.thicknesses) > 0:
            self.thicknesses.pop()
            self.conductivities.pop()
            self._draw_model()

    # --- Utility accessors ---
    def get_model(self):
        """Return (thicknesses, conductivities)"""
        return self.thicknesses, self.conductivities
    
    def set_model(self, thicknesses, conductivities):
        thicks = list(thicknesses)
        conds = list(conductivities)
        if len(thicks) != len(conds) - 1:
            raise ValueError(
                "`thicknesses` array must have one less element than the `conductivity` array"
            )
        self.thicknesses = thicks
        self.conductivities = conds
        self.autoscale()

