import ipywidgets as widgets
from ipycanvas import MultiCanvas, hold_canvas, Canvas
import math, random
import ipyevents
import numpy as np
import json
from IPython.display import Javascript, display
import base64

def rand_color():
    return f"rgb({random.randint(80,220)}, {random.randint(80,220)}, {random.randint(80,220)})"

def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def point_in_poly(x, y, poly):
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i+1)%n]
        if ((y1>y) != (y2>y)):
            xinters = (y-y1)*(x2-x1)/(y2-y1+1e-12) + x1
            if x < xinters:
                inside = not inside
    return inside

def dist_to_seg(px, py, ax, ay, bx, by):
    abx, aby = bx-ax, by-ay
    apx, apy = px-ax, py-ay
    ab2 = abx*abx + aby*aby
    if ab2 == 0:
        return math.hypot(px-ax, py-ay), 0
    t = max(0, min(1, (apx*abx + apy*aby) / ab2))
    cx, cy = ax + t*abx, ay + t*aby
    return math.hypot(px-cx, py-cy), t

def nearest_edge(poly, x, y, thresh):
    best = (None, None, None)
    best_d = float("inf")
    for i in range(len(poly)):
        ax, ay = poly[i]
        bx, by = poly[(i+1)%len(poly)]
        d, t = dist_to_seg(x, y, ax, ay, bx, by)
        if d < best_d:
            best_d = d
            best = (i, d, t)
    return best if best_d <= thresh else (None, None, None)

class PolygonEditor(widgets.VBox):
    vertex_radius = 6
    click_thresh = 8
    edge_thresh = 8

    monitor = widgets.Output()

    def __init__(self, width=500, height=400, points=None, layout=None):
        """
        Polygon editor with multiple modes.
        
        Parameters
        ----------
        width, height : int
            Canvas size.
        """
        super().__init__(layout=layout)
        
        # canvas layers: 0=background, 1=polygons
        self.canvas = MultiCanvas(
            2,
            width=width,
            height=height,
            layout=widgets.Layout(
                border='1px solid #ccc'
            )
        )
        self.canvas[0].stroke_style = "black"
        # self.canvas[1].fill_style = "rgba(0, 150, 255, 0.3)"

        self.status = widgets.HTML()

        self.width = width
        self.height = height
        
        self.polygons = []       # list of polygons
        self.colors = []
        self.active_poly = []  # currently drawing polygon
        self.dragging_vertex = None     # (poly_idx, vert_idx) if dragging a vertex
        self.dragging_polygon = None
        
        # mouse events
        self.canvas.on_mouse_down(self._on_mouse_down)
        self.canvas.on_mouse_move(self._on_mouse_move)
        self.canvas.on_mouse_up(self._on_mouse_up)
        self.canvas.on_mouse_out(self._on_mouse_leave)
        
        # mode selector
        self.mode = widgets.ToggleButtons(
            options=[
                ('Pan', 'panning'),
                ('Select/Move Polygon', 'select'),
                ('Add Polygon', 'add_poly'),
                ('Add/Remove Vertices', 'edit_vertices'),
                ('Delete Polygon', 'delete')
            ]
        )
        self.mode.observe(self._set_mode, "value")
        self.mode.value = 'select'

        # View transform
        
        self.events = ipyevents.Event(
            source=self.canvas,
            watched_events=["wheel", "click"],
            prevent_default_action=True,
        )
        self.events.on_dom_event(self._handle_event)

        self.fit_btn = widgets.Button(description="Fit View")
        self.fit_btn.on_click(self.fit_view)

        self.save_btn = widgets.Button(description="Save Polygons", icon='download')
        self.save_btn.on_click(self.save_polygons)
        self.load_btn = widgets.FileUpload(
            description="Load Polygons",
            name="polygons.txt",
        )
        self.load_btn.observe(self._load_observe)

        if points is not None:
            points = np.atleast_2d(points)
        self._points = points

        # For panning
        self.panning = False
        self.last_mouse = None

        buttons = widgets.Box(
            [self.fit_btn, self.save_btn, self.load_btn],
            layout=widgets.Layout(
                display="flex",
                flex_flow="row wrap",   # horizontal, but wrap if needed
                justify_content="flex-end",  # align right
                width="100%"            # stretch container to cell width
            )
            )
        
        toolbar = widgets.HBox(
            [self.mode, buttons],
            layout=widgets.Layout(
                display='flex',
                justify_content='space-between',
                width='100%'
            )
        )
        self._poly_handles = set()
        
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0

        self.fit_view()

        self.location = widgets.HTML()

        info_box = widgets.HBox(
            [self.status, self.location],
            layout=widgets.Layout(
                display='flex',
                justify_content='space-between',
                width='100%'
            )
        )
        
        self.children = [self.canvas, toolbar, info_box, self.monitor]
        self._draw_grid()
        self._redraw()

    def on_poly_update(self, handle, remove=False):
        if not remove:
            self._poly_handles.add(handle)
        else:
            self._poly_handles.discard(handle)

    def _call_poly_handles(self):
        for handle in self._poly_handles:
            handle(self.polygons)

    def _draw_grid(self, spacing=64):
        """Draw grid in world coordinates that scales with zoom/pan."""
        ctx = self.canvas[0]
        with hold_canvas(ctx):
            ctx.clear()
            
            # Compute visible world bounds
            x0, y0 = self.to_world(0, self.canvas.height)
            x1, y1 = self.to_world(self.canvas.width, 0)

            spc = spacing
            sc = self.scale
            while sc >= 2:
                sc /= 2
                spc /= 2
            while sc <= 0.5:
                sc *= 2
                spc *= 2

            nx = int((x1 - x0)/spc)
            ny = int((y1 - y0)/spc)
            # xs = x0 + np.arange(nx + 1) * spc
            # ys = y0 + np.arange(ny + 1) * spc
            shift_x = (x0 // spc + 1) * spc - x0
            shift_y = (y0 // spc + 1) * spc - y0
            x0 += shift_x
            y0 += shift_y

            ctx.line_width = 1
            # Vertical lines
            for ix in range(nx + 1):
                x = x0 + ix * spc
                sx0, sy0 = self.to_screen(x, y0)
                sx1, sy1 = self.to_screen(x, y1)
                ctx.begin_path()
                ctx.move_to(sx0, 0)
                ctx.line_to(sx1, self.height)
                ctx.stroke()

            # Horizontal lines
            for iy in range(ny + 1):
                y = y0 + iy * spc
                sx0, sy0 = self.to_screen(x0, y)
                sx1, sy1 = self.to_screen(x1, y)
                ctx.begin_path()
                ctx.move_to(0, sy0)
                ctx.line_to(self.width, sy1)
                ctx.stroke()

            xax_s = self.to_screen(0, y0)[0]
            if xax_s > 0 and xax_s < self.width:
                ctx.line_width = 3
                ctx.begin_path()
                ctx.move_to(xax_s, 0)
                ctx.line_to(xax_s, self.height)
                ctx.stroke()

            yax_s = self.to_screen(x0, 0)[1]
            if yax_s > 0 and yax_s < self.height:
                ctx.line_width = 3
                ctx.begin_path()
                ctx.move_to(0, yax_s)
                ctx.line_to(self.width, yax_s)
                ctx.stroke()

            for pt in self._points:
                ctx.fill_circle(*self.to_screen(*pt), 10)


    def _set_status(self, txt):
        self.status.value = f"<b>Mode:</b> {self.mode.value} — {txt}"

    def to_world(self, x, y):
        """Convert screen -> world coords"""
        return ((x - self.width/2) / self.scale - self.offset_x,
                -((y - self.height/2) / self.scale - self.offset_y))

    def to_screen(self, x, y):
        """Convert world -> screen coords"""
        return ((x + self.offset_x) * self.scale + self.width/2,
                (-y + self.offset_y ) * self.scale + self.height/2)

    @monitor.capture()
    def fit_view(self, b=None):
        self.monitor.clear_output()
        mins = []
        maxs = []
        if self._points is not None:
            mins.append(np.min(self._points, axis=0))
            maxs.append(np.max(self._points, axis=0))
        if self.polygons:
            for ply in self.polygons:
                mins.append(np.min(ply, axis=0))
                maxs.append(np.max(ply, axis=0))

        reset = not mins and not maxs
        if not reset:
            mins = np.min(mins, axis=0)
            maxs = np.max(maxs, axis=0)
        reset |= np.all(mins == maxs)
        if reset:
            self.scale = 1.0
            self.offset_x = self.offset_y = 0
            self._draw_grid()
            self._redraw()
            return

        min_x, min_y = mins
        max_x, max_y = maxs

        w = (max_x - min_x) * 1.25
        h = (max_y - min_y) * 1.25

        self.scale = min(self.width / (w + 1E-8), self.height / (h + 1E-8))
        self.offset_x = -(min_x + max_x) / 2
        self.offset_y = (min_y + max_y) / 2
        self._draw_grid()
        self._redraw()

    # Save/Load
    # ---------
    @monitor.capture()
    def save_polygons(self, button, file_name="polygons.txt"):
        self.monitor.clear_output()
        with(self.monitor):
            polys = json.dumps({"polygons":self.polygons}).encode("utf-8")
            payload = base64.b64encode(polys).decode()
    
            js = f"""
            var link = document.createElement('a');
            link.href = 'data:text/plain;base64,{payload}';
            link.download = '{file_name}';
            link.click();
            """
            display(Javascript(js))
        self.monitor.clear_output()
    
    @monitor.capture()
    def _load_observe(self, change):
        self.monitor.clear_output()
        success = False
        if (val:=self.load_btn.value):
            data = val[0]['content'].tobytes().decode("utf-8")
            polys = json.loads(data)["polygons"]
            self.add_polygons(polys)
            success = True
            if not success:
                monitor.clear()
                raise IOError(f"Unable to read polygons from file {val[0]['name']}")
            if success:
                self.load_btn.value = tuple()
                

    def load_polygons(self, file_name=None):
        with open(file_name) as f:
            polys = json.load(file_name)["polygons"]
            self.add_polygons(polys)

    # --------------------
    # Mode Handling
    # --------------------
    @monitor.capture()
    def _set_mode(self, change):
        self.monitor.clear_output()
        if change['new'] == 'add_poly':
            self._set_status("Click to start polygon, click first vertex to close.")
        elif change['new'] == 'select':
            if self.active_poly: self._close_active()
            self._set_status("Drag vertices or polygons to move.")
        elif change['new'] == 'edit_vertices':
            if self.active_poly: self._close_active()
            self._set_status("Click vertex to delete, click edge to insert vertex.")
        elif change['new'] == 'delete':
            if self.active_poly: self._close_active()
            self._set_status("Click inside a polygon to delete it.")
        

    # --------------------
    # Event Handlers
    # --------------------
    @monitor.capture()
    def _on_mouse_down(self, x, y):
        self.monitor.clear_output()

        mode = self.mode.value
        
        if mode == 'panning':
            self.panning = True
            self.last_mouse = (x, y)
            return

        if mode == "select":
            self._handle_drag_start(x, y)
        elif mode == "add_poly":
            self._handle_add(x, y)
        elif mode == "edit_vertices":
            self._handle_edit(x, y)
        elif mode == "delete":
            self._handle_delete(x, y)

    def _on_mouse_move(self, x, y):
        wx, wy = self.to_world(x, y)
        self.location.value = f"Cursor Location: {wx:.4f}, {wy:.4f}"
        if self.panning and self.last_mouse:
            dx = x - self.last_mouse[0]
            dy = y - self.last_mouse[1]
            soffx, soffy = self.to_screen(self.offset_x, self.offset_y)
            soffx += dx
            soffy -= dy
            offx, offy = self.to_world(soffx, soffy)
            self.offset_x = offx
            self.offset_y = offy
            self.last_mouse = (x, y)
            self._draw_grid()
            self._redraw()
            return

        x, y = self.to_world(x, y)
        mode = self.mode.value
        if mode == "select":
            dragging_vertex = self.dragging_vertex
            dragging_polygon = self.dragging_polygon
            if dragging_vertex:
                pi, vi = dragging_vertex
                self.polygons[pi][vi] = (x, y)
                self._redraw()
            elif dragging_polygon:
                pi, ox, oy, opoly = dragging_polygon
                dx, dy = x - ox, y - oy
                self.polygons[pi] = [(vx + dx, vy + dy) for vx, vy in opoly]
                self._redraw()

    def _on_mouse_up(self, x, y):
        self.panning = False
        self.last_mouse = None
        self.dragging_vertex = None
        self.dragging_polygon = None

    def _on_mouse_leave(self, x, y):
        self.location.value = ""
        self._on_mouse_up(x, y)

    @monitor.capture()
    def _handle_event(self, event):
        if event["type"] == "wheel":
            self.monitor.clear_output()
            # Zoom around mouse location
            mx = (self.canvas.width * event["relativeX"]) / event['boundingRectWidth']
            my = (self.canvas.height * event["relativeY"]) / event['boundingRectHeight']
            old = self.to_world(mx, my)
            factor = 10/9 if event["deltaY"] < 0 else 9/10
            self.scale *= factor
            new = self.to_world(mx, my)
            # need to shift offset such that the new_x and new_y are the same as
            # mx and my
            # wx = x / self.scale - self.offset_x

            self.offset_x += new[0] - old[0]
            self.offset_y -= new[1] - old[1]
            self._draw_grid()
            self._redraw()

    @monitor.capture()
    def _close_active(self):
        self.monitor.clear_output()
        if len(self.active_poly) > 2:
            self.polygons.append(self.active_poly[:])
            self.colors.append(rand_color())
        self.active_poly = []
        self._set_status("Polygon closed, switched to Select/Move mode.")
        self.mode.value = 'select'
        self._redraw()

    @monitor.capture()
    def _on_mouse_up(self, x, y):
        self.monitor.clear_output()
        self.panning = False
        self.last_mouse = None
        self.dragging_vertex = None
        self.dragging_polygon = None

    # --------------------
    # Mode-specific
    # --------------------

    def _handle_add(self, x, y):
        x, y = self.to_world(x, y)
        current = self.active_poly
        if current and len(current) > 2 and dist((x, y), current[0]) <= self.click_thresh / self.scale:
            self._close_active()
        else:
            current.append((x, y))
        self._redraw()


    def _handle_drag_start(self, x, y):
        wx, wy = self.to_world(x, y)
        pi, vi = self._find_vertex(wx, wy)
        if vi is not None:
            self.dragging_vertex = (pi, vi)
        else:
            pi = self._find_polygon(wx, wy)
            if pi is not None:
                self.dragging_polygon = (pi, wx, wy, [(vx, vy) for vx, vy in self.polygons[pi]])
            else:
                self.panning = True
                self.last_mouse = (x, y)


    def _handle_edit(self, x, y):
        x, y = self.to_world(x, y)

        polygons = self.polygons
        pi, vi = self._find_vertex(x, y)
        if vi is not None:
            if len(polygons[pi]) > 3:
                polygons[pi].pop(vi)
                self._set_status(f"Deleted vertex from polygon {pi}.")
        else:
            pi = self._find_polygon(x, y)
            if pi is not None:
                ei, d, t = nearest_edge(polygons[pi], x, y, self.edge_thresh / self.scale)
                if ei is not None:
                    ax, ay = polygons[pi][ei]
                    bx, by = polygons[pi][(ei+1)%len(polygons[pi])]
                    nx, ny = ax + t*(bx-ax), ay + t*(by-ay)
                    polygons[pi].insert(ei+1, (nx, ny))
                    self._set_status(f"Inserted vertex into polygon {pi}.")
        self._redraw()


    def _handle_delete(self, x, y):
        x, y = self.to_world(x, y)
        pi = self._find_polygon(x, y)
        if pi is not None:
            del self.polygons[pi]
            del self.colors[pi]
            self._set_status(f"Deleted polygon {pi}.")
        self._redraw()

    # --------------------
    # Drawing
    # --------------------


    def _redraw(self):
        canvas = self.canvas[1]
        polygons = self.polygons
        colors = self.colors
        current = self.active_poly
        
        with hold_canvas(canvas):
            canvas.clear()
            for poly, col in zip(polygons, colors):
                if len(poly) >= 3:
                    canvas.fill_style = col
                    canvas.begin_path()
                    canvas.move_to(*self.to_screen(*poly[0]))
                    for p in poly[1:]:
                        canvas.line_to(*self.to_screen(*p))
                    canvas.close_path()
                    canvas.fill()
                    canvas.stroke()
                for (x, y) in poly:
                    x, y = self.to_screen(x, y)
                    canvas.fill_style = "white"
                    canvas.fill_circle(x, y, self.vertex_radius)
                    canvas.stroke_circle(x, y, self.vertex_radius)
    
            if current:
                canvas.stroke_style = "blue"
                canvas.begin_path()
                canvas.move_to(*self.to_screen(*current[0]))
                for p in current[1:]:
                    canvas.line_to(*self.to_screen(*p))
                canvas.stroke()
                for (x, y) in current:
                    x, y = self.to_screen(x, y)
                    canvas.fill_style = "blue"
                    canvas.fill_circle(x, y, self.vertex_radius)
            else:
                self._call_poly_handles()

    # --------------------
    # Geometry utils
    # --------------------

    def _find_vertex(self, x, y):
        # Reverse the search because of the order they are drawn.
        for pi in range(len(self.polygons)-1, -1, -1):
            poly = self.polygons[pi]
            for vi, (vx, vy) in enumerate(poly):
                if dist((x, y), (vx, vy)) <= self.click_thresh / self.scale:
                    return pi, vi
        return None, None

    def _find_polygon(self, x, y):
        # Reverse the search because of the order they are drawn.
        for pi in range(len(self.polygons)-1, -1, -1):
            poly = self.polygons[pi]
            if point_in_poly(x, y, poly):
                return pi
        return None

    # --------------------
    # Public API
    # --------------------
    def clear(self):
        self.polygons = []
        self._redraw()

    def get_polygons(self):
        return self.polygons

    def add_polygons(self, polys):
        polys = np.atleast_2d(polys)
        if polys.ndim == 2:
            polys = polys[None, ...]
        for poly in polys:
            self.polygons.append(poly.tolist())
            self.colors.append(rand_color())
        self._redraw()
        self.mode.value = "select"
