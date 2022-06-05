import sys
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except:
    print('odak.visualize.plotly requires plotly and kaleido to be installed: pip install plotly and kaleido')
import numpy as np
from ..wave import calculate_phase, calculate_amplitude, calculate_intensity


class surfaceshow():
    """
    A class for general purpose surface plotting using plotly.
    """

    def __init__(self, title='plot', shape=[1000, 1000], labels=['x', 'y', 'z'], types=['linear', 'linear', 'linear'], font_size=12, tick_no=[10, 10, 10], margin=[65, 50, 65, 90], camera=[1.87, 0.88, -0.64], colorbarlength=0.75):
        """
        Class for plotting detectors.

        Parameters
        ---------- 
        title          : str
                         Title of the plot.
        shape          : list
                         Resolution of the figure to be generated.
        labels         : list
                         Labels for x and y axes.
        types          : list
                         Types of axes, it can be `linear`, `log`, `data`, or `category`. For more see ploty.layout.scene.xaxis.type.
        font_size      : int
                         Font size.
        tick_no        : list
                         Number of ticks along each axis.
        margin         : list
                         Margins in plotting.
        camera         : list
                         Scene camera location along X,Y,Z axes.
        colorbarlength : float
                         Length of the colorbar.
        """
        self.settings = {
            'title': title,
            'color scale': 'Portland',
                           'x label': labels[0],
                           'y label': labels[1],
                           'z label': labels[2],
                           'x axis type': types[0],
                           'y axis type': types[1],
                           'z axis type': types[2],
                           'font size': font_size,
                           'x axis tick number': tick_no[0],
                           'y axis tick number': tick_no[1],
                           'z axis tick number': tick_no[2],
                           'width': shape[0],
                           'height': shape[1],
                           'margin': margin,
                           'camera': camera,
                           'colorbar length': colorbarlength,
        }
        self.fig = make_subplots(
            rows=1,
            cols=1,
            specs=[
                [
                    {"type": "scene"},
                ],
            ],
        )

    def save_image(self, filename):
        """
        Definition to save the figure.

        Parameters
        ----------
        filename    : str
                      Filename.
        """
        self.fig.write_image(filename)

    def show(self):
        """
        Definition to show the plot.
        """
        self.fig.update_layout(
            autosize=False,
            width=self.settings['width'],
            height=self.settings['height'],
            scene=dict(
                aspectmode='manual',
                aspectratio=dict(x=1., y=1., z=1.),
                xaxis=dict(title=self.settings['x label'], type=self.settings['x axis type'],
                           nticks=self.settings['x axis tick number']),
                yaxis=dict(title=self.settings['y label'], type=self.settings['y axis type'],
                           nticks=self.settings['y axis tick number']),
                zaxis=dict(title=self.settings['z label'], type=self.settings['z axis type'],
                           nticks=self.settings['z axis tick number']),
            ),
            title=self.settings['title'],
            font=dict(
                size=self.settings['font size'],
            ),
            margin=dict(
                l=self.settings['margin'][0],
                r=self.settings['margin'][1],
                b=self.settings['margin'][2],
                t=self.settings['margin'][3]
            ),
            scene_camera_eye=dict(
                x=self.settings['camera'][0],
                y=self.settings['camera'][1],
                z=self.settings['camera'][2]
            ),
        )
        self.fig.show()

    def add_surface(self, data_x, data_y, data_z, label='', mode='lines+markers', opacity=1., contour=False):
        """
        Definition to add data to the plot.

        Parameters
        ----------
        data_x      : ndarray
                      X axis data to be plotted.
        data_y      : ndarray
                      Y axis data to be plotted.
        label       : str
                      Label of the plot.  
        mode        : str
                      Mode for the plot, it can be either lines+markers, lines or markers.
        opacity     : float
                      Opacity of the plot. The value must be between one to zero. Zero is fully trasnparent, while one is opaque.
        """
        self.fig.add_trace(
            go.Surface(
                x=data_x,
                y=data_y,
                z=data_z,
                surfacecolor=data_z,
                colorscale=self.settings['color scale'],
                opacity=opacity,
                colorbar=dict(len=self.settings['colorbar length']),
                contours={
                    'z': {
                        'show': contour,
                    },
                }
            ),
            row=1,
            col=1,
        )


class plotshow():
    """
    A class for general purpose 1D plotting using plotly.
    """

    def __init__(self, subplot_titles=['plot'], shape=[1000, 1000], font_size=16, labels=['x', 'y'], margin=[65, 50, 65, 90], colorbarlength=0.75, rows=1, cols=1):
        """
        Class for plotting detectors.

        Parameters
        ----------
        subplot_titles : list
                         Titles of plots.
                         Resolution of the figure to be generated.
        font_size      : int
                         Font size.                       
        labels         : list
                         Labels for x and y axes.
        margin         : list
                         Margins in plotting.
        colorbarlength : float
                         Length of the colorbar.                         
        """
        self.settings = {
            'subplot titles': subplot_titles,
            'color scale': 'Portland',
                           'font size': font_size,
                           'x label': labels[0],
                           'y label': labels[1],
                           'margin': margin,
                           'colorbar length': colorbarlength,
                           'width': shape[0],
                           'height': shape[1],
        }
        specs = []
        new_row = []
        for row in range(rows):
            new_col = []
            for col in range(cols):
                new_col.append({"type": "xy"})
            specs.append(new_col)
        self.fig = make_subplots(
            rows=rows,
            cols=cols,
            specs=specs,
            subplot_titles=subplot_titles
        )

    def show(self):
        """
        Definition to show the plot.
        """
        self.fig.layout = go.Layout(
            xaxis=dict(title=self.settings['x label']),
            yaxis=dict(title=self.settings['y label'])
        )
        self.fig.update_layout(
            autosize=False,
            width=self.settings['width'],
            height=self.settings['height'],
            scene=dict(
                aspectmode='manual',
                aspectratio=dict(x=1., y=1., z=1.),
            ),
            font=dict(
                size=self.settings['font size'],
            ),
            margin=dict(
                l=self.settings['margin'][0],
                r=self.settings['margin'][1],
                b=self.settings['margin'][2],
                t=self.settings['margin'][3]
            ),

        )
        self.fig.show()

    def save_image(self, filename):
        """
        Definition to save the figure.

        Parameters
        ----------
        filename    : str
                      Filename.
        """
        self.fig.write_image(filename)

    def add_plot(self, data_x, data_y=None, label='', mode='lines+markers', row=1, col=1):
        """
        Definition to add data to the plot.

        Parameters
        ----------
        data_x      : ndarray
                      X axis data to be plotted.
        data_y      : ndarray
                      Y axis data to be plotted.
        label       : str
                      Label of the plot.  
        mode        : str
                      Mode for the plot, it can be either lines+markers, lines or markers.
        """
        if type(data_y) == type(None):
            data_y = np.arange(0, data_x.shape[0])
        self.fig.add_trace(
            go.Scatter(
                x=data_x,
                y=data_y,
                mode=mode,
                name=label,
            ),
            row=row,
            col=col
        )


class detectorshow():
    """
    A class for visualizing detectors using plotly.
    """

    def __init__(self, row_titles=['Field 1'], subplot_titles=['Amplitude', 'Phase', 'Intensity'], title='detector', rows=1, cols=1, show_intensity=False, show_amplitude=True, show_phase=True, shape=[1000, 1000], margin=[65, 50, 65, 90]):
        """
        Class for plotting detectors.

        Parameters
        ----------
        subplot_titles : list
                         Titles of plots.
        title          : str
                         Title of plots.
        rows           : int
                         Number of rows/fields.
        show_intensity : bool
                         Flag to show intensity.
        show_amplitude : bool
                         Flag to show amplitude.
        show_phase     : bool
                         Flag to show phase.
        """
        m = 0
        if show_intensity == True:
            m += 1
        if show_amplitude == True:
            m += 1
        if show_phase == True:
            m += 1

        self.settings = {
            'title': title,
            'subplot titles': subplot_titles,
            'row titles': row_titles,
            'color scale': 'Portland',
            'sub column no': m,
            'column number': cols*m,
            'row number': rows,
            'show amplitude': show_amplitude,
            'show phase': show_phase,
            'show intensity': show_intensity,
            'width': shape[0],
            'height': shape[1],
            'margin': margin
        }
        specs = []
        for i in range(0, self.settings["row number"]):
            new_row = []
            for j in range(0, self.settings["column number"]):
                new_row.append({"type": "heatmap"})
            specs.append(new_row)
        self.fig = make_subplots(
            rows=self.settings["row number"],
            cols=self.settings["column number"],
            specs=specs,
            subplot_titles=self.settings["subplot titles"],
            row_titles=self.settings["row titles"]
        )

    def show(self):
        """
        Definition to show the plot.
        """
        self.fig.update_layout(
            autosize=True,
            width=self.settings['width'],
            height=self.settings['height'],
            scene=dict(
                aspectmode='manual',
                aspectratio=dict(x=1., y=1., z=1.),
            ),
            margin=dict(
                l=self.settings['margin'][0],
                r=self.settings['margin'][1],
                b=self.settings['margin'][2],
                t=self.settings['margin'][3]
            ),
        )
        self.fig.show()

    def save_image(self, filename):
        """
        Definition to save the figure. Always first show then save.

        Parameters
        ----------
        filename    : str
                      Filename.
        """
        self.fig.write_image(filename)

    def add_field(self, field, row=1, col=1, showscale=False):
        """
        Definition to add a point to the figure.

        Parameters
        ----------
        field          : ndarray
                         Field to be displayed.
        row            : int
                         Row number.
        col            : int
                         Column number.
        showscale      : bool
                         Set True to show color bar.
        """
        amplitude = calculate_amplitude(field)
        phase = calculate_phase(field, deg=True)
        intensity = calculate_intensity(field)
        col = (col-1)*(self.settings["sub column no"])+1
        if self.settings["show amplitude"] == True:
            self.fig.add_trace(
                go.Heatmap(
                    z=amplitude,
                    colorscale=self.settings['color scale'],
                    showscale=showscale
                ),
                row=row,
                col=col
            )
            col += 1

        if self.settings["show phase"] == True:
            self.fig.add_trace(
                go.Heatmap(
                    z=phase,
                    colorscale=self.settings['color scale'],
                    showscale=showscale
                ),
                row=row,
                col=col
            )
            col += 1

        if self.settings["show intensity"] == True:
            self.fig.add_trace(
                go.Heatmap(
                    z=intensity,
                    colorscale=self.settings['color scale'],
                    showscale=showscale
                ),
                row=row,
                col=col
            )
            col += 1


class rayshow():
    """
    A class for visualizing rays using plotly.
    """

    def __init__(self, rows=1, columns=1, subplot_titles=["Ray visualization"], opacity=0.5, line_width=1., marker_size=1., color_scale='Inferno'):
        """
        Class for plotting rays.

        Parameters
        ----------
        rows           : int
                         Number of rows.
        columns        : int
                         Number of columns.
        subplot_titles : list
                         Titles of plots.
        opacity        : float
                         Opacity of the markers or lines.
        line_width     : float
                         Line width of the lines.
        marker_size    : float
                         Marker size of the markers.
        color_scale    : str
                         Color scale to be used. 
        """
        self.settings = {
            'rows': rows,
            'color scale': color_scale,
            'columns': columns,
            'subplot titles': subplot_titles,
            'opacity': opacity,
            'line width': line_width,
            'marker size': marker_size
        }
        specs = []
        for i in range(0, columns):
            new_row = []
            for j in range(0, rows):
                new_row.append({"type": "scene"},)
            specs.append(new_row)
        self.fig = make_subplots(
            rows=self.settings['rows'],
            cols=self.settings['columns'],
            subplot_titles=self.settings['subplot titles'],
            specs=specs
        )

    def show(self):
        """
        Definition to show the plot.
        """
        self.fig.update_layout(
            scene=dict(
                aspectmode='manual',
                aspectratio=dict(x=1., y=1., z=1.),
            ),
        )
        self.fig.show()

    def add_point(self, point, row=1, column=1, color='red'):
        """
        Definition to add a point to the figure.

        Parameters
        ----------
        point          : ndarray
                         Point(s).
        row            : int
                         Row number of the figure.
        column         : int
                         Column number of the figure.

        """
        self.fig.add_trace(
            go.Scatter3d(
                x=point[:, 0].flatten(),
                y=point[:, 1].flatten(),
                z=point[:, 2].flatten(),
                mode='markers',
                marker=dict(
                    size=self.settings["marker size"],
                    color=color,
                    opacity=self.settings["opacity"]
                ),
            ),
            row=row,
            col=column
        )

    def add_line(self, point_start, point_end, row=1, column=1, color='red'):
        """
        Definition to add a ray to the figure.

        Parameters
        ----------
        point_start    : ndarray
                         Starting point(s).
        point_end      : ndarray
                         Ending point(s).
        row            : int
                         Row number of the figure.
        column         : int
                         Column number of the figure.
        color          : str
                         Color of the lune to be drawn.
        """
        if len(point_start.shape) == 1:
            point_start = point_start.reshape((1, 3))
        if len(point_end.shape) == 1:
            point_end = point_end.reshape((1, 3))
        if point_start.shape != point_end.shape:
            print('Size mismatch in line plot. Sizes are {} and {}.'.format(
                point_start.shape, point_end.shape))
            sys.exit()
        for point_id in range(0, point_start.shape[0]):
            points = np.array(
                [
                    point_start[point_id],
                    point_end[point_id]
                ]
            )
            points = points.reshape((2, 3))
            self.fig.add_trace(
                go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode='lines',
                    line=dict(
                        width=self.settings["line width"],
                        color=color,
                    ),
                    opacity=self.settings["opacity"]
                ),
                row=row,
                col=column
            )

    def add_surface(self, data_x, data_y, data_z, surface_color, row=1, column=1, label='', mode='lines+markers', opacity=1., contour=False):
        """
        Definition to add data to the plot.

        Parameters
        ----------
        data_x        : ndarray
                        X axis data to be plotted.
        data_y        : ndarray
                        Y axis data to be plotted.
        data_z        : ndarray
                        Z axis data to be plotted.
        surface_color : ndarray
                        Colors of the surface.
        label         : str
                        Label of the plot.  
        mode          : str
                        Mode for the plot, it can be either lines+markers, lines or markers.
        opacity       : float
                        Opacity of the plot. The value must be between one to zero. Zero is fully trasnparent, while one is opaque.
        """
        self.fig.add_trace(
            go.Surface(
                x=data_x,
                y=data_y,
                z=data_z,
                surfacecolor=surface_color,
                colorscale=self.settings['color scale'],
                opacity=opacity,
                contours={
                    'z': {
                        'show': contour,
                    },
                }
            ),
            row=row,
            col=column,
        )
