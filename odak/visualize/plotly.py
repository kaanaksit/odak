import plotly.graph_objects as go
import sys
from plotly.subplots import make_subplots
from odak import np
from odak.wave.parameters import calculate_phase,calculate_amplitude,calculate_intensity


class detectorshow():
    """
    A class for visualizing detectors using plotly.
    """
    def __init__(self,subplot_titles=['Amplitude','Phase','Intensity'],title='detector'):
        """
        Class for plotting detectors.

        Parameters
        ----------
        subplot_titles : list
                         Titles of plots.
        """
        self.settings   = {
                           'title'          : title,
                           'subplot titles' : subplot_titles,
                           'color scale'    : 'Portland'
                          }
        self.fig = make_subplots(
                                 rows=1,
                                 cols=3,
                                 specs=[
                                        [
                                         {"type": "xy"},
                                         {"type": "xy"},
                                         {"type": "xy"}
                                        ],
                                       ],
                                 subplot_titles=subplot_titles
                                )
    def show(self):
        """
        Definition to show the plot.
        """
        self.fig.update_layout(
                               scene = dict(
                                            aspectmode  = 'manual',
                                            aspectratio = dict(x=1.,y=1.,z=1.),
                                           ),
                              )
        self.fig.show()

    def add_field(self,field):
        """
        Definition to add a point to the figure.

        Parameters
        ----------
        field          : ndarray
                         Field to be displayed.
        """
        amplitude = calculate_amplitude(field)
        phase     = calculate_phase(field,deg=True)
        intensity = calculate_intensity(field)
        if np.__name__ == 'cupy':
            amplitude = np.asnumpy(amplitude)
            phase     = np.asnumpy(phase)
            intensity = np.asnumpy(intensity)
        self.fig.add_trace(
                           go.Heatmap(
                                      z=amplitude,
                                      colorscale=self.settings['color scale']
                                     ),
                           row=1,
                           col=1
                          )

        self.fig.add_trace(
                           go.Heatmap(
                                      z=phase,
                                      colorscale=self.settings['color scale']
                                     ),
                           row=1,
                           col=2
                          )

        self.fig.add_trace(
                           go.Heatmap(
                                      z=intensity,
                                      colorscale=self.settings['color scale']
                                     ),
                           row=1,
                           col=3
                          )


class rayshow():
    """
    A class for visualizing rays using plotly.
    """
    def __init__(self,rows=1,columns=1,subplot_titles=["Ray visualization"],color='red',opacity=0.5,line_width=1.,marker_size=1.):
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
        color          : str
                         Color.
        opacity        : float
                         Opacity of the markers or lines.
        line_width     : float
                         Line width of the lines.
        marker_size    : float
                         Marker size of the markers.
        """
        self.settings  = {
                          'rows'           : rows,
                          'columns'        : columns,
                          'subplot titles' : subplot_titles,
                          'color'          : color,
                          'opacity'        : opacity,
                          'line width'     : line_width,
                          'marker size'    : marker_size
                         }
        specs          = []
        for i in range(0,columns):
            new_row = []
            for j in range(0,rows):
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
                               scene = dict(
                                            aspectmode  = 'manual',
                                            aspectratio = dict(x=1.,y=1.,z=1.),
                                           ),
                              )
        self.fig.show()

    def add_point(self,point,row=1,column=1):
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
        if np.__name__ == 'cupy':
            point = np.asnumpy(point)
        self.fig.add_trace(
                           go.Scatter3d(
                                        x=point[:,0].flatten(),
                                        y=point[:,1].flatten(),
                                        z=point[:,2].flatten(),
                                        mode='markers',
                                        marker=dict(
                                                    size=self.settings["marker size"],
                                                    color=self.settings["color"],
                                                    opacity=self.settings["opacity"]
                                                   ),
                                       ),
                           row=row,
                           col=column
                          )

    def add_line(self,point_start,point_end,row=1,column=1):
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
        """
        if np.__name__ == 'cupy':
            point_start = np.asnumpy(point_start)
            point_end   = np.asnumpy(point_end)
        if len(point_start.shape) == 1:
            point_start = point_start.reshape((1,3))
        if len(point_end.shape) == 1:
            point_end   = point_end.reshape((1,3))
        if point_start.shape != point_end.shape:
            print('Size mismatch in line plot. Sizes are {} and {}.'.format(point_start.shape,point_end.shape))
            sys.exit()
        for point_id in range(0,point_start.shape[0]):
            points = np.array(
                              [
                               point_start[point_id],
                               point_end[point_id]
                              ]
                             )
            points = points.reshape((2,3))
            if np.__name__ == 'cupy':
                points = np.asnumpy(points)
            self.fig.add_trace(
                               go.Scatter3d(
                                            x=points[:,0],
                                            y=points[:,1],
                                            z=points[:,2],
                                            mode='lines',
                                            line=dict(
                                                      width=self.settings["line width"],
                                                      color=self.settings["color"],
                                                     ),
                                            opacity=self.settings["opacity"]
                                           ),
                               row=row,
                               col=column
                              )

