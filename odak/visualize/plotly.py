import plotly.graph_objects as go
import sys
from plotly.subplots import make_subplots
from odak import np
from odak.wave.parameters import calculate_phase,calculate_amplitude,calculate_intensity

class surfaceshow():
    """
    A class for general purpose surface plotting using plotly.
    """
    def __init__(self,title='plot',labels=['x','y','z'],types=['linear','linear','linear'],font_size=12,tick_no=[10,10,10]):
        """
        Class for plotting detectors.

        Parameters
        ----------
        title         : str
                        Title of the plot.
        labels        : list
                        Labels for x and y axes.
        types         : list
                        Types of axes, it can be `linear`, `log`, `data`, or `category`. For more see ploty.layout.scene.xaxis.type.
        font_size     : int
                        Font size.
        tick_no       : list
                        Number of ticks along each axis.
        """
        self.settings   = {
                           'title'              : title,
                           'color scale'        : 'Portland',
                           'x label'            : labels[0],
                           'y label'            : labels[1],
                           'z label'            : labels[2],
                           'x axis type'        : types[0],
                           'y axis type'        : types[1],
                           'z axis type'        : types[2],
                           'font size'          : font_size,
                           'x axis tick number' : tick_no[0],
                           'y axis tick number' : tick_no[1],
                           'z axis tick number' : tick_no[2],
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
    def show(self):
        """
        Definition to show the plot.
        """
        self.fig.update_layout(
                               scene = dict(
                                            aspectmode  = 'manual',
                                            aspectratio = dict(x=1.,y=1.,z=1.),
                                            xaxis=dict(title=self.settings['x label'],type=self.settings['x axis type'],nticks=self.settings['x axis tick number']),
                                            yaxis=dict(title=self.settings['y label'],type=self.settings['y axis type'],nticks=self.settings['y axis tick number']),
                                            zaxis=dict(title=self.settings['z label'],type=self.settings['z axis type'],nticks=self.settings['z axis tick number']),
                                           ),
                               title = self.settings['title'],
                               font  = dict(
                                            size=self.settings['font size'],
                                           )
                              )
        self.fig.show()

    def add_surface(self,data_x,data_y,data_z,label='',mode='lines+markers',opacity=1.,contour=False):
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
                                      contours= {
                                                 'z':{
                                                      'show':contour,
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
    def __init__(self,subplot_titles=['plot'],labels=['x','y']):
        """
        Class for plotting detectors.

        Parameters
        ----------
        subplot_titles : list
                         Titles of plots.
        labels         : list
                         Labels for x and y axes.
        """
        self.settings   = {
                           'subplot titles' : subplot_titles,
                           'color scale'    : 'Portland',
                           'x label'        : labels[0],
                           'y label'        : labels[1]
                          }
        self.fig = make_subplots(
                                 rows=1,
                                 cols=1,
                                 specs=[
                                        [
                                         {"type": "xy"},
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
        self.fig.layout = go.Layout(
                                    xaxis=dict(title=self.settings['x label']),
                                    yaxis=dict(title=self.settings['y label'])
                                   )
        self.fig.show()

    def add_plot(self,data_x,data_y=None,label='',mode='lines+markers'):
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
        if type(data_y) == None:
           data_y = np.arange(0,data_x.shape[0])
        self.fig.add_trace(
                           go.Scatter(
                                      x=data_x,
                                      y=data_y,
                                      mode=mode,
                                      name=label,
                                     ),
                           row=1,
                           col=1
                          )

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

