??? quote end "Narrate section"
    <audio controls="controls">
         <source type="audio/mp3" src="../media/geometric_optics.mp3"></source>
    </audio>


# Modeling light with rays


Modeling light plays a crucial role in describing events based on light and helps designing mechanisms based on light (e.g., Realistic graphics in a video game, display or camera).
This chapter introduces the most basic description of light using geometric rays, also known as raytracing.
Raytracing has a long history, from ancient Greece or Islamic scholars to Physicists in the last couple of centuries or current Computer Graphics scientists.
We will not cover the history of raytracing.
Instead, we will focus on how we implement things to build "things" with them in the future.
As we provide algorithmic examples to support our descriptions, readers should be able to simulate light on their computers using the provided descriptions.


??? question end "Are there other good resources on modeling light with rays?"
    When I first started coding Odak, the first paper I read was on raytracing. 
    Thus, I recommend that paper for any starter:
    
    * [Spencer, G. H., and M. V. R. K. Murty. "General ray-tracing procedure." JOSA 52, no. 6 (1962): 672-678.](https://doi.org/10.1364/JOSA.52.000672)
    
    Beyond this paper, there are several resources that I can recommend for curious readers:
    
    * [Shirley, Peter. "Ray tracing in one weekend." Amazon Digital Services LLC 1 (2018): 4.](https://www.realtimerendering.com/raytracing/Ray%20Tracing%20in%20a%20Weekend.pdf)
    * [Morgan McGuire (2021). The Graphics Codex. Casual Effects.](https://graphicscodex.com/)


## Ray description :material-alert-decagram:{ .mdx-pulse title="Too important!" }


:octicons-info-24: Informative ·
:octicons-beaker-24: Practical


We have to define what "a ray" is.
A ray has a starting point in Euclidean space ($x_0, y_0, z_0 \in \mathbb{R}$).
We also have to define direction cosines to provide a direction for rays.
Direction cosines are three angles of a ray between the XYZ axis and that ray ($\theta_x, \theta_y, \theta_z \in \mathbb{R}$).
To calculate direction cosines, we must choose a point on that ray as $x_1, y_1,$ and $z_1$, and calculate its distance to the starting point of $x_0, y_0$ and $z_0$:

$$
x_{distance} = x_1 - x_0, \\
y_{distance} = y_1 - y_0, \\
z_{distance} = z_1 - z_0.
$$


Then, we can also calculate the Euclidian distance between starting point and the point chosen:

$$
s = \sqrt{x_{distance}^2 + y_{distance}^2 + z_{distance}^2}.
$$

Thus, we describe each direction cosines as:

$$
cos(\theta_x) = \frac{x_{distance}}{s}, \\
cos(\theta_y) = \frac{y_{distance}}{s}, \\
cos(\theta_z) = \frac{z_{distance}}{s}.
$$


Now that we know how to define a ray with a starting point, $x_0, y_0, z_0$ and a direction cosine, $cos(\theta_x), cos(\theta_y), cos(\theta_z)$, let us carefully analyze the parameters, returns, and source code of the provided two following functions in `odak` dedicated to creating a ray or rays.


=== ":octicons-file-code-16: `odak.learn.raytracing.create_ray`"

    ::: odak.learn.raytracing.create_ray

=== ":octicons-file-code-16: `odak.learn.raytracing.create_ray_from_two_points`"

    ::: odak.learn.raytracing.create_ray_from_two_points


In the future, we must find out where a ray lands after a certain amount of propagation distance for various purposes, which we will describe in this chapter.
For that purpose, let us also create a utility function that propagates a  ray to some distance, $d$, using $x_0, y_0, z_0$ and $cos(\theta_x), cos(\theta_y), cos(\theta_z)$:

$$
x_{new} = x_0 + cos(\theta_x) d,\\
y_{new} = y_0 + cos(\theta_y) d,\\
z_{new} = z_0 + cos(\theta_z) d.
$$


Let us also check the function provided below to understand its source code, parameters, and returns.
This function will serve as a utility function to propagate a ray or a batch of rays in our future simulations.


=== ":octicons-file-code-16: `odak.learn.raytracing.propagate_a_ray`"

    ::: odak.learn.raytracing.propagate_a_ray


It is now time for us to put what we have learned so far into an actual code.
We can create many rays using the two functions, `odak.learn.raytracing.create_ray_from_two_points` and `odak.learn.raytracing.create_ray`.
However, to do so, we need to have many points in both cases.
For that purpose, let's carefully review this utility function provided below.
This utility function can generate grid samples from a plane with some tilt, and we can also define the center of our samples to position points anywhere in Euclidean space.


=== ":octicons-file-code-16: `odak.learn.tools.grid_sample`"

    ::: odak.learn.tools.grid_sample


The below script provides a sample use case for the functions provided above.
I also leave comments near some lines explaining the code in steps.


=== ":octicons-file-code-16: `test_learn_ray.py`"

    ```python 
    --8<-- "test/test_learn_ray.py"
    ```

    1. Required libraries are imported.
    2. Defining a starting point, in order X, Y and Z locations.
       Size of starting point could be [1] or [1, 1].
    3. Defining some end points on a plane in grid fashion.
    4. `odak.learn.raytracing.create_ray_from_two_points` is verified with an example! Let's move on to `odak.learn.raytracing.create_ray`.
    5. Creating starting points with `odak.learn.tools.grid_sample` and defining some angles as the direction using `torch.randn`.
       Note that the angles are in degrees.
    6. `odak.learn.raytracing.create_ray` is verified with an example!
    7. `odak.learn.raytracing.propagate_a_ray` is verified with an example!
    8. Uncomment for enabling the visualizations.
    9. Uncomment for enabling the visualizations.


The above code also has parts that are commented.
We commented out these lines intentionally to avoid running it at every run.
Let me talk about these commented functions as well.
Odak offers a tidy approach to simple visualizations through packages called [Plotly](https://plotly.com/) and [Kaleido]().
To make these lines work by uncommenting them, you must first install `plotly` in your work environment.
This installation is as simple as `pip3 install plotly kaleido` in a Linux system.
As you install these packages and uncomment these lines, the code will produce a visualization similar to the one below.
Note that this is an interactive visualization where you can interact with your mouse clicks to rotate, shift, and zoom.
In this visualization, we visualize a single ray (green line) starting from our defined starting point (red dot) and ending at one of the `end_points` (blue dot).
We also highlight three axes with black lines to provide a reference frame.
Although `odak.visualize.plotly` offers us methods to visualize rays quickly for debugging, it is highly suggested to stick to a low number of lines when using it.
The proper way to draw many rays lies in modern path-tracing renderers such as [Blender](https://www.blender.org/).


<div>                        <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-2.12.1.min.js"></script>                <div id="ed004ef3-daf7-46d1-8220-01fd9627e15b" class="plotly-graph-div" style="height:100%; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("ed004ef3-daf7-46d1-8220-01fd9627e15b")) {                    Plotly.newPlot(                        "ed004ef3-daf7-46d1-8220-01fd9627e15b",                        [{"marker":{"color":"red","opacity":0.5,"size":3.0},"mode":"markers","scene":"scene","x":[5.0],"y":[5.0],"z":[0.0],"type":"scatter3d"},{"marker":{"color":"blue","opacity":0.5,"size":3.0},"mode":"markers","scene":"scene","x":[-20.0],"y":[-20.0],"z":[10.0],"type":"scatter3d"},{"line":{"color":"green","width":3.0},"mode":"lines","opacity":0.5,"scene":"scene","x":[5.0,-20.0],"y":[5.0,-20.0],"z":[0.0,10.0],"type":"scatter3d"},{"marker":{"color":"black","opacity":0.5,"size":3.0},"mode":"markers","scene":"scene","x":[-20.0],"y":[5.0],"z":[0.0],"type":"scatter3d"},{"line":{"color":"black","width":3.0},"mode":"lines","opacity":0.5,"scene":"scene","x":[5.0,-20.0],"y":[5.0,5.0],"z":[0.0,0.0],"type":"scatter3d"},{"marker":{"color":"black","opacity":0.5,"size":3.0},"mode":"markers","scene":"scene","x":[5.0],"y":[-20.0],"z":[0.0],"type":"scatter3d"},{"line":{"color":"black","width":3.0},"mode":"lines","opacity":0.5,"scene":"scene","x":[5.0,5.0],"y":[5.0,-20.0],"z":[0.0,0.0],"type":"scatter3d"},{"marker":{"color":"black","opacity":0.5,"size":3.0},"mode":"markers","scene":"scene","x":[5.0],"y":[5.0],"z":[10.0],"type":"scatter3d"},{"line":{"color":"black","width":3.0},"mode":"lines","opacity":0.5,"scene":"scene","x":[5.0,5.0],"y":[5.0,5.0],"z":[0.0,10.0],"type":"scatter3d"}],                        {"annotations":[{"font":{"size":16},"showarrow":false,"text":"Ray visualization","x":0.5,"xanchor":"center","xref":"paper","y":1.0,"yanchor":"bottom","yref":"paper"}],"scene":{"domain":{"x":[0.0,1.0],"y":[0.0,1.0]}},"template":{"data":{"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}}},                        {"responsive": true}                    )                };                            </script>        </div>


## Intersecting rays with a triangle :material-alert-decagram:{ .mdx-pulse title="Too important!" }


:octicons-info-24: Informative ·
:octicons-beaker-24: Practical


=== ":octicons-file-code-16: `test_learn_ray_intersect_w_a_triangle.py`"

    ```python 
    --8<-- "test/test_learn_ray_intersect_w_a_triangle.py"
    ```

    1. Uncomment for running visualization.
    2. Returning intersection normals as new rays, distances from starting point of input rays and a check which returns True if intersection points are inside the triangle.


<div>                        <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-2.12.1.min.js"></script>                <div id="c80aca20-5911-4b37-b845-05ce008aeee3" class="plotly-graph-div" style="height:100%; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("c80aca20-5911-4b37-b845-05ce008aeee3")) {                    Plotly.newPlot(                        "c80aca20-5911-4b37-b845-05ce008aeee3",                        [{"line":{"color":"black","width":3.0},"mode":"lines","opacity":0.5,"scene":"scene","x":[-5.0,5.0],"y":[-5.0,-5.0],"z":[10.0,10.0],"type":"scatter3d"},{"line":{"color":"black","width":3.0},"mode":"lines","opacity":0.5,"scene":"scene","x":[-5.0,0.0],"y":[-5.0,5.0],"z":[10.0,10.0],"type":"scatter3d"},{"line":{"color":"black","width":3.0},"mode":"lines","opacity":0.5,"scene":"scene","x":[5.0,0.0],"y":[-5.0,5.0],"z":[10.0,10.0],"type":"scatter3d"},{"marker":{"color":"blue","opacity":0.5,"size":3.0},"mode":"markers","scene":"scene","x":[-10.0,-10.0,-10.0,-10.0,-10.0,-5.0,-5.0,-5.0,-5.0,-5.0,0.0,0.0,0.0,0.0,0.0,5.0,5.0,5.0,5.0,5.0,10.0,10.0,10.0,10.0,10.0],"y":[-10.0,-5.0,0.0,5.0,10.0,-10.0,-5.0,0.0,5.0,10.0,-10.0,-5.0,0.0,5.0,10.0,-10.0,-5.0,0.0,5.0,10.0,-10.0,-5.0,0.0,5.0,10.0],"z":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"type":"scatter3d"},{"line":{"color":"blue","width":3.0},"mode":"lines","opacity":0.5,"scene":"scene","x":[-10.0,-3.0],"y":[-10.0,-3.0],"z":[0.0,10.0],"type":"scatter3d"},{"line":{"color":"blue","width":3.0},"mode":"lines","opacity":0.5,"scene":"scene","x":[-10.0,-3.0],"y":[-5.0,-1.5],"z":[0.0,10.0],"type":"scatter3d"},{"line":{"color":"blue","width":3.0},"mode":"lines","opacity":0.5,"scene":"scene","x":[-10.0,-3.0],"y":[0.0,0.0],"z":[0.0,10.0],"type":"scatter3d"},{"line":{"color":"blue","width":3.0},"mode":"lines","opacity":0.5,"scene":"scene","x":[-10.0,-3.0],"y":[5.0,1.5],"z":[0.0,10.0],"type":"scatter3d"},{"line":{"color":"blue","width":3.0},"mode":"lines","opacity":0.5,"scene":"scene","x":[-10.0,-3.0],"y":[10.0,3.0],"z":[0.0,10.0],"type":"scatter3d"},{"line":{"color":"blue","width":3.0},"mode":"lines","opacity":0.5,"scene":"scene","x":[-5.0,-1.5],"y":[-10.0,-3.0],"z":[0.0,10.0],"type":"scatter3d"},{"line":{"color":"blue","width":3.0},"mode":"lines","opacity":0.5,"scene":"scene","x":[-5.0,-1.5],"y":[-5.0,-1.5],"z":[0.0,10.0],"type":"scatter3d"},{"line":{"color":"blue","width":3.0},"mode":"lines","opacity":0.5,"scene":"scene","x":[-5.0,-1.5000002],"y":[0.0,0.0],"z":[0.0,10.0],"type":"scatter3d"},{"line":{"color":"blue","width":3.0},"mode":"lines","opacity":0.5,"scene":"scene","x":[-5.0,-1.5],"y":[5.0,1.5],"z":[0.0,10.0],"type":"scatter3d"},{"line":{"color":"blue","width":3.0},"mode":"lines","opacity":0.5,"scene":"scene","x":[-5.0,-1.5],"y":[10.0,3.0],"z":[0.0,10.0],"type":"scatter3d"},{"line":{"color":"blue","width":3.0},"mode":"lines","opacity":0.5,"scene":"scene","x":[0.0,0.0],"y":[-10.0,-3.0],"z":[0.0,10.0],"type":"scatter3d"},{"line":{"color":"blue","width":3.0},"mode":"lines","opacity":0.5,"scene":"scene","x":[0.0,0.0],"y":[-5.0,-1.5000002],"z":[0.0,10.0],"type":"scatter3d"},{"line":{"color":"blue","width":3.0},"mode":"lines","opacity":0.5,"scene":"scene","x":[0.0,0.0],"y":[0.0,0.0],"z":[0.0,10.0],"type":"scatter3d"},{"line":{"color":"blue","width":3.0},"mode":"lines","opacity":0.5,"scene":"scene","x":[0.0,0.0],"y":[5.0,1.5000002],"z":[0.0,10.0],"type":"scatter3d"},{"line":{"color":"blue","width":3.0},"mode":"lines","opacity":0.5,"scene":"scene","x":[0.0,0.0],"y":[10.0,3.0],"z":[0.0,10.0],"type":"scatter3d"},{"line":{"color":"blue","width":3.0},"mode":"lines","opacity":0.5,"scene":"scene","x":[5.0,1.5],"y":[-10.0,-3.0],"z":[0.0,10.0],"type":"scatter3d"},{"line":{"color":"blue","width":3.0},"mode":"lines","opacity":0.5,"scene":"scene","x":[5.0,1.5],"y":[-5.0,-1.5],"z":[0.0,10.0],"type":"scatter3d"},{"line":{"color":"blue","width":3.0},"mode":"lines","opacity":0.5,"scene":"scene","x":[5.0,1.5000002],"y":[0.0,0.0],"z":[0.0,10.0],"type":"scatter3d"},{"line":{"color":"blue","width":3.0},"mode":"lines","opacity":0.5,"scene":"scene","x":[5.0,1.5],"y":[5.0,1.5],"z":[0.0,10.0],"type":"scatter3d"},{"line":{"color":"blue","width":3.0},"mode":"lines","opacity":0.5,"scene":"scene","x":[5.0,1.5],"y":[10.0,3.0],"z":[0.0,10.0],"type":"scatter3d"},{"line":{"color":"blue","width":3.0},"mode":"lines","opacity":0.5,"scene":"scene","x":[10.0,3.0],"y":[-10.0,-3.0],"z":[0.0,10.0],"type":"scatter3d"},{"line":{"color":"blue","width":3.0},"mode":"lines","opacity":0.5,"scene":"scene","x":[10.0,3.0],"y":[-5.0,-1.5],"z":[0.0,10.0],"type":"scatter3d"},{"line":{"color":"blue","width":3.0},"mode":"lines","opacity":0.5,"scene":"scene","x":[10.0,3.0],"y":[0.0,0.0],"z":[0.0,10.0],"type":"scatter3d"},{"line":{"color":"blue","width":3.0},"mode":"lines","opacity":0.5,"scene":"scene","x":[10.0,3.0],"y":[5.0,1.5],"z":[0.0,10.0],"type":"scatter3d"},{"line":{"color":"blue","width":3.0},"mode":"lines","opacity":0.5,"scene":"scene","x":[10.0,3.0],"y":[10.0,3.0],"z":[0.0,10.0],"type":"scatter3d"},{"marker":{"color":["green","green","red","red","red","green","green","green","green","red","green","green","green","green","green","green","green","green","green","red","green","green","red","red","red"],"opacity":0.5,"size":3.0},"mode":"markers","scene":"scene","x":[-3.0,-3.0,-3.0,-3.0,-3.0,-1.5,-1.5,-1.5000002,-1.5,-1.5,0.0,0.0,0.0,0.0,0.0,1.5,1.5,1.5000002,1.5,1.5,3.0,3.0,3.0,3.0,3.0],"y":[-3.0,-1.5,0.0,1.5,3.0,-3.0,-1.5,0.0,1.5,3.0,-3.0,-1.5000002,0.0,1.5000002,3.0,-3.0,-1.5,0.0,1.5,3.0,-3.0,-1.5,0.0,1.5,3.0],"z":[10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0],"type":"scatter3d"}],                        {"annotations":[{"font":{"size":16},"showarrow":false,"text":"Ray visualization","x":0.5,"xanchor":"center","xref":"paper","y":1.0,"yanchor":"bottom","yref":"paper"}],"scene":{"aspectmode":"manual","aspectratio":{"x":1.0,"y":1.0,"z":1.0},"domain":{"x":[0.0,1.0],"y":[0.0,1.0]}},"template":{"data":{"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}}},                        {"responsive": true}                    )                };                            </script>        </div>
