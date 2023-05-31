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


=== ":octicons-file-code-16: `odak.learn.raytracing.propagate_a_ray"

    ::: odak.learn.raytracing.propagate_a_ray


