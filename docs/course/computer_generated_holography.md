??? quote end "Narrate section"
    <audio controls="controls">
         <source type="audio/mp3" src="../media/computer_generated_holography.mp3"></source>
    </audio>


# Computer-Generated Holography


In this section, we introduce Computer-Generated Holography (CGH) [@born2013principles, @goodman2005introduction] as another emerging method to simulate light.
CGH offers an upgraded but more computationally expensive way to simulating light concerning the raytracing method described in the previous section.
This section dives deep into CGH and will explain how CGH differs from raytracing as we go.


## What is holography?


:octicons-info-24: Informative 


Holography is a method in Optical sciences to represent light distribution using amplitude and phase of light.
In much simpler terms, holography describes light distribution emitted from an object, scene, or illumination source over a surface.
The primary difference of holography concerning raytracing is that it accounts not only amplitude or intensity of light but also the phase of light.
In raytracing, the smallest building block that defines light is a ray, whereas, in holography, the building block is a light distribution over surfaces.
In other terms, while raytracing traces rays, holography deals with surface-to-surface light transfer.


??? tip end "Did you know this source?"
    There is an active repository on GitHub, where latest CGH papers relevant to display technologies are listed.
    Visit [GitHub:bchao1/awesome-holography](https://github.com/bchao1/awesome-holography) for more.



### What is a hologram?


:octicons-info-24: Informative 


Hologram is either a surface or a volume that modifies the light distribution of incoming light in terms of phase and amplitude.
Diffraction gratings, Holographic Optical Elements, or Metasurfaces are good examples of holograms.


### What is Computer-Generated Holography?


:octicons-info-24: Informative 


It is the computerized version (discrete sampling) of holography.
In other terms, whenever you can program the phase or amplitude of light, this will get us to Computer-Generated Holography.


??? question end "Where can I find an extensive summary on CGH?"
    You may be wondering about the greater physical details of CGH.
    In this case, we suggest our readers watch the video below.
    Please watch this video for an extensive summary on CGH [@kavakli2022optimizing].
    <center> <iframe width="560" height="315" src="https://www.youtube.com/embed/z_AtSgct6_I" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe> </center>


## Defining a slice of a lightfield :material-alert-decagram:{ .mdx-pulse title="Too important!" }


:octicons-info-24: Informative ·
:octicons-beaker-24: Practical


CGH deals with generating optical fields that capture light from various scenes.
CGH often describes these optical fields (a.k.a. lightfields, holograms) as planes.
So in CGH, light travels from plane to plane, as depicted below.
Roughly, CGH deals with plane to plane interaction of light, whereas raytracing is a ray or beam oriented description of light.


<figure markdown>
  ![Image title](media/hologram_generation.png){ width="600" }
  <figcaption>A rendering showing how a slice (a.k.a. lightfield, optical field, hologram) propagates from one plane to another plane.</figcaption>
</figure>


In other words, in CGH, you define everything as a "lightfield," including light sources, materials, and objects.
Thus, we must first determine how to describe the mentioned lightfield in a computer.
So that we can run CGH simulations effectively.


A lightfield is a planar slice in the context of CGH, as depicted in the above figure.
This planar field is a pixelated 2D surface (could be represented as a matrix).
The pixels in this 2D slice hold values for the amplitude of light, $A$, and the phase of the light, $\phi$ at each pixel.
Whereas in classical raytracing, a ray only holds the amplitude or intensity of light.
With a caveat, though, raytracing could also be made to care about the phase of light. 
Still, it will then arrive with all the complications of raytracing, like sampling enough rays or describing scenes accurately.


Each pixel in this planar lightfield slice encapsulates the $A$ and $\phi$ as $A cos(wt + \phi)$.
If you recall our description of light, we explain that light is an electromagnetic phenomenon.
Here, we model the oscillating electric field of light with $A cos(wt + \phi)$ shown in our previous light description.
Note that if we stick to $A cos(wt + \phi)$, each time two fields intersect, we have to deal with trigonometric conversion complexities like sampled in this example:


$$
A_0 cos(wt + \phi_0) + A_1 cos(wt + \phi_1),
$$


Where the indices zero and one indicate the first and second fields, and we have to identify the right trigonometric conversion to deal with this sum.


Instead of complicated trigonometric conversions, what people do in CGH is to rely on complex numbers as a proxy to these trigonometric conversions.
In its proxy form, a pixel value in a field is converted into $A e^{-j \phi}$, where $j$ represents a complex number ($\sqrt{-1}$).
Thus, with this new proxy representation, the same intersection problem we dealt with using sophisticated trigonometry before could be turned into something as simple as $A_0 A_1 e^{-j(\phi_0 +\phi_1)}$.


In the above summation of two fields, the resulting field follows an exact sum of the two collided fields.
On the other hand, in raytracing, often, when a ray intersects with another ray, it will be left unchanged and continue its path.
However, in the case of lightfields, they form a new field.
This feature is called interference of light, which is not introduced in raytracing, and often raytracing omits this feature.
As you can tell from also the summation, two fields could enhance the resulting field (constructive interference) by converging to a brighter intensity, or these two fields could cancel out each other (destructive interference) and lead to the absence of light --total darkness--.

There are various examples of interference in nature.
For example, the blue color of a butterfly wing results from interference, as biology typically does not produce blue-colored pigments in nature.
More examples of light interference from daily lives are provided in the figure below.


<figure markdown>
  ![Image title](media/interference_examples.png){ width="600" }
  <figcaption>Two photographs showin some examples of light interference: (left) thin oil film creates rainbow interference patterns (CC BY-SA 2.5 by Wikipedia user John) and a soup bubble interference with light and creates vivid reflections (CC BY-SA 3.0 by Wikipedia user Brocken Inaglory).</figcaption>
</figure>


We have established an easy way to describe a field with a proxy complex number form.
This way, we avoided complicated trigonometric conversions. 
Let us look into how we use that in an actual simulation.
Firstly, we can define two separate matrices to represent a field using real numbers:


```python
import torch

amplitude = torch.tensor(100, 100, dtype = torch.float64)
phase = torch.tensor(100, 100, dtype = torch.float64)
```


In this above example, we define two matrices with `100 x 100` dimensions.
Each matrix holds floating point numbers, and they are real numbers.
To convert the amplitude and phase into a field, we must define the field as suggested in our previous description.
Instead of going through the same mathematical process for every piece of our future codes, we can rely on a utility function in odak to create fields consistently and coherently across all our future developments.
The utility function we will review is `odak.learn.wave.generate_complex_field()`:


=== ":octicons-file-code-16: `odak.learn.wave.generate_complex_field`"

    ::: odak.learn.wave.generate_complex_field


Let us use this utility function to expand our previous code snippet and show how we can generate a complex field using that:


```python
import torch
import odak # (1)

amplitude = torch.tensor(100, 100, dtype = torch.float64)
phase = torch.tensor(100, 100, dtype = torch.float64)
field = odak.learn.wave.generate_complex_field(amplitude, phase) # (2)
```

1. Adding `odak` to our imports.
2. Generating a field using `odak.learn.wave.generate_complex_field`.

## Propagating a field in free space :material-alert-decagram:{ .mdx-pulse title="Too important!" }


:octicons-info-24: Informative ·
:octicons-beaker-24: Practical


The next question we have to ask is related to the field we generated in our previous example.
In raytracing, we propagate rays in space, whereas in CGH, we propagate a field described over a surface onto another target surface.
So we need a transfer function that projects our field on another target surface.
That is the point where free space beam propagation comes into play.
As the name implies, free space beam propagation deals with propagating light in free space from one surface to another.
This entire process of propagation is also referred to as light transport in the domains of Computer Graphics.
In the rest of this section, we will explore means to simulate beam propagation on a computer.


??? tip end "A good news for Matlab fans!"
    We will indeed use `odak` to explore beam propagation.
    However, there is also a book in the literature, `(Numerical simulation of optical wave propagation: With examples in MATLAB by Jason D. Schmidt](https://www.spiedigitallibrary.org/ebooks/PM/Numerical-Simulation-of-Optical-Wave-Propagation-with-Examples-in-MATLAB/eISBN-9780819483270/10.1117/3.866274?SSO=1)`[@schmidt2010numerical], that provides a crash course on beam propagation using MATLAB.


