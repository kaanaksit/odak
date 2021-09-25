# Holographic light transport
Odak contains essential ingredients for research and development targeting Computer-Generated Holography.
We consult the beginners in this matter to `Goodman's Introduction to Fourier Optics` book (ISBN-13:  978-0974707723) and `Principles of optics: electromagnetic theory of propagation, interference and diffraction of light` from Max Born and Emil Wolf (ISBN 0-08-26482-4).
This engineering note will provide a crash course on how light travels from a phase-only hologram to an image plane.


| <img src="https://github.com/kunguz/odak/raw/master/docs/notes/holographic_light_transport_files/hologram_generation.png" width="640" alt/> |
|:--:| 
| *Holographic image reconstruction. A collimated beam with a homogenous amplitude distribution (A=1) illuminates a phase-only hologram $u_0(x,y)$. Light from this hologram diffracts and arrive at an image plane $u(x,y)$ at a distance of z. Diffracted beams from each hologram pixel interfere at the image plane and, finally, reconstruct a target image.


As depicted in above figure, when such holograms are illuminated with a collimated coherent light (e.g. laser), these holograms can reconstruct an intended optical field at target depth levels.
How light travels from a hologram to a parallel image plane is commonly described using Rayleigh-Sommerfeld diffraction integrals (For more, consult `Heurtley, J. C. (1973). Scalar Rayleigh–Sommerfeld and Kirchhoff diffraction integrals: a comparison of exact evaluations for axial points. JOSA, 63(8), 1003-1008.`).
The first solution of the Rayleigh-Sommerfeld integral, also known as the Huygens-Fresnel principle, is expressed as follows:

<a href="https://www.codecogs.com/eqnedit.php?latex=u(x,y)=\frac{1}{j\lambda}&space;\int\!\!\!\!\int&space;u_0(x,y)\frac{e^{jkr}}{r}cos(\theta)dxdy" target="_blank"><img src="https://latex.codecogs.com/gif.latex?u(x,y)=\frac{1}{j\lambda}&space;\int\!\!\!\!\int&space;u_0(x,y)\frac{e^{jkr}}{r}cos(\theta)dxdy" title="u(x,y)=\frac{1}{j\lambda} \int\!\!\!\!\int u_0(x,y)\frac{e^{jkr}}{r}cos(\theta)dxdy" /></a>

where field at a target image plane, u(x,y), is calculated by integrating over every point of hologram's field, u_0(x,y).
Note that, for the above equation, r represents the optical path between a selected point over a hologram and a selected point in the image plane, theta represents the angle between these two points, k represents the wavenumber 2*pi/lambda and lambda represents the wavelength of light.
In this described light transport model, optical fields, u_0(x,y) and u(x,y), are represented with a complex value,
<a href="https://www.codecogs.com/eqnedit.php?latex=u_0(x,y)=A(x,y)e^{j\phi(x,y)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?u_0(x,y)=A(x,y)e^{j\phi(x,y)}" title="u_0(x,y)=A(x,y)e^{j\phi(x,y)}" /></a>

where A represents the spatial distribution of amplitude and phi represents the spatial distribution of phase across a hologram plane.
The described holographic light transport model is often simplified into a single convolution with a fixed spatially invariant complex kernel, h(x,y) (`Sypek, Maciej. "Light propagation in the Fresnel region. New numerical approach." Optics communications 116.1-3 (1995): 43-48.`).

<a href="https://www.codecogs.com/eqnedit.php?latex=u(x,y)=u_0(x,y)&space;*&space;h(x,y)&space;=\mathcal{F}^{-1}(\mathcal{F}(u_0(x,y))&space;\mathcal{F}(h(x,y)))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?u(x,y)=u_0(x,y)&space;*&space;h(x,y)&space;=\mathcal{F}^{-1}(\mathcal{F}(u_0(x,y))&space;\mathcal{F}(h(x,y)))" title="u(x,y)=u_0(x,y) * h(x,y) =\mathcal{F}^{-1}(\mathcal{F}(u_0(x,y)) \mathcal{F}(h(x,y)))" /></a>

There are multiple variants of this simplified approach:

* `Matsushima, Kyoji, and Tomoyoshi Shimobaba. "Band-limited angular spectrum method for numerical simulation of free-space propagation in far and near fields." Optics express 17.22 (2009): 19662-19673.`,
* `Zhang, Wenhui, Hao Zhang, and Guofan Jin. "Band-extended angular spectrum method for accurate diffraction calculation in a wide propagation range." Optics letters 45.6 (2020): 1543-1546.`,
* `Zhang, Wenhui, Hao Zhang, and Guofan Jin. "Adaptive-sampling angular spectrum method with full utilization of space-bandwidth product." Optics Letters 45.16 (2020): 4416-4419.`

In many cases, people choose to use the most common form of h described as

<a href="https://www.codecogs.com/eqnedit.php?latex=h(x,y)=\frac{e^{jkz}}{j\lambda&space;z}&space;e^{\frac{jk}{2z}&space;(x^2&plus;y^2)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h(x,y)=\frac{e^{jkz}}{j\lambda&space;z}&space;e^{\frac{jk}{2z}&space;(x^2&plus;y^2)}" title="h(x,y)=\frac{e^{jkz}}{j\lambda z} e^{\frac{jk}{2z} (x^2+y^2)}" /></a>

where z represents the distance between a hologram plane and a target image plane.
Note that beam propagation can also be learned for physical setups to avoid imperfections in a setup and to improve the image quality at an image plane:

* `Peng, Yifan, et al. "Neural holography with camera-in-the-loop training." ACM Transactions on Graphics (TOG) 39.6 (2020): 1-14.`,
* `Chakravarthula, Praneeth, et al. "Learned hardware-in-the-loop phase retrieval for holographic near-eye displays." ACM Transactions on Graphics (TOG) 39.6 (2020): 1-18.`,
* `Kavaklı, Koray, Hakan Urey, and Kaan Akşit. "Learned holographic light transport." Applied Optics (2021).`.
