[![DOI](https://zenodo.org/badge/3987171.svg)](https://zenodo.org/badge/latestdoi/3987171) 
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0) 


# Odak
Odak (pronounced "O-dac") is the fundamental library for scientific computing in optical sciences, computer graphics, and visual perception.
Odak is also the toolkit for the research conducted in [Computational Light Laboratory](https://complightlab.com).
To learn more about what Odak can do to help your design, experimentation, and development, consult our [documentation](https://kaanaksit.github.io/odak/)!


## Getting Started

### Installing
For the most recent guidance on installing Odak, please consult to [installation documentation](https://kaanaksit.github.io/odak/odak/installation/).
Odak can be installed using [pip](https://pypi.org/project/pip):

```bash
pip3 install odak
```
or you can follow this, but remember that it will install the latest version in the repository this way:

```bash
pip3 install git+https://github.com/kaanaksit/odak --upgrade
```

or this:

```bash
git clone git@github.com:kaanaksit/odak.git
cd odak
pip3 install -r requirements.txt
pip3 install -e .
```

### Usage and examples
You can import Odak and start designing your next in Optics, Computer Graphics, or Perception! 
We prepared a [documentation](https://kaanaksit.com/odak/) on usage and much more.
For absolute begineers can [learn light, computation and odak with our Computatioonal Light.](https://kaanaksit.com/odak/course)

Here is a simple example of raytracing with odak:

```python
import odak
import torch

starting_point = torch.tensor([0., 0., 0.])
end_point = torch.tensor([1., 1., 5.])
rays = odak.learn.raytracing.create_ray_from_two_points(
                                                        starting_point,
                                                        end_point
                                                       )

triangle = torch.tensor([[
                          [-5., -5., 5.],
                          [ 5., -5., 5.],
                          [ 0.,  5., 5.]
                         ]])

normals, distance, _, _, check = odak.learn.raytracing.intersect_w_triangle(
                                                                            rays,
                                                                            triangle
                                                                           )
print('intersection point is {}. Surface normal cosines are {}.'.format(normals[0, 0], normals[0, 1]))
```

Here is a simple example of computer-generated holography with odak:
```python
import odak
import torch


wavelength = 532e-9
pixel_pitch = 8e-6 
distance = 5e-3
propagation_type = 'Angular Spectrum'
k = odak.learn.wave.wavenumber(wavelength)


amplitude = torch.zeros(500, 500)
amplitude[200:300, 200:300 ] = 1.
phase = torch.randn_like(amplitude) * 2 * odak.pi
hologram = odak.learn.wave.generate_complex_field(amplitude, phase)


image_plane = odak.learn.wave.propagate_beam(
                                             hologram,
                                             k,
                                             distance,
                                             pixel_pitch,
                                             wavelength,
                                             propagation_type,
                                             zero_padding = [True, False, True]
                                            )
image_intensity = odak.learn.wave.calculate_amplitude(image_plane) ** 2 
odak.learn.tools.save_image(
                            'image_intensity.png', 
                            image_intensity, 
                            cmin = 0., 
                            cmax = 1.
                           )
```

For more of these examples, you can either check our [course documentation](https://kaanaksit.com/odak/course) or visit our [unit tests](https://github.com/kaanaksit/odak/tree/master/test) to get inspired.

## Sample Projects that uses Odak
Here are some sample projects that uses `odak`:

* [ChromaCorrect: Prescription Correction in Virtual Reality Headsets through Perceptual Guidance](http://complightlab.com/ChromaCorrect/)
* [HoloBeam: Paper-Thin Near-Eye Displays](https://complightlab.com/publications/holobeam/)
* [Optimizing Vision and Visuals: Lectures on Cameras, Displays and Perception](https://github.com/complight/cameras-displays-perception-course)
* [Realistic Defocus Blur for Multiplane Computer-Generated Holography](https://github.com/complight/realistic_defocus)
* [Metameric Varifocal Computer-Generated Holography](https://github.com/complight/metameric_holography)
* [Learned Holographic Light Transport](https://github.com/complight/realistic_holography)


## How to cite
For adding the link of this repository in your publication, please use [Zenodo's citation](https://zenodo.org/badge/latestdoi/3987171). 
If you have used `odak` in your research project, please consider citing any of the following works:


```bibtex
@inproceedings{akcsit2023flexible,
  title={Flexible modeling of next-generation displays using a differentiable toolkit},
  author={Ak{\c{s}}it, Kaan and Kavakl{\i}, Koray},
  booktitle={Practical Holography XXXVII: Displays, Materials, and Applications},
  volume={12445},
  pages={131--132},
  year={2023},
  organization={SPIE}
}
```

```bibtex
@inproceedings{kavakli2022introduction,
  title={Introduction to Odak: a Differentiable Toolkit for Optical Sciences, Vision Sciences and Computer Graphics},
  author={Kavakl{\i}, Koray and Ak{\c{s}}it, Kaan},
  booktitle={Frontiers in Optics},
  pages={FTu1A--1},
  year={2022},
  organization={Optica Publishing Group}
}
```

```bibtex
@incollection{kavakli2022optimizing,
  title={Optimizing vision and visuals: lectures on cameras, displays and perception},
  author={Kavakli, Koray and Walton, David Robert and Antipa, Nick and Mantiuk, Rafa{\l} and Lanman, Douglas and Ak{\c{s}}it, Kaan},
  booktitle={ACM SIGGRAPH 2022 Courses},
  pages={1--66},
  year={2022}
}

```
