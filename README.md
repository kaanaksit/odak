[![DOI](https://zenodo.org/badge/3987171.svg)](https://zenodo.org/badge/latestdoi/3987171) [![License: MPL 2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0) [![Inline docs](https://img.shields.io/readthedocs/odak)](https://kunguz.github.io/odak/)

# Odak
Odak (pronounced "O-dac") is the fundamental library for scientific computing in optical and visual perception sciences.
Odak includes modules for geometric [3D raytracing](odak/raytracing/), [Jones calculus](odak/jones), [wave optics](odak/wave), and [a set of tools](odak/tools) to ease pain in [measurement](odak/measurement), [exporting/importing CAD](odak/tools/asset.py), and [visualization](odak/visualize) during a design process. 
We have generated a set of recipes that go well with machine learning approaches compatible with the PyTorch learning framework as provided [here](odak/learn). 
We have created many [test scripts](test/) to inspire how you use Odak and helping your design process. 
Finally, we have created a [distribution system](odak/manager) to process tasks in parallel across multiple computing resources within the same network. 
Odak can either run using CPUs or automatically switch to [NVIDIA GPUs](odak/__init__.py#L8).
Consult to our [documentation](https://kunguz.github.io/odak) for more!

## Getting Started

### Installing
Odak can be installed using [pip](https://pypi.org/project/pip):

```bash
pip3 install git+https://github.com/kunguz/odak
```

or:

```bash
pip3 install odak
```

### Usage
You can import Odak and start designing your next in Optics, Computer Graphics or Perception! 
We prepared a [documentation](https://kunguz.github.io/odak) on usage and much more.
