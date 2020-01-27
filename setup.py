import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
      name             = "odak",
      version          = "0.1.3",
      author           = "Kaan Akşit",
      author_email     = "kunguz@gmail.com",
      description      = (
                          "Odak (pronounced O-dawec) is the fundamental Python library"
                          "for scientific computing in optical sciences."
                          "Odak includes modules for geometric raytracing and wave optics."
                         ),
      license          = "",
      keywords         = "example documentation tutorial",
      url              = "https://github.com/kunguz/odak",
      install_requires = [
                          "numpy",
                         ],
      packages         = [
                          'odak',
                          'odak/raytracing',
                          'odak/jones',
                          'odak/tools',
                          'odak/wave',
                          'odak/visualize',
                          'odak/visualize/blender',
                          'odak/manager',
                          'odak/oldschool'
                         ],
      long_description = read('README.md'),
      classifiers      = [
                          "Development Status :: 2 - Pre-Alpha",
                          "Intended Audience :: Science/Research",
                          "Intended Audience :: Developers",
                          "Topic :: Scientific/Engineering :: Physics",
                          "Programming Language :: Python",
                          "License :: OSI Approved :: Apache Software License",
                         ],
      python_requires='>=3.7.5',
      )
