import os
from setuptools import setup, find_packages

read = lambda fname: open(os.path.join(os.path.dirname(__file__), fname)).read()

thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = os.path.join(thelibFolder, "requirements.txt")
install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

setup(
    name="odak",
    version="0.2.7",
    author="Kaan Akşit",
    author_email="kaanaksit@kaanaksit.com",
    description="Odak, the fundamental Python library for scientific computing in optical sciences.",
    license=read("LICENSE.txt"),
    keywords="optics, holography, perception, graphics",
    url="https://github.com/kaanaksit/odak",
    install_requires=install_requires,
    packages=find_packages(where="."),
    package_dir={"odak": "odak"},
    package_data={"odak": ["catalog/data/*.json", "learn/models/*.json"]},
    data_files=[
        (
            "",
            [
                "LICENSE.txt",
                "README.md",
                "THANKS.txt",
                "requirements.txt",
                "short_readme.md",
            ],
        )
    ],
    long_description=read("short_readme.md"),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python",
        "License :: OSI Approved :: MPL-2.0",
    ],
    python_requires=">=3.9",
)
