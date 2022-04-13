# Installation
We use odak with Linux operating systems. 
Therefore, we don't know if it can work with Windows or Mac operating systems.
Odak can be installed in multiple ways. 
However, our recommended method for installing Odak is using [pip](https://pypi.org/project/pip) distribution system. 
We update Odak within pip with each new version.
Thus, the most straightforward way to install Odak is to use the below command in a Linux shell:

```bash
pip3 install odak
```
Note that Odak is in constant development. 
One may want to install the latest and greatest odak in the source repository for their reasons.
In this case, our recommended method is to rely on pip for installing Odak from the source using:

```bash
pip3 install git+https://github.com/kunguz/odak
```

One can also install Odak without pip by first getting a local copy and installing using Python. 
Such an installation can be conducted using:

```bash
git clone git@github.com:kunguz/odak.git
cd odak
pip3 install -r requirements.txt
python3 setup.py install
```

## Notes before running
Some notes should be highlighted to users, and these include:

* Odak installs `PyTorch` that only uses `CPU`. 
To properly install `PyTorch` with GPU support, please consult [PyTorch website](https://pytorch.org).

## Testing an installation
After installing Odak, one can test if Odak has been appropriately installed with its dependencies by running the unit tests.
To be able to run unit tests, make sure to have `pytest` installed:

```bash
pip3 install -U pytest
```

Once `pytest` is installed, unit tests can be run by calling:
 
```bash
cd odak
pytest
```
The tests should return no error.
However, if an error is encountered, please [start a new issue](https://github.com/kunguz/odak/issues) to help us be aware of the issue.
