# Installation
We encourage users to use virtual environments in their development pipeline when working with or developing odak.
You can simply create and activate a virtual environment by using the following syntax:

```bash
python3 -m venv odak
source odak/bin/activate
```

Once activated, you can install odak using the previous instructions.
To deactivate the virtual environment, use `deactivate` command in your terminal.

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
pip3 install git+https://github.com/kaanaksit/odak
```

One can also install Odak without pip by first getting a local copy and installing using Python. 
Such an installation can be conducted using:

```bash
git clone git@github.com:kaanaksit/odak.git
cd odak
pip3 install -r requirements.txt
pip3 install -e .
```


## Uninstalling the Development version
If you have to remove the development version of `odak`, you can first try:

```bash
pip3 uninstall odak
sudo pip3 uninstall odak
```

And if for some reason, you are still able to import `odak` after that, check `easy-install.pth` file which is typically found `~/.local/lib/pythonX/site-packages`, where `~` refers to your home directory and `X` refers to your Python version.
In that file, if you see odak's directory listed, delete it.
This will help you remove development version of `odak`.


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
However, if an error is encountered, please [start a new issue](https://github.com/kaanaksit/odak/issues) to help us be aware of the issue.

## Notes before running
Some notes should be highlighted to users, and these include:

* Odak installs `PyTorch` that only uses `CPU`. 
To properly install `PyTorch` with GPU support, please consult [PyTorch website](https://pytorch.org).
