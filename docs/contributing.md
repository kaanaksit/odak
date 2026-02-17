# Contributing to Odak 🚀

Odak is in constant development. 
We shape Odak according to the most current needs in our scientific research.
We welcome both users and developers in the open-source community as long as they have good intentions (e.g., scientific research).
For the most recent description of Odak, please consult [our description](beginning.md).
If you are planning to use Odak for industrial purposes, please reach out to [Kaan Akşit](mailto:kaanaksit@kaanaksit.com). 📧
All of the Odak contributors are listed in our [`THANKS.txt`](https://github.com/kaanaksit/odak/blob/master/THANKS.txt) and added to [`CITATION.cff`](https://github.com/kaanaksit/odak/blob/master/CITATION.cff) regardless of how much they contribute to the project.
Their names are also included in our [Digital Object Identifier (DOI) page](https://zenodo.org/record/5526684). 📚

## Contributing process 🤝
Contributions to Odak can come in different forms.
It can either be code or documentation related contributions.
Historically, Odak has evolved through scientific collaboration, in which authors of Odak identified a collaborative project with a new potential contributor.
You can always reach out to [Kaan Akşit](mailto:kaanaksit@kaanaksit.com) to query your idea for potential collaborations in the future. 
Another potential place to identify likely means to improve odak is to address outstanding [issues of Odak](https://github.com/kaanaksit/odak/issues). 🐛

### Code 🧑‍💻
Odak's `odak` directory contains the source code. 
To add to it, please make sure that you can install and test Odak on your local computer.
The [installation documentation](installation.md) contains routines for installation and testing, please follow that page carefully.

We typically work with `pull requests`. 
If you want to add new code to Odak, please do not hesitate to fork Odak's git repository and have your modifications on your fork at first.
Once you test the modified version, please do not hesitate to initiate a pull request.
We will revise your code, and if found suitable, it will be merged to the master branch.
Remember to follow `numpy` convention while adding documentation to your newly added functions to Odak.

Another thing to mention is regarding to the code quality and standard.
Although it hasn't been strictly followed since the start of Odak, note that Odak follows code conventions of `flake8`, which can be installed using:

```
pip3 install flake8
```

You can always check for code standard violations in Odak by running these two commands:

```
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
```

There are tools that can automatically fix code in terms of following standards.
One primary tool that we are aware of is `autopep8`, which can be installed using:

```
pip3 install autopep8
```

Please once you are ready to have a pull request, make sure to add a unit test for your additions in `test` folder, and make sure to test all unit tests by running `pytest`.
If your system do not have `pytest` installed, it can be installed using:

```
pip3 install pytest
```

It is recommended to run the full test suite to make sure your changes don't break existing functionality.
You can also run specific tests related to your changes for quicker testing.

### Documentation 📖
Under Odak's source's root directory, you will find a folder named `docs`.
This directory contains all the necessary information to generate the pages in this documentation.
If you are interested in improving the documentation of Odak, this directory is the place where you will be adding things.

Odak's documentation is built using [`mkdocs`](https://www.mkdocs.org/).
At this point, I assume that you have successfully installed Odak on your system.
If you haven't yet, please follow [installation documentation](installation.md).
To be able to run documentation locally, make sure to have the correct dependencies installed properly:

```
pip3 install plyfile
pip3 install Pillow
pip3 install tqdm
pip3 install mkdocs-material
pip3 install mkdocstrings
```

Once you have dependencies appropriately installed, navigate to the source directory of Odak in your hard drive and run a test server:

```
cd odak
mkdocs serve
```

If all goes well, you should see a bunch of lines on your terminal, and the final lines should look similar to these:

```
INFO     -  Documentation built in 4.45 seconds
INFO     -  [22:15:22] Serving on http://127.0.0.1:8000/odak/
INFO     -  [22:15:23] Browser connected: http://127.0.0.1:8000/odak/
```

At this point, you can start your favourite browser and navigate to [`http://127.0.0.1:8000/odak`](http://127.0.0.1:8000/odak) to view documentation locally.
This local viewing is essential as it can help you view your changes locally on the spot before actually committing.
One last thing to mention here is the fact that Odak's `docs` folder's structure is self-explanatory.
It follows `markdown` rules, and `mkdocsstrings` style is `numpy`.

## Code of Conduct 🤝
Please note that all contributors are expected to follow our code of conduct in all project interactions.
We are committed to providing a harassment-free experience for everyone, regardless of gender, gender identity and expression, sexual orientation, disability, physical appearance, body size, race, age, or religion. 🌍

## Reporting Issues 🐛
If you encounter any bugs or issues please check if the issue has already been reported in our [issue tracker](https://github.com/kaanaksit/odak/issues).
If you are reporting a new issue, please include:
- A clear and descriptive title 📌  
- A detailed explanation of the problem 💬
- Steps to reproduce the issue 🔄
- Information about your environment (OS, Python version, etc.) 🖥️
