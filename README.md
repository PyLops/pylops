![PyLops](https://github.com/PyLops/pylops/blob/master/docs/source/_static/pylops_b.png)

[![PyPI version](https://badge.fury.io/py/pylops.svg)](https://badge.fury.io/py/pylops)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/pylops/badges/version.svg)](https://anaconda.org/conda-forge/pylops)
[![Build Status](https://travis-ci.com/PyLops/pylops.svg?branch=master)](https://travis-ci.com/PyLops/pylops)
[![AzureDevOps Status](https://dev.azure.com/matteoravasi/PyLops/_apis/build/status/PyLops.pylops?branchName=master)](https://dev.azure.com/matteoravasi/PyLops/_build/latest?definitionId=2&branchName=master)
[![GithubAction Status](https://github.com/mrava87/pylops/actions/workflows/build.yaml/badge.svg)](https://github.com/mrava87/pylops/actions/workflows/build.yaml)
[![Documentation Status](https://readthedocs.org/projects/pylops/badge/?version=latest)](https://pylops.readthedocs.io/en/latest/?badge=latest)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/17fd60b4266347d8890dd6b64f2c0807)](https://www.codacy.com/gh/PyLops/pylops/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=PyLops/pylops&amp;utm_campaign=Badge_Grade)
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/17fd60b4266347d8890dd6b64f2c0807)](https://www.codacy.com/gh/PyLops/pylops/dashboard?utm_source=github.com&utm_medium=referral&utm_content=PyLops/pylops&utm_campaign=Badge_Coverage)
![OS-support](https://img.shields.io/badge/OS-linux,win,osx-850A8B.svg)
[![Slack Status](https://img.shields.io/badge/chat-slack-green.svg)](https://pylops.slack.com)

A [NUMFOCUS](https://numfocus.org/sponsored-projects/affiliated-projects) Affiliated Project.

## Objective
This Python library is inspired by the MATLAB [Spot – A Linear-Operator Toolbox](http://www.cs.ubc.ca/labs/scl/spot/) project.

Linear operators and inverse problems are at the core of many of the most used algorithms
in signal processing, image processing, and remote sensing. When dealing with small-scale problems,
the Python numerical scientific libraries [numpy](http://www.numpy.org)
and [scipy](https://www.scipy.org/scipylib/index.html) allow allow performing many
of the underlying matrix operations (e.g., computation of matrix-vector products and manipulation of matrices)
in a simple and compact way.

Many useful operators, however, do not lend themselves to an explicit matrix
representation when used to solve large-scale problems. PyLops operators, on the other hand, still represent a matrix
and can be treated in a similar way, but do not rely on the explicit creation of a dense (or sparse) matrix itself. Conversely,
the forward and adjoint operators are represented by small pieces of codes that mimic the effect of the matrix
on a vector or another matrix.

Luckily, many iterative methods (e.g. cg, lsqr) do not need to know the individual entries of a matrix to solve a linear system.
Such solvers only require the computation of forward and adjoint matrix-vector products as done for any of the PyLops operators.

Here is a simple example showing how a dense first-order first derivative operator can be created,
applied and inverted using numpy/scipy commands:
```python
import numpy as np
from scipy.linalg import lstsq

nx = 7
x = np.arange(nx) - (nx-1)/2

D = np.diag(0.5*np.ones(nx-1), k=1) - \
    np.diag(0.5*np.ones(nx-1), k=-1)
D[0] = D[-1] = 0 # take away edge effects

# y = Dx
y = np.dot(D, x)
# x = D'y
xadj = np.dot(D.T, y)
# xinv = D^-1 y
xinv = lstsq(D, y)[0]
```
and similarly using PyLops commands:
```python
from pylops import FirstDerivative

nx = 7
x = range(-(nx // 2), nx // 2 + (1 if nx % 2 else 0))

Dlop = FirstDerivative(nx, dtype='float64')

# y = Dx
y = Dlop * x
# x = D'y
xadj = Dlop.H * y
# xinv = D^-1 y
xinv = Dlop / y
```

Note how this second approach does not require creating a dense matrix, reducing both the memory load and the computational cost of
applying a derivative to an input vector x. Moreover, the code becomes even more compact and expressive than in the previous case
letting the user focus on the formulation of equations of the forward problem to be solved by inversion.


## Project structure
This repository is organized as follows:
* **pylops**:       python library containing various linear operators and auxiliary routines
* **pytests**:    set of pytests
* **testdata**:   sample datasets used in pytests and documentation
* **docs**:       sphinx documentation
* **examples**:   set of python script examples for each linear operator to be embedded in documentation using sphinx-gallery
* **tutorials**:  set of python script tutorials to be embedded in documentation using sphinx-gallery

## Getting started

You need **Python 3.6 or greater**.

#### From PyPi

If you want to use PyLops within your codes,
install it in your Python environment by typing the following command in your terminal:

```
pip install pylops
```

Open a python terminal and type:

```
import pylops
```

If you do not see any error, you should be good to go, enjoy!

#### From Conda-forge

Alternatively, you can install PyLops using the conda-forge distribution by typing the following command in your terminal:

```
conda install -c conda-forge pylops
```

#### From Github

You can also directly install from the main repository (although this is not reccomended)

```
pip install git+https://git@github.com/PyLops/pylops.git@master
```

#### From Docker

Finally, if you simply want to try PyLops but do not have Python in your
local machine, you can use our [Docker](https://www.docker.com) image. After installing Docker in your computer,
type the following command in your terminal (note that this will take some time the first time
you type it as you will download and install the docker image):

```
docker run -it -v /path/to/local/folder:/home/jupyter/notebook -p 8888:8888 mrava87/pylops:notebook
```

This will give you an address that you can put in your browser and will open a jupyter-notebook enviroment with PyLops
and other basic Python libraries installed. Here `/path/to/local/folder` is the absolute path of a local folder
on your computer where you will create a notebook (or containing notebooks that you want to continue working on). Note that
anything you do to the notebook(s) will be saved in your local folder.

A larger image with Conda distribution is also available. Simply use `conda_notebook` instead of `notebook` in the
previous command.

## Contributing

*Feel like contributing to the project? Adding new operators or tutorial?*

We advise using the [Anaconda Python distribution](https://www.anaconda.com/download)
to ensure that all the dependencies are installed via the `Conda` package manager. Follow
the following instructions and read carefully the [CONTRIBUTING](CONTRIBUTING.md) file before getting started.

### 1. Fork and clone the repository

Execute the following command in your terminal:

```
git clone https://github.com/your_name_here/pylops.git
```

### 2. Install PyLops in a new Conda environment
To ensure that further development of PyLops is performed within the same environment (i.e., same dependencies) as
that defined by ``requirements-dev.txt`` or ``environment-dev.yml`` files, we suggest to work off a new Conda enviroment.

The first time you clone the repository run the following command:
```
make dev-install_conda
```
To ensure that everything has been setup correctly, run tests:
```
make tests
```
Make sure no tests fail, this guarantees that the installation has been successfull.

Remember to always activate the conda environment every time you open a new terminal by typing:
```
source activate pylops
```

## Documentation
The official documentation of PyLops is available [here](https://pylops.readthedocs.io/).

Visit this page to get started learning about different operators and their applications as well as how to
create new operators yourself and make it to the ``Contributors`` list.

Moreover, if you have installed PyLops using the *developer environment* you can also build the documentation locally by
typing the following command:
```
make doc
```
Once the documentation is created, you can make any change to the source code and rebuild the documentation by
simply typing
```
make docupdate
```
Note that if a new example or tutorial is created (and if any change is made to a previously available example or tutorial)
you are required to rebuild the entire documentation before your changes will be visible.


## History
PyLops was initially written by [Equinor](https://www.equinor.com).
It is a flexible and scalable python library for large-scale optimization with linear
operators that can be tailored to our needs, and as contribution to the free software community.
Since June 2021, PyLops is a [NUMFOCUS](https://numfocus.org/sponsored-projects/affiliated-projects)
Affiliated Project.

## Citing
When using ``pylops`` in scientific publications, please cite the following paper:

- Ravasi, M., and Vasconcelos I., *PyLops--A linear-operator Python library for scalable algebra and optimization*,
  SoftwareX, (2020). [link](https://www.sciencedirect.com/science/article/pii/S2352711019301086).

## Tutorials
A list of video tutorials to learn more about PyLops:

- PyDataGlobal 2020: Youtube video [links](https://github.com/PyLops/pylops_pydata2020).
- Swung Rendezvous 2021: Youtube video [links](https://www.youtube.com/watch?v=rot1K1xr5H4).
- Transform 2021: Youtube video [links](https://www.youtube.com/watch?v=4GaVtE1ciLw).

## Contributors
* Matteo Ravasi, mrava87
* Carlos da Costa, cako
* Dieter Werthmüller, prisae
* Tristan van Leeuwen, TristanvanLeeuwen
* Leonardo Uieda, leouieda
* Filippo Broggini, filippo82
* Tyler Hughes, twhughes
* Lyubov Skopintseva, lskopintseva
* Francesco Picetti, fpicetti
* Alan Richardson, ar4
* BurningKarl, BurningKarl
* Nick Luiken, NickLuiken
