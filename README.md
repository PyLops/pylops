![PyLops](https://github.com/PyLops/pylops/blob/master/docs/source/_static/pylops_b.png)

[![NUMFOCUS](https://img.shields.io/badge/powered%20by-NumFOCUS-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)](https://numfocus.org/sponsored-projects/affiliated-projects)
[![PyPI version](https://badge.fury.io/py/pylops.svg)](https://badge.fury.io/py/pylops)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/pylops/badges/version.svg)](https://anaconda.org/conda-forge/pylops)
[![AzureDevOps Status](https://dev.azure.com/matteoravasi/PyLops/_apis/build/status/PyLops.pylops?branchName=dev)](https://dev.azure.com/matteoravasi/PyLops/_build/latest?definitionId=9&branchName=dev)
[![GithubAction Status](https://github.com/PyLops/pylops/actions/workflows/build.yaml/badge.svg?branch=dev)](https://github.com/PyLops/pylops/actions/workflows/build.yaml)
[![Documentation Status](https://readthedocs.org/projects/pylops/badge/?version=stable)](https://pylops.readthedocs.io/en/stable/?badge=stable)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/17fd60b4266347d8890dd6b64f2c0807)](https://www.codacy.com/gh/PyLops/pylops/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=PyLops/pylops&amp;utm_campaign=Badge_Grade)
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/17fd60b4266347d8890dd6b64f2c0807)](https://www.codacy.com/gh/PyLops/pylops/dashboard?utm_source=github.com&utm_medium=referral&utm_content=PyLops/pylops&utm_campaign=Badge_Coverage)
![OS-support](https://img.shields.io/badge/OS-linux,win,osx-850A8B.svg)
[![Slack Status](https://img.shields.io/badge/chat-slack-green.svg)](https://pylops.slack.com)
![PyPI downloads](https://img.shields.io/pypi/dm/pylops.svg?label=Pypi%20downloads)
![Conda downloads](https://img.shields.io/conda/dn/conda-forge/pylops.svg?label=Conda%20downloads)


# A Linear Operator Library for Python
PyLops is an open-source Python library focused on providing a backend-agnostic, idiomatic, matrix-free library of linear operators and related computations.
It is inspired by the iconic MATLAB [Spot – A Linear-Operator Toolbox](http://www.cs.ubc.ca/labs/scl/spot/) project.


## Installation
To get the most out of PyLops straight out of the box, we recommend `conda` to install PyLops:
```bash
conda install -c conda-forge pylops
```
You can also install with `pip`:
```bash
pip install pylops
```

See the docs ([Installation](https://pylops.readthedocs.io/en/stable/installation.html)) for more information about dependencies and performance.

## Why PyLops?
Linear operators and inverse problems are at the core of many of the most used algorithms in signal processing, image processing, and remote sensing.
For small-scale problems, matrices can be explicitly computed and manipulated with Python numerical scientific libraries such as [NumPy](http://www.numpy.org) and [SciPy](https://www.scipy.org/scipylib/index.html).

On the other hand, large-scale problems often feature matrices that are prohibitive in size—but whose operations can be described by simple functions.
PyLops exploits this to represent linear operators not as array of numbers, but by *functions which describe matrix-vector products*.

Indeed, many iterative methods (e.g. cg, lsqr) were designed to not rely on the elements of the matrix, only on the result of matrix-vector products.
PyLops offers many linear operators (derivatives, convolutions, FFTs and manyh more) as well as solvers for a variety of problems (e.g., least-squares and sparse inversion).
With these two ingredients, PyLops can describe and solve a variety of linear inverse problems which appear in many different areas.

## Example: A finite-difference operator

A first-order, central finite-difference derivative operator denoted D can be described either as a matrix (array of numbers), or as weighed stencil summation:

```python
import numpy as np

# Setup
nx = 7
x = np.arange(nx) - (nx-1)/2

# Matrix
D_mat = 0.5 * (np.diag(np.ones(nx-1), k=1) - np.diag(np.ones(nx-1), k=-1))
D_mat[0] = D_mat[-1] = 0 # remove edge effects

# Function: Stencil summation
def central_diff(x):
    y = np.zeros_like(x)
    y[1:-1] = 0.5 * (x[2:] - x[:-2])
    return y

# y = Dx
y = D_mat @ x
y_fun = central_diff(x)
print(np.allclose(y, y_fun)) # True
```

The matrix formulation can easily be paired with a SciPy least-squares solver to approximately invert the matrix, but this requires us to have an explicit representation for D (in this case, ``D_mat``):
```python
from scipy.linalg import lstsq

# xinv = D^-1 y
xinv = lstsq(D_mat, y)[0]
```
Relying on the functional approach, PyLops wraps a function similar to ``central_diff`` into the [``FirstDerivative``](https://pylops.readthedocs.io/en/stable/api/generated/pylops.FirstDerivative.html#pylops.FirstDerivative) operator, defining not only the forward mode (Dx) but also the transpose mode (Dᵀy).
In fact, it goes even further as the forward slash operator applies least-squares inversion!
```python
from pylops import FirstDerivative

D_op = FirstDerivative(nx, dtype='float64')

# y = Dx
y = D_op @ x
# xinv = D^-1 y
xinv_op = D_op / y

print(np.allclose(xinv, xinv_op)) # True
```

Note how the code becomes even more compact and expressive than in the previous case letting the user focus on the formulation of equations of the forward problem to be solved by inversion.
PyLops offers many other linear operators, as well as the ability to implement your own in a way that seamlessly interfaces with the rest of the ecosystem.


## Contributing

*Feel like contributing to the project? Adding new operators or tutorial?*

Follow the instructions detailed in the [CONTRIBUTING](CONTRIBUTING.md) file before getting started.

## Documentation
The official documentation of PyLops is available [here](https://pylops.readthedocs.io/).

Visit this page to get started learning about different operators and their applications as well as how to
create new operators yourself and make it to the ``Contributors`` list.

## History
PyLops was initially written by [Equinor](https://www.equinor.com).
It is a flexible and scalable python library for large-scale optimization with linear
operators that can be tailored to our needs, and as contribution to the free software community.
Since June 2021, PyLops is a [NUMFOCUS](https://numfocus.org/sponsored-projects/affiliated-projects)
Affiliated Project.

## Citing
When using PyLops in scientific publications, please cite the following paper:


- Ravasi, M., and I. Vasconcelos, 2020, *<b>PyLops—A linear-operator Python library for scalable algebra and optimization</b>*,
  SoftwareX, 11, 100361. doi: 10.1016/j.softx.2019.100361 [(link)](https://www.sciencedirect.com/science/article/pii/S2352711019301086)

## Tutorials
A list of video tutorials to learn more about PyLops:

- Transform 2022: Youtube video [links](https://www.youtube.com/watch?v=RIeVkuY_ivQ).
- Transform 2021: Youtube video [links](https://www.youtube.com/watch?v=4GaVtE1ciLw).
- Swung Rendezvous 2021: Youtube video [links](https://www.youtube.com/watch?v=rot1K1xr5H4).
- PyDataGlobal 2020: Youtube video [links](https://github.com/PyLops/pylops_pydata2020).

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
* BurningKarl, BurningKarl
* Muhammad Izzatullah, izzatum
* Juan Daniel Romero, jdromerom
* Aniket Singh Rawat, dikwickley
* Rohan Babbar, rohanbabbar04
* Wei Zhang, ZhangWeiGeo
* Fedor Goncharov, fedor-goncharov
* Alex Rakowski, alex-rakowski
* David Sollberger, solldavid
* Gustavo Coelho, guaacoelho
