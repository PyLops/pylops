![PyLops](docs/source/_static/pylops_b.png)

## Objective
This Python library is inspired by the MATLAB [Spot – A Linear-Operator Toolbox](http://www.cs.ubc.ca/labs/scl/spot/) project.

Linear operators and inverse problems are at the core of many of the most used algorithms
in signal processing, image processing, and remote sensing. When dealing with small-scale problems,
the Python numerical scientific libraries [numpy](http://www.numpy.org)
and [scipy](https://www.scipy.org/scipylib/index.html) allow to perform many
of the underlying matrix operations (e.g., computation of matrix-vector products and manipulation of matrices)
in a simple and compact way.

Many useful operators, however, do not lend themselves to an explicit matrix
representation when used to solve large-scale problems. PyLops operators, on the other hand, still represent a matrix
and can be treated in a similar way, but do not rely on the explicit creation of a dense (or sparse) matrix itself. Conversely,
the forward and adjoint operators are represented by small pieces of codes that mimic the effect of the matrix
on a vector or another matrix.

Luckily, many iterative methods (e.g. cg, lsqr) do not need to know the individual entries of a matrix to solve a linear system.
Such solvers only require the computation of forward and adjoint matrix-vector products as done for any of the PyLops operators.

Here is simple example showing how a dense first-order first derivative operator can be created,
applied and inverted using numpy/scipy commands:
```python
nx = 7
x = np.arange(nx) - (nx-1)/2

D = np.diag(0.5*np.ones(nx-1),k=1) - np.diag(0.5*np.ones(nx-1),-1)
D[0] = D[-1] = 0 # take away edge effects

# y = Dx
y = np.dot(D,x)
# x = D'y
xadj = np.dot(D.T,y)
# xinv = D^-1 y
xinv = scipy.linalg.lstsq(D, y]
```
and similarly using PyLops commands:
```python
Dlop = FirstDerivative(nx, dtype='float64')
# y = Dx
y = Dlop*x
# x = D'y
xadj = Dlop.H*y
# xinv = D^-1 y
xinv = D / y
```

Note how this second approach does not require creating a dense matrix, reducing both the memory load and the computational cost of
applying a derivative to an input vector x. Moreover, the code becomes even more compact and espressive than in the previous case
letting the user focus on the formulation of equations of the forward problem to be solved by inversion.


## Project structure
This repository is organized as follows:
* **lops**:       python library containing various linear operators and auxiliary routines
* **pytests**:    set of pytests
* **testdata**:   sample datasets used in pytests and documentation
* **docs**:       sphinx documentation
* **examples**:   set of python script examples for each linear operator to be embedded in documentation using sphinx-gallery
* **tutorials**:  set of python script tutorials to be embedded in documentation using sphinx-gallery

## Getting started

We advise using the [Anaconda Python distribution](https://www.anaconda.com/download)
to ensure that all the dependencies are installed via the ``Conda`` package manager.

### 1. Clone the repository

Execute the following in your terminal:

```
git clone git@bitbucket.org:mravasi/pylops.git
```

### 2a. Installation for users (Your own environment)

The first time you clone the repository run the following command:
```
make install
```
to install the dependencies of PyLops and the PyLops library in your own active environment.

### 2b. Installation for users (New Conda environment)
The first time you clone the repository, create a new envionment and install the PyLops library
by running the following command:
```
make install_conda
```
Remember to always activate the conda environment every time you open a new *bash* shell by typing:
```
source activate pylops
```

### 3. Installation environment for developers (New Conda environment)
To ensure that further development of PyLops is performed within the same enviroment (i.e., same dependencies) as
that defined by ``requirements.txt`` and ``environment.yml`` files, we suggest to work off a new Conda enviroment.

The first time you clone the repository run the following command:
```
make dev-install_conda
```
To ensure that everything has been setup correctly, run tests:
```
make tests
```
Make sure no tests fail, this guarantees that the installation has been successfull.

Again, if using Conda environment, remember to always activate the conda environment every time you open
a new terminal by typing:
```
source activate lops
```

## Documentation
The official documentation of PyLops is available at *COMING SOON*.

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
PyLops was initially written and it is currently maintained by [Equinor](https://www.equinor.com).
It is a flexible and scalable python library for large-scale optimization with linear
operators that can be tailored to our needs, and as contribution to the free software community.


## Contributors
* Matteo Ravasi, Equinor