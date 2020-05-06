.. _installation:

Installation
============

The PyLops project stives to create a library that is easy to install in
any environment and has a very limited number of dependencies. However,
since *Python2* will retire soon, we have decided to only focus on a
*Python3* implementation. If you are still using *Python2*, hurry up!

For this reason you will need **Python 3.6 or greater** to get started.


Dependencies
------------

Our mandatory dependencies are limited to:

* `numpy <http://www.numpy.org>`_
* `scipy <http://www.scipy.org/scipylib/index.html>`_

We advise using the `Anaconda Python distribution <https://www.anaconda.com/download>`_
to ensure that these dependencies are installed via the ``Conda`` package manager. This
is not just a pure stylistic choice but comes with some *hidden* advantages, such as the linking to
``Intel MKL`` library (i.e., a highly optimized BLAS library created by Intel).

If you simply want to use PyLops for teaching purposes or for small-scale examples, this should not
really affect you. However, if you are interested in getting better code performance,
read carefully the :ref:`performance` page.


Optional dependencies
---------------------

PyLops's optional dependencies refer to those dependencies that we do not include
in our ``requirements.txt`` and ``environment.yml`` files and thus are not strictly
needed nor installed directly as part of a standar installation (see below for details)

However, we sometimes implement additional back-ends (referred to as ``engine`` in the code)
for some of our operators in order to improve their performance.
To do so, we rely on third-party libraries. Those libraries are generally added to the
list of our optional dependencies.
If you are not after code performance, you may simply stick to the mandatory dependencies
and pylops will ensure to always fallback to one of those for any linear operator.

If you are instead after code performance, take a look at the *Optional dependencies* section
in the :ref:`performance` page.


Step-by-step installation for users
-----------------------------------

Python environment
~~~~~~~~~~~~~~~~~~

Activate your Python environment, and simply type the following command in your terminal
to install the PyPi distribution:

.. code-block:: bash

   >> pip install pylops

If using Conda, you can also install our conda-forge distribution via:

.. code-block:: bash

   >> conda install -c conda-forge pylops

Note that using the ``conda-forge`` distribution is reccomended as all the dependencies (both mandatory
and optional) will be correctly installed for you, while only mandatory dependencies are installed
using the ``pip`` distribution.

Alternatively, to access the latest source from github:

.. code-block:: bash

   >> pip install https://github.com/equinor/pylops/archive/master.zip

or just clone the repository

.. code-block:: bash

   >> git clone https://github.com/equinor/pylops.git

or download the zip file from the repository (green button in the top right corner of the
main github repo page) and install PyLops from terminal using the command:

.. code-block:: bash

   >> make install

Docker
~~~~~~

If you want to try PyLops but do not have Python in your
local machine, you can use our `Docker <https://www.docker.com>`_ image instead.

After installing Docker in your computer, type the following command in your terminal
(note that this will take some time the first time you type it as you will download and install the docker image):

.. code-block:: bash

   >> docker run -it -v /path/to/local/folder:/home/jupyter/notebook -p 8888:8888 mrava87/pylops:notebook

This will give you an address that you can put in your browser and will open a jupyter-notebook enviroment with PyLops
and other basic Python libraries installed. Here `/path/to/local/folder` is the absolute path of a local folder
on your computer where you will create a notebook (or containing notebooks that you want to continue working on). Note that
anything you do to the notebook(s) will be saved in your local folder.

A larger image with Conda distribution is also available. Simply use `conda_notebook` instead of `notebook` in the
previous command.


Step-by-step installation for developers
----------------------------------------
Fork and clone the repository by executing the following in your terminal:

.. code-block:: bash

   >> git clone https://github.com/your_name_here/pylops.git

The first time you clone the repository run the following command:

.. code-block:: bash

   >> make dev-install

If you prefer to build a new Conda enviroment just for PyLops, run the following command:

.. code-block:: bash

   >> make dev-install_conda

To ensure that everything has been setup correctly, run tests:

.. code-block:: bash

    >> make tests

Make sure no tests fail, this guarantees that the installation has been successfull.

If using Conda environment, always remember to activate the conda environment every time you open
a new *bash* shell by typing:

.. code-block:: bash

   >> source activate pylops