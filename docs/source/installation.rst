.. _installation:

Installation
============

The PyLops project stives to create a library that is easy to install in
any environment and has a very limited number of dependencies. However,
since *Python2* will retire soon, we have decided to only focus on a
*Python3* implementation. If you are still using *Python2*, hurry up!

For this reason you will need **Python 3.6.4 or greater** to get started.


Dependencies
------------

Our dependencies are limited to:

* `numpy <http://www.numpy.org>`_
* `scipy <http://www.scipy.org/scipylib/index.html>`_

Nevertheless, we advise using the `Anaconda Python distribution <https://www.anaconda.com/download>`_
to ensure that these dependencies are installed via the ``Conda`` package manager. This
is not just a pure stylistic choice but comes with some *hidden* advantages, such as the linking to
``Intel MKL`` library (i.e., a highly optimized ``BLAS library`` created by Intel).

If you simply want to use PyLops for teaching purposes or for small-scale examples, this should not
really affect you. However, if you are interested in getting better code performance,
read carefully the :ref:`performance` page.


Optional dependencies
---------------------

So far PyLops has no optional dependencies. However we will soon start using libraries to
improve the performance of some of our operators and add such libraries to the optional dependencies.
Again, if you are after code performance, take a look at the *Optional dependencies* section in
the :ref:`performance` page.


Step-by-step installation for users
-----------------------------------

Simply type the following command in your terminal:

.. code-block:: bash

   >> pip install pylops

Alternatively, to access the latest source from github:

.. code-block:: bash

   >> pip install https://github.com/Statoil/pylops/archive/master.zip

or just clone the repository

.. code-block:: bash

   >> git clone https://github.com/Statoil/pylops.git

or download the zip file of the repository (green button in the top right corner of the main github repo page) and
install PyLops from terminal using the command:

.. code-block:: bash

   >> make install


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