.. _installation:

Installation
============

You need **Python 3.6.4 or greater**.

We advise using the `Anaconda Python distribution <https://www.anaconda.com/download>`__
to ensure that all the dependencies are installed via the ``Conda`` package manager.

Step-by-step installation for users
-----------------------------------

Simply type the following command in your terminal:

.. code-block:: bash

   >> pip install pylops

Alternatively, you can clone the repository

.. code-block:: bash

   >> git clone git@github.com:Statoil/pylops.git

or download the zip file of the repository (green button in the top right corner of the main github repo page).


Step-by-step installation for developers
----------------------------------------
Fork and clone the repository by executing the following in your terminal:

.. code-block:: bash

   >> git clone git@github.com:your_name_here/pylops.git

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

Again, if using Conda environment, remember to always activate the conda environment every time you open
a new *bash* shell by typing:

.. code-block:: bash

   >> source activate lops