.. _contributing:

Contributing
############

Contributions are welcome and greatly appreciated!

The best way to get in touch with the core developers and maintainers is to
join the `PyLops slack channel <https://pylops.slack.com/>`_ as well as
open new *Issues* directly from the `GitHub repo <https://github.com/PyLops/pylops>`_.

Moreover, take a look at the :ref:`roadmap` page for a list of current ideas
for improvements and additions to the PyLops library.


Welcomed contributions
**********************

Bug reports
===========

Report bugs at https://github.com/PyLops/pylops/issues.

If you are playing with the PyLops library and find a bug, please report it including:

* Your operating system name and version.
* Any details about your Python environment.
* Detailed steps to reproduce the bug.

New operators and features
==========================

Open an issue at https://github.com/PyLops/pylops/issues with tag *enhancement*.

If you are proposing a new operator or a new feature:

* Explain in detail how it should work.
* Keep the scope as narrow as possible, to make it easier to implement.


Fix issues
==========

There is always a backlog of issues that need to be dealt with.
Look through the `GitHub Issues <https://github.com/PyLops/pylops/issues>`_ for operator/feature requests or bugfixes.


Add examples or improve documentation
=====================================

Writing new operators is not the only way to get involved and contribute. Create examples with existing operators
as well as improving the documentation of existing operators is as important as making new operators and very much
encouraged.


Step-by-step instructions for contributing
******************************************

Ready to contribute?

1. Follow all instructions in :ref:`DevInstall`.

2. Create a branch for local development, usually starting from the dev branch:

.. code-block:: bash

   >> git checkout -b name-of-your-branch dev

Now you can make your changes locally.

3. When you're done making changes, check that your code follows the guidelines for :ref:`addingoperator` and
that the both old and new tests pass successfully:

.. code-block:: bash

   >> make tests

If you have access to a GPU, it is advised also that old and new tests run with the CuPy 
backend pass successfully:

.. code-block:: bash

   >> make tests_gpu

4. Run flake8 to check the quality of your code:

.. code-block:: bash

   >> make lint

Note that PyLops does not enforce full compliance with flake8, rather this is used as a
guideline and will also be run as part of our CI.
Make sure to limit to a minimum flake8 warnings before making a PR.

5. Update the docs

.. code-block:: bash

   >> make docupdate


6. Commit your changes and push your branch to GitHub:

.. code-block:: bash

   >> git add .
   >> git commit -m "Your detailed description of your changes."
   >> git push origin name-of-your-branch

Remember to add ``-u`` when pushing the branch for the first time.
We recommend using `Conventional Commits <https://www.conventionalcommits.org/en/v1.0.0/#summary>`_ to
format your commit messages, but this is not enforced.

7. Submit a pull request through the GitHub website.


Pull Request Guidelines
***********************

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include new tests for all the core routines that have been developed.
2. If the pull request adds functionality, the docs should be updated accordingly.
3. Ensure that the updated code passes all tests.

Project structure
#################
This repository is organized as follows:
* **pylops**:     Python library containing various linear operators and auxiliary routines
* **pytests**:    set of pytests
* **testdata**:   sample datasets used in pytests and documentation
* **docs**:       Sphinx documentation
* **examples**:   set of python script examples for each linear operator to be embedded in documentation using sphinx-gallery
* **tutorials**:  set of python script tutorials to be embedded in documentation using sphinx-gallery
