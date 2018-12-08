# Contributing

Contributions are welcome and greatly appreciated!


## Types of Contributions

### Report Bugs

Report bugs at https://github.com/Statoil/pylops/issues

If you are playing with the PyLops library and find a bug, please
reporting it including:

* Your operating system name and version.
* Any details about your Python environment.
* Detailed steps to reproduce the bug.

### Propose New Operators or Features

The best way to send feedback is to open an issue at
https://github.com/Statoil/pylops/issues
with tag *enhancement*.

If you are proposing a new operator or a new feature:

* Explain in detail how it should work.
* Keep the scope as narrow as possible, to make it easier to implement.

### Implement Operators or Features
Look through the Git issues for operator or feature requests.
Anything tagged with *enhancement* is open to whoever wants to
implement it.

### Add Examples or improve Documentation
Writing new operators is not the only way to get involved and
contribute. Create examples with existing operators as well as
improving the documentation of existing operators is as important
as making new operators and very much encouraged.


## Getting Started to contribute

Ready to contribute?

1. Fork the `PyLops` repo.

2. Clone your fork locally:
    ```
    git clone https://github.com/your_name_here/pylops.git
    ```

3. Follow the installation instructions for *developers* that you find
in the README.md or in the online documentation.
Ensure that you are able to *pass all the tests before moving forward*.

4. Create a branch for local development:
    ```
    git checkout -b name-of-your-branch
    ```
    Now you can make your changes locally.

5. When you're done making changes, check that old and new tests pass
succesfully:
    ```
    python setup.py test
    ```

6. Commit your changes and push your branch to GitLab::
    ```
    git add .
    git commit -m "Your detailed description of your changes."
    git push origin name-of-your-branch
    ```
    Remember to add ``-u`` when pushing the branch for the first time.

7. Submit a pull request through the GitHub website.


### Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include new tests for all the core routines that have been developed.
2. If the pull request adds functionality, the docs should be updated accordingly.
