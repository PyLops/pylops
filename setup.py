import os
from distutils.core import setup
from setuptools import find_packages
from setuptools_scm import get_version


# Project description
descr = """
        Python library implementing some of the most common linear operators
        without requiring to explicitly create a dense (or sparse) matrix.
        """

# Utility function to read the README file.
def read(file_name):
    """print README file"""
    return open(os.path.join(os.path.dirname(__file__), file_name)).read()


# Setup
setup(
    name='pylops',
    version=get_version(root='.', relative_to=__file__),
    description=descr,
    long_description=open('README.md').read(),
    keywords=['algebra',
              'inverse problems',
              'large-scale optimization'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.6',
        'Topic :: Mathematics :: Inverse Problems',
    ],
    author='mrava',
    author_email='mrava@equinor.com',
    install_requires=['numpy', 'scipy', 'matplotlib'],
    packages=find_packages(exclude=['pytests']),
    use_scm_version=True,
    setup_requires=['pytest-runner', 'setuptools_scm'],
    test_suite='pytests',
    tests_require=['pytest'],
    zip_safe=True)
