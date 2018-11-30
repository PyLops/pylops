import os
from distutils.core import setup
from setuptools import find_packages
from setuptools_scm import get_version


def src(pth):
    return os.path.join(os.path.dirname(__file__), pth)

# Project description
descr = """
        Python library implementing some of the most common linear operators
        without requiring to explicitly create a dense (or sparse) matrix.
        """

# Utility function to read the README file.
def read(file_name):
    """print README file"""
    return open(src(file_name)).read()


# Setup
setup(
    name='pylops',
    version=get_version(root='.',
                        relative_to=__file__,
                        write_to=src('lops/version.py')),
    description=descr,
    long_description=open(src('README.md')).read(),
    keywords=['algebra',
              'inverse problems',
              'large-scale optimization'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    author='mrava',
    author_email='mrava@equinor.com',
    install_requires=['numpy', 'scipy', 'matplotlib', 'setuptools_scm'],
    packages=find_packages(exclude=['pytests']),
    use_scm_version=True,
    setup_requires=['pytest-runner'],
    test_suite='pytests',
    tests_require=['pytest'],
    zip_safe=True)
