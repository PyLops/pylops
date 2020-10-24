import os
from setuptools import setup, find_packages

def src(pth):
    return os.path.join(os.path.dirname(__file__), pth)

# Project description
descr = """
        Python library implementing linear operators to allow solving large-scale optimization
        problems without requiring to explicitly create a dense (or sparse) matrix.
        """

# Setup
setup(
    name='pylops',
    description=descr,
    long_description=open(src('README.md')).read(),
    long_description_content_type='text/markdown',
    keywords=['algebra',
              'inverse problems',
              'large-scale optimization'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    author='mrava',
    author_email='mrava@equinor.com',
    install_requires=['numpy >= 1.15.0', 'scipy >= 1.4.0'],
    extras_require={'advanced': ['llvmlite', 'numba',
                                 'pyfftw', 'PyWavelets',
                                 'scikit-fmm', 'spgl1']},
    packages=find_packages(exclude=['pytests']),
    use_scm_version=dict(root = '.',
                         relative_to = __file__,
                         write_to = src('pylops/version.py')),
    setup_requires=['pytest-runner', 'setuptools_scm'],
    test_suite='pytests',
    tests_require=['pytest'],
    zip_safe=True)
