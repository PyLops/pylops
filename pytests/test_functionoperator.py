"""
test_functionoperator.py

Test module for FunctionOperator. Tests 32 and 64 bit float and complex number
by wrapping a matrix multiplication as a FunctionOperator.
Also provides a good starting point for new tests.
"""
import itertools
import pytest

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.sparse.linalg import lsqr

from pylops.utils import dottest
from pylops.basicoperators import FunctionOperator

PARS_LISTS = [
        [11, 21],  # nr
        [11, 21],  # nc
        ['float32', 'float64', 'complex64', 'complex128'],  # dtypes
        ]
PARS = []
for nr, nc, dtype in itertools.product(*PARS_LISTS):
    PARS += [{'nr': nr,
              'nc': nc,
              'imag': 0 if dtype.startswith('float') else 1j,
              'dtype': dtype}]


@pytest.mark.parametrize("par", PARS)
def test_FunctionOperator(par):
    """Dot-test and inversion for FunctionOperator operator.
    """
    np.random.seed(10)
    G = np.matrix(np.random.normal(0, 1, (par['nr'], par['nc'])) +
                  np.random.normal(0, 1, (par['nr'], par['nc']))*par['imag'],
                  dtype=par['dtype'])

    def forward_f(x):
        return G @ x
    def adjoint_f(y):
        return G.H @ y

    if par['nr'] == par['nc']:
        Fop = FunctionOperator(forward_f, adjoint_f, par['nr'],
                               dtype=par['dtype'])
    else:
        Fop = FunctionOperator(forward_f, adjoint_f, par['nr'], par['nc'],
                               dtype=par['dtype'])

    assert dottest(Fop, par['nr'], par['nc'],
                   complexflag=0 if par['imag'] == 0 else 3)

    x = (np.ones(par['nc']) +
         np.ones(par['nc'])*par['imag']).astype(par['dtype'])
    y = (np.ones(par['nr']) +
         np.ones(par['nr'])*par['imag']).astype(par['dtype'])

    F_x = Fop @ x
    FH_y = Fop.H @ y

    G_x = np.squeeze(np.asarray(G @ x))
    GH_y = np.squeeze(np.asarray(G.H @ y))

    assert_array_equal(F_x, G_x)
    assert_array_equal(FH_y, GH_y)

    # Only test inversion for square or overdetermined systems
    if par['nc'] <= par['nr']:
        xlsqr = lsqr(Fop, F_x, damp=0, iter_lim=100, show=0)[0]
        assert_array_almost_equal(x, xlsqr, decimal=4)


@pytest.mark.parametrize("par", PARS)
def test_FunctionOperator_NoAdjoint(par):
    """Forward and adjoint for FunctionOperator operator where the adjoint
    is not implemented.
    """
    np.random.seed(10)
    G = np.matrix(np.random.normal(0, 1, (par['nr'], par['nc'])) +
                  np.random.normal(0, 1, (par['nr'], par['nc']))*par['imag'],
                  dtype=par['dtype'])

    def forward_f(x):
        return G @ x

    if par['nr'] == par['nc']:
        Fop = FunctionOperator(forward_f, par['nr'], dtype=par['dtype'])
    else:
        Fop = FunctionOperator(forward_f, par['nr'], par['nc'],
                               dtype=par['dtype'])

    x = (np.ones(par['nc']) +
         np.ones(par['nc'])*par['imag']).astype(par['dtype'])
    y = (np.ones(par['nr']) +
         np.ones(par['nr'])*par['imag']).astype(par['dtype'])

    F_x = Fop @ x
    G_x = np.squeeze(np.asarray(G @ x))
    assert_array_equal(F_x, G_x)

    try:
        FH_x = Fop.H @ y
        GH_x = G.H @ y
        assert_array_equal(FH_x, GH_x)
    except NotImplementedError:
        pass