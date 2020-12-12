import pytest

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from pylops import MemoizeOperator
from pylops.basicoperators import MatrixMult, VStack, HStack, Diagonal, Zero
from pylops.optimization.solver import cgls

par1 = {'ny': 11, 'nx': 11,
        'imag': 0, 'dtype':'float32'}  # square real
par1j = {'ny': 11, 'nx': 11,
         'imag': 1j, 'dtype':'complex64'} # square imag


@pytest.mark.parametrize("par", [(par1), (par1j)])
def test_memoize_evals(par):
    """Check nevals counter when same model/data vectors are inputted
    to the operator
    """
    A = np.random.normal(0, 10, (par['ny'], par['nx'])).astype('float32') + \
        par['imag'] * np.random.normal(0, 10, (par['ny'], par['nx'])).astype('float32')
    Aop = MatrixMult(A, dtype=par['dtype'])
    Amemop = MemoizeOperator(Aop, max_neval=2)

    # 1st evaluation
    Amemop * np.ones(par['nx'])
    assert Amemop.neval == 1
    # repeat 1st evaluation multiple times
    for _ in range(2):
        Amemop * np.ones(par['nx'])
    assert Amemop.neval == 1
    # 2nd evaluation
    Amemop * np.full(par['nx'], 2)  # same
    assert Amemop.neval == 2
    # 3rd evaluation (np.ones goes out of store)
    Amemop * np.full(par['nx'], 3)  # same
    assert Amemop.neval == 3
    # 4th evaluation
    Amemop * np.ones(par['nx'])
    assert Amemop.neval == 4


@pytest.mark.parametrize("par", [(par1j), ])
def test_memoize_evals(par):
    """Inversion of problem with real model and complex data, using two
    equivalent approaches: 1. complex operator enforcing the output of adjoint
    to be real, 2. joint system of equations for real and complex parts
    """
    rdtype = np.real(np.ones(1, dtype=par['dtype'])).dtype
    A = np.random.normal(0, 10, (par['ny'], par['nx'])).astype(rdtype) + \
        par['imag'] * np.random.normal(0, 10, (par['ny'], par['nx'])).astype(
            rdtype)
    Aop = MatrixMult(A, dtype=par['dtype'])
    x = np.ones(par['nx'], dtype=rdtype)
    y = Aop * x

    # Approach 1
    Aop1 = Aop.toreal(forw=False, adj=True)
    xinv1 = Aop1.div(y)
    assert_array_almost_equal(x, xinv1)

    # Approach 2
    Amop = MemoizeOperator(Aop, max_neval=10)
    Aop2 = VStack([Amop.toreal(), Amop.toimag()])
    y2 = np.concatenate([np.real(y), np.imag(y)])
    xinv2 = Aop2.div(y2)
    assert_array_almost_equal(x, xinv2)

