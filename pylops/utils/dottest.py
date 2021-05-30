import numpy as np
from pylops.utils.backend import get_module, to_numpy


def dottest(Op, nr=None, nc=None, tol=1e-6, complexflag=0, raiseerror=True, verb=False,
            backend='numpy'):
    r"""Dot test.

    Generate random vectors :math:`\mathbf{u}` and :math:`\mathbf{v}`
    and perform dot-test to verify the validity of forward and adjoint
    operators. This test can help to detect errors in the operator
    implementation.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Linear operator to test.
    nr : :obj:`int`
        Number of rows of operator (i.e., elements in data)
    nc : :obj:`int`
        Number of columns of operator (i.e., elements in model)
    tol : :obj:`float`, optional
        Dottest tolerance
    complexflag : :obj:`bool`, optional
        generate random vectors with real (0) or complex numbers
        (1: only model, 2: only data, 3:both)
    raiseerror : :obj:`bool`, optional
        Raise error or simply return ``False`` when dottest fails
    verb : :obj:`bool`, optional
        Verbosity
    backend : :obj:`str`, optional
        Backend used for dot test computations (``numpy`` or ``cupy``). This
        parameter will be used to choose how to create the random vectors.

    Raises
    ------
    ValueError
        If dot-test is not verified within chosen tolerance.

    Notes
    -----
    A dot-test is mathematical tool used in the development of numerical
    linear operators.

    More specifically, a correct implementation of forward and adjoint for
    a linear operator should verify the following *equality*
    within a numerical tolerance:

    .. math::
        (\mathbf{Op}*\mathbf{u})^H*\mathbf{v} =
        \mathbf{u}^H*(\mathbf{Op}^H*\mathbf{v})

    """
    ncp = get_module(backend)

    if nr is None:
        nr = Op.shape[0]
    if nc is None:
        nc = Op.shape[1]

    assert (nr, nc) == Op.shape, 'Provided nr and nc do not match operator shape'

    # make u and v vectors
    if complexflag != 0:
        rdtype = np.real(np.ones(1, Op.dtype)).dtype

    if complexflag in (0, 2):
        u = ncp.random.randn(nc).astype(Op.dtype)
    else:
        u = ncp.random.randn(nc).astype(rdtype) + \
            1j * ncp.random.randn( nc).astype(rdtype)

    if complexflag in (0, 1):
        v = ncp.random.randn(nr).astype(Op.dtype)
    else:
        v = ncp.random.randn(nr).astype(rdtype) + \
            1j * ncp.random.randn(nr).astype(rdtype)

    y = Op.matvec(u)   # Op * u
    x = Op.rmatvec(v)  # Op'* v

    if getattr(Op, 'clinear', True):
        yy = ncp.vdot(y, v) # (Op  * u)' * v
        xx = ncp.vdot(u, x) # u' * (Op' * v)
    else:
        # Op is only R-linear, so treat complex numbers as elements of R^2
        yy = ncp.dot(y.real, v.real) + ncp.dot(y.imag, v.imag)
        xx = ncp.dot(u.real, x.real) + ncp.dot(u.imag, x.imag)

    # convert back to numpy (in case cupy arrays were used), make into a numpy
    # array and extract the first element. This is ugly but allows to handle
    # complex numbers in subsequent prints also when using cupy arrays.
    xx, yy = np.array([to_numpy(xx)])[0], np.array([to_numpy(yy)])[0]

    # evaluate if dot test is passed
    if complexflag == 0:
        if np.abs((yy - xx) / ((yy + xx + 1e-15) / 2)) < tol:
            if verb: print('Dot test passed, v^T(Opu)=%f - u^T(Op^Tv)=%f'
                           % (yy, xx))
            return True
        else:
            if raiseerror:
                raise ValueError('Dot test failed, v^T(Opu)=%f - u^T(Op^Tv)=%f'
                                 % (yy, xx))
            if verb: print('Dot test failed, v^T(Opu)=%f - u^T(Op^Tv)=%f'
                           % (yy, xx))
            return False
    else:
        checkreal = np.abs((np.real(yy) - np.real(xx)) /
                           ((np.real(yy) + np.real(xx)+1e-15) / 2)) < tol
        checkimag = np.abs((np.imag(yy) - np.imag(xx)) /
                           ((np.imag(yy) + np.imag(xx)+1e-15) / 2)) < tol
        if checkreal and checkimag:
            if verb:
                print('Dot test passed, v^T(Opu)=%f%+fi - u^T(Op^Tv)=%f%+fi'
                      % (yy.real, yy.imag, xx.real, xx.imag))
            return True
        else:
            if raiseerror:
                raise ValueError('Dot test failed, v^H(Opu)=%f%+fi '
                                 '- u^H(Op^Hv)=%f%+fi'
                                 % (yy.real, yy.imag, xx.real, xx.imag))
            if verb:
                print('Dot test failed, v^H(Opu)=%f%+fi - u^H(Op^Hv)=%f%+fi'
                      % (yy.real, yy.imag, xx.real, xx.imag))
            return False
