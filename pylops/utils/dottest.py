import numpy as np
from pylops.utils.backend import get_module


def dottest(Op, nr, nc, tol=1e-6, complexflag=0, raiseerror=True, verb=False,
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

    if complexflag in (0, 2):
        u = ncp.random.randn(nc)
    else:
        u = ncp.random.randn(nc) + 1j*ncp.random.randn(nc)

    if complexflag in (0, 1):
        v = ncp.random.randn(nr)
    else:
        v = ncp.random.randn(nr) + 1j*ncp.random.randn(nr)

    y = Op.matvec(u)   # Op * u
    x = Op.rmatvec(v)  # Op'* v

    if complexflag == 0:
        yy = ncp.dot(y, v) # (Op  * u)' * v
        xx = ncp.dot(u, x) # u' * (Op' * v)
    else:
        yy = ncp.vdot(y, v) # (Op  * u)' * v
        xx = ncp.vdot(u, x) # u' * (Op' * v)

    if complexflag == 0:
        if ncp.abs((yy - xx) / ((yy + xx + 1e-15) / 2)) < tol:
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
        checkreal = ncp.abs((ncp.real(yy) - ncp.real(xx)) /
                           ((ncp.real(yy) + ncp.real(xx)+1e-15) / 2)) < tol
        checkimag = ncp.abs((ncp.real(yy) - ncp.real(xx)) /
                           ((ncp.real(yy) + ncp.real(xx)+1e-15) / 2)) < tol
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
