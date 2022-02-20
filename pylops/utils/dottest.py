import warnings
import numpy as np

from pylops.utils.backend import get_module, to_numpy


def dottest(
    Op,
    nr=None,
    nc=None,
    rtol=1e-6,
    complexflag=0,
    raiseerror=True,
    verb=False,
    backend="numpy",
    atol=1e-21,
    tol=None,  # deprecated, use rtol
):
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
    rtol : :obj:`float`, optional
        Relative dottest tolerance
        .. versionadded:: 1.18.1
    complexflag : :obj:`bool`, optional
        Generate random vectors with

        * ``0``: Real entries for model and data

        * ``1``: Complex entries for model and real entries for data

        * ``2``: Real entries for model and complex entries for data

        * ``3``: Complex entries for model and  data
    raiseerror : :obj:`bool`, optional
        Raise error or simply return ``False`` when dottest fails
    verb : :obj:`bool`, optional
        Verbosity
    backend : :obj:`str`, optional
        Backend used for dot test computations (``numpy`` or ``cupy``). This
        parameter will be used to choose how to create the random vectors.
    atol : :obj:`float`, optional
        Absolute dottest tolerance
        .. versionadded:: 1.18.1
    tol : :obj:`float`, optional
        Dottest tolerance
        .. deprecated:: 2.0.0
            Use ``rtol`` instead.

    Raises
    ------
    ValueError
        If dot-test is not verified within chosen tolerances.

    Notes
    -----
    A dot-test is mathematical tool used in the development of numerical
    linear operators.

    More specifically, a correct implementation of forward and adjoint for
    a linear operator should verify the following *equality*
    within a numerical tolerance:

    .. math::
        (\mathbf{Op}\,\mathbf{u})^H\mathbf{v} =
        \mathbf{u}^H(\mathbf{Op}^H\mathbf{v})

    """
    if tol is not None:
        warnings.warn(
            "tol will be deprecated in version 2.0.0, use rtol instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        rtol = tol

    ncp = get_module(backend)

    if nr is None:
        nr = Op.shape[0]
    if nc is None:
        nc = Op.shape[1]

    if (nr, nc) != Op.shape:
        raise AssertionError("Provided nr and nc do not match operator shape")

    # make u and v vectors
    rdtype = np.ones(1, Op.dtype).real.dtype

    u = ncp.random.randn(nc).astype(rdtype)
    if complexflag not in (0, 2):
        u = u + 1j * ncp.random.randn(nc).astype(rdtype)

    v = ncp.random.randn(nr).astype(rdtype)
    if complexflag not in (0, 1):
        v = v + 1j * ncp.random.randn(nr).astype(rdtype)

    y = Op.matvec(u)  # Op * u
    x = Op.rmatvec(v)  # Op'* v

    if getattr(Op, "clinear", True):
        yy = ncp.vdot(y, v)  # (Op  * u)' * v
        xx = ncp.vdot(u, x)  # u' * (Op' * v)
    else:
        # Op is only R-linear, so treat complex numbers as elements of R^2
        yy = ncp.dot(y.real, v.real) + ncp.dot(y.imag, v.imag)
        xx = ncp.dot(u.real, x.real) + ncp.dot(u.imag, x.imag)

    # convert back to numpy (in case cupy arrays were used), make into a numpy
    # array and extract the first element. This is ugly but allows to handle
    # complex numbers in subsequent prints also when using cupy arrays.
    xx, yy = np.array([to_numpy(xx)])[0], np.array([to_numpy(yy)])[0]

    # evaluate if dot test passed
    passed = np.isclose(xx, yy, rtol, atol)

    # verbosity or error raising
    if (not passed and raiseerror) or verb:
        passed_status = "passed" if passed else "failed"
        msg = f"Dot test {passed_status}, v^H(Opu)={yy} - u^H(Op^Hv)={xx}"
        if not passed and raiseerror:
            raise ValueError(msg)
        else:
            print(msg)

    return passed
