import os

if int(os.environ.get("TEST_CUPY_PYLOPS", 0)):
    import cupy as np
    from cupy.testing import assert_array_almost_equal, assert_array_equal
    from cupyx.scipy.sparse import rand

    backend = "cupy"
else:
    import numpy as np
    from numpy.testing import assert_array_almost_equal, assert_array_equal
    from scipy.sparse import rand

    backend = "numpy"
import numpy as npp
import pytest

from pylops.basicoperators import (
    Conj,
    Flip,
    Identity,
    Imag,
    LinearRegression,
    MatrixMult,
    Real,
    Regression,
    Roll,
    Sum,
    Symmetrize,
    ToCupy,
    Zero,
)
from pylops.optimization.basic import lsqr
from pylops.utils import dottest

par1 = {"ny": 11, "nx": 11, "imag": 0, "dtype": "float64"}  # square real
par2 = {"ny": 21, "nx": 11, "imag": 0, "dtype": "float64"}  # overdetermined real
par1j = {"ny": 11, "nx": 11, "imag": 1j, "dtype": "complex128"}  # square complex
par2j = {
    "ny": 21,
    "nx": 11,
    "imag": 1j,
    "dtype": "complex128",
}  # overdetermined complex
par3 = {"ny": 11, "nx": 21, "imag": 0, "dtype": "float64"}  # underdetermined real

np.random.seed(10)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_Regression(par):
    """Dot-test, inversion and apply for Regression operator"""
    np.random.seed(10)
    order = 4
    t = np.arange(par["ny"], dtype=np.float64)
    LRop = Regression(t, order=order, dtype=par["dtype"])
    assert dottest(LRop, par["ny"], order + 1, backend=backend)

    x = np.array([1.0, 2.0, 0.0, 3.0, -1.0], dtype=np.float64)
    xlsqr = lsqr(
        LRop,
        LRop * x,
        x0=np.zeros_like(x),
        damp=1e-10,
        niter=300,
        atol=0,
        btol=0,
        conlim=np.inf,
        show=0,
    )[0]
    assert_array_almost_equal(x, xlsqr, decimal=3)

    y = LRop * x
    y1 = LRop.apply(t, x)
    assert_array_almost_equal(y, y1, decimal=3)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_LinearRegression(par):
    """Dot-test and inversion for LinearRegression operator"""
    np.random.seed(10)
    t = np.arange(par["ny"], dtype=np.float32)
    LRop = LinearRegression(t, dtype=par["dtype"])
    assert dottest(LRop, par["ny"], 2, backend=backend)

    x = np.array([1.0, 2.0], dtype=np.float64)
    xlsqr = lsqr(
        LRop,
        LRop * x,
        x0=np.zeros_like(x),
        damp=1e-10,
        niter=300,
        atol=0,
        btol=0,
        conlim=np.inf,
        show=0,
    )[0]
    assert_array_almost_equal(x, xlsqr, decimal=3)

    y = LRop * x
    y1 = LRop.apply(t, x)
    assert_array_almost_equal(y, y1, decimal=3)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_MatrixMult(par):
    """Dot-test and inversion for MatrixMult operator"""
    np.random.seed(10)
    G = np.random.normal(0, 10, (par["ny"], par["nx"])).astype("float32") + par[
        "imag"
    ] * np.random.normal(0, 10, (par["ny"], par["nx"])).astype("float32")
    Gop = MatrixMult(G, dtype=par["dtype"])
    assert dottest(
        Gop,
        par["ny"],
        par["nx"],
        complexflag=0 if par["imag"] == 0 else 3,
        backend=backend,
    )

    x = np.ones(par["nx"]) + par["imag"] * np.ones(par["nx"])
    xlsqr = lsqr(
        Gop,
        Gop * x,
        x0=np.zeros_like(x),
        damp=1e-20,
        niter=300,
        atol=1e-8,
        btol=1e-8,
        show=0,
    )[0]
    assert_array_almost_equal(x, xlsqr, decimal=4)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_MatrixMult_sparse(par):
    """Dot-test and inversion for MatrixMult operator using sparse
    matrix
    """
    np.random.seed(10)
    G = rand(par["ny"], par["nx"], density=0.75).astype("float32") + par["imag"] * rand(
        par["ny"], par["nx"], density=0.75
    ).astype("float32")

    Gop = MatrixMult(G, dtype=par["dtype"])
    assert dottest(
        Gop,
        par["ny"],
        par["nx"],
        complexflag=0 if par["imag"] == 1 else 3,
        backend=backend,
    )

    x = np.ones(par["nx"]) + par["imag"] * np.ones(par["nx"])
    xlsqr = lsqr(
        Gop,
        Gop * x,
        x0=np.zeros_like(x),
        damp=1e-20,
        niter=300,
        atol=1e-8,
        btol=1e-8,
        show=0,
    )[0]
    assert_array_almost_equal(x, xlsqr, decimal=4)


@pytest.mark.parametrize("par", [(par1j), (par2j)])
def test_MatrixMult_complexcast(par):
    """Automatic upcasting of MatrixMult operator dtype based on complex
    matrix
    """
    np.random.seed(10)
    G = rand(par["ny"], par["nx"], density=0.75).astype("float32") + par["imag"] * rand(
        par["ny"], par["nx"], density=0.75
    ).astype("float32")

    Gop = MatrixMult(G, dtype="float32")
    assert Gop.dtype == "complex64"


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_MatrixMult_repeated(par):
    """Dot-test and inversion for test_MatrixMult operator repeated
    along another dimension
    """
    np.random.seed(10)
    G = np.random.normal(0, 10, (par["ny"], par["nx"])).astype("float32") + par[
        "imag"
    ] * np.random.normal(0, 10, (par["ny"], par["nx"])).astype("float32")
    Gop = MatrixMult(G, otherdims=5, dtype=par["dtype"])
    assert dottest(
        Gop,
        par["ny"] * 5,
        par["nx"] * 5,
        complexflag=0 if par["imag"] == 1 else 3,
        backend=backend,
    )

    x = (np.ones((par["nx"], 5)) + par["imag"] * np.ones((par["nx"], 5))).ravel()
    xlsqr = lsqr(
        Gop,
        Gop * x,
        x0=np.zeros_like(x),
        damp=1e-20,
        niter=300,
        atol=1e-8,
        btol=1e-8,
        show=0,
    )[0]
    assert_array_almost_equal(x, xlsqr, decimal=4)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j), (par3)])
def test_Identity_inplace(par):
    """Dot-test, forward and adjoint for Identity operator"""
    np.random.seed(10)
    Iop = Identity(par["ny"], par["nx"], dtype=par["dtype"], inplace=True)
    assert dottest(
        Iop,
        par["ny"],
        par["nx"],
        complexflag=0 if par["imag"] == 0 else 3,
        backend=backend,
    )

    x = np.ones(par["nx"]) + par["imag"] * np.ones(par["nx"])
    y = Iop * x
    x1 = Iop.H * y

    assert_array_almost_equal(
        x[: min(par["ny"], par["nx"])], y[: min(par["ny"], par["nx"])], decimal=4
    )
    assert_array_almost_equal(
        x[: min(par["ny"], par["nx"])], x1[: min(par["ny"], par["nx"])], decimal=4
    )


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j), (par3)])
def test_Identity_noinplace(par):
    """Dot-test, forward and adjoint for Identity operator (not in place)"""
    np.random.seed(10)
    Iop = Identity(par["ny"], par["nx"], dtype=par["dtype"], inplace=False)
    assert dottest(
        Iop,
        par["ny"],
        par["nx"],
        complexflag=0 if par["imag"] == 0 else 3,
        backend=backend,
    )

    x = np.ones(par["nx"]) + par["imag"] * np.ones(par["nx"])
    y = Iop * x
    x1 = Iop.H * y

    assert_array_almost_equal(
        x[: min(par["ny"], par["nx"])], y[: min(par["ny"], par["nx"])], decimal=4
    )
    assert_array_almost_equal(
        x[: min(par["ny"], par["nx"])], x1[: min(par["ny"], par["nx"])], decimal=4
    )

    # change value in x and check it doesn't change in y
    x[0] = 10
    assert x[0] != y[0]


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j), (par3)])
def test_Zero(par):
    """Dot-test, forward and adjoint for Zero operator"""
    np.random.seed(10)
    Zop = Zero(par["ny"], par["nx"], dtype=par["dtype"])
    assert dottest(Zop, par["ny"], par["nx"], backend=backend)

    x = np.ones(par["nx"]) + par["imag"] * np.ones(par["nx"])
    y = Zop * x
    x1 = Zop.H * y

    assert_array_almost_equal(y, np.zeros(par["ny"]))
    assert_array_almost_equal(x1, np.zeros(par["nx"]))


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_Flip1D(par):
    """Dot-test, forward and adjoint for Flip operator on 1d signal"""
    np.random.seed(10)
    x = np.arange(par["ny"]) + par["imag"] * np.arange(par["ny"])

    Fop = Flip(par["ny"], dtype=par["dtype"])
    assert dottest(Fop, par["ny"], par["ny"], backend=backend)

    y = Fop * x
    xadj = Fop.H * y
    assert_array_equal(x, xadj)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_Flip2D(par):
    """Dot-test, forward and adjoint for Flip operator on 2d signal"""
    np.random.seed(10)
    x = {}
    x["0"] = np.outer(np.arange(par["ny"]), np.ones(par["nx"])) + par[
        "imag"
    ] * np.outer(np.arange(par["ny"]), np.ones(par["nx"]))
    x["1"] = np.outer(np.ones(par["ny"]), np.arange(par["nx"])) + par[
        "imag"
    ] * np.outer(np.ones(par["ny"]), np.arange(par["nx"]))

    for axis in [0, 1]:
        Fop = Flip(
            (par["ny"], par["nx"]),
            axis=axis,
            dtype=par["dtype"],
        )
        assert dottest(
            Fop, par["ny"] * par["nx"], par["ny"] * par["nx"], backend=backend
        )

        y = Fop * x[str(axis)].ravel()
        xadj = Fop.H * y.ravel()
        xadj = xadj.reshape(par["ny"], par["nx"])
        assert_array_equal(x[str(axis)], xadj)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_Flip3D(par):
    """Dot-test, forward and adjoint for Flip operator on 3d signal"""
    np.random.seed(10)
    x = {}
    x["0"] = np.outer(np.arange(par["ny"]), np.ones(par["nx"]))[
        :, :, np.newaxis
    ] * np.ones(par["nx"]) + par["imag"] * np.outer(
        np.arange(par["ny"]), np.ones(par["nx"])
    )[
        :, :, np.newaxis
    ] * np.ones(
        par["nx"]
    )

    x["1"] = np.outer(np.ones(par["ny"]), np.arange(par["nx"]))[
        :, :, np.newaxis
    ] * np.ones(par["nx"]) + par["imag"] * np.outer(
        np.ones(par["ny"]), np.arange(par["nx"])
    )[
        :, :, np.newaxis
    ] * np.ones(
        par["nx"]
    )
    x["2"] = np.outer(np.ones(par["ny"]), np.ones(par["nx"]))[
        :, :, np.newaxis
    ] * np.arange(par["nx"]) + par["imag"] * np.outer(
        np.ones(par["ny"]), np.ones(par["nx"])
    )[
        :, :, np.newaxis
    ] * np.arange(
        par["nx"]
    )

    for axis in [0, 1, 2]:
        Fop = Flip(
            (par["ny"], par["nx"], par["nx"]),
            axis=axis,
            dtype=par["dtype"],
        )
        assert dottest(
            Fop,
            par["ny"] * par["nx"] * par["nx"],
            par["ny"] * par["nx"] * par["nx"],
            backend=backend,
        )

        y = Fop * x[str(axis)].ravel()
        xadj = Fop.H * y.ravel()
        xadj = xadj.reshape(par["ny"], par["nx"], par["nx"])
        assert_array_equal(x[str(axis)], xadj)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j), (par3)])
def test_Symmetrize1D(par):
    """Dot-test, forward and inverse for Symmetrize operator on 1d signal"""
    np.random.seed(10)
    x = np.arange(par["ny"]) + par["imag"] * np.arange(par["ny"])

    Sop = Symmetrize(par["ny"], dtype=par["dtype"])
    dottest(Sop, par["ny"] * 2 - 1, par["ny"], verb=True, backend=backend)

    y = Sop * x
    xinv = Sop / y
    assert_array_almost_equal(x, xinv, decimal=3)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j), (par3)])
def test_Symmetrize2D(par):
    """Dot-test, forward and inverse for Symmetrize operator on 2d signal"""
    np.random.seed(10)
    x = {}
    x["0"] = np.outer(np.arange(par["ny"]), np.ones(par["nx"])) + par[
        "imag"
    ] * np.outer(np.arange(par["ny"]), np.ones(par["nx"]))
    x["1"] = np.outer(np.ones(par["ny"]), np.arange(par["nx"])) + par[
        "imag"
    ] * np.outer(np.ones(par["ny"]), np.arange(par["nx"]))

    for axis in [0, 1]:
        Sop = Symmetrize(
            (par["ny"], par["nx"]),
            axis=axis,
            dtype=par["dtype"],
        )
        y = Sop * x[str(axis)].ravel()
        assert dottest(Sop, y.size, par["ny"] * par["nx"], backend=backend)

        xinv = Sop / y
        assert_array_almost_equal(x[str(axis)].ravel(), xinv, decimal=3)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j), (par3)])
def test_Symmetrize3D(par):
    """Dot-test, forward and adjoint for Symmetrize operator on 3d signal"""
    np.random.seed(10)
    x = {}
    x["0"] = np.outer(np.arange(par["ny"]), np.ones(par["nx"]))[
        :, :, np.newaxis
    ] * np.ones(par["nx"]) + par["imag"] * np.outer(
        np.arange(par["ny"]), np.ones(par["nx"])
    )[
        :, :, np.newaxis
    ] * np.ones(
        par["nx"]
    )

    x["1"] = np.outer(np.ones(par["ny"]), np.arange(par["nx"]))[
        :, :, np.newaxis
    ] * np.ones(par["nx"]) + par["imag"] * np.outer(
        np.ones(par["ny"]), np.arange(par["nx"])
    )[
        :, :, np.newaxis
    ] * np.ones(
        par["nx"]
    )
    x["2"] = np.outer(np.ones(par["ny"]), np.ones(par["nx"]))[
        :, :, np.newaxis
    ] * np.arange(par["nx"]) + par["imag"] * np.outer(
        np.ones(par["ny"]), np.ones(par["nx"])
    )[
        :, :, np.newaxis
    ] * np.arange(
        par["nx"]
    )

    for axis in [0, 1, 2]:
        Sop = Symmetrize(
            (par["ny"], par["nx"], par["nx"]),
            axis=axis,
            dtype=par["dtype"],
        )
        y = Sop * x[str(axis)].ravel()
        assert dottest(Sop, y.size, par["ny"] * par["nx"] * par["nx"], backend=backend)

        xinv = Sop / y
        assert_array_almost_equal(x[str(axis)].ravel(), xinv, decimal=3)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j), (par3)])
def test_Roll1D(par):
    """Dot-test, forward and adjoint for Roll operator on 1d signal"""
    np.random.seed(10)
    x = np.arange(par["ny"]) + par["imag"] * np.arange(par["ny"])

    Rop = Roll(par["ny"], shift=2, dtype=par["dtype"])
    assert dottest(Rop, par["ny"], par["ny"], backend=backend)

    y = Rop * x
    xadj = Rop.H * y
    assert_array_almost_equal(x, xadj, decimal=3)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j), (par3)])
def test_Roll2D(par):
    """Dot-test, forward and inverse for Roll operator on 2d signal"""
    np.random.seed(10)
    x = {}
    x["0"] = np.outer(np.arange(par["ny"]), np.ones(par["nx"])) + par[
        "imag"
    ] * np.outer(np.arange(par["ny"]), np.ones(par["nx"]))
    x["1"] = np.outer(np.ones(par["ny"]), np.arange(par["nx"])) + par[
        "imag"
    ] * np.outer(np.ones(par["ny"]), np.arange(par["nx"]))

    for axis in [0, 1]:
        Rop = Roll(
            (par["ny"], par["nx"]),
            axis=axis,
            shift=-2,
            dtype=par["dtype"],
        )
        y = Rop * x[str(axis)].ravel()
        assert dottest(
            Rop, par["ny"] * par["nx"], par["ny"] * par["nx"], backend=backend
        )

        xadj = Rop.H * y
        assert_array_almost_equal(x[str(axis)].ravel(), xadj, decimal=3)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j), (par3)])
def test_Roll3D(par):
    """Dot-test, forward and adjoint for Roll operator on 3d signal"""
    np.random.seed(10)
    x = {}
    x["0"] = np.outer(np.arange(par["ny"]), np.ones(par["nx"]))[
        :, :, np.newaxis
    ] * np.ones(par["nx"]) + par["imag"] * np.outer(
        np.arange(par["ny"]), np.ones(par["nx"])
    )[
        :, :, np.newaxis
    ] * np.ones(
        par["nx"]
    )

    x["1"] = np.outer(np.ones(par["ny"]), np.arange(par["nx"]))[
        :, :, np.newaxis
    ] * np.ones(par["nx"]) + par["imag"] * np.outer(
        np.ones(par["ny"]), np.arange(par["nx"])
    )[
        :, :, np.newaxis
    ] * np.ones(
        par["nx"]
    )
    x["2"] = np.outer(np.ones(par["ny"]), np.ones(par["nx"]))[
        :, :, np.newaxis
    ] * np.arange(par["nx"]) + par["imag"] * np.outer(
        np.ones(par["ny"]), np.ones(par["nx"])
    )[
        :, :, np.newaxis
    ] * np.arange(
        par["nx"]
    )

    for axis in [0, 1, 2]:
        Rop = Roll(
            (par["ny"], par["nx"], par["nx"]),
            axis=axis,
            shift=3,
            dtype=par["dtype"],
        )
        y = Rop * x[str(axis)].ravel()
        assert dottest(
            Rop,
            par["ny"] * par["nx"] * par["nx"],
            par["ny"] * par["nx"] * par["nx"],
            backend=backend,
        )

        xinv = Rop.H * y
        assert_array_almost_equal(x[str(axis)].ravel(), xinv, decimal=3)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j), (par3)])
def test_Sum2D(par):
    """Dot-test for Sum operator on 2d signal"""
    for axis in [0, 1]:
        dim_d = [par["ny"], par["nx"]]
        dim_d.pop(axis)
        Sop = Sum(dims=(par["ny"], par["nx"]), axis=axis, dtype=par["dtype"])
        assert dottest(Sop, npp.prod(dim_d), par["ny"] * par["nx"], backend=backend)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j), (par3)])
def test_Sum2D_forceflat(par):
    """Dot-test for Sum operator on 2d signal with forceflat"""
    np.random.seed(10)
    flat_dimsd = par["ny"]
    flat_dims = par["ny"] * par["nx"]
    x = np.random.randn(flat_dims) + par["imag"] * np.random.randn(flat_dims)

    Sop_True = Sum((par["ny"], par["nx"]), axis=-1, forceflat=True)
    y = Sop_True @ x
    xadj = Sop_True.H @ y
    assert y.shape == (flat_dimsd,)
    assert xadj.shape == (flat_dims,)

    Sop_None = Sum((par["ny"], par["nx"]), axis=-1)
    y = Sop_None @ x
    xadj = Sop_None.H @ y
    assert y.shape == (par["ny"],)
    assert xadj.shape == (par["ny"], par["nx"])

    Sop_False = Sum((par["ny"], par["nx"]), axis=-1, forceflat=False)
    y = Sop_False @ x
    xadj = Sop_False.H @ y
    assert y.shape == (par["ny"],)
    assert xadj.shape == (par["ny"], par["nx"])

    with pytest.raises(ValueError):
        Sop_True * Sop_False.H

    Sop = Sop_True * Sop_None.H
    assert Sop.forceflat is True

    Sop = Sop_False * Sop_None.H
    assert Sop.forceflat is False

    Sop = Sop_None * Sop_None.H
    assert Sop.forceflat is None


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j), (par3)])
def test_Sum3D(par):
    """Dot-test, forward and adjoint for Sum operator on 3d signal"""
    for axis in [0, 1, 2]:
        dim_d = [par["ny"], par["nx"], par["nx"]]
        dim_d.pop(axis)
        Sop = Sum(dims=(par["ny"], par["nx"], par["nx"]), axis=axis, dtype=par["dtype"])
        assert dottest(
            Sop, npp.prod(dim_d), par["ny"] * par["nx"] * par["nx"], backend=backend
        )


@pytest.mark.parametrize("par", [(par1j), (par2j)])
def test_Real(par):
    """Dot-test, forward and adjoint for Real operator"""
    Rop = Real(dims=(par["ny"], par["nx"]), dtype=par["dtype"])
    if np.dtype(par["dtype"]).kind == "c":
        complexflag = 3
    else:
        complexflag = 0
    assert dottest(
        Rop,
        par["ny"] * par["nx"],
        par["ny"] * par["nx"],
        complexflag=complexflag,
        backend=backend,
    )

    np.random.seed(10)
    x = np.random.randn(par["nx"] * par["ny"]) + par["imag"] * np.random.randn(
        par["nx"] * par["ny"]
    )
    y = Rop * x
    assert_array_equal(y, np.real(x))
    y = np.random.randn(par["nx"] * par["ny"]) + par["imag"] * np.random.randn(
        par["nx"] * par["ny"]
    )
    x = Rop.H * y
    assert_array_equal(x, np.real(y) + 0j)


@pytest.mark.parametrize("par", [(par1j), (par2j)])
def test_Imag(par):
    """Dot-test, forward and adjoint for Imag operator"""
    Iop = Imag(dims=(par["ny"], par["nx"]), dtype=par["dtype"])
    if np.dtype(par["dtype"]).kind == "c":
        complexflag = 3
    else:
        complexflag = 0
    assert dottest(
        Iop,
        par["ny"] * par["nx"],
        par["ny"] * par["nx"],
        complexflag=complexflag,
        backend=backend,
    )

    np.random.seed(10)
    x = np.random.randn(par["nx"] * par["ny"]) + par["imag"] * np.random.randn(
        par["nx"] * par["ny"]
    )
    y = Iop * x
    assert_array_equal(y, np.imag(x))
    y = np.random.randn(par["nx"] * par["ny"]) + par["imag"] * np.random.randn(
        par["nx"] * par["ny"]
    )
    x = Iop.H * y
    if np.dtype(par["dtype"]).kind == "c":
        assert_array_equal(x, 0 + 1j * np.real(y))
    else:
        assert_array_equal(x, 0)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j), (par3)])
def test_Conj(par):
    """Dot-test, forward and adjoint for Conj operator"""
    Cop = Conj(dims=(par["ny"], par["nx"]), dtype=par["dtype"])
    if np.dtype(par["dtype"]).kind == "c":
        complexflag = 3
    else:
        complexflag = 0
    assert dottest(
        Cop,
        par["ny"] * par["nx"],
        par["ny"] * par["nx"],
        complexflag=complexflag,
        backend=backend,
    )

    np.random.seed(10)
    x = np.random.randn(par["nx"] * par["ny"]) + par["imag"] * np.random.randn(
        par["nx"] * par["ny"]
    )
    y = Cop * x
    xadj = Cop.H * y
    assert_array_equal(x, xadj)
    assert_array_equal(y, np.conj(x))
    assert_array_equal(xadj, np.conj(y))


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j), (par3)])
def test_ToCupy(par):
    """Forward and adjoint for ToCupy operator (checking that it works also
    when cupy is not available)
    """
    Top = ToCupy(par["nx"], dtype=par["dtype"])

    np.random.seed(10)
    x = npp.random.randn(par["nx"]) + par["imag"] * npp.random.randn(par["nx"])
    y = Top * x
    xadj = Top.H * y
    assert_array_equal(x, xadj)
    assert_array_equal(y, np.asarray(x))
