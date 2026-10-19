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
par1s = {"ny": 11, "nx": 11, "imag": 0, "dtype": "float32"}  # square real (fp32)
par2s = {
    "ny": 21,
    "nx": 11,
    "imag": 0,
    "dtype": "float32",
}  # overdetermined real (fp32)
par1j = {"ny": 11, "nx": 11, "imag": 1j, "dtype": "complex128"}  # square complex
par2j = {
    "ny": 21,
    "nx": 11,
    "imag": 1j,
    "dtype": "complex128",
}  # overdetermined complex
par3 = {"ny": 11, "nx": 21, "imag": 0, "dtype": "float64"}  # underdetermined real
par3s = {
    "ny": 11,
    "nx": 21,
    "imag": 0,
    "dtype": "float64",
}  # underdetermined real (fp32)

par_reg_real = {**par1, "order": 2}
par_reg_complex = {**par1j, "order": 2}

np.random.seed(10)


@pytest.mark.parametrize("par", [par_reg_real, par_reg_complex])
@pytest.mark.parametrize("axis_dtype", ["int64", "float64"])
def test_Regression_axis_dtype(par, axis_dtype):
    """Preserve coefficient dtype with integer or real sample positions."""
    np.random.seed(10)
    order = par["order"]
    t = np.arange(par["ny"], dtype=axis_dtype)
    op = Regression(t, order=order, dtype=par["dtype"])
    coefficients = (
        np.random.normal(size=order + 1)
        + par["imag"] * np.random.normal(size=order + 1)
    ).astype(par["dtype"])
    design = np.stack([t.astype(par["dtype"]) ** i for i in range(order + 1)], axis=1)
    expected = design @ coefficients
    actual = op @ coefficients
    assert_array_almost_equal(actual, expected)
    assert actual.dtype == expected.dtype
    assert_array_almost_equal(op.apply(t, coefficients), expected)
    assert_array_equal(op.taxis, t)
    assert dottest(
        op, par["ny"], order + 1, complexflag=3 if par["imag"] else 0, backend=backend
    )
    recovered = lsqr(op, actual, niter=30, atol=1e-12, btol=1e-12)[0]
    assert_array_almost_equal(recovered, coefficients, decimal=5)


@pytest.mark.parametrize("par", [(par1), (par2), (par1s), (par2s)])
def test_Regression(par):
    """Dot-test, inversion and apply for Regression operator"""
    np.random.seed(10)

    order = 4
    t = np.arange(par["ny"], dtype=par["dtype"])
    LRop = Regression(t, order=order, dtype=par["dtype"])
    assert dottest(
        LRop,
        par["ny"],
        order + 1,
        rtol=1e-4 if par["dtype"] == np.float32 else 1e-6,
        backend=backend,
    )

    x = np.array([1.0, 2.0, 0.0, 3.0, -1.0], dtype=par["dtype"])

    # forward vs apply method
    y = LRop * x
    y1 = LRop.apply(t, x)
    assert y.dtype == par["dtype"]
    assert_array_almost_equal(y, y1, decimal=3)

    # inversion
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


@pytest.mark.parametrize("par", [(par1), (par2), (par1s), (par2s)])
def test_LinearRegression(par):
    """Dot-test and inversion for LinearRegression operator"""
    np.random.seed(10)

    t = np.arange(par["ny"], dtype=par["dtype"])
    LRop = LinearRegression(t, dtype=par["dtype"])
    assert dottest(
        LRop,
        par["ny"],
        2,
        rtol=1e-4 if par["dtype"] == np.float32 else 1e-6,
        backend=backend,
    )

    x = np.array([1.0, 2.0], dtype=par["dtype"])

    # forward vs apply method
    y = LRop * x
    y1 = LRop.apply(t, x)
    assert y.dtype == par["dtype"]
    assert_array_almost_equal(y, y1, decimal=3)

    # inversion
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


@pytest.mark.parametrize("par", [(par1), (par2), (par1s), (par2s), (par1j), (par2j)])
def test_MatrixMult(par):
    """Dot-test and inversion for MatrixMult operator"""
    np.random.seed(10)
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype

    G = np.random.normal(0, 10, (par["ny"], par["nx"])).astype(dtype) + par[
        "imag"
    ] * np.random.normal(0, 10, (par["ny"], par["nx"])).astype(dtype)
    Gop = MatrixMult(G, dtype=par["dtype"])
    assert dottest(
        Gop,
        par["ny"],
        par["nx"],
        complexflag=0 if par["imag"] == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    x = np.ones(par["nx"], dtype=dtype) + par["imag"] * np.ones(par["nx"], dtype=dtype)
    y = Gop * x
    xlsqr = lsqr(
        Gop,
        y,
        x0=np.zeros_like(x),
        damp=1e-20,
        niter=300,
        atol=1e-8,
        btol=1e-8,
        show=0,
    )[0]
    assert y.dtype == par["dtype"]
    assert_array_almost_equal(x, xlsqr, decimal=4)


@pytest.mark.parametrize("par", [(par1), (par2), (par1s), (par2s), (par1j), (par2j)])
def test_MatrixMult_sparse(par):
    """Dot-test and inversion for MatrixMult operator using sparse
    matrix
    """
    np.random.seed(10)
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype

    G = rand(par["ny"], par["nx"], density=0.75).astype(dtype) + par["imag"] * rand(
        par["ny"], par["nx"], density=0.75
    ).astype(dtype)

    Gop = MatrixMult(G, dtype=par["dtype"])
    assert dottest(
        Gop,
        par["ny"],
        par["nx"],
        complexflag=0 if par["imag"] == 1 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    x = np.ones(par["nx"], dtype=dtype) + par["imag"] * np.ones(par["nx"], dtype=dtype)
    y = Gop * x
    xlsqr = lsqr(
        Gop,
        y,
        x0=np.zeros_like(x),
        damp=1e-20,
        niter=300,
        atol=1e-8,
        btol=1e-8,
        show=0,
    )[0]
    assert y.dtype == par["dtype"]
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


@pytest.mark.parametrize("par", [(par1), (par2), (par1s), (par2s), (par1j), (par2j)])
def test_MatrixMult_repeated(par):
    """Dot-test and inversion for test_MatrixMult operator repeated
    along another dimension
    """
    np.random.seed(10)
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype

    G = np.random.normal(0, 10, (par["ny"], par["nx"])).astype(dtype) + par[
        "imag"
    ] * np.random.normal(0, 10, (par["ny"], par["nx"])).astype(dtype)
    Gop = MatrixMult(G, otherdims=5, dtype=par["dtype"])
    assert dottest(
        Gop,
        par["ny"] * 5,
        par["nx"] * 5,
        complexflag=0 if par["imag"] == 1 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    x = (
        np.ones((par["nx"], 5), dtype=dtype)
        + par["imag"] * np.ones((par["nx"], 5), dtype=dtype)
    ).ravel()
    y = Gop * x
    xlsqr = lsqr(
        Gop,
        y,
        x0=np.zeros_like(x),
        damp=1e-20,
        niter=300,
        atol=1e-8,
        btol=1e-8,
        show=0,
    )[0]
    assert y.dtype == par["dtype"]
    assert_array_almost_equal(x, xlsqr, decimal=4)


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par1s), (par2s), (par1j), (par2j), (par3), (par3s)]
)
def test_Identity_inplace(par):
    """Dot-test, forward and adjoint for Identity operator"""
    np.random.seed(10)
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype

    Iop = Identity(par["ny"], par["nx"], dtype=par["dtype"], inplace=True)
    assert dottest(
        Iop,
        par["ny"],
        par["nx"],
        complexflag=0 if par["imag"] == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    x = np.ones(par["nx"], dtype=dtype) + par["imag"] * np.ones(par["nx"], dtype=dtype)
    y = Iop * x
    x1 = Iop.H * y

    assert y.dtype == par["dtype"]
    assert x1.dtype == par["dtype"]
    assert_array_almost_equal(
        x[: min(par["ny"], par["nx"])], y[: min(par["ny"], par["nx"])], decimal=4
    )
    assert_array_almost_equal(
        x[: min(par["ny"], par["nx"])], x1[: min(par["ny"], par["nx"])], decimal=4
    )


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par1s), (par2s), (par1j), (par2j), (par3), (par3s)]
)
def test_Identity_noinplace(par):
    """Dot-test, forward and adjoint for Identity operator (not in place)"""
    np.random.seed(10)
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype

    Iop = Identity(par["ny"], par["nx"], dtype=par["dtype"], inplace=False)
    assert dottest(
        Iop,
        par["ny"],
        par["nx"],
        complexflag=0 if par["imag"] == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    x = np.ones(par["nx"], dtype=dtype) + par["imag"] * np.ones(par["nx"], dtype=dtype)
    y = Iop * x
    x1 = Iop.H * y

    assert y.dtype == par["dtype"]
    assert x1.dtype == par["dtype"]
    assert_array_almost_equal(
        x[: min(par["ny"], par["nx"])], y[: min(par["ny"], par["nx"])], decimal=4
    )
    assert_array_almost_equal(
        x[: min(par["ny"], par["nx"])], x1[: min(par["ny"], par["nx"])], decimal=4
    )

    # change value in x and check it doesn't change in y
    x[0] = 10
    assert x[0] != y[0]


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par1s), (par2s), (par1j), (par2j), (par3), (par3s)]
)
def test_Zero(par):
    """Dot-test, forward and adjoint for Zero operator"""
    np.random.seed(10)
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype

    Zop = Zero(par["ny"], par["nx"], dtype=par["dtype"])
    assert dottest(
        Zop,
        par["ny"],
        par["nx"],
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    x = np.ones(par["nx"], dtype=dtype) + par["imag"] * np.ones(par["nx"], dtype=dtype)
    y = Zop * x
    x1 = Zop.H * y

    assert y.dtype == par["dtype"]
    assert x1.dtype == par["dtype"]
    assert_array_almost_equal(y, np.zeros(par["ny"]))
    assert_array_almost_equal(x1, np.zeros(par["nx"]))


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par1s), (par2s), (par1j), (par2j), (par3), (par3s)]
)
def test_Flip1D(par):
    """Dot-test, forward and adjoint for Flip operator on 1d signal"""
    np.random.seed(10)
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype

    x = np.arange(par["ny"], dtype=dtype) + par["imag"] * np.arange(
        par["ny"], dtype=dtype
    )

    Fop = Flip(par["ny"], dtype=par["dtype"])
    assert dottest(
        Fop,
        par["ny"],
        par["ny"],
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    y = Fop * x
    xadj = Fop.H * y
    assert y.dtype == par["dtype"]
    assert xadj.dtype == par["dtype"]
    assert_array_equal(x, xadj)


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par1s), (par2s), (par1j), (par2j), (par3), (par3s)]
)
@pytest.mark.parametrize("axis", [0, 1])
def test_Flip2D(par, axis):
    """Dot-test, forward and adjoint for Flip operator on 2d signal"""
    np.random.seed(10)
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype

    x = {}
    x["0"] = np.outer(
        np.arange(par["ny"], dtype=dtype), np.ones(par["nx"], dtype=dtype)
    ) + par["imag"] * np.outer(
        np.arange(par["ny"], dtype=dtype), np.ones(par["nx"], dtype=dtype)
    )
    x["1"] = np.outer(
        np.ones(par["ny"], dtype=dtype), np.arange(par["nx"], dtype=dtype)
    ) + par["imag"] * np.outer(
        np.ones(par["ny"], dtype=dtype), np.arange(par["nx"], dtype=dtype)
    )

    Fop = Flip(
        (par["ny"], par["nx"]),
        axis=axis,
        dtype=par["dtype"],
    )
    assert dottest(
        Fop,
        par["ny"] * par["nx"],
        par["ny"] * par["nx"],
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    y = Fop * x[str(axis)].ravel()
    xadj = Fop.H * y.ravel()
    xadj = xadj.reshape(par["ny"], par["nx"])
    assert y.dtype == par["dtype"]
    assert xadj.dtype == par["dtype"]
    assert_array_equal(x[str(axis)], xadj)


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par1s), (par2s), (par1j), (par2j), (par3), (par3s)]
)
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_Flip3D(par, axis):
    """Dot-test, forward and adjoint for Flip operator on 3d signal"""
    np.random.seed(10)
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype

    x = {}
    x["0"] = np.outer(
        np.arange(par["ny"], dtype=dtype), np.ones(par["nx"], dtype=dtype)
    )[:, :, np.newaxis] * np.ones(par["nx"], dtype=dtype) + par["imag"] * np.outer(
        np.arange(par["ny"], dtype=dtype), np.ones(par["nx"], dtype=dtype)
    )[:, :, np.newaxis] * np.ones(par["nx"], dtype=dtype)

    x["1"] = np.outer(
        np.ones(par["ny"], dtype=dtype), np.arange(par["nx"], dtype=dtype)
    )[:, :, np.newaxis] * np.ones(par["nx"], dtype=dtype) + par["imag"] * np.outer(
        np.ones(par["ny"], dtype=dtype), np.arange(par["nx"], dtype=dtype)
    )[:, :, np.newaxis] * np.ones(par["nx"], dtype=dtype)
    x["2"] = np.outer(np.ones(par["ny"], dtype=dtype), np.ones(par["nx"], dtype=dtype))[
        :, :, np.newaxis
    ] * np.arange(par["nx"], dtype=dtype) + par["imag"] * np.outer(
        np.ones(par["ny"], dtype=dtype), np.ones(par["nx"], dtype=dtype)
    )[:, :, np.newaxis] * np.arange(par["nx"], dtype=dtype)

    Fop = Flip(
        (par["ny"], par["nx"], par["nx"]),
        axis=axis,
        dtype=par["dtype"],
    )
    assert dottest(
        Fop,
        par["ny"] * par["nx"] * par["nx"],
        par["ny"] * par["nx"] * par["nx"],
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    y = Fop * x[str(axis)].ravel()
    xadj = Fop.H * y.ravel()
    xadj = xadj.reshape(par["ny"], par["nx"], par["nx"])
    assert y.dtype == par["dtype"]
    assert xadj.dtype == par["dtype"]
    assert_array_equal(x[str(axis)], xadj)


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par1s), (par2s), (par1j), (par2j), (par3), (par3s)]
)
def test_Symmetrize1D(par):
    """Dot-test, forward and inverse for Symmetrize operator on 1d signal"""
    np.random.seed(10)
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype

    x = np.arange(par["ny"], dtype=dtype) + par["imag"] * np.arange(
        par["ny"], dtype=dtype
    )

    Sop = Symmetrize(par["ny"], dtype=par["dtype"])
    dottest(
        Sop,
        par["ny"] * 2 - 1,
        par["ny"],
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    y = Sop * x
    xinv = Sop / y
    assert y.dtype == par["dtype"]
    assert_array_almost_equal(x, xinv, decimal=3)


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par1s), (par2s), (par1j), (par2j), (par3), (par3s)]
)
@pytest.mark.parametrize("axis", [0, 1])
def test_Symmetrize2D(par, axis):
    """Dot-test, forward and inverse for Symmetrize operator on 2d signal"""
    np.random.seed(10)
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype

    x = {}
    x["0"] = np.outer(
        np.arange(par["ny"], dtype=dtype), np.ones(par["nx"], dtype=dtype)
    ) + par["imag"] * np.outer(
        np.arange(par["ny"], dtype=dtype), np.ones(par["nx"], dtype=dtype)
    )
    x["1"] = np.outer(
        np.ones(par["ny"], dtype=dtype), np.arange(par["nx"], dtype=dtype)
    ) + par["imag"] * np.outer(
        np.ones(par["ny"], dtype=dtype), np.arange(par["nx"], dtype=dtype)
    )

    Sop = Symmetrize(
        (par["ny"], par["nx"]),
        axis=axis,
        dtype=par["dtype"],
    )
    y = Sop * x[str(axis)].ravel()
    assert dottest(
        Sop,
        y.size,
        par["ny"] * par["nx"],
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    xinv = Sop / y
    assert y.dtype == par["dtype"]
    assert_array_almost_equal(x[str(axis)].ravel(), xinv, decimal=3)


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par1s), (par2s), (par1j), (par2j), (par3), (par3s)]
)
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_Symmetrize3D(par, axis):
    """Dot-test, forward and adjoint for Symmetrize operator on 3d signal"""
    np.random.seed(10)
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype

    x = {}
    x["0"] = np.outer(
        np.arange(par["ny"], dtype=dtype), np.ones(par["nx"], dtype=dtype)
    )[:, :, np.newaxis] * np.ones(par["nx"], dtype=dtype) + par["imag"] * np.outer(
        np.arange(par["ny"], dtype=dtype), np.ones(par["nx"], dtype=dtype)
    )[:, :, np.newaxis] * np.ones(par["nx"], dtype=dtype)

    x["1"] = np.outer(
        np.ones(par["ny"], dtype=dtype), np.arange(par["nx"], dtype=dtype)
    )[:, :, np.newaxis] * np.ones(par["nx"], dtype=dtype) + par["imag"] * np.outer(
        np.ones(par["ny"], dtype=dtype), np.arange(par["nx"], dtype=dtype)
    )[:, :, np.newaxis] * np.ones(par["nx"], dtype=dtype)
    x["2"] = np.outer(np.ones(par["ny"], dtype=dtype), np.ones(par["nx"], dtype=dtype))[
        :, :, np.newaxis
    ] * np.arange(par["nx"], dtype=dtype) + par["imag"] * np.outer(
        np.ones(par["ny"], dtype=dtype), np.ones(par["nx"], dtype=dtype)
    )[:, :, np.newaxis] * np.arange(par["nx"], dtype=dtype)

    Sop = Symmetrize(
        (par["ny"], par["nx"], par["nx"]),
        axis=axis,
        dtype=par["dtype"],
    )
    y = Sop * x[str(axis)].ravel()
    assert dottest(
        Sop,
        y.size,
        par["ny"] * par["nx"] * par["nx"],
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    xinv = Sop / y
    assert y.dtype == par["dtype"]
    assert_array_almost_equal(x[str(axis)].ravel(), xinv, decimal=3)


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par1s), (par2s), (par1j), (par2j), (par3), (par3s)]
)
def test_Roll1D(par):
    """Dot-test, forward and adjoint for Roll operator on 1d signal"""
    np.random.seed(10)
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype

    x = np.arange(par["ny"], dtype=dtype) + par["imag"] * np.arange(
        par["ny"], dtype=dtype
    )

    Rop = Roll(par["ny"], shift=2, dtype=par["dtype"])
    assert dottest(
        Rop,
        par["ny"],
        par["ny"],
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    y = Rop * x
    xadj = Rop.H * y
    assert y.dtype == par["dtype"]
    assert xadj.dtype == par["dtype"]
    assert_array_almost_equal(x, xadj, decimal=3)


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par1s), (par2s), (par1j), (par2j), (par3), (par3s)]
)
@pytest.mark.parametrize("axis", [0, 1])
def test_Roll2D(par, axis):
    """Dot-test, forward and inverse for Roll operator on 2d signal"""
    np.random.seed(10)
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype

    x = {}
    x["0"] = np.outer(
        np.arange(par["ny"], dtype=dtype), np.ones(par["nx"], dtype=dtype)
    ) + par["imag"] * np.outer(
        np.arange(par["ny"], dtype=dtype), np.ones(par["nx"], dtype=dtype)
    )
    x["1"] = np.outer(
        np.ones(par["ny"], dtype=dtype), np.arange(par["nx"], dtype=dtype)
    ) + par["imag"] * np.outer(
        np.ones(par["ny"], dtype=dtype), np.arange(par["nx"], dtype=dtype)
    )

    Rop = Roll(
        (par["ny"], par["nx"]),
        axis=axis,
        shift=-2,
        dtype=par["dtype"],
    )
    y = Rop * x[str(axis)].ravel()
    assert dottest(
        Rop,
        par["ny"] * par["nx"],
        par["ny"] * par["nx"],
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    xadj = Rop.H * y
    assert y.dtype == par["dtype"]
    assert xadj.dtype == par["dtype"]
    assert_array_almost_equal(x[str(axis)].ravel(), xadj, decimal=3)


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par1s), (par2s), (par1j), (par2j), (par3), (par3s)]
)
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_Roll3D(par, axis):
    """Dot-test, forward and adjoint for Roll operator on 3d signal"""
    np.random.seed(10)
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype

    x = {}
    x["0"] = np.outer(
        np.arange(par["ny"], dtype=dtype), np.ones(par["nx"], dtype=dtype)
    )[:, :, np.newaxis] * np.ones(par["nx"], dtype=dtype) + par["imag"] * np.outer(
        np.arange(par["ny"], dtype=dtype), np.ones(par["nx"], dtype=dtype)
    )[:, :, np.newaxis] * np.ones(par["nx"], dtype=dtype)

    x["1"] = np.outer(
        np.ones(par["ny"], dtype=dtype), np.arange(par["nx"], dtype=dtype)
    )[:, :, np.newaxis] * np.ones(par["nx"], dtype=dtype) + par["imag"] * np.outer(
        np.ones(par["ny"], dtype=dtype), np.arange(par["nx"], dtype=dtype)
    )[:, :, np.newaxis] * np.ones(par["nx"], dtype=dtype)
    x["2"] = np.outer(np.ones(par["ny"], dtype=dtype), np.ones(par["nx"], dtype=dtype))[
        :, :, np.newaxis
    ] * np.arange(par["nx"], dtype=dtype) + par["imag"] * np.outer(
        np.ones(par["ny"], dtype=dtype), np.ones(par["nx"], dtype=dtype)
    )[:, :, np.newaxis] * np.arange(par["nx"], dtype=dtype)

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
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    xinv = Rop.H * y
    assert y.dtype == par["dtype"]
    assert_array_almost_equal(x[str(axis)].ravel(), xinv, decimal=3)


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par1s), (par2s), (par1j), (par2j), (par3), (par3s)]
)
@pytest.mark.parametrize("axis", [0, 1])
def test_Sum2D(par, axis):
    """Dot-test for Sum operator on 2d signal"""
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype
    dim_d = [par["ny"], par["nx"]]
    dim_d.pop(axis)
    Sop = Sum(dims=(par["ny"], par["nx"]), axis=axis, dtype=par["dtype"])
    assert dottest(
        Sop,
        npp.prod(dim_d),
        par["ny"] * par["nx"],
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    x = np.arange(par["nx"] * par["ny"], dtype=dtype) + par["imag"] * np.arange(
        par["nx"] * par["ny"], dtype=dtype
    )
    y = Sop * x.ravel()
    xadj = Sop.H * y
    assert y.dtype == par["dtype"]
    assert xadj.dtype == par["dtype"]


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

    with pytest.raises(ValueError, match="Operators have conflicting forceflat"):
        Sop_True * Sop_False.H

    Sop = Sop_True * Sop_None.H
    assert Sop.forceflat is True

    Sop = Sop_False * Sop_None.H
    assert Sop.forceflat is False

    Sop = Sop_None * Sop_None.H
    assert Sop.forceflat is None


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par1s), (par2s), (par1j), (par2j), (par3), (par3s)]
)
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_Sum3D(par, axis):
    """Dot-test, forward and adjoint for Sum operator on 3d signal"""
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype
    dim_d = [par["ny"], par["nx"], par["nx"]]
    dim_d.pop(axis)
    Sop = Sum(dims=(par["ny"], par["nx"], par["nx"]), axis=axis, dtype=par["dtype"])
    assert dottest(
        Sop,
        npp.prod(dim_d),
        par["ny"] * par["nx"] * par["nx"],
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )
    x = np.arange(par["ny"] * par["nx"] * par["nx"], dtype=dtype) + par[
        "imag"
    ] * np.arange(par["ny"] * par["nx"] * par["nx"], dtype=dtype)
    y = Sop * x.ravel()
    xadj = Sop.H * y
    assert y.dtype == par["dtype"]
    assert xadj.dtype == par["dtype"]


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


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par1s), (par2s), (par1j), (par2j), (par3), (par3s)]
)
def test_Conj(par):
    """Dot-test, forward and adjoint for Conj operator"""
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype
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
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    np.random.seed(10)
    x = np.random.randn(par["nx"] * par["ny"]).astype(dtype) + par[
        "imag"
    ] * np.random.randn(par["nx"] * par["ny"]).astype(dtype)
    y = Cop * x
    xadj = Cop.H * y
    assert y.dtype == par["dtype"]
    assert xadj.dtype == par["dtype"]
    assert_array_equal(x, xadj)
    assert_array_equal(y, np.conj(x))
    assert_array_equal(xadj, np.conj(y))


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par1s), (par2s), (par1j), (par2j), (par3), (par3s)]
)
def test_ToCupy(par):
    """Forward and adjoint for ToCupy operator (checking that it works also
    when cupy is not available)
    """
    np.random.seed(10)
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype

    Top = ToCupy(par["nx"], dtype=par["dtype"])

    x = np.random.randn(par["nx"]).astype(dtype) + par["imag"] * np.random.randn(
        par["nx"]
    ).astype(dtype)
    y = Top * x
    xadj = Top.H * y
    assert y.dtype == par["dtype"]
    assert xadj.dtype == par["dtype"]
    assert_array_equal(x, xadj)
    assert_array_equal(y, np.asarray(x))
