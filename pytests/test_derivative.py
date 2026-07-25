import os

if int(os.environ.get("TEST_CUPY_PYLOPS", 0)):
    import cupy as np
    from cupy.testing import assert_array_almost_equal, assert_array_equal

    backend = "cupy"
else:
    import numpy as np
    from numpy.testing import assert_array_almost_equal, assert_array_equal

    backend = "numpy"
import pytest

from pylops.basicoperators import (
    FirstDerivative,
    FirstDirectionalDerivative,
    Gradient,
    Laplacian,
    SecondDerivative,
    SecondDirectionalDerivative,
)
from pylops.utils import dottest

par1 = {
    "nz": 10,
    "ny": 30,
    "nx": 40,
    "dz": 1.0,
    "dy": 1.0,
    "dx": 1.0,
    "edge": False,
}  # even with unitary sampling
par2 = {
    "nz": 10,
    "ny": 30,
    "nx": 40,
    "dz": 0.4,
    "dy": 2.0,
    "dx": 0.5,
    "edge": False,
}  # even with non-unitary sampling
par3 = {
    "nz": 11,
    "ny": 51,
    "nx": 61,
    "dz": 1.0,
    "dy": 1.0,
    "dx": 1.0,
    "edge": False,
}  # odd with unitary sampling
par4 = {
    "nz": 11,
    "ny": 51,
    "nx": 61,
    "dz": 0.4,
    "dy": 2.0,
    "dx": 0.5,
    "edge": False,
}  # odd with non-unitary sampling
par1e = {
    "nz": 10,
    "ny": 30,
    "nx": 40,
    "dz": 1.0,
    "dy": 1.0,
    "dx": 1.0,
    "edge": True,
}  # even with unitary sampling
par2e = {
    "nz": 10,
    "ny": 30,
    "nx": 40,
    "dz": 0.4,
    "dy": 2.0,
    "dx": 0.5,
    "edge": True,
}  # even with non-unitary sampling
par3e = {
    "nz": 11,
    "ny": 51,
    "nx": 61,
    "dz": 1.0,
    "dy": 1.0,
    "dx": 1.0,
    "edge": True,
}  # odd with unitary sampling
par4e = {
    "nz": 11,
    "ny": 51,
    "nx": 61,
    "dz": 0.4,
    "dy": 2.0,
    "dx": 0.5,
    "edge": True,
}  # odd with non-unitary sampling

np.random.seed(10)


def test_FirstDerivative_invalid():
    """FirstDerivative raises NotImplementedError for invalid kind/order"""
    # invalid centered order
    with pytest.raises(NotImplementedError, match="order must be"):
        FirstDerivative(10, kind="centered", order=4)
    # invalid kind
    with pytest.raises(NotImplementedError, match="kind must be"):
        FirstDerivative(10, kind="invalid")


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1e), (par2e), (par3e), (par4e)]
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_FirstDerivative_centered(par, dtype):
    """Dot-test and forward/adjoint for FirstDerivative operator (centered stencil)"""
    np.random.seed(10)
    for order in (3, 5):
        # 1d
        D1op = FirstDerivative(
            par["nx"],
            sampling=par["dx"],
            edge=par["edge"],
            order=order,
            dtype=dtype,
        )
        assert dottest(
            D1op,
            par["nx"],
            par["nx"],
            rtol=5e-4 if dtype == np.float32 else 1e-6,
            backend=backend,
        )

        x = (par["dx"] * np.arange(par["nx"], dtype=dtype)) ** 2
        yana = 2 * par["dx"] * np.arange(par["nx"], dtype=dtype)
        y = D1op * x
        xadj = D1op.H * y
        assert y.dtype == dtype
        assert xadj.dtype == dtype
        assert_array_almost_equal(
            y[order // 2 : -order // 2], yana[order // 2 : -order // 2], decimal=1
        )

        # 2d - derivative on 1st direction
        D1op = FirstDerivative(
            (par["ny"], par["nx"]),
            axis=0,
            sampling=par["dy"],
            edge=par["edge"],
            order=order,
            dtype=dtype,
        )
        assert dottest(
            D1op,
            par["ny"] * par["nx"],
            par["ny"] * par["nx"],
            rtol=5e-4 if dtype == np.float32 else 1e-6,
            backend=backend,
        )

        x = np.outer(
            (par["dy"] * np.arange(par["ny"], dtype=dtype)) ** 2,
            np.ones(par["nx"], dtype=dtype),
        )
        yana = np.outer(
            2 * par["dy"] * np.arange(par["ny"], dtype=dtype),
            np.ones(par["nx"], dtype=dtype),
        )
        y = D1op * x
        xadj = D1op.H * y
        assert y.dtype == dtype
        assert xadj.dtype == dtype
        assert_array_almost_equal(
            y.reshape(par["ny"], par["nx"])[order // 2 : -order // 2],
            yana[order // 2 : -order // 2],
            decimal=1,
        )

        # 2d - derivative on 2nd direction
        D1op = FirstDerivative(
            (par["ny"], par["nx"]),
            axis=1,
            sampling=par["dx"],
            edge=par["edge"],
            order=order,
            dtype=dtype,
        )
        assert dottest(
            D1op,
            par["ny"] * par["nx"],
            par["ny"] * par["nx"],
            rtol=5e-4 if dtype == np.float32 else 1e-6,
            backend=backend,
        )

        yana = np.zeros((par["ny"], par["nx"]), dtype=dtype)
        y = D1op * x.ravel()
        xadj = D1op.H * y
        assert y.dtype == dtype
        assert xadj.dtype == dtype
        assert_array_almost_equal(
            y.reshape(par["ny"], par["nx"])[order // 2 : -order // 2],
            yana[order // 2 : -order // 2],
            decimal=1,
        )

        # 3d - derivative on 1st direction
        D1op = FirstDerivative(
            (par["nz"], par["ny"], par["nx"]),
            axis=0,
            sampling=par["dz"],
            edge=par["edge"],
            order=order,
            dtype=dtype,
        )
        assert dottest(
            D1op,
            par["nz"] * par["ny"] * par["nx"],
            par["nz"] * par["ny"] * par["nx"],
            rtol=5e-4 if dtype == np.float32 else 1e-6,
            backend=backend,
        )

        x = np.outer(
            (par["dz"] * np.arange(par["nz"], dtype=dtype)) ** 2,
            np.ones((par["ny"], par["nx"]), dtype=dtype),
        ).reshape(par["nz"], par["ny"], par["nx"])
        yana = np.outer(
            2 * par["dz"] * np.arange(par["nz"], dtype=dtype),
            np.ones((par["ny"], par["nx"]), dtype=dtype),
        ).reshape(par["nz"], par["ny"], par["nx"])
        y = D1op * x.ravel()
        xadj = D1op.H * y
        assert y.dtype == dtype
        assert xadj.dtype == dtype
        assert_array_almost_equal(
            y.reshape(par["nz"], par["ny"], par["nx"])[order // 2 : -order // 2],
            yana[order // 2 : -order // 2],
            decimal=1,
        )

        # 3d - derivative on 2nd direction
        D1op = FirstDerivative(
            (par["nz"], par["ny"], par["nx"]),
            axis=1,
            sampling=par["dy"],
            edge=par["edge"],
            order=order,
            dtype=dtype,
        )
        assert dottest(
            D1op,
            par["nz"] * par["ny"] * par["nx"],
            par["nz"] * par["ny"] * par["nx"],
            rtol=5e-4 if dtype == np.float32 else 1e-6,
            backend=backend,
        )

        yana = np.zeros((par["nz"], par["ny"], par["nx"]), dtype=dtype)
        y = D1op * x.ravel()
        xadj = D1op.H * y
        assert y.dtype == dtype
        assert xadj.dtype == dtype
        assert_array_almost_equal(
            y.reshape(par["nz"], par["ny"], par["nx"])[order // 2 : -order // 2],
            yana[order // 2 : -order // 2],
            decimal=1,
        )

        # 3d - derivative on 3rd direction
        D1op = FirstDerivative(
            (par["nz"], par["ny"], par["nx"]),
            axis=2,
            sampling=par["dx"],
            edge=par["edge"],
            order=order,
            dtype=dtype,
        )
        assert dottest(
            D1op,
            par["nz"] * par["ny"] * par["nx"],
            par["nz"] * par["ny"] * par["nx"],
            rtol=5e-4 if dtype == np.float32 else 1e-6,
            backend=backend,
        )

        yana = np.zeros((par["nz"], par["ny"], par["nx"]), dtype=dtype)
        y = D1op * x.ravel()
        xadj = D1op.H * y
        assert y.dtype == dtype
        assert xadj.dtype == dtype
        assert_array_almost_equal(
            y.reshape(par["nz"], par["ny"], par["nx"])[order // 2 : -order // 2],
            yana[order // 2 : -order // 2],
            decimal=1,
        )


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1e), (par2e), (par3e), (par4e)]
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_FirstDerivative_forwaback(par, dtype):
    """Dot-test for FirstDerivative operator and forward/adjoint
    (forward and backward stencils). Note that the analytical
    expression cannot be validated in this case
    """
    np.random.seed(10)
    for kind in ("forward", "backward"):
        # 1d
        D1op = FirstDerivative(
            par["nx"], sampling=par["dx"], edge=par["edge"], kind=kind, dtype=dtype
        )
        assert dottest(
            D1op,
            par["nx"],
            par["nx"],
            rtol=5e-4 if dtype == np.float32 else 1e-6,
            backend=backend,
        )

        x = (par["dx"] * np.arange(par["nx"])) ** 2
        y = D1op * x
        xadj = D1op.H * y
        assert y.dtype == dtype
        assert xadj.dtype == dtype

        # 2d - derivative on 1st direction
        D1op = FirstDerivative(
            (par["ny"], par["nx"]),
            axis=0,
            sampling=par["dy"],
            edge=par["edge"],
            kind=kind,
            dtype=dtype,
        )
        assert dottest(
            D1op,
            par["ny"] * par["nx"],
            par["ny"] * par["nx"],
            rtol=5e-4 if dtype == np.float32 else 1e-6,
            backend=backend,
        )

        x = np.outer(
            (par["dy"] * np.arange(par["ny"], dtype=dtype)) ** 2,
            np.ones(par["nx"], dtype=dtype),
        )
        y = D1op * x
        xadj = D1op.H * y
        assert y.dtype == dtype
        assert xadj.dtype == dtype

        # 2d - derivative on 2nd direction
        D1op = FirstDerivative(
            (par["ny"], par["nx"]),
            axis=1,
            sampling=par["dx"],
            edge=par["edge"],
            kind=kind,
            dtype=dtype,
        )
        assert dottest(
            D1op,
            par["ny"] * par["nx"],
            par["ny"] * par["nx"],
            rtol=5e-4 if dtype == np.float32 else 1e-6,
            backend=backend,
        )

        y = D1op * x.ravel()
        xadj = D1op.H * y
        assert y.dtype == dtype
        assert xadj.dtype == dtype

        # 3d - derivative on 1st direction
        D1op = FirstDerivative(
            (par["nz"], par["ny"], par["nx"]),
            axis=0,
            sampling=par["dz"],
            edge=par["edge"],
            kind=kind,
            dtype=dtype,
        )
        assert dottest(
            D1op,
            par["nz"] * par["ny"] * par["nx"],
            par["nz"] * par["ny"] * par["nx"],
            rtol=5e-4 if dtype == np.float32 else 1e-6,
            backend=backend,
        )

        x = np.outer(
            (par["dz"] * np.arange(par["nz"], dtype=dtype)) ** 2,
            np.ones((par["ny"], par["nx"]), dtype=dtype),
        ).reshape(par["nz"], par["ny"], par["nx"])
        y = D1op * x.ravel()
        xadj = D1op.H * y
        assert y.dtype == dtype
        assert xadj.dtype == dtype

        # 3d - derivative on 2nd direction
        D1op = FirstDerivative(
            (par["nz"], par["ny"], par["nx"]),
            axis=1,
            sampling=par["dy"],
            edge=par["edge"],
            kind=kind,
            dtype=dtype,
        )
        assert dottest(
            D1op,
            par["nz"] * par["ny"] * par["nx"],
            par["nz"] * par["ny"] * par["nx"],
            rtol=5e-4 if dtype == np.float32 else 1e-6,
            backend=backend,
        )

        y = D1op * x.ravel()
        xadj = D1op.H * y
        assert y.dtype == dtype
        assert xadj.dtype == dtype

        # 3d - derivative on 3rd direction
        D1op = FirstDerivative(
            (par["nz"], par["ny"], par["nx"]),
            axis=2,
            sampling=par["dx"],
            edge=par["edge"],
            kind=kind,
            dtype=dtype,
        )
        assert dottest(
            D1op,
            par["nz"] * par["ny"] * par["nx"],
            par["nz"] * par["ny"] * par["nx"],
            rtol=5e-4 if dtype == np.float32 else 1e-6,
            backend=backend,
        )

        y = D1op * x.ravel()
        xadj = D1op.H * y
        assert y.dtype == dtype
        assert xadj.dtype == dtype


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1e), (par2e), (par3e), (par4e)]
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_SecondDerivative_centered(par, dtype):
    """Dot-test and forward/adjoint for SecondDerivative operator (centered stencil)
    The test is based on the fact that the central stencil is exact for polynomials of
    degree 3.
    """
    np.random.seed(10)
    x = par["dx"] * np.arange(par["nx"], dtype=dtype)
    y = par["dy"] * np.arange(par["ny"], dtype=dtype)
    z = par["dz"] * np.arange(par["nz"], dtype=dtype)

    xx, yy = np.meshgrid(x, y)  # produces arrays of size (ny,nx)
    xxx, yyy, zzz = np.meshgrid(x, y, z)  # produces arrays of size (ny,nx,nz)

    # 1d
    D2op = SecondDerivative(
        par["nx"], sampling=par["dx"], edge=par["edge"], dtype=dtype
    )
    assert dottest(
        D2op,
        par["nx"],
        par["nx"],
        rtol=5e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    # polynomial f(x) = x^3, f''(x) = 6x
    f = x**3
    dfana = 6 * x
    df = D2op * f
    fadj = D2op.H * df
    assert df.dtype == dtype
    assert fadj.dtype == dtype
    assert_array_almost_equal(df[1:-1], dfana[1:-1], decimal=1)

    # 2d - derivative on 1st direction
    D2op = SecondDerivative(
        (par["ny"], par["nx"]),
        axis=0,
        sampling=par["dy"],
        edge=par["edge"],
        dtype=dtype,
    )
    assert dottest(
        D2op,
        par["ny"] * par["nx"],
        par["ny"] * par["nx"],
        rtol=5e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    # polynomial f(x,y) = y^3, f_{yy}(x,y) = 6y
    f = yy**3
    dfana = 6 * yy
    df = D2op * f.ravel()
    fadj = D2op.H * df
    assert df.dtype == dtype
    assert fadj.dtype == dtype
    assert_array_almost_equal(
        df.reshape(par["ny"], par["nx"])[1:-1, :], dfana[1:-1, :], decimal=1
    )

    # 2d - derivative on 2nd direction
    D2op = SecondDerivative(
        (par["ny"], par["nx"]),
        axis=1,
        sampling=par["dx"],
        edge=par["edge"],
        dtype=dtype,
    )
    assert dottest(
        D2op,
        par["ny"] * par["nx"],
        par["ny"] * par["nx"],
        rtol=5e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    # polynomial f(x,y) = x^3, f_{xx}(x,y) = 6x
    f = xx**3
    dfana = 6 * xx
    df = D2op * f.ravel()
    fadj = D2op.H * df
    assert df.dtype == dtype
    assert fadj.dtype == dtype
    assert_array_almost_equal(
        df.reshape(par["ny"], par["nx"])[:, 1:-1], dfana[:, 1:-1], decimal=1
    )

    # 3d - derivative on 1st direction
    D2op = SecondDerivative(
        (par["ny"], par["nx"], par["nz"]),
        axis=0,
        sampling=par["dy"],
        edge=par["edge"],
        dtype=dtype,
    )
    assert dottest(
        D2op,
        par["nz"] * par["ny"] * par["nx"],
        par["nz"] * par["ny"] * par["nx"],
        rtol=5e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    # polynomial f(x,y,z) = y^3, f_{yy}(x,y,z) = 6y
    f = yyy**3
    dfana = 6 * yyy
    df = D2op * f.ravel()
    fadj = D2op.H * df
    assert df.dtype == dtype
    assert fadj.dtype == dtype
    assert_array_almost_equal(
        df.reshape(par["ny"], par["nx"], par["nz"])[1:-1, :, :],
        dfana[1:-1, :, :],
        decimal=1,
    )

    # 3d - derivative on 2nd direction
    D2op = SecondDerivative(
        (par["ny"], par["nx"], par["nz"]),
        axis=1,
        sampling=par["dx"],
        edge=par["edge"],
        dtype=dtype,
    )
    assert dottest(
        D2op,
        par["nz"] * par["ny"] * par["nx"],
        par["nz"] * par["ny"] * par["nx"],
        rtol=5e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    # polynomial f(x,y,z) = x^3, f_{xx}(x,y,z) = 6x
    f = xxx**3
    dfana = 6 * xxx
    df = D2op * f.ravel()
    fadj = D2op.H * df
    assert df.dtype == dtype
    assert fadj.dtype == dtype
    assert_array_almost_equal(
        df.reshape(par["ny"], par["nx"], par["nz"])[:, 1:-1, :],
        dfana[:, 1:-1, :],
        decimal=1,
    )

    # 3d - derivative on 3rd direction
    D2op = SecondDerivative(
        (par["ny"], par["nx"], par["nz"]),
        axis=2,
        sampling=par["dz"],
        edge=par["edge"],
        dtype=dtype,
    )
    assert dottest(
        D2op,
        par["nz"] * par["ny"] * par["nx"],
        par["ny"] * par["nx"] * par["nz"],
        rtol=5e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    # polynomial f(x,y,z) = z^3, f_{zz}(x,y,z) = 6z
    f = zzz**3
    dfana = 6 * zzz
    df = D2op * f.ravel()
    fadj = D2op.H * df
    assert df.dtype == dtype
    assert fadj.dtype == dtype
    assert_array_almost_equal(
        df.reshape(par["ny"], par["nx"], par["nz"])[:, :, 1:-1],
        dfana[:, :, 1:-1],
        decimal=1,
    )


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1e), (par2e), (par3e), (par4e)]
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_SecondDerivative_forwaback(par, dtype):
    """Dot-test for SecondDerivative operator (forward and backward stencils).
    Note that the analytical expression cannot be validated in this case
    """
    np.random.seed(10)
    x = par["dx"] * np.arange(par["nx"], dtype=dtype)
    y = par["dy"] * np.arange(par["ny"], dtype=dtype)
    z = par["dz"] * np.arange(par["nz"], dtype=dtype)

    xx, _ = np.meshgrid(x, y)  # produces arrays of size (ny,nx)
    xxx, _, _ = np.meshgrid(x, y, z)  # produces arrays of size (ny,nx,nz)

    for _ in ("forward", "backward"):
        # 1d
        D2op = SecondDerivative(
            par["nx"],
            sampling=par["dx"],
            edge=par["edge"],
            kind="forward",
            dtype=dtype,
        )
        assert dottest(
            D2op,
            par["nx"],
            par["nx"],
            rtol=5e-4 if dtype == np.float32 else 1e-6,
            backend=backend,
        )
        y = D2op * x
        xadj = D2op.H * y
        assert y.dtype == dtype
        assert xadj.dtype == dtype

        # 2d - derivative on 1st direction
        D2op = SecondDerivative(
            (par["ny"], par["nx"]),
            axis=0,
            sampling=par["dy"],
            edge=par["edge"],
            kind="forward",
            dtype=dtype,
        )
        assert dottest(
            D2op,
            par["ny"] * par["nx"],
            par["ny"] * par["nx"],
            rtol=5e-4 if dtype == np.float32 else 1e-6,
            backend=backend,
        )
        y = D2op * xx.ravel()
        xadj = D2op.H * y
        assert y.dtype == dtype
        assert xadj.dtype == dtype

        # 2d - derivative on 2nd direction
        D2op = SecondDerivative(
            (par["ny"], par["nx"]),
            axis=1,
            sampling=par["dx"],
            edge=par["edge"],
            kind="forward",
            dtype=dtype,
        )
        assert dottest(
            D2op,
            par["ny"] * par["nx"],
            par["ny"] * par["nx"],
            rtol=5e-4 if dtype == np.float32 else 1e-6,
            backend=backend,
        )
        y = D2op * xx.ravel()
        xadj = D2op.H * y
        assert y.dtype == dtype
        assert xadj.dtype == dtype

        # 3d - derivative on 1st direction
        D2op = SecondDerivative(
            (par["ny"], par["nx"], par["nz"]),
            axis=0,
            sampling=par["dy"],
            edge=par["edge"],
            kind="forward",
            dtype=dtype,
        )
        assert dottest(
            D2op,
            par["nz"] * par["ny"] * par["nx"],
            par["nz"] * par["ny"] * par["nx"],
            rtol=5e-4 if dtype == np.float32 else 1e-6,
            backend=backend,
        )
        y = D2op * xxx.ravel()
        xadj = D2op.H * y
        assert y.dtype == dtype
        assert xadj.dtype == dtype

        # 3d - derivative on 2nd direction
        D2op = SecondDerivative(
            (par["ny"], par["nx"], par["nz"]),
            axis=1,
            sampling=par["dx"],
            edge=par["edge"],
            kind="forward",
            dtype=dtype,
        )
        assert dottest(
            D2op,
            par["nz"] * par["ny"] * par["nx"],
            par["nz"] * par["ny"] * par["nx"],
            rtol=5e-4 if dtype == np.float32 else 1e-6,
            backend=backend,
        )
        y = D2op * xxx.ravel()
        xadj = D2op.H * y
        assert y.dtype == dtype
        assert xadj.dtype == dtype

        # 3d - derivative on 3rd direction
        D2op = SecondDerivative(
            (par["ny"], par["nx"], par["nz"]),
            axis=2,
            sampling=par["dz"],
            edge=par["edge"],
            kind="forward",
            dtype=dtype,
        )
        assert dottest(
            D2op,
            par["nz"] * par["ny"] * par["nx"],
            par["ny"] * par["nx"] * par["nz"],
            rtol=5e-4 if dtype == np.float32 else 1e-6,
            backend=backend,
        )
        y = D2op * xxx.ravel()
        xadj = D2op.H * y
        assert y.dtype == dtype
        assert xadj.dtype == dtype


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1e), (par2e), (par3e), (par4e)]
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_Laplacian(par, dtype):
    """Dot-test for Laplacian operator"""
    np.random.seed(10)
    xx = np.ones((par["ny"], par["nx"]), dtype=dtype)
    xxx = np.ones((par["nz"], par["ny"], par["nx"]), dtype=dtype)

    # 2d - symmetrical
    Dlapop = Laplacian(
        (par["ny"], par["nx"]),
        axes=(0, 1),
        weights=(1, 1),
        sampling=(par["dy"], par["dx"]),
        edge=par["edge"],
        dtype=dtype,
    )
    assert dottest(
        Dlapop,
        par["ny"] * par["nx"],
        par["ny"] * par["nx"],
        rtol=5e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )
    y = Dlapop * xx.ravel()
    xadj = Dlapop.H * y
    assert y.dtype == dtype
    assert xadj.dtype == dtype

    # 2d - asymmetrical
    Dlapop = Laplacian(
        (par["ny"], par["nx"]),
        axes=(0, 1),
        weights=(1, 2),
        sampling=(par["dy"], par["dx"]),
        edge=par["edge"],
        dtype=dtype,
    )
    assert dottest(
        Dlapop,
        par["ny"] * par["nx"],
        par["ny"] * par["nx"],
        rtol=5e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )
    y = Dlapop * xx.ravel()
    xadj = Dlapop.H * y
    assert y.dtype == dtype
    assert xadj.dtype == dtype

    # 3d - symmetrical on 1st and 2nd direction
    Dlapop = Laplacian(
        (par["nz"], par["ny"], par["nx"]),
        axes=(0, 1),
        weights=(1, 1),
        sampling=(par["dy"], par["dx"]),
        edge=par["edge"],
        dtype=dtype,
    )
    assert dottest(
        Dlapop,
        par["nz"] * par["ny"] * par["nx"],
        par["nz"] * par["ny"] * par["nx"],
        rtol=5e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )
    y = Dlapop * xxx.ravel()
    xadj = Dlapop.H * y
    assert y.dtype == dtype
    assert xadj.dtype == dtype

    # 3d - symmetrical on 1st and 2nd direction
    Dlapop = Laplacian(
        (par["nz"], par["ny"], par["nx"]),
        axes=(0, 1),
        weights=(1, 1),
        sampling=(par["dy"], par["dx"]),
        edge=par["edge"],
        dtype=dtype,
    )
    assert dottest(
        Dlapop,
        par["nz"] * par["ny"] * par["nx"],
        par["nz"] * par["ny"] * par["nx"],
        rtol=5e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )
    y = Dlapop * xxx.ravel()
    xadj = Dlapop.H * y
    assert y.dtype == dtype
    assert xadj.dtype == dtype

    # 3d - symmetrical on all directions
    Dlapop = Laplacian(
        (par["nz"], par["ny"], par["nx"]),
        axes=(0, 1, 2),
        weights=(1, 1, 1),
        sampling=(par["dz"], par["dx"], par["dx"]),
        edge=par["edge"],
        dtype=dtype,
    )
    assert dottest(
        Dlapop,
        par["nz"] * par["ny"] * par["nx"],
        par["nz"] * par["ny"] * par["nx"],
        rtol=5e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )
    y = Dlapop * xxx.ravel()
    xadj = Dlapop.H * y
    assert y.dtype == dtype
    assert xadj.dtype == dtype


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1e), (par2e), (par3e), (par4e)]
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_Gradient(par, dtype):
    """Dot-test for Gradient operator"""
    np.random.seed(10)
    xx = np.ones((par["ny"], par["nx"]), dtype=dtype)
    xxx = np.ones((par["nz"], par["ny"], par["nx"]), dtype=dtype)

    for kind in ("forward", "centered", "backward"):
        # 2d
        Gop = Gradient(
            (par["ny"], par["nx"]),
            sampling=(par["dy"], par["dx"]),
            edge=par["edge"],
            kind=kind,
            dtype=dtype,
        )
        assert dottest(
            Gop,
            2 * par["ny"] * par["nx"],
            par["ny"] * par["nx"],
            rtol=5e-4 if dtype == np.float32 else 1e-6,
            backend=backend,
        )
        y = Gop * xx.ravel()
        xadj = Gop.H * y
        assert y.dtype == dtype
        assert xadj.dtype == dtype

        # 3d
        Gop = Gradient(
            (par["nz"], par["ny"], par["nx"]),
            sampling=(par["dz"], par["dy"], par["dx"]),
            edge=par["edge"],
            kind=kind,
            dtype=dtype,
        )
        assert dottest(
            Gop,
            3 * par["nz"] * par["ny"] * par["nx"],
            par["nz"] * par["ny"] * par["nx"],
            rtol=5e-4 if dtype == np.float32 else 1e-6,
            backend=backend,
        )
        y = Gop * xxx.ravel()
        xadj = Gop.H * y
        assert y.dtype == dtype
        assert xadj.dtype == dtype


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1e), (par2e), (par3e), (par4e)]
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_FirstDirectionalDerivative(par, dtype):
    """Dot-test for FirstDirectionalDerivative operator"""
    np.random.seed(10)
    xx = np.ones((par["ny"], par["nx"]), dtype=dtype)
    xxx = np.ones((par["nz"], par["ny"], par["nx"]), dtype=dtype)

    for kind in ("forward", "centered", "backward"):
        # 2d
        Fdop = FirstDirectionalDerivative(
            (par["ny"], par["nx"]),
            v=(np.sqrt(2.0) / 2.0 * np.ones(2)).astype(dtype),
            sampling=(par["dy"], par["dx"]),
            edge=par["edge"],
            kind=kind,
            dtype=dtype,
        )
        assert dottest(
            Fdop,
            par["ny"] * par["nx"],
            par["ny"] * par["nx"],
            rtol=5e-4 if dtype == np.float32 else 1e-6,
            backend=backend,
        )
        y = Fdop * xx.ravel()
        xadj = Fdop.H * y
        assert y.dtype == dtype
        assert xadj.dtype == dtype

        # 3d
        Fdop = FirstDirectionalDerivative(
            (par["nz"], par["ny"], par["nx"]),
            v=(np.ones(3) / np.sqrt(3)).astype(dtype),
            sampling=(par["dz"], par["dy"], par["dx"]),
            edge=par["edge"],
            kind=kind,
            dtype=dtype,
        )
        assert dottest(
            Fdop,
            par["nz"] * par["ny"] * par["nx"],
            par["nz"] * par["ny"] * par["nx"],
            rtol=5e-4 if dtype == np.float32 else 1e-6,
            backend=backend,
        )
        y = Fdop * xxx.ravel()
        xadj = Fdop.H * y
        assert y.dtype == dtype
        assert xadj.dtype == dtype


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1e), (par2e), (par3e), (par4e)]
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_SecondDirectionalDerivative(par, dtype):
    """Dot-test for test_SecondDirectionalDerivative operator"""
    np.random.seed(10)
    xx = np.ones((par["ny"], par["nx"]), dtype=dtype)
    xxx = np.ones((par["nz"], par["ny"], par["nx"]), dtype=dtype)

    # 2d
    Fdop = SecondDirectionalDerivative(
        (par["ny"], par["nx"]),
        v=np.sqrt(2.0) / 2.0 * np.ones(2),
        sampling=(par["dy"], par["dx"]),
        edge=par["edge"],
        dtype=dtype,
    )
    assert dottest(
        Fdop,
        par["ny"] * par["nx"],
        par["ny"] * par["nx"],
        rtol=5e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )
    y = Fdop * xx.ravel()
    xadj = Fdop.H * y
    assert y.dtype == dtype
    assert xadj.dtype == dtype

    # 3d
    Fdop = SecondDirectionalDerivative(
        (par["nz"], par["ny"], par["nx"]),
        v=np.ones(3) / np.sqrt(3),
        sampling=(par["dz"], par["dy"], par["dx"]),
        edge=par["edge"],
        dtype=dtype,
    )
    assert dottest(
        Fdop,
        par["nz"] * par["ny"] * par["nx"],
        par["nz"] * par["ny"] * par["nx"],
        rtol=5e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )
    y = Fdop * xxx.ravel()
    xadj = Fdop.H * y
    assert y.dtype == dtype
    assert xadj.dtype == dtype


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1e), (par2e), (par3e), (par4e)]
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_SecondDirectionalDerivative_verticalderivative(par, dtype):
    """Compare vertical derivative for SecondDirectionalDerivative operator
    and SecondDerivative
    """
    np.random.seed(10)
    Fop = FirstDerivative((par["ny"], par["nx"]), axis=0, edge=par["edge"], dtype=dtype)
    F2op = Fop.H * Fop

    F2dop = SecondDirectionalDerivative(
        (par["ny"], par["nx"]), v=np.array([1, 0]), edge=par["edge"], dtype=dtype
    )

    x = np.random.normal(0.0, 1.0, (par["ny"], par["nx"]))
    assert_array_equal(-F2op * x.ravel(), F2dop * x.ravel())
