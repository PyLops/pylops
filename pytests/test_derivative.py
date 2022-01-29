import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

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


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1e), (par2e), (par3e), (par4e)]
)
def test_FirstDerivative_centered(par):
    """Dot-test and forward for FirstDerivative operator (centered stencil)"""
    # 1d
    D1op = FirstDerivative(
        par["nx"], sampling=par["dx"], edge=par["edge"], dtype="float32"
    )
    assert dottest(D1op, par["nx"], par["nx"], tol=1e-3)

    x = (par["dx"] * np.arange(par["nx"])) ** 2
    yana = 2 * par["dx"] * np.arange(par["nx"])
    y = D1op * x
    assert_array_almost_equal(y[1:-1], yana[1:-1], decimal=1)

    # 2d - derivative on 1st direction
    D1op = FirstDerivative(
        par["ny"] * par["nx"],
        dims=(par["ny"], par["nx"]),
        dir=0,
        sampling=par["dy"],
        edge=par["edge"],
        dtype="float32",
    )
    assert dottest(D1op, par["ny"] * par["nx"], par["ny"] * par["nx"], tol=1e-3)

    x = np.outer((par["dy"] * np.arange(par["ny"])) ** 2, np.ones(par["nx"]))
    yana = np.outer(2 * par["dy"] * np.arange(par["ny"]), np.ones(par["nx"]))
    y = D1op * x.ravel()
    y = y.reshape(par["ny"], par["nx"])
    assert_array_almost_equal(y[1:-1], yana[1:-1], decimal=1)

    # 2d - derivative on 2nd direction
    D1op = FirstDerivative(
        par["ny"] * par["nx"],
        dims=(par["ny"], par["nx"]),
        dir=1,
        sampling=par["dx"],
        edge=par["edge"],
        dtype="float32",
    )
    assert dottest(D1op, par["ny"] * par["nx"], par["ny"] * par["nx"], tol=1e-3)

    x = np.outer((par["dy"] * np.arange(par["ny"])) ** 2, np.ones(par["nx"]))
    yana = np.zeros((par["ny"], par["nx"]))
    y = D1op * x.ravel()
    y = y.reshape(par["ny"], par["nx"])
    assert_array_almost_equal(y[1:-1], yana[1:-1], decimal=1)

    # 3d - derivative on 1st direction
    D1op = FirstDerivative(
        par["nz"] * par["ny"] * par["nx"],
        dims=(par["nz"], par["ny"], par["nx"]),
        dir=0,
        sampling=par["dz"],
        edge=par["edge"],
        dtype="float32",
    )
    assert dottest(
        D1op,
        par["nz"] * par["ny"] * par["nx"],
        par["nz"] * par["ny"] * par["nx"],
        tol=1e-3,
    )

    x = np.outer(
        (par["dz"] * np.arange(par["nz"])) ** 2, np.ones((par["ny"], par["nx"]))
    ).reshape(par["nz"], par["ny"], par["nx"])
    yana = np.outer(
        2 * par["dz"] * np.arange(par["nz"]), np.ones((par["ny"], par["nx"]))
    ).reshape(par["nz"], par["ny"], par["nx"])
    y = D1op * x.ravel()
    y = y.reshape(par["nz"], par["ny"], par["nx"])
    assert_array_almost_equal(y[1:-1], yana[1:-1], decimal=1)

    # 3d - derivative on 2nd direction
    D1op = FirstDerivative(
        par["nz"] * par["ny"] * par["nx"],
        dims=(par["nz"], par["ny"], par["nx"]),
        dir=1,
        sampling=par["dy"],
        edge=par["edge"],
        dtype="float32",
    )
    assert dottest(
        D1op,
        par["nz"] * par["ny"] * par["nx"],
        par["nz"] * par["ny"] * par["nx"],
        tol=1e-3,
    )

    x = np.outer(
        (par["dz"] * np.arange(par["nz"])) ** 2, np.ones((par["ny"], par["nx"]))
    ).reshape(par["nz"], par["ny"], par["nx"])
    yana = np.zeros((par["nz"], par["ny"], par["nx"]))
    y = D1op * x.ravel()
    y = y.reshape(par["nz"], par["ny"], par["nx"])
    assert_array_almost_equal(y[1:-1], yana[1:-1], decimal=1)

    # 3d - derivative on 3rd direction
    D1op = FirstDerivative(
        par["nz"] * par["ny"] * par["nx"],
        dims=(par["nz"], par["ny"], par["nx"]),
        dir=2,
        sampling=par["dx"],
        edge=par["edge"],
        dtype="float32",
    )
    assert dottest(
        D1op,
        par["nz"] * par["ny"] * par["nx"],
        par["nz"] * par["ny"] * par["nx"],
        tol=1e-3,
    )

    yana = np.zeros((par["nz"], par["ny"], par["nx"]))
    y = D1op * x.ravel()
    y = y.reshape(par["nz"], par["ny"], par["nx"])
    assert_array_almost_equal(y[1:-1], yana[1:-1], decimal=1)


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1e), (par2e), (par3e), (par4e)]
)
def test_FirstDerivative_forwaback(par):
    """Dot-test for FirstDerivative operator (forward and backward
    stencils). Note that the analytical expression cannot be validated in this
    case
    """
    for kind in ("forward", "backward"):
        # 1d
        D1op = FirstDerivative(
            par["nx"], sampling=par["dx"], edge=par["edge"], kind=kind, dtype="float32"
        )
        assert dottest(D1op, par["nx"], par["nx"], tol=1e-3)

        # 2d - derivative on 1st direction
        D1op = FirstDerivative(
            par["ny"] * par["nx"],
            dims=(par["ny"], par["nx"]),
            dir=0,
            sampling=par["dy"],
            edge=par["edge"],
            kind=kind,
            dtype="float32",
        )
        assert dottest(D1op, par["ny"] * par["nx"], par["ny"] * par["nx"], tol=1e-3)

        # 2d - derivative on 2nd direction
        D1op = FirstDerivative(
            par["ny"] * par["nx"],
            dims=(par["ny"], par["nx"]),
            dir=1,
            sampling=par["dx"],
            edge=par["edge"],
            kind=kind,
            dtype="float32",
        )
        assert dottest(D1op, par["ny"] * par["nx"], par["ny"] * par["nx"], tol=1e-3)

        # 3d - derivative on 1st direction
        D1op = FirstDerivative(
            par["nz"] * par["ny"] * par["nx"],
            dims=(par["nz"], par["ny"], par["nx"]),
            dir=0,
            sampling=par["dz"],
            edge=par["edge"],
            kind=kind,
            dtype="float32",
        )
        assert dottest(
            D1op,
            par["nz"] * par["ny"] * par["nx"],
            par["nz"] * par["ny"] * par["nx"],
            tol=1e-3,
        )

        # 3d - derivative on 2nd direction
        D1op = FirstDerivative(
            par["nz"] * par["ny"] * par["nx"],
            dims=(par["nz"], par["ny"], par["nx"]),
            dir=1,
            sampling=par["dy"],
            edge=par["edge"],
            kind=kind,
            dtype="float32",
        )
        assert dottest(
            D1op,
            par["nz"] * par["ny"] * par["nx"],
            par["nz"] * par["ny"] * par["nx"],
            tol=1e-3,
        )

        # 3d - derivative on 3rd direction
        D1op = FirstDerivative(
            par["nz"] * par["ny"] * par["nx"],
            dims=(par["nz"], par["ny"], par["nx"]),
            dir=2,
            sampling=par["dx"],
            edge=par["edge"],
            kind=kind,
            dtype="float32",
        )
        assert dottest(
            D1op,
            par["nz"] * par["ny"] * par["nx"],
            par["nz"] * par["ny"] * par["nx"],
            tol=1e-3,
        )


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1e), (par2e), (par3e), (par4e)]
)
def test_SecondDerivative_centered(par):
    """Dot-test and forward for SecondDerivative operator (centered stencil)
    The test is based on the fact that the central stencil is exact for polynomials of
    degree 3.
    """

    x = par["dx"] * np.arange(par["nx"])
    y = par["dy"] * np.arange(par["ny"])
    z = par["dz"] * np.arange(par["nz"])

    xx, yy = np.meshgrid(x, y)  # produces arrays of size (ny,nx)
    xxx, yyy, zzz = np.meshgrid(x, y, z)  # produces arrays of size (ny,nx,nz)

    # 1d
    D2op = SecondDerivative(
        par["nx"], sampling=par["dx"], edge=par["edge"], dtype="float32"
    )
    assert dottest(D2op, par["nx"], par["nx"], tol=1e-3)

    # polynomial f(x) = x^3, f''(x) = 6x
    f = x ** 3
    dfana = 6 * x
    df = D2op * f
    assert_array_almost_equal(df[1:-1], dfana[1:-1], decimal=1)

    # 2d - derivative on 1st direction
    D2op = SecondDerivative(
        par["ny"] * par["nx"],
        dims=(par["ny"], par["nx"]),
        dir=0,
        sampling=par["dy"],
        edge=par["edge"],
        dtype="float32",
    )

    assert dottest(D2op, par["ny"] * par["nx"], par["ny"] * par["nx"], tol=1e-3)

    # polynomial f(x,y) = y^3, f_{yy}(x,y) = 6y
    f = yy ** 3
    dfana = 6 * yy
    df = D2op * f.ravel()
    df = df.reshape(par["ny"], par["nx"])
    assert_array_almost_equal(df[1:-1, :], dfana[1:-1, :], decimal=1)

    # 2d - derivative on 2nd direction
    D2op = SecondDerivative(
        par["ny"] * par["nx"],
        dims=(par["ny"], par["nx"]),
        dir=1,
        sampling=par["dx"],
        edge=par["edge"],
        dtype="float32",
    )

    assert dottest(D2op, par["ny"] * par["nx"], par["ny"] * par["nx"], tol=1e-3)

    # polynomial f(x,y) = x^3, f_{xx}(x,y) = 6x
    f = xx ** 3
    dfana = 6 * xx
    df = D2op * f.ravel()
    df = df.reshape(par["ny"], par["nx"])
    assert_array_almost_equal(df[:, 1:-1], dfana[:, 1:-1], decimal=1)

    # 3d - derivative on 1st direction
    D2op = SecondDerivative(
        par["nz"] * par["ny"] * par["nx"],
        dims=(par["ny"], par["nx"], par["nz"]),
        dir=0,
        sampling=par["dy"],
        edge=par["edge"],
        dtype="float32",
    )

    assert dottest(
        D2op,
        par["nz"] * par["ny"] * par["nx"],
        par["nz"] * par["ny"] * par["nx"],
        tol=1e-3,
    )

    # polynomial f(x,y,z) = y^3, f_{yy}(x,y,z) = 6y
    f = yyy ** 3
    dfana = 6 * yyy
    df = D2op * f.ravel()
    df = df.reshape(par["ny"], par["nx"], par["nz"])

    assert_array_almost_equal(df[1:-1, :, :], dfana[1:-1, :, :], decimal=1)

    # 3d - derivative on 2nd direction
    D2op = SecondDerivative(
        par["nz"] * par["ny"] * par["nx"],
        dims=(par["ny"], par["nx"], par["nz"]),
        dir=1,
        sampling=par["dx"],
        edge=par["edge"],
        dtype="float32",
    )

    assert dottest(
        D2op,
        par["nz"] * par["ny"] * par["nx"],
        par["nz"] * par["ny"] * par["nx"],
        tol=1e-3,
    )

    # polynomial f(x,y,z) = x^3, f_{xx}(x,y,z) = 6x
    f = xxx ** 3
    dfana = 6 * xxx
    df = D2op * f.ravel()
    df = df.reshape(par["ny"], par["nx"], par["nz"])

    assert_array_almost_equal(df[:, 1:-1, :], dfana[:, 1:-1, :], decimal=1)

    # 3d - derivative on 3rd direction
    D2op = SecondDerivative(
        par["nz"] * par["ny"] * par["nx"],
        dims=(par["ny"], par["nx"], par["nz"]),
        dir=2,
        sampling=par["dz"],
        edge=par["edge"],
        dtype="float32",
    )

    assert dottest(
        D2op,
        par["nz"] * par["ny"] * par["nx"],
        par["ny"] * par["nx"] * par["nz"],
        tol=1e-3,
    )

    # polynomial f(x,y,z) = z^3, f_{zz}(x,y,z) = 6z
    f = zzz ** 3
    dfana = 6 * zzz
    df = D2op * f.ravel()
    df = df.reshape(par["ny"], par["nx"], par["nz"])

    assert_array_almost_equal(df[:, :, 1:-1], dfana[:, :, 1:-1], decimal=1)


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1e), (par2e), (par3e), (par4e)]
)
def test_SecondDerivative_forward(par):
    """Dot-test for SecondDerivative operator (forward stencil).
    Note that the analytical expression cannot be validated in this case
    """
    x = par["dx"] * np.arange(par["nx"])
    y = par["dy"] * np.arange(par["ny"])
    z = par["dz"] * np.arange(par["nz"])

    xx, yy = np.meshgrid(x, y)  # produces arrays of size (ny,nx)
    xxx, yyy, zzz = np.meshgrid(x, y, z)  # produces arrays of size (ny,nx,nz)

    # 1d
    D2op = SecondDerivative(
        par["nx"], sampling=par["dx"], edge=par["edge"], kind="forward", dtype="float32"
    )
    assert dottest(D2op, par["nx"], par["nx"], tol=1e-3)

    # 2d - derivative on 1st direction
    D2op = SecondDerivative(
        par["ny"] * par["nx"],
        dims=(par["ny"], par["nx"]),
        dir=0,
        sampling=par["dy"],
        edge=par["edge"],
        kind="forward",
        dtype="float32",
    )

    assert dottest(D2op, par["ny"] * par["nx"], par["ny"] * par["nx"], tol=1e-3)

    # 2d - derivative on 2nd direction
    D2op = SecondDerivative(
        par["ny"] * par["nx"],
        dims=(par["ny"], par["nx"]),
        dir=1,
        sampling=par["dx"],
        edge=par["edge"],
        kind="forward",
        dtype="float32",
    )

    assert dottest(D2op, par["ny"] * par["nx"], par["ny"] * par["nx"], tol=1e-3)

    # 3d - derivative on 1st direction
    D2op = SecondDerivative(
        par["nz"] * par["ny"] * par["nx"],
        dims=(par["ny"], par["nx"], par["nz"]),
        dir=0,
        sampling=par["dy"],
        edge=par["edge"],
        kind="forward",
        dtype="float32",
    )

    assert dottest(
        D2op,
        par["nz"] * par["ny"] * par["nx"],
        par["nz"] * par["ny"] * par["nx"],
        tol=1e-3,
    )

    # 3d - derivative on 2nd direction
    D2op = SecondDerivative(
        par["nz"] * par["ny"] * par["nx"],
        dims=(par["ny"], par["nx"], par["nz"]),
        dir=1,
        sampling=par["dx"],
        edge=par["edge"],
        kind="forward",
        dtype="float32",
    )

    assert dottest(
        D2op,
        par["nz"] * par["ny"] * par["nx"],
        par["nz"] * par["ny"] * par["nx"],
        tol=1e-3,
    )

    # 3d - derivative on 3rd direction
    D2op = SecondDerivative(
        par["nz"] * par["ny"] * par["nx"],
        dims=(par["ny"], par["nx"], par["nz"]),
        dir=2,
        sampling=par["dz"],
        edge=par["edge"],
        kind="forward",
        dtype="float32",
    )

    assert dottest(
        D2op,
        par["nz"] * par["ny"] * par["nx"],
        par["ny"] * par["nx"] * par["nz"],
        tol=1e-3,
    )


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1e), (par2e), (par3e), (par4e)]
)
def test_Laplacian(par):
    """Dot-test for Laplacian operator"""
    # 2d - symmetrical
    Dlapop = Laplacian(
        (par["ny"], par["nx"]),
        dirs=(0, 1),
        weights=(1, 1),
        sampling=(par["dy"], par["dx"]),
        edge=par["edge"],
        dtype="float32",
    )
    assert dottest(Dlapop, par["ny"] * par["nx"], par["ny"] * par["nx"], tol=1e-3)

    # 2d - asymmetrical
    Dlapop = Laplacian(
        (par["ny"], par["nx"]),
        dirs=(0, 1),
        weights=(1, 2),
        sampling=(par["dy"], par["dx"]),
        edge=par["edge"],
        dtype="float32",
    )
    assert dottest(Dlapop, par["ny"] * par["nx"], par["ny"] * par["nx"], tol=1e-3)

    # 3d - symmetrical on 1st and 2nd direction
    Dlapop = Laplacian(
        (par["nz"], par["ny"], par["nx"]),
        dirs=(0, 1),
        weights=(1, 1),
        sampling=(par["dy"], par["dx"]),
        edge=par["edge"],
        dtype="float32",
    )
    assert dottest(
        Dlapop,
        par["nz"] * par["ny"] * par["nx"],
        par["nz"] * par["ny"] * par["nx"],
        tol=1e-3,
    )

    # 3d - symmetrical on 1st and 2nd direction
    Dlapop = Laplacian(
        (par["nz"], par["ny"], par["nx"]),
        dirs=(0, 1),
        weights=(1, 1),
        sampling=(par["dy"], par["dx"]),
        edge=par["edge"],
        dtype="float32",
    )
    assert dottest(
        Dlapop,
        par["nz"] * par["ny"] * par["nx"],
        par["nz"] * par["ny"] * par["nx"],
        tol=1e-3,
    )


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1e), (par2e), (par3e), (par4e)]
)
def test_Gradient(par):
    """Dot-test for Gradient operator"""
    for kind in ("forward", "centered", "backward"):
        # 2d
        Gop = Gradient(
            (par["ny"], par["nx"]),
            sampling=(par["dy"], par["dx"]),
            edge=par["edge"],
            kind=kind,
            dtype="float32",
        )
        assert dottest(Gop, 2 * par["ny"] * par["nx"], par["ny"] * par["nx"], tol=1e-3)

        # 3d
        Gop = Gradient(
            (par["nz"], par["ny"], par["nx"]),
            sampling=(par["dz"], par["dy"], par["dx"]),
            edge=par["edge"],
            kind=kind,
            dtype="float32",
        )
        assert dottest(
            Gop,
            3 * par["nz"] * par["ny"] * par["nx"],
            par["nz"] * par["ny"] * par["nx"],
            tol=1e-3,
        )


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1e), (par2e), (par3e), (par4e)]
)
def test_FirstDirectionalDerivative(par):
    """Dot-test for FirstDirectionalDerivative operator"""
    for kind in ("forward", "centered", "backward"):
        # 2d
        Fdop = FirstDirectionalDerivative(
            (par["ny"], par["nx"]),
            v=np.sqrt(2.0) / 2.0 * np.ones(2),
            sampling=(par["dy"], par["dx"]),
            edge=par["edge"],
            kind=kind,
            dtype="float32",
        )
        assert dottest(Fdop, par["ny"] * par["nx"], par["ny"] * par["nx"], tol=1e-3)

        # 3d
        Fdop = FirstDirectionalDerivative(
            (par["nz"], par["ny"], par["nx"]),
            v=np.ones(3) / np.sqrt(3),
            sampling=(par["dz"], par["dy"], par["dx"]),
            edge=par["edge"],
            kind=kind,
            dtype="float32",
        )
        assert dottest(
            Fdop,
            par["nz"] * par["ny"] * par["nx"],
            par["nz"] * par["ny"] * par["nx"],
            tol=1e-3,
        )


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1e), (par2e), (par3e), (par4e)]
)
def test_SecondDirectionalDerivative(par):
    """Dot-test for test_SecondDirectionalDerivative operator"""
    # 2d
    Fdop = SecondDirectionalDerivative(
        (par["ny"], par["nx"]),
        v=np.sqrt(2.0) / 2.0 * np.ones(2),
        sampling=(par["dy"], par["dx"]),
        edge=par["edge"],
        dtype="float32",
    )
    assert dottest(Fdop, par["ny"] * par["nx"], par["ny"] * par["nx"], tol=1e-3)

    # 3d
    Fdop = SecondDirectionalDerivative(
        (par["nz"], par["ny"], par["nx"]),
        v=np.ones(3) / np.sqrt(3),
        sampling=(par["dz"], par["dy"], par["dx"]),
        edge=par["edge"],
        dtype="float32",
    )
    assert dottest(
        Fdop,
        par["nz"] * par["ny"] * par["nx"],
        par["nz"] * par["ny"] * par["nx"],
        tol=1e-3,
    )


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1e), (par2e), (par3e), (par4e)]
)
def test_SecondDirectionalDerivative_verticalderivative(par):
    """Compare vertical derivative for SecondDirectionalDerivative operator
    and SecondDerivative
    """
    Fop = FirstDerivative(
        par["ny"] * par["nx"],
        (par["ny"], par["nx"]),
        dir=0,
        edge=par["edge"],
        dtype="float32",
    )
    F2op = Fop.H * Fop

    F2dop = SecondDirectionalDerivative(
        (par["ny"], par["nx"]), v=np.array([1, 0]), edge=par["edge"], dtype="float32"
    )

    x = np.random.normal(0.0, 1.0, (par["ny"], par["nx"]))
    assert_array_equal(-F2op * x.ravel(), F2dop * x.ravel())
