import os

if int(os.environ.get("TEST_CUPY_PYLOPS", 0)):
    import cupy as np
    from cupy.testing import assert_array_almost_equal
    from cupyx.scipy.signal.windows import triang

    backend = "cupy"
else:
    import numpy as np
    from numpy.testing import assert_array_almost_equal
    from scipy.signal.windows import triang

    backend = "numpy"
import pytest

from pylops.optimization.basic import lsqr
from pylops.signalprocessing import (
    Convolve1D,
    Convolve2D,
    ConvolveND,
    NonStationaryConvolve1D,
    NonStationaryConvolve2D,
    NonStationaryConvolve3D,
    NonStationaryFilters1D,
    NonStationaryFilters2D,
)
from pylops.utils import dottest

# filters
nfilts = (5, 7)
nfilts3 = (5, 5, 7)

h1 = triang(nfilts[0], sym=True)
h2 = np.outer(triang(nfilts[0], sym=True), triang(nfilts[1], sym=True))
h3 = np.outer(
    triang(nfilts[0], sym=True),
    np.outer(triang(nfilts[0], sym=True), triang(nfilts[1], sym=True)[np.newaxis]).T,
).reshape(nfilts3)

h1stat = np.vstack([h1, h1, h1])
h1ns = np.vstack([h1, -h1, 2 * h1])

h2stat = np.vstack(
    [
        h2.ravel(),
        h2.ravel(),
        h2.ravel(),
        h2.ravel(),
        h2.ravel(),
        h2.ravel(),
    ]
).reshape(3, 2, nfilts[0], nfilts[1])
h2ns = np.vstack(
    [
        2 * h2.ravel(),
        h2.ravel(),
        h2.ravel(),
        h2.ravel(),
        -h2.ravel(),
        2 * h2.ravel(),
    ]
).reshape(3, 2, nfilts[0], nfilts[1])

h3stat = np.vstack(
    [
        h3.ravel(),
    ]
    * 8
).reshape(2, 2, 2, nfilts[0], nfilts[0], nfilts[1])
h3ns = np.vstack(
    [
        2 * h3.ravel(),
        h3.ravel(),
        h3.ravel(),
        h3.ravel(),
        h3.ravel(),
        h3.ravel(),
        -h3.ravel(),
        2 * h3.ravel(),
    ]
).reshape(2, 2, 2, nfilts[0], nfilts[0], nfilts[1])


par1_1d = {
    "nz": 21,
    "nx": 31,
    "axis": 0,
}  # first direction
par2_1d = {
    "nz": 21,
    "nx": 31,
    "axis": 1,
}  # second direction

par_2d = {
    "nz": 21,
    "nx": 31,
}


@pytest.mark.parametrize("par", [(par_2d)])
def test_even_filter(par):
    """Check error is raised if filter has even size"""
    with pytest.raises(ValueError):
        _ = NonStationaryConvolve1D(
            dims=par["nx"],
            hs=h1ns[..., :-1],
            ih=(int(par["nx"] // 4), int(2 * par["nx"] // 4), int(3 * par["nx"] // 4)),
        )
    with pytest.raises(ValueError):
        _ = NonStationaryConvolve2D(
            dims=(par["nx"], par["nz"]),
            hs=h2ns[..., :-1],
            ihx=(int(par["nx"] // 4), int(2 * par["nx"] // 4), int(3 * par["nx"] // 4)),
            ihz=(int(par["nz"] // 4), int(3 * par["nz"] // 4)),
        )

    with pytest.raises(ValueError):
        _ = NonStationaryFilters1D(
            inp=np.arange(par["nx"]),
            hsize=nfilts[0] - 1,
            ih=(int(par["nx"] // 4), int(2 * par["nx"] // 4), int(3 * par["nx"] // 4)),
        )
    with pytest.raises(ValueError):
        _ = NonStationaryFilters2D(
            inp=np.ones((par["nx"], par["nz"])),
            hshape=(nfilts[0] - 1, nfilts[1] - 1),
            ihx=(int(par["nx"] // 4), int(2 * par["nx"] // 4), int(3 * par["nx"] // 4)),
            ihz=(int(par["nz"] // 4), int(3 * par["nz"] // 4)),
        )


@pytest.mark.parametrize("par", [(par_2d)])
def test_ih_irregular(par):
    """Check error is raised if ih (or ihx/ihz) are irregularly sampled"""
    with pytest.raises(ValueError):
        _ = NonStationaryConvolve1D(
            dims=par["nx"],
            hs=h1ns,
            ih=(10, 11, 15),
        )
    with pytest.raises(ValueError):
        _ = NonStationaryConvolve2D(
            dims=(par["nx"], par["nz"]),
            hs=h2ns,
            ihx=(10, 11, 15),
            ihz=(int(par["nz"] // 4), int(3 * par["nz"] // 4)),
        )


@pytest.mark.parametrize("par", [(par_2d)])
def test_unknown_engine_2d(par):
    """Check error is raised if unknown engine is passed"""
    with pytest.raises(NotImplementedError):
        _ = NonStationaryConvolve2D(
            dims=(par["nx"], par["nz"]),
            hs=h2ns,
            ihx=(int(par["nx"] // 3), int(2 * par["nx"] // 3)),
            ihz=(int(par["nz"] // 3), int(2 * par["nz"] // 3)),
            engine="foo",
        )
    with pytest.raises(NotImplementedError):
        _ = NonStationaryFilters2D(
            inp=np.ones((par["nx"], par["nz"])),
            hshape=(nfilts[0] - 1, nfilts[1] - 1),
            ihx=(int(par["nx"] // 3), int(2 * par["nx"] // 3)),
            ihz=(int(par["nz"] // 3), int(2 * par["nz"] // 3)),
            engine="foo",
        )


@pytest.mark.parametrize("par", [(par1_1d), (par2_1d)])
def test_NonStationaryConvolve1D(par):
    """Dot-test and inversion for NonStationaryConvolve1D operator"""
    # 1D
    if par["axis"] == 0:
        Cop = NonStationaryConvolve1D(
            dims=par["nx"],
            hs=h1ns,
            ih=(int(par["nx"] // 4), int(2 * par["nx"] // 4), int(3 * par["nx"] // 4)),
            dtype="float64",
        )
        assert dottest(Cop, par["nx"], par["nx"], backend=backend)

        x = np.zeros((par["nx"]))
        x[par["nx"] // 2] = 1.0
        xlsqr = lsqr(
            Cop,
            Cop * x,
            x0=np.zeros_like(x),
            damp=1e-20,
            niter=200,
            atol=1e-8,
            btol=1e-8,
            show=0,
        )[0]
        assert_array_almost_equal(x, xlsqr, decimal=1)

    # 1D on 2D
    nfilt = par["nx"] if par["axis"] == 0 else par["nz"]
    Cop = NonStationaryConvolve1D(
        dims=(par["nx"], par["nz"]),
        hs=h1ns,
        ih=(int(nfilt // 4), int(2 * nfilt // 4), int(3 * nfilt // 4)),
        axis=par["axis"],
        dtype="float64",
    )
    assert dottest(Cop, par["nx"] * par["nz"], par["nx"] * par["nz"], backend=backend)

    x = np.zeros((par["nx"], par["nz"]))
    x[
        int(par["nx"] / 2 - 3) : int(par["nx"] / 2 + 3),
        int(par["nz"] / 2 - 3) : int(par["nz"] / 2 + 3),
    ] = 1.0
    x = x.ravel()
    xlsqr = lsqr(
        Cop,
        Cop * x,
        x0=np.zeros_like(x),
        damp=1e-20,
        niter=400,
        atol=1e-8,
        btol=1e-8,
        show=0,
    )[0]
    assert_array_almost_equal(x, xlsqr, decimal=1)


@pytest.mark.parametrize("par", [(par1_1d)])
def test_StationaryConvolve1D(par):
    """Check that Convolve1D and NonStationaryConvolve1D return same result for
    stationary filter"""
    np.random.seed(10)
    Cop = NonStationaryConvolve1D(
        dims=par["nx"],
        hs=h1stat,
        ih=(int(par["nx"] // 4), int(2 * par["nx"] // 4), int(3 * par["nx"] // 4)),
        dtype="float64",
    )
    Cop_stat = Convolve1D(
        dims=par["nx"],
        h=h1,
        offset=nfilts[0] // 2,
        dtype="float64",
    )

    x = np.random.normal(0, 1, par["nx"])
    assert_array_almost_equal(Cop_stat * x, Cop * x, decimal=10)


@pytest.mark.parametrize("par", [(par_2d)])
def test_NonStationaryConvolve2D(par):
    """Dot-test and inversion for NonStationaryConvolve2D operator"""
    Cop = NonStationaryConvolve2D(
        dims=(par["nx"], par["nz"]),
        hs=h2ns,
        ihx=(int(par["nx"] // 4), int(2 * par["nx"] // 4), int(3 * par["nx"] // 4)),
        ihz=(int(par["nz"] // 4), int(3 * par["nz"] // 4)),
        engine="numpy" if backend == "numpy" else "cuda",
        dtype="float64",
    )
    assert dottest(Cop, par["nx"] * par["nz"], par["nx"] * par["nz"], backend=backend)

    x = np.zeros((par["nx"], par["nz"]))
    x[
        int(par["nx"] / 2 - 3) : int(par["nx"] / 2 + 3),
        int(par["nz"] / 2 - 3) : int(par["nz"] / 2 + 3),
    ] = 1.0
    x = x.ravel()
    xlsqr = lsqr(
        Cop,
        Cop * x,
        x0=np.zeros_like(x),
        damp=1e-20,
        niter=300,
        atol=1e-8,
        btol=1e-8,
        show=0,
    )[0]
    assert_array_almost_equal(x, xlsqr, decimal=1)


@pytest.mark.parametrize("par", [(par_2d)])
def test_StationaryConvolve2D(par):
    """Check that Convolve2D and NonStationaryConvolve2D return same result for
    stationary filter"""
    Cop = NonStationaryConvolve2D(
        dims=(par["nx"], par["nz"]),
        hs=h2stat,
        ihx=(int(par["nx"] // 4), int(2 * par["nx"] // 4), int(3 * par["nx"] // 4)),
        ihz=(int(par["nz"] // 4), int(3 * par["nz"] // 4)),
        engine="numpy" if backend == "numpy" else "cuda",
        dtype="float64",
    )
    Cop_stat = Convolve2D(
        dims=(par["nx"], par["nz"]),
        h=h2,
        offset=(nfilts[0] // 2, nfilts[1] // 2),
        dtype="float64",
    )
    x = np.random.normal(0, 1, (par["nx"], par["nz"]))

    assert_array_almost_equal(Cop_stat * x, Cop * x, decimal=10)


@pytest.mark.parametrize(
    "par",
    [
        (par1_1d),
    ],
)
def test_NonStationaryFilters1D(par):
    """Dot-test and inversion for NonStationaryFilters2D operator"""
    x = np.zeros((par["nx"]))
    x[par["nx"] // 4], x[par["nx"] // 2], x[3 * par["nx"] // 4] = 1.0, 1.0, 1.0
    Cop = NonStationaryFilters1D(
        inp=x,
        hsize=nfilts[0],
        ih=(int(par["nx"] // 4), int(2 * par["nx"] // 4), int(3 * par["nx"] // 4)),
        dtype="float64",
    )
    assert dottest(Cop, par["nx"], 3 * nfilts[0], backend=backend)

    h1lsqr = lsqr(
        Cop, Cop * h1ns, x0=np.zeros_like(h1ns), damp=1e-20, niter=200, show=0
    )[0]
    assert_array_almost_equal(h1ns, h1lsqr, decimal=1)


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par_2d)])
def test_NonStationaryFilters2D(par):
    """Dot-test and inversion for NonStationaryFilters2D operator"""
    x = np.zeros((par["nx"], par["nz"]))
    x[int(par["nx"] // 4)] = 1.0
    x[int(par["nx"] // 2)] = 1.0
    x[int(3 * par["nx"] // 4)] = 1.0

    Cop = NonStationaryFilters2D(
        inp=x,
        hshape=nfilts,
        ihx=(int(par["nx"] // 4), int(2 * par["nx"] // 4), int(3 * par["nx"] // 4)),
        ihz=(int(par["nz"] // 4), int(3 * par["nz"] // 4)),
        dtype="float64",
    )
    assert dottest(
        Cop, par["nx"] * par["nz"], 6 * nfilts[0] * nfilts[1], backend=backend
    )

    h2lsqr = lsqr(
        Cop,
        Cop * h2ns.ravel(),
        x0=np.zeros_like(h2ns).ravel(),
        damp=1e-20,
        niter=400,
        show=0,
    )[0]
    assert_array_almost_equal(h2ns.ravel(), h2lsqr, decimal=1)


@pytest.mark.parametrize("par", [(par_2d)])
def test_NonStationaryConvolve3D(par):
    """Dot-test and inversion for NonStationaryConvolve3D operator"""
    Cop = NonStationaryConvolve3D(
        dims=(par["nx"], par["nx"], par["nz"]),
        hs=h3ns,
        ihx=(int(par["nx"] // 4), int(3 * par["nx"] // 4)),
        ihy=(int(par["nx"] // 4), int(3 * par["nx"] // 4)),
        ihz=(int(par["nz"] // 4), int(3 * par["nz"] // 4)),
        engine="numpy" if backend == "numpy" else "cuda",
        dtype="float64",
    )
    assert dottest(
        Cop,
        par["nx"] * par["nx"] * par["nz"],
        par["nx"] * par["nx"] * par["nz"],
        backend=backend,
    )

    x = np.zeros((par["nx"], par["nx"], par["nz"]))
    x[
        int(par["nx"] / 2 - 3) : int(par["nx"] / 2 + 3),
        int(par["nx"] / 2 - 3) : int(par["nx"] / 2 + 3),
        int(par["nz"] / 2 - 3) : int(par["nz"] / 2 + 3),
    ] = 1.0
    x = x.ravel()
    xlsqr = lsqr(
        Cop,
        Cop * x,
        x0=np.zeros_like(x),
        damp=1e-20,
        niter=40,
        atol=1e-8,
        btol=1e-8,
        show=0,
    )[0]
    # given the size of the problem, we can only run few iterations and test accuracy up to 30%
    assert np.linalg.norm(x - xlsqr) / np.linalg.norm(x) < 0.3


@pytest.mark.parametrize("par", [(par_2d)])
def test_StationaryConvolve3D(par):
    """Check that Convolve3D and NonStationaryConvolve3D return same result for
    stationary filter"""
    Cop = NonStationaryConvolve3D(
        dims=(par["nx"], par["nx"], par["nz"]),
        hs=h3stat,
        ihx=(int(par["nx"] // 4), int(3 * par["nx"] // 4)),
        ihy=(int(par["nx"] // 4), int(3 * par["nx"] // 4)),
        ihz=(int(par["nz"] // 4), int(3 * par["nz"] // 4)),
        engine="numpy" if backend == "numpy" else "cuda",
        dtype="float64",
    )
    Cop_stat = ConvolveND(
        dims=(par["nx"], par["nx"], par["nz"]),
        h=h3,
        offset=(nfilts[0] // 2, nfilts[0] // 2, nfilts[1] // 2),
        dtype="float64",
    )
    x = np.random.normal(0, 1, (par["nx"], par["nx"], par["nz"]))

    assert_array_almost_equal(Cop_stat * x, Cop * x, decimal=10)
