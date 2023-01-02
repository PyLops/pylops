import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from scipy.signal.windows import triang
from scipy.sparse.linalg import lsqr

from pylops.signalprocessing import (
    Convolve1D,
    Convolve2D,
    NonStationaryConvolve1D,
    NonStationaryConvolve2D,
)
from pylops.utils import dottest

# filters
nfilt = (5, 7)
h1 = triang(nfilt[0], sym=True)
h2 = np.outer(triang(nfilt[0], sym=True), triang(nfilt[1], sym=True))
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
).reshape(3, 2, nfilt[0], nfilt[1])
h2ns = np.vstack(
    [
        2 * h2.ravel(),
        h2.ravel(),
        h2.ravel(),
        h2.ravel(),
        -h2.ravel(),
        2 * h2.ravel(),
    ]
).reshape(3, 2, nfilt[0], nfilt[1])

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
        assert dottest(Cop, par["nx"], par["nx"])

        x = np.zeros((par["nx"]))
        x[par["nx"] // 2] = 1.0
        xlsqr = lsqr(Cop, Cop * x, damp=1e-20, iter_lim=200, show=0)[0]
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
    assert dottest(Cop, par["nx"] * par["nz"], par["nx"] * par["nz"])

    x = np.zeros((par["nx"], par["nz"]))
    x[
        int(par["nx"] / 2 - 3) : int(par["nx"] / 2 + 3),
        int(par["nz"] / 2 - 3) : int(par["nz"] / 2 + 3),
    ] = 1.0
    x = x.ravel()
    xlsqr = lsqr(Cop, Cop * x, damp=1e-20, iter_lim=400, show=0)[0]
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
        offset=nfilt[0] // 2,
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
        dtype="float64",
    )
    assert dottest(Cop, par["nx"] * par["nz"], par["nx"] * par["nz"])

    x = np.zeros((par["nx"], par["nz"]))
    x[
        int(par["nx"] / 2 - 3) : int(par["nx"] / 2 + 3),
        int(par["nz"] / 2 - 3) : int(par["nz"] / 2 + 3),
    ] = 1.0
    x = x.ravel()
    xlsqr = lsqr(Cop, Cop * x, damp=1e-20, iter_lim=400, show=0)[0]
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
        dtype="float64",
    )
    Cop_stat = Convolve2D(
        dims=(par["nx"], par["nz"]),
        h=h2,
        offset=(nfilt[0] // 2, nfilt[1] // 2),
        dtype="float64",
    )
    x = np.random.normal(0, 1, (par["nx"], par["nz"]))

    assert_array_almost_equal(Cop_stat * x, Cop * x, decimal=10)
