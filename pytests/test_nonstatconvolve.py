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
h1s = np.vstack([h1, h1, h1])
h2s = np.vstack(
    [
        h2.ravel(),
        h2.ravel(),
        h2.ravel(),
        h2.ravel(),
        h2.ravel(),
        h2.ravel(),
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
            hs=h1s[..., :-1],
            ih=(int(par["nx"] // 4), int(2 * par["nx"] // 4), int(3 * par["nx"] // 4)),
        )
    with pytest.raises(ValueError):
        _ = NonStationaryConvolve2D(
            dims=(par["nx"], par["nz"]),
            hs=h2s[..., :-1],
            ihx=(int(par["nx"] // 4), int(2 * par["nx"] // 4), int(3 * par["nx"] // 4)),
            ihz=(int(par["nz"] // 4), int(3 * par["nz"] // 4)),
        )


@pytest.mark.parametrize("par", [(par_2d)])
def test_ih_irregular(par):
    """Check error is raised if ih (or ihx/ihz) are irregularly sampled"""
    with pytest.raises(ValueError):
        _ = NonStationaryConvolve1D(
            dims=par["nx"],
            hs=h1s,
            ih=(10, 11, 15),
        )
    with pytest.raises(ValueError):
        _ = NonStationaryConvolve2D(
            dims=(par["nx"], par["nz"]),
            hs=h2s,
            ihx=(10, 11, 15),
            ihz=(int(par["nz"] // 4), int(3 * par["nz"] // 4)),
        )


@pytest.mark.parametrize("par", [(par_2d)])
def test_unknown_engine_2d(par):
    """Check error is raised if unknown engine is passed"""
    with pytest.raises(NotImplementedError):
        _ = NonStationaryConvolve2D(
            dims=(par["nx"], par["nz"]),
            hs=h2s,
            ihx=(int(par["nx"] // 3), int(2 * par["nx"] // 3)),
            ihz=(int(par["nz"] // 3), int(2 * par["nz"] // 3)),
            engine="foo",
        )


@pytest.mark.parametrize("par", [(par1_1d), (par2_1d)])
def test_NonStationaryConvolve1D(par):
    """Dot-test and inversion for NonStationaryConvolve1D operator"""
    np.random.seed(10)
    # 1D
    if par["axis"] == 0:
        Cop = NonStationaryConvolve1D(
            dims=par["nx"],
            hs=h1s,
            ih=(int(par["nx"] // 4), int(2 * par["nx"] // 4), int(3 * par["nx"] // 4)),
            dtype="float64",
        )
        assert dottest(Cop, par["nx"], par["nx"])

        x = np.zeros((par["nx"]))
        x[par["nx"] // 2] = 1.0
        xlsqr = lsqr(Cop, Cop * x, damp=1e-20, iter_lim=200, show=0)[0]
        assert_array_almost_equal(x, xlsqr, decimal=1)

    # 1D on 2D
    Cop = NonStationaryConvolve1D(
        dims=(par["nx"], par["nz"]),
        hs=h1s,
        ih=(int(par["nx"] // 4), int(2 * par["nx"] // 4), int(3 * par["nx"] // 4)),
        axis=par["axis"],
        dtype="float64",
    )
    assert dottest(Cop, par["nz"] * par["nx"], par["nz"] * par["nx"])

    x = np.zeros((par["nx"], par["nz"]))
    x[
        int(par["nz"] / 2 - 3) : int(par["nz"] / 2 + 3),
        int(par["nx"] / 2 - 3) : int(par["nx"] / 2 + 3),
    ] = 1.0
    x = x.ravel()
    xlsqr = lsqr(Cop, Cop * x, damp=1e-20, iter_lim=200, show=0)[0]
    assert_array_almost_equal(x, xlsqr, decimal=1)


@pytest.mark.parametrize("par", [(par_2d)])
def test_NonStationaryConvolve2D(par):
    """Dot-test and inversion for NonStationaryConvolve2D operator"""
    Cop = NonStationaryConvolve2D(
        dims=(par["nx"], par["nz"]),
        hs=h2s,
        ihx=(int(par["nx"] // 4), int(2 * par["nx"] // 4), int(3 * par["nx"] // 4)),
        ihz=(int(par["nz"] // 4), int(3 * par["nz"] // 4)),
        dtype="float64",
    )
    assert dottest(Cop, par["nz"] * par["nx"], par["nz"] * par["nx"])

    x = np.zeros((par["nx"], par["nz"]))
    x[
        int(par["nz"] / 2 - 3) : int(par["nz"] / 2 + 3),
        int(par["nx"] / 2 - 3) : int(par["nx"] / 2 + 3),
    ] = 1.0
    x = x.ravel()
    xlsqr = lsqr(Cop, Cop * x, damp=1e-20, iter_lim=200, show=0)[0]
    assert_array_almost_equal(x, xlsqr, decimal=1)
