import os

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from pylops.basicoperators import FunctionOperator
from pylops.signalprocessing import Seislet
from pylops.signalprocessing.seislet import _predict_haar, _predict_lin, _predict_trace
from pylops.utils import dottest

par1 = {
    "nx": 16,
    "nt": 30,
    "dx": 10,
    "dt": 0.004,
    "level": None,
}  # nx power of 2, max level
par2 = {
    "nx": 16,
    "nt": 30,
    "dx": 10,
    "dt": 0.004,
    "level": 2,
}  # nx power of 2, smaller level
par3 = {
    "nx": 13,
    "nt": 30,
    "dx": 10,
    "dt": 0.004,
    "level": 2,
}  # nx not power of 2, max level

np.random.seed(10)


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1)])
@pytest.mark.parametrize("slope", [-0.2, 0.0, 0.3])
def test_predict_trace(par, slope):
    """Dot-test for _predict_trace operator"""
    t = np.arange(par["nt"]) * par["dt"]
    Fop = FunctionOperator(
        lambda x, slope=slope: _predict_trace(x, t, par["dt"], par["dx"], slope),
        lambda x, slope=slope: _predict_trace(
            x, t, par["dt"], par["dx"], slope, adj=True
        ),
        par["nt"],
        par["nt"],
    )
    dottest(Fop, par["nt"], par["nt"])


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1)])
@pytest.mark.parametrize(
    "predictor", [_predict_haar, _predict_lin], ids=["haar", "linear"]
)
@pytest.mark.parametrize("repeat", [0, 1, 2])
@pytest.mark.parametrize("backward", [False, True])
def test_predict(par, predictor, repeat, backward):
    """Dot-test for _predict operator"""

    def _predict_reshape(
        predictor, traces, nt, nx, dt, dx, slopes, repeat=0, backward=False, adj=False
    ):
        return predictor(
            traces.reshape(nx, nt),
            dt,
            dx,
            slopes,
            repeat=repeat,
            backward=backward,
            adj=adj,
        )

    slope = np.random.normal(0, 0.1, (2 ** (repeat + 1) * par["nx"], par["nt"]))
    Fop = FunctionOperator(
        lambda x, predictor=predictor, slope=slope, backward=backward: _predict_reshape(
            predictor,
            x,
            par["nt"],
            par["nx"],
            par["dt"],
            par["dx"],
            slope,
            backward=backward,
        ),
        lambda x, predictor=predictor, slope=slope, backward=backward: _predict_reshape(
            predictor,
            x,
            par["nt"],
            par["nx"],
            par["dt"],
            par["dx"],
            slope,
            backward=backward,
            adj=True,
        ),
        par["nt"] * par["nx"],
        par["nt"] * par["nx"],
    )
    dottest(Fop, par["nt"] * par["nx"], par["nt"] * par["nx"])


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par2), (par3)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("kind", ["haar", "linear"])
def test_Seislet(par, dtype, kind):
    """Dot-test and forward/adjoint/inverse for Seislet"""
    slope = np.random.normal(0, 0.1, (par["nx"], par["nt"])).astype(dtype)

    Sop = Seislet(
        slope,
        sampling=(par["dx"], par["dt"]),
        level=par["level"],
        kind=kind,
        dtype=dtype,
    )
    dottest(
        Sop,
        Sop.shape[0],
        par["nx"] * par["nt"],
        rtol=1e-3 if dtype == np.float32 else 1e-6,
    )

    x = np.random.normal(0, 0.1, par["nx"] * par["nt"]).astype(dtype)
    y = Sop * x
    xadj = Sop.H * y
    xinv = Sop.inverse(y)
    assert y.dtype == dtype
    assert xadj.dtype == dtype
    assert_array_almost_equal(x, xinv, decimal=3 if dtype == np.float32 else 6)
