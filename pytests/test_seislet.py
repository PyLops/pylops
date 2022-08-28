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
    "dtype": "float32",
}  # nx power of 2, max level
par2 = {
    "nx": 16,
    "nt": 30,
    "dx": 10,
    "dt": 0.004,
    "level": 2,
    "dtype": "float32",
}  # nx power of 2, smaller level
par3 = {
    "nx": 13,
    "nt": 30,
    "dx": 10,
    "dt": 0.004,
    "level": 2,
    "dtype": "float32",
}  # nx not power of 2, max level

np.random.seed(10)


@pytest.mark.parametrize("par", [(par1)])
def test_predict_trace(par):
    """Dot-test for _predict_trace operator"""
    t = np.arange(par["nt"]) * par["dt"]
    for slope in [-0.2, 0.0, 0.3]:
        Fop = FunctionOperator(
            lambda x: _predict_trace(x, t, par["dt"], par["dx"], slope),
            lambda x: _predict_trace(x, t, par["dt"], par["dx"], slope, adj=True),
            par["nt"],
            par["nt"],
        )
        dottest(Fop, par["nt"], par["nt"])


@pytest.mark.parametrize("par", [(par1)])
def test_predict(par):
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

    for predictor in (_predict_haar, _predict_lin):
        for repeat in (0, 1, 2):
            slope = np.random.normal(0, 0.1, (2 ** (repeat + 1) * par["nx"], par["nt"]))
            for backward in (False, True):
                Fop = FunctionOperator(
                    lambda x: _predict_reshape(
                        predictor,
                        x,
                        par["nt"],
                        par["nx"],
                        par["dt"],
                        par["dx"],
                        slope,
                        backward=backward,
                    ),
                    lambda x: _predict_reshape(
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


@pytest.mark.parametrize("par", [(par1), (par2), (par3)])
def test_Seislet(par):
    """Dot-test and forward-inverse for Seislet"""
    slope = np.random.normal(0, 0.1, (par["nx"], par["nt"]))

    for kind in ("haar", "linear"):
        Sop = Seislet(
            slope,
            sampling=(par["dx"], par["dt"]),
            level=par["level"],
            kind=kind,
            dtype=par["dtype"],
        )
        dottest(Sop, Sop.shape[0], par["nx"] * par["nt"])

        x = np.random.normal(0, 0.1, par["nx"] * par["nt"])
        y = Sop * x
        xinv = Sop.inverse(y)
        assert_array_almost_equal(x, xinv)
