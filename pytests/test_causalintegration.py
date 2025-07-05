import os

if int(os.environ.get("TEST_CUPY_PYLOPS", 0)):
    import cupy as np
    from cupy.testing import assert_array_almost_equal

    backend = "cupy"
else:
    import numpy as np
    from numpy.testing import assert_array_almost_equal

    backend = "numpy"
import itertools

import pytest

from pylops.basicoperators import CausalIntegration, FirstDerivative
from pylops.optimization.basic import lsqr
from pylops.utils import dottest
from pylops.utils.backend import get_module_name

par1 = {
    "nt": 20,
    "nx": 101,
    "dt": 1.0,
    "imag": 0,
    "dtype": "float64",
}  # even samples, real, unitary step
par2 = {
    "nt": 21,
    "nx": 101,
    "dt": 1.0,
    "imag": 0,
    "dtype": "float64",
}  # odd samples, real, unitary step
par3 = {
    "nt": 20,
    "nx": 101,
    "dt": 0.3,
    "imag": 0,
    "dtype": "float64",
}  # even samples, real, non-unitary step
par4 = {
    "nt": 21,
    "nx": 101,
    "dt": 0.3,
    "imag": 0,
    "dtype": "float64",
}  # odd samples, real, non-unitary step
par1j = {
    "nt": 20,
    "nx": 101,
    "dt": 1.0,
    "imag": 1j,
    "dtype": "complex128",
}  # even samples, complex, unitary step
par2j = {
    "nt": 21,
    "nx": 101,
    "dt": 1.0,
    "imag": 1j,
    "dtype": "complex128",
}  # odd samples, complex, unitary step
par3j = {
    "nt": 20,
    "nx": 101,
    "dt": 0.3,
    "imag": 1j,
    "dtype": "complex128",
}  # even samples, complex, non-unitary step
par4j = {
    "nt": 21,
    "nx": 101,
    "dt": 0.3,
    "imag": 1j,
    "dtype": "complex128",
}  # odd samples, complex, non-unitary step

np.random.seed(0)


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1j), (par2j), (par3j), (par4j)]
)
def test_CausalIntegration1d(par):
    """Dot-test and inversion for CausalIntegration operator for 1d signals"""
    t = np.arange(par["nt"]) * par["dt"]
    x = t + par["imag"] * t

    for kind, rf in itertools.product(("full", "half", "trapezoidal"), (False, True)):
        rf = rf if get_module_name == "numpy" else False
        Cop = CausalIntegration(
            par["nt"],
            sampling=par["dt"],
            kind=kind,
            removefirst=rf,
            dtype=par["dtype"],
        )
        rf1 = 1 if rf else 0
        assert dottest(
            Cop,
            par["nt"] - rf1,
            par["nt"],
            complexflag=0 if par["imag"] == 0 else 3,
            backend=backend,
        )

        # test analytical integration and derivative inversion only for
        # cases where a zero c is required
        if kind != "full" and not rf:
            # numerical integration
            y = Cop * x
            # analytical integration
            yana = (
                t**2 / 2.0
                - t[0] ** 2 / 2.0
                + par["imag"] * (t**2 / 2.0 - t[0] ** 2 / 2.0)
            )

            assert_array_almost_equal(y, yana[rf1:], decimal=4)

            # numerical derivative
            Dop = FirstDerivative(
                par["nt"] - rf1, sampling=par["dt"], dtype=par["dtype"]
            )
            xder = Dop * y.ravel()

            # derivative by inversion
            xinv = lsqr(
                Cop,
                y,
                x0=np.zeros_like(x),
                niter=100,
                atol=0,
                btol=0,
                conlim=np.inf,
                show=0,
            )[0]

            assert_array_almost_equal(x[:-1], xder[:-1], decimal=4)
            assert_array_almost_equal(x, xinv, decimal=4)


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1j), (par2j), (par3j), (par4j)]
)
def test_CausalIntegration2d(par):
    """Dot-test and inversion for CausalIntegration operator for 2d signals"""
    dt = 0.2 * par["dt"]  # need lower frequency in sinusoids for stability
    t = np.arange(par["nt"]) * dt
    x = np.outer(np.sin(t), np.ones(par["nx"])) + par["imag"] * np.outer(
        np.sin(t), np.ones(par["nx"])
    )

    for kind, rf in itertools.product(("full", "half", "trapezoidal"), (False, True)):
        rf = rf if get_module_name == "numpy" else False
        Cop = CausalIntegration(
            (par["nt"], par["nx"]),
            sampling=dt,
            axis=0,
            kind=kind,
            removefirst=rf,
            dtype=par["dtype"],
        )
        rf1 = 1 if rf else 0
        assert dottest(
            Cop,
            (par["nt"] - rf1) * par["nx"],
            par["nt"] * par["nx"],
            complexflag=0 if par["imag"] == 0 else 3,
            backend=backend,
        )

        # test analytical integration and derivative inversion only for
        # cases where a zero c is required
        if kind != "full" and not rf:
            # numerical integration
            y = Cop * x.ravel()
            y = y.reshape(par["nt"], par["nx"])

            # analytical integration
            yana = (
                np.outer(-np.cos(t), np.ones(par["nx"]))
                + np.cos(t[0])
                + par["imag"]
                * (np.outer(-np.cos(t), np.ones(par["nx"])) + np.cos(t[0]))
            )
            yana = yana.reshape(par["nt"], par["nx"])

            assert_array_almost_equal(y, yana, decimal=2)

            # numerical derivative
            Dop = FirstDerivative(
                (par["nt"], par["nx"]), axis=0, sampling=dt, dtype=par["dtype"]
            )
            xder = Dop * y.ravel()
            xder = xder.reshape(par["nt"], par["nx"])

            # derivative by inversion
            xinv = lsqr(
                Cop,
                y.ravel(),
                x0=np.zeros_like(x).ravel(),
                niter=100,
                atol=0,
                btol=0,
                conlim=np.inf,
                show=0,
            )[0]
            xinv = xinv.reshape(par["nt"], par["nx"])

            assert_array_almost_equal(x[:-1], xder[:-1], decimal=2)
            assert_array_almost_equal(x, xinv, decimal=2)
