import os

if int(os.environ.get("TEST_CUPY_PYLOPS", 0)):
    import cupy as np

    backend = "cupy"
else:
    import numpy as np

    backend = "numpy"
import numpy as npp
import pytest

from pylops.utils import dottest
from pylops.waveeqprocessing import BlendingContinuous, BlendingGroup, BlendingHalf

par = {"nt": 101, "ns": 50, "nr": 20, "dtype": "float64"}

d = np.random.normal(0, 1, (par["ns"], par["nr"], par["nt"]))
dt = 0.004


@pytest.mark.parametrize("par", [(par)])
def test_Blending_continuous(par):
    """Dot-test for continuous Blending operator"""
    npp.random.seed(0)
    # ignition times
    overlap = 0.5
    ignition_times = 2.0 * npp.random.rand(par["ns"]) - 1.0
    ignition_times += (
        npp.arange(0, overlap * par["nt"] * par["ns"], overlap * par["nt"]) * dt
    )
    ignition_times[0] = 0.0

    Bop = BlendingContinuous(
        par["nt"],
        par["nr"],
        par["ns"],
        dt,
        ignition_times,
        dtype=par["dtype"],
    )
    assert dottest(
        Bop,
        Bop.nttot * par["nr"],
        par["nt"] * par["ns"] * par["nr"],
        backend=backend,
    )


@pytest.mark.parametrize("par", [(par)])
def test_Blending_group(par):
    """Dot-test for group Blending operator"""
    npp.random.seed(0)
    group_size = 2
    n_groups = par["ns"] // group_size
    ignition_times = 0.8 * npp.random.rand(par["ns"])

    Bop = BlendingGroup(
        par["nt"],
        par["nr"],
        par["ns"],
        dt,
        ignition_times.reshape(group_size, n_groups),
        n_groups=n_groups,
        group_size=group_size,
        dtype=par["dtype"],
    )
    assert dottest(
        Bop,
        par["nt"] * n_groups * par["nr"],
        par["nt"] * par["ns"] * par["nr"],
        backend=backend,
    )


@pytest.mark.parametrize("par", [(par)])
def test_Blending_half(par):
    """Dot-test for half Blending operator"""
    npp.random.seed(0)
    group_size = 2
    n_groups = par["ns"] // group_size
    ignition_times = 0.8 * npp.random.rand(par["ns"])

    Bop = BlendingHalf(
        par["nt"],
        par["nr"],
        par["ns"],
        dt,
        ignition_times.reshape(group_size, n_groups),
        n_groups=n_groups,
        group_size=group_size,
        dtype=par["dtype"],
    )
    assert dottest(
        Bop,
        par["nt"] * n_groups * par["nr"],
        par["nt"] * par["ns"] * par["nr"],
        backend=backend,
    )
