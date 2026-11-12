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

par = {"nt": 101, "ns": 50, "nr": 20}

par1, par2, par3 = par.copy(), par.copy(), par.copy()
par1["dtype"] = np.float64
par2["dtype"] = np.float32
par3["dtype"] = "float64"  # default dtype provided as string

d = np.random.normal(0, 1, (par["ns"], par["nr"], par["nt"]))
dt = 0.004


@pytest.mark.parametrize("par", [(par1), (par2)])
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
        rtol=1e-4 if par["dtype"] == np.float32 else 1e-6,
        backend=backend,
    )

    # Forward and adjoint
    db = Bop * d.astype(par["dtype"])
    dadj = Bop.H * db
    assert db.dtype == par["dtype"]
    assert dadj.dtype == par["dtype"]


@pytest.mark.parametrize("par", [(par1), (par2), (par3)])
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
    assert isinstance(Bop.dtype, npp.dtype)
    assert Bop.dtype == npp.dtype(par["dtype"])
    assert dottest(
        Bop,
        par["nt"] * n_groups * par["nr"],
        par["nt"] * par["ns"] * par["nr"],
        rtol=1e-4 if par["dtype"] == np.float32 else 1e-6,
        backend=backend,
    )

    # Forward and adjoint
    db = Bop * d.astype(par["dtype"])
    dadj = Bop.H * db
    assert db.dtype == par["dtype"]
    assert dadj.dtype == par["dtype"]


@pytest.mark.parametrize("par", [(par1), (par2), (par3)])
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
    assert isinstance(Bop.dtype, npp.dtype)
    assert Bop.dtype == npp.dtype(par["dtype"])
    assert dottest(
        Bop,
        par["nt"] * n_groups * par["nr"],
        par["nt"] * par["ns"] * par["nr"],
        rtol=1e-4 if par["dtype"] == np.float32 else 1e-6,
        backend=backend,
    )

    # Forward and adjoint
    db = Bop * d.astype(par["dtype"])
    dadj = Bop.H * db
    assert db.dtype == par["dtype"]
    assert dadj.dtype == par["dtype"]
