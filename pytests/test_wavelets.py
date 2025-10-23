import os

import numpy as np
import pytest

from pylops.utils.wavelets import gaussian, klauder, ormsby, ricker

par1 = {"nt": 21, "dt": 0.004}  # odd samples
par2 = {"nt": 20, "dt": 0.004}  # even samples


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par2)])
def test_gaussian(par):
    """Create gaussian wavelet and check size and central value"""
    t = np.arange(par["nt"]) * par["dt"]
    wav, twav, wcenter = gaussian(t, std=10)

    assert twav.size == (par["nt"] - 1 if par["nt"] % 2 == 0 else par["nt"]) * 2 - 1
    assert wav.shape[0] == (par["nt"] - 1 if par["nt"] % 2 == 0 else par["nt"]) * 2 - 1
    assert wav[wcenter] == 1


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par2)])
def test_klauder(par):
    """Create klauder wavelet and check size and central value"""
    t = np.arange(par["nt"]) * par["dt"]
    wav, twav, wcenter = klauder(t, f=(10, 20))

    assert twav.size == (par["nt"] - 1 if par["nt"] % 2 == 0 else par["nt"]) * 2 - 1
    assert wav.shape[0] == (par["nt"] - 1 if par["nt"] % 2 == 0 else par["nt"]) * 2 - 1
    assert wav[wcenter] == 1


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par2)])
def test_ormsby(par):
    """Create ormsby wavelet and check size and central value"""
    t = np.arange(par["nt"]) * par["dt"]
    wav, twav, wcenter = ormsby(t, f=(5, 10, 25, 30))

    assert twav.size == (par["nt"] - 1 if par["nt"] % 2 == 0 else par["nt"]) * 2 - 1
    assert wav.shape[0] == (par["nt"] - 1 if par["nt"] % 2 == 0 else par["nt"]) * 2 - 1
    assert wav[wcenter] == 1


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par2)])
def test_ricker(par):
    """Create ricker wavelet and check size and central value"""
    t = np.arange(par["nt"]) * par["dt"]
    wav, twav, wcenter = ricker(t, f0=20)

    assert twav.size == (par["nt"] - 1 if par["nt"] % 2 == 0 else par["nt"]) * 2 - 1
    assert wav.shape[0] == (par["nt"] - 1 if par["nt"] % 2 == 0 else par["nt"]) * 2 - 1
    assert wav[wcenter] == 1
