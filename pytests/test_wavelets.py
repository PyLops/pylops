import numpy as np
import pytest

from pylops.utils.wavelets import gaussian, ricker

par1 = {"nt": 21, "dt": 0.004}  # odd samples
par2 = {"nt": 20, "dt": 0.004}  # even samples


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_ricker(par):
    """Create ricker wavelet and check size and central value"""
    t = np.arange(par["nt"]) * par["dt"]
    wav, twav, wcenter = ricker(t, f0=20)

    assert twav.size == (par["nt"] - 1 if par["nt"] % 2 == 0 else par["nt"]) * 2 - 1
    assert wav.shape[0] == (par["nt"] - 1 if par["nt"] % 2 == 0 else par["nt"]) * 2 - 1
    assert wav[wcenter] == 1


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_gaussian(par):
    """Create gaussian wavelet and check size and central value"""
    t = np.arange(par["nt"]) * par["dt"]
    wav, twav, wcenter = gaussian(t, std=10)

    assert twav.size == (par["nt"] - 1 if par["nt"] % 2 == 0 else par["nt"]) * 2 - 1
    assert wav.shape[0] == (par["nt"] - 1 if par["nt"] % 2 == 0 else par["nt"]) * 2 - 1
    assert wav[wcenter] == 1
