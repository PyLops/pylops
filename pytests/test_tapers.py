import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pylops.utils.tapers import taper2d, taper3d

par1 = {
    "nt": 21,
    "nspat": (11, 13),
    "ntap": (3, 5),
    "tapertype": "hanning",
}  # hanning, odd samples and taper
par2 = {
    "nt": 20,
    "nspat": (12, 16),
    "ntap": (4, 6),
    "tapertype": "hanning",
}  # hanning, even samples and taper
par3 = {
    "nt": 21,
    "nspat": (11, 13),
    "ntap": (3, 5),
    "tapertype": "cosine",
}  # cosine, odd samples and taper
par4 = {
    "nt": 20,
    "nspat": (12, 16),
    "ntap": (4, 6),
    "tapertype": "cosine",
}  # cosine, even samples and taper
par5 = {
    "nt": 21,
    "nspat": (11, 13),
    "ntap": (3, 5),
    "tapertype": "cosinesquare",
}  # cosinesquare, odd samples and taper
par6 = {
    "nt": 20,
    "nspat": (12, 16),
    "ntap": (4, 6),
    "tapertype": "cosinesquare",
}  # cosinesquare, even samples and taper
par7 = {
    "nt": 21,
    "nspat": (11, 13),
    "ntap": (3, 5),
    "tapertype": "cosinesqrt",
}  # cosinesqrt, odd samples and taper
par8 = {
    "nt": 20,
    "nspat": (12, 16),
    "ntap": (4, 6),
    "tapertype": "cosinesqrt",
}  # cosinesqrt, even samples and taper


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par5), (par6), (par7), (par8)]
)
def test_taper2d(par):
    """Create taper wavelet and check size and values"""
    tap = taper2d(par["nt"], par["nspat"][0], par["ntap"][0], par["tapertype"])

    assert tap.shape == (par["nspat"][0], par["nt"])
    assert_array_equal(tap[0], np.zeros(par["nt"]))
    assert_array_equal(tap[-1], np.zeros(par["nt"]))
    assert_array_equal(tap[par["ntap"][0] + 1], np.ones(par["nt"]))
    assert_array_equal(tap[par["nspat"][0] // 2], np.ones(par["nt"]))


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par5), (par6), (par7), (par8)]
)
def test_taper3d(par):
    """Create taper wavelet and check size and values"""
    tap = taper3d(par["nt"], par["nspat"], par["ntap"], par["tapertype"])

    assert tap.shape == (par["nspat"][0], par["nspat"][1], par["nt"])
    assert_array_equal(tap[0][0], np.zeros(par["nt"]))
    assert_array_equal(tap[-1][-1], np.zeros(par["nt"]))
    assert_array_equal(tap[par["ntap"][0], par["ntap"][1]], np.ones(par["nt"]))
    assert_array_equal(
        tap[par["nspat"][0] // 2, par["nspat"][1] // 2], np.ones(par["nt"])
    )
