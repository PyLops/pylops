import os

import numpy as np
import pytest

from pylops.signalprocessing import DTCWT

# currently test only if numpy<2.0.0 is installed...
np_version = np.__version__.split(".")

par1 = {"ny": 10, "nx": 10, "dtype": "float64"}
par2 = {"ny": 50, "nx": 50, "dtype": "float64"}


def sequential_array(shape):
    num_elements = np.prod(shape)
    seq_array = np.arange(1, num_elements + 1)
    result = seq_array.reshape(shape)
    return result


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par2)])
def test_dtcwt1D_input1D(par):
    """Test for DTCWT with 1D input"""
    if int(np_version[0]) >= 2:
        return

    t = sequential_array((par["ny"],))

    for level in range(1, 10):
        Dtcwt = DTCWT(dims=t.shape, level=level, dtype=par["dtype"])
        x = Dtcwt @ t
        y = Dtcwt.H @ x

        np.testing.assert_allclose(t, y)


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par2)])
def test_dtcwt1D_input2D(par):
    """Test for DTCWT with 2D input (forward-inverse pair)"""
    if int(np_version[0]) >= 2:
        return

    t = sequential_array(
        (
            par["ny"],
            par["ny"],
        )
    )

    for level in range(1, 10):
        Dtcwt = DTCWT(dims=t.shape, level=level, dtype=par["dtype"])
        x = Dtcwt @ t
        y = Dtcwt.H @ x

        np.testing.assert_allclose(t, y)


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par2)])
def test_dtcwt1D_input3D(par):
    """Test for DTCWT with 3D input (forward-inverse pair)"""
    if int(np_version[0]) >= 2:
        return

    t = sequential_array((par["ny"], par["ny"], par["ny"]))

    for level in range(1, 10):
        Dtcwt = DTCWT(dims=t.shape, level=level, dtype=par["dtype"])
        x = Dtcwt @ t
        y = Dtcwt.H @ x

        np.testing.assert_allclose(t, y)


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par2)])
def test_dtcwt1D_birot(par):
    """Test for DTCWT birot (forward-inverse pair)"""
    if int(np_version[0]) >= 2:
        return

    birots = ["antonini", "legall", "near_sym_a", "near_sym_b"]

    t = sequential_array(
        (
            par["ny"],
            par["ny"],
        )
    )

    for _b in birots:
        print(f"birot {_b}")
        Dtcwt = DTCWT(dims=t.shape, biort=_b, dtype=par["dtype"])
        x = Dtcwt @ t
        y = Dtcwt.H @ x

        np.testing.assert_allclose(t, y)
