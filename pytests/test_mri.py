import os

import numpy as np
import pytest

from pylops.medical import MRI2D
from pylops.utils import dottest, mkl_fft_enabled

par1 = {
    "ny": 32,
    "nx": 64,
    "dtype": "complex128",
    "fft_engine": "numpy",
}  # even input, numpy engine
par2 = {
    "ny": 32,
    "nx": 64,
    "dtype": "complex128",
    "fft_engine": "scipy",
}  # even input, scipy engine
par3 = {
    "ny": 32,
    "nx": 64,
    "dtype": "complex128",
    "fft_engine": "mkl_fft",
}  # even input, mkl_fft engine
par4 = {
    "ny": 33,
    "nx": 65,
    "dtype": "complex128",
    "fft_engine": "numpy",
}  # odd input, numpy engine
par5 = {
    "ny": 33,
    "nx": 65,
    "dtype": "complex128",
    "fft_engine": "scipy",
}  # even input, scipy engine
par6 = {
    "ny": 33,
    "nx": 65,
    "dtype": "complex128",
    "fft_engine": "mkl_fft",
}  # odd input, mkl_fft engine


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
def test_MRI2D_invalid_mask():
    """Test MRI2D operator with invalid mask string"""
    with pytest.raises(ValueError, match="mask must be"):
        MRI2D(
            dims=(32, 64),
            mask="invalid-mask",
            nlines=16,
            dtype="complex128",
        )


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
def test_MRI2D_invalid_engine():
    """Test MRI2D operator with invalid engine"""
    with pytest.raises(ValueError, match="engine must be"):
        MRI2D(
            dims=(32, 64),
            mask="vertical-reg",
            fft_engine="invalid-engine",
            dtype="complex128",
        )


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
def test_MRI2D_vertical_reg_invalid_perc_center():
    """Test MRI2D operator with vertical-reg mask and non-zero perc_center"""
    with pytest.raises(ValueError, match="perc_center must be 0.0"):
        MRI2D(
            dims=(32, 64),
            mask="vertical-reg",
            nlines=16,
            perc_center=0.1,
            dtype="complex128",
        )


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
def test_MRI2D_vertical_mask_invalid_nlines():
    """Test MRI2D operator with vertical mask and invalid nlines"""
    with pytest.raises(ValueError, match="nlines and perc_center"):
        MRI2D(
            dims=(32, 64),
            mask="vertical-uni",
            nlines=60,
            perc_center=0.5,
            dtype="complex128",
        )


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4), (par5), (par6)])
def test_MRI2D_mask_array(par):
    """Dot-test and forward/adjoint for MRI2D operator with numpy array mask"""
    if par["fft_engine"] == "mkl_fft" and not mkl_fft_enabled:
        pytest.skip("mkl_fft is not installed")
    np.random.seed(10)

    # Create a random mask
    mask = np.zeros((par["ny"], par["nx"]), dtype=bool)
    nselected = int(par["ny"] * par["nx"] * 0.3)
    indices = np.random.choice(par["ny"] * par["nx"], nselected, replace=False)
    mask.flat[indices] = True
    mask = mask.astype(par["dtype"])

    Mop = MRI2D(
        dims=(par["ny"], par["nx"]),
        mask=mask,
        fft_engine=par["fft_engine"],
        dtype=par["dtype"],
    )
    assert dottest(
        Mop,
        par["ny"] * par["nx"],
        par["ny"] * par["nx"],
        complexflag=2,
    )

    x = np.random.normal(0, 1, (par["ny"], par["nx"]))
    y = Mop * x.ravel()
    xadj = Mop.H * y

    assert y.shape[0] == par["ny"] * par["nx"]
    assert xadj.shape[0] == par["ny"] * par["nx"]


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4), (par5), (par6)])
def test_MRI2D_vertical_reg(par):
    """Dot-test and forward/adjoint for MRI2D operator with vertical-reg mask"""
    if par["fft_engine"] == "mkl_fft" and not mkl_fft_enabled:
        pytest.skip("mkl_fft is not installed")
    np.random.seed(10)

    nlines = 16
    Mop = MRI2D(
        dims=(par["ny"], par["nx"]),
        mask="vertical-reg",
        nlines=nlines,
        perc_center=0.0,
        fft_engine=par["fft_engine"],
        dtype=par["dtype"],
    )
    assert len(Mop.mask) == nlines
    assert dottest(
        Mop,
        par["ny"] * nlines,
        par["ny"] * par["nx"],
        complexflag=2,
    )

    x = np.random.normal(0, 1, (par["ny"], par["nx"]))
    y = Mop * x.ravel()
    xadj = Mop.H * y

    assert y.shape[0] == par["ny"] * nlines
    assert xadj.shape[0] == par["ny"] * par["nx"]


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4), (par5), (par6)])
def test_MRI2D_vertical_mask_regularity(par):
    """Test that vertical-reg mask produces regularly spaced lines"""
    if par["fft_engine"] == "mkl_fft" and not mkl_fft_enabled:
        pytest.skip("mkl_fft is not installed")
    np.random.seed(10)

    nlines = 8
    Mop = MRI2D(
        dims=(par["ny"], par["nx"]),
        mask="vertical-reg",
        nlines=nlines,
        perc_center=0.0,
        fft_engine=par["fft_engine"],
        dtype=par["dtype"],
    )
    mask_indices = Mop.mask

    # Check that indices are regularly spaced
    if len(mask_indices) > 1:
        steps = np.diff(np.sort(mask_indices))
        # All steps should be approximately equal (within rounding)
        assert np.allclose(steps, steps[0], atol=1)


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4), (par5), (par6)])
def test_MRI2D_vertical_uni(par):
    """Dot-test and forward/adjoint for MRI2D operator with vertical-uni mask"""
    if par["fft_engine"] == "mkl_fft" and not mkl_fft_enabled:
        pytest.skip("mkl_fft is not installed")
    np.random.seed(10)

    nlines = 16
    perc_center = 0.1
    Mop = MRI2D(
        dims=(par["ny"], par["nx"]),
        mask="vertical-uni",
        nlines=nlines,
        perc_center=perc_center,
        fft_engine=par["fft_engine"],
        dtype=par["dtype"],
    )
    nlines_total = nlines + int(perc_center * par["nx"])
    assert len(Mop.mask) == nlines_total
    assert dottest(
        Mop,
        par["ny"] * (nlines + int(perc_center * par["nx"])),
        par["ny"] * par["nx"],
        complexflag=2,
    )

    x = np.random.normal(0, 1, (par["ny"], par["nx"]))
    y = Mop * x.ravel()
    xadj = Mop.H * y

    assert y.shape[0] == par["ny"] * nlines_total
    assert xadj.shape[0] == par["ny"] * par["nx"]


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4), (par5), (par6)])
def test_MRI2D_vertical_uni_no_center(par):
    """Test MRI2D operator with vertical mask and no center lines"""
    if par["fft_engine"] == "mkl_fft" and not mkl_fft_enabled:
        pytest.skip("mkl_fft is not installed")
    np.random.seed(10)

    nlines = 16
    Mop = MRI2D(
        dims=(par["ny"], par["nx"]),
        mask="vertical-uni",
        nlines=nlines,
        perc_center=0.0,
        fft_engine=par["fft_engine"],
        dtype=par["dtype"],
    )

    assert dottest(
        Mop,
        par["ny"] * nlines,
        par["ny"] * par["nx"],
        complexflag=2,
    )
    assert len(Mop.mask) == nlines

    x = np.random.normal(0, 1, (par["ny"], par["nx"]))
    y = Mop * x.ravel()
    xadj = Mop.H * y

    assert y.shape[0] == par["ny"] * nlines
    assert xadj.shape[0] == par["ny"] * par["nx"]


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4), (par5), (par6)])
def test_MRI2D_radial_reg(par):
    """Dot-test and forward/adjoint for MRI2D operator with radial-reg mask"""
    if par["fft_engine"] == "mkl_fft" and not mkl_fft_enabled:
        pytest.skip("mkl_fft is not installed")
    np.random.seed(10)

    nlines = 8
    Mop = MRI2D(
        dims=(par["ny"], par["nx"]),
        mask="radial-reg",
        nlines=nlines,
        fft_engine=par["fft_engine"],
        dtype=par["dtype"],
    )

    # For radial masks, output size depends on the number of points in the mask
    npoints = Mop.mask.shape[1]
    assert dottest(
        Mop,
        npoints,
        par["ny"] * par["nx"],
        complexflag=2,
    )
    assert Mop.mask.shape[0] == 2  # x and y coordinates

    x = np.random.normal(0, 1, (par["ny"], par["nx"]))
    y = Mop * x
    xadj = Mop.H * y

    assert y.size == npoints
    assert xadj.shape == (par["ny"], par["nx"])


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4), (par5), (par6)])
def test_MRI2D_radial_uni(par):
    """Dot-test and forward/adjoint for MRI2D operator with radial-uni mask"""
    if par["fft_engine"] == "mkl_fft" and not mkl_fft_enabled:
        pytest.skip("mkl_fft is not installed")
    np.random.seed(10)

    nlines = 8
    Mop = MRI2D(
        dims=(par["ny"], par["nx"]),
        mask="radial-uni",
        nlines=nlines,
        fft_engine=par["fft_engine"],
        dtype=par["dtype"],
    )

    # For radial masks, output size depends on the number of points in the mask
    npoints = Mop.mask.shape[1]
    assert dottest(
        Mop,
        npoints,
        par["ny"] * par["nx"],
        complexflag=2,
    )
    assert Mop.mask.shape[0] == 2  # x and y coordinates

    x = np.random.normal(0, 1, (par["ny"], par["nx"]))
    y = Mop * x
    xadj = Mop.H * y

    assert y.size == npoints
    assert xadj.shape == (par["ny"], par["nx"])
