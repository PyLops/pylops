import numpy as np
import pytest

from pylops.utils.metrics import mae, mse, psnr, snr

par1 = {"nx": 11, "dtype": "float64"}  # float64
par2 = {"nx": 11, "dtype": "float32"}  # float32


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_mae(par):
    """Check MAE with same vector and vector of zeros"""
    xref = np.ones(par["nx"])
    xcmp = np.zeros(par["nx"])

    maesame = mae(xref, xref)
    maecmp = mae(xref, xcmp)
    assert maesame == 0.0
    assert maecmp == 1.0


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_mse(par):
    """Check MSE with same vector and vector of zeros"""
    xref = np.ones(par["nx"])
    xcmp = np.zeros(par["nx"])

    msesame = mse(xref, xref)
    msecmp = mse(xref, xcmp)
    assert msesame == 0.0
    assert msecmp == 1.0


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_snr(par):
    """Check SNR with same vector and vector of zeros"""
    xref = np.random.normal(0, 1, par["nx"])
    xcmp = np.zeros(par["nx"])

    snrsame = snr(xref, xref)
    snrcmp = snr(xref, xcmp)
    assert snrsame == np.inf
    assert snrcmp == 0.0


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_psnr(par):
    """Check PSNR with same vector and vector of zeros"""
    xref = np.ones(par["nx"])
    xcmp = np.zeros(par["nx"])

    psnrsame = psnr(xref, xref, xmax=1.0)
    psnrcmp = psnr(xref, xcmp, xmax=1.0)
    assert psnrsame == np.inf
    assert psnrcmp == 0.0
