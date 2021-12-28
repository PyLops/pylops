import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from scipy.sparse.linalg import lsqr

from pylops.signalprocessing import FFT, FFT2D, FFTND
from pylops.utils import dottest

par1 = {
    "nt": 41,
    "nx": 31,
    "ny": 10,
    "nfft": None,
    "real": False,
    "engine": "numpy",
    "ifftshift_before": False,
    "dtype": np.complex128,
}  # nfft=nt, complex input, numpy engine
par2 = {
    "nt": 41,
    "nx": 31,
    "ny": 10,
    "nfft": 64,
    "real": False,
    "engine": "numpy",
    "ifftshift_before": False,
    "dtype": np.complex64,
}  # nfft>nt, complex input, numpy engine
par3 = {
    "nt": 41,
    "nx": 31,
    "ny": 10,
    "nfft": None,
    "real": True,
    "engine": "numpy",
    "ifftshift_before": False,
    "dtype": np.float64,
}  # nfft=nt, real input, numpy engine
par4 = {
    "nt": 41,
    "nx": 31,
    "ny": 10,
    "nfft": 64,
    "real": True,
    "engine": "numpy",
    "ifftshift_before": False,
    "dtype": np.float64,
}  # nfft>nt, real input, numpy engine
par5 = {
    "nt": 41,
    "nx": 31,
    "ny": 10,
    "nfft": 64,
    "real": True,
    "engine": "numpy",
    "ifftshift_before": True,
    "dtype": np.float32,
}  # nfft>nt, real input and ifftshift_before, numpy engine
par1w = {
    "nt": 41,
    "nx": 31,
    "ny": 10,
    "nfft": None,
    "real": False,
    "engine": "fftw",
    "ifftshift_before": False,
    "dtype": np.complex128,
}  # nfft=nt, complex input, fftw engine
par2w = {
    "nt": 41,
    "nx": 31,
    "ny": 10,
    "nfft": 64,
    "real": False,
    "engine": "fftw",
    "ifftshift_before": False,
    "dtype": np.complex128,
}  # nfft>nt, complex input, fftw engine
par3w = {
    "nt": 41,
    "nx": 31,
    "ny": 10,
    "nfft": None,
    "real": True,
    "engine": "fftw",
    "ifftshift_before": False,
    "dtype": np.float64,
}  # nfft=nt, real input, fftw engine
par4w = {
    "nt": 41,
    "nx": 31,
    "ny": 10,
    "nfft": 64,
    "real": True,
    "engine": "fftw",
    "ifftshift_before": False,
    "dtype": np.float32,
}  # nfft>nt, real input, fftw engine

np.random.seed(5)


@pytest.mark.parametrize("par", [(par1)])
def test_unknown_engine(par):
    """Check error is raised if unknown engine is passed"""
    with pytest.raises(NotImplementedError):
        _ = FFT(
            dims=[par["nt"]],
            nfft=par["nfft"],
            sampling=0.005,
            real=par["real"],
            engine="foo",
        )


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par5), (par1w), (par2w), (par3w), (par4w)]
)
def test_FFT_1dsignal(par):
    """Dot-test and inversion for FFT operator for 1d signal"""
    decimal = 3 if np.real(np.ones(1, par["dtype"])).dtype == np.float32 else 8

    dt = 0.005
    t = np.arange(par["nt"]) * dt
    f0 = 10
    x = np.sin(2 * np.pi * f0 * t)
    x = x.astype(par["dtype"])
    nfft = par["nt"] if par["nfft"] is None else par["nfft"]

    FFTop = FFT(
        dims=[par["nt"]],
        nfft=nfft,
        sampling=dt,
        real=par["real"],
        engine=par["engine"],
        dtype=par["dtype"],
    )

    if par["real"]:
        assert dottest(
            FFTop, nfft // 2 + 1, par["nt"], complexflag=2, tol=10 ** (-decimal)
        )
    else:
        assert dottest(FFTop, nfft, par["nt"], complexflag=2, tol=10 ** (-decimal))
        assert dottest(FFTop, nfft, par["nt"], complexflag=3, tol=10 ** (-decimal))

    y = FFTop * x
    xadj = FFTop.H * y  # adjoint is same as inverse for fft
    xinv = lsqr(FFTop, y, damp=1e-10, iter_lim=10, show=0)[0]

    assert_array_almost_equal(x, xadj, decimal=decimal)
    assert_array_almost_equal(x, xinv, decimal=decimal)

    if not par["real"]:
        FFTop_fftshift = FFT(
            dims=[par["nt"]],
            nfft=nfft,
            sampling=dt,
            real=par["real"],
            ifftshift_before=par["ifftshift_before"],
            fftshift_after=True,
            engine=par["engine"],
            dtype=par["dtype"],
        )
        assert_array_almost_equal(FFTop_fftshift.f, np.fft.fftshift(FFTop.f))

        y_fftshift = FFTop_fftshift * x
        assert_array_almost_equal(y_fftshift, np.fft.fftshift(y))

        xadj = FFTop_fftshift.H * y_fftshift  # adjoint is same as inverse for fft
        xinv = lsqr(FFTop_fftshift, y_fftshift, damp=1e-10, iter_lim=10, show=0)[0]
        assert_array_almost_equal(x, xadj, decimal=decimal)
        assert_array_almost_equal(x, xinv, decimal=decimal)


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par5), (par1w), (par2w), (par3w), (par4w)]
)
def test_FFT_2dsignal(par):
    """Dot-test and inversion for fft operator for 2d signal
    (fft on single dimension)
    """
    decimal = 3 if np.real(np.ones(1, par["dtype"])).dtype == np.float32 else 8

    dt = 0.005
    nt, nx = par["nt"], par["nx"]
    t = np.arange(nt) * dt
    f0 = 10
    d = np.outer(np.sin(2 * np.pi * f0 * t), np.arange(nx) + 1)
    d = d.astype(par["dtype"])

    # 1st dimension
    nfft = par["nt"] if par["nfft"] is None else par["nfft"]
    FFTop = FFT(
        dims=(nt, nx),
        dir=0,
        nfft=nfft,
        sampling=dt,
        real=par["real"],
        engine=par["engine"],
        dtype=par["dtype"],
    )

    if par["real"]:
        assert dottest(
            FFTop, (nfft // 2 + 1) * nx, nt * nx, complexflag=2, tol=10 ** (-decimal)
        )
    else:
        assert dottest(FFTop, nfft * nx, nt * nx, complexflag=2, tol=10 ** (-decimal))
        assert dottest(FFTop, nfft * nx, nt * nx, complexflag=3, tol=10 ** (-decimal))

    D = FFTop * d.ravel()
    dadj = FFTop.H * D  # adjoint is same as inverse for fft
    dinv = lsqr(FFTop, D, damp=1e-10, iter_lim=10, show=0)[0]

    dadj = np.real(dadj.reshape(nt, nx))
    dinv = np.real(dinv.reshape(nt, nx))

    assert_array_almost_equal(d, dadj, decimal=decimal)
    assert_array_almost_equal(d, dinv, decimal=decimal)

    if not par["real"]:
        FFTop_fftshift = FFT(
            dims=(nt, nx),
            dir=0,
            nfft=nfft,
            sampling=dt,
            real=par["real"],
            fftshift_after=True,
            engine=par["engine"],
            dtype=par["dtype"],
        )
        assert_array_almost_equal(FFTop_fftshift.f, np.fft.fftshift(FFTop.f))

        D_fftshift = FFTop_fftshift * d.flatten()
        D2 = np.fft.fftshift(D.reshape(nfft, nx), axes=0).flatten()
        assert_array_almost_equal(D_fftshift, D2)

        dadj = FFTop_fftshift.H * D_fftshift  # adjoint is same as inverse for fft
        dinv = lsqr(FFTop_fftshift, D_fftshift, damp=1e-10, iter_lim=10, show=0)[0]

        dadj = np.real(dadj.reshape(nt, nx))
        dinv = np.real(dinv.reshape(nt, nx))

        assert_array_almost_equal(d, dadj, decimal=decimal)
        assert_array_almost_equal(d, dinv, decimal=decimal)

    # 2nd dimension
    nfft = par["nx"] if par["nfft"] is None else par["nfft"]
    FFTop = FFT(
        dims=(nt, nx),
        dir=1,
        nfft=nfft,
        sampling=dt,
        real=par["real"],
        engine=par["engine"],
        dtype=par["dtype"],
    )

    if par["real"]:
        assert dottest(
            FFTop, nt * (nfft // 2 + 1), nt * nx, complexflag=2, tol=10 ** (-decimal)
        )
    else:
        assert dottest(FFTop, nt * nfft, nt * nx, complexflag=2, tol=10 ** (-decimal))
        assert dottest(FFTop, nt * nfft, nt * nx, complexflag=3, tol=10 ** (-decimal))

    D = FFTop * d.ravel()
    dadj = FFTop.H * D  # adjoint is inverse for fft
    dinv = lsqr(FFTop, D, damp=1e-10, iter_lim=10, show=0)[0]

    dadj = np.real(dadj.reshape(nt, nx))
    dinv = np.real(dinv.reshape(nt, nx))

    assert_array_almost_equal(d, dadj, decimal=decimal)
    assert_array_almost_equal(d, dinv, decimal=decimal)

    if not par["real"]:
        FFTop_fftshift = FFT(
            dims=(nt, nx),
            dir=1,
            nfft=nfft,
            sampling=dt,
            real=par["real"],
            fftshift_after=True,
            engine=par["engine"],
            dtype=par["dtype"],
        )
        assert_array_almost_equal(FFTop_fftshift.f, np.fft.fftshift(FFTop.f))

        D_fftshift = FFTop_fftshift * d.flatten()
        D2 = np.fft.fftshift(D.reshape(nt, nfft), axes=1).flatten()
        assert_array_almost_equal(D_fftshift, D2)

        dadj = FFTop_fftshift.H * D_fftshift  # adjoint is same as inverse for fft
        dinv = lsqr(FFTop_fftshift, D_fftshift, damp=1e-10, iter_lim=10, show=0)[0]

        dadj = np.real(dadj.reshape(nt, nx))
        dinv = np.real(dinv.reshape(nt, nx))

        assert_array_almost_equal(d, dadj, decimal=decimal)
        assert_array_almost_equal(d, dinv, decimal=decimal)


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par5), (par1w), (par2w), (par3w), (par4w)]
)
def test_FFT_3dsignal(par):
    """Dot-test and inversion for fft operator for 3d signal
    (fft on single dimension)
    """
    decimal = 3 if np.real(np.ones(1, par["dtype"])).dtype == np.float32 else 8

    dt = 0.005
    nt, nx, ny = par["nt"], par["nx"], par["ny"]
    t = np.arange(nt) * dt
    f0 = 10
    d = np.outer(np.sin(2 * np.pi * f0 * t), np.arange(nx) + 1)
    d = np.tile(d[:, :, np.newaxis], [1, 1, ny])
    d = d.astype(par["dtype"])

    # 1st dimension
    nfft = par["nt"] if par["nfft"] is None else par["nfft"]
    FFTop = FFT(
        dims=(nt, nx, ny),
        dir=0,
        nfft=nfft,
        sampling=dt,
        real=par["real"],
        engine=par["engine"],
        dtype=par["dtype"],
    )

    if par["real"]:
        assert dottest(
            FFTop,
            (nfft // 2 + 1) * nx * ny,
            nt * nx * ny,
            complexflag=2,
            tol=10 ** (-decimal),
        )
    else:
        assert dottest(
            FFTop, nfft * nx * ny, nt * nx * ny, complexflag=2, tol=10 ** (-decimal)
        )
        assert dottest(
            FFTop, nfft * nx * ny, nt * nx * ny, complexflag=3, tol=10 ** (-decimal)
        )

    D = FFTop * d.ravel()
    dadj = FFTop.H * D  # adjoint is same as inverse for fft
    dinv = lsqr(FFTop, D, damp=1e-10, iter_lim=10, show=0)[0]

    dadj = np.real(dadj.reshape(nt, nx, ny))
    dinv = np.real(dinv.reshape(nt, nx, ny))

    assert_array_almost_equal(d, dadj, decimal=decimal)
    assert_array_almost_equal(d, dinv, decimal=decimal)

    # 2nd dimension
    nfft = par["nx"] if par["nfft"] is None else par["nfft"]
    FFTop = FFT(
        dims=(nt, nx, ny),
        dir=1,
        nfft=nfft,
        sampling=dt,
        real=par["real"],
        engine=par["engine"],
        dtype=par["dtype"],
    )

    if par["real"]:
        assert dottest(
            FFTop,
            nt * (nfft // 2 + 1) * ny,
            nt * nx * ny,
            complexflag=2,
            tol=10 ** (-decimal),
        )
    else:
        assert dottest(
            FFTop, nt * nfft * ny, nt * nx * ny, complexflag=2, tol=10 ** (-decimal)
        )
        assert dottest(
            FFTop, nt * nfft * ny, nt * nx * ny, complexflag=3, tol=10 ** (-decimal)
        )

    D = FFTop * d.ravel()
    dadj = FFTop.H * D  # adjoint is inverse for fft
    dinv = lsqr(FFTop, D, damp=1e-10, iter_lim=10, show=0)[0]

    dadj = np.real(dadj.reshape(nt, nx, ny))
    dinv = np.real(dinv.reshape(nt, nx, ny))

    assert_array_almost_equal(d, dadj, decimal=decimal)
    assert_array_almost_equal(d, dinv, decimal=decimal)

    # 3rd dimension
    nfft = par["ny"] if par["nfft"] is None else par["nfft"]
    FFTop = FFT(
        dims=(nt, nx, ny),
        dir=2,
        nfft=nfft,
        sampling=dt,
        real=par["real"],
        engine=par["engine"],
        dtype=par["dtype"],
    )

    if par["real"]:
        assert dottest(
            FFTop,
            nt * nx * (nfft // 2 + 1),
            nt * nx * ny,
            complexflag=2,
            tol=10 ** (-decimal),
        )
    else:
        assert dottest(
            FFTop, nt * nx * nfft, nt * nx * ny, complexflag=2, tol=10 ** (-decimal)
        )
        assert dottest(
            FFTop, nt * nx * nfft, nt * nx * ny, complexflag=3, tol=10 ** (-decimal)
        )

    D = FFTop * d.ravel()
    dadj = FFTop.H * D  # adjoint is inverse for fft
    dinv = lsqr(FFTop, D, damp=1e-10, iter_lim=10, show=0)[0]

    dadj = np.real(dadj.reshape(nt, nx, ny))
    dinv = np.real(dinv.reshape(nt, nx, ny))

    assert_array_almost_equal(d, dadj, decimal=decimal)
    assert_array_almost_equal(d, dinv, decimal=decimal)

    if not par["real"]:
        FFTop_fftshift = FFT(
            dims=(nt, nx, ny),
            dir=2,
            nfft=nfft,
            sampling=dt,
            real=par["real"],
            fftshift_after=True,
            engine=par["engine"],
            dtype=par["dtype"],
        )
        assert_array_almost_equal(FFTop_fftshift.f, np.fft.fftshift(FFTop.f))

        D_fftshift = FFTop_fftshift * d.flatten()
        D2 = np.fft.fftshift(D.reshape(nt, nx, nfft), axes=2).flatten()
        assert_array_almost_equal(D_fftshift, D2)

        dadj = FFTop_fftshift.H * D_fftshift  # adjoint is same as inverse for fft
        dinv = lsqr(FFTop_fftshift, D_fftshift, damp=1e-10, iter_lim=10, show=0)[0]

        dadj = np.real(dadj.reshape(nt, nx, ny))
        dinv = np.real(dinv.reshape(nt, nx, ny))

        assert_array_almost_equal(d, dadj, decimal=decimal)
        assert_array_almost_equal(d, dinv, decimal=decimal)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_FFT2D(par):
    """Dot-test and inversion for FFT2D operator for 2d signal"""
    decimal = 3 if np.real(np.ones(1, par["dtype"])).dtype == np.float32 else 8

    dt, dx = 0.005, 5
    t = np.arange(par["nt"]) * dt
    f0 = 10
    nfft1 = par["nt"] if par["nfft"] is None else par["nfft"]
    nfft2 = par["nx"] if par["nfft"] is None else par["nfft"]
    d = np.outer(np.sin(2 * np.pi * f0 * t), np.arange(par["nx"]) + 1)
    d = d.astype(par["dtype"])

    # first fft on dir 1
    FFTop = FFT2D(
        dims=(par["nt"], par["nx"]),
        nffts=(nfft1, nfft2),
        sampling=(dt, dx),
        real=par["real"],
        dirs=(0, 1),
    )

    if par["real"]:
        assert dottest(
            FFTop,
            nfft1 * (nfft2 // 2 + 1),
            par["nt"] * par["nx"],
            complexflag=2,
            tol=10 ** (-decimal),
        )
    else:
        assert dottest(
            FFTop,
            nfft1 * nfft2,
            par["nt"] * par["nx"],
            complexflag=2,
            tol=10 ** (-decimal),
        )
        assert dottest(
            FFTop,
            nfft1 * nfft2,
            par["nt"] * par["nx"],
            complexflag=3,
            tol=10 ** (-decimal),
        )

    D = FFTop * d.ravel()
    dadj = FFTop.H * D  # adjoint is inverse for fft
    dinv = lsqr(FFTop, D, damp=1e-10, iter_lim=100, show=0)[0]

    dadj = np.real(dadj).reshape(par["nt"], par["nx"])
    dinv = np.real(dinv).reshape(par["nt"], par["nx"])

    assert_array_almost_equal(d, dadj, decimal=decimal)
    assert_array_almost_equal(d, dinv, decimal=decimal)

    # first fft on dir 0
    FFTop = FFT2D(
        dims=(par["nt"], par["nx"]),
        nffts=(nfft2, nfft1),
        sampling=(dx, dt),
        real=par["real"],
        dirs=(1, 0),
    )

    if par["real"]:
        assert dottest(
            FFTop,
            nfft2 * (nfft1 // 2 + 1),
            par["nt"] * par["nx"],
            complexflag=2,
            tol=10 ** (-decimal),
        )
    else:
        assert dottest(
            FFTop,
            nfft1 * nfft2,
            par["nt"] * par["nx"],
            complexflag=2,
            tol=10 ** (-decimal),
        )
        assert dottest(
            FFTop,
            nfft1 * nfft2,
            par["nt"] * par["nx"],
            complexflag=3,
            tol=10 ** (-decimal),
        )

    D = FFTop * d.ravel()
    dadj = FFTop.H * D  # adjoint is inverse for fft
    dinv = lsqr(FFTop, D, damp=1e-10, iter_lim=100, show=0)[0]

    dadj = np.real(dadj).reshape(par["nt"], par["nx"])
    dinv = np.real(dinv).reshape(par["nt"], par["nx"])

    assert_array_almost_equal(d, dadj, decimal=decimal)
    assert_array_almost_equal(d, dinv, decimal=decimal)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_FFT3D(par):
    """Dot-test and inversion for FFTND operator for 3d signal"""
    decimal = 3 if np.real(np.ones(1, par["dtype"])).dtype == np.float32 else 8

    dt, dx, dy = 0.005, 5, 2
    t = np.arange(par["nt"]) * dt
    f0 = 10
    nfft1 = par["nt"] if par["nfft"] is None else par["nfft"]
    nfft2 = par["nx"] if par["nfft"] is None else par["nfft"]
    nfft3 = par["ny"] if par["nfft"] is None else par["nfft"]
    d = np.outer(np.sin(2 * np.pi * f0 * t), np.arange(par["nx"]) + 1)
    d = np.tile(d[:, :, np.newaxis], [1, 1, par["ny"]])
    d = d.astype(par["dtype"])

    # first fft on dir 2
    FFTop = FFTND(
        dims=(par["nt"], par["nx"], par["ny"]),
        nffts=(nfft1, nfft2, nfft3),
        dirs=(0, 1, 2),
        sampling=(dt, dx, dy),
        real=par["real"],
    )

    if par["real"]:
        assert dottest(
            FFTop,
            nfft1 * nfft2 * (nfft3 // 2 + 1),
            par["nt"] * par["nx"] * par["ny"],
            complexflag=2,
            tol=10 ** (-decimal),
        )
    else:
        assert dottest(
            FFTop,
            nfft1 * nfft2 * nfft3,
            par["nt"] * par["nx"] * par["ny"],
            complexflag=2,
            tol=10 ** (-decimal),
        )
        assert dottest(
            FFTop,
            nfft1 * nfft2 * nfft3,
            par["nt"] * par["nx"] * par["ny"],
            complexflag=3,
            tol=10 ** (-decimal),
        )

    D = FFTop * d.ravel()
    dadj = FFTop.H * D  # adjoint is inverse for fft
    dinv = lsqr(FFTop, D, damp=1e-10, iter_lim=100, show=0)[0]

    dadj = np.real(dadj).reshape(par["nt"], par["nx"], par["ny"])
    dinv = np.real(dinv).reshape(par["nt"], par["nx"], par["ny"])

    assert_array_almost_equal(d, dadj, decimal=decimal)
    assert_array_almost_equal(d, dinv, decimal=decimal)

    # first fft on dir 1
    FFTop = FFTND(
        dims=(par["nt"], par["nx"], par["ny"]),
        nffts=(nfft1, nfft3, nfft2),
        dirs=(0, 2, 1),
        sampling=(dt, dy, dx),
        real=par["real"],
    )

    if par["real"]:
        assert dottest(
            FFTop,
            nfft1 * nfft3 * (nfft2 // 2 + 1),
            par["nt"] * par["nx"] * par["ny"],
            complexflag=2,
            tol=10 ** (-decimal),
        )
    else:
        assert dottest(
            FFTop,
            nfft1 * nfft2 * nfft3,
            par["nt"] * par["nx"] * par["ny"],
            complexflag=2,
            tol=10 ** (-decimal),
        )
        assert dottest(
            FFTop,
            nfft1 * nfft2 * nfft3,
            par["nt"] * par["nx"] * par["ny"],
            complexflag=3,
            tol=10 ** (-decimal),
        )

    D = FFTop * d.ravel()
    dadj = FFTop.H * D  # adjoint is inverse for fft
    dinv = lsqr(FFTop, D, damp=1e-10, iter_lim=100, show=0)[0]

    dadj = np.real(dadj).reshape(par["nt"], par["nx"], par["ny"])
    dinv = np.real(dinv).reshape(par["nt"], par["nx"], par["ny"])

    assert_array_almost_equal(d, dadj, decimal=decimal)
    assert_array_almost_equal(d, dinv, decimal=decimal)

    # first fft on dir 0
    FFTop = FFTND(
        dims=(par["nt"], par["nx"], par["ny"]),
        nffts=(nfft2, nfft3, nfft1),
        dirs=(1, 2, 0),
        sampling=(dx, dy, dt),
        real=par["real"],
    )

    if par["real"]:
        assert dottest(
            FFTop,
            nfft2 * nfft3 * (nfft1 // 2 + 1),
            par["nt"] * par["nx"] * par["ny"],
            complexflag=2,
            tol=10 ** (-decimal),
        )
    else:
        assert dottest(
            FFTop,
            nfft1 * nfft2 * nfft3,
            par["nt"] * par["nx"] * par["ny"],
            complexflag=2,
            tol=10 ** (-decimal),
        )
        assert dottest(
            FFTop,
            nfft1 * nfft2 * nfft3,
            par["nt"] * par["nx"] * par["ny"],
            complexflag=3,
            tol=10 ** (-decimal),
        )

    D = FFTop * d.ravel()
    dadj = FFTop.H * D  # adjoint is inverse for fft
    dinv = lsqr(FFTop, D, damp=1e-10, iter_lim=100, show=0)[0]

    dadj = np.real(dadj).reshape(par["nt"], par["nx"], par["ny"])
    dinv = np.real(dinv).reshape(par["nt"], par["nx"], par["ny"])

    assert_array_almost_equal(d, dadj, decimal=decimal)
    assert_array_almost_equal(d, dinv, decimal=decimal)
