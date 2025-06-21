import itertools
import os

if int(os.environ.get("TEST_CUPY_PYLOPS", 0)):
    import cupy as np
    from cupy.testing import assert_array_almost_equal

    backend = "cupy"
else:
    import numpy as np
    from numpy.testing import assert_array_almost_equal

    backend = "numpy"
import numpy as npp
import pytest

from pylops.optimization.basic import lsqr
from pylops.signalprocessing import FFT, FFT2D, FFTND
from pylops.utils import dottest


# Utility function
def _choose_random_axes(ndim, n_choices=2):
    """Chooses `n_choices` random axes given an array of `ndim` dimensions.
    Examples:
        _choose_random_axes(2, 1) may return any of [0], [1], [-2] or [-1]
        _choose_random_axes(3, 2) may return any of [0, 1], [1, 0], [-2, -1],
            [-1, -2], [-2, 1], [1, -2], [0, -1] or [-1, 0].
    """
    if ndim < n_choices:
        raise ValueError("ndim < n_choices")
    axes_choices = list(range(-ndim, ndim))
    axes = []
    for _ in range(n_choices):
        axis_chosen = npp.random.choice(axes_choices)
        # Remove chosen and its symmetrical counterpart
        axes_choices.remove(axis_chosen)
        axes_choices.remove(axis_chosen - (1 if axis_chosen >= 0 else -1) * ndim)
        axes += [axis_chosen]
    return axes


par1 = {
    "nt": 41,
    "nx": 31,
    "ny": 10,
    "nfft": None,
    "real": False,
    "engine": "numpy",
    "ifftshift_before": False,
    "dtype": np.complex128,
    "kwargs": {},
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
    "kwargs": {},
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
    "kwargs": {},
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
    "kwargs": {},
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
    "kwargs": {},
}  # nfft>nt, real input and ifftshift_before, numpy engine
par6 = {
    "nt": 41,
    "nx": 31,
    "ny": 10,
    "nfft": 16,
    "real": False,
    "engine": "numpy",
    "ifftshift_before": False,
    "dtype": np.complex128,
    "kwargs": {},
}  # nfft<nt, complex input, scipy engine
par1s = {
    "nt": 41,
    "nx": 31,
    "ny": 10,
    "nfft": None,
    "real": False,
    "engine": "numpy",
    "ifftshift_before": False,
    "dtype": np.complex128,
    "kwargs": {},
}  # nfft=nt, complex input, scipy engine
par2s = {
    "nt": 41,
    "nx": 31,
    "ny": 10,
    "nfft": 64,
    "real": False,
    "engine": "numpy",
    "ifftshift_before": False,
    "dtype": np.complex64,
    "kwargs": {},
}  # nfft>nt, complex input, scipy engine
par3s = {
    "nt": 41,
    "nx": 31,
    "ny": 10,
    "nfft": None,
    "real": True,
    "engine": "numpy",
    "ifftshift_before": False,
    "dtype": np.float64,
    "kwargs": {},
}  # nfft=nt, real input, scipy engine
par4s = {
    "nt": 41,
    "nx": 31,
    "ny": 10,
    "nfft": 64,
    "real": True,
    "engine": "numpy",
    "ifftshift_before": False,
    "dtype": np.float64,
    "kwargs": {},
}  # nfft>nt, real input, scipy engine
par5s = {
    "nt": 41,
    "nx": 31,
    "ny": 10,
    "nfft": 16,
    "real": False,
    "engine": "numpy",
    "ifftshift_before": False,
    "dtype": np.complex128,
    "kwargs": {},
}  # nfft<nt, complex input, scipy engine
par6s = {
    "nt": 41,
    "nx": 31,
    "ny": 10,
    "nfft": 16,
    "real": False,
    "engine": "numpy",
    "ifftshift_before": False,
    "dtype": np.complex128,
    "kwargs": {"workers": 2},
}  # nfft<nt, complex input, scipy engine with workers
par1w = {
    "nt": 41,
    "nx": 31,
    "ny": 10,
    "nfft": None,
    "real": False,
    "engine": "fftw",
    "ifftshift_before": False,
    "dtype": np.complex128,
    "kwargs": {},
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
    "kwargs": {},
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
    "kwargs": {},
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
    "kwargs": {},
}  # nfft>nt, real input, fftw engine
par5w = {
    "nt": 41,
    "nx": 31,
    "ny": 10,
    "nfft": 16,
    "real": False,
    "engine": "fftw",
    "ifftshift_before": False,
    "dtype": np.complex128,
    "kwargs": {},
}  # nfft<nt, complex input, fftw engine

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


dtype_precision = [
    (np.float16, 1),
    (np.float32, 4),
    (np.float64, 11),
]
if backend == "numpy":
    dtype_precision.append((np.longdouble, 11))

par_lists_fft_small_real = dict(
    dtype_precision=dtype_precision,
    norm=["ortho", "none", "1/n"],
    ifftshift_before=[False, True],
    engine=["numpy", "fftw", "scipy"],
)
# Generate all combinations of the above parameters
pars_fft_small_real = [
    dict(zip(par_lists_fft_small_real.keys(), value))
    for value in itertools.product(*par_lists_fft_small_real.values())
]


@pytest.mark.parametrize("par", pars_fft_small_real)
def test_FFT_small_real(par):
    np.random.seed(5)

    if backend == "numpy" or (backend == "cupy" and par["engine"] == "numpy"):
        dtype, decimal = par["dtype_precision"]
        norm = par["norm"]
        ifftshift_before = par["ifftshift_before"]
        engine = par["engine"]

        x = np.array([1, 0, -1, 1], dtype=dtype)

        FFTop = FFT(
            dims=x.shape,
            axis=0,
            norm=norm,
            real=True,
            ifftshift_before=ifftshift_before,
            dtype=dtype,
            engine=engine,
        )
        FFTop.f = np.asarray(FFTop.f)
        y = FFTop * x.ravel()

        if norm == "ortho":
            y_true = np.array([0.5, 1 + 0.5j, -0.5], dtype=FFTop.cdtype)
        elif norm == "none":
            y_true = np.array([1, 2 + 1j, -1], dtype=FFTop.cdtype)
        elif norm == "1/n":
            y_true = np.array([0.25, 0.5 + 0.25j, -0.25], dtype=FFTop.cdtype)

        y_true[1:-1] *= np.sqrt(2)  # Zero and Nyquist
        if ifftshift_before:
            # `ifftshift_before`` is useful when the time-axis is centered around zero as
            # it ensures the time axis to starts at zero:
            #     [-2, -1, 0, 1] ---ifftshift--> [0, 1, -2, -1]
            # This does not alter the amplitude of the FFT, but does alter the phase. To
            # match the results without ifftshift, we need to add a phase shift opposite to
            # the one introduced by FFT as given below. See "An FFT Primer for physicists",
            # by Thomas Kaiser.
            # https://www.iap.uni-jena.de/iapmedia/de/Lecture/Computational+Photonics/CoPho19_supp_FFT_primer.pdf
            x0 = -np.ceil(len(x) / 2)
            y_true *= np.exp(2 * np.pi * 1j * FFTop.f * x0)

        assert_array_almost_equal(y, y_true, decimal=decimal)
        assert dottest(
            FFTop, len(y), len(x), complexflag=0, rtol=10 ** (-decimal), backend=backend
        )
        assert dottest(
            FFTop, len(y), len(x), complexflag=2, rtol=10 ** (-decimal), backend=backend
        )

        x_inv = FFTop / y
        x_inv = x_inv.reshape(x.shape)
        assert_array_almost_equal(x_inv, x, decimal=decimal)


par_lists_fft_random_real = dict(
    shape=[
        npp.random.randint(1, 20, size=(1,)),
        npp.random.randint(1, 20, size=(2,)),
        npp.random.randint(1, 20, size=(3,)),
    ],
    dtype_precision=dtype_precision,
    ifftshift_before=[False, True],
    engine=["numpy", "fftw", "scipy"],
)
pars_fft_random_real = [
    dict(zip(par_lists_fft_random_real.keys(), value))
    for value in itertools.product(*par_lists_fft_random_real.values())
]


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1,
    reason="Dot-test failure with CuPy enabled when running entire test suite",
)
@pytest.mark.parametrize("par", pars_fft_random_real)
def test_FFT_random_real(par):
    np.random.seed(5)

    shape = par["shape"]
    dtype, decimal = par["dtype_precision"]
    ifftshift_before = par["ifftshift_before"]

    x = np.random.randn(*shape).astype(dtype)

    # Select an axis to apply FFT on. It can be any integer
    # in [0,..., ndim-1] but also in [-ndim, ..., -1]
    axis = _choose_random_axes(x.ndim, n_choices=1)[0]

    FFTop = FFT(
        dims=x.shape,
        axis=axis,
        ifftshift_before=ifftshift_before,
        real=True,
        dtype=dtype,
    )
    x = x.ravel()
    y = FFTop * x

    # Ensure inverse and adjoint recover x
    xadj = FFTop.H * y  # adjoint is same as inverse for fft
    xinv = lsqr(FFTop, y, damp=0, niter=10, atol=1e-8, btol=1e-8, show=0)[0].ravel()
    assert_array_almost_equal(x, xadj, decimal=decimal)
    assert_array_almost_equal(x, xinv, decimal=decimal)

    # Dot tests
    nr, nc = FFTop.shape
    assert dottest(FFTop, nr, nc, complexflag=0, rtol=10 ** (-decimal), backend=backend)
    assert dottest(FFTop, nr, nc, complexflag=2, rtol=10 ** (-decimal), backend=backend)


dtype_precision_cpx = [
    (np.complex64, 4),
    (np.complex128, 11),
]
if backend == "numpy":
    dtype_precision_cpx.append((np.clongdouble, 11))

par_lists_fft_small_cpx = dict(
    dtype_precision=dtype_precision_cpx,
    norm=["ortho", "none", "1/n"],
    ifftshift_before=[False, True],
    fftshift_after=[False, True],
    engine=["numpy", "fftw", "scipy"],
)
pars_fft_small_cpx = [
    dict(zip(par_lists_fft_small_cpx.keys(), value))
    for value in itertools.product(*par_lists_fft_small_cpx.values())
]


@pytest.mark.parametrize("par", pars_fft_small_cpx)
def test_FFT_small_complex(par):
    np.random.seed(5)
    dtype, decimal = par["dtype_precision"]
    norm = par["norm"]
    ifftshift_before = par["ifftshift_before"]
    fftshift_after = par["fftshift_after"]

    x = np.array([1, 2 - 1j, -1j, -1 + 2j], dtype=dtype)

    FFTop = FFT(
        dims=x.shape,
        axis=0,
        norm=norm,
        ifftshift_before=ifftshift_before,
        fftshift_after=fftshift_after,
        dtype=dtype,
    )

    # Compute FFT of x independently
    if norm == "ortho":
        y_true = np.array([1, -1 - 1j, -1j, 2 + 2j], dtype=FFTop.cdtype)
    elif norm == "none":
        y_true = np.array([2, -2 - 2j, -2j, 4 + 4j], dtype=FFTop.cdtype)
    elif norm == "1/n":
        y_true = np.array([0.5, -0.5 - 0.5j, -0.5j, 1 + 1j], dtype=FFTop.cdtype)

    if fftshift_after:
        y_true = np.fft.fftshift(y_true)
    if ifftshift_before:
        x0 = -np.ceil(x.shape[0] / 2)
        y_true *= np.exp(2 * np.pi * 1j * np.asarray(FFTop.f) * x0)

    # Compute FFT with FFTop and compare with y_true
    y = FFTop * x.ravel()
    assert_array_almost_equal(y, y_true, decimal=decimal)
    assert dottest(
        FFTop, *FFTop.shape, complexflag=3, rtol=10 ** (-decimal), backend=backend
    )

    x_inv = FFTop / y
    x_inv = x_inv.reshape(x.shape)
    assert_array_almost_equal(x_inv, x, decimal=decimal)


dtype_precision_cpx1 = [
    (np.float16, 1),
    (np.float32, 3),
    (np.float64, 11),
    (np.complex64, 3),
    (np.complex128, 11),
]
if backend == "numpy":
    dtype_precision_cpx1.append((np.longdouble, 11))
    dtype_precision_cpx1.append((np.clongdouble, 11))
par_lists_fft_random_cpx = dict(
    shape=[
        npp.random.randint(1, 20, size=(1,)),
        npp.random.randint(1, 20, size=(2,)),
        npp.random.randint(1, 20, size=(3,)),
    ],
    dtype_precision=dtype_precision_cpx1,
    ifftshift_before=[False, True],
    fftshift_after=[False, True],
    engine=["numpy", "fftw", "scipy"],
)
pars_fft_random_cpx = [
    dict(zip(par_lists_fft_random_cpx.keys(), value))
    for value in itertools.product(*par_lists_fft_random_cpx.values())
]


@pytest.mark.parametrize("par", pars_fft_random_cpx)
def test_FFT_random_complex(par):
    np.random.seed(5)
    if backend == "numpy" or (backend == "cupy" and par["engine"] == "numpy"):
        shape = par["shape"]
        dtype, decimal = par["dtype_precision"]
        ifftshift_before = par["ifftshift_before"]
        fftshift_after = par["fftshift_after"]
        engine = par["engine"]

        x = np.random.randn(*shape).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            x += 1j * np.random.randn(*shape).astype(dtype)

        # Select an axis to apply FFT on. It can be any integer
        # in [0,..., ndim-1] but also in [-ndim, ..., -1]
        axis = _choose_random_axes(x.ndim, n_choices=1)[0]

        FFTop = FFT(
            dims=x.shape,
            axis=axis,
            ifftshift_before=ifftshift_before,
            fftshift_after=fftshift_after,
            dtype=dtype,
            engine=engine,
        )

        # Compute FFT of x independently
        y_true = np.fft.fft(x, axis=axis, norm="ortho")
        if fftshift_after:
            y_true = np.fft.fftshift(y_true, axes=int(axis))
        if ifftshift_before:
            y_true = np.swapaxes(y_true, axis, -1)
            x0 = -np.ceil(x.shape[axis] / 2)
            phase_correction = np.exp(2 * np.pi * 1j * np.asarray(FFTop.f) * x0)
            y_true *= phase_correction
            y_true = np.swapaxes(y_true, -1, axis)
        y_true = y_true.ravel()

        # Compute FFT with FFTop and compare with y_true
        x = x.ravel()
        y = FFTop * x
        assert_array_almost_equal(y, y_true, decimal=decimal)

        # Ensure inverse and adjoint recover x
        xadj = FFTop.H * y  # adjoint is same as inverse for fft
        xinv = lsqr(FFTop, y, damp=0, niter=10, atol=1e-8, btol=1e-8, show=0)[0].ravel()
        assert_array_almost_equal(x, xadj, decimal=decimal)
        assert_array_almost_equal(x, xinv, decimal=decimal)

        # Dot tests
        nr, nc = FFTop.shape
        assert dottest(
            FFTop, nr, nc, complexflag=0, rtol=10 ** (-decimal), backend=backend
        )
        assert dottest(
            FFTop, nr, nc, complexflag=2, rtol=10 ** (-decimal), backend=backend
        )
        if np.issubdtype(dtype, np.complexfloating):
            assert dottest(
                FFTop, nr, nc, complexflag=1, rtol=10 ** (-decimal), backend=backend
            )
            assert dottest(
                FFTop, nr, nc, complexflag=3, rtol=10 ** (-decimal), backend=backend
            )


par_lists_fft2d_random_real = dict(
    shape=[
        npp.random.randint(1, 5, size=(2,)),
        npp.random.randint(1, 5, size=(3,)),
        npp.random.randint(1, 5, size=(4,)),
    ],
    dtype_precision=dtype_precision,
    ifftshift_before=[False, True],
    engine=["numpy", "scipy"],
)
pars_fft2d_random_real = [
    dict(zip(par_lists_fft2d_random_real.keys(), value))
    for value in itertools.product(*par_lists_fft2d_random_real.values())
]


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1,
    reason="Dot-test failure with CuPy enabled when running entire test suite",
)
@pytest.mark.parametrize("par", pars_fft2d_random_real)
def test_FFT2D_random_real(par):
    np.random.seed(5)
    if backend == "numpy" or (backend == "cupy" and par["engine"] == "numpy"):
        shape = par["shape"]
        dtype, decimal = par["dtype_precision"]
        ifftshift_before = par["ifftshift_before"]
        engine = par["engine"]

        x = np.random.randn(*shape).astype(dtype)

        # Select an axis to apply FFT on. It can be any integer
        # in [0,..., ndim-1] but also in [-ndim, ..., -1]
        # However, dimensions cannot be repeated
        axes = _choose_random_axes(x.ndim, n_choices=2)

        FFTop = FFT2D(
            dims=x.shape,
            axes=axes,
            ifftshift_before=ifftshift_before,
            real=True,
            dtype=dtype,
            engine=engine,
        )
        x = x.ravel()
        y = FFTop * x

        # Ensure inverse and adjoint recover x
        xadj = FFTop.H * y  # adjoint is same as inverse for fft
        xinv = lsqr(FFTop, y, damp=0, niter=10, atol=1e-8, btol=1e-8, show=0)[0].ravel()
        assert_array_almost_equal(x, xadj, decimal=decimal)
        assert_array_almost_equal(x, xinv, decimal=decimal)

        # Dot tests
        nr, nc = FFTop.shape
        assert dottest(
            FFTop, nr, nc, complexflag=0, rtol=10 ** (-decimal), backend=backend
        )
        assert dottest(
            FFTop, nr, nc, complexflag=2, rtol=10 ** (-decimal), backend=backend
        )


par_lists_fft2d_random_cpx = dict(
    shape=[
        npp.random.randint(1, 5, size=(2,)),
        npp.random.randint(1, 5, size=(3,)),
        npp.random.randint(1, 5, size=(5,)),
    ],
    dtype_precision=dtype_precision_cpx1,
    ifftshift_before=itertools.product([False, True], [False, True]),
    fftshift_after=itertools.product([False, True], [False, True]),
    engine=["numpy", "scipy"],
)
# Generate all combinations of the above parameters
pars_fft2d_random_cpx = [
    dict(zip(par_lists_fft2d_random_cpx.keys(), value))
    for value in itertools.product(*par_lists_fft2d_random_cpx.values())
]


@pytest.mark.parametrize("par", pars_fft2d_random_cpx)
def test_FFT2D_random_complex(par):
    np.random.seed(5)
    if backend == "numpy" or (backend == "cupy" and par["engine"] == "numpy"):
        shape = par["shape"]
        dtype, decimal = par["dtype_precision"]
        ifftshift_before = par["ifftshift_before"]
        fftshift_after = par["fftshift_after"]
        engine = par["engine"]

        x = np.random.randn(*shape).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            x += 1j * np.random.randn(*shape).astype(dtype)

        # Select an axis to apply FFT on. It can be any integer
        # in [0,..., ndim-1] but also in [-ndim, ..., -1]
        # However, dimensions cannot be repeated
        axes = _choose_random_axes(x.ndim, n_choices=2)

        FFTop = FFT2D(
            dims=x.shape,
            axes=axes,
            ifftshift_before=ifftshift_before,
            fftshift_after=fftshift_after,
            dtype=dtype,
            engine=engine,
        )

        # Compute FFT of x independently
        x_ishift = x.copy()
        for axis, ishift in zip(axes, ifftshift_before):
            if ishift:
                x_ishift = np.fft.ifftshift(x_ishift, axes=int(axis))
        y_true = np.fft.fft2(x_ishift, axes=axes, norm="ortho")
        for axis, fshift in zip(axes, fftshift_after):
            if fshift:
                y_true = np.fft.fftshift(y_true, axes=int(axis))
        y_true = y_true.ravel()

        # Compute FFT with FFTop and compare with y_true
        x = x.ravel()
        y = FFTop * x
        assert_array_almost_equal(y, y_true, decimal=decimal)

        # Ensure inverse and adjoint recover x
        xadj = FFTop.H * y  # adjoint is same as inverse for fft
        xinv = lsqr(FFTop, y, damp=0, niter=10, atol=1e-8, btol=1e-8, show=0)[0].ravel()
        assert_array_almost_equal(x, xadj, decimal=decimal)
        assert_array_almost_equal(x, xinv, decimal=decimal)

        # Dot tests
        nr, nc = FFTop.shape
        assert dottest(
            FFTop, nr, nc, complexflag=0, rtol=10 ** (-decimal), backend=backend
        )
        assert dottest(
            FFTop, nr, nc, complexflag=2, rtol=10 ** (-decimal), backend=backend
        )
        if np.issubdtype(dtype, np.complexfloating):
            assert dottest(
                FFTop, nr, nc, complexflag=1, rtol=10 ** (-decimal), backend=backend
            )
            assert dottest(
                FFTop, nr, nc, complexflag=3, rtol=10 ** (-decimal), backend=backend
            )


par_lists_fftnd_random_real = dict(
    shape=[
        npp.random.randint(1, 5, size=(3,)),
        npp.random.randint(1, 5, size=(4,)),
    ],
    dtype_precision=dtype_precision,
    engine=["numpy", "scipy"],
)
pars_fftnd_random_real = [
    dict(zip(par_lists_fftnd_random_real.keys(), value))
    for value in itertools.product(*par_lists_fftnd_random_real.values())
]


@pytest.mark.parametrize("par", pars_fftnd_random_real)
def test_FFTND_random_real(par):
    np.random.seed(5)
    if backend == "numpy" or (backend == "cupy" and par["engine"] == "numpy"):
        shape = par["shape"]
        dtype, decimal = par["dtype_precision"]
        engine = par["engine"]

        x = np.random.randn(*shape).astype(dtype)

        # Select an axis to apply FFT on. It can be any integer
        # in [0,..., ndim-1] but also in [-ndim, ..., -1]
        # However, dimensions cannot be repeated
        n_choices = npp.random.randint(3, x.ndim + 1)
        axes = _choose_random_axes(x.ndim, n_choices=n_choices)

        # Trying out all posibilities is very cumbersome, let's select some shifts randomly
        ifftshift_before = npp.random.choice([False, True], size=n_choices)

        FFTop = FFTND(
            dims=x.shape,
            axes=axes,
            ifftshift_before=ifftshift_before,
            real=True,
            dtype=dtype,
            engine=engine,
        )
        x = x.ravel()
        y = FFTop * x

        # Ensure inverse and adjoint recover x
        xadj = FFTop.H * y  # adjoint is same as inverse for fft
        xinv = lsqr(FFTop, y, damp=0, niter=10, atol=1e-8, btol=1e-8, show=0)[0].ravel()
        assert_array_almost_equal(x, xadj, decimal=decimal)
        assert_array_almost_equal(x, xinv, decimal=decimal)

        # Dot tests
        nr, nc = FFTop.shape
        assert dottest(
            FFTop, nr, nc, complexflag=0, rtol=10 ** (-decimal), backend=backend
        )
        assert dottest(
            FFTop, nr, nc, complexflag=2, rtol=10 ** (-decimal), backend=backend
        )


par_lists_fftnd_random_cpx = dict(
    shape=[
        npp.random.randint(1, 5, size=(3,)),
        npp.random.randint(1, 5, size=(5,)),
    ],
    dtype_precision=dtype_precision_cpx1,
    engine=["numpy", "scipy"],
)
# Generate all combinations of the above parameters
pars_fftnd_random_cpx = [
    dict(zip(par_lists_fftnd_random_cpx.keys(), value))
    for value in itertools.product(*par_lists_fftnd_random_cpx.values())
]


@pytest.mark.parametrize("par", pars_fftnd_random_cpx)
def test_FFTND_random_complex(par):
    np.random.seed(5)
    shape = par["shape"]
    dtype, decimal = par["dtype_precision"]
    engine = par["engine"]

    x = np.random.randn(*shape).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        x += 1j * np.random.randn(*shape).astype(dtype)

    # Select an axis to apply FFT on. It can be any integer
    # in [0,..., ndim-1] but also in [-ndim, ..., -1]
    # However, dimensions cannot be repeated
    n_choices = npp.random.randint(3, x.ndim + 1)
    axes = _choose_random_axes(x.ndim, n_choices=n_choices)

    # Trying out all posibilities is very cumbersome, let's select some shifts randomly
    ifftshift_before = npp.random.choice([False, True], size=n_choices)
    fftshift_after = npp.random.choice([True, False], size=n_choices)

    FFTop = FFTND(
        dims=x.shape,
        axes=axes,
        ifftshift_before=ifftshift_before,
        fftshift_after=fftshift_after,
        dtype=dtype,
        engine=engine,
    )

    # Compute FFT of x independently
    x_ishift = x.copy()
    for axis, ishift in zip(axes, ifftshift_before):
        if ishift:
            x_ishift = np.fft.ifftshift(x_ishift, axes=int(axis))
    y_true = np.fft.fft2(x_ishift, axes=axes, norm="ortho")
    for axis, fshift in zip(axes, fftshift_after):
        if fshift:
            y_true = np.fft.fftshift(y_true, axes=int(axis))
    y_true = y_true.ravel()

    # Compute FFT with FFTop and compare with y_true
    x = x.ravel()
    y = FFTop * x
    assert_array_almost_equal(y, y_true, decimal=decimal)

    # Ensure inverse and adjoint recover x
    xadj = FFTop.H * y  # adjoint is same as inverse for fft
    xinv = lsqr(FFTop, y, damp=0, niter=10, atol=1e-8, btol=1e-8, show=0)[0].ravel()
    assert_array_almost_equal(x, xadj, decimal=decimal)
    assert_array_almost_equal(x, xinv, decimal=decimal)

    # Dot tests
    nr, nc = FFTop.shape
    assert dottest(FFTop, nr, nc, complexflag=0, rtol=10 ** (-decimal), backend=backend)
    assert dottest(FFTop, nr, nc, complexflag=2, rtol=10 ** (-decimal), backend=backend)
    if np.issubdtype(dtype, np.complexfloating):
        assert dottest(
            FFTop, nr, nc, complexflag=1, rtol=10 ** (-decimal), backend=backend
        )
        assert dottest(
            FFTop, nr, nc, complexflag=3, rtol=10 ** (-decimal), backend=backend
        )


par_lists_fft2dnd_small_cpx = dict(
    dtype_precision=dtype_precision_cpx,
    norm=["ortho", "none", "1/n"],
    engine=["numpy", "scipy"],
)
pars_fft2dnd_small_cpx = [
    dict(zip(par_lists_fft2dnd_small_cpx.keys(), value))
    for value in itertools.product(*par_lists_fft2dnd_small_cpx.values())
]


@pytest.mark.parametrize("par", pars_fft2dnd_small_cpx)
def test_FFT2D_small_complex(par):
    np.random.seed(5)
    dtype, decimal = par["dtype_precision"]
    norm = par["norm"]

    x = np.array(
        [
            [1, 2 - 1j, -1j, -1 + 2j],
            [2 - 1j, -1j, -1 - 2j, 1],
            [-1j, -1 - 2j, 1, 2 - 1j],
            [-1 - 2j, 1, 2 - 1j, -1j],
        ]
    )

    FFTop = FFT2D(
        dims=x.shape,
        axes=(0, 1),
        norm=norm,
        dtype=dtype,
    )

    # Compute FFT of x independently
    y_true = np.array(
        [
            [8 - 12j, -4, -4j, 4],
            [4j, 4 - 8j, -4j, 4],
            [4j, -4, 4j, 4],
            [4j, -4, -4j, 4 + 16j],
        ],
        dtype=FFTop.cdtype,
    )  # Backward
    if norm == "ortho":
        y_true /= 4
    elif norm == "1/n":
        y_true /= 16

    # Compute FFT with FFTop and compare with y_true
    y = FFTop * x.ravel()
    y = y.reshape(FFTop.dimsd)
    assert_array_almost_equal(y, y_true, decimal=decimal)
    assert dottest(
        FFTop, *FFTop.shape, complexflag=3, rtol=10 ** (-decimal), backend=backend
    )

    x_inv = FFTop / y.ravel()
    x_inv = x_inv.reshape(x.shape)
    assert_array_almost_equal(x_inv, x, decimal=decimal)


@pytest.mark.parametrize("par", pars_fft2dnd_small_cpx)
def test_FFTND_small_complex(par):
    np.random.seed(5)
    dtype, decimal = par["dtype_precision"]
    norm = par["norm"]

    x = np.array(
        [
            [1, 2 - 1j, -1j, -1 + 2j],
            [2 - 1j, -1j, -1 - 2j, 1],
            [-1j, -1 - 2j, 1, 2 - 1j],
            [-1 - 2j, 1, 2 - 1j, -1j],
        ]
    )

    FFTop = FFTND(
        dims=x.shape,
        axes=(0, 1),
        norm=norm,
        dtype=dtype,
    )

    # Compute FFT of x independently
    y_true = np.array(
        [
            [8 - 12j, -4, -4j, 4],
            [4j, 4 - 8j, -4j, 4],
            [4j, -4, 4j, 4],
            [4j, -4, -4j, 4 + 16j],
        ],
        dtype=FFTop.cdtype,
    )  # Backward
    if norm == "ortho":
        y_true /= 4
    elif norm == "1/n":
        y_true /= 16

    # Compute FFT with FFTop and compare with y_true
    y = FFTop * x.ravel()
    y = y.reshape(FFTop.dimsd)
    assert_array_almost_equal(y, y_true, decimal=decimal)
    assert dottest(
        FFTop, *FFTop.shape, complexflag=3, rtol=10 ** (-decimal), backend=backend
    )

    x_inv = FFTop / y.ravel()
    x_inv = x_inv.reshape(x.shape)
    assert_array_almost_equal(x_inv, x, decimal=decimal)


@pytest.mark.parametrize(
    "par",
    [
        (par1),
        (par2),
        (par3),
        (par4),
        (par5),
        (par6),
        (par1s),
        (par2s),
        (par3s),
        (par4s),
        (par5s),
        (par1w),
        (par2w),
        (par3w),
        (par4w),
        (par5w),
    ],
)
def test_FFT_1dsignal(par):
    np.random.seed(5)
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
        **par["kwargs"],
    )

    if par["real"]:
        assert dottest(
            FFTop,
            nfft // 2 + 1,
            par["nt"],
            complexflag=2,
            rtol=10 ** (-decimal),
            backend=backend,
        )
    else:
        assert dottest(
            FFTop,
            nfft,
            par["nt"],
            complexflag=2,
            rtol=10 ** (-decimal),
            backend=backend,
        )
        assert dottest(
            FFTop,
            nfft,
            par["nt"],
            complexflag=3,
            rtol=10 ** (-decimal),
            backend=backend,
        )

    y = FFTop * x
    xadj = FFTop.H * y  # adjoint is same as inverse for fft
    xinv = lsqr(FFTop, y, damp=1e-10, niter=10, atol=1e-8, btol=1e-8, show=0)[0]

    # check all signal if nt>nfft and only up to nfft if nfft<nt
    imax = par["nt"] if par["nfft"] is None else min([par["nt"], par["nfft"]])
    assert_array_almost_equal(x[:imax], xadj[:imax], decimal=decimal)
    assert_array_almost_equal(x[:imax], xinv[:imax], decimal=decimal)

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
        xinv = lsqr(
            FFTop_fftshift,
            y_fftshift,
            damp=1e-10,
            niter=10,
            atol=1e-8,
            btol=1e-8,
            show=0,
        )[0]
        assert_array_almost_equal(x[:imax], xadj[:imax], decimal=decimal)
        assert_array_almost_equal(x[:imax], xinv[:imax], decimal=decimal)


@pytest.mark.parametrize(
    "par",
    [
        (par1),
        (par2),
        (par3),
        (par4),
        (par5),
        (par6),
        (par1s),
        (par2s),
        (par3s),
        (par4s),
        (par5s),
        (par1w),
        (par2w),
        (par3w),
        (par4w),
        (par5w),
    ],
)
def test_FFT_2dsignal(par):
    """Dot-test and inversion for fft operator for 2d signal
    (fft on single dimension)
    """
    np.random.seed(5)
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
        axis=0,
        nfft=nfft,
        sampling=dt,
        real=par["real"],
        engine=par["engine"],
        dtype=par["dtype"],
        **par["kwargs"],
    )

    if par["real"]:
        assert dottest(
            FFTop,
            (nfft // 2 + 1) * nx,
            nt * nx,
            complexflag=2,
            rtol=10 ** (-decimal),
            backend=backend,
        )
    else:
        assert dottest(
            FFTop,
            nfft * nx,
            nt * nx,
            complexflag=2,
            rtol=10 ** (-decimal),
            backend=backend,
        )
        assert dottest(
            FFTop,
            nfft * nx,
            nt * nx,
            complexflag=3,
            rtol=10 ** (-decimal),
            backend=backend,
        )

    D = FFTop * d.ravel()
    dadj = FFTop.H * D  # adjoint is same as inverse for fft
    dinv = lsqr(FFTop, D, damp=1e-10, niter=10, atol=1e-8, btol=1e-8, show=0)[0]

    dadj = np.real(dadj.reshape(nt, nx))
    dinv = np.real(dinv.reshape(nt, nx))

    # check all signal if nt>nfft and only up to nfft if nfft<nt
    imax = par["nt"] if par["nfft"] is None else min([par["nt"], par["nfft"]])
    assert_array_almost_equal(d[:imax], dadj[:imax], decimal=decimal)
    assert_array_almost_equal(d[:imax], dinv[:imax], decimal=decimal)

    if not par["real"]:
        FFTop_fftshift = FFT(
            dims=(nt, nx),
            axis=0,
            nfft=nfft,
            sampling=dt,
            real=par["real"],
            fftshift_after=True,
            engine=par["engine"],
            dtype=par["dtype"],
            **par["kwargs"],
        )
        assert_array_almost_equal(FFTop_fftshift.f, np.fft.fftshift(FFTop.f))

        D_fftshift = FFTop_fftshift * d.flatten()
        D2 = np.fft.fftshift(D.reshape(nfft, nx), axes=0).flatten()
        assert_array_almost_equal(D_fftshift, D2)

        dadj = FFTop_fftshift.H * D_fftshift  # adjoint is same as inverse for fft
        dinv = lsqr(
            FFTop_fftshift,
            D_fftshift,
            damp=1e-10,
            niter=10,
            atol=1e-8,
            btol=1e-8,
            show=0,
        )[0]

        dadj = np.real(dadj.reshape(nt, nx))
        dinv = np.real(dinv.reshape(nt, nx))

        assert_array_almost_equal(d[:imax], dadj[:imax], decimal=decimal)
        assert_array_almost_equal(d[:imax], dinv[:imax], decimal=decimal)

    # 2nd dimension
    nfft = par["nx"] if par["nfft"] is None else par["nfft"]
    FFTop = FFT(
        dims=(nt, nx),
        axis=1,
        nfft=nfft,
        sampling=dt,
        real=par["real"],
        engine=par["engine"],
        dtype=par["dtype"],
        **par["kwargs"],
    )

    if par["real"]:
        assert dottest(
            FFTop,
            nt * (nfft // 2 + 1),
            nt * nx,
            complexflag=2,
            rtol=10 ** (-decimal),
            backend=backend,
        )
    else:
        assert dottest(
            FFTop,
            nt * nfft,
            nt * nx,
            complexflag=2,
            rtol=10 ** (-decimal),
            backend=backend,
        )
        assert dottest(
            FFTop,
            nt * nfft,
            nt * nx,
            complexflag=3,
            rtol=10 ** (-decimal),
            backend=backend,
        )

    D = FFTop * d.ravel()
    dadj = FFTop.H * D  # adjoint is inverse for fft
    dinv = lsqr(FFTop, D, damp=1e-10, niter=10, atol=1e-8, btol=1e-8, show=0)[0]

    dadj = np.real(dadj.reshape(nt, nx))
    dinv = np.real(dinv.reshape(nt, nx))

    # check all signal if nx>nfft and only up to nfft if nfft<nx
    imax = par["nx"] if par["nfft"] is None else min([par["nx"], par["nfft"]])
    assert_array_almost_equal(d[:, :imax], dadj[:, :imax], decimal=decimal)
    assert_array_almost_equal(d[:, :imax], dinv[:, :imax], decimal=decimal)

    if not par["real"]:
        FFTop_fftshift = FFT(
            dims=(nt, nx),
            axis=1,
            nfft=nfft,
            sampling=dt,
            real=par["real"],
            fftshift_after=True,
            engine=par["engine"],
            dtype=par["dtype"],
            **par["kwargs"],
        )
        assert_array_almost_equal(FFTop_fftshift.f, np.fft.fftshift(FFTop.f))

        D_fftshift = FFTop_fftshift * d.flatten()
        D2 = np.fft.fftshift(D.reshape(nt, nfft), axes=1).flatten()
        assert_array_almost_equal(D_fftshift, D2)

        dadj = FFTop_fftshift.H * D_fftshift  # adjoint is same as inverse for fft
        dinv = lsqr(
            FFTop_fftshift,
            D_fftshift,
            damp=1e-10,
            niter=10,
            atol=1e-8,
            btol=1e-8,
            show=0,
        )[0]

        dadj = np.real(dadj.reshape(nt, nx))
        dinv = np.real(dinv.reshape(nt, nx))

        assert_array_almost_equal(d[:, :imax], dadj[:, :imax], decimal=decimal)
        assert_array_almost_equal(d[:, :imax], dinv[:, :imax], decimal=decimal)


@pytest.mark.parametrize(
    "par",
    [
        (par1),
        (par2),
        (par3),
        (par4),
        (par5),
        (par6),
        (par1s),
        (par2s),
        (par3s),
        (par4s),
        (par5s),
        (par1w),
        (par2w),
        (par3w),
        (par4w),
        (par5w),
    ],
)
def test_FFT_3dsignal(par):
    """Dot-test and inversion for fft operator for 3d signal
    (fft on single dimension)
    """
    np.random.seed(5)
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
        axis=0,
        nfft=nfft,
        sampling=dt,
        real=par["real"],
        engine=par["engine"],
        dtype=par["dtype"],
        **par["kwargs"],
    )

    if par["real"]:
        assert dottest(
            FFTop,
            (nfft // 2 + 1) * nx * ny,
            nt * nx * ny,
            complexflag=2,
            rtol=10 ** (-decimal),
            backend=backend,
        )
    else:
        assert dottest(
            FFTop,
            nfft * nx * ny,
            nt * nx * ny,
            complexflag=2,
            rtol=10 ** (-decimal),
            backend=backend,
        )
        assert dottest(
            FFTop,
            nfft * nx * ny,
            nt * nx * ny,
            complexflag=3,
            rtol=10 ** (-decimal),
            backend=backend,
        )

    D = FFTop * d.ravel()
    dadj = FFTop.H * D  # adjoint is same as inverse for fft
    dinv = lsqr(FFTop, D, damp=1e-10, niter=10, atol=1e-8, btol=1e-8, show=0)[0]

    dadj = np.real(dadj.reshape(nt, nx, ny))
    dinv = np.real(dinv.reshape(nt, nx, ny))

    # check all signal if nt>nfft and only up to nfft if nfft<nt
    imax = nt if nfft is None else min([nt, nfft])
    assert_array_almost_equal(d[:imax], dadj[:imax], decimal=decimal)
    assert_array_almost_equal(d[:imax], dinv[:imax], decimal=decimal)

    # 2nd dimension
    nfft = par["nx"] if par["nfft"] is None else par["nfft"]
    FFTop = FFT(
        dims=(nt, nx, ny),
        axis=1,
        nfft=nfft,
        sampling=dt,
        real=par["real"],
        engine=par["engine"],
        dtype=par["dtype"],
        **par["kwargs"],
    )

    if par["real"]:
        assert dottest(
            FFTop,
            nt * (nfft // 2 + 1) * ny,
            nt * nx * ny,
            complexflag=2,
            rtol=10 ** (-decimal),
            backend=backend,
        )
    else:
        assert dottest(
            FFTop,
            nt * nfft * ny,
            nt * nx * ny,
            complexflag=2,
            rtol=10 ** (-decimal),
            backend=backend,
        )
        assert dottest(
            FFTop,
            nt * nfft * ny,
            nt * nx * ny,
            complexflag=3,
            rtol=10 ** (-decimal),
            backend=backend,
        )

    D = FFTop * d.ravel()
    dadj = FFTop.H * D  # adjoint is inverse for fft
    dinv = lsqr(FFTop, D, damp=1e-10, niter=10, atol=1e-8, btol=1e-8, show=0)[0]

    dadj = np.real(dadj.reshape(nt, nx, ny))
    dinv = np.real(dinv.reshape(nt, nx, ny))

    # check all signal if nx>nfft and only up to nfft if nfft<nx
    imax = nx if nfft is None else min([nx, nfft])
    assert_array_almost_equal(d[:, :imax], dadj[:, :imax], decimal=decimal)
    assert_array_almost_equal(d[:, :imax], dinv[:, :imax], decimal=decimal)

    # 3rd dimension
    nfft = par["ny"] if par["nfft"] is None else par["nfft"]
    FFTop = FFT(
        dims=(nt, nx, ny),
        axis=2,
        nfft=nfft,
        sampling=dt,
        real=par["real"],
        engine=par["engine"],
        dtype=par["dtype"],
        **par["kwargs"],
    )

    if par["real"]:
        assert dottest(
            FFTop,
            nt * nx * (nfft // 2 + 1),
            nt * nx * ny,
            complexflag=2,
            rtol=10 ** (-decimal),
            backend=backend,
        )
    else:
        assert dottest(
            FFTop,
            nt * nx * nfft,
            nt * nx * ny,
            complexflag=2,
            rtol=10 ** (-decimal),
            backend=backend,
        )
        assert dottest(
            FFTop,
            nt * nx * nfft,
            nt * nx * ny,
            complexflag=3,
            rtol=10 ** (-decimal),
            backend=backend,
        )

    D = FFTop * d.ravel()
    dadj = FFTop.H * D  # adjoint is inverse for fft
    dinv = lsqr(FFTop, D, damp=1e-10, niter=10, atol=1e-8, btol=1e-8, show=0)[0]

    dadj = np.real(dadj.reshape(nt, nx, ny))
    dinv = np.real(dinv.reshape(nt, nx, ny))

    # check all signal if ny>nfft and only up to nfft if nfft<ny
    imax = ny if nfft is None else min([ny, nfft])
    assert_array_almost_equal(d[..., :imax], dadj[..., :imax], decimal=decimal)
    assert_array_almost_equal(d[..., :imax], dinv[..., :imax], decimal=decimal)

    if not par["real"]:
        FFTop_fftshift = FFT(
            dims=(nt, nx, ny),
            axis=2,
            nfft=nfft,
            sampling=dt,
            real=par["real"],
            fftshift_after=True,
            engine=par["engine"],
            dtype=par["dtype"],
            **par["kwargs"],
        )
        assert_array_almost_equal(FFTop_fftshift.f, np.fft.fftshift(FFTop.f))

        D_fftshift = FFTop_fftshift * d.flatten()
        D2 = np.fft.fftshift(D.reshape(nt, nx, nfft), axes=2).flatten()
        assert_array_almost_equal(D_fftshift, D2)

        dadj = FFTop_fftshift.H * D_fftshift  # adjoint is same as inverse for fft
        dinv = lsqr(
            FFTop_fftshift,
            D_fftshift,
            damp=1e-10,
            niter=10,
            atol=1e-8,
            btol=1e-8,
            show=0,
        )[0]

        dadj = np.real(dadj.reshape(nt, nx, ny))
        dinv = np.real(dinv.reshape(nt, nx, ny))

        assert_array_almost_equal(d[..., :imax], dadj[..., :imax], decimal=decimal)
        assert_array_almost_equal(d[..., :imax], dinv[..., :imax], decimal=decimal)


@pytest.mark.parametrize(
    "par",
    [
        (par1),
        (par2),
        (par3),
        (par4),
        (par6),
        (par1s),
        (par2s),
        (par3s),
        (par4s),
        (par5s),
    ],
)
def test_FFT2D(par):
    """Dot-test and inversion for FFT2D operator for 2d signal"""
    np.random.seed(5)
    decimal = 3 if np.real(np.ones(1, par["dtype"])).dtype == np.float32 else 8

    dt, dx = 0.005, 5
    t = np.arange(par["nt"]) * dt
    f0 = 10
    nfft1 = par["nt"] if par["nfft"] is None else par["nfft"]
    nfft2 = par["nx"] if par["nfft"] is None else par["nfft"]
    d = np.outer(np.sin(2 * np.pi * f0 * t), np.arange(par["nx"]) + 1)
    d = d.astype(par["dtype"])

    # first fft on axis 1
    FFTop = FFT2D(
        dims=(par["nt"], par["nx"]),
        nffts=(nfft1, nfft2),
        sampling=(dt, dx),
        real=par["real"],
        axes=(0, 1),
        engine=par["engine"],
        dtype=par["dtype"],
        **par["kwargs"],
    )

    if par["real"]:
        assert dottest(
            FFTop,
            nfft1 * (nfft2 // 2 + 1),
            par["nt"] * par["nx"],
            complexflag=2,
            rtol=10 ** (-decimal),
            backend=backend,
        )
    else:
        assert dottest(
            FFTop,
            nfft1 * nfft2,
            par["nt"] * par["nx"],
            complexflag=2,
            rtol=10 ** (-decimal),
            backend=backend,
        )
        assert dottest(
            FFTop,
            nfft1 * nfft2,
            par["nt"] * par["nx"],
            complexflag=3,
            rtol=10 ** (-decimal),
            backend=backend,
        )

    D = FFTop * d.ravel()
    dadj = FFTop.H * D  # adjoint is inverse for fft
    dinv = lsqr(FFTop, D, damp=1e-10, niter=100, atol=1e-8, btol=1e-8, show=0)[0]

    dadj = np.real(dadj).reshape(par["nt"], par["nx"])
    dinv = np.real(dinv).reshape(par["nt"], par["nx"])

    # check all signal if nt>nfft and only up to nfft if nfft<nt
    imax1 = par["nt"] if nfft1 is None else min([par["nt"], nfft1])
    imax2 = par["nx"] if nfft2 is None else min([par["nx"], nfft2])
    assert_array_almost_equal(d[:imax1, :imax2], dadj[:imax1, :imax2], decimal=decimal)
    assert_array_almost_equal(d[:imax1, :imax2], dinv[:imax1, :imax2], decimal=decimal)

    # first fft on axis 0
    FFTop = FFT2D(
        dims=(par["nt"], par["nx"]),
        nffts=(nfft2, nfft1),
        sampling=(dx, dt),
        real=par["real"],
        axes=(1, 0),
        dtype=par["dtype"],
        **par["kwargs"],
    )

    if par["real"]:
        assert dottest(
            FFTop,
            nfft2 * (nfft1 // 2 + 1),
            par["nt"] * par["nx"],
            complexflag=2,
            rtol=10 ** (-decimal),
            backend=backend,
        )
    else:
        assert dottest(
            FFTop,
            nfft1 * nfft2,
            par["nt"] * par["nx"],
            complexflag=2,
            rtol=10 ** (-decimal),
            backend=backend,
        )
        assert dottest(
            FFTop,
            nfft1 * nfft2,
            par["nt"] * par["nx"],
            complexflag=3,
            rtol=10 ** (-decimal),
            backend=backend,
        )

    D = FFTop * d.ravel()
    dadj = FFTop.H * D  # adjoint is inverse for fft
    dinv = lsqr(FFTop, D, damp=1e-10, niter=100, atol=1e-8, btol=1e-8, show=0)[0]

    dadj = np.real(dadj).reshape(par["nt"], par["nx"])
    dinv = np.real(dinv).reshape(par["nt"], par["nx"])

    # check all signal if nt>nfft and only up to nfft if nfft<nt
    assert_array_almost_equal(d[:imax1, :imax2], dadj[:imax1, :imax2], decimal=decimal)
    assert_array_almost_equal(d[:imax1, :imax2], dinv[:imax1, :imax2], decimal=decimal)


@pytest.mark.parametrize(
    "par",
    [
        (par1),
        (par2),
        (par3),
        (par4),
        (par6),
        (par1s),
        (par2s),
        (par3s),
        (par4s),
        (par5s),
    ],
)
def test_FFT3D(par):
    """Dot-test and inversion for FFTND operator for 3d signal"""
    np.random.seed(5)
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

    # first fft on axis 2
    FFTop = FFTND(
        dims=(par["nt"], par["nx"], par["ny"]),
        nffts=(nfft1, nfft2, nfft3),
        axes=(0, 1, 2),
        sampling=(dt, dx, dy),
        real=par["real"],
        dtype=par["dtype"],
        **par["kwargs"],
    )

    if par["real"]:
        assert dottest(
            FFTop,
            nfft1 * nfft2 * (nfft3 // 2 + 1),
            par["nt"] * par["nx"] * par["ny"],
            complexflag=2,
            rtol=10 ** (-decimal),
            backend=backend,
        )
    else:
        assert dottest(
            FFTop,
            nfft1 * nfft2 * nfft3,
            par["nt"] * par["nx"] * par["ny"],
            complexflag=2,
            rtol=10 ** (-decimal),
            backend=backend,
        )
        assert dottest(
            FFTop,
            nfft1 * nfft2 * nfft3,
            par["nt"] * par["nx"] * par["ny"],
            complexflag=3,
            rtol=10 ** (-decimal),
            backend=backend,
        )

    D = FFTop * d.ravel()
    dadj = FFTop.H * D  # adjoint is inverse for fft
    dinv = lsqr(FFTop, D, damp=1e-10, niter=100, atol=1e-8, btol=1e-8, show=0)[0]

    dadj = np.real(dadj).reshape(par["nt"], par["nx"], par["ny"])
    dinv = np.real(dinv).reshape(par["nt"], par["nx"], par["ny"])

    # check all signal if nt>nfft and only up to nfft if nfft<nt
    imax1 = par["nt"] if nfft1 is None else min([par["nt"], nfft1])
    imax2 = par["nx"] if nfft2 is None else min([par["nx"], nfft2])
    imax3 = par["ny"] if nfft3 is None else min([par["ny"], nfft3])
    assert_array_almost_equal(
        d[:imax1, :imax2, :imax3], dadj[:imax1, :imax2, :imax3], decimal=decimal
    )
    assert_array_almost_equal(
        d[:imax1, :imax2, :imax3], dinv[:imax1, :imax2, :imax3], decimal=decimal
    )

    # first fft on axis 1
    FFTop = FFTND(
        dims=(par["nt"], par["nx"], par["ny"]),
        nffts=(nfft1, nfft3, nfft2),
        axes=(0, 2, 1),
        sampling=(dt, dy, dx),
        real=par["real"],
    )

    if par["real"]:
        assert dottest(
            FFTop,
            nfft1 * nfft3 * (nfft2 // 2 + 1),
            par["nt"] * par["nx"] * par["ny"],
            complexflag=2,
            rtol=10 ** (-decimal),
            backend=backend,
        )
    else:
        assert dottest(
            FFTop,
            nfft1 * nfft2 * nfft3,
            par["nt"] * par["nx"] * par["ny"],
            complexflag=2,
            rtol=10 ** (-decimal),
            backend=backend,
        )
        assert dottest(
            FFTop,
            nfft1 * nfft2 * nfft3,
            par["nt"] * par["nx"] * par["ny"],
            complexflag=3,
            rtol=10 ** (-decimal),
            backend=backend,
        )

    D = FFTop * d.ravel()
    dadj = FFTop.H * D  # adjoint is inverse for fft
    dinv = lsqr(FFTop, D, damp=1e-10, niter=100, atol=1e-8, btol=1e-8, show=0)[0]

    dadj = np.real(dadj).reshape(par["nt"], par["nx"], par["ny"])
    dinv = np.real(dinv).reshape(par["nt"], par["nx"], par["ny"])

    assert_array_almost_equal(
        d[:imax1, :imax2, :imax3], dadj[:imax1, :imax2, :imax3], decimal=decimal
    )
    assert_array_almost_equal(
        d[:imax1, :imax2, :imax3], dinv[:imax1, :imax2, :imax3], decimal=decimal
    )

    # first fft on axis 0
    FFTop = FFTND(
        dims=(par["nt"], par["nx"], par["ny"]),
        nffts=(nfft2, nfft3, nfft1),
        axes=(1, 2, 0),
        sampling=(dx, dy, dt),
        real=par["real"],
        dtype=par["dtype"],
        **par["kwargs"],
    )

    if par["real"]:
        assert dottest(
            FFTop,
            nfft2 * nfft3 * (nfft1 // 2 + 1),
            par["nt"] * par["nx"] * par["ny"],
            complexflag=2,
            rtol=10 ** (-decimal),
            backend=backend,
        )
    else:
        assert dottest(
            FFTop,
            nfft1 * nfft2 * nfft3,
            par["nt"] * par["nx"] * par["ny"],
            complexflag=2,
            rtol=10 ** (-decimal),
            backend=backend,
        )
        assert dottest(
            FFTop,
            nfft1 * nfft2 * nfft3,
            par["nt"] * par["nx"] * par["ny"],
            complexflag=3,
            rtol=10 ** (-decimal),
            backend=backend,
        )

    D = FFTop * d.ravel()
    dadj = FFTop.H * D  # adjoint is inverse for fft
    dinv = lsqr(FFTop, D, damp=1e-10, niter=100, atol=1e-8, btol=1e-8, show=0)[0]

    dadj = np.real(dadj).reshape(par["nt"], par["nx"], par["ny"])
    dinv = np.real(dinv).reshape(par["nt"], par["nx"], par["ny"])

    assert_array_almost_equal(
        d[:imax1, :imax2, :imax3], dadj[:imax1, :imax2, :imax3], decimal=decimal
    )
    assert_array_almost_equal(
        d[:imax1, :imax2, :imax3], dinv[:imax1, :imax2, :imax3], decimal=decimal
    )
