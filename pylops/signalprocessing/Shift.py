import numpy as np
from numpy.core.multiarray import normalize_axis_index

from pylops.basicoperators import Diagonal
from pylops.signalprocessing import FFT
from pylops.utils._internal import _value_or_list_like_to_array


def Shift(
    dims,
    shift,
    axis=-1,
    nfft=None,
    sampling=1.0,
    real=False,
    engine="numpy",
    dtype="complex128",
    name="S",
    **kwargs_fftw
):
    r"""Shift operator

    Apply fractional shift in the frequency domain along an ``axis``
    of a multi-dimensional array of size ``dims``.

    Parameters
    ----------
    dims : :obj:`tuple`
        Number of samples for each dimension
    shift : :obj:`float` or :obj:`numpy.ndarray`
        Fractional shift to apply in the same unit as ``sampling``. For multi-dimensional inputs,
        this can be a scalar to apply to every trace along the chosen axis or an array of shifts
        to be applied to each trace.
    axis : :obj:`int`, optional
        .. versionadded:: 2.0.0

        Axis along which shift is applied
    nfft : :obj:`int`, optional
        Number of samples in Fourier Transform (same as input if ``nfft=None``)
    sampling : :obj:`float`, optional
        Sampling step :math:`\Delta t`.
    real : :obj:`bool`, optional
        Model to which fft is applied has real numbers (``True``) or not
        (``False``). Used to enforce that the output of adjoint of a real
        model is real.
    engine : :obj:`str`, optional
        Engine used for fft computation (``numpy``, ``scipy``, or ``fftw``). Choose
        ``numpy`` when working with CuPy arrays.
    dtype : :obj:`str`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
        .. versionadded:: 2.0.0

        Name of operator (to be used by :func:`pylops.utils.describe.describe`)
    **kwargs_fftw
            Arbitrary keyword arguments
            for :py:class:`pyfftw.FTTW`

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly
        (``True``) or not (``False``)

    Raises
    ------
    ValueError
        If ``dims`` is provided and ``axis`` is bigger than ``len(dims)``
    NotImplementedError
        If ``engine`` is neither ``numpy``, ``scipy``, nor ``fftw``

    Notes
    -----
    The Shift operator applies the forward Fourier transform, an element-wise
    complex scaling, and inverse fourier transform

    .. math::
         \mathbf{y}= \mathbf{F}^{-1} \mathbf{S} \mathbf{F} \mathbf{x}

    Here :math:`\mathbf{S}` is a diagonal operator that scales the Fourier
    transformed input by :math:`e^{-j2\pi f t_S}`, where :math:`t_S` is the
    chosen ``shift``.

    """
    Fop = FFT(
        dims,
        axis=axis,
        nfft=nfft,
        sampling=sampling,
        real=real,
        engine=engine,
        dtype=dtype,
        **kwargs_fftw
    )
    if isinstance(dims, int):
        dimsdiag = None
    else:
        dimsdiag = list(dims)
        dimsdiag[axis] = len(Fop.f)

    shift = _value_or_list_like_to_array(shift)

    if shift.size == 1:
        shift = np.exp(-1j * 2 * np.pi * Fop.f * shift)
        Sop = Diagonal(shift, dims=dimsdiag, axis=axis, dtype=Fop.cdtype)
    else:
        # add dimensions to shift to match dimensions of model and data
        axis = normalize_axis_index(axis, len(dims))
        fdims = np.ones(shift.ndim + 1, dtype=int)
        fdims[axis] = Fop.f.size
        f = Fop.f.reshape(fdims)
        sdims = np.ones(shift.ndim + 1, dtype=int)
        sdims[:axis] = shift.shape[:axis]
        sdims[axis + 1 :] = shift.shape[axis:]
        shift = np.exp(-1j * 2 * np.pi * f * shift.reshape(sdims))
        Sop = Diagonal(shift, dtype=Fop.cdtype)
    Op = Fop.H * Sop * Fop
    Op.dims = Op.dimsd = Fop.dims
    # force dtype to that of input (FFT always upcasts it to complex)
    Op.dtype = dtype
    Op.name = name
    return Op
