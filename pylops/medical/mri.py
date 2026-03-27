__all__ = [
    "MRI2D",
]

import warnings
from typing import Optional, Union

import numpy as np

from pylops import LinearOperator
from pylops.basicoperators import Diagonal, Restriction
from pylops.signalprocessing import FFT2D, Bilinear
from pylops.utils.backend import get_module
from pylops.utils.typing import (
    DTypeLike,
    InputDimsLike,
    NDArray,
    Tfftengine_nsm,
    Tmriengine,
    Tmrimask,
)


class MRI2D(LinearOperator):
    r"""2D Magnetic Resonance Imaging

    Apply 2D Magnetic Resonance Imaging operator to obtain a k-space data (i.e.,
    undersampled Fourier representation of the model).

    Parameters
    ----------
    dims : :obj:`list` or :obj:`int`
        Number of samples for each dimension. Must be 2-dimensional and of size
        :math:`n_y \times n_x`
    mask : :obj:`str` or :obj:`numpy.ndarray`
        Mask to be applied in the Fourier domain:

        - :obj:`numpy.ndarray`: a 2-dimensional array of size :math:`n_y \times n_x`
          with 1 in the selected locations;
        - ``vertical-reg``: mask with vertical lines (regularly sampled around the
          second dimension);
        - ``vertical-uni``: mask with vertical lines (irregularly sampled around the
          second dimension, with lines drawn from a uniform distribution);
        - ``radial-reg``: mask with radial lines (regularly sampled around the
          :math:`-\pi/\pi` angles);
        - ``radial-uni``: mask with radial lines (irregularly sampled around the
          :math:`-\pi/\pi` angles, with angles drawn from a uniform distribution);
    nlines : :obj:`int`, optional
        Number of lines in the k-space. Not required if ``mask`` is passed as array.
    perc_center : :obj:`float`, optional
        Percentage of total lines to retain in the center. Not required if ``mask``
        is passed as array.
    engine : :obj:`str`, optional
        Engine used for computation (``numpy`` or ``jax``).
    fft_engine : :obj:`str`, optional
        Engine used for fft computation (``numpy`` or ``scipy`` or ``mkl_fft``).
    dtype : :obj:`str`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
        Name of operator (to be used by :func:`pylops.utils.describe.describe`)
    **kwargs_fft
        Arbitrary keyword arguments to be passed to the selected fft method

    Attributes
    ----------
    mask : :obj:`numpy.ndarray`
        Mask applied in the Fourier domain.
    ROp : :obj:`pylops.Restriction` or :obj:`pylops.Diagonal` or :obj:`pylops.signalprocessing.Bilinear`
        Operator that applies the mask in the Fourier domain.
    dims : :obj:`tuple`
        Shape of the array after the adjoint, but before flattening.

        For example, ``x_reshaped = (Op.H * y.ravel()).reshape(Op.dims)``.
    dimsd : :obj:`tuple`
        Shape of the array after the forward, but before flattening.

        For example, ``y_reshaped = (Op * x.ravel()).reshape(Op.dimsd)``.
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved
        explicitly (``True``) or not (``False``)

    Raises
    ------
    ValueError
        If ``mask`` is not one of the accepted strings or a numpy array.
    ValueError
        If ``fft_engine`` is neither ``numpy``, ``fftw``, nor ``scipy``.
    ValueError
        If ``nlines`` or ``perc_center`` are not specified when providing ``mask``
        as string.
    ValueError
        If ``perc_center`` is greater than 0 when using ``vertical-reg`` mask.

    Notes
    -----
    The MRI2D operator applies 2-dimensional Fourier transform to the model,
    followed by a subsampling with a given ``mask``:

    .. math::
        \mathbf{d} = \mathbf{R} \mathbf{F}_{k} \mathbf{m}

    where :math:`\mathbf{F}_{k}` is the 2-dimensional Fourier transform and
    :math:`\mathbf{R}` is the mask.

    """

    def __init__(
        self,
        dims: InputDimsLike,
        mask: Union[Tmrimask, NDArray],
        nlines: Optional[int] = None,
        perc_center: Optional[float] = 0.1,
        engine: Tmriengine = "numpy",
        fft_engine: Tfftengine_nsm = "numpy",
        dtype: DTypeLike = "complex128",
        name: str = "M",
        **kwargs_fft,
    ) -> None:
        self._mask_type = mask if isinstance(mask, str) else "mask"
        self.engine = engine
        self.fft_engine = fft_engine

        # Validate inputs
        if engine == "jax" and fft_engine != "numpy":
            warnings.warn("When engine='jax', fft_engine is forced to 'numpy'")
            self.fft_engine = "numpy"
        if isinstance(mask, str) and mask not in (
            "vertical-reg",
            "vertical-uni",
            "radial-reg",
            "radial-uni",
        ):
            raise ValueError(
                "mask must be a numpy array, 'vertical-reg', 'vertical-uni', 'radial-reg', or 'radial-uni'"
            )
        if self.fft_engine not in ["numpy", "scipy", "mkl_fft"]:
            raise ValueError("fft_engine must be 'numpy', 'scipy', or 'mkl_fft'")
        if isinstance(mask, str) and (nlines is None or perc_center is None):
            raise ValueError(
                "nlines and perc_center must be specified providing mask as string"
            )
        if isinstance(mask, str) and mask == "vertical-reg" and perc_center > 0.0:
            raise ValueError("perc_center must be 0.0 when using 'vertical-reg' mask")

        # Create mask
        self.mask: NDArray
        if self._mask_type == "mask":
            self.mask = mask
        elif "vertical" in self._mask_type:
            self.mask = self._vertical_mask(
                dims,
                nlines,
                perc_center,
                uniform=True if "reg" in self._mask_type else False,
            )
        elif "radial" in self._mask_type:
            self.mask = self._radial_mask(
                dims, nlines, uniform=True if "reg" in self._mask_type else False
            )

        # Convert mask to appropriate backend
        ncp = get_module(self.engine)
        self.mask = ncp.asarray(self.mask)

        # Create operator
        self.ROp, Op = self._calc_op(
            dims=dims,
            mask_type=mask if isinstance(mask, str) else "mask",
            mask=self.mask,
            fft_engine=self.fft_engine,
            dtype=dtype,
            **kwargs_fft,
        )
        super().__init__(Op=Op, name=name)

    @staticmethod
    def _vertical_mask(
        dims: InputDimsLike, nlines: int, perc_center: float, uniform: bool = True
    ) -> NDArray:
        """Create vertical mask"""
        nlines_center = int(perc_center * dims[1])
        if (nlines + nlines_center) > dims[1]:
            raise ValueError(
                "nlines and perc_center produce a number of lines "
                "greater than the total number of lines of the k-space"
                f"({nlines + nlines_center}>{dims[1]})"
            )

        if nlines_center == 0:
            # No lines from the center
            if uniform:
                step = dims[1] // nlines
                mask = np.arange(0, dims[1], step)[:nlines]
            else:
                rng = np.random.default_rng()
                mask = rng.choice(np.arange(dims[1]), nlines, replace=False)
        else:
            # Lines taken from the center
            istart_center = dims[1] // 2 - nlines_center // 2
            iend_center = dims[1] // 2 + nlines_center // 2 + (nlines_center % 2)
            ilines_center = np.arange(istart_center, iend_center)

            # Other lines
            if uniform:
                nlines_left = nlines // 2 + nlines % 2
                step_left = istart_center // nlines_left
                ilines_left = np.arange(0, istart_center, step_left)[:nlines_left]
                nlines_right = nlines // 2
                step_right = (dims[1] - iend_center) // nlines_left
                ilines_right = np.arange(iend_center, dims[1], step_right)[
                    :nlines_right
                ]
                mask = np.sort(np.hstack((ilines_left, ilines_center, ilines_right)))
            else:
                rng = np.random.default_rng()
                ilines_other = np.hstack(
                    (np.arange(0, istart_center), np.arange(iend_center, dims[1]))
                )
                ilines_other = rng.choice(ilines_other, nlines, replace=False)
                mask = np.sort(np.hstack((ilines_center, ilines_other)))
        return mask

    @staticmethod
    def _radial_mask(dims: InputDimsLike, nlines: int, uniform: bool = True) -> NDArray:
        """Create radial mask"""
        npoints_per_line = dims[1] - 1

        # Define angles
        if uniform:
            thetas = np.linspace(0, np.pi, nlines, endpoint=False)
        else:
            rng = np.random.default_rng()
            thetas = rng.uniform(-np.pi, np.pi, nlines)

        # Create lines
        lines = []
        for theta in thetas:
            if theta == np.pi / 2:
                # Create vertical line
                xline = np.zeros(npoints_per_line)
                yline = np.linspace(
                    -dims[1] // 2 + 1, dims[1] // 2 - 1, npoints_per_line, endpoint=True
                )
            elif np.tan(theta) >= 0:
                # Create lines for positive angles
                xmax = min(dims[1] // 2, (dims[0] // 2) / np.tan(theta))
                xline = np.linspace(
                    -xmax,
                    min(xmax, dims[0] // 2 - 1 - (dims[0] + 1) % 2),
                    npoints_per_line,
                    endpoint=True,
                )
                yline = np.tan(theta) * xline
            else:
                # Create lines for negative angles
                xmin = max(-dims[1] // 2 + 1, (dims[0] // 2) / np.tan(theta))
                xline = np.linspace(
                    xmin, min(-xmin, dims[0] // 2 - 1), npoints_per_line, endpoint=True
                )
                yline = np.tan(theta) * xline
            xline, yline = xline + dims[0] // 2, yline + dims[1] // 2
            lines.append(np.vstack((xline, yline)))
        mask = np.concatenate(lines, axis=1)
        # Remove points beyond domain allowed by Bilinear operator
        # and duplicate points
        mask = mask[:, mask[0] < dims[0] - 1]
        mask = mask[:, mask[1] < dims[1] - 1]
        mask = np.unique(mask, axis=1)
        return mask

    def _matvec(self, x: NDArray) -> NDArray:
        return super()._matvec(x)

    def _rmatvec(self, x: NDArray) -> NDArray:
        return super()._rmatvec(x)

    @staticmethod
    def _calc_op(
        dims: InputDimsLike,
        mask_type: str,
        mask: NDArray,
        fft_engine: Tfftengine_nsm,
        dtype: DTypeLike,
        **kwargs_fft,
    ):
        """Calculate MRI operator"""
        fop = FFT2D(
            dims,
            nffts=dims,
            fftshift_after=True,
            engine=fft_engine,
            dtype=dtype,
            **kwargs_fft,
        )
        if mask_type == "mask":
            rop = Diagonal(mask, dtype=dtype)
        elif "vertical" in mask_type:
            rop = Restriction(dims, mask, axis=-1, forceflat=True, dtype=dtype)
        elif "radial" in mask_type:
            rop = Bilinear(mask, dims, dtype=dtype)
        return rop, rop @ fop
