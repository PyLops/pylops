__all__ = [
    "InterpCubicSpline",
]

from dataclasses import dataclass
from functools import cached_property, partial
from typing import Callable, Final, Literal, Tuple, Union

import numpy as np
from scipy.linalg import get_lapack_funcs
from scipy.sparse import csr_matrix

from pylops import LinearOperator
from pylops.signalprocessing._interp_utils import _clip_iava_above_last_sample_index
from pylops.utils._internal import _value_or_sized_to_tuple
from pylops.utils.backend import get_normalize_axis_index
from pylops.utils.decorators import reshaped
from pylops.utils.typing import (
    DTypeLike,
    FloatingNDArray,
    InexactNDArray,
    InputDimsLike,
    IntNDArray,
    SamplingLike,
)

ONE_SIXTH: Final[float] = 1.0 / 6.0
TWO_THIRDS: Final[float] = 2.0 / 3.0


def _second_order_finite_differences_zero_padded(
    x: InexactNDArray,
    pad_width: tuple[tuple[int, int], ...],
) -> InexactNDArray:
    """
    Computes the second order finite differences of ``x`` along axis 0 and pads the
    result with ``pad_width[0][0]`` leading and `pad_width[0][1]`` trailing zeros
    along the same axis.

    Parameters
    ----------
    x : :obj:`numpy.ndarray` of shape ``(n,)`` or shape ``(m, n)``
        The input array.
        It is processed along axis 0.
    pad_width : ((:obj:`int`, :obj:`int`), ...) of length ``x.ndim``
        The ``pad_width`` argument to pass to ``numpy.pad`` to achieve the subsequent
        zero padding along axis 0.

    Returns
    -------
    x_diffs : :obj:`numpy.ndarray` of shape ``(n,)`` or shape ``(m, n)``
        The second order finite differences of ``x`` padded with leading and trailing
        zeros along axis 0.

    """

    return np.pad(
        np.diff(
            x,
            n=2,
            axis=0,
        ),
        pad_width=pad_width,
        mode="constant",
        constant_values=0.0,
    )


def _second_order_finite_differences_zero_padded_transposed(
    x: InexactNDArray,
    x_slice: slice,
    pad_width: tuple[tuple[int, int], ...],
) -> InexactNDArray:
    """
    Computes the transposed operation of the second order finite differences operator
    with subsequent zero padding along axis 0, i.e.,

    - ``x[0 : x_slices[axis].start,]`` and ``x[x_slices[axis].stop ::]`` are not
        considered,
    - ``x[x_slice]`` is padded with ``pad_width[0][0]`` leading and ``pad_width[0][1]``
        trailing zeros,
    - the second order finite differences of the padded array are computed

    Parameters
    ----------
    x : :obj:`numpy.ndarray` of shape ``(n,)``  or shape ``(m, n)``
        The input array.
        It is processed along axis 0.
    x_slices : (:obj:`slice`, ...) of length ``x.ndim``
        The slices to extract from ``x`` along each dimension to ignore leading and
        trailing elements along axis 0.
    pad_width : ((:obj:`int`, :obj:`int`), ...) of length ``x.ndim``
        The ``pad_width`` argument to pass to ``numpy.pad`` to achieve the zero
        padding along axis 0.

    Returns
    -------
    x_diffs : :obj:`numpy.ndarray` of shape ``(n,)``  or shape ``(m, n)``
        The result of the transposed second order finite differences operator with
        subsequent zero padding along axis 0.

    """

    return np.diff(
        np.pad(
            x[x_slice],
            pad_width=pad_width,
            mode="constant",
            constant_values=0.0,
        ),
        n=2,
        axis=0,
    )


@dataclass
class _TridiagonalMatrix:
    """
    Represents a tridiagonal matrix with

    - the main diagonal ``.main_diagonal``,
    - the super-diagonal ``.super_diagonal``,
    - the sub-diagonal ``.sub_diagonal``.

    """

    main_diagonal: InexactNDArray
    super_diagonal: InexactNDArray
    sub_diagonal: InexactNDArray

    def __post_init__(self) -> None:
        """
        Validates the input.

        """

        for which in ("main", "super", "sub"):
            ndim = getattr(self, f"{which}_diagonal").ndim
            if ndim != 1:
                raise ValueError(
                    f"Expected {which} diagonal to be a 1-dimensional Array, but it is "
                    f"{ndim}-dimensional."
                )

        main_diagonal_dtype = self.main_diagonal.dtype.type
        main_diagonal_size = self.main_diagonal.size
        for which in ("super", "sub"):
            diag = getattr(self, f"{which}_diagonal")
            dtype = diag.dtype.type
            size = diag.size

            if dtype != main_diagonal_dtype:
                raise TypeError(
                    f"Expected {which} diagonal to have the same dtype as the main "
                    f"diagonal, but its dtype is {repr(dtype)} and the main diagonal "
                    f"has dtype {repr(main_diagonal_dtype)}."
                )

            if size != main_diagonal_size - 1:
                raise ValueError(
                    f"Expected {which} diagonal to have 1 entry less than the main "
                    f"diagonal, but it has {size} entries and the main diagonal has "
                    f"{main_diagonal_size} entries."
                )

        return

    def __iter__(self):
        """
        Returns an iterator over the sub-diagonal, main diagonal and super-diagonal
        (in that order) as required for the LAPACK routines ``?gttrf``.

        """

        yield self.sub_diagonal
        yield self.main_diagonal
        yield self.super_diagonal

        return

    def __len__(self) -> int:
        """
        Returns the number of rows of the tridiagonal matrix.

        """

        return self.main_diagonal.size

    @property
    def dtype(self) -> DTypeLike:
        return self.main_diagonal.dtype

    @property
    def T(self) -> "_TridiagonalMatrix":
        """
        Returns the transpose of the tridiagonal matrix.

        """

        return _TridiagonalMatrix(
            main_diagonal=self.main_diagonal,
            super_diagonal=self.sub_diagonal,
            sub_diagonal=self.super_diagonal,
        )


@dataclass
class _BandedLUDecomposition:
    """
    Represents the LU decomposition of a general banded matrix as performed by the
    LAPACK routines ``?gbtrf``.
    This class was implemented for spline interpolations between only 2 data points
    because the class :class:`_BandedLUDecomposition` uses the LAPACK routines
    ``?gttrf`` that cannot handle 2 x 2 tridiagonal matrices.

    """

    lu_banded: InexactNDArray
    pivot_indices: IntNDArray
    num_sub: int
    num_super: int

    @staticmethod
    def from_tridiagonal_matrix(
        matrix: _TridiagonalMatrix,
        lapack_factorizer: Callable,
    ) -> "_BandedLUDecomposition":
        """
        Computes the LU decomposition of a tridiagonal matrix using the LAPACK routine
        ``?gbtrf``.

        Parameters
        ----------
        matrix : :obj:`_TridiagonalMatrix`
            The tridiagonal matrix to decompose.
        lapack_factorizer : callable
            The LAPACK routine ``?gbtrf`` to use for the decomposition.

        Returns
        -------
        lu_decomposition : :obj:`_BandedLUDecomposition`
            The LU decomposition of the tridiagonal matrix.

        """

        banded_representation = np.empty(
            shape=(4, len(matrix)),
            dtype=matrix.dtype,
        )
        banded_representation[1, 1::] = matrix.super_diagonal
        banded_representation[2, ::] = matrix.main_diagonal
        banded_representation[3, 0:-1] = matrix.sub_diagonal

        (lu_banded, pivot_indices, info,) = lapack_factorizer(
            ab=banded_representation,
            kl=1,
            ku=1,
        )

        if info == 0:
            return _BandedLUDecomposition(
                lu_banded=lu_banded,
                pivot_indices=pivot_indices,
                num_sub=1,
                num_super=1,
            )

        raise np.linalg.LinAlgError(
            f"Could not LU-factorize tridiagonal matrix! Got {info=}."
        )

    def solve(
        self,
        rhs: InexactNDArray,
        lapack_solver: Callable,
    ) -> InexactNDArray:
        """
        Solves the linear system of equations ``A @ x = rhs`` where ``A`` is the
        tridiagonal matrix represented by the LU decomposition. For this, the LAPACK
        routine ``?gbtrs`` is used.

        Parameters
        ----------
        rhs : :obj:`numpy.ndarray` of shape ``(n,)``  or shape ``(m, n)``
            The right-hand side(s) of the linear system of equations.
            It is processed along axis 0, i.e.,

            - a 1D-Array is treated as a single right hand side.
            - each colum of a 2D-Array is treated as a single right hand side.

        lapack_solver : callable
            The LAPACK routine ``?gbtrs`` to use for solving the system.

        Returns
        -------
        x : :obj:`numpy.ndarray` of shape ``(n,)``  or shape ``(m, n)``
            The solution of the linear system(s) of equations.
            For 2D-Arrays, the ``j``-th column is the solution of the respective
            ``rhs[::, j]``.

        """

        x, info = lapack_solver(
            self.lu_banded,
            self.num_sub,
            self.num_super,
            rhs,
            self.pivot_indices,
        )

        if info == 0:
            return x

        raise np.linalg.LinAlgError(
            f"Could not solve LU-factorization of tridiagonal matrix! Got {info=}."
        )


@dataclass
class _TridiagonalLUDecomposition:
    """
    Represents the LU decomposition of a tridiagonal matrix as performed by the LAPACK
    routines ``?gttrf``.

    """

    sub_diagonal_lu: InexactNDArray
    main_diagonal_lu: InexactNDArray
    super_diagonal_lu: InexactNDArray
    super_two_diagonal_lu: InexactNDArray
    pivot_indices: IntNDArray

    def __iter__(self):
        """
        Returns an iterator over the sub-diagonal, main diagonal, super-diagonal,
        the second super-diagonal (filled by pivoting) and the pivot indices
        (in that order) as required for the LAPACK routines ``?gttrs``.

        """

        yield self.sub_diagonal_lu
        yield self.main_diagonal_lu
        yield self.super_diagonal_lu
        yield self.super_two_diagonal_lu
        yield self.pivot_indices

        return

    @staticmethod
    def from_tridiagonal_matrix(
        matrix: _TridiagonalMatrix,
        lapack_factorizer: Callable,
    ) -> "_TridiagonalLUDecomposition":
        """
        Computes the LU decomposition of a tridiagonal matrix using the LAPACK routine
        ``?gttrf``.

        Parameters
        ----------
        matrix : :obj:`_TridiagonalMatrix`
            The tridiagonal matrix to decompose.
        lapack_factorizer : callable
            The LAPACK routine ``?gttrf`` to use for the decomposition.

        Returns
        -------
        lu_decomposition : :obj:`_TridiagonalLUDecomposition`
            The LU decomposition of the tridiagonal matrix.

        """

        (
            sub_diagonal_lu,
            main_diagonal_lu,
            super_diagonal_lu,
            super_two_diagonal_lu,
            pivot_indices,
            info,
        ) = lapack_factorizer(*matrix)

        if info == 0:
            return _TridiagonalLUDecomposition(
                sub_diagonal_lu=sub_diagonal_lu,
                main_diagonal_lu=main_diagonal_lu,
                super_diagonal_lu=super_diagonal_lu,
                super_two_diagonal_lu=super_two_diagonal_lu,
                pivot_indices=pivot_indices,
            )

        raise np.linalg.LinAlgError(
            f"Could not LU-factorize tridiagonal matrix! Got {info=}."
            f"Could not LU-factorize tridiagonal matrix! Got {info=}."
        )

    def solve(
        self,
        rhs: InexactNDArray,
        lapack_solver: Callable,
    ) -> InexactNDArray:
        """
        Solves the linear system of equations ``A @ x = rhs`` where ``A`` is the
        tridiagonal matrix represented by the LU decomposition. For this, the LAPACK
        routine ``?gttrs`` is used.

        Parameters
        ----------
        rhs : :obj:`numpy.ndarray` of shape ``(n,)``  or shape ``(m, n)``
            The right-hand side(s) of the linear system of equations.
            It is processed along axis 0, i.e.,

            - a 1D-Array is treated as a single right hand side.
            - each colum of a 2D-Array is treated as a single right hand side.

        lapack_solver : callable
            The LAPACK routine ``?gttrs`` to use for solving the system.

        Returns
        -------
        x : :obj:`numpy.ndarray` of shape ``(n,)``  or shape ``(m, n)``
            The solution of the linear system(s) of equations.
            For 2D-Arrays, the ``j``-th column is the solution of the respective
            ``rhs[::, j]``.

        """

        x, info = lapack_solver(*self, rhs)

        if info == 0:
            return x

        raise np.linalg.LinAlgError(
            f"Could not solve LU-factorization of tridiagonal matrix! Got {info=}."
        )


def _make_cubic_spline_left_hand_side(
    dims: int,
) -> _TridiagonalMatrix:
    """
    Constructs the banded matrix ``A`` for the linear system of equations ``A @ m = b``
    where

    - ``A`` is a diagonally dominant tridiagonal matrix with the main diagonal
        being ``[1, 2/3, 2/3, ..., 2/3, 2/3, 1]``, the super-diagonal being
        ``[0, 1/6, 1/6, ..., 1/6, 1/6]`` and the sub-diagonal being
        ``[1/6, 1/6, ..., 1/6, 1/6, 0]``,
    - ``m`` is the unknown vector of spline coefficients,
    - ``b`` is the second order finite differences of the ``y`` values for which the
        spline should interpolate.

    Parameters
    ----------
    dims : :obj:`int`
        The number of points the spline should interpolate.

    Returns
    -------
    matrix_a : :obj:`TridiagonalMatrix`
        The tridiagonal matrix ``A``.

    """

    main_diag = np.empty(shape=(dims,), dtype=np.float64)
    super_diag = np.empty(shape=(dims - 1,), dtype=np.float64)

    main_diag[0] = 1.0
    main_diag[1 : dims - 1] = TWO_THIRDS
    main_diag[dims - 1] = 1.0

    super_diag[0] = 0.0
    super_diag[1 : dims - 1] = ONE_SIXTH

    return _TridiagonalMatrix(
        main_diagonal=main_diag,
        super_diagonal=super_diag,
        sub_diagonal=np.flip(super_diag).copy(),  # to avoid a view
    )


def _make_cubic_spline_x_csr(
    dims: int,
    iava: FloatingNDArray,
    base_indices: IntNDArray,
    iava_remainders: FloatingNDArray,
) -> csr_matrix:
    """
    Constructs the specifications ``data``, ``indices``, and ``indptr`` for a
    :obj:`scipy.sparse.csr_matrix` ``X`` that can be used to interpolate the
    equally spaced ``y`` values with a cubic spline like ``X @ Q @ y`` where

    - ``X`` is the interpolation matrix to be constructed,
    - ``Q`` is the linear operator that obtains the spline coefficients ``y``
        (a vertical concatenation of the ``y`` values and the spline coefficients
        ``m``),
    - ``y`` are the values to be interpolated.

    Parameters
    ----------
    dims : :obj:`int`
        The number of points the spline interpolation takes as input.
    iava : :obj:`numpy.ndarray` of shape ``(n,)`` and dtype ``numpy.float64``
        Floating indices of the locations to which the spline should interpolate.
    base_indices : :obj:`numpy.ndarray` of shape ``(n,)`` and dtype ``numpy.int64``
        The indices of the respective first data point in ``y`` for the intervals
        in which the corresponding ``iava`` values lie.
    iava_remainders : :obj:`numpy.ndarray` of shape ``(n,)`` and dtype ``numpy.float64``
        The remainders of the ``iava`` values after subtracting the respective
        ``base_indices``.

    Returns
    -------
    cubic_spline_interp_matrix : :obj:`scipy.sparse.csr_matrix`
        The ``X`` is CSR-format.

    """

    # some auxiliary variables are required
    iava_remainders_cubic = (  # (x - x[i])³
        iava_remainders * iava_remainders * iava_remainders
    )
    one_minus_iava_remainders = 1.0 - iava_remainders  # (x[i + 1] - x)
    one_minus_iava_remainders_cubic = (  # (x[i + 1] - x)³
        one_minus_iava_remainders
        * one_minus_iava_remainders
        * one_minus_iava_remainders
    )

    # for each data point, except for the first and the last one, we need 4 entries
    # to multiply with ``y[i]``, ``y[i + 1]``, ``m[i]``, and ``m[i + 1]``;

    data = np.column_stack(
        (
            one_minus_iava_remainders,
            iava_remainders,
            ONE_SIXTH * (one_minus_iava_remainders_cubic - one_minus_iava_remainders),
            ONE_SIXTH * (iava_remainders_cubic - iava_remainders),
        )
    ).ravel()

    indices = np.add.outer(
        base_indices,
        np.array([0, 1, dims, dims + 1], dtype=np.int64),
    ).ravel()

    indptr = np.arange(0, 4 * (iava.size + 1), 4, dtype=np.int64)

    return csr_matrix(
        (
            data,
            indices,
            indptr,
        ),
        shape=(iava.size, 2 * dims),
    )


class InterpCubicSpline(LinearOperator):
    r"""
    Cubic spline interpolation operator.

    Interpolate a regularly sampled input vector along ``axis`` at the fractional
    positions ``iava`` using a cubic spline, i.e., a :math:`C^{2}`-continuous piecewise
    third order polynomial.

    Currently, only cubic splines with natural boundary conditions are supported, i.e.,
    the second derivatives at the first and last sampling point are both zero.

    .. note:: The vector ``iava`` should contain unique values only. If the same
      fractional index is present multiple times, an error will be raised. Elements that
      exceed the last index ``dims[axis] - 1`` are clipped to the closest float value
      right below ``dims[axis] - 1`` to avoid extrapolation.

    Parameters
    ----------
    dims : :obj:`list` or :obj:`int`
        Number of samples for each dimension.
        A cubic spline requires ``dims[axis] > 2``.
    iava : :obj:`list` or :obj:`numpy.ndarray`
        Floating indices of the locations to which the spline should interpolate.
    bc_type : :obj:`str`, optional
        The type of boundary condition.
        Currently, only ``"natural"`` is supported.
    axis : :obj:`int`, optional
        Axis along which interpolation is applied.
        By default, the interpolation is carried out over the last axis.
    dtype : ``numpy.dtype``-like, optional
        The data type of the input and output arrays.
        For complex input, both the real and the imaginary parts are interpolated
        separately.
        Only double precision versions of ``numpy.inexact`` are supported, i.e., either
        ``"float64"`` (default) or ``"complex128"``.
        Multiplication of the operator with data with less precise data types will
        result in a type promotion.
    name : :obj:`str`, optional
        Name of operator (to be used by :func:`pylops.utils.describe.describe`).

    Attributes
    ----------
    dims : :obj:`tuple`
        Shape of the array after the adjoint, but before flattening.
        For example, ``x_reshaped = (Op.H * y.ravel()).reshape(Op.dims)``.
    dimsd : :obj:`tuple`
        Shape of the array after the forward, but before flattening.
        For example, ``y_reshaped = (Op * x.ravel()).reshape(Op.dimsd)``.
    shape : :obj:`tuple`
        Operator shape.

    Raises
    ------
    ValueError
        If ``dims[axis] < 2``.
    ValueError
        If the ``iava`` contains duplicate values.
    NotImplementError
        If ``bc_type != "natural"``.
    TypeError
        If ``dtype`` is neither ``numpy.float64`` nor ``numpy.complex128``.

    See Also
    --------
    pylops.signalprocessing.Interp : General interpolation operator
    :py:class:`scipy.interpolate.CubicSpline` : An equivalent implementation of the forward operator :math:`\mathbf{S}\mathbf{x}`

    References
    ----------

    .. [1] Wikipedia (German), Spline Interpolation
       https://de.wikipedia.org/wiki/Spline-Interpolation#Kubisch_(Polynome_3._Grades)

    Notes
    -----
    Cubic spline interpolation of an :math:`\left(N\times 1\right)`-vector
    :math:`\mathbf{x}` at the :math:`L` fractional positions ``iava`` can be represented
    as :math:`\mathbf{y}=\mathbf{S}\mathbf{x}` where :math:`\mathbf{S}` is the cubic
    spline operator.

    The :math:`\left(N\times 1\right)`-grid-point vector at which :math:`\mathbf{x}` was
    equidistantly sampled is denoted by the *"knot"* vector :math:`\mathbf{k}`, i.e.,
    :math:`\mathbf{x}_{j} = f\left(\mathbf{k}_{j}\right)`. Furthermore, the vector
    ``iava`` is denoted by :math:`t` in the following mathematical expressions.

    When ``iava[i]`` (:math:`t_{i}`) is mapped to the :math:`j`-th knot-to-knot-interval
    :math:`\mathbf{k}_{j}\le t_{i}\lt\mathbf{k}_{j + 1}`, the corresponding polynomial
    :math:`s_{j}\left(t\right)` can be expressed as

    .. math::
        s_{j}\left(t\right) = p_{j,0}\left(t\right)\cdot\mathbf{x}_{j}
        + p_{j,1}\left(t\right)\cdot\mathbf{x}_{j + 1}
        + p_{j,2}\left(t\right)\cdot\mathbf{m}_{j}
        + p_{j,3}\left(t\right)\cdot\mathbf{m}_{j + 1}

    where the :math:`\left(N\times 1\right)`-vector :math:`\mathbf{m}` holds, the second
    order derivatives of the cubic spline at its knots and the base polynomials are
    given by

    - :math:`p_{j,0}\left(t\right) = \mathbf{k}_{j + 1} - t`
    - :math:`p_{j,1}\left(t\right) = t - \mathbf{k}_{j}`
    - :math:`p_{j,2}\left(t\right) = \frac{1}{6}\cdot\left(\left(\mathbf{k}_{j + 1} - t\right)^{3} - \left(\mathbf{k}_{j + 1} - t\right)\right)`
    - :math:`p_{j,3}\left(t\right) = \frac{1}{6}\cdot\left(\left(t - \mathbf{k}_{j}\right)^{3} - \left(t - \mathbf{k}_{j}\right)\right)`

    Applying this 4-term linear combination to all points in ``iava`` leads to the
    matrix representation

    .. math::
        \mathbf{S}\mathbf{x} = \mathbf{P}\begin{bmatrix}
            \mathbf{x} \\
            \mathbf{m}
        \end{bmatrix}

    where each row of the very sparse
    :math:`\left(L\times 2N\right)`-base-polynomial-matrix :math:`\mathbf{P}` has only
    4 non-zero entries:

    .. math::
        \mathbf{P}_{i} = \begin{bmatrix}
            \dots & 0 & \dots & p_{j,0}\left(t_{i}\right) & p_{j,1}\left(t_{i}\right) &
            \dots & 0 & \dots & p_{j,2}\left(t_{i}\right) & p_{j,3}\left(t_{i}\right) &
            \dots & 0 & \dots
        \end{bmatrix}

    at the indices :math:`j`, :math:`j + 1`, :math:`j + N`, and :math:`j + N + 1`,
    respectively.

    :math:`\mathbf{P}` forms the first linear operator of :math:`\mathbf{S}` that
    carries out the linear combination and requires the vectors :math:`\mathbf{x}` and
    :math:`\mathbf{m}` as combination weights. Those need to be obtained from the second
    linear operator :math:`\mathbf{F}`

    .. math::
        \mathbf{F}\mathbf{x} = \begin{bmatrix}
            \mathbf{x} \\
            \mathbf{m}
        \end{bmatrix}

    For cubic splines, :math:`\mathbf{F}` is given by

    .. math::
        \mathbf{F} = \begin{bmatrix}
            \mathbf{I}_{N} \\
            \mathbf{B}^{-1}\mathbf{D}_{2}
        \end{bmatrix}


    :math:`\mathbf{I}_{N}` is the :math:`\left(N\times N\right)`-identity matrix.
    The :math:`\left(N\times N\right)`-tridiagonal or -near-tridiagonal matrix
    :math:`\mathbf{B}` and the :math:`\left(N\times N\right)`-second-order-finite-
    difference matrix :math:`\mathbf{D}_{2}` originate from the linear system

    .. math::
        \mathbf{B}\mathbf{m}=\mathbf{D}_{2}\mathbf{x}

    that needs to be solved for :math:`\mathbf{m}`, the second order derivatives of the
    cubic spline at its knots, by

    .. math::
        \mathbf{m}=\mathbf{B}^{-1}\mathbf{D}_{2}\mathbf{x}

    Assuming :math:`\mathbf{x}` was sampled at equidistant knots, :math:`\mathbf{B}`
    simplifies to

    .. math::
        \mathbf{B} = \frac{1}{6}\cdot\begin{bmatrix}
            \mu_{0} & \lambda_{0} & 0 & 0 & 0 & \dots & 0 & 0 & 0 & 0 & \theta_{0} \\
            1 & 4 & 1 & 0 & 0 & \dots & 0 & 0 & 0 & 0 & 0 \\
            0 & 1 & 4 & 1 & 0 & \dots & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 1 & 4 & 1 & \dots & 0 & 0 & 0 & 0 & 0 \\
            \vdots & \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots & \vdots & \vdots & \vdots \\
            0 & 0 & 0 & 0 & 0 & \dots & 1 & 4 & 1 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & \dots & 0 & 1 & 4 & 1 & 0 \\
            0 & 0 & 0 & 0 & 0 & \dots & 0 & 0 & 1 & 4 & 1 \\
            \theta_{N-1} & 0 & 0 & 0 & 0 & \dots & 0 & 0 & 0 & \lambda_{N-1} & \mu_{N-1}
        \end{bmatrix}

    The special values :math:`\mu_{i}`, :math:`\lambda_{i}`, :math:`\theta_{i}` for the
    first (:math:`i = 0`) and last row  (:math:`i = N - 1`) are determined by the
    boundary conditions, i.e., the behaviour prescribed at the first and last knot. See
    below for different boundary conditions and their corresponding special values.

    Similarly, the second order finite difference matrix :math:`\mathbf{D}_{2}` can be
    reduced to

    .. math::
        \mathbf{D}_{2} = \begin{bmatrix}
            \ & \ & \ & \ & \ & \mathbf{d}_{0} & \ & \ & \ & \ \\
            1 & -2 & 1 & 0 & 0 & \dots & 0 & 0 & 0 & 0 & 0 \\
            0 & 1 & -2 & 1 & 0 & \dots & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 1 & -2 & 1 & \dots & 0 & 0 & 0 & 0 & 0 \\
            \vdots & \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots & \vdots & \vdots & \vdots \\
            0 & 0 & 0 & 0 & 0 & \dots & 1 & -2 & 1 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & \dots & 0 & 1 & -2 & 1 & 0 \\
            0 & 0 & 0 & 0 & 0 & \dots & 0 & 0 & 1 & -2 & 1 \\
            \ & \ & \ & \ & \ & \mathbf{d}_{N-1} & \ & \ & \ & \
        \end{bmatrix}

    Again, the special rows :math:`\mathbf{d}_{i}` depend on the boundary conditions.

    For the "natural" boundary condition for which the second derivatives of the spline
    are exactly zero at the boundaries, the special values and rows are

    - :math:`\mu_{0} = \mu_{N - 1} = 6`
    - :math:`\lambda_{0} = \lambda_{N - 1} = 0`
    - :math:`\theta_{0} = \theta_{N - 1} = 0`
    - :math:`\mathbf{d}_{0} = \mathbf{d}_{N - 1}` are all zeroes

    and :math:`\mathbf{B}` is thus truly tridiagonal.

    In summary, the cubic spline interpolation can be decomposed as

    .. math::
        \mathbf{y}=\mathbf{S}\mathbf{x}=\mathbf{P}\mathbf{F}\mathbf{x}

    The adjoint operator :math:`\mathbf{S}^{H}` can be derived by rearranging the
    involved operators. All of them are purely real and consequently, a transpose is
    sufficient. This yields

    .. math::
        \mathbf{S}^{H}=\mathbf{S}^{T} = \mathbf{F}^{T}\mathbf{P}^{T} = \begin{bmatrix}
            \mathbf{I}_{N} & {\mathbf{D}_{2}}^{T} \mathbf{B}^{-T}
        \end{bmatrix} \mathbf{P}^{T}

    where :math:`\mathbf{B}^{-T} = \left(\mathbf{B}^{-1}\right)^{T} = \left(\mathbf{B}^{T}\right)^{-1}`.

    Computationally,

    - :math:`\mathbf{D}_{2}` and :math:`{\mathbf{D}_{2}}^{T}` can be computed
      efficiently via finite differences with appropriate boundary handling
    - :math:`\mathbf{B}` and :math:`\mathbf{B}^{T}` are individually factorized via
      partially pivoted tridiagonal :math:`LU`-decompositions that are kept in memory
    - :math:`\mathbf{B}^{-1}\mathbf{z}` and :math:`\mathbf{B}^{-T}\mathbf{z}` then
      rely on highly optimized backward- and forward-substitutions with those
      factorizations to solve the linear systems
    - :math:`\mathbf{P}` is represented as a :py:class:`scipy.sparse.csr_matrix` which
      involves a tiny bit of overhead during operator initialization, but allows for
      a simple transpose.

    """

    def __init__(
        self,
        dims: Union[int, InputDimsLike],
        iava: SamplingLike,
        bc_type: Literal["natural"] = "natural",
        axis: int = -1,
        dtype: DTypeLike = "float64",
        name: str = "S",
    ) -> None:

        # Input Validation and Standardization

        dims = _value_or_sized_to_tuple(dims)
        axis = get_normalize_axis_index()(axis, len(dims))
        num_cols = dims[axis]

        if num_cols < 2:
            raise ValueError(
                f"A cubic spline requires at least 2 data points to interpolate, but "
                f"got {dims[axis]=}."
            )

        iava = np.asarray(iava, dtype=np.float64)
        _clip_iava_above_last_sample_index(iava=iava, sample_size=num_cols)

        if isinstance(bc_type, str) and bc_type.lower() in {"natural"}:
            self.bc_type = bc_type.lower()

        else:
            raise NotImplementedError(
                f"Cubic spline interpolation currently only supports 'natural' "
                f"boundaries, but got {bc_type=}"
            )

        dtype = np.dtype(dtype)
        if dtype.type not in {np.float64, np.complex128}:
            raise TypeError(
                f"Expected dtype fo cubic spline interpolator to be either float64 or "
                f"complex128 to achieve the required accuracy, but got {dtype}."
            )

        # Operator Initialization

        dimsd = list(dims)
        dimsd[axis] = len(iava)
        dimsd = tuple(dimsd)

        self.dims: Tuple = dims
        self.dimsd: Tuple = dimsd
        self.iava: FloatingNDArray = iava
        self.axis: int = axis

        super().__init__(
            dtype=dtype,
            dims=dims,
            dimsd=dimsd,
            name=name,
        )

        # Pre-computations of the Tridiagonal Systems

        # NOTE: the LU-factorization will always be performed on ``float64`` while
        #       the LU-solve type depends on the actual dtype, which might also be
        #       complex
        if num_cols >= 3:
            lapack_factorizer = ("gttrf",)
            lapack_solver = ("gttrs",)
            lu_format = _TridiagonalLUDecomposition
        else:
            lapack_factorizer = ("gbtrf",)
            lapack_solver = ("gbtrs",)
            lu_format = _BandedLUDecomposition

        self._tridiag_factorize = get_lapack_funcs(
            lapack_factorizer,
            dtype=self.iava.dtype,
        )[0]
        self._tridiag_lu_solve = get_lapack_funcs(
            lapack_solver,
            dtype=self.dtype,
        )[0]

        lhs_matrix: _TridiagonalMatrix = _make_cubic_spline_left_hand_side(
            dims=num_cols
        )

        self._lhs_B_matrix_lu = lu_format.from_tridiagonal_matrix(
            matrix=lhs_matrix,
            lapack_factorizer=self._tridiag_factorize,
        )
        self._lhs_B_matrix_transposed_lu = lu_format.from_tridiagonal_matrix(
            matrix=lhs_matrix.T,
            lapack_factorizer=self._tridiag_factorize,
        )

        # Pre-computation of the Interpolator Matrices

        base_indices = np.clip(
            self.iava.astype(np.int64),  # already rounds down
            a_min=0,
            a_max=num_cols - 2,
        )

        self._P_matrix: csr_matrix = _make_cubic_spline_x_csr(
            dims=num_cols,
            iava=self.iava,
            base_indices=base_indices,
            iava_remainders=self.iava - base_indices,
        )
        self._P_matrix_transposed: csr_matrix = self._P_matrix.transpose().tocsr()  # type: ignore

        self._matmat_difference_method = partial(
            _second_order_finite_differences_zero_padded,
            pad_width=((1, 1), (0, 0)),
        )
        self._rmatmat_difference_method = partial(
            _second_order_finite_differences_zero_padded_transposed,
            x_slice=slice(1, num_cols - 1),
            pad_width=((2, 2), (0, 0)),
        )

        return

    @cached_property
    def num_cols(self) -> int:
        return self.dims[self.axis]

    @reshaped(swapaxis=True, axis=0)
    def _matvec(self, x: InexactNDArray) -> InexactNDArray:
        x_reshaped = x.reshape(x.shape[0], -1)

        m_coeffs = self._lhs_B_matrix_lu.solve(
            rhs=self._matmat_difference_method(x_reshaped),
            lapack_solver=self._tridiag_lu_solve,
        )
        return (
            self._P_matrix
            @ np.concatenate(
                (
                    x_reshaped,
                    m_coeffs,
                ),
                axis=0,
            )
        ).reshape(-1, *x.shape[1:])

    @reshaped(swapaxis=True, axis=0)
    def _rmatvec(self, x: InexactNDArray) -> InexactNDArray:

        x_mod = self._P_matrix_transposed @ x.reshape(x.shape[0], -1)

        return (
            x_mod[0 : self.num_cols]
            + self._rmatmat_difference_method(
                self._lhs_B_matrix_transposed_lu.solve(
                    rhs=x_mod[self.num_cols : x_mod.size],
                    lapack_solver=self._tridiag_lu_solve,
                )
            )
        ).reshape(self.num_cols, *x.shape[1:])
