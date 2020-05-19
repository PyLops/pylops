import logging
import numpy as np
from pylops import LinearOperator
from pylops.basicoperators import Restriction, Diagonal, MatrixMult, Transpose

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)

def _checkunique(iava):
    _, count = np.unique(iava, return_counts=True)
    if np.any(count > 1):
        raise ValueError('Repeated values in iava array')

def _nearestinterp(M, iava, dims=None, dir=0, dtype='float64'):
    """Nearest neighbour interpolation.
    """
    iava = np.round(iava).astype(np.int)
    _checkunique(iava)
    return Restriction(M, iava, dims=dims, dir=dir, dtype=dtype), iava

def _linearinterp(M, iava, dims=None, dir=0, dtype='float64'):
    """Linear interpolation.
    """
    if np.issubdtype(iava.dtype, np.integer):
        iava = iava.astype(np.float)
    if dims is None:
        lastsample = M
        dimsd = None
    else:
        lastsample = dims[dir]
        dimsd = list(dims)
        dimsd[dir] = len(iava)
        dimsd = tuple(dimsd)

    # ensure that samples are not beyond the last sample, in that case set to
    # penultimate sample and raise a warning
    outside = (iava >= lastsample - 1)
    if sum(outside) > 0:
        logging.warning('at least one value is beyond penultimate sample, '
                        'forced to be at penultimate sample')
    iava[outside] = lastsample - 1 - 1e-10
    _checkunique(iava)

    # find indices and weights
    iva_l = np.floor(iava).astype(np.int)
    iva_r = iva_l + 1
    weights = iava - iva_l

    # create operators
    Op = Diagonal(1 - weights, dims=dimsd, dir=dir, dtype=dtype) * \
         Restriction(M, iva_l, dims=dims, dir=dir, dtype=dtype) + \
         Diagonal(weights, dims=dimsd, dir=dir, dtype=dtype) * \
         Restriction(M, iva_r, dims=dims, dir=dir, dtype=dtype)
    return Op, iava


def _sincinterp(M, iava, dims=None, dir=0, dtype='float64'):
    """Sinc interpolation.
    """
    _checkunique(iava)
    # create sinc interpolation matrix
    nreg = M if dims is None else dims[dir]
    ireg = np.arange(nreg)
    sinc = np.tile(iava[:, np.newaxis], (1, nreg)) - \
           np.tile(ireg, (len(iava), 1))
    sinc = np.sinc(sinc)

    # identify additional dimensions and create MatrixMult operator
    otherdims = None
    if dims is not None:
        otherdims = np.array(dims)
        otherdims = np.roll(otherdims, -dir)
        otherdims = otherdims[1:]
        print('dims', dims)
        print('otherdims', otherdims)
    Op = MatrixMult(sinc, dims=otherdims, dtype=dtype)

    # create Transpose operator that brings dir to first dimension
    if dir > 0:
        axes = np.arange(len(dims), dtype=np.int)
        axes = np.roll(axes, -dir)
        dimsd =  list(dims)
        dimsd[dir] = len(iava)
        Top = Transpose(dims, axes=axes, dtype=dtype)
        T1op = Transpose(dimsd, axes=axes, dtype=dtype)
        Op = T1op.H * Op * Top
    return Op


def Interp(M, iava, dims=None, dir=0, kind='linear', dtype='float64'):
    r"""Interpolation operator.

    Apply interpolation along direction ``dir``
    from regularly sampled input vector into fractionary positions ``iava``
    using one of the following algorithms:

    - *Nearest neighbour* interpolation
      is a thin wrapper around :obj:`pylops.Restriction` at ``np.round(iava)``
      locations.

    - *Linear interpolation* extracts values from input vector
      at locations ``np.floor(iava)`` and ``np.floor(iava)+1`` and linearly
      combines them in forward mode, places weighted versions of the
      interpolated values at locations ``np.floor(iava)`` and
      ``np.floor(iava)+1`` in an otherwise zero vector in adjoint mode.

    - *Sinc interpolation* performs sinc interpolation at locations ``iava``.
      Note that this is the most accurate method but it has higher computational
      cost as it involves multiplying the input data by a matrix of size
      :math:`N \times M`.

    .. note:: The vector ``iava`` should contain unique values. If the same
      index is repeated twice an error will be raised. This also applies when
      values beyond the last element of the input array for
      *linear interpolation* as those values are forced to be just before this
      element.

    Parameters
    ----------
    M : :obj:`int`
        Number of samples in model.
    iava : :obj:`list` or :obj:`numpy.ndarray`
         Floating indices of locations of available samples for interpolation.
    dims : :obj:`list`, optional
        Number of samples for each dimension
        (``None`` if only one dimension is available)
    dir : :obj:`int`, optional
        Direction along which restriction is applied.
    kind : :obj:`str`, optional
        Kind of interpolation (``nearest``, ``linear``, and ``sinc`` are
        currently supported)
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Returns
    -------
    op : :obj:`pylops.LinearOperator`
        Linear intepolation operator
    iava : :obj:`list` or :obj:`numpy.ndarray`
        Corrected indices of locations of available samples
        (samples at ``M-1`` or beyond are forced to be at ``M-1-eps``)

    Raises
    ------
    ValueError
        If the vector ``iava`` contains repeated values.
    NotImplementedError
        If ``kind`` is not ``nearest``, ``linear`` or ``sinc``

    See Also
    --------
    pylops.Restriction : Restriction operator

    Notes
    -----
    *Linear interpolation* of a subset of :math:`N` values at locations
    ``iava`` from an input (or model) vector :math:`\mathbf{x}` of size
    :math:`M` can be expressed as:

    .. math::

        y_i = (1-w_i) x_{l^{l}_i} + w_i x_{l^{r}_i}
        \quad \forall i=1,2,...,N

    where :math:`\mathbf{l^l}=[\lfloor l_1 \rfloor, \lfloor l_2 \rfloor,...,
    \lfloor l_N \rfloor]` and :math:`\mathbf{l^r}=[\lfloor l_1 \rfloor +1,
    \lfloor l_2 \rfloor +1,...,
    \lfloor l_N \rfloor +1]` are vectors containing the indeces
    of the original array at which samples are taken, and
    :math:`\mathbf{w}=[l_1 - \lfloor l_1 \rfloor, l_2 - \lfloor l_2 \rfloor,
    ..., l_N - \lfloor l_N \rfloor]` are the linear interpolation weights.
    This operator can be implemented by simply summing two
    :class:`pylops.Restriction` operators which are weighted
    using :class:`pylops.basicoperators.Diagonal` operators.

    *Sinc interpolation* of a subset of :math:`N` values at locations
    ``iava`` from an input (or model) vector :math:`\mathbf{x}` of size
    :math:`M` can be expressed as:

    .. math::

        y_i = \sum_{j=0}^{M} x_j sinc(i-j) \quad \forall i=1,2,...,N

    This operator can be implemented using the :class:`pylops.MatrixMult`
    operator with a matrix containing the values of the sinc function at all
    :math:`i,j` possible combinations.

    """
    if kind == 'nearest':
        interpop, iava = _nearestinterp(M, iava, dims=dims, dir=dir, dtype=dtype)
    elif kind == 'linear':
        interpop, iava = _linearinterp(M, iava, dims=dims, dir=dir, dtype=dtype)
    elif kind == 'sinc':
        interpop = _sincinterp(M, iava, dims=dims, dir=dir, dtype=dtype)
    else:
        raise NotImplementedError('kind is not correct...')
    return LinearOperator(interpop), iava
