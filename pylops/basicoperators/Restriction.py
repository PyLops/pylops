import numpy as np
import numpy.ma as np_ma

from pylops import LinearOperator
from pylops.utils.backend import get_array_module, to_cupy_conditional


def _compute_iavamask(dims, dir, iava, ncp):
    """Compute restriction mask when using cupy arrays"""
    otherdims = np.array(dims)
    otherdims = np.delete(otherdims, dir)
    iavamask = ncp.zeros(dims[dir], dtype=int)
    iavamask[iava] = 1
    iavamask = ncp.moveaxis(
        ncp.broadcast_to(iavamask, list(otherdims) + [dims[dir]]), -1, dir
    )
    iavamask = ncp.where(iavamask.ravel() == 1)[0]
    return iavamask


class Restriction(LinearOperator):
    r"""Restriction (or sampling) operator.

    Extract subset of values from input vector at locations ``iava``
    in forward mode and place those values at locations ``iava``
    in an otherwise zero vector in adjoint mode.

    Parameters
    ----------
    M : :obj:`int`
        Number of samples in model.
    iava : :obj:`list` or :obj:`numpy.ndarray`
        Integer indices of available samples for data selection.
    dims : :obj:`list`
        Number of samples for each dimension
        (``None`` if only one dimension is available)
    dir : :obj:`int`, optional
        Direction along which restriction is applied.
    dtype : :obj:`str`, optional
        Type of elements in input array.
    inplace : :obj:`bool`, optional
        Work inplace (``True``) or make a new copy (``False``). By default,
        data is a reference to the model (in forward) and model is a reference
        to the data (in adjoint).

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved
        explicitly (``True``) or not (``False``)

    See Also
    --------
    pylops.signalprocessing.Interp : Interpolation operator

    Notes
    -----
    Extraction (or *sampling*) of a subset of :math:`N` values at locations
    ``iava`` from an input (or model) vector :math:`\mathbf{x}` of size
    :math:`M` can be expressed as:

    .. math::

        y_i = x_{l_i}  \quad \forall i=0,1,\ldots,N-1

    where :math:`\mathbf{l}=[l_0, l_1,\ldots, l_{N-1}]` is a vector containing the indeces
    of the original array at which samples are taken.

    Conversely, in adjoint mode the available values in the data vector
    :math:`\mathbf{y}` are placed at locations
    :math:`\mathbf{l}=[l_0, l_1,\ldots, l_{M-1}]` in the model vector:

    .. math::

        x_{l_i} = y_i  \quad \forall i=0,1,\ldots,N-1

    and :math:`x_{j}=0` for :math:`j \neq l_i` (i.e., at all other locations in input
    vector).

    """

    def __init__(self, M, iava, dims=None, dir=0, dtype="float64", inplace=True):
        ncp = get_array_module(iava)
        self.M = M
        self.dir = dir
        self.iava = iava
        if dims is None:
            self.N = len(iava)
            self.dims = (self.M,)
            self.reshape = False
        else:
            if np.prod(dims) != self.M:
                raise ValueError("product of dims must equal M!")
            else:
                self.dims = dims  # model dimensions
                self.dimsd = list(dims)  # data dimensions
                self.dimsd[self.dir] = len(iava)
                self.iavareshape = (
                    [1] * self.dir
                    + [len(self.iava)]
                    + [1] * (len(self.dims) - self.dir - 1)
                )
                self.N = np.prod(self.dimsd)
                self.reshape = True

                # currently cupy does not support put_along_axis, so we need to
                # explicitely create a list of indices in the n-dimensional
                # model space which will be used in _rmatvec to place the input
                if ncp != np:
                    self.iavamask = _compute_iavamask(dims, dir, iava, ncp)
        self.inplace = inplace
        self.shape = (self.N, self.M)
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        ncp = get_array_module(x)
        if not self.inplace:
            x = x.copy()
        if not self.reshape:
            y = x[self.iava]
        else:
            x = ncp.reshape(x, self.dims)
            y = ncp.take(x, self.iava, axis=self.dir)
            y = y.ravel()
        return y

    def _rmatvec(self, x):
        ncp = get_array_module(x)
        if not self.inplace:
            x = x.copy()
        if not self.reshape:
            y = ncp.zeros(self.dims, dtype=self.dtype)
            y[self.iava] = x
        else:
            x = ncp.reshape(x, self.dimsd)
            if ncp == np:
                y = ncp.zeros(self.dims, dtype=self.dtype)
                ncp.put_along_axis(
                    y, ncp.reshape(self.iava, self.iavareshape), x, axis=self.dir
                )
            else:
                if not hasattr(self, "iavamask"):
                    self.iava = to_cupy_conditional(x, self.iava)
                    self.iavamask = _compute_iavamask(
                        self.dims, self.dir, self.iava, ncp
                    )
                y = ncp.zeros(int(self.M), dtype=self.dtype)
                y[self.iavamask] = x.ravel()
            y = y.ravel()
        return y

    def mask(self, x):
        """Apply mask to input signal returning a signal of same size with
        values at ``iava`` locations and ``0`` at other locations

        Parameters
        ----------
        x : :obj:`numpy.ndarray` or :obj:`cupy.ndarray`
            Input array (can be either flattened or not)

        Returns
        ----------
        y : :obj:`numpy.ma.core.MaskedArray`
            Masked array.

        """
        ncp = get_array_module(x)
        if ncp != np:
            iava = ncp.asnumpy(self.iava)
        else:
            iava = self.iava.copy()

        y = np_ma.array(np.zeros(self.dims), mask=np.ones(self.dims), dtype=self.dtype)
        if self.reshape:
            x = np.reshape(x, self.dims)
            x = np.swapaxes(x, self.dir, 0)
            y = np.swapaxes(y, self.dir, 0)
        y.mask[iava] = False
        if ncp == np:
            y[iava] = x[self.iava]
        else:
            y[iava] = ncp.asnumpy(x)[iava]
        if self.reshape:
            y = np.swapaxes(y, 0, self.dir)
        return y
