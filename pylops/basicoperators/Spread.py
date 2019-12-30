import logging
import numpy as np
from pylops import LinearOperator

try:
    from numba import jit
    from ._Spread_numba import _matvec_numba_table, _rmatvec_numba_table, \
        _matvec_numba_onthefly, _rmatvec_numba_onthefly
except ModuleNotFoundError:
    jit = None
    jit_message = 'Numba not available, reverting to numpy.'
except Exception as e:
    jit = None
    jit_message = 'Failed to import numba (error:%s), use numpy.' % e

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)


class Spread(LinearOperator):
    r"""Spread operator.

    Spread values from the input model vector arranged as a 2-dimensional
    array of size :math:`[n_{x0} \times n_{t0}]` into the data vector of size
    :math:`[n_x \times n_t]`. Spreading is performed along parametric curves
    provided as look-up table of pre-computed indices (``table``)
    or computed on-the-fly using a function handle (``fh``).

    In adjont mode, values from the data vector are instead stacked
    along the same parametric curves.

    Parameters
    ----------
    dims : :obj:`tuple`
        Dimensions of model vector (vector will be reshaped internally into
        a two-dimensional array of size :math:`[n_{x0} \times n_{t0}]`,
        where the first dimension is the spreading/stacking direction)
    dimsd : :obj:`tuple`
        Dimensions of model vector (vector will be reshaped internal into
        a two-dimensional array of size :math:`[n_x \times n_t]`)
    table : :obj:`np.ndarray`, optional
        Look-up table of indeces of size
        :math:`[n_{x0} \times n_{t0} \times n_x]` (if ``None`` use function
        handle ``fh``)
    dtable : :obj:`np.ndarray`, optional
        Look-up table of decimals remainders for linear interpolation of size
        :math:`[n_{x0} \times n_{t0} \times n_x]` (if ``None`` use function
        handle ``fh``)
    fh : :obj:`np.ndarray`, optional
        Function handle that returns an index (and a fractional value in case
        of ``interp=True``) to be used for spreading/stacking given indices
        in :math:`x0` and :math:`t` axes (if ``None`` use look-up table
        ``table``)
    interp : :obj:`bool`, optional
        Apply linear interpolation (``True``) or nearest interpolation
        (``False``) during stacking/spreading along parametric curve. To be
        used only if ``engine='numba'``, inferred directly from the number of
        outputs of ``fh`` for ``engine='numpy'``
    engine : :obj:`str`, optional
        Engine used for fft computation (``numpy`` or ``numba``). Note that
        ``numba`` can only be used when providing a look-up table
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly (``True``) or
        not (``False``)

    Raises
    ------
    KeyError
        If ``engine`` is neither ``numpy`` nor ``numba``
    NotImplementedError
        If both ``table`` and ``fh`` are not provided
    ValueError
        If ``table`` has shape different from
        :math:`[n_{x0} \times n_t0 \times n_x]`

    Notes
    -----
    The Spread operator applies the following linear transform in forward mode
    to the model vector after reshaping it into a 2-dimensional array of size
    :math:`[n_x \times n_t]`:

    .. math::
        m(x0, t_0) \rightarrow d(x, t=f(x0, x, t_0))

    where :math:`f(x0, x, t)` is a mapping function that returns a value t
    given values :math:`x0`, :math:`x`, and  :math:`t_0`.

    In adjoint mode, the model is reconstructed by means of the following
    stacking operation:

    .. math::
        m(x0, t_0) = \int{d(x, t=f(x0, x, t_0))} dx

    Note that ``table`` (or ``fh``)  must return integer numbers
    representing indices in the axis :math:`t`. However it also possible to
    perform linear interpolation as part of the spreading/stacking process by
    providing the decimal part of the mapping function (:math:`t - \lfloor
    t \rfloor`) either in ``dtable`` input parameter or as second value in
    the return of ``fh`` function.

    """
    def __init__(self, dims, dimsd, table=None, dtable=None,
                 fh=None, interp=False, engine='numpy', dtype='float64'):
        if not engine in ['numpy', 'numba']:
            raise KeyError('engine must be numpy or numba')
        if engine == 'numba' and jit is not None:
            self.engine = 'numba'
        else:
            if engine == 'numba' and jit is None:
                logging.warning(jit_message)
            self.engine = 'numpy'

        # axes
        self.dims, self.dimsd = dims, dimsd
        self.nx0, self.nt0 = self.dims[0], self.dims[1]
        self.nx, self.nt = self.dimsd[0], self.dimsd[1]
        self.table = table
        self.dtable = dtable
        self.fh = fh

        # find out if mapping is in table of function handle
        if table is None and fh is None:
            raise NotImplementedError('provide either table or fh...')
        elif table is not None:
            if self.table.shape != (self.nx0, self.nt0, self.nx):
                raise ValueError('table must have shape [nx0 x nt0 x nx]')
            self.usetable = True
            if np.any(self.table > self.nt):
                raise ValueError('values in table must be smaller than nt')
        else:
            self.usetable = False

        # find out if linear interpolation has to be carried out
        self.interp = False
        if self.usetable:
            if dtable is not None:
                if self.dtable.shape != (self.nx0, self.nt0, self.nx):
                    raise ValueError('dtable must have shape [nx0 x nt x nx]')
                self.interp = True
        else:
            if self.engine == 'numba':
                self.interp = interp
            else:
                if len(fh(0, 0)) == 2:
                    self.interp = True
        self.shape = (int(np.prod(self.dimsd)), int(np.prod(self.dims)))
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec_numpy(self, x):
        x = x.reshape(self.dims)
        y = np.zeros(self.dimsd, dtype=self.dtype)
        for it in range(self.dims[1]):
            for ix0 in range(self.dims[0]):
                if self.usetable:
                    indices = self.table[ix0, it]
                    if self.interp:
                        dindices = self.dtable[ix0, it]
                else:
                    if self.interp:
                        indices, dindices = self.fh(ix0, it)
                    else:
                        indices = self.fh(ix0, it)
                mask = np.argwhere(~np.isnan(indices))
                if mask.size > 0:
                    indices = (indices[mask]).astype(np.int)
                    if not self.interp:
                        y[mask, indices] += x[ix0, it]
                    else:
                        y[mask, indices] += (1-dindices[mask])*x[ix0, it]
                        y[mask, indices + 1] += dindices[mask] * x[ix0, it]
        return y.ravel()

    def _rmatvec_numpy(self, x):
        x = x.reshape(self.dimsd)
        y = np.zeros(self.dims, dtype=self.dtype)
        for it in range(self.dims[1]):
            for ix0 in range(self.dims[0]):
                if self.usetable:
                    indices = self.table[ix0, it]
                    if self.interp:
                        dindices = self.dtable[ix0, it]
                else:
                    if self.interp:
                        indices, dindices = self.fh(ix0, it)
                    else:
                        indices = self.fh(ix0, it)
                mask = np.argwhere(~np.isnan(indices))
                if mask.size > 0:
                    indices = (indices[mask]).astype(np.int)
                    if not self.interp:
                        y[ix0, it] = np.sum(x[mask, indices])
                    else:
                        y[ix0, it] = \
                            np.sum(x[mask, indices]*(1-dindices[mask])) + \
                            np.sum(x[mask, indices+1]*dindices[mask])
        return y.ravel()

    def _matvec(self, x):
        if self.engine == 'numba':
            y = np.zeros(self.dimsd, dtype=self.dtype)
            if self.usetable:
                y = _matvec_numba_table(x, y, self.dims, self.interp,
                                        self.table,
                                        self.table if self.dtable is None
                                        else self.dtable)
            else:
                y = _matvec_numba_onthefly(x, y, self.dims, self.interp,
                                           self.fh)
        else:
            y = self._matvec_numpy(x)
        return y

    def _rmatvec(self, x):
        if self.engine == 'numba':
            y = np.zeros(self.dims, dtype=self.dtype)
            if self.usetable:
                y = _rmatvec_numba_table(x, y, self.dims, self.dimsd,
                                         self.interp, self.table,
                                         self.table if self.dtable is None
                                         else self.dtable)
            else:
                y = _rmatvec_numba_onthefly(x, y, self.dims, self.dimsd,
                                            self.interp, self.fh)
        else:
            y = self._rmatvec_numpy(x)
        return y
