import logging

import numpy as np

from pylops import LinearOperator

try:
    from numba import jit

    from ._Spread_numba import (
        _matvec_numba_onthefly,
        _matvec_numba_table,
        _rmatvec_numba_onthefly,
        _rmatvec_numba_table,
    )
except ModuleNotFoundError:
    jit = None
    jit_message = "Numba not available, reverting to numpy."
except Exception as e:
    jit = None
    jit_message = "Failed to import numba (error:%s), use numpy." % e

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


class Spread(LinearOperator):
    r"""Spread operator.

    Spread values from the input model vector arranged as a 2-dimensional
    array of size :math:`[n_{x_0} \times n_{t_0}]` into the data vector of size
    :math:`[n_x \times n_t]`. Note that the value at each single pair
    :math:`(x_0, t_0)` in the input is spread over the entire :math:`x` axis
    in the output.

    Spreading is performed along parametric curves provided as look-up table
    of pre-computed indices (``table``) or computed on-the-fly using a
    function handle (``fh``).

    In adjont mode, values from the data vector are instead stacked
    along the same parametric curves.

    Parameters
    ----------
    dims : :obj:`tuple`
        Dimensions of model vector (vector will be reshaped internally into
        a two-dimensional array of size :math:`[n_{x_0} \times n_{t_0}]`,
        where the first dimension is the spreading direction)
    dimsd : :obj:`tuple`
        Dimensions of data vector (vector will be reshaped internal into
        a two-dimensional array of size :math:`[n_x \times n_t]`,
        where the first dimension is the stacking direction)
    table : :obj:`np.ndarray`, optional
        Look-up table of indices of size
        :math:`[n_{x_0} \times n_{t_0} \times n_x]` (if ``None`` use function
        handle ``fh``). When ``dtable`` is not provided, the  ``data`` will be created
        as follows

        .. code-block:: python

            data[ix, table[ix0, it0, ix]] += model[ix0, it0]

        .. note::
            When using ``table`` without ``dtable``, its elements must be
            between 0 and :math:`n_{t_0} - 1` (or ``numpy.nan``).

    dtable : :obj:`np.ndarray`, optional
        Look-up table of decimals remainders for linear interpolation of size
        :math:`[n_{x_0} \times n_{t_0} \times n_x]` (if ``None`` use function
        handle ``fh``). When provided, the ``data`` will be created as follows

        .. code-block:: python

            data[ix, table[ix0, it0, ix]]     += (1 - dtable[ix0, it0, ix]) * model[ix0, it0]
            data[ix, table[ix0, it0, ix] + 1] +=      dtable[ix0, it0, ix]  * model[ix0, it0]

        .. note::
            When using ``table`` and ``dtable``, the elements of ``table`` indices must be
            between 0 and :math:`n_{t_0} - 2`  (or ``numpy.nan``).

    fh : :obj:`callable`, optional
        If ``None`` will use look-up table ``table``. When provided, should be a
        function which takes indices ``ix0`` and ``it0`` and returns
        an array of size :math:`n_x` containing each respective time index.
        Alternatively, if linear interpolation is required, it should output in
        addition to the time indices, a weight for interpolation with linear
        interpolation, to be used as follows

        .. code-block:: python

            data[ix, index]     += (1 - dindices[ix]) * model[ix0, it0]
            data[ix, index + 1] +=      dindices[ix]  * model[ix0, it0]

        where ``index`` refers to a time index in the first array returned by ``fh``
        and ``dindices`` refers to the weight in the second array returned by ``fh``.

        .. note::
            When using ``fh`` with one output (time indices), the time indices must be
            between 0 and :math:`n_{t_0} - 1` (or ``numpy.nan``). When using ``fh`` with two outputs
            (time indices and weights), they must be within the between 0 and
            :math:`n_{t_0} - 2` (or ``numpy.nan``).

    interp : :obj:`bool`, optional
        Use only if engine ``engine='numba'``. Apply linear interpolation (``True``)
        or nearest interpolation (``False``) during stacking/spreading along
        parametric curve.
        When using ``engine="numpy"``, it will be inferred directly from ``fh`` or
        the presence of ``dtable``.
    engine : :obj:`str`, optional
        Engine used for spread computation (``numpy`` or ``numba``). Note that
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
        :math:`[n_{x_0} \times n_{t_0} \times n_x]`

    Notes
    -----
    The Spread operator applies the following linear transform in forward mode
    to the model vector after reshaping it into a 2-dimensional array of size
    :math:`[n_x \times n_t]`:

    .. math::
        m(x_0, t_0) \rightarrow d(x, t=f(x_0, x, t_0)) \quad \forall x

    where :math:`f(x_0, x, t)` is a mapping function that returns a value :math:`t`
    given values :math:`x_0`, :math:`x`, and  :math:`t_0`. Note that for each
    :math:`(x_0, t_0)` pair, spreading is done over the entire :math:`x` axis
    in the data domain.

    In adjoint mode, the model is reconstructed by means of the following
    stacking operation:

    .. math::
        m(x_0, t_0) = \int{d(x, t=f(x_0, x, t_0))} \,\mathrm{d}x

    Note that ``table`` (or ``fh``)  must return integer numbers
    representing indices in the axis :math:`t`. However it also possible to
    perform linear interpolation as part of the spreading/stacking process by
    providing the decimal part of the mapping function (:math:`t - \lfloor
    t \rfloor`) either in ``dtable`` input parameter or as second value in
    the return of ``fh`` function.

    """

    def __init__(
        self,
        dims,
        dimsd,
        table=None,
        dtable=None,
        fh=None,
        interp=None,
        engine="numpy",
        dtype="float64",
    ):
        if engine not in ["numpy", "numba"]:
            raise KeyError("engine must be numpy or numba")
        if engine == "numba" and jit is not None:
            self.engine = "numba"
        else:
            if engine == "numba" and jit is None:
                logging.warning(jit_message)
            self.engine = "numpy"

        # axes
        self.dims, self.dimsd = dims, dimsd
        self.nx0, self.nt0 = self.dims[0], self.dims[1]
        self.nx, self.nt = self.dimsd[0], self.dimsd[1]
        self.table = table
        self.dtable = dtable
        self.fh = fh

        # find out if mapping is in table of function handle
        if table is None and fh is None:
            raise NotImplementedError("provide either table or fh.")
        elif table is not None:
            if fh is not None:
                raise ValueError("provide only one of table or fh.")
            if self.table.shape != (self.nx0, self.nt0, self.nx):
                raise ValueError("table must have shape [nx0 x nt0 x nx]")
            self.usetable = True
            if np.any(self.table > self.nt):
                raise ValueError("values in table must be smaller than nt")
        else:
            self.usetable = False

        # find out if linear interpolation has to be carried out
        self.interp = False
        if self.usetable:
            if dtable is not None:
                if self.dtable.shape != (self.nx0, self.nt0, self.nx):
                    raise ValueError("dtable must have shape [nx0 x nt x nx]")
                self.interp = True
        else:
            if self.engine == "numba":
                self.interp = interp
            else:
                if len(fh(0, 0)) == 2:
                    self.interp = True
        if interp is not None and self.interp != interp:
            logging.warning("interp has been overridden to %r.", self.interp)
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
                    indices = (indices[mask]).astype(int)
                    if not self.interp:
                        y[mask, indices] += x[ix0, it]
                    else:
                        y[mask, indices] += (1 - dindices[mask]) * x[ix0, it]
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
                    indices = (indices[mask]).astype(int)
                    if not self.interp:
                        y[ix0, it] = np.sum(x[mask, indices])
                    else:
                        y[ix0, it] = np.sum(
                            x[mask, indices] * (1 - dindices[mask])
                        ) + np.sum(x[mask, indices + 1] * dindices[mask])
        return y.ravel()

    def _matvec(self, x):
        if self.engine == "numba":
            y = np.zeros(self.dimsd, dtype=self.dtype)
            if self.usetable:
                y = _matvec_numba_table(
                    x,
                    y,
                    self.dims,
                    self.interp,
                    self.table,
                    self.table if self.dtable is None else self.dtable,
                )
            else:
                y = _matvec_numba_onthefly(x, y, self.dims, self.interp, self.fh)
        else:
            y = self._matvec_numpy(x)
        return y

    def _rmatvec(self, x):
        if self.engine == "numba":
            y = np.zeros(self.dims, dtype=self.dtype)
            if self.usetable:
                y = _rmatvec_numba_table(
                    x,
                    y,
                    self.dims,
                    self.dimsd,
                    self.interp,
                    self.table,
                    self.table if self.dtable is None else self.dtable,
                )
            else:
                y = _rmatvec_numba_onthefly(
                    x, y, self.dims, self.dimsd, self.interp, self.fh
                )
        else:
            y = self._rmatvec_numpy(x)
        return y
