import numpy as np

from pylops import LinearOperator


class CausalIntegration(LinearOperator):
    r"""Causal integration.

    Apply causal integration to a multi-dimensional array along ``dir`` axis.

    Parameters
    ----------
    N : :obj:`int`
        Number of samples in model.
    dims : :obj:`list`, optional
        Number of samples for each dimension
        (``None`` if only one dimension is available)
    dir : :obj:`int`, optional
        Direction along which smoothing is applied.
    sampling : :obj:`float`, optional
        Sampling step ``dx``.
    halfcurrent : :obj:`bool`, optional
        Add half of current value (``True``) or the entire value (``False``).
        This will be *deprecated* in v2.0.0, use instead `kind=half` to obtain
        the same behaviour.
    dtype : :obj:`str`, optional
        Type of elements in input array.
    kind : :obj:`str`, optional
        Integration kind (``full``, ``half``, or ``trapezoidal``).
    removefirst : :obj:`bool`, optional
        Remove first sample (``True``) or not (``False``).

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly (``True``)
        or not (``False``)

    Notes
    -----
    The CausalIntegration operator applies a causal integration to any chosen
    direction of a multi-dimensional array.

    For simplicity, given a one dimensional array, the causal integration is:

    .. math::
        y(t) = \int\limits_{-\infty}^t x(\tau) \,\mathrm{d}\tau

    which can be discretised as :

    .. math::
        y[i] = \sum_{j=0}^i x[j] \,\Delta t

    or

    .. math::
        y[i] = \left(\sum_{j=0}^{i-1} x[j] + 0.5x[i]\right) \,\Delta t

    or

    .. math::
        y[i] = \left(\sum_{j=1}^{i-1} x[j] + 0.5x[0] + 0.5x[i]\right) \,\Delta t

    where :math:`\Delta t` is the ``sampling`` interval, and assuming the signal is zero
    before sample :math:`j=0`. In our implementation, the
    choice to add :math:`x[i]` or :math:`0.5x[i]` is made by selecting ``kind=full``
    or ``kind=half``, respectively. The choice to add :math:`0.5x[i]` and
    :math:`0.5x[0]` instead of made by selecting the ``kind=trapezoidal``.

    Note that the causal integral of a signal will depend, up to a constant,
    on causal start of the signal. For example if :math:`x(\tau) = t^2` the
    resulting indefinite integration is:

    .. math::
        y(t) = \int \tau^2 \,\mathrm{d}\tau = \frac{t^3}{3} + C

    However, if we apply a first derivative to :math:`y` always obtain:

    .. math::
        x(t) = \frac{\mathrm{d}y}{\mathrm{d}t} = t^2

    no matter the choice of :math:`C`.

    """

    def __init__(
        self,
        N,
        dims=None,
        dir=-1,
        sampling=1,
        halfcurrent=True,
        dtype="float64",
        kind="full",
        removefirst=False,
    ):
        self.N = N
        self.dir = dir
        self.sampling = sampling
        self.kind = kind
        if kind == "full" and halfcurrent:  # ensure backcompatibility
            self.kind = "half"
        self.removefirst = removefirst
        # define samples to remove from output
        rf = 0
        if removefirst:
            rf = 1 if dims is None else self.N // dims[self.dir]
        if dims is None:
            self.dims = [self.N, 1]
            self.dimsd = [self.N - rf, 1]
            self.reshape = False
        else:
            if np.prod(dims) != self.N:
                raise ValueError("product of dims must equal N!")
            else:
                self.dims = dims
                self.dimsd = list(dims)
                if self.removefirst:
                    self.dimsd[self.dir] -= 1
                self.reshape = True
        self.shape = (self.N - rf, self.N)
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        if self.reshape:
            x = np.reshape(x, self.dims)
        if self.dir != -1:
            x = np.swapaxes(x, self.dir, -1)
        y = self.sampling * np.cumsum(x, axis=-1)
        if self.kind in ("half", "trapezoidal"):
            y -= self.sampling * x / 2.0
        if self.kind == "trapezoidal":
            y[..., 1:] -= self.sampling * x[..., 0:1] / 2.0
        if self.removefirst:
            y = y[..., 1:]
        if self.dir != -1:
            y = np.swapaxes(y, -1, self.dir)
        return y.ravel()

    def _rmatvec(self, x):
        if self.reshape:
            x = np.reshape(x, self.dimsd)
        if self.removefirst:
            x = np.insert(x, 0, 0, axis=self.dir)
        if self.dir != -1:
            x = np.swapaxes(x, self.dir, -1)
        xflip = np.flip(x, axis=-1)
        if self.kind == "half":
            y = self.sampling * (np.cumsum(xflip, axis=-1) - xflip / 2.0)
        elif self.kind == "trapezoidal":
            y = self.sampling * (np.cumsum(xflip, axis=-1) - xflip / 2.0)
            y[..., -1] = self.sampling * np.sum(xflip, axis=-1) / 2.0
        else:
            y = self.sampling * np.cumsum(xflip, axis=-1)
        y = np.flip(y, axis=-1)
        if self.dir != -1:
            y = np.swapaxes(y, -1, self.dir)
        return y.ravel()
