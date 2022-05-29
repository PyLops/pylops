import numpy as np

from pylops import LinearOperator
from pylops.utils._internal import _value_or_list_like_to_tuple
from pylops.utils.decorators import reshaped


class CausalIntegration(LinearOperator):
    r"""Causal integration.

    Apply causal integration to a multi-dimensional array along ``axis``.

    Parameters
    ----------
    dims : :obj:`list` or :obj:`int`
        Number of samples for each dimension
    axis : :obj:`int`, optional
        .. versionadded:: 2.0.0

        Axis along which the model is integrated.
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
    name : :obj:`str`, optional
        .. versionadded:: 2.0.0

        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

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
        dims,
        axis=-1,
        sampling=1,
        halfcurrent=True,
        dtype="float64",
        kind="full",
        removefirst=False,
        name="C",
    ):
        self.axis = axis
        self.sampling = sampling
        # backwards compatible
        self.kind = "half" if kind == "full" and halfcurrent else kind
        self.removefirst = removefirst
        dims = _value_or_list_like_to_tuple(dims)
        dimsd = list(dims)
        if self.removefirst:
            dimsd[self.axis] -= 1
        super().__init__(dtype=np.dtype(dtype), dims=dims, dimsd=dimsd, name=name)

    @reshaped(swapaxis=True)
    def _matvec(self, x):
        y = self.sampling * np.cumsum(x, axis=-1)
        if self.kind in ("half", "trapezoidal"):
            y -= self.sampling * x / 2.0
        if self.kind == "trapezoidal":
            y[..., 1:] -= self.sampling * x[..., 0:1] / 2.0
        if self.removefirst:
            y = y[..., 1:]
        return y

    @reshaped(swapaxis=True)
    def _rmatvec(self, x):
        if self.removefirst:
            x = np.insert(x, 0, 0, axis=-1)
        xflip = np.flip(x, axis=-1)
        if self.kind == "half":
            y = self.sampling * (np.cumsum(xflip, axis=-1) - xflip / 2.0)
        elif self.kind == "trapezoidal":
            y = self.sampling * (np.cumsum(xflip, axis=-1) - xflip / 2.0)
            y[..., -1] = self.sampling * np.sum(xflip, axis=-1) / 2.0
        else:
            y = self.sampling * np.cumsum(xflip, axis=-1)
        y = np.flip(y, axis=-1)
        return y
