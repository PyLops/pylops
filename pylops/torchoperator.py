__all__ = [
    "TorchOperator",
]


from math import prod

import numpy as np

from pylops import LinearOperator
from pylops.utils import deps

if deps.torch_enabled:
    from pylops._torchoperator import _TorchOperator
else:
    torch_message = (
        "Torch package not installed. In order to be able to use"
        'the torchoperator module run "pip install torch" or'
        '"conda install -c pytorch torch".'
    )
from pylops.utils.typing import TensorTypeLike


class TorchOperator:
    """Wrap a PyLops operator into a Torch function.

    This class can be used to wrap a pylops operator into a
    torch function. Doing so, users can mix native torch functions (e.g.
    basic linear algebra operations, neural networks, etc.) and pylops
    operators.

    Since all operators in PyLops are linear operators, a Torch function is
    simply implemented by using the forward operator for its forward pass
    and the adjoint operator for its backward (gradient) pass.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        PyLops operator
    batch : :obj:`bool`, optional
        Input has single sample (``False``) or batch of samples (``True``).
        If ``batch=False``, the input must be a 1-d Torch tensor of size
        ``Op.shape[1]`` for ``flatten=True`` or a tensor of size ``Op.dims``
        for ``flatten=False``; if ``batch=True``, the input must be a 2-d Torch
        tensor of size ``[nbatch, Op.shape[1]]`` for ``flatten=True`` or a tensor
        of size ``[nbatch, *Op.dims]`` for ``flatten=False``, where ``nbatch`` is
        the size of the batch. The output follows the same convention with
        ``Op.shape[0]`` and ``Op.dimsd``. When using :meth:`adjoint`, the roles of
        ``Op.shape[1]``/``Op.dims`` and ``Op.shape[0]``/``Op.dimsd`` are swapped.
    flatten : :obj:`bool`, optional
        Input beyond the batch dimension is flattened (``True``) or not (``False``).
        See ``batch`` for more details.
    device : :obj:`str`, optional
        Device to be used when applying operator (``cpu`` or ``gpu``)
    devicetorch : :obj:`str`, optional
        Device to be assigned the output of the operator to (any Torch-compatible device)

    Attributes
    ----------
    transpf : :obj:`numpy.ndarray`
        *Deprecated*, will be removed in v3.0.0.
        Indices to transpose batch to last dimension before applying ``Op``
    transpb : :obj:`numpy.ndarray`
        *Deprecated*, will be removed in v3.0.0.
        Indices to transpose back batch to first dimension after applying ``Op``

    """

    def __init__(
        self,
        Op: LinearOperator,
        batch: bool = False,
        flatten: bool | None = True,
        device: str = "cpu",
        devicetorch: str = "cpu",
    ) -> None:
        if not deps.torch_enabled:
            raise NotImplementedError(torch_message)
        self.device = device
        self.devicetorch = devicetorch
        self.dtype = np.dtype(Op.dtype)
        self.dims, self.dimsd = Op.dims, Op.dimsd
        self.name = Op.name
        # define transpose indices to move the batch axis before and after
        # applying the forward operator (deprecated, kept only for backward
        # compatibility of the transpf and transpb attributes)
        self.transpf = np.roll(np.arange(2 if flatten else len(self.dims) + 1), -1)
        self.transpb = np.roll(np.arange(2 if flatten else len(self.dimsd) + 1), 1)
        if batch and not flatten:
            # batched and unflattened
            # flatten inputs of size ``[nbatch, *Op.dims]`` (or
            # ``[nbatch, *Op.dimsd]``), transpose, apply matmat (or rmatmat),
            # transpose back, and reshape outputs to ``[nbatch, *Op.dimsd]``
            # (or ``[nbatch, *Op.dims]``)
            self.matvec = lambda x: Op.matmat(x.reshape(x.shape[0], -1).T).T.reshape(
                x.shape[0], *self.dimsd
            )
            self.rmatvec = lambda x: Op.rmatmat(x.reshape(x.shape[0], -1).T).T.reshape(
                x.shape[0], *self.dims
            )
        elif batch:
            # batched and flattened
            # transpose, apply matmat (or rmatmat), and transpose back
            self.matvec = lambda x: Op.matmat(x.T).T
            self.rmatvec = lambda x: Op.rmatmat(x.T).T
        elif not flatten:
            # unbatched and unflattened
            # flatten inputs of size ``Op.dims`` (or ``Op.dimsd``), apply
            # matvec (or rmatvec), and reshape outputs to ``Op.dimsd``
            # (or ``Op.dims``)
            self.matvec = lambda x: Op.matvec(x.ravel()).reshape(self.dimsd)
            self.rmatvec = lambda x: Op.rmatvec(x.ravel()).reshape(self.dims)
        else:
            # unbatched and flattened
            # use directly matvec/rmatvec
            self.matvec = Op.matvec
            self.rmatvec = Op.rmatvec
        self.Top = _TorchOperator.apply

    def __call__(self, x):
        return self.apply(x)

    def __repr__(self):
        M, N = prod(self.dimsd), prod(self.dims)
        if self.dtype is None:
            dt = "unspecified dtype"
        else:
            dt = "dtype=" + str(self.dtype)

        return "<%dx%d %s with %s>" % (M, N, self.__class__.__name__, dt)

    def apply(self, x: TensorTypeLike) -> TensorTypeLike:
        """Apply forward pass to input vector

        Parameters
        ----------
        x : :obj:`torch.Tensor`
            Input array

        Returns
        -------
        y : :obj:`torch.Tensor`
            Output array resulting from the application of the operator to ``x``.

        """
        return self.Top(x, self.matvec, self.rmatvec, self.device, self.devicetorch)

    # alias for forward pass
    forward = apply

    def adjoint(self, x: TensorTypeLike) -> TensorTypeLike:
        """Apply adjoint pass to input vector

        Parameters
        ----------
        x : :obj:`torch.Tensor`
            Input array

        Returns
        -------
        y : :obj:`torch.Tensor`
            Output array resulting from the application of the adjoint operator to ``x``.

        """
        return self.Top(x, self.rmatvec, self.matvec, self.device, self.devicetorch)
