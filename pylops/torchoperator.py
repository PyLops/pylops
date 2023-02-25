__all__ = [
    "TorchOperator",
]

from typing import Optional, Callable

import numpy as np

from pylops import LinearOperator
from pylops.utils import deps, NDArray

if deps.torch_enabled:
    from pylops._torchoperator import _TorchOperator
else:
    torch_message = (
        "Torch package not installed. In order to be able to use"
        'the twoway module run "pip install torch" or'
        '"conda install -c pytorch torch".'
    )
from pylops.utils.typing import TensorTypeLike


class TorchOperator(LinearOperator):
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
        If ``batch==False`` the input must be a 1-d Torch tensor or a tensor of
        size equal to ``Op.dims``; if ``batch==True`` the input must be a 2-d Torch
        tensor with batches along the first dimension or a tensor of size equal to
        ``[nbatch, *Op.dims]`` where ``nbatch`` is the size of the batch
    flatten : :obj:`bool`, optional
        Input is flattened along ``Op.dims`` (``True``) or not (``False``)
    device : :obj:`str`, optional
        Device to be used when applying operator (``cpu`` or ``gpu``)
    devicetorch : :obj:`str`, optional
        Device to be assigned the output of the operator to (any Torch-compatible device)

    """

    def __init__(
        self,
        Op: LinearOperator,
        batch: bool = False,
        flatten: Optional[bool] = True,
        device: str = "cpu",
        devicetorch: str = "cpu",
    ) -> None:
        if not deps.torch_enabled:
            raise NotImplementedError(torch_message)
        self.device = device
        self.devicetorch = devicetorch
        super().__init__(
            dtype=np.dtype(Op.dtype), dims=Op.dims, dimsd=Op.dims, name=Op.name
        )
        # define transpose indices to bring batch to last dimension before applying
        # pylops forward and adjoint (this will call matmat and rmatmat)
        self.transpf = np.roll(np.arange(2 if flatten else len(self.dims) + 1), -1)
        self.transpb = np.roll(np.arange(2 if flatten else len(self.dims) + 1), 1)
        self.batch = batch
        self.Op = Op
        self._register_torchop()
        self.Top = _TorchOperator.apply

    def _register_torchop(self):
        # choose _matvec and _rmatvec
        self._hmatvec: Callable
        self._hrmatvec: Callable

        if not self.batch:
            self._hmatvec = lambda x: self.Op @ x
            self._hrmatvec = lambda x: self.Op.H @ x
        else:
            self._hmatvec = lambda x: (self.Op @ x.transpose(self.transpf)).transpose(self.transpb)
            self._hrmatvec = lambda x: (self.Op.H @ x.transpose(self.transpf)).transpose(self.transpb)

    def _matvec(self, x: NDArray) -> NDArray:
        return self._hmatvec(x)

    def _rmatvec(self, x: NDArray) -> NDArray:
        return self._hrmatvec(x)

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
        return self.Top(x, self._hmatvec, self._hrmatvec, self.device, self.devicetorch)
