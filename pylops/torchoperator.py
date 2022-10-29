__all__ = [
    "TorchOperator",
]

from typing import Optional, Union

import numpy as np

from pylops import LinearOperator
from pylops.utils import deps

if deps.torch_enabled:
    from pylops._torchoperator import _TorchOperator
else:
    torch_message = (
        "Torch package not installed. In order to be able to use"
        'the twoway module run "pip install torch" or'
        '"conda install -c pytorch torch".'
    )
from pylops.utils.typing import InputDimsLike, TensorTypeLike


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
        If ``batch==False`` the input must be a 1-d Torch tensor,
        if ``batch==True`` the input must be a 2-d Torch tensor with
        batches along the first dimension. Alternatively, if ``dims`` is
        provided the input can be a NDarray (with batches along the first
        dimension)
    dims : :obj:`list` or :obj:`int`, optional
        Number of samples for each dimension (excluding batch size)
    device : :obj:`str`, optional
        Device to be used when applying operator (``cpu`` or ``gpu``)
    devicetorch : :obj:`str`, optional
        Device to be assigned the output of the operator to (any Torch-compatible device)

    """

    def __init__(
        self,
        Op: LinearOperator,
        batch: bool = False,
        dims: Optional[Union[int, InputDimsLike]] = None,
        device: str = "cpu",
        devicetorch: str = "cpu",
    ) -> None:
        if not deps.torch_enabled:
            raise NotImplementedError(torch_message)
        self.device = device
        self.devicetorch = devicetorch
        # define transpose indices to bring batch to last dimension before applying
        # pylops forward and adjoint (this will call matmat and rmatmat)
        self.transpf = np.roll(np.arange(2 if dims is None else len(dims) + 1), -1)
        self.transpb = np.roll(np.arange(2 if dims is None else len(dims) + 1), 1)
        if not batch:
            self.matvec = Op.matvec
            self.rmatvec = Op.rmatvec
        else:
            self.matvec = lambda x: (Op @ x.transpose(self.transpf)).transpose(
                self.transpb
            )
            self.rmatvec = lambda x: (Op.H @ x.transpose(self.transpf)).transpose(
                self.transpb
            )
        self.Top = _TorchOperator.apply
        super().__init__(
            dtype=np.dtype(Op.dtype), dims=Op.dims, dimsd=Op.dims, name=Op.name
        )

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
