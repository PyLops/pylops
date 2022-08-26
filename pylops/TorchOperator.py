from pylops.utils import deps

if deps.torch_enabled:
    import torch
    from torch.utils.dlpack import from_dlpack, to_dlpack
else:
    torch_message = (
        "Torch package not installed. In order to be able to use"
        'the twoway module run "pip install torch" or'
        '"conda install -c pytorch torch".'
    )
if deps.cupy_enabled:
    import cupy as cp
else:
    cp = None


class _TorchOperator(torch.autograd.Function):
    """Wrapper class for PyLops operators into Torch functions"""

    @staticmethod
    def forward(ctx, x, forw, adj, device):
        ctx.forw = forw
        ctx.adj = adj
        ctx.device = device

        # prepare input
        if ctx.device == "cpu":
            # bring x to cpu and numpy
            x = x.cpu().detach().numpy()
        else:
            # pass x to cupy using DLPack
            x = cp.fromDlpack(to_dlpack(x))

        # apply forward operator
        y = ctx.forw(x)

        # prepare output
        if ctx.device == "cpu":
            # move y to torch and device
            y = torch.from_numpy(y)
        else:
            # move y to torch and device
            y = from_dlpack(y.toDlpack())
        return y

    @staticmethod
    def backward(ctx, y):
        # prepare input
        if ctx.device == "cpu":
            y = y.cpu().detach().numpy()
        else:
            # pass x to cupy using DLPack
            y = cp.fromDlpack(to_dlpack(y))

        # apply adjoint operator
        x = ctx.adj(y)

        # prepare output
        if ctx.device == "cpu":
            x = torch.from_numpy(x)
        else:
            x = from_dlpack(x.toDlpack())
        return x, None, None, None, None


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
        If ``batch==False`` the input must be a 1-d Torch tensor,
        if `batch==False`` the input must be a 2-d Torch tensor with
        batches along the first dimension
    device : :obj:`str`, optional
        Device to be used for output vectors when ``Op`` is a pylops operator

    Returns
    -------
    y : :obj:`torch.Tensor`
        Output array resulting from the application of the operator to ``x``.

    """

    def __init__(self, Op, batch=False, device="cpu"):
        self.device = device
        if not batch:
            self.matvec = Op.matvec
            self.rmatvec = Op.rmatvec
        else:
            self.matvec = lambda x: Op.matmat(x.T, kfirst=True).T
            self.rmatvec = lambda x: Op.rmatmat(x.T, kfirst=True).T
        self.Top = _TorchOperator.apply

    def apply(self, x):
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
        return self.Top(x, self.matvec, self.rmatvec, self.device)