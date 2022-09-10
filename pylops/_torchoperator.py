import logging

from pylops.utils import deps

if deps.torch_enabled:
    import torch
    from torch.utils.dlpack import from_dlpack, to_dlpack

if deps.cupy_enabled:
    import cupy as cp


class _TorchOperator(torch.autograd.Function):
    """Wrapper class for PyLops operators into Torch functions"""

    @staticmethod
    def forward(ctx, x, forw, adj, device, devicetorch):
        ctx.forw = forw
        ctx.adj = adj
        ctx.device = device
        ctx.devicetorch = devicetorch

        # check if data is moved to cpu and warn user
        if ctx.device == "cpu" and ctx.devicetorch != "cpu":
            logging.warning(
                "pylops operator will be applied on the cpu "
                "whilst the input torch vector is on "
                "%s, this may lead to poor performance" % ctx.devicetorch
            )

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
            y = torch.from_numpy(y).to(ctx.devicetorch)
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
            x = torch.from_numpy(x).to(ctx.devicetorch)
        else:
            x = from_dlpack(x.toDlpack())
        return x, None, None, None, None, None
