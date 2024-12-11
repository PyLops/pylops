import pylops
from pylops.utils import deps

pytensor_message = deps.pytensor_import("the pytensor module")

if pytensor_message is not None:

    class PyTensorOperator:
        """PyTensor Op which applies a PyLops Linear Operator, including gradient support.

        This class "converts" a PyLops `LinearOperator` class into a PyTensor `Op`.
        This applies the `LinearOperator` in "forward-mode" in `self.perform`, and applies
        its adjoint when computing the vector-Jacobian product (`self.grad`), as that is
        the analytically correct gradient for linear operators. This class should pass
        `pytensor.gradient.verify_grad`.

        Parameters
        ----------
        LOp : pylops.LinearOperator
        """

        def __init__(self, LOp: pylops.LinearOperator) -> None:
            if not deps.pytensor_enabled:
                raise NotImplementedError(pytensor_message)

else:
    import pytensor.tensor as pt
    from pytensor.graph.basic import Apply
    from pytensor.graph.op import Op

    class _PyTensorOperatorNoGrad(Op):
        """PyTensor Op which applies a PyLops Linear Operator, excluding gradient support.

        This class "converts" a PyLops `LinearOperator` class into a PyTensor `Op`.
        This applies the `LinearOperator` in "forward-mode" in `self.perform`.

        Parameters
        ----------
        LOp : pylops.LinearOperator
        """

        __props__ = ("dims", "dimsd", "shape")

        def __init__(self, LOp: pylops.LinearOperator) -> None:
            self._LOp = LOp
            self.dims = self._LOp.dims
            self.dimsd = self._LOp.dimsd
            self.shape = self._LOp.shape
            super().__init__()

        def make_node(self, x) -> Apply:
            x = pt.as_tensor_variable(x)
            inputs = [x]
            outputs = [pt.tensor(dtype=x.type.dtype, shape=self._LOp.dimsd)]
            return Apply(self, inputs, outputs)

        def perform(
            self, node: Apply, inputs: list, output_storage: list[list[None]]
        ) -> None:
            (x,) = inputs
            (yt,) = output_storage
            yt[0] = self._LOp @ x

    class PyTensorOperator(_PyTensorOperatorNoGrad):
        """PyTensor Op which applies a PyLops Linear Operator, including gradient support.

        This class "converts" a PyLops `LinearOperator` class into a PyTensor `Op`.
        This applies the `LinearOperator` in "forward-mode" in `self.perform`, and applies
        its adjoint when computing the vector-Jacobian product (`self.grad`), as that is
        the analytically correct gradient for linear operators. This class should pass
        `pytensor.gradient.verify_grad`.

        Parameters
        ----------
        LOp : pylops.LinearOperator
        """

        def __init__(self, LOp: pylops.LinearOperator) -> None:
            super().__init__(LOp)
            self._gradient_op = _PyTensorOperatorNoGrad(self._LOp.H)

        def grad(
            self, inputs: list[pt.TensorVariable], output_grads: list[pt.TensorVariable]
        ):
            return [self._gradient_op(output_grads[0])]
