__all__ = [
    "JaxOperator",
]

from typing import Any, NewType

from pylops import LinearOperator
from pylops.utils import deps

if deps.jax_enabled:
    import jax
    import jaxlib

    jaxarray_type = jaxlib.xla_extension.ArrayImpl
else:
    jaxarray_type = Any

JaxType = NewType("JaxType", jaxarray_type)


class JaxOperator(LinearOperator):
    def __init__(self, Op: LinearOperator) -> None:
        super().__init__(
            dtype=Op.dtype,
            dims=Op.dims,
            dimsd=Op.dimsd,
            clinear=Op.clinear,
            explicit=Op.explicit,
            forceflat=Op.forceflat,
            name=Op.name,
        )
        self._matvec = jax.jit(Op._matvec)
        self._rmatvec = jax.jit(Op._rmatvec)

    def _rmatvecad(self, x: JaxType, y: JaxType) -> JaxType:
        """Vector-Jacobian products

        JIT-compiled Vector-Jacobian product

        Parameters
        ----------
        x : :obj:`jaxlib.xla_extension.ArrayImpl`
            Input array
        y : :obj:`jaxlib.xla_extension.ArrayImpl`
            Output array (where to store the
            Vector-Jacobian product)

        Returns
        ----------
        y : :obj:`jaxlib.xla_extension.ArrayImpl`
            Output array

        """
        _, f_vjp = jax.vjp(self._matvec, x)
        return jax.jit(f_vjp)(y)
