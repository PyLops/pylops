__all__ = [
    "JaxOperator",
]


import jax

from pylops import LinearOperator


class JaxOperator(LinearOperator):
    def __init__(self, Op):
        super().__init__(dtype=Op.dtype, dims=Op.dims, dimsd=Op.dimsd, name=Op.name)
        self._matvec = jax.jit(Op._matvec)
        self._rmatvec = jax.jit(Op._rmatvec)

    def _rmatvecad(self, x, y):
        _, f_vjp = jax.vjp(self._matvec, x)
        return jax.jit(f_vjp)(y)
