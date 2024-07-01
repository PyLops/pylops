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
            explicit=False,
            forceflat=Op.forceflat,
            name=Op.name,
        )
        self._matvec = jax.jit(Op._matvec)
        self._rmatvec = jax.jit(Op._rmatvec)

    def __call__(self, x, *args, **kwargs):
        return self._matvec(x)

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
        return jax.jit(f_vjp)(y)[0]

    def rmatvecad(self, x: JaxType, y: JaxType) -> JaxType:
        """Adjoint matrix-vector multiplication with AD

        Parameters
        ----------
        x : :obj:`jaxlib.xla_extension.ArrayImpl`
            Input array
        y : :obj:`jaxlib.xla_extension.ArrayImpl`
            Output array (where to store the
            Vector-Jacobian product)

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Output array of shape (N,) or (N,1)

        """
        M, N = self.shape

        if x.shape != (M,) and x.shape != (M, 1):
            raise ValueError("dimension mismatch")

        y = self._rmatvecad(x, y)

        if x.ndim == 1:
            y = y.reshape(N)
        elif x.ndim == 2:
            y = y.reshape(N, 1)
        else:
            raise ValueError("invalid shape returned by user-defined rmatvecad()")
        return y
