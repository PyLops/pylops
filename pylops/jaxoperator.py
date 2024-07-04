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
    jax_message = (
        "JAX package not installed. In order to be able to use"
        'the jaxoperator module run "pip install jax" or'
        '"conda install -c conda-forge jax".'
    )
    jaxarray_type = Any

JaxType = NewType("JaxType", jaxarray_type)


class JaxOperator(LinearOperator):
    def __init__(self, Op: LinearOperator) -> None:
        if not deps.jax_enabled:
            raise NotImplementedError(jax_message)
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
            Input array for forward
        y : :obj:`jaxlib.xla_extension.ArrayImpl`
            Input array for adjoint

        Returns
        -------
        xadj : :obj:`jaxlib.xla_extension.ArrayImpl`
            Output array

        """
        _, f_vjp = jax.vjp(self._matvec, x)
        xadj = jax.jit(f_vjp)(y)[0]
        return xadj

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
            raise ValueError(
                f"Dimension mismatch. Got {x.shape}, but expected {(M, 1)} or {(M,)}."
            )

        y = self._rmatvecad(x, y)

        if x.ndim == 1:
            y = y.reshape(N)
        elif x.ndim == 2:
            y = y.reshape(N, 1)
        else:
            raise ValueError(
                f"Invalid shape returned by user-defined rmatvecad(). "
                f"Expected 2-d ndarray or matrix, not {x.ndim}-d ndarray"
            )
        return y
