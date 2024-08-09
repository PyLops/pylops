__all__ = [
    "JaxOperator",
]

from typing import Any, NewType

from pylops import LinearOperator
from pylops.utils import deps

if deps.jax_enabled:
    import jax

    jaxarrayin_type = jax.typing.ArrayLike
    jaxarrayout_type = jax.Array
else:
    jax_message = (
        "JAX package not installed. In order to be able to use"
        'the jaxoperator module run "pip install jax" or'
        '"conda install -c conda-forge jax".'
    )
    jaxarrayin_type = Any
    jaxarrayout_type = Any

JaxTypeIn = NewType("JaxTypeIn", jaxarrayin_type)
JaxTypeOut = NewType("JaxTypeOut", jaxarrayout_type)


class JaxOperator(LinearOperator):
    """Enable JAX backend for PyLops operator.

    This class can be used to wrap a pylops operator to enable the JAX
    backend. Doing so, users can run all of the methods of a pylops
    operator with JAX arrays. Moreover, the forward and adjoint
    are internally just-in-time compiled, and other JAX functionalities
    such as automatic differentiation and automatic vectorization
    are enabled.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        PyLops operator

    """

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

    def _rmatvecad(self, x: JaxTypeIn, y: JaxTypeIn) -> JaxTypeOut:
        _, f_vjp = jax.vjp(self._matvec, x)
        xadj = jax.jit(f_vjp)(y)[0]
        return xadj

    def rmatvecad(self, x: JaxTypeIn, y: JaxTypeIn) -> JaxTypeOut:
        """Vector-Jacobian product

        JIT-compiled Vector-Jacobian product

        Parameters
        ----------
        x : :obj:`jax.Array`
            Input array for forward
        y : :obj:`jax.Array`
            Input array for adjoint

        Returns
        -------
        xadj : :obj:`jax.typing.ArrayLike`
            Output array

        """
        M, N = self.shape

        if x.shape != (M,) and x.shape != (M, 1):
            raise ValueError(
                f"Dimension mismatch. Got {x.shape}, but expected  ({M},) or ({M}, 1)."
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
