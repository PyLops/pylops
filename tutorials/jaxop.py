r"""
21. JAX Operator
================
This tutorial is aimed at introducing the :class:`pylops.JaxOperator` operator. This
represents the entry-point to the JAX backend of PyLops.

More specifically, by wrapping any of PyLops' operators into a
:class:`pylops.JaxOperator` one can:

- apply forward, adjoint and use any of PyLops solver with JAX arrays;
- enable automatic differentiation;
- enable automatic vectorization.

Moreover, both the forward and adjoint are internally just-in-time compiled
to enable any further optimization provided by JAX.

In this example we will consider a :class:`pylops.MatrixMult` operator and
showcase how to use it in conjunction with :class:`pylops.JaxOperator`
to enable the different JAX functionalities mentioned above.

"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import pylops

plt.close("all")
np.random.seed(10)

###############################################################################
# Let's start by creating a :class:`pylops.MatrixMult` operator. We will then
# perform the dot-test as well as apply the forward and adjoint operations to
# JAX arrays.

n = 4
G = np.random.normal(0, 1, (n, n)).astype("float32")
Gopjax = pylops.JaxOperator(pylops.MatrixMult(jnp.array(G), dtype="float32"))

# dottest
pylops.utils.dottest(Gopjax, n, n, backend="jax", verb=True, atol=1e-3)

# forward
xjnp = jnp.ones(n, dtype="float32")
yjnp = Gopjax @ xjnp

# adjoint
xadjjnp = Gopjax.H @ yjnp

###############################################################################
# We can now use one of PyLops solvers to invert the operator

xcgls = pylops.optimization.basic.cgls(
    Gopjax, yjnp, x0=jnp.zeros(n), niter=100, tol=1e-10, show=True
)[0]
print("Inverse: ", xcgls)

###############################################################################
# Let's see how we can empower the automatic differentiation capabilities
# of JAX to obtain the adjoint of our operator without having to implement it.
# Although in PyLops the adjoint of any of operators is hand-written (and
# optimized), it may be useful in some cases to quickly implement the forward
# pass of a new operator and get the adjoint for free. This could be extremely
# beneficial during the prototyping stage of an operator before embarking in
# implementing an efficient hand-written adjoint.

xadjjnpad = Gopjax.rmatvecad(xjnp, yjnp)

print("Hand-written Adjoint: ", xadjjnp)
print("AD Adjoint: ", xadjjnpad)

###############################################################################
# And more in general how we can combine any of JAX native operations with a
# PyLops operator.


def fun(x):
    y = Gopjax(x)
    loss = jnp.sum(y)
    return loss


xgrad = jax.grad(fun)(xjnp)
print("Grad: ", xgrad)

###############################################################################
# We turn now our attention to automatic vectorization, which is very useful
# if we want to apply the same operator to multiple vectors. In PyLops we can
# easily do so by using the ``matmat`` and ``rmatmat`` methods, however under
# the hood what these methods do is to simply run a for...loop and call the
# corresponding ``matvec`` / ``rmatvec`` methods multiple times. On the other
# hand, JAX is able to automatically add a batch axis at the beginning of
# operator. Moreover, this can be seamlessly combined with `jax.jit` to
# further improve performance.

auto_batch_matvec = jax.jit(jax.vmap(Gopjax._matvec))
xs = jnp.stack([xjnp, xjnp])
ys = auto_batch_matvec(xs)

print("Original output: ", yjnp)
print("AV Output 1: ", ys[0])
print("AV Output 1: ", ys[1])
