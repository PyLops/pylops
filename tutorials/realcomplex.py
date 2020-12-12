r"""
17. Real/Complex Inversion
==========================
In this tutorial we will discuss two equivalent approaches to the solution
of inverse problems with real-valued model vector and complex-valued data vector.
In other words, we consider a modelling operator
:math:`\mathbf{A}:\mathbb{F}^m \to \mathbb{C}^n` (which could be the case
for example for the real FFT).

Mathematically speaking, this problem can be solved equivalently by inverting
the complex-valued problem:

.. math::
   \mathbf{y} = \mathbf{A} \mathbf{x}

or the real-valued augmented system

.. math::
   \begin{bmatrix}
       Re(\mathbf{y})  \\
       Im(\mathbf{y})
   \end{bmatrix} =
   \begin{bmatrix}
       Re(\mathbf{A})  \\
       Im(\mathbf{A})
   \end{bmatrix}  \mathbf{x}

Whilst we already know how to solve the first problem, let's see how we can
solve the second one by taking advantage of the ``real`` method of the
:class:`pylops.LinearOperator` object. We will also wrap our linear operator
into a :class:`pylops.MemoizeOperator` which remembers the last N model and
data vectors and by-passes the computation of the forward and/or adjoint pass
whenever the same pair reappears. This is very useful in our case when we
want to compute the real and the imag components of

"""
import numpy as np
import matplotlib.pyplot as plt
import pylops

plt.close('all')
np.random.seed(0)

###############################################################################
# To start we create the forward problem

n = 5
x = np.arange(n) + 1.

# make A
Ar = np.random.normal(0, 1, (n, n))
Ai = np.random.normal(0, 1, (n, n))
A = Ar + 1j * Ai
Aop = pylops.MatrixMult(A, dtype=np.complex)
y = Aop @ x

###############################################################################
# Let's check we can solve this problem using the first formulation
A1op = Aop.toreal(forw=False, adj=True)
xinv = A1op.div(y)

print('xinv=%s\n' % xinv)

###############################################################################
# Let's now see how we formulate the second problem
Amop = pylops.MemoizeOperator(Aop, max_neval=10)
Arop = Amop.toreal()
Aiop = Amop.toimag()

A1op = pylops.VStack([Arop, Aiop])
y1 = np.concatenate([np.real(y), np.imag(y)])
xinv1 = np.real(A1op.div(y1))

print('xinv1=%s\n' % xinv1)
