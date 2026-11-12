r"""
Memoized forward and adjoint evaluations
========================================

This example shows how to use the :class:`pylops.MemoizeOperator`
operator to cache repeated applications of an operator.
"""

import numpy as np

import pylops

###############################################################################
# Let's start by creating a :class:`pylops.Diagonal` operator and wrapping it
# into a :class:`pylops.MemoizeOperator` operator choosing ``max_neval=2``
# (meaning that at most 4 evaluations are cached).
Op = pylops.Diagonal(np.array([2.0, 3.0, 4.0]))
MOp = pylops.MemoizeOperator(Op, max_neval=4)

###############################################################################
# Now we first apply twice the forward pass of operator (and check that only
# one entry is memoized), followed by one adjoint pass of operator (and check
# that now two entries are memoized)

x = np.array([1.0, -2.0, 0.5])
y = MOp @ x
y = MOp @ x
print("Number of evaluations: ", MOp.neval)

xadj = MOp.H @ y
print("Number of evaluations: ", MOp.neval)

###############################################################################
# Repeating either calculation reuses its own cached result.
print("y - ycached = ", y - MOp @ x)
print("xadj - xadjcached = ", xadj - MOp.H @ y)
print("Number of evaluations: ", MOp.neval)

###############################################################################
# Finally, we can see that similar inputs are not interchangeable: any
# small change triggers a new memoization.

x_nearby = x.copy()
x_nearby[0] += 1e-6
y_nearby = MOp @ x_nearby
print("Number of evaluations: ", MOp.neval)
