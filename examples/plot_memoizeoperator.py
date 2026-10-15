r"""
Memoized forward and adjoint evaluations
=======================================

:class:`pylops.MemoizeOperator` caches repeated applications of an operator.
The forward and adjoint directions have separate cache entries, since an
adjoint generally does not undo a forward application.
"""

import numpy as np

import pylops

###############################################################################
# A non-unitary diagonal operator makes the distinction explicit:
# applying the adjoint to ``A @ x`` gives ``A.H @ A @ x``, rather than ``x``.
op = pylops.Diagonal(np.array([2.0, 3.0, 4.0]))
memo = pylops.MemoizeOperator(op, max_neval=2)
x = np.array([1.0, -2.0, 0.5])
y = memo @ x
z = memo.H @ y
np.testing.assert_allclose(z, op.H @ (op @ x))
assert memo.neval == 2

###############################################################################
# Repeating either calculation reuses its own cached result. Both entries
# share the two-evaluation budget, and the stored vectors remain (x, y) pairs.
np.testing.assert_allclose(memo @ x, y)
np.testing.assert_allclose(memo.H @ y, z)
assert memo.neval == 2
assert len(memo.store) == 2
