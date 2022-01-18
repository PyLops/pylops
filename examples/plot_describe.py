r"""
Describe
========
This example focuses on the usage of the :func:`pylops.utils.describe.describe`
method, which allows expressing any PyLops operator into its equivalent
mathematical representation. This is done with the aid of
`sympy <https://docs.sympy.org>`_, a Python library for symbolic computing

"""
import matplotlib.pyplot as plt
import numpy as np

import pylops
from pylops.utils.describe import describe

plt.close("all")

###############################################################################
# Let's start by defining 3 PyLops operators. Note that once an operator is
# defined we can attach a name to the operator; by doing so, this name will
# be used in the mathematical description of the operator. Alternatively,
# the describe method will randomly choose a name for us.

A = pylops.MatrixMult(np.ones((10, 5)))
A.name = "A"
B = pylops.Diagonal(np.ones(5))
B.name = "A"
C = pylops.MatrixMult(np.ones((10, 5)))

# Simple operator
describe(A)

# Transpose
AT = A.T
describe(AT)

# Adjoint
AH = A.H
describe(AH)

# Scaled
A3 = 3 * A
describe(A3)

# Sum
D = A + C
describe(D)

###############################################################################
# So good so far. Let's see what happens if we accidentally call two different
# operators with the same name. You will see that PyLops catches that and
# changes the name for us (and provides us with a nice warning!)

D = A * B
describe(D)

###############################################################################
# We can move now to something more complicated using various composition
# operators

H = pylops.HStack((A * B, C * B))
describe(H)

H = pylops.Block([[A * B, C], [A, A]])
describe(H)

###############################################################################
# Finally, note that you can get the best out of the describe method if working
# inside a Jupyter notebook. There, the mathematical expression will be
# rendered using a LeTex format!
