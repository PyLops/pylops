import numpy as np

from pylops.basicoperators import BlockDiag, Diagonal, HStack, MatrixMult
from pylops.utils.describe import describe


def test_describe():
    """Testing the describe method. As it is is difficult to verify that the
    output is correct, at this point we merely test that no error arises when
    applying this method to a variety of operators
    """
    A = MatrixMult(np.ones((10, 5)))
    A.name = "A"
    B = Diagonal(np.ones(5))
    B.name = "A"
    C = MatrixMult(np.ones((10, 5)))
    C.name = "C"

    AT = A.T
    AH = A.H
    A3 = 3 * A
    D = A + C
    E = D * B
    F = (A + C) * B + A
    G = HStack((A * B, C * B))
    H = BlockDiag((F, G))

    describe(A)
    describe(AT)
    describe(AH)
    describe(A3)
    describe(D)
    describe(E)
    describe(F)
    describe(G)
    describe(H)
