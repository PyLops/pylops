"""
Real
====

This example shows how to use the :py:class:`pylops.basicoperators.Real`
operator.
This operator returns the real part of the data in forward and adjoint mode,
but the forward output will be a real number, while the adjoint output will
be a complex number with a zero-valued imaginary part.
"""
import matplotlib.pyplot as plt
import numpy as np

import pylops

plt.close("all")

###############################################################################
# Let's define a Real operator :math:`\mathbf{\Re}` to extract the real
# component of the input.

M = 5
x = np.arange(M) + 1j * np.arange(M)[::-1]
Rop = pylops.basicoperators.Real(M, dtype="complex128")

y = Rop * x
xadj = Rop.H * y

_, axs = plt.subplots(1, 3, figsize=(10, 4))
axs[0].plot(np.real(x), lw=2, label="Real")
axs[0].plot(np.imag(x), lw=2, label="Imag")
axs[0].legend()
axs[0].set_title("Input")
axs[1].plot(np.real(y), lw=2, label="Real")
axs[1].plot(np.imag(y), lw=2, label="Imag")
axs[1].legend()
axs[1].set_title("Forward of Input")
axs[2].plot(np.real(xadj), lw=2, label="Real")
axs[2].plot(np.imag(xadj), lw=2, label="Imag")
axs[2].legend()
axs[2].set_title("Adjoint of Forward")
