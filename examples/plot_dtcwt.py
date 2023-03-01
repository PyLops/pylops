"""
Dual-Tree Complex Wavelet Transform
=========================
This example shows how to use the :py:class:`pylops.signalprocessing.DCT` operator.
This operator performs the 1D Dual-Tree Complex Wavelet Transform on a (single or multi-dimensional)
input array.
"""

import matplotlib.pyplot as plt
import numpy as np

import pylops

plt.close("all")

###############################################################################
# Let's define a 1D array x of having random values

n = 50
x = np.random.rand(n,)

###############################################################################
# We create the DTCWT operator with shape of our input array. DTCWT transform
# gives a Pyramid object that is flattened out `y`.

DOp = pylops.signalprocessing.DTCWT(dims=x.shape)
y = DOp @ x
xadj = DOp.H @ y

plt.figure(figsize=(8, 5))
plt.plot(x, "k", label="input array")
plt.plot(y, "r", label="transformed array")
plt.plot(xadj, "--b", label="transformed array")
plt.title("Dual-Tree Complex Wavelet Transform 1D")
plt.legend()
plt.tight_layout()

#################################################################################
# To get the Pyramid object use the `get_pyramid` method.
# We can get the Highpass signal and Lowpass signal from it

pyr = DOp.get_pyramid(y)

plt.figure(figsize=(10, 5))
plt.plot(x, "--b", label="orignal signal")
plt.plot(pyr.lowpass, "k", label="lowpass")
plt.plot(pyr.highpasses[0], "r", label="highpass level 1 signal")
plt.plot(pyr.highpasses[1], "b", label="highpass level 2 signal")
plt.plot(pyr.highpasses[2], "g", label="highpass level 3 signal")

plt.title("DTCWT Pyramid Object")
plt.legend()
plt.tight_layout()

###################################################################################
# DTCWT can also be performed on multi-dimension arrays. The number of levels can also
# be defined using the `nlevels`

n = 10
m = 2

x = np.random.rand(n, m)

DOp = pylops.signalprocessing.DTCWT(dims=x.shape, nlevels=5)
y = DOp @ x
xadj = DOp.H @ y

plt.figure(figsize=(8, 5))
plt.plot(x, "k", label="input array")
plt.plot(y, "r", label="transformed array")
plt.plot(xadj, "--b", label="transformed array")
plt.title("Dual-Tree Complex Wavelet Transform 1D on ND array")
plt.legend()
plt.tight_layout()
