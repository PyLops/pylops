r"""
2D Radon Transform
==================
This example shows how to use the :py:class:`pylops.signalprocessing.Radon2D`
operator to apply the Radon Transform to a signal. In our implementation both
linear, parabolic and hyperbolic parametrization can be chosen.
"""
import numpy as np
import matplotlib.pyplot as plt

import pylops

plt.close('all')

###############################################################################
# Let's start by creating an empty 2d matrix of size :math:`n_{p_x} \times n_t`
# and add a single spike in it. We will see that applying the forward
# Radon operator will result in a single event (linear, parabolic or
# hyperbolic) in the resulting data vector.
nt, nh = 41, 51
npx, pxmax = 41, 1e-2

dt, dh = 0.005, 1
t = np.arange(nt) * dt
h = np.arange(nh) * dh
px = np.linspace(0, pxmax, npx)

x = np.zeros((npx, nt))
x[4, nt//2] = 1

###############################################################################
# We can now define our operators for different parametric curves and apply
# them to the input model vector. We also apply the adjoint to the resulting
# data vector.
RLop = pylops.signalprocessing.Radon2D(t, h, px, centeredh=True,
                                       kind='linear', interp='nearest')
RPop = pylops.signalprocessing.Radon2D(t, h, px, centeredh=True,
                                       kind='parabolic', interp='nearest')
RHop = pylops.signalprocessing.Radon2D(t, h, px, centeredh=True,
                                       kind='hyperbolic', interp='nearest')

# forward
yL = RLop * x.flatten()
yP = RPop * x.flatten()
yH = RHop * x.flatten()
yL = yL.reshape(nh, nt)
yP = yP.reshape(nh, nt)
yH = yH.reshape(nh, nt)

# adjoint
xadjL = RLop.H * yL.flatten()
xadjP = RPop.H * yP.flatten()
xadjH = RHop.H * yH.flatten()
xadjL = xadjL.reshape(npx, nt)
xadjP = xadjP.reshape(npx, nt)
xadjH = xadjH.reshape(npx, nt)

###############################################################################
# Let's finally visualize the input model in the Radon domain, the data, and
# the adjoint model the different parametric curves.

fig, axs = plt.subplots(2, 4, figsize=(10, 6))
axs[0][0].imshow(x.T, vmin=-1, vmax=1, cmap='seismic_r',
                 extent=(px[0], px[-1], t[-1], t[0]))
axs[0][0].set_title('Input model')
axs[0][0].axis('tight')
axs[0][1].imshow(yL.T, vmin=-1, vmax=1, cmap='seismic_r',
                 extent=(h[0], h[-1], t[-1], t[0]))
axs[0][1].set_title('Linear data')
axs[0][1].axis('tight')
axs[0][2].imshow(yP.T, vmin=-1, vmax=1, cmap='seismic_r',
                 extent=(h[0], h[-1], t[-1], t[0]))
axs[0][2].set_title('Parabolic data')
axs[0][2].axis('tight')
axs[0][3].imshow(yH.T, vmin=-1, vmax=1, cmap='seismic_r',
                 extent=(h[0], h[-1], t[-1], t[0]))
axs[0][3].set_title('Hyperbolic data')
axs[0][3].axis('tight')
axs[1][1].imshow(xadjL.T, vmin=-20, vmax=20, cmap='seismic_r',
                 extent=(px[0], px[-1], t[-1], t[0]))
axs[1][0].axis('off')
axs[1][1].set_title('Linear adjoint')
axs[1][1].axis('tight')
axs[1][2].imshow(xadjP.T, vmin=-20, vmax=20, cmap='seismic_r',
                 extent=(px[0], px[-1], t[-1], t[0]))
axs[1][2].set_title('Parabolic adjoint')
axs[1][2].axis('tight')
axs[1][3].imshow(xadjH.T, vmin=-20, vmax=20, cmap='seismic_r',
                 extent=(px[0], px[-1], t[-1], t[0]))
axs[1][3].set_title('Hyperbolic adjoint')
axs[1][3].axis('tight')
fig.tight_layout()

###############################################################################
# As we can see in the bottom figures, the adjoint Radon Transform is far
# from being close to the inverse Radon Transform, i.e.
# :math:`\mathbf{R^H}\mathbf{R} \neq \mathbf{I}` (compared to the case of FFT
# where the adjoint and inverse are equivalent, i.e.
# :math:`\mathbf{F^H}\mathbf{F} = \mathbf{I}`). In fact when we apply the
# adjoint Radon Transform we obtain a *model* that
# is a smoothed version of the original model polluted by smearing and
# artifacts. In tutorial :ref:`sphx_glr_tutorials_radonfiltering.py` we will
# exploit a sparsity-promiting Radon transform to perform filtering of unwanted
# signals from an input data.
