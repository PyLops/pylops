"""
Uniform Discrete Curvelet Transform
===================================
This example shows how to use the :py:class:`pylops.signalprocessing.UDCT` operator to perform the
Uniform Discrete Curvelet Transform on a (multi-dimensional) input array.
"""


from ucurv import *
import matplotlib.pyplot as plt
import pylops
plt.close("all")

if False:
    sz = [512, 512]
    cfg = [[3, 3], [6,6]]
    res = len(cfg)
    rsq = zoneplate(sz)
    img = rsq - np.mean(rsq)

    transform = udct(sz, cfg, complex = False, high = "curvelet")

    imband = ucurvfwd(img, transform)
    plt.figure(figsize = (20, 60))
    print(imband.keys())
    plt.imshow(np.abs(ucurv2d_show(imband, transform))) 
    # plt.show()

    recon = ucurvinv(imband, transform)

    err = img - recon
    print(np.max(np.abs(err)))
    plt.figure(figsize = (20, 60))
    plt.imshow(np.real(np.concatenate((img, recon, err), axis = 1)))

    plt.figure()
    plt.imshow(np.abs(np.fft.fftshift(np.fft.fftn(err))))   
    # plt.show()

################################################################################


sz = [256, 256]
cfg = [[3,3],[6,6]]
x = np.random.rand(256*256)
y = np.random.rand(262144)
F = pylops.signalprocessing.UDCT(sz,cfg)
print(np.dot(y,F*x))
print(np.dot(x,F.T*y))