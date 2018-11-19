"""
Tapers
======
This example shows how to create some basic tapers in 1d, 2d, and 3d
using the :py:mod:`lops.utils.tapers` module.
"""
import matplotlib.pyplot as plt

import lops

############################################
# Let's first define the time and space axes
par = {'ox':-200, 'dx':2, 'nx':201,
       'oy':-100, 'dy':2, 'ny':101,
       'ot':0, 'dt':0.004, 'nt':501,
       'ntapx': 21, 'ntapy': 31}

############################################
# We can now create tapers in 1d
tap_han = lops.utils.tapers.hanningtaper(par['nx'],
                                         par['ntapx'])
tap_cos = lops.utils.tapers.cosinetaper(par['nx'], False)
tap_cos2 = lops.utils.tapers.cosinetaper(par['nx'], True)

plt.figure()
plt.plot(tap_han, 'r', label='hanning')
plt.plot(tap_cos, 'k', label='cosine')
plt.plot(tap_cos2, 'b', label='cosine square')
plt.title('Tapers')
plt.legend()

############################################
# Similarly we can create 2d and 3d tapers with any of the tapers above
tap2d = lops.utils.tapers.taper2d(par['nt'], par['nx'],
                                  par['ntapx'], plotflag='True')

tap3d = lops.utils.tapers.taper3d(par['nt'], (par['ny'], par['nx']),
                                  (par['ntapy'], par['ntapx']),
                                  plotflag='True')
