import pytest

import numpy as np
from numpy.testing import assert_array_equal

from pylops.utils.wavelets import ricker
from pylops.utils.seismicevents import makeaxis, linear2d, linear3d
from pylops.utils.seismicevents import parabolic2d, hyperbolic2d, hyperbolic3d

# Wavelet
wav = ricker(np.arange(41)*0.004, f0=10)[0]

par1 = {'ot': 0, 'dt': 1, 'nt': 300,
        'ox': 0, 'dx': 2, 'nx': 200,
        'oy': 0, 'dy': 2, 'ny': 100} # even axis

par2 = {'ot': 0, 'dt': 1, 'nt': 301,
        'ox': -200, 'dx': 2, 'nx': 201,
        'oy': -100, 'dy': 2, 'ny': 101} # odd axis, centered to 0


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_makeaxis(par):
    """Verify makeaxis creation
    """
    # Create t, x, and y axis
    t, _, x, y = makeaxis(par)

    # Check axis lenght
    assert len(t) == par['nt']
    assert len(x) == par['nx']
    assert len(y) == par['ny']

    # Check axis initial and end values
    assert t[0] == par['ot']
    assert t[-1] == par['ot'] + par['dt'] * (par['nt'] - 1)
    assert x[0] == par['ox']
    assert x[-1] == par['ox'] + par['dx'] * (par['nx'] - 1)
    assert y[0] == par['oy']
    assert y[-1] == par['oy'] + par['dy'] * (par['ny'] - 1)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_linear2d(par):
    """Create small dataset with an horizontal event and check that output
    contains the event at correct time and correct amplitude
    """
    # Data creation
    v = 1
    t0 = 50
    theta = 0.
    amp = 0.6

    # Create axes
    t, _, x, _ = makeaxis(par)

    # Create data
    d, dwav = linear2d(x, t, v, t0, theta, amp, wav)

    # Assert shape
    assert d.shape[0] == par['nx']
    assert d.shape[1] == par['nt']

    assert dwav.shape[0] == par['nx']
    assert dwav.shape[1] == par['nt']

    # Assert correct position of event
    assert_array_equal(d[:, t0], amp*np.ones(par['nx']))


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_multilinear2d(par):
    """Create small dataset with several horizontal events and check that output
    contains the events at correct time and correct amplitude
    """
    # Data creation
    v = 1
    t0 = (50, 130)
    theta = (0., 0.)
    amp = (0.6, 1)

    # Create axes
    t, _, x, _ = makeaxis(par)

    # Create data
    d, dwav = linear2d(x, t, v, t0, theta, amp, wav)

    # Assert shape
    assert d.shape[0] == par['nx']
    assert d.shape[1] == par['nt']

    assert dwav.shape[0] == par['nx']
    assert dwav.shape[1] == par['nt']

    # Assert correct position of event
    assert_array_equal(d[:, t0[0]],
                       amp[0]*np.ones(par['nx']))
    assert_array_equal(d[:, t0[1]],
                       amp[1] * np.ones(par['nx']))


@pytest.mark.parametrize("par", [(par2)])
def test_parabolic2d(par):
    """Create small dataset with a parabolic event and check that output
    contains the event apex at correct time and correct amplitude
    """
    # Data creation
    t0 = 50
    px = 0
    pxx = 1e-1
    amp = 0.6

    # Create axes
    t, _, x, _ = makeaxis(par)

    # Create data
    d, dwav = parabolic2d(x, t, t0, px, pxx, amp, np.ones(1))

    # Assert shape
    assert d.shape[0] == par['nx']
    assert d.shape[1] == par['nt']

    assert dwav.shape[0] == par['nx']
    assert dwav.shape[1] == par['nt']

    # Assert correct position of event
    assert_array_equal(d[par['nx']//2, t0], amp)


@pytest.mark.parametrize("par", [(par2)])
def test_hyperbolic2d(par):
    """Create small dataset with a hyperbolic event and check that output
    contains the event apex at correct time and correct amplitude
    """
    # Data creation
    t0 = 50
    vrms = 1
    amp = 0.6

    # Create axes
    t, _, x, _ = makeaxis(par)

    # Create data
    d, dwav = hyperbolic2d(x, t, t0, vrms, amp, wav)

    # Assert shape
    assert d.shape[0] == par['nx']
    assert d.shape[1] == par['nt']

    assert dwav.shape[0] == par['nx']
    assert dwav.shape[1] == par['nt']

    # Assert correct position of event
    assert_array_equal(d[par['nx']//2, t0], amp)
    assert_array_equal(dwav[par['nx']//2, t0], amp)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_linear3d(par):
    """Create small dataset with an horizontal event and check output
    contains event at correct time and correct amplitude
    """
    # Data creation
    v = 1
    t0 = 50
    theta = 0.
    phi = 0.
    amp = 0.6

    # Create axes
    t, _, x, y = makeaxis(par)

    # Create data
    d, dwav = linear3d(x, y, t, v, t0, theta, phi, amp, wav)

    #Assert shape
    assert d.shape[0] == par['ny']
    assert d.shape[1] == par['nx']
    assert d.shape[2] == par['nt']

    assert dwav.shape[0] == par['ny']
    assert dwav.shape[1] == par['nx']
    assert dwav.shape[2] == par['nt']

    # Assert correct position of event
    assert_array_equal(d[:, :, t0],
                       amp*np.ones((par['ny'], par['nx'])))
    assert_array_equal(dwav[:, :, t0],
                       amp * np.ones((par['ny'], par['nx'])))


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_multilinear3d(par):
    """Create small dataset with several linear events and check output
    contains the events at correct time and correct amplitude
    """
    # Data creation
    v = 1
    t0 = (50, 130)
    theta = (0., 0.)
    phi = (0., 0.)
    amp = (0.6, 1)

    # Create axes
    t, _, x, y = makeaxis(par)

    # Create data
    d, dwav = linear3d(x, y, t, v, t0, theta, phi, amp, wav)

    #Assert shape
    assert d.shape[0] == par['ny']
    assert d.shape[1] == par['nx']
    assert d.shape[2] == par['nt']

    assert dwav.shape[0] == par['ny']
    assert dwav.shape[1] == par['nx']
    assert dwav.shape[2] == par['nt']

    # Assert correct position of event
    assert_array_equal(d[:, :, t0[0]],
                       amp[0]*np.ones((par['ny'], par['nx'])))
    assert_array_equal(d[:, :, t0[1]],
                       amp[1]*np.ones((par['ny'], par['nx'])))


@pytest.mark.parametrize("par", [(par2)])
def test_hyperbolic3d(par):
    """Create small dataset with several hyperbolic events and check output
    contains the events at correct time and correct amplitude
    """
    # Data creation
    t0 = 50
    vrms_x = 1.
    vrms_y = 1.
    amp = 0.6

    # Create axes
    t, _, x, y = makeaxis(par)

    # Create data
    d, dwav = hyperbolic3d(x, y, t, t0, vrms_x, vrms_y, amp, wav)

    #Assert shape
    assert d.shape[0] == par['ny']
    assert d.shape[1] == par['nx']
    assert d.shape[2] == par['nt']

    assert dwav.shape[0] == par['ny']
    assert dwav.shape[1] == par['nx']
    assert dwav.shape[2] == par['nt']

    # Assert correct position of event
    assert_array_equal(d[par['ny']//2, par['nx']//2, t0], amp)
