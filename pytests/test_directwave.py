import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.signal import convolve

from pylops.waveeqprocessing.marchenko import directwave

# Test data
inputfile2d = "testdata/marchenko/input.npz"
inputfile3d = "testdata/marchenko/direct3D.npz"

# Parameters
vel = 2400.0  # velocity


def test_direct2D():
    """Check consistency of analytical 2D Green's function with FD modelling"""
    inputdata = np.load(inputfile2d)

    # Receivers
    r = inputdata["r"]
    nr = r.shape[1]

    # Virtual points
    vs = inputdata["vs"]

    # Time axis
    t = inputdata["t"]
    dt, nt = t[1] - t[0], len(t)

    # FD GF
    G0FD = inputdata["G0sub"]
    wav = inputdata["wav"]
    wav_c = np.argmax(wav)

    G0FD = np.apply_along_axis(convolve, 0, G0FD, wav, mode="full")
    G0FD = G0FD[wav_c:][:nt]

    # Analytic GF
    trav = np.sqrt((vs[0] - r[0]) ** 2 + (vs[1] - r[1]) ** 2) / vel
    G0ana = directwave(wav, trav, nt, dt, nfft=nt, derivative=False)

    # Differentiate to get same response as in FD modelling
    G0ana = np.diff(G0ana, axis=0)
    G0ana = np.vstack([G0ana, np.zeros(nr)])

    assert_array_almost_equal(
        G0FD / np.max(np.abs(G0FD)), G0ana / np.max(np.abs(G0ana)), decimal=1
    )


def test_direct3D():
    """Check consistency of analytical 3D Green's function with FD modelling"""
    inputdata = np.load(inputfile3d)

    # Receivers
    r = inputdata["r"]
    nr = r.shape[0]

    # Virtual points
    vs = inputdata["vs"]

    # Time axis
    t = inputdata["t"]
    dt, nt = t[1] - t[0], len(t)

    # FD GF
    G0FD = inputdata["G0"][:, :nr]
    wav = inputdata["wav"]
    wav_c = np.argmax(wav)

    G0FD = np.apply_along_axis(convolve, 0, G0FD, wav, mode="full")
    G0FD = G0FD[wav_c:][:nt]

    # Analytic GF
    dist = np.sqrt(
        (vs[0] - r[:, 0]) ** 2 + (vs[1] - r[:, 1]) ** 2 + (vs[2] - r[:, 2]) ** 2
    )
    trav = dist / vel
    G0ana = directwave(
        wav, trav, nt, dt, nfft=nt, dist=dist, kind="3d", derivative=False
    )

    # Differentiate to get same response as in FD modelling
    G0ana = np.diff(G0ana, axis=0)
    G0ana = np.vstack([G0ana, np.zeros(nr)])

    assert_array_almost_equal(
        G0FD / np.max(np.abs(G0FD)), G0ana / np.max(np.abs(G0ana)), decimal=1
    )
