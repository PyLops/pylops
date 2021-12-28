import multiprocessing

import numpy as np
import pytest

from pylops.basicoperators import Restriction
from pylops.utils.seismicevents import linear2d, linear3d, makeaxis
from pylops.utils.wavelets import ricker
from pylops.waveeqprocessing.seismicinterpolation import SeismicInterpolation

np.random.seed(5)

# avoid timeout in travis and azure-pipeline(linux) for numba
# if multiprocessing.cpu_count() >= 4:
#    engine = 'numba'
# else:
engine = "numpy"

# params
par = {
    "oy": 0,
    "dy": 2,
    "ny": 30,
    "ox": 0,
    "dx": 2,
    "nx": 10,
    "ot": 0,
    "dt": 0.004,
    "nt": 40,
    "f0": 25,
}

v = 1500
t0 = [0.05, 0.1, 0.12]
theta = [0, 30, -60]
phi = [0, 50, 30]
amp = [1.0, -2, 0.5]

perc_subsampling = 0.7
nysub = int(np.round(par["ny"] * perc_subsampling))
iava = np.sort(np.random.permutation(np.arange(par["ny"]))[:nysub])

taxis, taxis2, xaxis, yaxis = makeaxis(par)
wav = ricker(taxis[:41], f0=par["f0"])[0]

# 2d model
_, x2d = linear2d(yaxis, taxis, v, t0, theta, amp, wav)
_, x3d = linear3d(xaxis, yaxis, taxis, v, t0, theta, phi, amp, wav)

# Create restriction operator
Rop2d = Restriction(
    par["ny"] * par["nt"], iava, dims=(par["ny"], par["nt"]), dir=0, dtype="float64"
)
y2d = Rop2d * x2d.ravel()
y2d = y2d.reshape(nysub, par["nt"])
Rop3d = Restriction(
    par["ny"] * par["nx"] * par["nt"],
    iava,
    dims=(par["ny"], par["nx"], par["nt"]),
    dir=0,
    dtype="float64",
)
y3d = Rop3d * x3d.ravel()
y3d = y3d.reshape(nysub, par["nx"], par["nt"])


par1_2d = {
    "kind": "spatial",
    "kwargs": dict(epsRs=[np.sqrt(0.1)], damp=np.sqrt(1e-4), iter_lim=20, show=0),
}
par2_2d = {
    "kind": "fk",
    "kwargs": dict(
        nffts=(2 ** 9, 2 ** 9),
        sampling=(par["dy"], par["dt"]),
        niter=20,
        eps=1e-2,
        eigsiter=4,
    ),
}
par3_2d = {
    "kind": "radon-linear",
    "kwargs": dict(
        paxis=np.linspace(-1e-3, 1e-3, 50),
        centeredh=True,
        niter=20,
        eps=1e-1,
        eigsiter=4,
    ),
}
par4_2d = {
    "kind": "sliding",
    "kwargs": dict(
        paxis=np.linspace(-1e-3, 1e-3, 50),
        nwin=12,
        nwins=3,
        nover=3,
        design=False,
        niter=20,
        eps=1e-1,
        eigsiter=4,
    ),
}
par1_2d.update(par)
par2_2d.update(par)
par3_2d.update(par)
par4_2d.update(par)

par1_3d = par1_2d
par2_3d = {
    "kind": "fk",
    "kwargs": dict(
        nffts=(2 ** 7, 2 ** 7, 2 ** 8),
        returninfo=False,
        sampling=(par["dy"], par["dx"], par["dt"]),
        niter=20,
        eps=5e-2,
        alpha=1e0,
        show=False,
    ),
}
par3_3d = {
    "kind": "radon-linear",
    "kwargs": dict(
        paxis=np.linspace(-1e-3, 1e-3, 21),
        p1axis=np.linspace(-1e-3, 1e-3, 50),
        centeredh=True,
        niter=20,
        eps=1e-3,
        alpha=1.3e-6,
        show=False,
    ),
}
par4_3d = {
    "kind": "sliding",
    "kwargs": dict(
        paxis=np.linspace(-1e-3, 1e-3, 21),
        p1axis=np.linspace(-1e-3, 1e-3, 21),
        nwin=(12, 5),
        nwins=(3, 2),
        nover=(3, 2),
        design=True,
        niter=20,
        eps=1e-2,
        alpha=1.3e-4,
        show=False,
    ),
}
par1_3d.update(par)
par2_3d.update(par)
par3_3d.update(par)
par4_3d.update(par)


@pytest.mark.parametrize("par", [(par1_2d), (par2_2d), (par3_2d), (par4_2d)])
def test_SeismicInterpolation2d(par):
    """Dot-test and inversion for SeismicInterpolation in 2d"""
    xinv, _, _ = SeismicInterpolation(
        y2d,
        par["ny"],
        iava,
        kind=par["kind"],
        spataxis=yaxis,
        taxis=taxis,
        engine=engine,
        dottest=True,
        **par["kwargs"]
    )
    assert np.linalg.norm(x2d - xinv) / np.linalg.norm(xinv) < 2e-1


# , (par3_3d), (par4_3d)])
@pytest.mark.parametrize("par", [(par1_3d), (par2_3d)])
def test_SeismicInterpolation3d(par):
    """Dot-test and inversion for SeismicInterpolation in 3d"""
    xinv, _, _ = SeismicInterpolation(
        y3d,
        (par["ny"], par["nx"]),
        iava,
        kind=par["kind"],
        spataxis=yaxis,
        spat1axis=xaxis,
        taxis=taxis,
        engine=engine,
        dottest=True,
        **par["kwargs"]
    )
    # remove edges before checking inversion if using sliding windows
    if par["kind"] == "sliding":
        win0in = par["kwargs"]["nover"][0]
        win0end = (par["kwargs"]["nwin"][0] - par["kwargs"]["nover"][0]) * par[
            "kwargs"
        ]["nwins"][0] - par["kwargs"]["nover"][0]
        win1in = par["kwargs"]["nover"][1]
        win1end = (par["kwargs"]["nwin"][1] - par["kwargs"]["nover"][1]) * par[
            "kwargs"
        ]["nwins"][1] - par["kwargs"]["nover"][1]
        x3dwin = x3d[win0in:win0end, win1in:win1end]
        xinvwin = xinv[win0in:win0end, win1in:win1end]
    else:
        x3dwin = x3d.copy()
        xinvwin = xinv.copy()
    assert np.linalg.norm(x3dwin - xinvwin) / np.linalg.norm(xinvwin) < 3e-1
