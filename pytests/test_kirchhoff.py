import os

import numpy as np

if int(os.environ.get("TEST_CUPY_PYLOPS", 0)):
    import cupy as np
    from cupy.testing import assert_array_almost_equal

    backend = "cupy"
else:
    import numpy as np
    from numpy.testing import assert_array_almost_equal

    backend = "numpy"
import numpy as npp
import pytest

from pylops.utils import dottest
from pylops.utils.wavelets import ricker
from pylops.waveeqprocessing.kirchhoff import Kirchhoff

np.random.seed(10)

PAR = {
    "ny": 3,
    "nx": 12,
    "nz": 20,
    "nt": 50,
    "dy": 3,
    "dx": 1,
    "dz": 2,
    "dt": 0.004,
    "nsy": 4,
    "nry": 3,
    "nsx": 6,
    "nrx": 2,
}

# Check if skfmm is available and by-pass tests using it otherwise. This is
# currently used in GPU CI not to run tests that require skfmm to compute the
# traveltime tables - should consider fixing it
try:
    import skfmm  # noqa: F401

    skfmm_enabled = True if backend == "numpy" else False
except ImportError:
    skfmm_enabled = False

v0 = 500
y = npp.arange(PAR["ny"]) * PAR["dy"]
x = npp.arange(PAR["nx"]) * PAR["dx"]
z = npp.arange(PAR["nz"]) * PAR["dz"]
t = npp.arange(PAR["nt"]) * PAR["dt"]

sy = npp.linspace(y.min(), y.max(), PAR["nsy"])
sx = npp.linspace(x.min(), x.max(), PAR["nsx"])
syy, sxx = npp.meshgrid(sy, sx, indexing="ij")
s2d = npp.vstack((sx, 2 * npp.ones(PAR["nsx"])))
s3d = npp.vstack((syy.ravel(), sxx.ravel(), 2 * npp.ones(PAR["nsx"] * PAR["nsy"])))

ry = npp.linspace(y.min(), y.max(), PAR["nry"])
rx = npp.linspace(x.min(), x.max(), PAR["nrx"])
ryy, rxx = npp.meshgrid(ry, rx, indexing="ij")
r2d = npp.vstack((rx, 2 * npp.ones(PAR["nrx"])))
r3d = npp.vstack((ryy.ravel(), rxx.ravel(), 2 * npp.ones(PAR["nrx"] * PAR["nry"])))

wav, _, wavc = ricker(t[:21], f0=40)

par1 = {"mode": "analytic", "dynamic": False}
par2 = {"mode": "eikonal", "dynamic": False}
par3 = {"mode": "byot", "dynamic": False}
par1d = {"mode": "analytic", "dynamic": True}
par2d = {"mode": "eikonal", "dynamic": True}
par3d = {"mode": "byot", "dynamic": True}

# paired=True: one receiver per source, at the source (offset=0) or shifted along x
# (common offset); "single" selects a single byot traveltime table (one column per pair)
sxp = npp.linspace(x.min(), x.max() - 2 * PAR["dx"], PAR["nsx"])
syyp, sxxp = npp.meshgrid(sy, sxp, indexing="ij")
s2dp = npp.vstack((sxp, 2 * npp.ones(PAR["nsx"])))
s3dp = npp.vstack((syyp.ravel(), sxxp.ravel(), 2 * npp.ones(PAR["nsx"] * PAR["nsy"])))

par1p = {
    "mode": "analytic",
    "dynamic": False,
    "single": False,
    "offset": 0,
    "aperture": None,
    "angleaperture": 90,
    "wavfilter": False,
}
par2p = {
    "mode": "eikonal",
    "dynamic": False,
    "single": False,
    "offset": 2,
    "aperture": 2,
    "angleaperture": 30,
    "wavfilter": True,
}
par3p = {
    "mode": "byot",
    "dynamic": False,
    "single": False,
    "offset": 0,
    "aperture": None,
    "angleaperture": 90,
    "wavfilter": False,
}
par4p = {
    "mode": "byot",
    "dynamic": False,
    "single": True,
    "offset": 2,
    "aperture": None,
    "angleaperture": 90,
    "wavfilter": True,
}
par1pd = {
    "mode": "analytic",
    "dynamic": True,
    "single": False,
    "offset": 2,
    "aperture": 2,
    "angleaperture": 30,
    "wavfilter": True,
}
par2pd = {
    "mode": "eikonal",
    "dynamic": True,
    "single": False,
    "offset": 0,
    "aperture": None,
    "angleaperture": 90,
    "wavfilter": False,
}
par3pd = {
    "mode": "byot",
    "dynamic": True,
    "single": False,
    "offset": 2,
    "aperture": 2,
    "angleaperture": 30,
    "wavfilter": False,
}


def _kirchhoff_paired(srcs, recs, par, dtype, paired, engine="numpy"):
    """Kirchhoff operator for the paired=True tests (all pairs if paired=False)"""
    ndims = srcs.shape[0]
    yy = None if ndims == 2 else y
    shape = (PAR["nx"], PAR["nz"]) if ndims == 2 else (PAR["ny"], PAR["nx"], PAR["nz"])
    trav, amp = None, None
    if par["mode"] == "byot":
        # tables of an analytic operator with the same geometry
        Aop = Kirchhoff(
            z,
            x,
            t,
            srcs,
            recs,
            v0,
            wav,
            wavc,
            y=yy,
            dynamic=par["dynamic"],
            mode="analytic",
        )
        trav = (Aop.trav_srcs.astype(dtype), Aop.trav_recs.astype(dtype))
        if par["dynamic"]:
            amp = (Aop.amp_srcs.astype(dtype), Aop.amp_recs.astype(dtype))
        if par["single"] and paired:
            trav = trav[0] + trav[1]
        elif par["single"]:
            trav = (trav[0][:, :, npp.newaxis] + trav[1][:, npp.newaxis, :]).reshape(
                trav[0].shape[0], -1
            )
    return Kirchhoff(
        z.astype(dtype),
        x.astype(dtype),
        t.astype(dtype),
        srcs.astype(dtype),
        recs.astype(dtype),
        v0 * npp.ones(shape, dtype=dtype) if par["mode"] == "eikonal" else v0,
        wav.astype(dtype),
        wavc,
        y=None if yy is None else yy.astype(dtype),
        mode=par["mode"],
        wavfilter=par["wavfilter"],
        dynamic=par["dynamic"],
        trav=trav,
        amp=amp,
        aperture=par["aperture"],
        angleaperture=par["angleaperture"],
        engine=engine,
        dtype=dtype,
        paired=paired,
    )


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
def test_kirchhoff_paired_nsnr():
    """Check error is raised if paired=True and ns differs from nr"""
    with pytest.raises(ValueError, match="paired=True requires as many sources"):
        _ = Kirchhoff(z, x, t, s2d, r2d, v0, wav, wavc, mode="analytic", paired=True)


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
def test_kirchhoff_paired_single_trav():
    """Check error is raised if paired=True and a single traveltime table has
    one column per source-receiver combination instead of one per pair"""
    trav = npp.zeros((PAR["nx"] * PAR["nz"], PAR["nsx"] * PAR["nsx"]))
    with pytest.raises(ValueError, match="needs one column per pair"):
        _ = Kirchhoff(
            z, x, t, s2dp, s2dp, v0, wav, wavc, mode="byot", trav=trav, paired=True
        )


def test_kirchhoff_paired_cuda():
    """Check error is raised if paired=True and engine='cuda'"""
    with pytest.raises(NotImplementedError, match="not implemented for paired=True"):
        _ = Kirchhoff(
            z,
            x,
            t,
            s2dp,
            s2dp,
            v0,
            wav,
            wavc,
            mode="analytic",
            engine="cuda",
            paired=True,
        )


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
def test_identify_geometry():
    """Identify geometry, check expected outputs"""
    # 2d
    (
        ndims,
        shiftdim,
        dims,
        ny,
        nx,
        nz,
        ns,
        nr,
        dy,
        dx,
        dz,
        dsamp,
        origin,
    ) = Kirchhoff._identify_geometry(z, x, s2d, r2d)
    assert ndims == 2
    assert shiftdim == 0
    assert [1, 2] == [1, 2]
    assert list(dims) == [PAR["nx"], PAR["nz"]]
    assert ny == 1
    assert nx == PAR["nx"]
    assert nz == PAR["nz"]
    assert ns == PAR["nsx"]
    assert nr == PAR["nrx"]
    assert list(dsamp) == [dx, dz]
    assert list(origin) == [0, 0]

    # 3d
    (
        ndims,
        shiftdim,
        dims,
        ny,
        nx,
        nz,
        ns,
        nr,
        dy,
        dx,
        dz,
        dsamp,
        origin,
    ) = Kirchhoff._identify_geometry(z, x, s3d, r3d, y=y)
    assert ndims == 3
    assert shiftdim == 1
    assert list(dims) == [PAR["ny"], PAR["nx"], PAR["nz"]]
    assert ny == PAR["ny"]
    assert nx == PAR["nx"]
    assert nz == PAR["nz"]
    assert ns == PAR["nsy"] * PAR["nsx"]
    assert nr == PAR["nry"] * PAR["nrx"]
    assert list(dsamp) == [dy, dx, dz]
    assert list(origin) == [0, 0, 0]


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
def test_traveltime_ana():
    """Check analytical traveltimes in homogenous medium for horizontal and
    vertical paths
    """
    src = np.array([100, 0])[:, np.newaxis]

    (
        trav_srcs_ana,
        trav_recs_ana,
        dist_srcs_ana,
        dist_recs_ana,
        _,
        _,
    ) = Kirchhoff._traveltime_table(
        np.arange(0, 200, 1), np.arange(0, 200, 1), src, src, v0, mode="analytic"
    )
    assert dist_srcs_ana[0, 0] + dist_recs_ana[0, 0] == 200
    assert trav_srcs_ana[0, 0] == 100 / v0
    assert trav_recs_ana[0, 0] == 100 / v0


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
def test_traveltime_table():
    """Compare analytical and eikonal traveltimes in homogenous medium"""
    if skfmm_enabled:
        # 2d
        (
            trav_ana,
            trav_srcs_ana,
            trav_recs_ana,
            _,
            _,
            _,
        ) = Kirchhoff._traveltime_table(z, x, s2d, r2d, v0, mode="analytic")

        (
            trav_eik,
            trav_srcs_eik,
            trav_recs_eik,
            _,
            _,
            _,
        ) = Kirchhoff._traveltime_table(
            z, x, s2d, r2d, v0 * np.ones((PAR["nx"], PAR["nz"])), mode="eikonal"
        )

        assert_array_almost_equal(trav_srcs_ana, trav_srcs_eik, decimal=2)
        assert_array_almost_equal(trav_recs_ana, trav_recs_ana, decimal=2)
        assert_array_almost_equal(trav_ana, trav_eik, decimal=2)

        # 3d
        (
            trav_srcs_ana,
            trav_recs_ana,
            _,
            _,
            _,
            _,
        ) = Kirchhoff._traveltime_table(z, x, s3d, r3d, v0, y=y, mode="analytic")

        (
            trav_srcs_eik,
            trav_recs_eik,
            _,
            _,
            _,
            _,
        ) = Kirchhoff._traveltime_table(
            z,
            x,
            s3d,
            r3d,
            v0 * np.ones((PAR["ny"], PAR["nx"], PAR["nz"])),
            y=y,
            mode="eikonal",
        )

        assert_array_almost_equal(trav_srcs_ana, trav_srcs_eik, decimal=2)
        assert_array_almost_equal(trav_recs_ana, trav_recs_eik, decimal=2)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par1d), (par2d), (par3d)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_kirchhoff2d(par, dtype):
    """Dot-test for Kirchhoff operator"""
    if backend == "cupy" and par["mode"] == "byot":
        pytest.skip("cuda engine not available for single trav table")
    vel = v0 * np.ones((PAR["nx"], PAR["nz"]), dtype=dtype)

    if par["mode"] == "byot":
        trav_srcs, trav_recs, _, _, _, _ = Kirchhoff._traveltime_table(
            z, x, s2d, r2d, v0, mode="analytic"
        )
        trav_srcs = trav_srcs.astype(dtype)
        trav_recs = trav_recs.astype(dtype)
        trav = trav_srcs.reshape(
            PAR["nx"] * PAR["nz"], PAR["nsx"], 1
        ) + trav_recs.reshape(PAR["nx"] * PAR["nz"], 1, PAR["nrx"])
        trav = trav.reshape(PAR["nx"] * PAR["nz"], PAR["nsx"] * PAR["nrx"])
        amp = None
    else:
        trav = None
        amp = None

    if skfmm_enabled or par["mode"] != "eikonal":
        Dop = Kirchhoff(
            z.astype(dtype),
            x.astype(dtype),
            t.astype(dtype),
            s2d.astype(dtype),
            r2d.astype(dtype),
            vel if par["mode"] == "eikonal" else v0,
            np.asarray(wav.astype(dtype)),
            wavc,
            y=None,
            trav=trav,
            amp=amp,
            mode=par["mode"],
            engine="numpy" if backend == "numpy" else "cuda",
            dtype=dtype,
        )
        if par["mode"] == "byot":
            Dop.trav = np.asarray(Dop.trav)
        else:
            Dop.trav_srcs = np.asarray(Dop.trav_srcs)
            Dop.trav_recs = np.asarray(Dop.trav_recs)
        if par["mode"] == "dynamic":
            Dop.amp_srcs = np.asarray(Dop.amp_srcs)
            Dop.amp_recs = np.asarray(Dop.amp_recs)

        assert dottest(
            Dop,
            PAR["nsx"] * PAR["nrx"] * PAR["nt"],
            PAR["nz"] * PAR["nx"],
            backend=backend,
            rtol=1e-4 if dtype == np.float32 else 1e-6,
        )

        xx = np.random.normal(0, 1, PAR["nx"] * PAR["nz"]).astype(dtype)
        yy = Dop * xx
        xadj = Dop.H * yy
        assert yy.dtype == dtype
        assert xadj.dtype == dtype


@pytest.mark.parametrize("par", [(par1), (par2), (par3)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_kirchhoff3d(par, dtype):
    """Dot-test for Kirchhoff operator"""
    if backend == "cupy" and par["mode"] == "byot":
        pytest.skip("cuda engine not available for single trav table")
    vel = v0 * np.ones((PAR["ny"], PAR["nx"], PAR["nz"]), dtype=dtype)

    if par["mode"] == "byot":
        trav_srcs, trav_recs, _, _, _, _ = Kirchhoff._traveltime_table(
            z, x, s3d, r3d, v0, y=y, mode="analytic"
        )
        trav_srcs = trav_srcs.astype(dtype)
        trav_recs = trav_recs.astype(dtype)
        trav = trav_srcs.reshape(
            PAR["ny"] * PAR["nx"] * PAR["nz"], PAR["nsy"] * PAR["nsx"], 1
        ) + trav_recs.reshape(
            PAR["ny"] * PAR["nx"] * PAR["nz"], 1, PAR["nry"] * PAR["nrx"]
        )
        trav = trav.reshape(
            PAR["ny"] * PAR["nx"] * PAR["nz"],
            PAR["nsy"] * PAR["nry"] * PAR["nsx"] * PAR["nrx"],
        )
    else:
        trav = None

    if skfmm_enabled or par["mode"] != "eikonal":
        Dop = Kirchhoff(
            z.astype(dtype),
            x.astype(dtype),
            t.astype(dtype),
            s3d.astype(dtype),
            r3d.astype(dtype),
            vel if par["mode"] == "eikonal" else v0,
            np.asarray(wav.astype(dtype)),
            wavc,
            y=y.astype(dtype),
            trav=trav,
            mode=par["mode"],
            engine="numpy" if backend == "numpy" else "cuda",
            dtype=dtype,
        )
        if par["mode"] == "byot":
            Dop.trav = np.asarray(Dop.trav)
        else:
            Dop.trav_srcs = np.asarray(Dop.trav_srcs)
            Dop.trav_recs = np.asarray(Dop.trav_recs)

        assert dottest(
            Dop,
            PAR["nsx"] * PAR["nrx"] * PAR["nsy"] * PAR["nry"] * PAR["nt"],
            PAR["nz"] * PAR["nx"] * PAR["ny"],
            backend=backend,
            rtol=1e-4 if dtype == np.float32 else 1e-6,
        )

        xx = np.random.normal(0, 1, PAR["ny"] * PAR["nx"] * PAR["nz"]).astype(dtype)
        yy = Dop * xx
        xadj = Dop.H * yy
        assert yy.dtype == dtype
        assert xadj.dtype == dtype


@pytest.mark.parametrize(
    "par",
    [
        (par1),
        (par1d),
    ],
)
def test_kirchhoff2d_trav_vs_travsrcrec(par):
    """Compare 2D Kirchhoff operator forward and adjoint when using trav (original behavior)
    or trav_src and trav_rec (new recommended behaviour)"""

    # new behaviour
    Dop = Kirchhoff(
        z,
        x,
        t,
        s2d,
        r2d,
        v0,
        wav,
        wavc,
        y=None,
        mode=par["mode"],
        dynamic=par["dynamic"],
        angleaperture=None,
        engine="numpy" if backend == "numpy" else "cuda",
    )
    Dop.trav_srcs = np.asarray(Dop.trav_srcs)
    Dop.trav_recs = np.asarray(Dop.trav_recs)
    if par["dynamic"]:
        Dop.amp_srcs = np.asarray(Dop.amp_srcs)
        Dop.amp_recs = np.asarray(Dop.amp_recs)

    # old behaviour
    trav = Dop.trav_srcs.reshape(
        PAR["nx"] * PAR["nz"], PAR["nsx"], 1
    ) + Dop.trav_recs.reshape(PAR["nx"] * PAR["nz"], 1, PAR["nrx"])
    trav = trav.reshape(PAR["nx"] * PAR["nz"], PAR["nsx"] * PAR["nrx"])
    if par["dynamic"]:
        amp = (Dop.amp_srcs, Dop.amp_recs)

    D1op = Kirchhoff(
        z,
        x,
        t,
        s2d,
        r2d,
        v0,
        wav,
        wavc,
        y=None,
        trav=trav,
        amp=amp if par["dynamic"] else None,
        mode=par["mode"],
        dynamic=par["dynamic"],
        angleaperture=None,
        engine="numpy" if backend == "numpy" else "cuda",
    )
    D1op.trav_srcs = np.asarray(D1op.trav_srcs)
    D1op.trav_recs = np.asarray(D1op.trav_recs)
    if par["dynamic"]:
        D1op.amp_srcs = np.asarray(D1op.amp_srcs)
        D1op.amp_recs = np.asarray(D1op.amp_recs)

    # forward
    xx = np.random.normal(0, 1, PAR["nx"] * PAR["nz"])
    assert_array_almost_equal(Dop @ xx, D1op @ xx, decimal=2)

    # adjoint
    yy = np.random.normal(0, 1, PAR["nrx"] * PAR["nsx"] * PAR["nt"])
    assert_array_almost_equal(Dop.H @ yy, D1op.H @ yy, decimal=2)


@pytest.mark.parametrize(
    "par",
    [
        (par1),
    ],
)
def test_kirchhoff3d_trav_vs_travsrcrec(par):
    """Compare 3D Kirchhoff operator forward and adjoint when using trav (original behavior)
    or trav_src and trav_rec (new recommended behaviour)"""

    # new behaviour
    Dop = Kirchhoff(
        z,
        x,
        t,
        s3d,
        r3d,
        v0,
        wav,
        wavc,
        y=y,
        mode=par["mode"],
        engine="numpy" if backend == "numpy" else "cuda",
    )
    Dop.trav_srcs = np.asarray(Dop.trav_srcs)
    Dop.trav_recs = np.asarray(Dop.trav_recs)
    if par["dynamic"]:
        Dop.amp_srcs = np.asarray(Dop.amp_srcs)
        Dop.amp_recs = np.asarray(Dop.amp_recs)

    # old behaviour
    trav = Dop.trav_srcs.reshape(
        PAR["ny"] * PAR["nx"] * PAR["nz"], PAR["nsy"] * PAR["nsx"], 1
    ) + Dop.trav_recs.reshape(
        PAR["ny"] * PAR["nx"] * PAR["nz"], 1, PAR["nry"] * PAR["nrx"]
    )
    trav = trav.reshape(
        PAR["ny"] * PAR["nx"] * PAR["nz"],
        PAR["nsy"] * PAR["nsx"] * PAR["nry"] * PAR["nrx"],
    )

    D1op = Kirchhoff(
        z,
        x,
        t,
        s3d,
        r3d,
        v0,
        wav,
        wavc,
        y=y,
        trav=trav,
        mode=par["mode"],
        engine="numpy" if backend == "numpy" else "cuda",
    )
    D1op.trav_srcs = np.asarray(D1op.trav_srcs)
    D1op.trav_recs = np.asarray(D1op.trav_recs)
    if par["dynamic"]:
        D1op.amp_srcs = np.asarray(D1op.amp_srcs)
        D1op.amp_recs = np.asarray(D1op.amp_recs)

    # forward
    xx = np.random.normal(0, 1, PAR["ny"] * PAR["nx"] * PAR["nz"])
    assert_array_almost_equal(Dop @ xx, D1op @ xx, decimal=2)

    # adjoint
    yy = np.random.normal(
        0, 1, PAR["nry"] * PAR["nrx"] * PAR["nsy"] * PAR["nsx"] * PAR["nt"]
    )
    assert_array_almost_equal(Dop.H @ yy, D1op.H @ yy, decimal=2)


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1,
    reason="engine='cuda' not available for paired=True",
)
@pytest.mark.parametrize(
    "par", [(par1p), (par2p), (par3p), (par4p), (par1pd), (par2pd), (par3pd)]
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_kirchhoff2d_paired(par, dtype):
    """Dot-test for Kirchhoff operator with paired=True"""
    if not skfmm_enabled and par["mode"] == "eikonal":
        pytest.skip("skfmm not available")
    r2dp = s2dp + npp.array([[par["offset"]], [0]])
    Dop = _kirchhoff_paired(s2dp, r2dp, par, dtype, paired=True)
    assert Dop.dimsd == (PAR["nsx"], PAR["nt"])
    assert dottest(
        Dop,
        PAR["nsx"] * PAR["nt"],
        PAR["nx"] * PAR["nz"],
        rtol=1e-4 if dtype == np.float32 else 1e-6,
    )

    if dtype == np.float32 and par["wavfilter"]:
        pytest.xfail(
            "wavfilter=True returns float64 data for float32 (also paired=False)"
        )
    xx = np.random.normal(0, 1, PAR["nx"] * PAR["nz"]).astype(dtype)
    yy = Dop * xx
    xadj = Dop.H * yy
    assert yy.dtype == dtype
    assert xadj.dtype == dtype


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1,
    reason="engine='cuda' not available for paired=True",
)
@pytest.mark.parametrize("par", [(par1p), (par2p), (par3p), (par4p), (par1pd)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_kirchhoff3d_paired(par, dtype):
    """Dot-test for Kirchhoff operator with paired=True"""
    if not skfmm_enabled and par["mode"] == "eikonal":
        pytest.skip("skfmm not available")
    r3dp = s3dp + npp.array([[0], [par["offset"]], [0]])
    Dop = _kirchhoff_paired(s3dp, r3dp, par, dtype, paired=True)
    assert Dop.dimsd == (PAR["nsy"] * PAR["nsx"], PAR["nt"])
    assert dottest(
        Dop,
        PAR["nsy"] * PAR["nsx"] * PAR["nt"],
        PAR["ny"] * PAR["nx"] * PAR["nz"],
        rtol=1e-4 if dtype == np.float32 else 1e-6,
    )

    if dtype == np.float32 and par["wavfilter"]:
        pytest.xfail(
            "wavfilter=True returns float64 data for float32 (also paired=False)"
        )
    xx = np.random.normal(0, 1, PAR["ny"] * PAR["nx"] * PAR["nz"]).astype(dtype)
    yy = Dop * xx
    xadj = Dop.H * yy
    assert yy.dtype == dtype
    assert xadj.dtype == dtype


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1,
    reason="engine='cuda' not available for paired=True",
)
@pytest.mark.parametrize(
    "par", [(par1p), (par2p), (par3p), (par4p), (par1pd), (par2pd), (par3pd)]
)
@pytest.mark.parametrize("engine", ["numpy", "numba"])
def test_kirchhoff2d_paired_vs_allpairs(par, engine):
    """Compare 2D Kirchhoff operator forward and adjoint with paired=True and
    with all source-receiver pairs restricted to the pairs (i, i)"""
    if not skfmm_enabled and par["mode"] == "eikonal":
        pytest.skip("skfmm not available")
    ns = PAR["nsx"]
    r2dp = s2dp + npp.array([[par["offset"]], [0]])
    Dop = _kirchhoff_paired(s2dp, r2dp, par, np.float64, paired=True, engine=engine)
    D1op = _kirchhoff_paired(s2dp, r2dp, par, np.float64, paired=False, engine=engine)

    # forward
    xx = np.random.normal(0, 1, PAR["nx"] * PAR["nz"])
    yy1 = (D1op @ xx).reshape(ns, ns, PAR["nt"])[npp.arange(ns), npp.arange(ns)]
    assert_array_almost_equal(Dop @ xx, yy1.ravel(), decimal=8)

    # adjoint
    yy = np.random.normal(0, 1, (ns, PAR["nt"]))
    yy1 = npp.zeros((ns, ns, PAR["nt"]))
    yy1[npp.arange(ns), npp.arange(ns)] = yy
    assert_array_almost_equal(Dop.H @ yy.ravel(), D1op.H @ yy1.ravel(), decimal=8)


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1,
    reason="engine='cuda' not available for paired=True",
)
@pytest.mark.parametrize("par", [(par1p), (par2p), (par4p), (par1pd)])
@pytest.mark.parametrize("engine", ["numpy", "numba"])
def test_kirchhoff3d_paired_vs_allpairs(par, engine):
    """Compare 3D Kirchhoff operator forward and adjoint with paired=True and
    with all source-receiver pairs restricted to the pairs (i, i)"""
    if not skfmm_enabled and par["mode"] == "eikonal":
        pytest.skip("skfmm not available")
    ns = PAR["nsy"] * PAR["nsx"]
    r3dp = s3dp + npp.array([[0], [par["offset"]], [0]])
    Dop = _kirchhoff_paired(s3dp, r3dp, par, np.float64, paired=True, engine=engine)
    D1op = _kirchhoff_paired(s3dp, r3dp, par, np.float64, paired=False, engine=engine)

    # forward
    xx = np.random.normal(0, 1, PAR["ny"] * PAR["nx"] * PAR["nz"])
    yy1 = (D1op @ xx).reshape(ns, ns, PAR["nt"])[npp.arange(ns), npp.arange(ns)]
    assert_array_almost_equal(Dop @ xx, yy1.ravel(), decimal=8)

    # adjoint
    yy = np.random.normal(0, 1, (ns, PAR["nt"]))
    yy1 = npp.zeros((ns, ns, PAR["nt"]))
    yy1[npp.arange(ns), npp.arange(ns)] = yy
    assert_array_almost_equal(Dop.H @ yy.ravel(), D1op.H @ yy1.ravel(), decimal=8)
