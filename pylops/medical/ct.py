__all__ = [
    "CT2D",
]

import warnings
from typing import Optional

import numpy as np

from pylops import LinearOperator
from pylops.utils import deps
from pylops.utils.backend import get_array_module, get_module_name, to_numpy
from pylops.utils.decorators import reshaped
from pylops.utils.typing import (
    DTypeLike,
    InputDimsLike,
    NDArray,
    Tctengine,
    Tctprojectortype,
    Tctprojgeom,
)

astra_message = deps.astra_import("the astra module")

if astra_message is None:
    import astra
    import astra.experimental


class CT2D(LinearOperator):
    r"""2D Computerized Tomography

    Apply 2D computerized tomography operator to model to obtain a
    2D sinogram.

    Note that the CT2D operator is an overload of the ``astra``
    implementation of the tomographic operator. Refer to
    https://www.astra-toolbox.com/ for a detailed description of the
    input parameters.

    Parameters
    ----------
    dims : :obj:`list` or :obj:`int`
        Number of samples for each dimension. Must be 2-dimensional and of size :math:`n_y \times n_x`
    det_width : :obj:`float`
        Detector width
    det_count : :obj:`int`
        Number of detectors
    thetas : :obj:`numpy.ndarray`
        Vector of angles in degrees
    engine : :obj:`str`
        Engine used for computation (``cpu`` or ``cuda``).
    proj_geom_type : :obj:`str`, optional
        Type of projection geometry (``parallel`` or ``fanflat``)
    source_origin_dist : :obj:`float`, optional
        Distance between source and origin (only for ``proj_geom_type=fanflat``)
    origin_detector_dist : :obj:`float`, optional
        Distance between origin and detector along the source-origin line
        (only for "proj_geom_type=fanflat")
    projector_type : :obj:`int`, optional
        Type of projection kernel: ``strip`` (default), ``line``, or ``linear`` for
        ``engine=cpu``, and ``cuda`` (i.e., hardware-accelerated ``linear``) for
        ``engine=cuda``. For ``fanflat`` geometry, ``linear`` kernel is not supported.
    dtype : :obj:`str`, optional
        Type of elements in input array. Note that internally all operations will be
        performed in float32 dtype because of ASTRA compatibility, and the output will
        be converted to the requested dtype afterwards.
    name : :obj:`str`, optional
        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Attributes
    ----------
    dims : :obj:`tuple`
        Shape of the array after the adjoint, but before flattening.

        For example, ``x_reshaped = (Op.H * y.ravel()).reshape(Op.dims)``.
    dimsd : :obj:`tuple`
        Shape of the array after the forward, but before flattening.

        For example, ``y_reshaped = (Op * x.ravel()).reshape(Op.dimsd)``.
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved
        explicitly (``True``) or not (``False``)

    Raises
    ------
    ValueError
        If a projector type other than 'cuda' is provided for 'cuda' engine.

    NotImplementedError
        If ASTRA toolbox could not be imported.

    Notes
    -----
    The CT2D operator applies parallel or fan beam computerized tomography operators
    to 2-dimensional objects and produces their corresponding sinograms.

    Mathematically the forward operator can be described as [1]_:

    .. math::
        s(r,\theta; i) = \int_l i(l(r,\theta)) dl

    where :math:`l(r,\theta)` is the summation line and :math:`i(x, y)`
    is the intensity map of the model. Here, :math:`\theta` refers to the angle
    between the y-axis (:math:`y`) and the summation line and :math:`r` is
    the distance from the origin of the summation line.

    .. [1] http://people.compute.dtu.dk/pcha/HDtomo/astra-introduction.pdf

    """

    def __init__(
        self,
        dims: InputDimsLike,
        det_width: float,
        det_count: int,
        thetas: NDArray,
        engine: Tctengine,
        proj_geom_type: Tctprojgeom = "parallel",
        source_origin_dist: Optional[float] = None,
        origin_detector_dist: Optional[float] = None,
        projector_type: Optional[Tctprojectortype] = None,
        dtype: DTypeLike = "float32",
        name: str = "C",
    ) -> None:
        if astra_message is not None:
            raise NotImplementedError(astra_message)

        self.dims = dims
        self.det_width = det_width
        self.det_count = det_count
        self.thetas = to_numpy(thetas)  # ASTRA can only consume angles as a NumPy array
        self.engine = engine
        self.proj_geom_type = proj_geom_type
        self.source_origin_dist = source_origin_dist
        self.origin_detector_dist = origin_detector_dist

        # make "strip" projector type default for cpu engine and only allow cuda otherwise
        if engine == "cpu":
            if projector_type is None:
                projector_type = "strip"
            # fanflat geometry projectors need an appropriate suffix (unless it's "cuda")
            if projector_type == "cuda":
                warnings.warn("'cuda' projector type specified with 'cpu' engine.")
            elif proj_geom_type == "fanflat":
                projector_type += "_fanflat"
        elif engine == "cuda":
            if projector_type in [None, "cuda"]:
                projector_type = "cuda"
            else:
                raise ValueError(
                    "Only 'cuda' projector type is supported for 'cuda' engine."
                )
        else:
            raise NotImplementedError("Engine must be 'cpu' or 'cuda'")
        self.projector_type = projector_type

        # create create volume and projection geometries as well as projector
        self._init_geometries()
        if engine == "cuda":
            # efficient GPU data exchange only implemented for 3D data in ASTRA, so we
            # emulate 2D geometry as 3D case with 1 slice
            self._init_1_slice_3d_geometries()

        super().__init__(
            dtype=np.dtype(dtype), dims=dims, dimsd=(len(thetas), det_count), name=name
        )

    def _init_geometries(self):
        self.vol_geom = astra.create_vol_geom(self.dims)
        if self.proj_geom_type == "parallel":
            self.proj_geom = astra.create_proj_geom(
                "parallel", self.det_width, self.det_count, self.thetas
            )
        elif self.proj_geom_type == "fanflat":
            self.proj_geom = astra.create_proj_geom(
                "fanflat",
                self.det_width,
                self.det_count,
                self.thetas,
                self.source_origin_dist,
                self.origin_detector_dist,
            )
        self.projector_id = astra.create_projector(
            self.projector_type, self.proj_geom, self.vol_geom
        )

    def _init_1_slice_3d_geometries(self):
        """Emulate 2D geometry as 3D to be able to use zero-copy GPU data exchange in ASTRA."""
        self._3d_vol_geom = astra.create_vol_geom(*self.dims, 1)
        if self.proj_geom_type == "parallel":
            self._3d_proj_geom = astra.create_proj_geom(
                "parallel3d", 1.0, self.det_width, 1, self.det_count, self.thetas
            )
        elif self.proj_geom_type == "fanflat":
            self._3d_proj_geom = astra.create_proj_geom(
                "cone",
                1.0,
                self.det_width,
                1,
                self.det_count,
                self.thetas,
                self.source_origin_dist,
                self.origin_detector_dist,
            )
        self._3d_projector_id = astra.create_projector(
            "cuda3d", self._3d_proj_geom, self._3d_vol_geom
        )

    def _astra_project(self, x, mode):
        """Perform forward/backward projection using ASTRA."""
        ncp = get_array_module(x)
        backend = get_module_name(ncp)
        x_dtype = x.dtype
        if x_dtype != ncp.float32:
            warnings.warn(
                "CT2D operator received input that is not of float32 dtype. It will "
                "be cast into float32 internally for ASTRA compatibility."
            )
            x = x.astype(ncp.float32)
        if self.engine == "cpu":
            x = to_numpy(x)
            if mode == "forward":
                y_id, y = astra.create_sino(x, self.projector_id)
            elif mode == "backward":
                y_id, y = astra.create_backprojection(x, self.projector_id)
            astra.data2d.delete(y_id)
        if self.engine == "cuda":
            # Use zero-copy GPU data exchange, which is only implemented for contiguous
            # 3D arrays in ASTRA
            if backend in ["numpy", "cupy"] and not x.flags["C_CONTIGUOUS"]:
                # JAX shouldn't have non-contiguous arrays here. In the worst case ASTRA
                # should throw an error later
                warnings.warn(
                    "CT2D operator received input that is not of contiguous. It will be "
                    "cast into a contiguous array internally for ASTRA compatibility."
                )
                x = ncp.ascontiguousarray(x)
            x = ncp.expand_dims(x, axis=0)
            if mode == "forward":
                y = ncp.empty_like(x, shape=astra.geom_size(self._3d_proj_geom))
                astra.experimental.direct_FP3D(self._3d_projector_id, x, y)
            elif mode == "backward":
                y = ncp.empty_like(x, shape=astra.geom_size(self._3d_vol_geom))
                astra.experimental.direct_BP3D(self._3d_projector_id, y, x)
        return y.astype(x_dtype)

    @reshaped
    def _matvec(self, x):
        return self._astra_project(x, mode="forward")

    @reshaped
    def _rmatvec(self, x):
        return self._astra_project(x, mode="backward")

    def __del__(self):
        if hasattr(self, "projector_id"):
            astra.projector.delete(self.projector_id)
        if hasattr(self, "_3d_projector_id"):
            astra.projector.delete(self._3d_projector_id)
