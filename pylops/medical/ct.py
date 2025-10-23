__all__ = [
    "CT2D",
]

import logging
from typing import Optional

import numpy as np

from pylops import LinearOperator
from pylops.utils import deps
from pylops.utils.decorators import reshaped
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


astra_message = deps.astra_import("the astra module")

if astra_message is None:
    import astra


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
    proj_geom_type : :obj:`str`, optional
        Type of projection geometry (``parallel`` or ``fanflat``)
    source_origin_dist : :obj:`float`, optional
        Distance between source and origin (only for ``proj_geom_type=fanflat``)
    origin_detector_dist : :obj:`float`, optional
        Distance between origin and detector along the source-origin line
        (only for "proj_geom_type=fanflat")
    projector_type : :obj:`int`, optional
        Type of projection geometry (``strip``, or ``line``, or ``linear``)
    dtype : :obj:`str`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved
        explicitly (``True``) or not (``False``)

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
        det_width: int,
        det_count: float,
        thetas: NDArray,
        proj_geom_type: Optional[str] = "parallel",
        source_origin_dist: float = None,
        origin_detector_dist: float = None,
        projector_type: Optional[str] = "strip",
        dtype: DTypeLike = "float64",
        name: str = "C",
    ) -> None:
        if astra_message is not None:
            raise NotImplementedError(astra_message)

        # create volume and projection geometries
        self.vol_geom = astra.create_vol_geom(dims)
        if proj_geom_type == "parallel":
            self.proj_geom = astra.create_proj_geom(
                proj_geom_type, det_width, det_count, thetas
            )
        else:
            self.proj_geom = astra.create_proj_geom(
                proj_geom_type,
                det_width,
                det_count,
                thetas,
                source_origin_dist,
                origin_detector_dist,
            )

        # create projector
        self.proj_id = astra.create_projector(
            projector_type, self.proj_geom, self.vol_geom
        )
        super().__init__(
            dtype=np.dtype(dtype), dims=dims, dimsd=(len(thetas), det_count), name=name
        )

    @reshaped
    def _matvec(self, x):
        y_id, y = astra.create_sino(x, self.proj_id)
        astra.data2d.delete(y_id)
        return y

    @reshaped
    def _rmatvec(self, x):
        y_id, y = astra.create_backprojection(x, self.proj_id)
        astra.data2d.delete(y_id)
        return y

    def __del__(self):
        astra.projector.delete(self.proj_id)
