from abc import ABCMeta, abstractmethod
from typing import Optional
from scipy.sparse.linalg import LinearOperator as spLinearOperator
from pylops.utils.typing import DTypeLike, ShapeLike, NDArray
from ..linearoperator import LinearOperator


class BaseLinearOperator(LinearOperator, metaclass=ABCMeta):
    def __init__(self, Op: Optional[spLinearOperator] = None, dtype: Optional[DTypeLike] = None,
                 shape: Optional[ShapeLike] = None, dims: Optional[ShapeLike] = None, dimsd: Optional[ShapeLike] = None,
                 clinear: Optional[bool] = None, explicit: Optional[bool] = None, name: Optional[str] = None) -> None:
        super().__init__(Op, dtype, shape, dims, dimsd, clinear, explicit, name)

    @abstractmethod
    def _matvec(self, x: NDArray) -> NDArray:
        """Matrix Vector Handler

        Parameters
        ----------
        x : :obj:`np.ndarray`
            Data
        """
        pass

    @abstractmethod
    def _rmatvec(self, x: NDArray) -> NDArray:
        """Adjoint matrix-vector handler

        Parameters
        ----------
        x : :obj:`np.ndarray`
            Data
        """
        pass
