from .base_linearoperator import BaseLinearOperator
from ..utils import NDArray


class Sample(BaseLinearOperator):

    def _matvec(self, x: NDArray) -> NDArray:
        pass

    def _rmatvec(self, x: NDArray) -> NDArray:
        pass
