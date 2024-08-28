__all__ = ["UDCT"]

from typing import Any, NewType, Union

import numpy as np

from pylops import LinearOperator
from pylops.utils import deps
from pylops.utils._internal import _value_or_sized_to_tuple
from pylops.utils.decorators import reshaped
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray
from ucurv import *

class UDCT(LinearOperator):
    def __init__(self, sz, cfg, complex = False, sparse = False, dtype=None):
        self.udct = udct(sz, cfg, complex, sparse)
        self.shape = (self.udct.len, np.prod(sz))
        self.dtype = np.dtype(dtype)
        self.explicit = False    
        self.rmatvec_count = 0
        self.matvec_count = 0
    def _matvec(self, x):
        img = x.reshape(self.udct.sz)
        band = ucurvfwd(img, self.udct)
        bvec = bands2vec(band)        
        return bvec

    def _rmatvec(self, x):
        band = vec2bands(x, self.udct)
        recon = ucurvinv(band, self.udct)
        recon2 = recon.reshape(self.udct.sz)
        return recon2