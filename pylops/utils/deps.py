__all__ = [
    "numba_enabled",
    "cupy_enabled",
    "cusignal_enabled",
]

import os
from importlib import util

cupy_enabled = (
    util.find_spec("cupy") is not None and int(os.getenv("CUPY_PYLOPS", 1)) == 1
)
cusignal_enabled = (
    util.find_spec("cusignal") is not None and int(os.getenv("CUSIGNAL_PYLOPS", 1)) == 1
)
numba_enabled = util.find_spec("numba") is not None
torch_enabled = util.find_spec("torch") is not None
