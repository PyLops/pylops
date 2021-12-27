import os
from importlib import util

numba_enabled = util.find_spec("numba") is not None
cupy_enabled = (
    util.find_spec("cupy") is not None and int(os.getenv("CUPY_PYLOPS", 1)) == 1
)
cusignal_enabled = (
    util.find_spec("cusignal") is not None and int(os.getenv("CUSIGNAL_PYLOPS", 1)) == 1
)
