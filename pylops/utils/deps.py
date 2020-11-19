from importlib import util

numba_enabled = util.find_spec("numba") is not None
cupy_enabled = util.find_spec("cupy") is not None
cusignal_enabled = util.find_spec("cusignal") is not None
