__all__ = [
    "cupy_enabled",
    "cusignal_enabled",
    "devito_enabled",
    "numba_enabled",
    "pyfftw_enabled",
    "pywt_enabled",
    "skfmm_enabled",
    "spgl1_enabled",
    "sympy_enabled",
    "torch_enabled",
    "mkl_fft_enabled"
]

import os
from importlib import util

# check package availability
cupy_enabled = (
    util.find_spec("cupy") is not None and int(os.getenv("CUPY_PYLOPS", 1)) == 1
)
cusignal_enabled = (
    util.find_spec("cusignal") is not None and int(os.getenv("CUSIGNAL_PYLOPS", 1)) == 1
)
devito_enabled = util.find_spec("devito") is not None
numba_enabled = util.find_spec("numba") is not None
pyfftw_enabled = util.find_spec("pyfftw") is not None
pywt_enabled = util.find_spec("pywt") is not None
skfmm_enabled = util.find_spec("skfmm") is not None
spgl1_enabled = util.find_spec("spgl1") is not None
sympy_enabled = util.find_spec("sympy") is not None
torch_enabled = util.find_spec("torch") is not None
mkl_fft_enabled = util.find_spec("mkl_fft") is not None


# error message at import of available package
def devito_import(message):
    if devito_enabled:
        try:
            import devito  # noqa: F401

            devito_message = None
        except Exception as e:
            devito_message = f"Failed to import devito (error:{e})."
    else:
        devito_message = (
            f"Devito not available. "
            f"In order to be able to use "
            f'{message} run "pip install devito".'
        )
    return devito_message


def numba_import(message):
    if numba_enabled:
        try:
            import numba  # noqa: F401

            numba_message = None
        except Exception as e:
            numba_message = f"Failed to import numba (error:{e}), use numpy."
    else:
        numba_message = (
            "Numba not available, reverting to numpy. "
            "In order to be able to use "
            f"{message} run "
            f'"pip install numba" or '
            f'"conda install numba".'
        )
    return numba_message


def pyfftw_import(message):
    if pyfftw_enabled:
        try:
            import pyfftw  # noqa: F401

            pyfftw_message = None
        except Exception as e:
            pyfftw_message = f"Failed to import pyfftw (error:{e}), use numpy."
    else:
        pyfftw_message = (
            "Pyfftw not available, reverting to numpy. "
            "In order to be able to use "
            f"{message} run "
            f'"pip install pyFFTW" or '
            f'"conda install -c conda-forge pyfftw".'
        )
    return pyfftw_message


def mkl_fft_import(message):
    if mkl_fft_enabled:
        try:
            from mkl_fft import _scipy_fft_backend  # noqa: F401
            from mkl_fft import _numpy_fft  # noqa: F401
            mkl_fft_message = None
        except Exception as e:
            mkl_fft_message = f"Failed to import pyfftw (error:{e}), use numpy."
    else:
        mkl_fft_message = (
            "mkl_fft not available, reverting to numpy. "
            "In order to be able to use "
            f"{message} run "
            f'"pip install mkl_fft" or '
            f'"conda install -c conda-forge mkl_fft".'
        )
    return mkl_fft_message


def pywt_import(message):
    if pywt_enabled:
        try:
            import pywt  # noqa: F401

            pywt_message = None
        except Exception as e:
            pywt_message = f"Failed to import pywt (error:{e})."
    else:
        pywt_message = (
            "Pywt not available. "
            "In order to be able to use "
            f"{message} run "
            f'"pip install PyWavelets" or '
            f'"conda install pywavelets".'
        )
    return pywt_message


def skfmm_import(message):
    if skfmm_enabled:
        try:
            import skfmm  # noqa: F401

            skfmm_message = None
        except Exception as e:
            skfmm_message = f"Failed to import skfmm (error:{e})."
    else:
        skfmm_message = (
            f"Skfmm package not installed. In order to be able to use "
            f"{message} run "
            f'"pip install scikit-fmm" or '
            f'"conda install -c conda-forge scikit-fmm".'
        )
    return skfmm_message


def spgl1_import(message):
    if spgl1_enabled:
        try:
            import spgl1  # noqa: F401

            spgl1_message = None
        except Exception as e:
            spgl1_message = f"Failed to import spgl1 (error:{e})."
    else:
        spgl1_message = (
            f"Spgl1 package not installed. In order to be able to use "
            f"{message} run "
            f'"pip install spgl1".'
        )
    return spgl1_message


def sympy_import(message):
    if sympy_enabled:
        try:
            import sympy  # noqa: F401

            sympy_message = None
        except Exception as e:
            sympy_message = f"Failed to import sympy (error:{e})."
    else:
        sympy_message = (
            f"Sympy package not installed. In order to be able to use "
            f"{message} run "
            f'"pip install sympy".'
        )
    return sympy_message
