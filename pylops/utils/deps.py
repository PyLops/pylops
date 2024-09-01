__all__ = [
    "cupy_enabled",
    "jax_enabled",
    "devito_enabled",
    "dtcwt_enabled",
    "ucurv_enabled",    
    "numba_enabled",
    "pyfftw_enabled",
    "pywt_enabled",
    "skfmm_enabled",
    "spgl1_enabled",
    "sympy_enabled",
    "torch_enabled",
]

import os
from importlib import import_module, util
from typing import Optional


# error message at import of available package
def cupy_import(message: Optional[str] = None) -> str:
    # detect if cupy is available and the user is expecting to be used
    cupy_test = (
        util.find_spec("cupy") is not None and int(os.getenv("CUPY_PYLOPS", 1)) == 1
    )
    # if cupy should be importable
    if cupy_test:
        # try importing it
        try:
            import_module("cupy")  # noqa: F401

            # if successful set the message to None.
            cupy_message = None
        # if unable to import but the package is installed
        except (ImportError, ModuleNotFoundError) as e:
            cupy_message = (
                f"Failed to import cupy, Falling back to CPU (error: {e}). "
                "Please ensure your CUDA environment is set up correctly "
                "for more details visit 'https://docs.cupy.dev/en/stable/install.html'"
            )
            print(UserWarning(cupy_message))
    # if cupy_test is False, it means not installed or environment variable set to 0
    else:
        cupy_message = (
            "Cupy package not installed or os.getenv('CUPY_PYLOPS') == 0. "
            f"In order to be able to use {message} "
            "ensure 'os.getenv('CUPY_PYLOPS') == 1' and run "
            "'pip install cupy'; "
            "for more details visit 'https://docs.cupy.dev/en/stable/install.html'"
        )

    return cupy_message


def jax_import(message: Optional[str] = None) -> str:
    jax_test = (
        util.find_spec("jax") is not None and int(os.getenv("JAX_PYLOPS", 1)) == 1
    )
    if jax_test:
        try:
            import_module("jax")  # noqa: F401

            jax_message = None
        except (ImportError, ModuleNotFoundError) as e:
            jax_message = (
                f"Failed to import jax, Falling back to numpy (error: {e}). "
                "Please ensure your environment is set up correctly "
                "for more details visit 'https://jax.readthedocs.io/en/latest/installation.html'"
            )
            print(UserWarning(jax_message))
    else:
        jax_message = (
            "Jax package not installed or os.getenv('JAX_PYLOPS') == 0. "
            f"In order to be able to use {message} "
            "ensure 'os.getenv('JAX_PYLOPS') == 1' and run "
            "'pip install jax'; "
            "for more details visit 'https://jax.readthedocs.io/en/latest/installation.html'"
        )

    return jax_message


def devito_import(message: Optional[str] = None) -> str:
    if devito_enabled:
        try:
            import_module("devito")  # noqa: F401

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


def dtcwt_import(message: Optional[str] = None) -> str:
    if dtcwt_enabled:
        try:
            import dtcwt  # noqa: F401

            dtcwt_message = None
        except Exception as e:
            dtcwt_message = f"Failed to import dtcwt (error:{e})."
    else:
        dtcwt_message = (
            f"Dtcwt not available. "
            f"In order to be able to use "
            f'{message} run "pip install dtcwt".'
        )
    return dtcwt_message

def ucurv_import(message: Optional[str] = None) -> str:
    if ucurv_enabled:
        try:
            import ucurv  # noqa: F401

            ucurv_message = None
        except Exception as e:
            ucurv_message = f"Failed to import ucurv (error:{e})."
    else:
        ucurv_message = (
            f"UCURV not available. "
            f"In order to be able to use "
            f'{message} run "pip install ucurv".'
        )
    return ucurv_message

def numba_import(message: Optional[str] = None) -> str:
    if numba_enabled:
        try:
            import_module("numba")  # noqa: F401

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


def pyfftw_import(message: Optional[str] = None) -> str:
    if pyfftw_enabled:
        try:
            import_module("pyfftw")  # noqa: F401

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


def pywt_import(message: Optional[str] = None) -> str:
    if pywt_enabled:
        try:
            import_module("pywt")  # noqa: F401

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


def skfmm_import(message: Optional[str] = None) -> str:
    if skfmm_enabled:
        try:
            import_module("skfmm")  # noqa: F401

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


def spgl1_import(message: Optional[str] = None) -> str:
    if spgl1_enabled:
        try:
            import_module("spgl1")  # noqa: F401

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


def sympy_import(message: Optional[str] = None) -> str:
    if sympy_enabled:
        try:
            import_module("sympy")  # noqa: F401

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


# Set package availability booleans
# cupy and jax: the package is imported to check everything is working correctly,
# if not the package is disabled. We do this here as these libraries are used as drop-in
# replacement for many numpy and scipy routines when cupy/jax arrays are provided.
# all other libraries: we simply check if the package is available and postpone its import
# to check everything is working correctly when a user tries to create an operator that requires
# such a package
cupy_enabled: bool = (
    True if (cupy_import() is None and int(os.getenv("CUPY_PYLOPS", 1)) == 1) else False
)
jax_enabled: bool = (
    True if (jax_import() is None and int(os.getenv("JAX_PYLOPS", 1)) == 1) else False
)
devito_enabled = util.find_spec("devito") is not None
dtcwt_enabled = util.find_spec("dtcwt") is not None
ucurv_enabled = util.find_spec("ucurv") is not None
numba_enabled = util.find_spec("numba") is not None
pyfftw_enabled = util.find_spec("pyfftw") is not None
pywt_enabled = util.find_spec("pywt") is not None
skfmm_enabled = util.find_spec("skfmm") is not None
spgl1_enabled = util.find_spec("spgl1") is not None
sympy_enabled = util.find_spec("sympy") is not None
torch_enabled = util.find_spec("torch") is not None
