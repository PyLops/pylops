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
]

import os
from importlib import import_module
from typing import Optional


def check_module_enabled(
    module: str,
    envrionment_str: Optional[str] = None,
    envrionment_val: Optional[int] = 1,
) -> bool:
    """
    Checks whether a specific module can be imported in the current Python environment.

    Args:
        module (str): The name of the module to check import state for.
        envrionment_str (Optional[str]): An optional environment variable name to check for. If provided,
            the function will return True only if the environment variable is set to the specified value.
            Defaults to None.
        envrionment_val (Optional[str]): The value to compare the environment variable against. Defaults to "1".

    Returns:
        bool: True if the module is available, False otherwise.
    """
    # try to import the module
    try:
        _ = import_module(module)  # noqa: F401
        # run envrionment check if needed
        if envrionment_str is not None:
            # return True if the value matches expected value
            return int(os.getenv(envrionment_str, envrionment_val)) == envrionment_val
        # if no environment check return True as import_module worked
        else:
            return True
    # if cannot import and provides expected Exceptions, return False
    except (ImportError, ModuleNotFoundError):
        return False
    # raise warning if anyother exception raised in import
    except Exception as e:
        raise UserWarning(f"Unexpceted Exception when importing {module}") from e


cupy_enabled = check_module_enabled("cupy", "CUPY_PYLOPS")
cusignal_enabled = check_module_enabled("cusignal", "CUSIGNAL_PYLOPS")
devito_enabled = check_module_enabled("devito")
numba_enabled = check_module_enabled("numba")
pyfftw_enabled = check_module_enabled("pyfftw")
pywt_enabled = check_module_enabled("pywt")
skfmm_enabled = check_module_enabled("skfmm")
spgl1_enabled = check_module_enabled("spgl1")
sympy_enabled = check_module_enabled("sympy")
torch_enabled = check_module_enabled("torch")


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
