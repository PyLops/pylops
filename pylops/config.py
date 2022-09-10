"""
Configuration
=============

The configuration module controls module-level behavior in PyLops.

You can either set behavior globally with getter/setter:

    get_ndarray_multiplication              Check the status of ndarray multiplication (True/False).
    set_ndarray_multiplication              Enable/disable ndarray multiplication.

or use context managers (with blocks):

    enabled_ndarray_multiplication          Enable ndarray multiplication within context.
    disabled_ndarray_multiplication         Disable ndarray multiplication within context.

"""
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator

__all__ = [
    "get_ndarray_multiplication",
    "set_ndarray_multiplication",
    "enabled_ndarray_multiplication",
    "disabled_ndarray_multiplication",
]


@dataclass
class Config:
    ndarray_multiplication: bool = True


_config = Config()


def get_ndarray_multiplication() -> bool:
    return _config.ndarray_multiplication


def set_ndarray_multiplication(val: bool) -> None:
    _config.ndarray_multiplication = val


@contextmanager
def enabled_ndarray_multiplication() -> Generator:
    enabled = get_ndarray_multiplication()
    set_ndarray_multiplication(True)
    try:
        yield enabled
    finally:
        set_ndarray_multiplication(enabled)


@contextmanager
def disabled_ndarray_multiplication() -> Generator:
    enabled = get_ndarray_multiplication()
    set_ndarray_multiplication(False)
    try:
        yield enabled
    finally:
        set_ndarray_multiplication(enabled)
