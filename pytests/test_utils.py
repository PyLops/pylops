import os

import pytest

from pylops import utils

try:
    import scooby
except ImportError:
    scooby = False


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
def test_report(capsys):
    out, _ = capsys.readouterr()  # Empty capsys

    # Reporting is done by the external package scooby.
    # We just ensure the shown packages do not change (core and optional).
    if scooby:
        out1 = utils.Report()
        out2 = scooby.Report(
            core=["numpy", "scipy", "pylops"],
            optional=["IPython", "matplotlib", "numba"],
        )

        # Ensure they're the same; exclude time to avoid errors.
        assert out1.__repr__()[115:] == out2.__repr__()[115:]

    else:  # soft dependency
        _ = utils.Report()
        out, _ = capsys.readouterr()  # Empty capsys
        assert "NOTE: `pylops.Report` requires `scooby`. Install it via" in out
