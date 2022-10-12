__all__ = ["Report"]

# scooby is a soft dependency for pylops
from typing import Optional

try:
    from scooby import Report as ScoobyReport
except ImportError:

    class ScoobyReport:
        def __init__(self, additional, core, optional, ncol, text_width, sort):
            print(
                "\nNOTE: `pylops.Report` requires `scooby`. Install it via"
                "\n      `pip install scooby` or "
                "`conda install -c conda-forge scooby`.\n"
            )


class Report(ScoobyReport):
    r"""Print date, time, and version information.

    Use ``scooby`` to print date, time, and package version information in any
    environment (Jupyter notebook, IPython console, Python console, QT
    console), either as html-table (notebook) or as plain text (anywhere).

    Always shown are the OS, number of CPU(s), ``numpy``, ``scipy``,
    ``pylops``, ``sys.version``, and time/date.

    Additionally shown are, if they can be imported, ``IPython``, ``numba``,
    and ``matplotlib``. It also shows MKL information, if available.

    All modules provided in ``add_pckg`` are also shown.

    .. note::

        The package ``scooby`` has to be installed in order to use ``Report``:
        ``pip install scooby`` or ``conda install -c conda-forge scooby``.


    Parameters
    ----------
    add_pckg : packages, optional
        Package or list of packages to add to output information (must be
        imported beforehand).

    ncol : int, optional
        Number of package-columns in html table (no effect in text-version);
        Defaults to 3.

    text_width : int, optional
        The text width for non-HTML display modes

    sort : bool, optional
        Sort the packages when the report is shown


    Examples
    --------
    >>> import pytest
    >>> import dateutil
    >>> from pylops import Report
    >>> Report()                            # Default values
    >>> Report(pytest)                      # Provide additional package
    >>> Report([pytest, dateutil], ncol=5)  # Set nr of columns

    """

    def __init__(
        self,
        add_pckg: Optional[list] = None,
        ncol: int = 3,
        text_width: int = 80,
        sort: bool = False,
    ) -> None:
        """Initiate a scooby.Report instance."""

        # Mandatory packages.
        core = ["numpy", "scipy", "pylops"]

        # Optional packages.
        optional = ["IPython", "matplotlib", "numba"]

        super().__init__(
            additional=add_pckg,
            core=core,
            optional=optional,
            ncol=ncol,
            text_width=text_width,
            sort=sort,
        )
