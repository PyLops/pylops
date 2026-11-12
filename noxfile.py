from __future__ import annotations

import shutil
from pathlib import Path

import nox

DIR = Path(__file__).parent.resolve()

nox.options.default_venv_backend = "uv"
nox.options.sessions = ["lint", "tests"]


@nox.session
def lint(session):
    """Run the Ruff linter."""
    session.install("ruff")
    session.run(
        "ruff",
        "check",
        "docs/source",
        "examples/",
        "pylops/",
        "pytests/",
        "tutorials/",
    )


@nox.session(python=["3.11", "3.12", "3.13", "3.14"])
def tests(session: nox.Session) -> None:
    """
    Run unit tests.
    """
    session.install("-e", ".[advanced,deep,stat]")
    session.install("--group", "dev")
    session.run("pytest", *session.posargs)


@nox.session(python=["3.11", "3.12", "3.13", "3.14"])
def build(session: nox.Session) -> None:
    """
    Build an SDist and wheel.
    """

    build_path = DIR.joinpath("build")
    if build_path.exists():
        shutil.rmtree(build_path)

    session.install("build")
    session.run("python", "-m", "build")
