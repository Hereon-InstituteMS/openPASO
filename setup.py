"""Packaging glue for the wheel; every static field lives in pyproject.toml.

The checkout keeps its flat layout -- src/core, src/backends, src/tools, ... are
imported by those names, with src/ on sys.path -- because tests, the coverage
harness and the docs all address the code that way. A wheel cannot ship
packages called ``core`` or ``tools`` (they would collide with anyone else's), so
the same directories are installed as ``openpaso.core``, ``openpaso.backends``
..., and ``openpaso/__init__.py`` puts its own directory on sys.path so the flat
imports inside the code keep resolving. ``data/`` (the coupling participants,
the post-mortems, the solver knowledge modules) becomes ``openpaso.data``.
"""
from setuptools import find_packages, setup

_src = find_packages("src")
setup(
    packages=["openpaso", "openpaso.data"] + ["openpaso." + p for p in _src],
    package_dir={"openpaso": "src", "openpaso.data": "data"},
    include_package_data=True,
)
