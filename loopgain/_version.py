"""Single source of truth for the package version.

Both ``loopgain/__init__.py`` and ``loopgain/telemetry.py`` import
``__version__`` from here so the value never drifts between
``__version__`` and the ``library_version`` field on telemetry payloads.
Update this file (and ``pyproject.toml``) for each release.
"""

__version__ = "0.2.0"
