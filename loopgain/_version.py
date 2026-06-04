"""Single source of truth for the package version.

``loopgain/__init__.py``, ``loopgain/telemetry.py`` (product receiver), and
``loopgain/funnel.py`` (opt-in funnel telemetry) all import ``__version__``
from here so the value never drifts between ``__version__`` and the
``library_version`` field on any telemetry payload. Update this file (and
``pyproject.toml``) for each release.
"""

__version__ = "0.4.0"
