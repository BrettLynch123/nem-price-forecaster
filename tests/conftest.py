"""
Pytest configuration — put the sidecar app/ on sys.path so the sidecar modules
(which import each other by bare name, e.g. ``from forecast_cache import ...``)
import cleanly during testing.

The sidecar and add-on app trees are byte-identical mirrors, so testing the
sidecar tree validates both.
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SIDECAR_APP = os.path.join(_REPO_ROOT, "sidecar", "app")

if _SIDECAR_APP not in sys.path:
    sys.path.insert(0, _SIDECAR_APP)
