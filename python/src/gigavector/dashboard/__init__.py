"""GigaVector dashboard - ships static files for the built-in web UI."""

import os

from .backend.server import DashboardServer


def get_static_dir() -> str:
    """Return the absolute path to the dashboard frontend directory.

        dashboard/frontend/
            index.html
            assets/   (favicon, logo, ...)
            styles/   (style.css, ...)
            src/      (index.js, ...)
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
