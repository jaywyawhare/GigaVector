"""GigaVector dashboard — ships static files for the built-in web UI."""

import os

from .server import DashboardServer


def get_static_dir() -> str:
    """Return the absolute path to the dashboard static files directory."""
    return os.path.join(os.path.dirname(__file__), "static")
