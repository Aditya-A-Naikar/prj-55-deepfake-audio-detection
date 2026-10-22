"""Web application package for PRJ_55 Audio Forensics Platform."""

from prj55.web.app import app
from prj55.web.service import ForensicsService

__all__ = ["app", "ForensicsService"]
