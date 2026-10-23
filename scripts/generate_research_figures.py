#!/usr/bin/env python3
"""Convenience CLI wrapper for reports/analysis/generate_research_figures.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "reports" / "analysis" / "generate_research_figures.py"

if __name__ == "__main__":
    cmd = [sys.executable, str(SCRIPT_PATH)] + sys.argv[1:]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    sys.exit(result.returncode)
