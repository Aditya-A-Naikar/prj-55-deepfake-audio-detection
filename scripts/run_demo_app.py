#!/usr/bin/env python3
"""Launcher script for PRJ_55 Audio Forensics Demonstration Platform (Plan 4.2).

Starts the local FastAPI application server using Uvicorn.
Provides access to the interactive web workspace, real-time AASIST-L inference,
channel degradation simulator, and forensic evidence dashboard.

Usage:
    python scripts/run_demo_app.py
    python scripts/run_demo_app.py --host 127.0.0.1 --port 8000
    python scripts/run_demo_app.py --device mps
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add repository root and src to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

import uvicorn  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PRJ_55 Audio Forensics Demonstration Platform Launcher"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host interface to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number to bind (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code modifications",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("  PRJ_55 Deepfake Audio Forensics Platform")
    print("  Architecture: AASIST-L (85,306 parameters)")
    print(f"  Interface:    http://{args.host}:{args.port}")
    print("  Status:       Ready for local verification and university defense")
    print("=" * 72)

    uvicorn.run(
        "prj55.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
