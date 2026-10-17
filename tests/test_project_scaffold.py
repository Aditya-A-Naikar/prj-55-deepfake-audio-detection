"""
test_project_scaffold.py
========================
PRJ_55 — Stage 0 Scaffold Integrity Tests

Verifies that the required directory and file structure exists after bootstrap.
These tests do NOT import any ML libraries, download data, or execute models.
They are designed to pass with only the Python standard library.

Run with:
    python -m pytest tests/test_project_scaffold.py -v
"""

import pathlib

# ── Resolve the project root (two levels up from this file) ─────────────────
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.resolve()


# ── Helper ───────────────────────────────────────────────────────────────────

def _path(*parts: str) -> pathlib.Path:
    """Return an absolute path relative to PROJECT_ROOT."""
    return PROJECT_ROOT.joinpath(*parts)


# ── Directory existence tests ────────────────────────────────────────────────

class TestDirectoryStructure:
    """All required directories must be present."""

    REQUIRED_DIRS = [
        "docs",
        "docs/decisions",
        "data",
        "models",
        "configs",
        "scripts",
        "src",
        "src/prj55",
        "tests",
        "reports",
    ]

    def test_required_directories_exist(self):
        missing = [d for d in self.REQUIRED_DIRS if not _path(d).is_dir()]
        assert not missing, (
            "Missing directories:\n" + "\n".join(f"  {PROJECT_ROOT / d}" for d in missing)
        )


# ── File existence tests ─────────────────────────────────────────────────────

class TestFileStructure:
    """All required files must be present and non-empty."""

    REQUIRED_FILES = [
        "README.md",
        ".gitignore",
        ".gitattributes",
        "pyproject.toml",
        "docs/project_charter.md",
        "docs/week1_execution_plan.md",
        "docs/experiment_log.md",
        "docs/data_governance.md",
        "docs/source_registry.md",
        "docs/decisions/ADR-001-compute-strategy.md",
        "data/README.md",
        "models/README.md",
        "configs/README.md",
        "scripts/README.md",
        "src/prj55/__init__.py",
        "tests/test_project_scaffold.py",
        "reports/README.md",
    ]

    def test_required_files_exist(self):
        missing = [f for f in self.REQUIRED_FILES if not _path(f).is_file()]
        assert not missing, (
            "Missing files:\n" + "\n".join(f"  {PROJECT_ROOT / f}" for f in missing)
        )

    def test_required_files_are_not_empty(self):
        empty = [
            f for f in self.REQUIRED_FILES
            if _path(f).is_file() and _path(f).stat().st_size == 0
        ]
        assert not empty, (
            "Empty files (should have content):\n"
            + "\n".join(f"  {PROJECT_ROOT / f}" for f in empty)
        )


# ── Package importability test ───────────────────────────────────────────────

class TestPackageImport:
    """The prj55 package must be importable and expose expected attributes."""

    def test_package_is_importable(self):
        """Import must succeed without installing any ML libraries."""
        import importlib
        import sys
        # Ensure src/ is on the path for editable-install-free testing
        src_path = str(PROJECT_ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        prj55 = importlib.import_module("prj55")
        assert prj55 is not None

    def test_package_version_attribute(self):
        import sys
        src_path = str(PROJECT_ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        import prj55
        assert hasattr(prj55, "__version__"), "__version__ attribute missing from prj55"
        assert prj55.__version__, "__version__ must be a non-empty string"

    def test_package_status_is_bootstrap(self):
        import sys
        src_path = str(PROJECT_ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        import prj55
        assert hasattr(prj55, "__status__"), "__status__ attribute missing from prj55"
        assert prj55.__status__ == "bootstrap", (
            f"Expected __status__ == 'bootstrap', got {prj55.__status__!r}"
        )


# ── .gitignore sanity tests ──────────────────────────────────────────────────

class TestGitignore:
    """The .gitignore must contain patterns for critical exclusions."""

    REQUIRED_PATTERNS = [
        ".env",
        "*.wav",
        "*.flac",
        "*.pt",
        "*.pth",
        "*.ckpt",
        "__pycache__",
        ".venv",
        "*.log",
    ]

    def _read_gitignore(self) -> str:
        return _path(".gitignore").read_text(encoding="utf-8")

    def test_gitignore_excludes_env_files(self):
        content = self._read_gitignore()
        assert ".env" in content, ".gitignore must exclude .env files"

    def test_gitignore_excludes_audio_files(self):
        content = self._read_gitignore()
        assert "*.wav" in content, ".gitignore must exclude *.wav"
        assert "*.flac" in content, ".gitignore must exclude *.flac"

    def test_gitignore_excludes_model_weights(self):
        content = self._read_gitignore()
        assert "*.pt" in content, ".gitignore must exclude *.pt"
        assert "*.ckpt" in content, ".gitignore must exclude *.ckpt"

    def test_gitignore_excludes_pycache(self):
        content = self._read_gitignore()
        assert "__pycache__" in content, ".gitignore must exclude __pycache__"

    def test_gitignore_excludes_venv(self):
        content = self._read_gitignore()
        assert ".venv" in content or "venv/" in content, (
            ".gitignore must exclude virtual environments"
        )


# ── pyproject.toml sanity ────────────────────────────────────────────────────

class TestPyprojectToml:
    """pyproject.toml must contain minimal required fields."""

    def _read(self) -> str:
        return _path("pyproject.toml").read_text(encoding="utf-8")

    def test_project_name_present(self):
        content = self._read()
        assert "prj55" in content.lower(), "pyproject.toml must reference the project name"

    def test_requires_python_present(self):
        content = self._read()
        assert "requires-python" in content, "pyproject.toml must specify requires-python"

    def test_no_ml_deps_pinned_yet(self):
        """Guard: no heavy ML libraries should be pinned in the bootstrap stage.

        Strips comment lines before checking so that in-file warning comments
        (e.g. '# Do NOT add torch, torchaudio …') do not trigger a false positive.
        """
        raw = self._read()
        # Remove comment lines (lines whose first non-whitespace character is '#')
        non_comment_lines = [
            line for line in raw.splitlines()
            if not line.lstrip().startswith("#")
        ]
        content = "\n".join(non_comment_lines)
        forbidden = ["torch", "torchaudio", "tensorflow", "keras"]
        found = [dep for dep in forbidden if dep in content]
        assert not found, (
            f"ML dependencies should not be pinned at bootstrap stage. Found: {found}"
        )
