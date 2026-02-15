"""Pytest configuration and shared fixtures for plume tests."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Add scripts to path
PLUME_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PLUME_ROOT / "scripts"))

SCRIPTS = PLUME_ROOT / "scripts"
FIXTURES_DIR = Path(__file__).parent / "fixtures"
MANIFEST_PATH = FIXTURES_DIR / "MANIFEST.json"


def _load_manifest():
    if not MANIFEST_PATH.exists():
        return []
    with open(MANIFEST_PATH) as f:
        data = json.load(f)
    forms = data.get("forms", data) if isinstance(data, dict) else data
    # Only include PDFs that actually exist on disk
    return [e for e in forms if (FIXTURES_DIR / e["filename"]).exists()]


ALL_FORMS = _load_manifest()
ACROFORM_PDFS = [e for e in ALL_FORMS if e.get("type") in ("acroform", "mixed", "both")]
GRAPHICAL_PDFS = [e for e in ALL_FORMS if e.get("type") == "graphical"]


@pytest.fixture
def fixtures_dir():
    return FIXTURES_DIR


@pytest.fixture
def tmp_output(tmp_path):
    return tmp_path


# --- Helpers used across test files ---

def run_extract(pdf_path, extra_args=None):
    """Run extract.py and return parsed JSON output."""
    cmd = [sys.executable, str(SCRIPTS / "extract.py"), str(pdf_path), "--pretty"]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"extract.py failed on {pdf_path}:\n{result.stderr}"
    return json.loads(result.stdout)


def run_fill(input_pdf, spec_path, output_pdf, strategy="auto"):
    """Run fill.py and return parsed JSON output."""
    cmd = [
        sys.executable, str(SCRIPTS / "fill.py"),
        str(input_pdf), str(spec_path), str(output_pdf),
        "--strategy", strategy,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"fill.py failed:\n{result.stderr}"
    return json.loads(result.stdout)


def run_verify(filled_pdf, spec_path, tolerance=5):
    """Run verify.py and return (report, exitcode)."""
    cmd = [
        sys.executable, str(SCRIPTS / "verify.py"),
        str(filled_pdf), str(spec_path),
        "--tolerance", str(tolerance), "--pretty",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return json.loads(result.stdout), result.returncode


def make_spec(tmp_path, spec_data):
    """Write a fill spec JSON and return its path."""
    spec_path = tmp_path / "fill_spec.json"
    spec_path.write_text(json.dumps(spec_data))
    return spec_path
