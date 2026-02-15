"""Pytest configuration and shared fixtures for plume tests."""

import json
import os
import sys
from pathlib import Path

import pytest

# Add scripts to path
PLUME_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PLUME_ROOT / "scripts"))

FIXTURES_DIR = Path(__file__).parent / "fixtures"
MANIFEST_PATH = FIXTURES_DIR / "MANIFEST.json"


@pytest.fixture
def fixtures_dir():
    return FIXTURES_DIR


@pytest.fixture
def manifest():
    if not MANIFEST_PATH.exists():
        pytest.skip("No test fixtures manifest found")
    with open(MANIFEST_PATH) as f:
        data = json.load(f)
    return data.get("forms", data) if isinstance(data, dict) else data


@pytest.fixture
def acroform_pdfs(manifest):
    """Return list of PDFs that have AcroForm fields."""
    pdfs = [
        entry for entry in manifest
        if entry.get("type") in ("acroform", "mixed")
        and (FIXTURES_DIR / entry["filename"]).exists()
    ]
    if not pdfs:
        pytest.skip("No AcroForm test PDFs available")
    return pdfs


@pytest.fixture
def graphical_pdfs(manifest):
    """Return list of graphical (non-AcroForm) PDFs."""
    pdfs = [
        entry for entry in manifest
        if entry.get("type") == "graphical"
        and (FIXTURES_DIR / entry["filename"]).exists()
    ]
    if not pdfs:
        pytest.skip("No graphical test PDFs available")
    return pdfs


@pytest.fixture
def all_pdfs(manifest):
    """Return all available test PDFs."""
    return [
        entry for entry in manifest
        if (FIXTURES_DIR / entry["filename"]).exists()
    ]


@pytest.fixture
def tmp_output(tmp_path):
    """Provide a temporary output directory."""
    return tmp_path
