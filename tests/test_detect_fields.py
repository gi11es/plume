"""Tests for the graphical PDF field detection system."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import (
    GRAPHICAL_PDFS, FIXTURES_DIR, SCRIPTS, PLUME_ROOT,
)


def run_detect_fields(pdf_path, extra_args=None):
    """Run detect_fields.py and return parsed JSON output."""
    cmd = [sys.executable, str(SCRIPTS / "detect_fields.py"), str(pdf_path), "--pretty"]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"detect_fields.py failed on {pdf_path}:\n{result.stderr}"
    return json.loads(result.stdout)


# ---------------------------------------------------------------------------
# Basic detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "entry", GRAPHICAL_PDFS, ids=[e["filename"] for e in GRAPHICAL_PDFS]
)
class TestDetectFieldsBasics:
    """Basic tests for field detection on all graphical PDFs."""

    def test_returns_valid_json(self, entry):
        pdf_path = FIXTURES_DIR / entry["filename"]
        result = run_detect_fields(pdf_path)
        assert "fields" in result
        assert "page" in result
        assert "page_size" in result
        assert "detection" in result
        assert "total_fields" in result

    def test_detects_some_fields(self, entry):
        pdf_path = FIXTURES_DIR / entry["filename"]
        result = run_detect_fields(pdf_path)
        assert result["total_fields"] > 0, f"No fields detected in {entry['filename']}"

    def test_page_size_sane(self, entry):
        pdf_path = FIXTURES_DIR / entry["filename"]
        result = run_detect_fields(pdf_path)
        ps = result["page_size"]
        assert ps["width"] > 50 and ps["width"] < 2000
        assert ps["height"] > 50 and ps["height"] < 2000

    def test_field_types_valid(self, entry):
        pdf_path = FIXTURES_DIR / entry["filename"]
        result = run_detect_fields(pdf_path)
        valid_types = {"text", "checkbox"}
        for field in result["fields"]:
            assert field["field_type"] in valid_types

    def test_fill_points_within_page(self, entry):
        pdf_path = FIXTURES_DIR / entry["filename"]
        result = run_detect_fields(pdf_path)
        ps = result["page_size"]
        for field in result["fields"]:
            fp = field["fill_point"]
            assert 0 <= fp["x"] <= ps["width"] + 5, (
                f"Fill point x={fp['x']} out of page width {ps['width']} "
                f"for field '{field.get('label', '?')}'"
            )
            assert 0 <= fp["y"] <= ps["height"] + 5, (
                f"Fill point y={fp['y']} out of page height {ps['height']} "
                f"for field '{field.get('label', '?')}'"
            )

    def test_labels_not_empty(self, entry):
        pdf_path = FIXTURES_DIR / entry["filename"]
        result = run_detect_fields(pdf_path)
        labeled = [f for f in result["fields"] if f.get("label")]
        assert len(labeled) > 0, "No fields have labels"

    def test_confidence_present(self, entry):
        pdf_path = FIXTURES_DIR / entry["filename"]
        result = run_detect_fields(pdf_path)
        assert result["confidence"] in ("high", "medium", "low")


# ---------------------------------------------------------------------------
# Specific PDF detection quality
# ---------------------------------------------------------------------------

class TestEUSchengenDetection:
    """EU Schengen visa should detect many fields and checkboxes."""

    def test_finds_text_fields(self):
        pdf_path = FIXTURES_DIR / "eu_schengen_visa_graphical.pdf"
        if not pdf_path.exists():
            pytest.skip("EU Schengen visa PDF not available")
        result = run_detect_fields(pdf_path)
        assert result["text_fields"] >= 30, (
            f"Expected >= 30 text fields, got {result['text_fields']}"
        )

    def test_finds_checkboxes(self):
        pdf_path = FIXTURES_DIR / "eu_schengen_visa_graphical.pdf"
        if not pdf_path.exists():
            pytest.skip("EU Schengen visa PDF not available")
        result = run_detect_fields(pdf_path)
        assert result["checkbox_fields"] >= 20, (
            f"Expected >= 20 checkboxes, got {result['checkbox_fields']}"
        )

    def test_finds_surname_field(self):
        pdf_path = FIXTURES_DIR / "eu_schengen_visa_graphical.pdf"
        if not pdf_path.exists():
            pytest.skip("EU Schengen visa PDF not available")
        result = run_detect_fields(pdf_path)
        labels = [f["label"].lower() for f in result["fields"]]
        assert any("surname" in lbl for lbl in labels), (
            "Expected to find a field with 'surname' in its label"
        )


class TestAUPassengerCardDetection:
    """AU passenger card should detect character boxes and checkboxes."""

    def test_finds_text_fields(self):
        pdf_path = FIXTURES_DIR / "au_incoming_passenger_graphical.pdf"
        if not pdf_path.exists():
            pytest.skip("AU passenger card PDF not available")
        result = run_detect_fields(pdf_path)
        assert result["text_fields"] >= 5, (
            f"Expected >= 5 text fields, got {result['text_fields']}"
        )

    def test_finds_checkboxes(self):
        pdf_path = FIXTURES_DIR / "au_incoming_passenger_graphical.pdf"
        if not pdf_path.exists():
            pytest.skip("AU passenger card PDF not available")
        result = run_detect_fields(pdf_path)
        assert result["checkbox_fields"] >= 10, (
            f"Expected >= 10 checkboxes, got {result['checkbox_fields']}"
        )

    def test_detects_character_boxes(self):
        pdf_path = FIXTURES_DIR / "au_incoming_passenger_graphical.pdf"
        if not pdf_path.exists():
            pytest.skip("AU passenger card PDF not available")
        result = run_detect_fields(pdf_path)
        char_box_fields = [f for f in result["fields"] if f.get("char_boxes")]
        assert len(char_box_fields) >= 1, (
            "Expected at least 1 character-box group field"
        )

    def test_has_colored_background_detection(self):
        pdf_path = FIXTURES_DIR / "au_incoming_passenger_graphical.pdf"
        if not pdf_path.exists():
            pytest.skip("AU passenger card PDF not available")
        result = run_detect_fields(pdf_path)
        # The AU card uses filled rectangles (white boxes on colored bg)
        assert result["detection"]["filled_rects"] > 0, (
            "Expected filled rectangle detection on AU passenger card"
        )


class TestJPCustomsDetection:
    """JP customs declaration should detect grid-based fields."""

    def test_finds_grid_cells(self):
        pdf_path = FIXTURES_DIR / "jp_customs_declaration_graphical.pdf"
        if not pdf_path.exists():
            pytest.skip("JP customs PDF not available")
        result = run_detect_fields(pdf_path)
        assert result["detection"]["grid_cells"] >= 20, (
            f"Expected >= 20 grid cells, got {result['detection']['grid_cells']}"
        )

    def test_finds_text_fields(self):
        pdf_path = FIXTURES_DIR / "jp_customs_declaration_graphical.pdf"
        if not pdf_path.exists():
            pytest.skip("JP customs PDF not available")
        result = run_detect_fields(pdf_path)
        assert result["text_fields"] >= 15, (
            f"Expected >= 15 text fields, got {result['text_fields']}"
        )


# ---------------------------------------------------------------------------
# Multi-page detection
# ---------------------------------------------------------------------------

class TestMultiPageDetection:
    """Test multi-page detection."""

    def test_all_pages_mode(self):
        if not GRAPHICAL_PDFS:
            pytest.skip("No graphical PDFs available")
        entry = GRAPHICAL_PDFS[0]
        pdf_path = FIXTURES_DIR / entry["filename"]
        result = run_detect_fields(pdf_path, ["--all-pages"])
        assert "pages" in result
        assert "num_pages" in result
        assert result["num_pages"] >= 1
        assert len(result["pages"]) == result["num_pages"]


# ---------------------------------------------------------------------------
# Grid overlay
# ---------------------------------------------------------------------------

class TestGridOverlay:
    """Test grid overlay generation."""

    def test_generates_grid_overlay(self, tmp_path):
        if not GRAPHICAL_PDFS:
            pytest.skip("No graphical PDFs available")
        entry = GRAPHICAL_PDFS[0]
        pdf_path = FIXTURES_DIR / entry["filename"]
        output_png = tmp_path / "grid.png"
        cmd = [
            sys.executable, str(SCRIPTS / "detect_fields.py"),
            str(pdf_path), "--grid-overlay", str(output_png),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0
        assert output_png.exists()
        assert output_png.stat().st_size > 0


# ---------------------------------------------------------------------------
# Annotated image
# ---------------------------------------------------------------------------

class TestAnnotatedImage:
    """Test annotated image generation."""

    def test_generates_annotated_image(self, tmp_path):
        if not GRAPHICAL_PDFS:
            pytest.skip("No graphical PDFs available")
        entry = GRAPHICAL_PDFS[0]
        pdf_path = FIXTURES_DIR / entry["filename"]
        output_png = tmp_path / "annotated.png"
        cmd = [
            sys.executable, str(SCRIPTS / "detect_fields.py"),
            str(pdf_path), "--annotate", str(output_png), "--pretty",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0
        assert output_png.exists()
        assert output_png.stat().st_size > 0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestDetectFieldsErrors:
    """Test error handling."""

    def test_nonexistent_pdf(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / "detect_fields.py"), "/nonexistent.pdf"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
