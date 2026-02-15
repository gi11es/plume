"""Tests for the PDF form filler."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pypdf import PdfReader

from conftest import (
    ALL_FORMS, ACROFORM_PDFS, GRAPHICAL_PDFS, FIXTURES_DIR,
    run_extract, run_fill, make_spec,
)


# ---------------------------------------------------------------------------
# Overlay fill
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "entry", GRAPHICAL_PDFS, ids=[e["filename"] for e in GRAPHICAL_PDFS]
)
class TestOverlayFill:
    """Test overlay filling on each graphical PDF."""

    def test_overlay_creates_output(self, entry, tmp_output):
        pdf_path = FIXTURES_DIR / entry["filename"]
        output_path = tmp_output / "filled.pdf"

        spec = make_spec(tmp_output, {
            "strategy": "overlay",
            "fields": [
                {"value": "Test Text", "x": 100, "y": 700, "page": 0,
                 "font_size": 10, "type": "text"}
            ],
        })

        result = run_fill(pdf_path, spec, output_path, "overlay")
        assert result["status"] == "success"
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_overlay_preserves_pages(self, entry, tmp_output):
        pdf_path = FIXTURES_DIR / entry["filename"]
        output_path = tmp_output / "filled.pdf"
        original = PdfReader(str(pdf_path))

        spec = make_spec(tmp_output, {
            "strategy": "overlay",
            "fields": [
                {"value": "X", "x": 100, "y": 700, "page": 0,
                 "font_size": 10, "type": "text"}
            ],
        })

        run_fill(pdf_path, spec, output_path, "overlay")
        filled = PdfReader(str(output_path))
        assert len(filled.pages) == len(original.pages)

    def test_overlay_multiple_fields(self, entry, tmp_output):
        pdf_path = FIXTURES_DIR / entry["filename"]
        output_path = tmp_output / "filled.pdf"

        spec = make_spec(tmp_output, {
            "strategy": "overlay",
            "fields": [
                {"value": "Field 1", "x": 100, "y": 700, "page": 0,
                 "font_size": 10, "type": "text"},
                {"value": "Field 2", "x": 100, "y": 680, "page": 0,
                 "font_size": 10, "type": "text"},
                {"value": "Field 3", "x": 100, "y": 660, "page": 0,
                 "font_size": 10, "type": "text"},
            ],
        })

        result = run_fill(pdf_path, spec, output_path, "overlay")
        assert result["fields_filled"] == 3


# ---------------------------------------------------------------------------
# AcroForm fill with readback verification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "entry", ACROFORM_PDFS, ids=[e["filename"] for e in ACROFORM_PDFS]
)
class TestAcroFormFill:
    """Test AcroForm filling on each AcroForm PDF, with readback verification."""

    def _get_text_fields(self, pdf_path):
        """Extract writable text fields from a PDF."""
        from extract import extract_structure
        structure = extract_structure(str(pdf_path))
        return [f for f in structure["acroform_fields"]
                if f["type"] == "text" and not f["readonly"]]

    def test_acroform_creates_output(self, entry, tmp_output):
        pdf_path = FIXTURES_DIR / entry["filename"]
        output_path = tmp_output / "filled.pdf"
        text_fields = self._get_text_fields(pdf_path)

        if not text_fields:
            pytest.skip("No writable text fields")

        spec = make_spec(tmp_output, {
            "strategy": "acroform",
            "fields": [
                {"name": text_fields[0]["name"], "value": "Plume Test", "type": "text"}
            ],
        })

        result = run_fill(pdf_path, spec, output_path, "acroform")
        assert result["status"] == "success"
        assert output_path.exists()

    def test_acroform_preserves_pages(self, entry, tmp_output):
        pdf_path = FIXTURES_DIR / entry["filename"]
        output_path = tmp_output / "filled.pdf"
        original = PdfReader(str(pdf_path))
        text_fields = self._get_text_fields(pdf_path)

        if not text_fields:
            pytest.skip("No writable text fields")

        spec = make_spec(tmp_output, {
            "strategy": "acroform",
            "fields": [
                {"name": text_fields[0]["name"], "value": "Test", "type": "text"}
            ],
        })

        run_fill(pdf_path, spec, output_path, "acroform")
        filled = PdfReader(str(output_path))
        assert len(filled.pages) == len(original.pages)

    def test_acroform_value_persists(self, entry, tmp_output):
        """After filling, read back the PDF and verify the field value is set."""
        pdf_path = FIXTURES_DIR / entry["filename"]
        output_path = tmp_output / "filled.pdf"
        text_fields = self._get_text_fields(pdf_path)

        if not text_fields:
            pytest.skip("No writable text fields")

        field_name = text_fields[0]["name"]
        test_value = "PlumeFillTest_42"

        spec = make_spec(tmp_output, {
            "strategy": "acroform",
            "fields": [
                {"name": field_name, "value": test_value, "type": "text"}
            ],
        })

        run_fill(pdf_path, spec, output_path, "acroform")

        # Read back and verify value was written
        from extract import extract_structure
        filled_struct = extract_structure(str(output_path))
        filled_field = next(
            (f for f in filled_struct["acroform_fields"] if f["name"] == field_name),
            None
        )
        assert filled_field is not None, f"Field '{field_name}' not found in filled PDF"
        assert filled_field["value"] == test_value, (
            f"Field '{field_name}' value is '{filled_field['value']}', expected '{test_value}'"
        )


# ---------------------------------------------------------------------------
# Auto strategy detection
# ---------------------------------------------------------------------------

class TestAutoStrategy:
    def test_auto_picks_acroform(self, tmp_output):
        if not ACROFORM_PDFS:
            pytest.skip("No AcroForm test PDFs")
        entry = ACROFORM_PDFS[0]
        pdf_path = FIXTURES_DIR / entry["filename"]
        output_path = tmp_output / "filled.pdf"

        from extract import extract_structure
        structure = extract_structure(str(pdf_path))
        text_fields = [f for f in structure["acroform_fields"]
                       if f["type"] == "text" and not f["readonly"]]
        if not text_fields:
            pytest.skip("No writable text fields")

        spec = make_spec(tmp_output, {
            "strategy": "auto",
            "fields": [{"name": text_fields[0]["name"], "value": "Test", "type": "text"}],
        })

        result = run_fill(pdf_path, spec, output_path, "auto")
        assert result["strategy"] == "acroform"

    def test_auto_picks_overlay(self, tmp_output):
        if not GRAPHICAL_PDFS:
            pytest.skip("No graphical test PDFs")
        entry = GRAPHICAL_PDFS[0]
        pdf_path = FIXTURES_DIR / entry["filename"]
        output_path = tmp_output / "filled.pdf"

        spec = make_spec(tmp_output, {
            "strategy": "auto",
            "fields": [{"value": "Test", "x": 100, "y": 700, "page": 0,
                         "font_size": 10, "type": "text"}],
        })

        result = run_fill(pdf_path, spec, output_path, "auto")
        assert result["strategy"] == "overlay"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_missing_input_pdf(self, tmp_output):
        spec = make_spec(tmp_output, {"strategy": "overlay", "fields": []})
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent.parent / "scripts" / "fill.py"),
             "/nonexistent.pdf", str(spec), str(tmp_output / "out.pdf")],
            capture_output=True, text=True,
        )
        assert result.returncode != 0

    def test_empty_fields(self, tmp_output):
        if not ALL_FORMS:
            pytest.skip("No test PDFs")
        entry = ALL_FORMS[0]
        pdf_path = FIXTURES_DIR / entry["filename"]
        output_path = tmp_output / "filled.pdf"

        spec = make_spec(tmp_output, {"strategy": "overlay", "fields": []})
        result = run_fill(pdf_path, spec, output_path, "overlay")
        assert result["status"] == "success"
        assert result["fields_filled"] == 0
