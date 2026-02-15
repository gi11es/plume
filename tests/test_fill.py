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
# Signature overlay
# ---------------------------------------------------------------------------

TEST_SIGNATURE = FIXTURES_DIR / "test_signature.png"


class TestSignatureFill:
    """Test signature image overlay integration."""

    def test_signature_overlay_creates_output(self, tmp_output):
        """A signature field in the spec produces a valid filled PDF."""
        if not GRAPHICAL_PDFS:
            pytest.skip("No graphical test PDFs")
        entry = GRAPHICAL_PDFS[0]
        pdf_path = FIXTURES_DIR / entry["filename"]
        output_path = tmp_output / "signed.pdf"

        spec = make_spec(tmp_output, {
            "strategy": "overlay",
            "fields": [
                {"type": "signature", "image_path": str(TEST_SIGNATURE),
                 "x": 100, "y": 200, "page": 0, "width": 150, "height": 50},
            ],
        })

        result = run_fill(pdf_path, spec, output_path, "overlay")
        assert result["status"] == "success"
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_signature_preserves_pages(self, tmp_output):
        """Signature overlay doesn't add or remove pages."""
        if not GRAPHICAL_PDFS:
            pytest.skip("No graphical test PDFs")
        entry = GRAPHICAL_PDFS[0]
        pdf_path = FIXTURES_DIR / entry["filename"]
        output_path = tmp_output / "signed.pdf"
        original = PdfReader(str(pdf_path))

        spec = make_spec(tmp_output, {
            "strategy": "overlay",
            "fields": [
                {"type": "signature", "image_path": str(TEST_SIGNATURE),
                 "x": 100, "y": 200, "page": 0, "width": 150, "height": 50},
            ],
        })

        run_fill(pdf_path, spec, output_path, "overlay")
        filled = PdfReader(str(output_path))
        assert len(filled.pages) == len(original.pages)

    def test_signature_with_text_fields(self, tmp_output):
        """Signature field works alongside regular text fields."""
        if not GRAPHICAL_PDFS:
            pytest.skip("No graphical test PDFs")
        entry = GRAPHICAL_PDFS[0]
        pdf_path = FIXTURES_DIR / entry["filename"]
        output_path = tmp_output / "signed.pdf"

        spec = make_spec(tmp_output, {
            "strategy": "overlay",
            "fields": [
                {"value": "Sophie Martin", "x": 100, "y": 700, "page": 0,
                 "font_size": 10, "type": "text"},
                {"type": "signature", "image_path": str(TEST_SIGNATURE),
                 "x": 100, "y": 200, "page": 0, "width": 150, "height": 50},
                {"value": "02/15/2026", "x": 300, "y": 200, "page": 0,
                 "font_size": 10, "type": "text"},
            ],
        })

        result = run_fill(pdf_path, spec, output_path, "overlay")
        assert result["fields_filled"] == 3
        assert output_path.stat().st_size > 0

    def test_signature_in_both_strategy(self, tmp_output):
        """Signature works in 'both' strategy (AcroForm + overlay with signature)."""
        if not ACROFORM_PDFS:
            pytest.skip("No AcroForm test PDFs")
        entry = ACROFORM_PDFS[0]
        pdf_path = FIXTURES_DIR / entry["filename"]
        output_path = tmp_output / "signed.pdf"

        from extract import extract_structure
        structure = extract_structure(str(pdf_path))
        text_fields = [f for f in structure["acroform_fields"]
                       if f["type"] == "text" and not f["readonly"]]
        if not text_fields:
            pytest.skip("No writable text fields")

        spec = make_spec(tmp_output, {
            "strategy": "both",
            "fields": [
                {"name": text_fields[0]["name"], "value": "Plume Test", "type": "text"},
                {"type": "signature", "image_path": str(TEST_SIGNATURE),
                 "x": 100, "y": 200, "page": 0, "width": 150, "height": 50},
            ],
        })

        result = run_fill(pdf_path, spec, output_path, "both")
        assert result["status"] == "success"
        assert result["strategy"] == "both"

    def test_missing_signature_image_skipped(self, tmp_output):
        """A signature field with a non-existent image path is silently skipped."""
        if not GRAPHICAL_PDFS:
            pytest.skip("No graphical test PDFs")
        entry = GRAPHICAL_PDFS[0]
        pdf_path = FIXTURES_DIR / entry["filename"]
        output_path = tmp_output / "signed.pdf"

        spec = make_spec(tmp_output, {
            "strategy": "overlay",
            "fields": [
                {"type": "signature", "image_path": "/nonexistent/sig.png",
                 "x": 100, "y": 200, "page": 0, "width": 150, "height": 50},
            ],
        })

        result = run_fill(pdf_path, spec, output_path, "overlay")
        assert result["status"] == "success"

    def test_signature_png_is_transparent(self):
        """Verify the test signature has a transparent background."""
        from PIL import Image
        img = Image.open(str(TEST_SIGNATURE))
        assert img.mode == "RGBA", f"Expected RGBA, got {img.mode}"
        # Check that some pixels are fully transparent
        pixels = list(img.get_flattened_data())
        transparent = [p for p in pixels if p[3] == 0]
        assert len(transparent) > 0, "Signature should have transparent pixels"
        # Check that some pixels are black (ink)
        black = [p for p in pixels if p[0] < 50 and p[1] < 50 and p[2] < 50 and p[3] > 200]
        assert len(black) > 0, "Signature should have black ink pixels"


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
