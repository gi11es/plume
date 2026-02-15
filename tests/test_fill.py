"""Tests for the PDF form filler."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pypdf import PdfReader

SCRIPTS = Path(__file__).parent.parent / "scripts"


def run_fill(input_pdf, spec, output_pdf, strategy="auto"):
    """Run fill.py and return parsed JSON output."""
    cmd = [
        sys.executable, str(SCRIPTS / "fill.py"),
        str(input_pdf), str(spec), str(output_pdf),
        "--strategy", strategy,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"fill.py failed: {result.stderr}"
    return json.loads(result.stdout)


def make_spec(tmp_path, spec_data):
    """Write a fill spec to a temp file and return its path."""
    spec_path = tmp_path / "fill_spec.json"
    spec_path.write_text(json.dumps(spec_data))
    return spec_path


class TestOverlayFill:
    """Test overlay (graphical) filling."""

    def test_overlay_creates_output(self, graphical_pdfs, fixtures_dir, tmp_output):
        entry = graphical_pdfs[0]
        pdf_path = fixtures_dir / entry["filename"]
        output_path = tmp_output / "filled.pdf"

        spec = make_spec(tmp_output, {
            "strategy": "overlay",
            "fields": [
                {"value": "Test Text", "x": 100, "y": 700, "page": 0, "font_size": 10, "type": "text"}
            ],
        })

        result = run_fill(pdf_path, spec, output_path, "overlay")
        assert result["status"] == "success"
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_overlay_preserves_pages(self, graphical_pdfs, fixtures_dir, tmp_output):
        entry = graphical_pdfs[0]
        pdf_path = fixtures_dir / entry["filename"]
        output_path = tmp_output / "filled.pdf"

        original = PdfReader(str(pdf_path))
        original_pages = len(original.pages)

        spec = make_spec(tmp_output, {
            "strategy": "overlay",
            "fields": [
                {"value": "Test", "x": 100, "y": 700, "page": 0, "font_size": 10, "type": "text"}
            ],
        })

        run_fill(pdf_path, spec, output_path, "overlay")
        filled = PdfReader(str(output_path))
        assert len(filled.pages) == original_pages

    def test_overlay_checkbox(self, graphical_pdfs, fixtures_dir, tmp_output):
        entry = graphical_pdfs[0]
        pdf_path = fixtures_dir / entry["filename"]
        output_path = tmp_output / "filled.pdf"

        spec = make_spec(tmp_output, {
            "strategy": "overlay",
            "fields": [
                {"value": "X", "x": 140, "y": 565, "page": 0, "font_size": 10,
                 "font": "Helvetica-Bold", "type": "checkbox"}
            ],
        })

        result = run_fill(pdf_path, spec, output_path, "overlay")
        assert result["status"] == "success"

    def test_overlay_multiple_fields(self, graphical_pdfs, fixtures_dir, tmp_output):
        entry = graphical_pdfs[0]
        pdf_path = fixtures_dir / entry["filename"]
        output_path = tmp_output / "filled.pdf"

        spec = make_spec(tmp_output, {
            "strategy": "overlay",
            "fields": [
                {"value": "Field 1", "x": 100, "y": 700, "page": 0, "font_size": 10, "type": "text"},
                {"value": "Field 2", "x": 100, "y": 680, "page": 0, "font_size": 10, "type": "text"},
                {"value": "Field 3", "x": 100, "y": 660, "page": 0, "font_size": 10, "type": "text"},
            ],
        })

        result = run_fill(pdf_path, spec, output_path, "overlay")
        assert result["fields_filled"] == 3


class TestAcroFormFill:
    """Test AcroForm field filling."""

    def test_acroform_creates_output(self, acroform_pdfs, fixtures_dir, tmp_output):
        entry = acroform_pdfs[0]
        pdf_path = fixtures_dir / entry["filename"]
        output_path = tmp_output / "filled.pdf"

        # Get actual field names
        from extract import extract_structure
        structure = extract_structure(str(pdf_path))
        text_fields = [f for f in structure["acroform_fields"] if f["type"] == "text" and not f["readonly"]]

        if not text_fields:
            pytest.skip("No writable text fields in test PDF")

        spec = make_spec(tmp_output, {
            "strategy": "acroform",
            "fields": [
                {"name": text_fields[0]["name"], "value": "Test Value", "type": "text"}
            ],
        })

        result = run_fill(pdf_path, spec, output_path, "acroform")
        assert result["status"] == "success"
        assert output_path.exists()

    def test_acroform_preserves_pages(self, acroform_pdfs, fixtures_dir, tmp_output):
        entry = acroform_pdfs[0]
        pdf_path = fixtures_dir / entry["filename"]
        output_path = tmp_output / "filled.pdf"

        original = PdfReader(str(pdf_path))
        original_pages = len(original.pages)

        from extract import extract_structure
        structure = extract_structure(str(pdf_path))
        text_fields = [f for f in structure["acroform_fields"] if f["type"] == "text" and not f["readonly"]]

        if not text_fields:
            pytest.skip("No writable text fields")

        spec = make_spec(tmp_output, {
            "strategy": "acroform",
            "fields": [{"name": text_fields[0]["name"], "value": "Test", "type": "text"}],
        })

        run_fill(pdf_path, spec, output_path, "acroform")
        filled = PdfReader(str(output_path))
        assert len(filled.pages) == original_pages


class TestAutoStrategy:
    """Test automatic strategy detection."""

    def test_auto_picks_acroform(self, acroform_pdfs, fixtures_dir, tmp_output):
        entry = acroform_pdfs[0]
        pdf_path = fixtures_dir / entry["filename"]
        output_path = tmp_output / "filled.pdf"

        from extract import extract_structure
        structure = extract_structure(str(pdf_path))
        text_fields = [f for f in structure["acroform_fields"] if f["type"] == "text" and not f["readonly"]]

        if not text_fields:
            pytest.skip("No writable text fields")

        spec = make_spec(tmp_output, {
            "strategy": "auto",
            "fields": [{"name": text_fields[0]["name"], "value": "Test", "type": "text"}],
        })

        result = run_fill(pdf_path, spec, output_path, "auto")
        assert result["strategy"] == "acroform"

    def test_auto_picks_overlay(self, graphical_pdfs, fixtures_dir, tmp_output):
        entry = graphical_pdfs[0]
        pdf_path = fixtures_dir / entry["filename"]
        output_path = tmp_output / "filled.pdf"

        spec = make_spec(tmp_output, {
            "strategy": "auto",
            "fields": [{"value": "Test", "x": 100, "y": 700, "page": 0, "font_size": 10, "type": "text"}],
        })

        result = run_fill(pdf_path, spec, output_path, "auto")
        assert result["strategy"] == "overlay"


class TestErrorHandling:
    """Test error cases for fill.py."""

    def test_missing_input_pdf(self, tmp_output):
        spec = make_spec(tmp_output, {"strategy": "overlay", "fields": []})
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / "fill.py"), "/nonexistent.pdf", str(spec), str(tmp_output / "out.pdf")],
            capture_output=True, text=True,
        )
        assert result.returncode != 0

    def test_empty_fields(self, all_pdfs, fixtures_dir, tmp_output):
        entry = all_pdfs[0]
        pdf_path = fixtures_dir / entry["filename"]
        output_path = tmp_output / "filled.pdf"

        spec = make_spec(tmp_output, {"strategy": "overlay", "fields": []})
        result = run_fill(pdf_path, spec, output_path, "overlay")
        assert result["status"] == "success"
        assert result["fields_filled"] == 0
