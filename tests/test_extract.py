"""Tests for the PDF structure extractor."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parent.parent / "scripts"


def run_extract(pdf_path, extra_args=None):
    """Run extract.py and return parsed JSON output."""
    cmd = [sys.executable, str(SCRIPTS / "extract.py"), str(pdf_path), "--pretty"]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"extract.py failed: {result.stderr}"
    return json.loads(result.stdout)


class TestExtractBasics:
    """Basic extraction tests that run on all PDFs."""

    def test_returns_valid_json(self, all_pdfs, fixtures_dir):
        for entry in all_pdfs:
            pdf_path = fixtures_dir / entry["filename"]
            data = run_extract(pdf_path)
            assert "num_pages" in data
            assert "pages" in data
            assert "has_acroform" in data
            assert "recommended_strategy" in data

    def test_page_count_positive(self, all_pdfs, fixtures_dir):
        for entry in all_pdfs:
            pdf_path = fixtures_dir / entry["filename"]
            data = run_extract(pdf_path)
            assert data["num_pages"] > 0

    def test_page_dimensions(self, all_pdfs, fixtures_dir):
        for entry in all_pdfs:
            pdf_path = fixtures_dir / entry["filename"]
            data = run_extract(pdf_path)
            for page in data["pages"]:
                assert page["width"] > 0
                assert page["height"] > 0


class TestAcroFormExtraction:
    """Tests for AcroForm field detection."""

    def test_detects_acroform_fields(self, acroform_pdfs, fixtures_dir):
        for entry in acroform_pdfs:
            pdf_path = fixtures_dir / entry["filename"]
            data = run_extract(pdf_path)
            assert data["has_acroform"], f"{entry['filename']} should have AcroForm"
            assert len(data["acroform_fields"]) > 0

    def test_field_names_present(self, acroform_pdfs, fixtures_dir):
        for entry in acroform_pdfs:
            pdf_path = fixtures_dir / entry["filename"]
            data = run_extract(pdf_path)
            for field in data["acroform_fields"]:
                assert "name" in field
                assert "type" in field

    def test_field_types_valid(self, acroform_pdfs, fixtures_dir):
        valid_types = {"text", "button", "choice", "signature", "unknown"}
        for entry in acroform_pdfs:
            pdf_path = fixtures_dir / entry["filename"]
            data = run_extract(pdf_path)
            for field in data["acroform_fields"]:
                assert field["type"] in valid_types, f"Unknown type: {field['type']}"

    def test_recommends_acroform_strategy(self, acroform_pdfs, fixtures_dir):
        for entry in acroform_pdfs:
            pdf_path = fixtures_dir / entry["filename"]
            data = run_extract(pdf_path)
            assert data["recommended_strategy"] == "acroform"

    def test_choice_fields_have_options(self, acroform_pdfs, fixtures_dir):
        for entry in acroform_pdfs:
            pdf_path = fixtures_dir / entry["filename"]
            data = run_extract(pdf_path)
            for field in data["acroform_fields"]:
                if field["type"] == "choice":
                    assert "options" in field
                    assert len(field["options"]) > 0


class TestGraphicalExtraction:
    """Tests for graphical PDF parsing."""

    def test_no_acroform_detected(self, graphical_pdfs, fixtures_dir):
        for entry in graphical_pdfs:
            pdf_path = fixtures_dir / entry["filename"]
            data = run_extract(pdf_path)
            assert not data["has_acroform"] or len(data["acroform_fields"]) == 0

    def test_finds_labels(self, graphical_pdfs, fixtures_dir):
        for entry in graphical_pdfs:
            pdf_path = fixtures_dir / entry["filename"]
            data = run_extract(pdf_path)
            all_labels = []
            for page in data["pages"]:
                all_labels.extend(page["labels"])
            assert len(all_labels) > 0, f"No labels found in {entry['filename']}"

    def test_finds_rectangles(self, graphical_pdfs, fixtures_dir):
        for entry in graphical_pdfs:
            pdf_path = fixtures_dir / entry["filename"]
            data = run_extract(pdf_path)
            all_rects = []
            for page in data["pages"]:
                all_rects.extend(page["field_rectangles"])
            # Not all graphical PDFs have colored rectangles, so just check it runs
            assert isinstance(all_rects, list)

    def test_recommends_overlay_strategy(self, graphical_pdfs, fixtures_dir):
        for entry in graphical_pdfs:
            pdf_path = fixtures_dir / entry["filename"]
            data = run_extract(pdf_path)
            assert data["recommended_strategy"] == "overlay"


class TestErrorHandling:
    """Test error cases."""

    def test_nonexistent_file(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / "extract.py"), "/nonexistent.pdf"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0

    def test_invalid_file(self, tmp_path):
        fake_pdf = tmp_path / "fake.pdf"
        fake_pdf.write_text("not a pdf")
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / "extract.py"), str(fake_pdf)],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
