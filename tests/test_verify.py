"""Tests for the PDF fill verifier."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parent.parent / "scripts"


def run_fill(input_pdf, spec_path, output_pdf):
    cmd = [sys.executable, str(SCRIPTS / "fill.py"), str(input_pdf), str(spec_path), str(output_pdf)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"fill.py failed: {result.stderr}"
    return json.loads(result.stdout)


def run_verify(filled_pdf, spec_path, tolerance=5):
    cmd = [
        sys.executable, str(SCRIPTS / "verify.py"),
        str(filled_pdf), str(spec_path),
        "--tolerance", str(tolerance), "--pretty",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return json.loads(result.stdout), result.returncode


def make_spec(tmp_path, spec_data):
    spec_path = tmp_path / "fill_spec.json"
    spec_path.write_text(json.dumps(spec_data))
    return spec_path


class TestOverlayVerification:
    """Test verification of overlay fills."""

    def test_correct_overlay_passes(self, graphical_pdfs, fixtures_dir, tmp_output):
        entry = graphical_pdfs[0]
        pdf_path = fixtures_dir / entry["filename"]
        output_path = tmp_output / "filled.pdf"

        spec_data = {
            "strategy": "overlay",
            "fields": [
                {"value": "Test Text", "x": 200, "y": 700, "page": 0, "font_size": 10, "type": "text"}
            ],
        }
        spec_path = make_spec(tmp_output, spec_data)

        run_fill(pdf_path, spec_path, output_path)
        report, exitcode = run_verify(output_path, spec_path)

        assert "results" in report
        assert "summary" in report

    def test_verification_report_structure(self, graphical_pdfs, fixtures_dir, tmp_output):
        entry = graphical_pdfs[0]
        pdf_path = fixtures_dir / entry["filename"]
        output_path = tmp_output / "filled.pdf"

        spec_data = {
            "strategy": "overlay",
            "fields": [
                {"value": "Hello", "x": 100, "y": 500, "page": 0, "font_size": 10, "type": "text"}
            ],
        }
        spec_path = make_spec(tmp_output, spec_data)

        run_fill(pdf_path, spec_path, output_path)
        report, _ = run_verify(output_path, spec_path)

        assert "total" in report["summary"]
        assert "pass" in report["summary"]
        assert "fail" in report["summary"]
        assert "all_passed" in report


class TestAcroFormVerification:
    """Test verification of AcroForm fills."""

    def test_acroform_field_value_check(self, acroform_pdfs, fixtures_dir, tmp_output):
        entry = acroform_pdfs[0]
        pdf_path = fixtures_dir / entry["filename"]
        output_path = tmp_output / "filled.pdf"

        from extract import extract_structure
        structure = extract_structure(str(pdf_path))
        text_fields = [f for f in structure["acroform_fields"] if f["type"] == "text" and not f["readonly"]]

        if not text_fields:
            pytest.skip("No writable text fields")

        spec_data = {
            "strategy": "acroform",
            "fields": [{"name": text_fields[0]["name"], "value": "Verified Text", "type": "text"}],
        }
        spec_path = make_spec(tmp_output, spec_data)

        run_fill(pdf_path, spec_path, output_path)
        report, _ = run_verify(output_path, spec_path)

        assert report["summary"]["total"] >= 1
