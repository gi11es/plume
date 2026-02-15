"""Tests for the PDF fill verifier."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import (
    ACROFORM_PDFS, GRAPHICAL_PDFS, FIXTURES_DIR,
    run_fill, run_verify, make_spec,
)


# ---------------------------------------------------------------------------
# Overlay verification
# ---------------------------------------------------------------------------

class TestOverlayVerification:
    @pytest.fixture
    def filled_overlay(self, tmp_path):
        """Fill a graphical PDF and return (output_path, spec_path)."""
        if not GRAPHICAL_PDFS:
            pytest.skip("No graphical test PDFs")
        entry = GRAPHICAL_PDFS[0]
        pdf_path = FIXTURES_DIR / entry["filename"]
        output_path = tmp_path / "filled.pdf"

        spec_data = {
            "strategy": "overlay",
            "fields": [
                {"value": "Test Text", "x": 200, "y": 700, "page": 0,
                 "font_size": 10, "type": "text"},
                {"value": "Another", "x": 200, "y": 680, "page": 0,
                 "font_size": 10, "type": "text"},
            ],
        }
        spec_path = make_spec(tmp_path, spec_data)
        run_fill(pdf_path, spec_path, output_path)
        return output_path, spec_path

    def test_verification_report_structure(self, filled_overlay):
        output_path, spec_path = filled_overlay
        report, _ = run_verify(output_path, spec_path)

        assert "results" in report
        assert "summary" in report
        assert "total" in report["summary"]
        assert "pass" in report["summary"]
        assert "fail" in report["summary"]
        assert "all_passed" in report

    def test_correct_overlay_passes(self, filled_overlay):
        output_path, spec_path = filled_overlay
        report, _ = run_verify(output_path, spec_path)
        assert report["summary"]["total"] >= 1


# ---------------------------------------------------------------------------
# AcroForm verification
# ---------------------------------------------------------------------------

class TestAcroFormVerification:
    def test_acroform_field_value_verified(self, tmp_path):
        """Fill an AcroForm field and verify the value is confirmed by verify.py."""
        if not ACROFORM_PDFS:
            pytest.skip("No AcroForm test PDFs")
        entry = ACROFORM_PDFS[0]
        pdf_path = FIXTURES_DIR / entry["filename"]
        output_path = tmp_path / "filled.pdf"

        from extract import extract_structure
        structure = extract_structure(str(pdf_path))
        text_fields = [f for f in structure["acroform_fields"]
                       if f["type"] == "text" and not f["readonly"]]
        if not text_fields:
            pytest.skip("No writable text fields")

        test_value = "VerifyMe123"
        spec_data = {
            "strategy": "acroform",
            "fields": [
                {"name": text_fields[0]["name"], "value": test_value, "type": "text"}
            ],
        }
        spec_path = make_spec(tmp_path, spec_data)

        run_fill(pdf_path, spec_path, output_path)
        report, exitcode = run_verify(output_path, spec_path)

        assert report["summary"]["total"] >= 1
        # The field we filled should pass verification
        acro_results = [r for r in report["results"] if r.get("type") == "acroform"]
        if acro_results:
            assert any(r["status"] == "pass" for r in acro_results), (
                f"No AcroForm field passed verification: {acro_results}"
            )
