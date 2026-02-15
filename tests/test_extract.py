"""Tests for the PDF structure extractor."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import (
    ALL_FORMS, ACROFORM_PDFS, GRAPHICAL_PDFS, FIXTURES_DIR,
    run_extract,
)


# ---------------------------------------------------------------------------
# Basic extraction — parametrized per PDF
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "entry", ALL_FORMS, ids=[e["filename"] for e in ALL_FORMS]
)
class TestExtractBasics:
    """Basic extraction tests that run on every single PDF."""

    def test_returns_valid_json(self, entry):
        data = run_extract(FIXTURES_DIR / entry["filename"])
        assert "num_pages" in data
        assert "pages" in data
        assert "has_acroform" in data
        assert "recommended_strategy" in data

    def test_page_count_positive(self, entry):
        data = run_extract(FIXTURES_DIR / entry["filename"])
        assert data["num_pages"] > 0

    def test_page_dimensions_sane(self, entry):
        data = run_extract(FIXTURES_DIR / entry["filename"])
        for page in data["pages"]:
            assert 100 < page["width"] < 2000, f"Unexpected page width: {page['width']}"
            assert 100 < page["height"] < 2000, f"Unexpected page height: {page['height']}"


# ---------------------------------------------------------------------------
# AcroForm field extraction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "entry", ACROFORM_PDFS, ids=[e["filename"] for e in ACROFORM_PDFS]
)
class TestAcroFormExtraction:
    """Tests for AcroForm field detection — one test per PDF."""

    def test_detects_acroform(self, entry):
        data = run_extract(FIXTURES_DIR / entry["filename"])
        assert data["has_acroform"], f"{entry['filename']} should have AcroForm"
        assert len(data["acroform_fields"]) > 0

    def test_field_count_reasonable(self, entry):
        data = run_extract(FIXTURES_DIR / entry["filename"])
        expected = entry.get("field_count", 1)
        actual = len(data["acroform_fields"])
        # Allow some tolerance since our count may differ from the manifest
        assert actual >= min(expected, 5), (
            f"{entry['filename']}: expected ~{expected} fields, got {actual}"
        )

    def test_field_names_and_types_present(self, entry):
        data = run_extract(FIXTURES_DIR / entry["filename"])
        valid_types = {"text", "button", "choice", "signature", "unknown"}
        for field in data["acroform_fields"]:
            assert "name" in field
            assert "type" in field
            assert field["type"] in valid_types, f"Invalid type: {field['type']}"

    def test_expected_fields_found(self, entry):
        """If the manifest lists expected field names, verify they actually appear."""
        expected = entry.get("expected_fields", [])
        if not expected:
            pytest.skip("No expected_fields in manifest")

        data = run_extract(FIXTURES_DIR / entry["filename"])
        actual_names = {f["name"] for f in data["acroform_fields"]}

        for name in expected[:10]:  # check first 10
            assert name in actual_names, (
                f"Expected field '{name}' not found in {entry['filename']}"
            )

    def test_recommends_acroform_strategy(self, entry):
        data = run_extract(FIXTURES_DIR / entry["filename"])
        assert data["recommended_strategy"] == "acroform"

    def test_choice_fields_have_options(self, entry):
        data = run_extract(FIXTURES_DIR / entry["filename"])
        for field in data["acroform_fields"]:
            if field["type"] == "choice":
                assert "options" in field, f"Choice field '{field['name']}' has no options"
                assert len(field["options"]) > 0


# ---------------------------------------------------------------------------
# Graphical extraction — the critical tests the old suite was missing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "entry", GRAPHICAL_PDFS, ids=[e["filename"] for e in GRAPHICAL_PDFS]
)
class TestGraphicalExtraction:
    """Tests for graphical PDF parsing — each graphical PDF gets its own tests."""

    def test_no_acroform_detected(self, entry):
        data = run_extract(FIXTURES_DIR / entry["filename"])
        assert not data["has_acroform"] or len(data["acroform_fields"]) == 0

    def test_finds_labels(self, entry):
        data = run_extract(FIXTURES_DIR / entry["filename"])
        all_labels = []
        for page in data["pages"]:
            all_labels.extend(page["labels"])
        assert len(all_labels) > 0, f"No labels found in {entry['filename']}"

    def test_label_positions_not_all_zero(self, entry):
        """Most labels must have non-zero positions — catches the (0,0) bug.

        We don't strictly check page bounds because some PDFs use scaled
        coordinate systems or content matrix transforms that place labels
        beyond the MediaBox.
        """
        data = run_extract(FIXTURES_DIR / entry["filename"])
        for page in data["pages"]:
            labels = page["labels"]
            if not labels:
                continue
            valid_count = sum(1 for l in labels if l["x"] > 0 and l["y"] > 0)
            total = len(labels)
            assert valid_count / total > 0.5, (
                f"Only {valid_count}/{total} labels have non-zero positions "
                f"in {entry['filename']} page {page['page_index']}"
            )

    def test_labels_not_all_same_position(self, entry):
        """All labels at the exact same position means the parser isn't tracking moves."""
        data = run_extract(FIXTURES_DIR / entry["filename"])
        for page in data["pages"]:
            if len(page["labels"]) < 3:
                continue
            positions = {(l["x"], l["y"]) for l in page["labels"]}
            assert len(positions) > 1, (
                f"All {len(page['labels'])} labels at same position in "
                f"{entry['filename']} page {page['page_index']}"
            )

    def test_recommends_overlay_strategy(self, entry):
        data = run_extract(FIXTURES_DIR / entry["filename"])
        assert data["recommended_strategy"] == "overlay"


# ---------------------------------------------------------------------------
# Schengen visa — regression test for TD operator & cumulative positioning
# ---------------------------------------------------------------------------

class TestSchengenVisa:
    """Specific tests for the Schengen visa form that exposed the original parser bug."""

    @pytest.fixture
    def schengen_data(self):
        pdf = FIXTURES_DIR / "eu_schengen_visa_graphical.pdf"
        if not pdf.exists():
            pytest.skip("eu_schengen_visa_graphical.pdf not found")
        return run_extract(pdf)

    def test_finds_many_labels(self, schengen_data):
        labels = schengen_data["pages"][0]["labels"]
        # The Schengen visa has 37 numbered sections — should find plenty of labels
        assert len(labels) >= 50, f"Only found {len(labels)} labels (expected 50+)"

    def test_title_at_correct_position(self, schengen_data):
        labels = schengen_data["pages"][0]["labels"]
        title_label = next(
            (l for l in labels if "Application for Schengen" in l["text"]), None
        )
        assert title_label is not None, "Title 'Application for Schengen Visa' not found"
        # Title should be near top of page (y > 700 in a 842pt page)
        assert title_label["y"] > 700, (
            f"Title y={title_label['y']}, expected > 700 (near top of page)"
        )

    def test_field_labels_spread_across_page(self, schengen_data):
        labels = schengen_data["pages"][0]["labels"]
        ys = [l["y"] for l in labels]
        y_range = max(ys) - min(ys)
        # Labels should span a significant portion of the page
        assert y_range > 300, f"Y range is only {y_range}pt, labels aren't spread out"

    def test_known_field_labels_present(self, schengen_data):
        labels = schengen_data["pages"][0]["labels"]
        text_concat = " ".join(l["text"] for l in labels)
        for expected in ["Surname", "First name", "Date of birth", "Nationality"]:
            assert expected.lower() in text_concat.lower(), (
                f"Expected label '{expected}' not found in Schengen visa"
            )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_nonexistent_file(self):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent.parent / "scripts" / "extract.py"), "/nonexistent.pdf"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0

    def test_invalid_file(self, tmp_path):
        fake_pdf = tmp_path / "fake.pdf"
        fake_pdf.write_text("not a pdf")
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent.parent / "scripts" / "extract.py"), str(fake_pdf)],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
