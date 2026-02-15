#!/usr/bin/env python3
"""Generate before/after screenshot pairs for all 10 graphical test PDFs.

For each PDF:
1. Takes a "before" screenshot of page 0
2. Runs extract.py to find label positions
3. Creates an overlay fill spec with representative field values
4. Runs fill.py to produce the filled PDF
5. Takes an "after" screenshot of the filled page 0

Output: assets/gallery/{basename}_before.png and {basename}_after.png
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import fitz  # PyMuPDF

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
FIXTURES_DIR = BASE_DIR / "tests" / "fixtures"
GALLERY_DIR = BASE_DIR / "assets" / "gallery"
EXTRACT_SCRIPT = BASE_DIR / "scripts" / "extract.py"
FILL_SCRIPT = BASE_DIR / "scripts" / "fill.py"

GALLERY_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# PDF list with page counts
# ---------------------------------------------------------------------------
PDFS = [
    "ca_td1_graphical.pdf",
    "eu_schengen_visa_graphical.pdf",
    "fr_schengen_visa_graphical.pdf",
    "de_en_schengen_visa_mixed.pdf",
    "it_schengen_visa_graphical.pdf",
    "jp_customs_declaration_graphical.pdf",
    "au_incoming_passenger_graphical.pdf",
    "pt_national_visa_graphical.pdf",
    "es_schengen_visa_graphical.pdf",
    "fr_long_stay_visa_graphical.pdf",
]

# ---------------------------------------------------------------------------
# Standard fill values
# ---------------------------------------------------------------------------
STANDARD_VALUES = {
    "surname": "MARTIN",
    "first_name": "Sophie",
    "dob": "15/03/1990",
    "place_of_birth": "Paris, France",
    "nationality": "French",
    "passport": "20FR12345",
}

# ---------------------------------------------------------------------------
# Label-keyword to value mapping
# Each entry: (list of keywords to match in label text, value to fill, offset_below)
# offset_below is how many PDF points below the label to place the fill text
# ---------------------------------------------------------------------------
KEYWORD_MAP = [
    (["surname", "family name", "last name", "familienname", "nom"],
     STANDARD_VALUES["surname"], 12),
    (["first name", "given name", "prenom", "vorname"],
     STANDARD_VALUES["first_name"], 12),
    (["date of birth", "birth date", "geburtsdatum", "date de naissance"],
     STANDARD_VALUES["dob"], 12),
    (["place of birth", "place and country of birth", "geburtsort", "lieu de naissance"],
     STANDARD_VALUES["place_of_birth"], 12),
    (["nationality", "nationalit", "staatsangehörigkeit"],
     STANDARD_VALUES["nationality"], 12),
    (["passport number", "passport no", "travel document number", "reisedokument"],
     STANDARD_VALUES["passport"], 12),
    (["id-number", "id number"],
     STANDARD_VALUES["passport"], 12),
]


def screenshot_page(pdf_path, page_num, output_png):
    """Render a single page of a PDF to a PNG at 150 DPI."""
    doc = fitz.open(str(pdf_path))
    page = doc[page_num]
    pix = page.get_pixmap(dpi=150)
    pix.save(str(output_png))
    doc.close()
    print(f"  Screenshot saved: {output_png.name}")


def run_extract(pdf_path):
    """Run extract.py and return parsed JSON structure."""
    result = subprocess.run(
        [sys.executable, str(EXTRACT_SCRIPT), str(pdf_path), "--page", "0"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  WARNING: extract.py failed: {result.stderr.strip()}")
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"  WARNING: Could not parse extract output")
        return None


def find_label_positions(structure):
    """Search labels for known keywords and return fill field specs."""
    if not structure or not structure.get("pages"):
        return []

    labels = structure["pages"][0].get("labels", [])
    fields = []
    used_keywords = set()

    for label in labels:
        text = label.get("text", "").lower().strip()
        x = label.get("x", 0)
        y = label.get("y", 0)
        page = label.get("page", 0)

        for keywords, value, offset in KEYWORD_MAP:
            kw_key = tuple(keywords)
            if kw_key in used_keywords:
                continue
            for kw in keywords:
                if kw.lower() in text:
                    # Place fill text below the label (y - offset in PDF coords)
                    fields.append({
                        "value": value,
                        "x": x,
                        "y": y - offset,
                        "page": page,
                        "font_size": 10,
                        "type": "text",
                    })
                    used_keywords.add(kw_key)
                    break

    return fields


def get_fixed_fields_for_pdf(basename):
    """Return fixed-position fields for PDFs with garbled labels."""
    # fr_long_stay_visa_graphical.pdf has custom font encoding,
    # so we use fixed coordinates based on the readable English labels
    # and the known form layout.
    if basename == "fr_long_stay_visa_graphical.pdf":
        return [
            # Surname — field 1 area (based on label at ~y=686)
            {"value": "MARTIN", "x": 150, "y": 670, "page": 0,
             "font_size": 10, "type": "text"},
            # First name — field area below surname
            {"value": "Sophie", "x": 150, "y": 630, "page": 0,
             "font_size": 10, "type": "text"},
            # Date of birth — around y=570
            {"value": "15/03/1990", "x": 150, "y": 558, "page": 0,
             "font_size": 10, "type": "text"},
            # Place of birth — around y=547
            {"value": "Paris, France", "x": 250, "y": 535, "page": 0,
             "font_size": 10, "type": "text"},
            # Nationality — around y=985 (page 1 in full doc, but
            # the readable label "Current nationality" is on page 0 area)
            {"value": "French", "x": 350, "y": 997, "page": 0,
             "font_size": 10, "type": "text"},
        ]
    return None


def get_custom_overrides(basename, fields):
    """Apply per-PDF adjustments to fields or add extra fields."""
    if basename == "ca_td1_graphical.pdf":
        # Canada TD1 uses YYYY/MM/DD format and slightly different field names
        for f in fields:
            if f["value"] == STANDARD_VALUES["dob"]:
                f["value"] = "1990/03/15"
            elif f["value"] == STANDARD_VALUES["surname"]:
                f["value"] = "Martin"

    elif basename == "jp_customs_declaration_graphical.pdf":
        # Supplement with Japan-specific fields
        extra = [
            {"value": "AF274", "x": 260, "y": 652, "page": 0,
             "font_size": 9, "type": "text"},
            {"value": "France", "x": 250, "y": 568, "page": 0,
             "font_size": 9, "type": "text"},
        ]
        # Fix name to combined format
        for f in fields:
            if f["value"] == STANDARD_VALUES["surname"]:
                f["value"] = "MARTIN"
            elif f["value"] == STANDARD_VALUES["first_name"]:
                f["value"] = "Sophie"
        fields.extend(extra)

    elif basename == "au_incoming_passenger_graphical.pdf":
        # Australian card — fields are in the left column
        for f in fields:
            if f["value"] == STANDARD_VALUES["surname"]:
                f["value"] = "MARTIN"

    return fields


def run_fill(input_pdf, spec_dict, output_pdf):
    """Run fill.py with the given spec."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp:
        json.dump(spec_dict, tmp)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [sys.executable, str(FILL_SCRIPT), str(input_pdf), tmp_path,
             str(output_pdf)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  WARNING: fill.py failed: {result.stderr.strip()}")
            return False
        return True
    finally:
        os.unlink(tmp_path)


def process_pdf(pdf_name):
    """Process a single PDF: extract, fill, screenshot."""
    pdf_path = FIXTURES_DIR / pdf_name
    basename = pdf_name.replace(".pdf", "")

    print(f"\nProcessing: {pdf_name}")

    if not pdf_path.exists():
        print(f"  ERROR: PDF not found at {pdf_path}")
        return False

    # 1. Before screenshot
    before_png = GALLERY_DIR / f"{basename}_before.png"
    screenshot_page(pdf_path, 0, before_png)

    # 2. Extract structure
    fixed_fields = get_fixed_fields_for_pdf(pdf_name)
    if fixed_fields:
        fields = fixed_fields
        print(f"  Using fixed coordinates ({len(fields)} fields)")
    else:
        structure = run_extract(pdf_path)
        fields = find_label_positions(structure)
        print(f"  Found {len(fields)} fields from label extraction")

    # 3. Apply per-PDF customizations
    fields = get_custom_overrides(pdf_name, fields)

    if not fields:
        print(f"  WARNING: No fields found, using fallback positions")
        fields = [
            {"value": "MARTIN", "x": 100, "y": 650, "page": 0,
             "font_size": 10, "type": "text"},
            {"value": "Sophie", "x": 100, "y": 620, "page": 0,
             "font_size": 10, "type": "text"},
            {"value": "15/03/1990", "x": 100, "y": 590, "page": 0,
             "font_size": 10, "type": "text"},
        ]

    # 4. Create fill spec and run fill.py
    spec = {
        "strategy": "overlay",
        "fields": fields,
    }

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        filled_pdf = tmp.name

    try:
        success = run_fill(pdf_path, spec, filled_pdf)
        if not success:
            print(f"  ERROR: Fill failed for {pdf_name}")
            return False

        # 5. After screenshot
        after_png = GALLERY_DIR / f"{basename}_after.png"
        screenshot_page(filled_pdf, 0, after_png)
    finally:
        if os.path.exists(filled_pdf):
            os.unlink(filled_pdf)

    return True


def main():
    print("=" * 60)
    print("Generating gallery screenshots for graphical PDFs")
    print("=" * 60)

    success_count = 0
    fail_count = 0

    for pdf_name in PDFS:
        ok = process_pdf(pdf_name)
        if ok:
            success_count += 1
        else:
            fail_count += 1

    print("\n" + "=" * 60)
    print(f"Done: {success_count} succeeded, {fail_count} failed")
    print("=" * 60)

    # Verify all expected files exist
    print("\nVerifying output files:")
    all_ok = True
    for pdf_name in PDFS:
        basename = pdf_name.replace(".pdf", "")
        for suffix in ("_before.png", "_after.png"):
            png_path = GALLERY_DIR / f"{basename}{suffix}"
            exists = png_path.exists()
            status = "OK" if exists else "MISSING"
            if not exists:
                all_ok = False
            print(f"  [{status}] {png_path.name}")

    if all_ok:
        print(f"\nAll 20 files present in {GALLERY_DIR}")
    else:
        print(f"\nSome files are missing!")
        sys.exit(1)


if __name__ == "__main__":
    main()
