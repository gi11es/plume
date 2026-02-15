#!/usr/bin/env python3
"""Generate before/after gallery screenshots for all AcroForm test PDFs.

For each AcroForm PDF in tests/fixtures/:
  1. Takes a "before" screenshot of page 0 at 150 DPI
  2. Fills representative fields using pymupdf's widget API
  3. Takes an "after" screenshot of the filled page 0

Output: assets/gallery/{name}_before.png and assets/gallery/{name}_after.png

Usage:
    python scripts/generate_gallery_acroform.py
"""

import os
import sys
import tempfile
from pathlib import Path

import fitz  # pymupdf

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"
GALLERY_DIR = PROJECT_ROOT / "assets" / "gallery"
GALLERY_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Fill specs: PDF basename -> list of (field_name, value) pairs
# ---------------------------------------------------------------------------

FILL_SPECS = {
    "us_w9_acroform.pdf": [
        ("topmostSubform[0].Page1[0].f1_01[0]", "Sophie Martin"),
        ("topmostSubform[0].Page1[0].f1_02[0]", "Martin Digital Consulting LLC"),
        ("topmostSubform[0].Page1[0].Address_ReadOrder[0].f1_07[0]", "742 Evergreen Terrace, Apt 3B"),
        ("topmostSubform[0].Page1[0].Address_ReadOrder[0].f1_08[0]", "Springfield, IL 62704"),
    ],
    "us_w4_acroform.pdf": [
        ("topmostSubform[0].Page1[0].Step1a[0].f1_01[0]", "Sophie"),
        ("topmostSubform[0].Page1[0].Step1a[0].f1_02[0]", "Martin"),
        ("topmostSubform[0].Page1[0].Step1a[0].f1_03[0]", "742 Evergreen Terrace"),
        ("topmostSubform[0].Page1[0].Step1a[0].f1_04[0]", "Springfield, IL 62704"),
        ("topmostSubform[0].Page1[0].f1_05[0]", "123-45-6789"),
    ],
    "us_i9_acroform.pdf": [
        ("Last Name Family Name from Section 1", "Martin"),
        ("First Name Given Name from Section 1", "Sophie"),
        ("First Name Given Name", "Sophie"),
    ],
    "us_sf86_security_acroform.pdf": [
        ("form1[0].Sections1-6[0].section5[0].TextField11[0]", "Martin"),
        ("form1[0].Sections1-6[0].section5[0].TextField11[1]", "Sophie"),
        ("form1[0].Sections1-6[0].section5[0].TextField11[2]", "Louise"),
    ],
    "de_kindergeld_acroform.pdf": [
        ("topmostSubform[0].Page1[0].Frage-1[0].#area[5].Name-Antragsteller[0]", "Martin"),
        ("topmostSubform[0].Page1[0].Frage-1[0].#area[5].Vorname-Antragsteller[0]", "Sophie"),
        ("topmostSubform[0].Page1[0].#area[1].Telefon[0]", "+49 30 1234 5678"),
    ],
    "es_modelo_030_acroform.pdf": [
        ("Nombre y apellidos del titular del órgano", "Sophie Martin"),
        ("NIF ORGANISMO PUBLICO", "B12345678"),
        ("DENOMINACION ORGANISMO PUBLICO", "Martin Digital Consulting SL"),
    ],
    "sample_foersom_acroform.pdf": [
        ("Given Name Text Box", "Sophie"),
        ("Family Name Text Box", "MARTIN"),
        ("Address 1 Text Box", "12 rue des Lilas"),
        ("City Text Box", "Paris"),
        ("Postcode Text Box", "75011"),
        ("Country Combo Box", "France"),
    ],
    "uk_n285_court_acroform.pdf": [
        ("Claimant", "Sophie Martin"),
        ("Defendant", "TechCorp Ltd"),
        ("In the [name] court", "Royal Courts of Justice"),
        ("Claim no", "QB-2026-001234"),
    ],
}


def screenshot_page(pdf_path: str, page_index: int, output_png: str, dpi: int = 150):
    """Render a single page of a PDF to a PNG screenshot."""
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    pix = page.get_pixmap(dpi=dpi)
    pix.save(output_png)
    doc.close()
    print(f"  Saved screenshot: {output_png}")


def fill_with_pymupdf(src_pdf: str, dst_pdf: str, fields: list[tuple[str, str]]):
    """Fill form fields using pymupdf's widget API and save to dst_pdf."""
    doc = fitz.open(src_pdf)
    field_map = {name: value for name, value in fields}
    filled_count = 0

    for page in doc:
        for widget in page.widgets():
            if widget.field_name in field_map:
                widget.field_value = field_map[widget.field_name]
                widget.update()
                filled_count += 1

    doc.save(dst_pdf)
    doc.close()
    print(f"  Filled {filled_count}/{len(fields)} fields -> {dst_pdf}")


def process_pdf(pdf_filename: str, fields: list[tuple[str, str]]):
    """Generate before/after screenshots for one PDF."""
    pdf_path = FIXTURES_DIR / pdf_filename
    stem = pdf_path.stem  # e.g. "us_w9_acroform"

    before_png = GALLERY_DIR / f"{stem}_before.png"
    after_png = GALLERY_DIR / f"{stem}_after.png"

    if not pdf_path.exists():
        print(f"  WARNING: {pdf_path} not found, skipping.")
        return False

    # 1. Before screenshot
    print(f"  Taking 'before' screenshot...")
    screenshot_page(str(pdf_path), 0, str(before_png))

    # 2. Fill with pymupdf widget API into a temp file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_filled = tmp.name

    try:
        print(f"  Filling form fields with pymupdf...")
        fill_with_pymupdf(str(pdf_path), tmp_filled, fields)

        # 3. After screenshot
        print(f"  Taking 'after' screenshot...")
        screenshot_page(tmp_filled, 0, str(after_png))
    finally:
        os.unlink(tmp_filled)

    return True


def main():
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Fixtures dir: {FIXTURES_DIR}")
    print(f"Gallery dir:  {GALLERY_DIR}")
    print()

    success_count = 0
    total = len(FILL_SPECS)

    for pdf_name, fields in FILL_SPECS.items():
        print(f"[{success_count + 1}/{total}] Processing {pdf_name}...")
        ok = process_pdf(pdf_name, fields)
        if ok:
            success_count += 1
        print()

    print(f"Done: {success_count}/{total} PDFs processed.")

    # Verify all expected files exist
    print("\nVerifying output files:")
    all_ok = True
    for pdf_name in FILL_SPECS:
        stem = Path(pdf_name).stem
        for suffix in ("_before.png", "_after.png"):
            png_path = GALLERY_DIR / f"{stem}{suffix}"
            exists = png_path.exists()
            status = "OK" if exists else "MISSING"
            size_info = ""
            if exists:
                size_kb = png_path.stat().st_size / 1024
                size_info = f" ({size_kb:.0f} KB)"
            print(f"  [{status}] {png_path.name}{size_info}")
            if not exists:
                all_ok = False

    if not all_ok:
        print("\nSome files are missing!")
        sys.exit(1)
    else:
        print(f"\nAll 16 gallery images generated successfully.")


if __name__ == "__main__":
    main()
