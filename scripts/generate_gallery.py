#!/usr/bin/env python3
"""Generate before/after gallery screenshots for all test PDFs.

Uses the real extract.py + fill.py pipeline — no separate rendering logic.
Each PDF must have a corresponding fill spec at tests/fill_specs/{stem}.json.

For AcroForm PDFs, fill.py sets field values directly.
For graphical PDFs, fill.py creates text overlays at exact coordinates.

Usage:
    python scripts/generate_gallery.py           # all PDFs
    python scripts/generate_gallery.py us_w9     # single PDF (substring match)
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import fitz  # PyMuPDF — for screenshots only

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"
FILL_SPECS_DIR = PROJECT_ROOT / "tests" / "fill_specs"
GALLERY_DIR = PROJECT_ROOT / "assets" / "gallery"
FILL_SCRIPT = PROJECT_ROOT / "scripts" / "fill.py"

GALLERY_DIR.mkdir(parents=True, exist_ok=True)


def screenshot_page(pdf_path: str, page_index: int, output_png: str, dpi: int = 150):
    """Render a single page of a PDF to a PNG screenshot."""
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    pix = page.get_pixmap(dpi=dpi)
    pix.save(output_png)
    doc.close()


def process_pdf(pdf_path: Path, spec_path: Path) -> bool:
    """Generate before/after screenshots for one PDF using fill.py."""
    import json as _json

    stem = pdf_path.stem
    print(f"  [{stem}]")

    with open(spec_path) as _f:
        _spec = _json.load(_f)

    # Determine which pages to screenshot
    # "gallery_pages" (list) takes precedence over "gallery_page" (int)
    gallery_pages = _spec.get("gallery_pages")
    if gallery_pages is None:
        gallery_pages = [_spec.get("gallery_page", 0)]

    # 2. Fill with fill.py
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_filled = tmp.name

    try:
        result = subprocess.run(
            [sys.executable, str(FILL_SCRIPT), str(pdf_path), str(spec_path), tmp_filled],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            print(f"    ERROR: fill.py failed: {result.stderr.strip()}")
            return False

        # Parse fill.py JSON output
        import json
        try:
            fill_result = json.loads(result.stdout)
            strategy = fill_result.get("strategy", "unknown")
            n_fields = fill_result.get("fields_filled", 0)
            print(f"    Filled: {n_fields} fields (strategy: {strategy})")
        except json.JSONDecodeError:
            print(f"    Filled (could not parse output)")

        # Generate before/after screenshots for each page
        for page_index in gallery_pages:
            if len(gallery_pages) == 1:
                # Single page: use original naming (no _pN suffix)
                before_png = GALLERY_DIR / f"{stem}_before.png"
                after_png = GALLERY_DIR / f"{stem}_after.png"
            else:
                # Multi-page: add _pN suffix
                before_png = GALLERY_DIR / f"{stem}_before_p{page_index}.png"
                after_png = GALLERY_DIR / f"{stem}_after_p{page_index}.png"

            screenshot_page(str(pdf_path), page_index, str(before_png))
            print(f"    Before: {before_png.name} ({before_png.stat().st_size // 1024} KB)")

            screenshot_page(tmp_filled, page_index, str(after_png))
            print(f"    After:  {after_png.name} ({after_png.stat().st_size // 1024} KB)")

    finally:
        if os.path.exists(tmp_filled):
            os.unlink(tmp_filled)

    return True


def main():
    filter_str = sys.argv[1] if len(sys.argv) > 1 else None

    # Find all PDFs that have fill specs
    specs = sorted(FILL_SPECS_DIR.glob("*.json"))
    if not specs:
        print(f"No fill specs found in {FILL_SPECS_DIR}")
        print("Create fill spec JSON files at tests/fill_specs/<pdf_stem>.json")
        sys.exit(1)

    pairs = []
    for spec_path in specs:
        stem = spec_path.stem
        pdf_path = FIXTURES_DIR / f"{stem}.pdf"
        if not pdf_path.exists():
            print(f"  WARNING: PDF not found for spec {spec_path.name}: {pdf_path}")
            continue
        if filter_str and filter_str not in stem:
            continue
        pairs.append((pdf_path, spec_path))

    print(f"Generating gallery screenshots for {len(pairs)} PDFs")
    print(f"  Fill specs: {FILL_SPECS_DIR}")
    print(f"  Gallery:    {GALLERY_DIR}")
    print()

    success = 0
    failed = 0

    for pdf_path, spec_path in pairs:
        ok = process_pdf(pdf_path, spec_path)
        if ok:
            success += 1
        else:
            failed += 1

    print(f"\nDone: {success} succeeded, {failed} failed")

    # Verify all expected files exist
    import json as _json_verify
    print("\nVerifying output files:")
    all_ok = True
    for pdf_path, spec_path in pairs:
        stem = pdf_path.stem
        with open(spec_path) as _fv:
            _sv = _json_verify.load(_fv)
        gp = _sv.get("gallery_pages")
        if gp is None:
            gp = [_sv.get("gallery_page", 0)]

        for page_index in gp:
            if len(gp) == 1:
                suffixes = ("_before.png", "_after.png")
            else:
                suffixes = (f"_before_p{page_index}.png", f"_after_p{page_index}.png")
            for suffix in suffixes:
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
        print(f"\nAll gallery images generated successfully.")


if __name__ == "__main__":
    main()
