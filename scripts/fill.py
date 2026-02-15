#!/usr/bin/env python3
"""Fill a PDF form using either AcroForm field values or text overlay.

Supports two strategies:
1. "acroform" — sets field values directly on interactive form fields
2. "overlay" — draws text at exact coordinates using reportlab, merges onto original

Usage:
    python fill.py <input.pdf> <fill_spec.json> <output.pdf> [--strategy auto]

The fill_spec.json format:
{
    "strategy": "overlay" | "acroform" | "auto",
    "fields": [
        {
            "name": "field_name",           # for acroform
            "value": "text to fill",
            "x": 134.36, "y": 720.4,       # for overlay
            "page": 0,
            "font_size": 10,
            "font": "Helvetica",
            "type": "text" | "checkbox" | "choice" | "signature" | "photo",
            "image_path": "/path/to/image.png",  # for signature/photo type
            "width": 150, "height": 50           # for signature (99x128 for photo)
        }
    ]
}
"""

import argparse
import io
import json
import sys
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.generic import IndirectObject
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def resolve(obj):
    while isinstance(obj, IndirectObject):
        obj = obj.get_object()
    return obj


# ---------------------------------------------------------------------------
# AcroForm filling
# ---------------------------------------------------------------------------

def fill_acroform(reader, writer, fields_spec):
    """Fill AcroForm fields by name."""
    field_values = {}
    for field in fields_spec:
        name = field.get("name")
        value = field.get("value")
        if name and value is not None:
            field_values[name] = value

    # Clone the entire document to preserve AcroForm
    writer.clone_reader_document_root(reader)

    # Strip XFA if present — XFA overrides AcroForm visual rendering,
    # and pypdf only updates AcroForm values (not XFA XML). Removing XFA
    # forces PDF viewers to render from the AcroForm layer instead.
    root = writer._root_object
    if "/AcroForm" in root:
        af = root["/AcroForm"]
        if hasattr(af, "get_object"):
            af = af.get_object()
        if "/XFA" in af:
            del af["/XFA"]

    # Update form fields on each page
    if field_values:
        for page in writer.pages:
            writer.update_page_form_field_values(page, field_values)

    return writer



# ---------------------------------------------------------------------------
# Overlay filling
# ---------------------------------------------------------------------------

def create_overlay(fields_spec, page_width, page_height):
    """Create a PDF overlay with text at specified coordinates."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(page_width, page_height))

    for field in fields_spec:
        x = field.get("x", 0)
        y = field.get("y", 0)
        value = field.get("value", "")
        font_size = field.get("font_size", 10)
        font_name = field.get("font", "Helvetica")
        field_type = field.get("type", "text")
        color = field.get("color")

        if color:
            c.setFillColorRGB(color.get("r", 0), color.get("g", 0), color.get("b", 0))
        else:
            c.setFillColorRGB(0, 0, 0)  # default black

        if field_type == "signature":
            # Draw a signature image (PNG with transparent background)
            image_path = field.get("image_path", "")
            if image_path and Path(image_path).exists():
                sig_w = field.get("width", 150)
                sig_h = field.get("height", 50)
                c.drawImage(ImageReader(image_path), x, y,
                            width=sig_w, height=sig_h, mask="auto")
        elif field_type == "photo":
            # Draw a passport/ID photo image
            image_path = field.get("image_path", "")
            if image_path and Path(image_path).exists():
                ph_w = field.get("width", 99)    # 35mm at 72dpi
                ph_h = field.get("height", 128)   # 45mm at 72dpi
                c.drawImage(ImageReader(image_path), x, y,
                            width=ph_w, height=ph_h, mask="auto")
        elif field_type == "checkbox":
            # Draw an X mark
            c.setFont(font_name, font_size)
            c.drawString(x, y, "X")
        elif field_type == "text":
            c.setFont(font_name, font_size)
            c.drawString(x, y, str(value))

    c.save()
    buf.seek(0)
    return buf


def fill_overlay(reader, fields_spec, output_path):
    """Fill a PDF by overlaying text at exact coordinates."""
    writer = PdfWriter()

    # Group fields by page
    fields_by_page = {}
    for field in fields_spec:
        pg = field.get("page", 0)
        fields_by_page.setdefault(pg, []).append(field)

    for pg_idx in range(len(reader.pages)):
        page = reader.pages[pg_idx]
        mediabox = page.get("/MediaBox")
        page_width = float(resolve(mediabox[2])) if mediabox else 595.28  # A4
        page_height = float(resolve(mediabox[3])) if mediabox else 841.89

        if pg_idx in fields_by_page:
            overlay_buf = create_overlay(fields_by_page[pg_idx], page_width, page_height)
            overlay_reader = PdfReader(overlay_buf)
            overlay_page = overlay_reader.pages[0]
            page.merge_page(overlay_page)

        writer.add_page(page)

    with open(output_path, "wb") as f:
        writer.write(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def fill_pdf(input_path, spec_path, output_path, strategy="auto"):
    """Fill a PDF based on the spec."""
    with open(spec_path, "r") as f:
        spec = json.load(f)

    reader = PdfReader(input_path)
    fields = spec.get("fields", [])
    strategy = spec.get("strategy", strategy)

    if strategy == "auto":
        # Check if the PDF has AcroForm fields
        has_acroform = "/AcroForm" in reader.trailer.get("/Root", {})
        acroform_fields = [f for f in fields if f.get("name")]
        overlay_fields = [f for f in fields if f.get("x") is not None]

        if has_acroform and acroform_fields:
            strategy = "acroform"
        elif overlay_fields:
            strategy = "overlay"
        else:
            strategy = "overlay"

    if strategy == "acroform":
        writer = PdfWriter()
        fill_acroform(reader, writer, fields)
        with open(output_path, "wb") as f:
            writer.write(f)
    elif strategy == "overlay":
        fill_overlay(reader, fields, output_path)
    elif strategy == "both":
        # First fill AcroForm fields, then overlay
        writer = PdfWriter()
        fill_acroform(reader, writer, [f for f in fields if f.get("name")])
        # Write intermediate
        intermediate = io.BytesIO()
        writer.write(intermediate)
        intermediate.seek(0)
        # Now overlay
        reader2 = PdfReader(intermediate)
        fill_overlay(reader2, [f for f in fields if f.get("x") is not None], output_path)
    else:
        print(f"Unknown strategy: {strategy}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps({
        "status": "success",
        "output": str(output_path),
        "strategy": strategy,
        "fields_filled": len(fields),
    }))


def main():
    parser = argparse.ArgumentParser(description="Fill a PDF form")
    parser.add_argument("input_pdf", help="Path to input PDF")
    parser.add_argument("fill_spec", help="Path to JSON fill specification")
    parser.add_argument("output_pdf", help="Path for output PDF")
    parser.add_argument("--strategy", default="auto", choices=["auto", "overlay", "acroform", "both"],
                        help="Fill strategy")
    args = parser.parse_args()

    if not Path(args.input_pdf).exists():
        print(json.dumps({"error": f"Input PDF not found: {args.input_pdf}"}), file=sys.stderr)
        sys.exit(1)
    if not Path(args.fill_spec).exists():
        print(json.dumps({"error": f"Fill spec not found: {args.fill_spec}"}), file=sys.stderr)
        sys.exit(1)

    fill_pdf(args.input_pdf, args.fill_spec, args.output_pdf, args.strategy)


if __name__ == "__main__":
    main()
