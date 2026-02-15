#!/usr/bin/env python3
"""Extract structure from a PDF: text labels, fillable fields, rectangles, checkboxes.

Handles two types of PDFs:
1. "Graphical" PDFs — fields are drawn as colored rectangles in the content stream
2. "AcroForm" PDFs — proper interactive form fields (text inputs, checkboxes, dropdowns)

Usage:
    python extract.py <input.pdf> [--page N]

Outputs JSON to stdout with the full structure.
"""

import argparse
import json
import re
import sys
import zlib
from pathlib import Path

from pypdf import PdfReader
from pypdf.generic import (
    ArrayObject,
    DictionaryObject,
    IndirectObject,
    NameObject,
    NumberObject,
    TextStringObject,
)


def resolve(obj):
    """Recursively resolve indirect references."""
    while isinstance(obj, IndirectObject):
        obj = obj.get_object()
    return obj


# ---------------------------------------------------------------------------
# AcroForm field extraction
# ---------------------------------------------------------------------------

FIELD_TYPE_MAP = {
    "/Tx": "text",
    "/Btn": "button",  # checkbox or radio
    "/Ch": "choice",   # dropdown or listbox
    "/Sig": "signature",
}


def extract_acroform_fields(reader):
    """Extract interactive form fields from AcroForm."""
    fields = []
    if "/AcroForm" not in reader.trailer.get("/Root", {}):
        return fields

    acroform = resolve(reader.trailer["/Root"]["/AcroForm"])
    if "/Fields" not in acroform:
        return fields

    field_list = resolve(acroform["/Fields"])
    _walk_fields(field_list, fields, reader, parent_name="")
    return fields


def _walk_fields(field_list, results, reader, parent_name=""):
    """Recursively walk AcroForm field tree."""
    for field_ref in field_list:
        field = resolve(field_ref)
        if not isinstance(field, DictionaryObject):
            continue

        # Field name
        partial_name = str(field.get("/T", ""))
        full_name = f"{parent_name}.{partial_name}" if parent_name else partial_name

        # Field type (may be inherited from parent)
        ft = str(field.get("/FT", ""))
        field_type = FIELD_TYPE_MAP.get(ft, ft.lstrip("/").lower() if ft else "unknown")

        # Current value
        value = field.get("/V")
        if value is not None:
            value = str(value)

        # Default value
        default = field.get("/DV")
        if default is not None:
            default = str(default)

        # Options (for choice fields)
        options = []
        if "/Opt" in field:
            opt_array = resolve(field["/Opt"])
            if isinstance(opt_array, ArrayObject):
                for opt in opt_array:
                    opt = resolve(opt)
                    if isinstance(opt, ArrayObject) and len(opt) >= 2:
                        options.append({"export": str(opt[0]), "display": str(opt[1])})
                    else:
                        options.append({"export": str(opt), "display": str(opt)})

        # Widget rect (position)
        rect = None
        page_index = None
        if "/Rect" in field:
            r = resolve(field["/Rect"])
            rect = [float(resolve(x)) for x in r]
        # Find which page this widget is on
        if "/P" in field:
            page_ref = field["/P"]
            for i, page in enumerate(reader.pages):
                if page.indirect_reference == page_ref or page.get_object() == resolve(page_ref):
                    page_index = i
                    break

        # Flags
        flags = int(field.get("/Ff", 0))
        is_readonly = bool(flags & (1 << 0))
        is_required = bool(flags & (1 << 1))
        is_multiline = bool(flags & (1 << 12)) if field_type == "text" else False

        # Check for checkbox/radio specifics
        is_checkbox = False
        is_radio = False
        if field_type == "button":
            is_radio = bool(flags & (1 << 15))
            is_checkbox = not is_radio

        entry = {
            "name": full_name,
            "type": field_type,
            "value": value,
            "default": default,
            "rect": rect,
            "page": page_index,
            "readonly": is_readonly,
            "required": is_required,
        }

        if field_type == "text" and is_multiline:
            entry["multiline"] = True
        if field_type == "button":
            entry["is_checkbox"] = is_checkbox
            entry["is_radio"] = is_radio
        if options:
            entry["options"] = options

        # Only add leaf fields (those with /FT or widgets)
        if ft or "/Rect" in field:
            results.append(entry)

        # Recurse into children (/Kids)
        if "/Kids" in field:
            kids = resolve(field["/Kids"])
            _walk_fields(kids, results, reader, parent_name=full_name)


# ---------------------------------------------------------------------------
# Content stream parsing (graphical PDFs)
# ---------------------------------------------------------------------------

def extract_content_stream(page):
    """Get the decoded content stream as a string."""
    contents = page.get("/Contents")
    if contents is None:
        return ""

    contents = resolve(contents)

    if isinstance(contents, ArrayObject):
        parts = []
        for item in contents:
            stream = resolve(item)
            data = stream.get_data() if hasattr(stream, "get_data") else b""
            if isinstance(data, bytes):
                parts.append(data.decode("latin-1", errors="replace"))
            else:
                parts.append(str(data))
        return "\n".join(parts)
    else:
        data = contents.get_data() if hasattr(contents, "get_data") else b""
        if isinstance(data, bytes):
            return data.decode("latin-1", errors="replace")
        return str(data)


def parse_content_stream(content, page_index):
    """Parse a PDF content stream to extract text, rectangles, and checkboxes."""
    labels = []
    rectangles = []
    checkboxes = []

    lines = content.split("\n")
    current_color = None
    current_stroke = None
    text_matrix = None
    current_font = None
    current_font_size = None

    # Track graphics state for rectangles
    i = 0
    all_tokens = content.replace("\r\n", "\n").replace("\r", "\n")

    # Parse rectangles and colored fills
    # Look for patterns like: R G B rg ... x y w h re f
    rect_pattern = re.compile(
        r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+rg\s+"  # fill color
        r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+re\s+"  # rectangle
        r"f\b"  # fill
    )

    for m in rect_pattern.finditer(all_tokens):
        r, g, b = float(m.group(1)), float(m.group(2)), float(m.group(3))
        x, y, w, h = float(m.group(4)), float(m.group(5)), float(m.group(6)), float(m.group(7))

        rect_info = {
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "color": {"r": r, "g": g, "b": b},
            "page": page_index,
        }

        # Classify: small squares → checkboxes, larger → fields
        abs_w, abs_h = abs(w), abs(h)
        if 8 <= abs_w <= 16 and 8 <= abs_h <= 16:
            checkboxes.append(rect_info)
        else:
            rectangles.append(rect_info)

    # Also look for stroke rectangles (unfilled checkboxes)
    stroke_rect_pattern = re.compile(
        r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+RG\s+"  # stroke color
        r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+re\s+"  # rectangle
        r"S\b"  # stroke
    )

    for m in stroke_rect_pattern.finditer(all_tokens):
        r, g, b = float(m.group(1)), float(m.group(2)), float(m.group(3))
        x, y, w, h = float(m.group(4)), float(m.group(5)), float(m.group(6)), float(m.group(7))
        abs_w, abs_h = abs(w), abs(h)
        if 8 <= abs_w <= 16 and 8 <= abs_h <= 16:
            checkboxes.append({
                "x": x, "y": y, "width": w, "height": h,
                "color": {"r": r, "g": g, "b": b},
                "stroke": True,
                "page": page_index,
            })

    # Parse text blocks (BT ... ET) with proper cumulative positioning
    # Handles Td, TD (cumulative), Tm (absolute), T* (next line), and nested moves
    text_block_pattern = re.compile(r"BT(.*?)ET", re.DOTALL)

    # Tokenizer for PDF operators within text blocks
    token_pattern = re.compile(
        r"(/\S+)"                                   # name like /F0
        r"|(\((?:[^()\\]|\\.)*\))"                  # string in parens (handles escapes)
        r"|\[((?:[^\[\]]|\((?:[^()\\]|\\.)*\))*)\]" # array [...]
        r"|([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"      # number
        r"|([A-Za-z*\'\"]+)"                         # operator
    )

    for block in text_block_pattern.finditer(all_tokens):
        block_content = block.group(1)

        # State within this text block
        tx, ty = 0, 0       # current text position (absolute)
        line_x, line_y = 0, 0  # line start position
        pending_nums = []
        block_font = current_font
        block_font_size = current_font_size

        pending_string = None
        pending_array = None

        for m in token_pattern.finditer(block_content):
            name, string, array, number, operator = m.groups()

            if number is not None:
                pending_nums.append(float(number))
            elif name is not None:
                pending_nums.append(name)
            elif string is not None:
                # String operand — store for upcoming Tj
                pending_string = string
            elif array is not None:
                # Array operand — store for upcoming TJ
                pending_array = array
            elif operator:
                op = operator
                if op == "Tf" and len(pending_nums) >= 2:
                    block_font = str(pending_nums[-2]).lstrip("/")
                    block_font_size = float(pending_nums[-1])
                elif op in ("Td", "TD") and len(pending_nums) >= 2:
                    dx = float(pending_nums[-2])
                    dy = float(pending_nums[-1])
                    line_x += dx
                    line_y += dy
                    tx, ty = line_x, line_y
                elif op == "Tm" and len(pending_nums) >= 6:
                    nums = [float(n) for n in pending_nums[-6:]]
                    tx, ty = nums[4], nums[5]
                    line_x, line_y = tx, ty
                elif op in ("T*", "Tstar"):
                    pass  # position doesn't change meaningfully without TL
                elif op == "Tj" and pending_string is not None:
                    text = pending_string[1:-1]  # strip parens
                    text = text.replace("\\(", "(").replace("\\)", ")").replace("\\\\", "\\")
                    if text.strip():
                        labels.append({
                            "text": text,
                            "x": tx,
                            "y": ty,
                            "font": block_font,
                            "font_size": block_font_size,
                            "page": page_index,
                        })
                elif op == "TJ" and pending_array is not None:
                    text_parts = []
                    for s in re.finditer(r"\((?:[^()\\]|\\.)*\)", pending_array):
                        part = s.group()[1:-1]
                        part = part.replace("\\(", "(").replace("\\)", ")").replace("\\\\", "\\")
                        text_parts.append(part)
                    text = "".join(text_parts)
                    if text.strip():
                        labels.append({
                            "text": text,
                            "x": tx,
                            "y": ty,
                            "font": block_font,
                            "font_size": block_font_size,
                            "page": page_index,
                        })

                pending_nums = []
                pending_string = None
                pending_array = None

    return labels, rectangles, checkboxes


def associate_labels_to_fields(labels, rectangles, tolerance=30):
    """Associate text labels with nearby field rectangles."""
    associations = []
    used_rects = set()

    for label in labels:
        best_rect = None
        best_dist = float("inf")

        for i, rect in enumerate(rectangles):
            if i in used_rects:
                continue
            # Check if the label is near the rect (to the left or above)
            label_right = label["x"] + len(label["text"]) * (label.get("font_size", 10) or 10) * 0.5
            rect_left = rect["x"]
            rect_bottom = rect["y"]

            # Label should be roughly at the same vertical level as the field
            y_diff = abs(label["y"] - rect_bottom)
            if y_diff > tolerance:
                continue

            # Label should be to the left of or near the field
            x_diff = rect_left - label["x"]
            if x_diff < -50:  # label is far to the right of field
                continue

            dist = (x_diff**2 + y_diff**2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_rect = i

        if best_rect is not None:
            associations.append({
                "label": label,
                "field_rect": rectangles[best_rect],
                "distance": best_dist,
            })
            used_rects.add(best_rect)

    return associations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def extract_structure(pdf_path, page_num=None):
    """Extract full structure from a PDF file."""
    reader = PdfReader(pdf_path)

    result = {
        "file": str(pdf_path),
        "num_pages": len(reader.pages),
        "has_acroform": False,
        "acroform_fields": [],
        "pages": [],
    }

    # Check for AcroForm fields
    acroform_fields = extract_acroform_fields(reader)
    if acroform_fields:
        result["has_acroform"] = True
        result["acroform_fields"] = acroform_fields

    # Parse content streams
    pages_to_process = range(len(reader.pages))
    if page_num is not None:
        pages_to_process = [page_num]

    for pg_idx in pages_to_process:
        page = reader.pages[pg_idx]
        mediabox = page.get("/MediaBox")
        page_width = float(resolve(mediabox[2])) if mediabox else 612
        page_height = float(resolve(mediabox[3])) if mediabox else 792

        content = extract_content_stream(page)
        labels, rectangles, checkboxes = parse_content_stream(content, pg_idx)
        associations = associate_labels_to_fields(labels, rectangles)

        page_info = {
            "page_index": pg_idx,
            "width": page_width,
            "height": page_height,
            "labels": labels,
            "field_rectangles": rectangles,
            "checkboxes": checkboxes,
            "label_field_associations": associations,
        }
        result["pages"].append(page_info)

    # Determine fill strategy
    if result["has_acroform"] and len(acroform_fields) > 0:
        fillable_fields = [f for f in acroform_fields if not f["readonly"]]
        if fillable_fields:
            result["recommended_strategy"] = "acroform"
        else:
            result["recommended_strategy"] = "overlay"
    else:
        result["recommended_strategy"] = "overlay"

    return result


def main():
    parser = argparse.ArgumentParser(description="Extract PDF form structure")
    parser.add_argument("pdf", help="Path to PDF file")
    parser.add_argument("--page", type=int, default=None, help="Process only this page (0-indexed)")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    args = parser.parse_args()

    if not Path(args.pdf).exists():
        print(json.dumps({"error": f"File not found: {args.pdf}"}), file=sys.stderr)
        sys.exit(1)

    result = extract_structure(args.pdf, args.page)
    indent = 2 if args.pretty else None
    print(json.dumps(result, indent=indent, ensure_ascii=False))


if __name__ == "__main__":
    main()
