#!/usr/bin/env python3
"""Verify that a filled PDF has text correctly positioned within field bounds.

Checks both:
1. Overlay text — extracts text from merged overlay and checks it lands within target rects
2. AcroForm fields — checks that field values are set correctly

Usage:
    python verify.py <filled.pdf> <fill_spec.json> [--tolerance 5]

Outputs a JSON report with pass/fail for each field.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from pypdf import PdfReader
from pypdf.generic import ArrayObject, DictionaryObject, IndirectObject


def resolve(obj):
    while isinstance(obj, IndirectObject):
        obj = obj.get_object()
    return obj


def extract_text_positions(page):
    """Extract text with positions from a PDF page."""
    texts = []
    contents = page.get("/Contents")
    if contents is None:
        return texts

    contents = resolve(contents)
    if isinstance(contents, ArrayObject):
        data_parts = []
        for item in contents:
            stream = resolve(item)
            d = stream.get_data() if hasattr(stream, "get_data") else b""
            if isinstance(d, bytes):
                data_parts.append(d.decode("latin-1", errors="replace"))
            else:
                data_parts.append(str(d))
        content_str = "\n".join(data_parts)
    else:
        d = contents.get_data() if hasattr(contents, "get_data") else b""
        content_str = d.decode("latin-1", errors="replace") if isinstance(d, bytes) else str(d)

    # Extract text blocks
    text_block_pattern = re.compile(r"BT\s(.*?)\sET", re.DOTALL)
    for block in text_block_pattern.finditer(content_str):
        block_content = block.group(1)

        td_match = re.search(r"([\d.]+)\s+([\d.]+)\s+Td", block_content)
        tm_match = re.search(
            r"([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s+Tm",
            block_content,
        )

        x, y = 0, 0
        if tm_match:
            x, y = float(tm_match.group(5)), float(tm_match.group(6))
        elif td_match:
            x, y = float(td_match.group(1)), float(td_match.group(2))

        text_parts = []
        for tj in re.finditer(r"\(([^)]*)\)\s*Tj", block_content):
            text_parts.append(tj.group(1))
        for tj in re.finditer(r"\[(.*?)\]\s*TJ", block_content):
            for s in re.finditer(r"\(([^)]*)\)", tj.group(1)):
                text_parts.append(s.group(1))

        if text_parts:
            text = "".join(text_parts).replace("\\(", "(").replace("\\)", ")").replace("\\\\", "\\")
            texts.append({"text": text, "x": x, "y": y})

    return texts


def check_text_in_bounds(text_pos, target_rect, tolerance=5):
    """Check if a text position falls within a target rectangle (with tolerance)."""
    x, y = text_pos["x"], text_pos["y"]
    rx, ry = target_rect["x"], target_rect["y"]
    rw, rh = target_rect["width"], target_rect["height"]

    # Handle negative width/height
    if rw < 0:
        rx, rw = rx + rw, -rw
    if rh < 0:
        ry, rh = ry + rh, -rh

    in_x = (rx - tolerance) <= x <= (rx + rw + tolerance)
    in_y = (ry - tolerance) <= y <= (ry + rh + tolerance)

    return in_x and in_y


def verify_acroform(reader, spec_fields):
    """Verify AcroForm field values."""
    results = []

    if "/AcroForm" not in reader.trailer.get("/Root", {}):
        for field in spec_fields:
            if field.get("name"):
                results.append({
                    "field": field["name"],
                    "status": "fail",
                    "reason": "No AcroForm in PDF",
                    "expected": field.get("value"),
                })
        return results

    # Get all field values
    form_fields = {}
    acroform = resolve(reader.trailer["/Root"]["/AcroForm"])
    if "/Fields" in acroform:
        _collect_field_values(resolve(acroform["/Fields"]), form_fields)

    for field in spec_fields:
        name = field.get("name")
        expected = field.get("value")
        if not name:
            continue

        actual = form_fields.get(name)
        if actual is None:
            results.append({
                "field": name,
                "status": "fail",
                "reason": "Field not found in PDF",
                "expected": expected,
            })
        elif str(actual) == str(expected):
            results.append({
                "field": name,
                "status": "pass",
                "expected": expected,
                "actual": str(actual),
            })
        else:
            results.append({
                "field": name,
                "status": "fail",
                "reason": "Value mismatch",
                "expected": expected,
                "actual": str(actual),
            })

    return results


def _collect_field_values(fields, result, prefix=""):
    for field_ref in fields:
        field = resolve(field_ref)
        if not isinstance(field, DictionaryObject):
            continue
        name = str(field.get("/T", ""))
        full_name = f"{prefix}.{name}" if prefix else name
        value = field.get("/V")
        if value is not None:
            result[full_name] = str(value)
        if "/Kids" in field:
            _collect_field_values(resolve(field["/Kids"]), result, prefix=full_name)


def verify_overlay(reader, spec_fields, tolerance=5):
    """Verify overlay text positions against expected field rects."""
    results = []

    # Group spec fields by page
    by_page = {}
    for field in spec_fields:
        if field.get("x") is not None:
            pg = field.get("page", 0)
            by_page.setdefault(pg, []).append(field)

    for pg_idx, fields in by_page.items():
        if pg_idx >= len(reader.pages):
            for f in fields:
                results.append({
                    "field": f.get("value", "?"),
                    "status": "fail",
                    "reason": f"Page {pg_idx} does not exist",
                })
            continue

        page_texts = extract_text_positions(reader.pages[pg_idx])

        for field in fields:
            expected_value = str(field.get("value", ""))
            target_x = field.get("x", 0)
            target_y = field.get("y", 0)
            target_rect = field.get("target_rect")

            # Find matching text in extracted texts
            found = False
            for text_entry in page_texts:
                if expected_value in text_entry["text"] or text_entry["text"] in expected_value:
                    found = True
                    if target_rect:
                        in_bounds = check_text_in_bounds(text_entry, target_rect, tolerance)
                        results.append({
                            "field": expected_value,
                            "status": "pass" if in_bounds else "fail",
                            "reason": None if in_bounds else "Text outside target rect",
                            "text_pos": {"x": text_entry["x"], "y": text_entry["y"]},
                            "target_rect": target_rect,
                        })
                    else:
                        # No target rect — just check text was placed near expected coords
                        dist = ((text_entry["x"] - target_x)**2 + (text_entry["y"] - target_y)**2)**0.5
                        ok = dist < tolerance * 3
                        results.append({
                            "field": expected_value,
                            "status": "pass" if ok else "warn",
                            "reason": None if ok else f"Text placed {dist:.1f}pt from target",
                            "text_pos": {"x": text_entry["x"], "y": text_entry["y"]},
                            "expected_pos": {"x": target_x, "y": target_y},
                        })
                    break

            if not found:
                results.append({
                    "field": expected_value,
                    "status": "fail",
                    "reason": "Text not found in page",
                    "expected_pos": {"x": target_x, "y": target_y},
                })

    return results


def verify_fill(filled_pdf_path, spec_path, tolerance=5):
    """Run full verification."""
    reader = PdfReader(filled_pdf_path)
    with open(spec_path) as f:
        spec = json.load(f)

    fields = spec.get("fields", [])
    strategy = spec.get("strategy", "auto")

    report = {
        "file": str(filled_pdf_path),
        "results": [],
        "summary": {"total": 0, "pass": 0, "fail": 0, "warn": 0},
    }

    acroform_fields = [f for f in fields if f.get("name")]
    overlay_fields = [f for f in fields if f.get("x") is not None]

    if strategy in ("acroform", "both", "auto") and acroform_fields:
        report["results"].extend(verify_acroform(reader, acroform_fields))

    if strategy in ("overlay", "both", "auto") and overlay_fields:
        report["results"].extend(verify_overlay(reader, overlay_fields, tolerance))

    for r in report["results"]:
        report["summary"]["total"] += 1
        status = r.get("status", "fail")
        if status in report["summary"]:
            report["summary"][status] += 1

    report["all_passed"] = report["summary"]["fail"] == 0

    return report


def main():
    parser = argparse.ArgumentParser(description="Verify PDF fill accuracy")
    parser.add_argument("filled_pdf", help="Path to filled PDF")
    parser.add_argument("fill_spec", help="Path to fill spec JSON")
    parser.add_argument("--tolerance", type=float, default=5, help="Position tolerance in points")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = parser.parse_args()

    if not Path(args.filled_pdf).exists():
        print(json.dumps({"error": f"File not found: {args.filled_pdf}"}), file=sys.stderr)
        sys.exit(1)

    report = verify_fill(args.filled_pdf, args.fill_spec, args.tolerance)
    indent = 2 if args.pretty else None
    print(json.dumps(report, indent=indent, ensure_ascii=False))

    sys.exit(0 if report["all_passed"] else 1)


if __name__ == "__main__":
    main()
