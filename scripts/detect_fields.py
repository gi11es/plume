#!/usr/bin/env python3
"""Detect fillable fields in graphical PDFs using visual layout analysis.

Uses multiple detection strategies:
1. Grid-line detection — finds horizontal/vertical lines forming a table grid
2. Filled-rectangle detection — finds white/colored boxes used as input fields
3. Character-box merging — merges groups of small adjacent rectangles into text fields
4. Checkbox detection — finds small square shapes
5. Underline detection — finds horizontal lines with labels above (write-on-line fields)
6. OCR fallback — for image-only PDFs where text is embedded as raster images

Usage:
    python detect_fields.py <input.pdf> [--page 0] [--pretty] [--annotate out.png]
    python detect_fields.py <input.pdf> --grid-overlay out.png   # coordinate grid for visual positioning

Output: JSON with detected fields, each with:
  - label: the field label text
  - cell_rect: {x0, y0, x1, y1} in PDF coordinates (bottom-left origin)
  - fill_point: {x, y} where text should be placed (PDF coordinates)
  - field_type: "text" or "checkbox"
  - font_size: recommended font size based on cell height
"""

import argparse
import json
import sys
from pathlib import Path

import fitz  # pymupdf


# ---------------------------------------------------------------------------
# Strategy 1: Grid-line detection
# ---------------------------------------------------------------------------

def detect_lines(page):
    """Detect horizontal and vertical grid lines from page drawings."""
    drawings = page.get_drawings()
    h_lines = []
    v_lines = []

    for d in drawings:
        r = d["rect"]
        # Thin horizontal lines (stroked or very thin filled rects)
        if r.height < 2.5 and r.width > 15:
            h_lines.append(r)
        # Thin vertical lines
        elif r.width < 2.5 and r.height > 8:
            v_lines.append(r)

    return h_lines, v_lines


def cluster_values(values, tolerance=3.0):
    """Cluster nearby coordinate values."""
    if not values:
        return []
    values = sorted(set(values))
    clusters = [[values[0]]]
    for v in values[1:]:
        if v - clusters[-1][-1] <= tolerance:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [sum(c) / len(c) for c in clusters]


def build_grid_cells(h_lines, v_lines, page_rect):
    """Build rectangular grid cells from horizontal and vertical lines."""
    h_ys = cluster_values([r.y0 for r in h_lines])
    v_xs = cluster_values([r.x0 for r in v_lines])

    cells = []
    for i in range(len(h_ys) - 1):
        y_top = h_ys[i]
        y_bot = h_ys[i + 1]
        if y_bot - y_top < 5:
            continue

        row_v_xs = set()
        for vl in v_lines:
            if vl.y0 <= y_top + 3 and vl.y1 >= y_bot - 3:
                row_v_xs.add(round(vl.x0, 1))
        row_v_xs = sorted(row_v_xs)

        if len(row_v_xs) < 2:
            matching_h = [h for h in h_lines if abs(h.y0 - y_top) < 3]
            if matching_h:
                x_left = min(h.x0 for h in matching_h)
                x_right = max(h.x1 for h in matching_h)
                cells.append(fitz.Rect(x_left, y_top, x_right, y_bot))
            continue

        for j in range(len(row_v_xs) - 1):
            x_left = row_v_xs[j]
            x_right = row_v_xs[j + 1]
            if x_right - x_left > 10:
                cells.append(fitz.Rect(x_left, y_top, x_right, y_bot))

    return cells


# ---------------------------------------------------------------------------
# Strategy 2: Filled-rectangle detection
# ---------------------------------------------------------------------------

def detect_filled_field_rects(page):
    """Detect filled rectangles that serve as input field areas.

    Common patterns:
    - White boxes on a colored background (e.g., AU passenger card)
    - Light-colored boxes (e.g., pale yellow/blue field highlights)
    """
    drawings = page.get_drawings()
    field_rects = []

    page_area = page.rect.width * page.rect.height

    # Check if the page has a colored background
    bg_rects = []
    for d in drawings:
        r = d["rect"]
        fill = d.get("fill")
        if fill and r.width * r.height > page_area * 0.3:
            is_white = all(c > 0.95 for c in fill)
            if not is_white:
                bg_rects.append((r, fill))

    has_colored_bg = len(bg_rects) > 0

    for d in drawings:
        r = d["rect"]
        fill = d.get("fill")
        if not fill:
            continue
        w, h = r.width, r.height

        # Skip very small (but not character boxes) or very large
        if w < 8 or h < 5 or h > 60 or w * h > page_area * 0.3:
            continue

        is_white = all(c > 0.95 for c in fill)
        is_black = all(c < 0.05 for c in fill)
        is_light = all(c > 0.8 for c in fill)

        # White rectangles on colored background → input fields
        if has_colored_bg and is_white:
            field_rects.append(r)
        # Light-colored (but not white/black) rectangles → highlighted fields
        elif is_light and not is_white and not is_black:
            field_rects.append(r)

    return field_rects


# ---------------------------------------------------------------------------
# Strategy 3: Character-box group merging
# ---------------------------------------------------------------------------

def merge_character_boxes(rects, tolerance=4.0):
    """Merge groups of small adjacent same-size rectangles into single fields.

    Detects patterns like individual character input boxes:
    □□□□□□□□  → single text field for passport number
    □□ □□ □□□□  → date field (dd/mm/yyyy) with gaps

    Returns list of (merged_rect, box_count) tuples.
    """
    if not rects:
        return []

    # Group rectangles by similar height AND y-position
    rows = {}
    for r in rects:
        # Key: (rounded height, rounded y position)
        key = (round(r.height, 0), round(r.y0, 1))
        rows.setdefault(key, []).append(r)

    merged = []
    for key, group in rows.items():
        if len(group) < 2:
            continue

        # Sort by x position
        group.sort(key=lambda r: r.x0)

        # Find runs of adjacent boxes
        current_run = [group[0]]
        for i in range(1, len(group)):
            prev = current_run[-1]
            curr = group[i]
            gap = curr.x0 - prev.x1
            # Allow small gaps (spacing between character boxes) up to 2x box width
            max_gap = max(tolerance, prev.width * 2)
            if gap <= max_gap and abs(curr.height - prev.height) < 3:
                current_run.append(curr)
            else:
                if len(current_run) >= 2:
                    merged_rect = fitz.Rect(
                        current_run[0].x0, current_run[0].y0,
                        current_run[-1].x1, current_run[-1].y1
                    )
                    merged.append((merged_rect, len(current_run)))
                current_run = [curr]

        # Don't forget the last run
        if len(current_run) >= 2:
            merged_rect = fitz.Rect(
                current_run[0].x0, current_run[0].y0,
                current_run[-1].x1, current_run[-1].y1
            )
            merged.append((merged_rect, len(current_run)))

    return merged


# ---------------------------------------------------------------------------
# Strategy 4: Checkbox detection
# ---------------------------------------------------------------------------

def detect_checkboxes(page):
    """Detect checkbox squares from drawings."""
    drawings = page.get_drawings()
    checkboxes = []
    for d in drawings:
        r = d["rect"]
        w, h = r.width, r.height
        if 5 <= w <= 14 and 5 <= h <= 14 and abs(w - h) < 3:
            fill = d.get("fill")
            # Include white-filled checkboxes and unfilled squares
            if fill is None or all(c > 0.9 for c in fill):
                checkboxes.append(r)
    # Deduplicate checkboxes at same position
    unique = []
    for cb in checkboxes:
        is_dup = False
        for u in unique:
            if abs(cb.x0 - u.x0) < 2 and abs(cb.y0 - u.y0) < 2:
                is_dup = True
                break
        if not is_dup:
            unique.append(cb)
    return unique


# ---------------------------------------------------------------------------
# Strategy 5: Underline detection (write-on-line fields)
# ---------------------------------------------------------------------------

def detect_underline_fields(page, h_lines, grid_cells, words):
    """Detect fields indicated by underlines with labels above or to the left.

    Common in simpler forms where a label like "Name: ______________" is used.
    Only active when grid detection finds few cells (otherwise grid lines would
    create too many false positives).
    """
    # Skip underline detection if the grid already found enough structure
    if len(grid_cells) >= 10:
        return []

    fields = []

    # Collect y-positions used by grid lines to avoid double-counting
    grid_ys = set()
    for cell in grid_cells:
        grid_ys.add(round(cell.y0, 0))
        grid_ys.add(round(cell.y1, 0))

    for line in h_lines:
        # Only consider lines that are likely underlines (not grid borders)
        if line.width < 30 or line.width > page.rect.width * 0.7:
            continue

        # Skip lines at y-positions already used by grid cells
        line_y = round(line.y0, 0)
        if any(abs(line_y - gy) < 5 for gy in grid_ys):
            continue

        # Look for labels directly above or to the left
        label_candidates = []
        for w in words:
            wx0, wy0, wx1, wy1, text = w[0], w[1], w[2], w[3], w[4]
            w_cy = (wy0 + wy1) / 2

            # Label above the line, horizontally overlapping
            if wy1 <= line.y0 + 2 and wy1 >= line.y0 - 15:
                if wx0 < line.x1 and wx1 > line.x0:
                    label_candidates.append((line.y0 - wy1, text))

            # Label to the left
            if wx1 <= line.x0 + 2 and wx1 >= line.x0 - 60:
                if abs(w_cy - line.y0) < 8:
                    label_candidates.append((line.x0 - wx1, text))

        label_candidates.sort()
        label = label_candidates[0][1] if label_candidates else ""

        # Require a meaningful label (at least 3 chars, not just punctuation)
        if label and len(label.strip()) >= 3:
            field_rect = fitz.Rect(line.x0, line.y0 - 12, line.x1, line.y0 + 2)
            fields.append({
                "label": label.strip(),
                "field_type": "text",
                "rect": field_rect,
                "source": "underline",
            })

    return fields


# ---------------------------------------------------------------------------
# Strategy 6: OCR fallback for image-only PDFs
# ---------------------------------------------------------------------------

def detect_with_ocr(page):
    """Use OCR to detect text in image-only PDFs.

    Falls back to pymupdf's built-in OCR support via Tesseract.
    Returns text blocks with positions that can be used for label association.
    """
    try:
        # pymupdf can perform OCR if tesseract is installed
        tp = page.get_textpage_ocr(flags=0, language="eng", dpi=300)
        words = page.get_text("words", textpage=tp)
        return words
    except Exception:
        # Tesseract not available — silently fall back
        return []


# ---------------------------------------------------------------------------
# Label association
# ---------------------------------------------------------------------------

def find_label_for_cell(cell, words):
    """Find label text inside a cell."""
    labels = []
    for w in words:
        wx0, wy0, wx1, wy1, text = w[0], w[1], w[2], w[3], w[4]
        if (wx0 >= cell.x0 - 2 and wx1 <= cell.x1 + 2 and
                wy0 >= cell.y0 - 2 and wy1 <= cell.y1 + 2):
            labels.append((wy0, wx0, text))
    labels.sort()
    return " ".join(t for _, _, t in labels) if labels else ""


def find_label_for_rect(rect, words, direction="left_or_above"):
    """Find the label text near a rectangle (to the left or above it)."""
    candidates = []
    for w in words:
        wx0, wy0, wx1, wy1, text = w[0], w[1], w[2], w[3], w[4]
        w_cy = (wy0 + wy1) / 2
        r_cy = (rect.y0 + rect.y1) / 2

        # Label to the left, vertically aligned
        if wx1 <= rect.x0 + 2 and wx1 >= rect.x0 - 120 and abs(w_cy - r_cy) < 10:
            dist = rect.x0 - wx1
            candidates.append((dist, text))
        # Label above, horizontally overlapping
        elif wy1 <= rect.y0 + 2 and wy1 >= rect.y0 - 20 and wx0 < rect.x1 and wx1 > rect.x0:
            dist = rect.y0 - wy1 + 100  # bias toward left labels
            candidates.append((dist, text))

    candidates.sort()
    # Collect nearby words for multi-word labels
    if candidates:
        label_parts = []
        for dist, text in candidates[:5]:
            if dist < candidates[0][0] + 80:
                label_parts.append(text)
        return " ".join(label_parts)
    return ""


def find_label_for_checkbox(cb, words, max_dist=100):
    """Find the label text near a checkbox."""
    best = None
    best_dist = max_dist
    # Check right
    for w in words:
        wx0, wy0, wx1, wy1, text = w[0], w[1], w[2], w[3], w[4]
        if wx0 > cb.x1 and abs((wy0 + wy1) / 2 - (cb.y0 + cb.y1) / 2) < 6:
            dist = wx0 - cb.x1
            if dist < best_dist:
                best = text
                best_dist = dist
    # Check left
    if not best:
        for w in words:
            wx0, wy0, wx1, wy1, text = w[0], w[1], w[2], w[3], w[4]
            if wx1 < cb.x0 and abs((wy0 + wy1) / 2 - (cb.y0 + cb.y1) / 2) < 6:
                dist = cb.x0 - wx1
                if dist < best_dist:
                    best = text
                    best_dist = dist
    return best or ""


# ---------------------------------------------------------------------------
# Fill point computation
# ---------------------------------------------------------------------------

def compute_fill_point_cell(cell, words, page_height):
    """Compute fill point for a grid cell (text placed below label)."""
    label_bottom = cell.y0
    for w in words:
        wx0, wy0, wx1, wy1, text = w[0], w[1], w[2], w[3], w[4]
        if (wx0 >= cell.x0 - 2 and wx1 <= cell.x1 + 2 and
                wy0 >= cell.y0 - 2 and wy1 <= cell.y1 + 2):
            label_bottom = max(label_bottom, wy1)

    fill_x_mu = cell.x0 + 3
    available = cell.y1 - label_bottom
    if available > 8:
        fill_y_mu = label_bottom + available * 0.6
    else:
        fill_y_mu = (cell.y0 + cell.y1) / 2 + 3

    font_size = min(10, max(6, int(available * 0.7))) if available > 8 else min(8, max(5, int(cell.height * 0.5)))
    return fill_x_mu, page_height - fill_y_mu, font_size


def compute_fill_point_rect(rect, page_height):
    """Compute fill point for a filled rectangle (text centered inside)."""
    fill_x_mu = rect.x0 + 2
    fill_y_mu = rect.y0 + rect.height * 0.65
    font_size = min(10, max(5, int(rect.height * 0.65)))
    return fill_x_mu, page_height - fill_y_mu, font_size


# ---------------------------------------------------------------------------
# Main detection
# ---------------------------------------------------------------------------

def detect_fields(pdf_path, page_num=0):
    """Detect all fillable fields on a page using multiple strategies."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    page_height = page.rect.height
    page_width = page.rect.width

    # Get text words (from PDF text stream — fast and accurate for most PDFs)
    words = page.get_text("words")

    # If very few words found, try OCR as fallback (image-only PDFs)
    used_ocr = False
    if len(words) < 5:
        ocr_words = detect_with_ocr(page)
        if len(ocr_words) > len(words):
            words = ocr_words
            used_ocr = True

    # Strategy 1: Grid-line cells
    h_lines, v_lines = detect_lines(page)
    grid_cells = build_grid_cells(h_lines, v_lines, page.rect)

    # Strategy 2: Filled rectangles
    filled_rects = detect_filled_field_rects(page)

    # Strategy 3: Character-box groups (merge small adjacent rects)
    char_box_groups = merge_character_boxes(filled_rects)

    # Remove individual character boxes that were merged into groups
    grouped_rects = set()
    for merged_rect, count in char_box_groups:
        for fr in filled_rects:
            if (fr.x0 >= merged_rect.x0 - 1 and fr.x1 <= merged_rect.x1 + 1 and
                    abs(fr.y0 - merged_rect.y0) < 2):
                grouped_rects.add(id(fr))

    remaining_filled = [fr for fr in filled_rects if id(fr) not in grouped_rects]

    # Strategy 4: Checkboxes
    checkboxes = detect_checkboxes(page)

    # Remove checkboxes that overlap with filled rects (avoid double-counting)
    checkbox_only = []
    for cb in checkboxes:
        is_field = False
        for fr in filled_rects:
            if id(fr) not in grouped_rects:
                overlap = fitz.Rect(cb).intersect(fr)
                if overlap.width > 0 and overlap.height > 0:
                    is_field = True
                    break
        if not is_field:
            checkbox_only.append(cb)

    # Strategy 5: Underline fields (only when grid detection is sparse)
    underline_fields = detect_underline_fields(page, h_lines, grid_cells, words)

    # Merge all cell-type detections, avoiding duplicates
    all_cells = list(grid_cells)

    # Add character-box groups
    char_box_cell_rects = []
    for merged_rect, count in char_box_groups:
        is_dup = False
        for gc in grid_cells:
            overlap = fitz.Rect(merged_rect).intersect(gc)
            if overlap.width > 0 and overlap.height > 0:
                if overlap.width * overlap.height > merged_rect.width * merged_rect.height * 0.3:
                    is_dup = True
                    break
        if not is_dup:
            all_cells.append(merged_rect)
            char_box_cell_rects.append(merged_rect)

    # Add remaining filled rects (those not merged into char-box groups)
    for fr in remaining_filled:
        # Skip tiny rects that might be individual character boxes we missed
        if fr.width < 15 and fr.height < 15:
            continue
        is_dup = False
        for gc in all_cells:
            overlap = fitz.Rect(fr).intersect(gc)
            if overlap.width > 0 and overlap.height > 0:
                if overlap.width * overlap.height > fr.width * fr.height * 0.3:
                    is_dup = True
                    break
        if not is_dup:
            all_cells.append(fr)

    # Build field list
    fields = []
    used_positions = set()

    for cell in all_cells:
        # Try finding label inside cell first (grid cells), then nearby (filled rects)
        label = find_label_for_cell(cell, words)
        is_grid_cell = cell in grid_cells
        is_char_box = cell in char_box_cell_rects

        if is_grid_cell:
            fill_x, fill_y, font_size = compute_fill_point_cell(cell, words, page_height)
        else:
            if not label:
                label = find_label_for_rect(cell, words)
            fill_x, fill_y, font_size = compute_fill_point_rect(cell, page_height)

        # For character-box fields, use smaller font to fit individual boxes
        if is_char_box:
            font_size = min(font_size, max(5, int(cell.height * 0.55)))

        if not label:
            continue

        # Deduplicate by position
        pos_key = (round(fill_x, 0), round(fill_y, 0))
        if pos_key in used_positions:
            continue
        used_positions.add(pos_key)

        field_entry = {
            "label": label.strip(),
            "field_type": "text",
            "cell_rect": {
                "x0": round(cell.x0, 1),
                "y0": round(page_height - cell.y1, 1),
                "x1": round(cell.x1, 1),
                "y1": round(page_height - cell.y0, 1),
            },
            "fill_point": {
                "x": round(fill_x, 1),
                "y": round(fill_y, 1),
            },
            "font_size": font_size,
            "cell_width": round(cell.width, 1),
            "cell_height": round(cell.height, 1),
        }
        if is_char_box:
            # Include box count for character-box fields
            for merged_rect, count in char_box_groups:
                if abs(merged_rect.x0 - cell.x0) < 1 and abs(merged_rect.y0 - cell.y0) < 1:
                    field_entry["char_boxes"] = count
                    break
        fields.append(field_entry)

    # Add underline fields
    for uf in underline_fields:
        rect = uf["rect"]
        fill_x = rect.x0 + 2
        fill_y = page_height - (rect.y0 + rect.height * 0.5)
        pos_key = (round(fill_x, 0), round(fill_y, 0))
        if pos_key in used_positions:
            continue
        # Check no existing cell already covers this area
        is_dup = False
        for cell in all_cells:
            overlap = rect.intersect(cell)
            if overlap.width > 0 and overlap.height > 0:
                is_dup = True
                break
        if is_dup:
            continue
        used_positions.add(pos_key)
        fields.append({
            "label": uf["label"],
            "field_type": "text",
            "cell_rect": {
                "x0": round(rect.x0, 1),
                "y0": round(page_height - rect.y1, 1),
                "x1": round(rect.x1, 1),
                "y1": round(page_height - rect.y0, 1),
            },
            "fill_point": {
                "x": round(fill_x, 1),
                "y": round(fill_y, 1),
            },
            "font_size": 8,
        })

    # Checkbox fields
    for cb in checkbox_only:
        label = find_label_for_checkbox(cb, words)
        center_x = (cb.x0 + cb.x1) / 2 - 3
        center_y = page_height - (cb.y0 + cb.y1) / 2 - 3

        pos_key = (round(center_x, 0), round(center_y, 0))
        if pos_key in used_positions:
            continue
        used_positions.add(pos_key)

        fields.append({
            "label": label.strip() if label else "(checkbox)",
            "field_type": "checkbox",
            "cell_rect": {
                "x0": round(cb.x0, 1),
                "y0": round(page_height - cb.y1, 1),
                "x1": round(cb.x1, 1),
                "y1": round(page_height - cb.y0, 1),
            },
            "fill_point": {
                "x": round(center_x, 1),
                "y": round(center_y, 1),
            },
            "font_size": min(8, int(cb.height - 2)),
        })

    doc.close()

    # Compute confidence: how well did detection work?
    text_fields = [f for f in fields if f["field_type"] == "text"]
    checkbox_fields = [f for f in fields if f["field_type"] == "checkbox"]
    text_word_count = len(words)
    confidence = "high"
    if text_word_count > 20 and len(text_fields) < 3:
        confidence = "low"
    elif text_word_count > 20 and len(text_fields) < 8:
        confidence = "medium"

    return {
        "page": page_num,
        "page_size": {"width": round(page_width, 1), "height": round(page_height, 1)},
        "detection": {
            "grid_lines": {"horizontal": len(h_lines), "vertical": len(v_lines)},
            "grid_cells": len(grid_cells),
            "filled_rects": len(filled_rects),
            "char_box_groups": len(char_box_groups),
            "checkboxes": len(checkbox_only),
            "underline_fields": len(underline_fields),
            "used_ocr": used_ocr,
        },
        "confidence": confidence,
        "total_fields": len(fields),
        "text_fields": len(text_fields),
        "checkbox_fields": len(checkbox_fields),
        "fields": fields,
    }


# ---------------------------------------------------------------------------
# Multi-page detection
# ---------------------------------------------------------------------------

def detect_all_pages(pdf_path):
    """Detect fields across all pages of a PDF."""
    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    doc.close()

    all_results = []
    for page_num in range(num_pages):
        result = detect_fields(pdf_path, page_num)
        all_results.append(result)

    return {
        "num_pages": num_pages,
        "pages": all_results,
        "total_fields": sum(r["total_fields"] for r in all_results),
        "total_text_fields": sum(r["text_fields"] for r in all_results),
        "total_checkbox_fields": sum(r["checkbox_fields"] for r in all_results),
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def annotate_page(pdf_path, page_num, fields_data, output_png, dpi=200):
    """Render page with field detection overlays."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    page_height = page.rect.height

    for field in fields_data["fields"]:
        cr = field["cell_rect"]
        mu_rect = fitz.Rect(cr["x0"], page_height - cr["y1"], cr["x1"], page_height - cr["y0"])
        fp = field["fill_point"]
        mu_fill_x = fp["x"]
        mu_fill_y = page_height - fp["y"]

        if field["field_type"] == "text":
            color = (0, 0.6, 0) if field.get("char_boxes") else (0, 0, 1)
            page.draw_rect(mu_rect, color=color, width=0.5)
            page.draw_circle(fitz.Point(mu_fill_x, mu_fill_y), 2, color=(1, 0, 0), fill=(1, 0, 0))
            # Draw label as tiny text
            label_text = field["label"][:30]
            try:
                page.insert_text(
                    fitz.Point(mu_rect.x0 + 1, mu_rect.y0 + 6),
                    label_text, fontsize=4, color=color,
                )
            except Exception:
                pass
        elif field["field_type"] == "checkbox":
            page.draw_rect(mu_rect, color=(0, 0.7, 0), width=0.5)
            page.draw_circle(fitz.Point(mu_fill_x, mu_fill_y), 2, color=(0, 0.7, 0), fill=(0, 0.7, 0))

    pix = page.get_pixmap(dpi=dpi)
    pix.save(output_png)
    doc.close()


def grid_overlay(pdf_path, page_num, output_png, dpi=150, step=50):
    """Render page with coordinate grid overlay for visual positioning."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    pw = page.rect.width
    ph = page.rect.height

    for x in range(0, int(pw) + 1, step):
        color = (0.7, 0.7, 1.0) if x % 100 else (0.3, 0.3, 1.0)
        width = 0.3 if x % 100 else 0.5
        page.draw_line(fitz.Point(x, 0), fitz.Point(x, ph), color=color, width=width)
        if x % 100 == 0:
            page.insert_text(fitz.Point(x + 1, 8), str(x), fontsize=6, color=(0, 0, 1))

    for y_mu in range(0, int(ph) + 1, step):
        pdf_y = ph - y_mu
        color = (1.0, 0.7, 0.7) if y_mu % 100 else (1.0, 0.3, 0.3)
        width = 0.3 if y_mu % 100 else 0.5
        page.draw_line(fitz.Point(0, y_mu), fitz.Point(pw, y_mu), color=color, width=width)
        if y_mu % 100 == 0:
            page.insert_text(fitz.Point(1, y_mu + 8), f"y={int(pdf_y)}", fontsize=6, color=(1, 0, 0))

    pix = page.get_pixmap(dpi=dpi)
    pix.save(output_png)
    doc.close()
    print(json.dumps({
        "page": page_num,
        "page_size": {"width": round(pw, 1), "height": round(ph, 1)},
        "grid_step": step,
        "dpi": dpi,
        "output": output_png,
        "note": "X labels in blue (PDF x-coordinate). Y labels in red (PDF y-coordinate, bottom-up).",
    }))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Detect fillable fields in graphical PDFs")
    parser.add_argument("input_pdf", help="Path to input PDF")
    parser.add_argument("--page", type=int, default=0, help="Page number (default: 0)")
    parser.add_argument("--all-pages", action="store_true", help="Detect fields on all pages")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    parser.add_argument("--annotate", help="Save annotated PNG showing detected fields")
    parser.add_argument("--grid-overlay", help="Save coordinate grid overlay PNG")
    args = parser.parse_args()

    if not Path(args.input_pdf).exists():
        print(json.dumps({"error": f"PDF not found: {args.input_pdf}"}), file=sys.stderr)
        sys.exit(1)

    if args.grid_overlay:
        grid_overlay(args.input_pdf, args.page, args.grid_overlay)
        return

    if args.all_pages:
        result = detect_all_pages(args.input_pdf)
    else:
        result = detect_fields(args.input_pdf, args.page)

    if args.annotate:
        if args.all_pages:
            # Annotate only the first page with most fields
            best = max(result["pages"], key=lambda p: p["total_fields"])
            annotate_page(args.input_pdf, best["page"], best, args.annotate)
        else:
            annotate_page(args.input_pdf, args.page, result, args.annotate)
        print(f"Annotated image saved to {args.annotate}", file=sys.stderr)

    indent = 2 if args.pretty else None
    print(json.dumps(result, indent=indent))


if __name__ == "__main__":
    main()
