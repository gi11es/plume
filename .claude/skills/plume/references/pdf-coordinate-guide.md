# PDF Coordinate System & Positioning Guide

## Coordinate System Basics

- PDF origin `(0, 0)` is at the **bottom-left** corner of the page
- X increases to the right, Y increases upward
- Standard page sizes:
  - **A4**: 595.28 x 841.89 points
  - **Letter**: 612 x 792 points
- 1 point = 1/72 inch

## Text Positioning in Overlay Mode

### Baseline Offset

When placing text inside a colored rectangle field:

```
baseline_y = field_rect_y + 2.4
```

The text baseline sits approximately **2.4 points above** the bottom edge of the field rectangle. This accounts for font descenders and visual centering.

### Left Padding

```
text_x = field_rect_x + 5.6
```

Text starts approximately **5.6 points** from the left edge of the field. This provides consistent left padding that matches typical form aesthetics.

### Examples

Given a field rectangle at `(134.36, 717.98)` with size `418.39 x 13.5`:
- Text position: `x = 134.36 + 5.6 = 140.0`, `y = 717.98 + 2.4 = 720.4`
- Font: Helvetica 10pt

Given a full-width field at `(42.52, 492.89)` with size `510.24 x 13.5`:
- Text position: `x = 42.52 + 5.5 = 48.0`, `y = 492.89 + 2.4 = 495.3`

## Checkbox Positioning

Checkboxes are typically drawn as small rectangles (8-16pt per side).

### Placing an X Mark

```
x_mark_x = checkbox_left + 1.8
x_mark_y = checkbox_bottom + 1.8
```

Use **Helvetica-Bold 10pt** for the "X" character. The 1.8pt offset centers the mark within a typical 11.5pt checkbox.

### Checkbox Detection

Checkboxes in graphical PDFs appear as:
1. Small filled/stroked rectangles (8-16pt sides)
2. 3D beveled buttons using light/dark triangle pairs (0.2510g dark, 0.7510g light)
3. Simple square outlines

## Content Stream Patterns

### Blue Input Fields
```
0.9098 0.9412 0.9882 rg    ← light blue fill color
134.3622 717.9772 418.3937 13.5000 re    ← rectangle
f*    ← fill
```

### 3D Checkbox (Beveled)
```
0.2510 g    ← dark gray (top-left shadow)
136.7397 574.1772 m
148.2397 574.1772 l
...
f
0.7510 g    ← light gray (bottom-right highlight)
148.2397 562.6772 m
...
f
```

### Text Labels
```
BT
0 g
42.5197 720.3672 Td    ← position
/F1 10.0000 Tf    ← font
(Nom et pr\351nom :) Tj    ← text with octal escapes
ET
```

## Font Handling

- Default overlay font: **Helvetica** (available in all PDF readers)
- For accented characters (é, è, à, etc.): Helvetica handles Latin-1 natively
- For non-Latin scripts: may need to register TTF fonts via reportlab

## Troubleshooting Alignment

1. **Text too high/low**: Adjust the Y baseline offset (try values between 1.5 and 3.5)
2. **Text too far left/right**: Adjust the X padding (try values between 4.0 and 7.0)
3. **Checkbox misplaced**: Verify you're using the correct checkbox rectangle (not a nearby one)
4. **Accents garbled**: Check font encoding — use `latin-1` for French/Spanish/German forms

## Multi-Page Forms

- Each page has its own coordinate system
- Field positions are relative to that page's MediaBox
- Specify `"page": N` (0-indexed) in the fill spec for each field
