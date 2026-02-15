---
name: plume
description: Fill any PDF form — graphical overlays or interactive AcroForm fields. Remembers your info across sessions.
argument-hint: <path-to-pdf>
allowed-tools: Bash, Read, Write, Edit, AskUserQuestion
---

# Plume — PDF Form Filler

You are filling a PDF form for the user. Follow these steps precisely.

## Constants

First, determine PLUME_ROOT by finding the directory containing this skill. The skill lives at `<PLUME_ROOT>/.claude/skills/plume/SKILL.md`, so resolve upward from the skill file location. Alternatively, run:
```bash
PLUME_ROOT=$(python3 -c "
import subprocess, os
result = subprocess.run(['readlink', '-f', os.path.expanduser('~/.claude/skills/plume')], capture_output=True, text=True)
skill_path = result.stdout.strip() if result.returncode == 0 else ''
if skill_path:
    print(os.path.dirname(os.path.dirname(os.path.dirname(skill_path))))
else:
    for d in [os.path.expanduser('~/Documents/personal/plume'), os.path.expanduser('~/plume')]:
        if os.path.isdir(d): print(d); break
")
SCRIPTS=$PLUME_ROOT/scripts
MEMORY=$PLUME_ROOT/memory/user-info.json
PYTHON=$PLUME_ROOT/.venv/bin/python
if [ ! -x "$PYTHON" ]; then PYTHON=python3; fi
```

## Step 1: Validate Input

The argument `$ARGUMENTS` is the path to the PDF file.

- Verify the file exists and is a PDF
- If no argument given, ask the user for the PDF path
- Set `INPUT_PDF` to the absolute path

## Step 2: Extract PDF Structure

Run:
```bash
$PYTHON $SCRIPTS/extract.py "$INPUT_PDF" --pretty
```

This outputs JSON with:
- `has_acroform`: whether the PDF has interactive form fields
- `acroform_fields`: list of AcroForm fields (name, type, options, rect)
- `pages[].labels`: text labels with positions
- `pages[].field_rectangles`: colored rectangles (input fields in graphical PDFs)
- `pages[].checkboxes`: small rectangles that are checkboxes
- `pages[].label_field_associations`: labels matched to nearby field rects
- `recommended_strategy`: "acroform" or "overlay"

Save this output — you'll need it for field mapping.

## Step 2b: Detect Fields in Graphical PDFs

If the PDF is graphical (no AcroForm fields, or `recommended_strategy` is "overlay"), run the field detector for more precise positioning:

```bash
$PYTHON $SCRIPTS/detect_fields.py "$INPUT_PDF" --all-pages --pretty
```

This uses multiple detection strategies:
1. **Grid-line detection** — finds table grids formed by horizontal/vertical lines
2. **Filled-rectangle detection** — finds white/colored input boxes on colored backgrounds
3. **Character-box merging** — groups of small adjacent rectangles → single text fields
4. **Checkbox detection** — small square shapes
5. **Underline detection** — horizontal lines with labels (write-on-line fields)
6. **OCR fallback** — for image-only PDFs (requires Tesseract)

Each detected field includes:
- `label`: the field label text
- `fill_point`: {x, y} in PDF coordinates where text should be placed
- `cell_rect`: bounding rectangle in PDF coordinates
- `font_size`: recommended size based on cell height
- `char_boxes`: number of character boxes (if applicable)

Check the `confidence` field:
- **high**: Auto-detection found good field coverage — use detected positions directly
- **medium/low**: Detection is incomplete — generate a grid overlay for visual positioning:
  ```bash
  $PYTHON $SCRIPTS/detect_fields.py "$INPUT_PDF" --grid-overlay /tmp/plume_grid.png --page 0
  ```
  Then **Read the grid overlay image** to visually determine exact coordinates.

Also generate an annotated image to verify detection accuracy:
```bash
$PYTHON $SCRIPTS/detect_fields.py "$INPUT_PDF" --annotate /tmp/plume_annotated.png --page 0
```
Blue boxes = text fields, green = character-box groups/checkboxes, red dots = fill points.

## Step 3: Understand the Form

**Read the PDF visually** using the Read tool to see the actual layout. Then:

1. List every fillable field you can identify (from the extract JSON, detect_fields output, and visual inspection)
2. For each field, note:
   - The label text (e.g., "Nom et prénom :")
   - The field type (text input, checkbox, dropdown, date, signature)
   - For overlay: the target rectangle coordinates from detect_fields
   - For AcroForm: the field name
3. Present the field list to the user for confirmation

## Step 4: Load Memory

Read the stored user info:
```bash
cat $MEMORY
```

Match stored data to form fields. For example:
- "Nom" / "Name" → `personal.full_name`
- "Adresse" / "Address" → `address.full`
- "Code postal" / "ZIP" → `address.zip`
- "Ville" / "City" → `address.city`

## Step 5: Ask for Missing Data

Show the user what you plan to fill (pre-populated from memory) and ask for any missing values.

Use `AskUserQuestion` for structured choices (checkboxes, dropdowns). For free text, just ask in conversation.

**Important**: Always confirm the full mapping before filling. The user must approve.

## Step 5b: Handle Signature Fields

If the form has a signature field:

0. First, ask the user whether to include a digital signature:
   > "Include a digital signature image? Note: some organizations require a wet ink
   > signature on the physical document. Include digital signature?"
   If the user declines, skip signature image placement entirely (still fill any
   AcroForm signature text fields if applicable).

1. Determine whose signature is needed. Forms may require signatures from:
   - The applicant (default) → stored as `$PLUME_ROOT/memory/signature.png`
   - A spouse/partner → stored as `$PLUME_ROOT/memory/signature_spouse.png`
   - A child → stored as `$PLUME_ROOT/memory/signature_child_<firstname>.png`
   (e.g., `signature_child_lucas.png`)

2. Check if a saved signature exists for the relevant person:
   ```bash
   ls $PLUME_ROOT/memory/signature.png 2>/dev/null          # applicant
   ls $PLUME_ROOT/memory/signature_spouse.png 2>/dev/null    # spouse
   ls $PLUME_ROOT/memory/signature_child_lucas.png 2>/dev/null  # child (example)
   ```

3. If no saved signature, prompt the user to draw one:
   ```bash
   $PYTHON $SCRIPTS/capture_signature.py /tmp/plume_signature.png
   ```
   This opens a GUI window where the user draws their signature (black on transparent background).

4. After capture, ask the user:
   - "Save this signature for future forms?" → cp to the appropriate path in `$PLUME_ROOT/memory/`
     (e.g., `signature.png` for applicant, `signature_spouse.png` for spouse)
   - "Use for this form only?" → use the temp path

5. If a saved signature exists, show it to the user (using Read on the PNG) and ask:
   - "Use your saved signature?" → use the saved path
   - "Draw a new one?" → run capture again

5. In the fill spec, use type `"signature"` for signature fields:
   ```json
   {"type": "signature", "image_path": "/path/to/signature.png", "x": 130, "y": 190, "page": 0, "width": 150, "height": 50}
   ```

6. **Preventing overlap on "both" strategy forms**: When using a signature image overlay on a form that also has AcroForm fields (strategy "both"), set the AcroForm signature text field value to `""` to prevent the typed name from rendering underneath the signature image.

7. **Signature placement — target the blank area**: Signature fields often have a text label
   (e.g., "Signature of Employee") occupying part of the field. When positioning the signature
   image, do NOT place it at the field's left edge if a label is there. Instead:
   - Identify the label text extent (x_start to x_end) and any other text in the row (e.g., "Date")
   - Place the signature image's x at label_x_end + 10pt (clear of the label)
   - Set width to fill the available blank space (up to the next label or field edge minus 10pt padding)
   - Set height to fit within the field row height (typically 15-25pt for standard forms)
   - The goal: the signature image should occupy the largest blank rectangle in the signing area,
     with no overlap on any printed text or labels

## Step 5c: Handle ID Photo Fields

If the form has a photo area (common on visa applications, ID forms):

1. Ask the user whether to include a digital photo:
   > "Include an ID photo in the form? Note: some organizations require a physical
   > photobooth photo glued to the form. Include digital photo?"
   If the user declines, skip photo placement entirely.

2. Determine whose photo is needed. Forms may require photos for:
   - The applicant (default) → stored as `$PLUME_ROOT/memory/photo.png`
   - A spouse/partner → stored as `$PLUME_ROOT/memory/photo_spouse.png`
   - A child → stored as `$PLUME_ROOT/memory/photo_child_<firstname>.png`
   (e.g., `photo_child_lucas.png`)

3. Check if a saved photo exists for the relevant person:
   ```bash
   ls $PLUME_ROOT/memory/photo.png 2>/dev/null            # applicant
   ls $PLUME_ROOT/memory/photo_spouse.png 2>/dev/null      # spouse
   ls $PLUME_ROOT/memory/photo_child_lucas.png 2>/dev/null # child (example)
   ```

4. If no saved photo, ask the user to provide one:
   - Accept a file path to an existing photo
   - The photo should be a front-facing passport-style image (35x45mm)

5. After receiving the photo, ask the user:
   - "Save this photo for future forms?" → cp to the appropriate path in `$PLUME_ROOT/memory/`
     (e.g., `photo.png` for applicant, `photo_spouse.png` for spouse)
   - "Use for this form only?" → use the provided path

6. If a saved photo exists, show it to the user (using Read on the PNG) and ask:
   - "Use your saved photo?" → use the saved path
   - "Provide a different one?" → accept new path

7. In the fill spec, use type `"photo"`:
   ```json
   {"type": "photo", "image_path": "/path/to/photo.png", "x": 460, "y": 700, "page": 0, "width": 99, "height": 128}
   ```
   Default dimensions: 99pt x 128pt (35mm x 45mm passport photo).

8. **Photo placement**: The photo box is typically a clearly marked rectangle on the form
   (often labeled "Photo" or "PHOTO"). Place the image to fill the photo box with 2pt
   inner padding on each side.

## Step 6: Create Fill Specification

Create a JSON fill spec at `/tmp/plume_fill_spec.json`:

### For AcroForm PDFs:
```json
{
  "strategy": "acroform",
  "fields": [
    {"name": "field_name", "value": "text value", "type": "text"},
    {"name": "checkbox_field", "value": "Yes", "type": "checkbox"},
    {"name": "dropdown_field", "value": "Option A", "type": "choice"}
  ]
}
```

### For Graphical (overlay) PDFs:
```json
{
  "strategy": "overlay",
  "fields": [
    {"value": "MARTIN Sophie", "x": 140, "y": 720.4, "page": 0, "font_size": 10, "type": "text"},
    {"value": "X", "x": 138.5, "y": 564.5, "page": 0, "font_size": 10, "font": "Helvetica-Bold", "type": "checkbox"}
  ]
}
```

### Overlay Positioning Rules:
- **Text baseline**: `field_rect_y + 2.4` (text sits ~2.4pt above field bottom)
- **Left padding**: `field_rect_x + 5.6` (text starts ~5.6pt from field left edge)
- **Checkbox X**: `checkbox_left + 1.8`, `checkbox_bottom + 1.8`
- **Font**: Helvetica 10pt for text, Helvetica-Bold 10pt for checkbox marks

### Checkbox Positioning Workflow:
1. Run `detect_fields.py --annotate` to see detected checkbox locations (green boxes)
2. Match each detected checkbox to its label by reading the PDF visually
3. Use the `fill_point` from detect_fields.py as the starting position
4. Verify by generating the filled PDF and reading it — the X must sit inside the checkbox square
5. Fine-tune: if the X appears offset, adjust x/y by 1-2pt
6. Common issue: forms with checkboxes at the same y-coordinate for different sections
   (e.g., Sex and Marital Status side by side) — verify which x-position maps to which option

See `references/pdf-coordinate-guide.md` for detailed positioning guidance.

## Step 7: Fill the PDF

```bash
$PYTHON $SCRIPTS/fill.py "$INPUT_PDF" /tmp/plume_fill_spec.json "$OUTPUT_PDF"
```

Set `OUTPUT_PDF` to the input path with `_filled` suffix (e.g., `form_filled.pdf`).

## Step 8: Verify — Visual Check

**Read the output PDF** using the Read tool to visually inspect the result. Check:
- All fields are filled
- Text is inside the correct field boundaries
- Checkboxes are properly marked
- No overlapping or misaligned text

## Step 9: Verify — Programmatic Check

```bash
$PYTHON $SCRIPTS/verify.py "$OUTPUT_PDF" /tmp/plume_fill_spec.json --pretty
```

Review the pass/fail report.

## Step 10: Iterate if Needed

If any fields are misaligned:
1. Adjust the coordinates in the fill spec (refer to the coordinate guide)
2. Re-run fill and verify
3. Maximum 3 iterations
4. If still off after 3 tries, show the user the current state and ask for guidance

## Step 11: Save Memory

Update `$MEMORY` with any new user data entered during this session. Merge new fields into existing categories — never overwrite existing data unless the user explicitly provides updated values.

## Step 12: Report

Tell the user:
- Output file path
- Which strategy was used (AcroForm vs overlay)
- How many fields were filled
- Any fields that couldn't be filled automatically
- Whether a signature was captured and saved
