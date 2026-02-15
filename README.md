<p align="center">
  <img src="assets/logo.png" alt="Plume" width="200">
</p>

<h1 align="center">Plume</h1>

<p align="center">
  <strong>A Claude Code skill that fills any PDF form — smart enough to read the form, remember your info, and verify its own work.</strong>
</p>

<p align="center">
  <a href="#how-it-works">How it works</a> &bull;
  <a href="#installation">Installation</a> &bull;
  <a href="#usage">Usage</a> &bull;
  <a href="#architecture">Architecture</a> &bull;
  <a href="#testing">Testing</a>
</p>

---

## What is this?

Plume is a [Claude Code](https://claude.ai/claude-code) skill that turns PDF form filling from a tedious chore into a single command. Point it at any PDF form — whether it's a French *autorisation parentale*, a US tax form, or a German *Anmeldung* — and it figures out the fields, fills them with your data, and verifies the result.

```
/plume ~/Downloads/tax-form-2026.pdf
```

It handles **both** types of PDF forms:
- **Interactive forms** (AcroForm) — text fields, checkboxes, dropdowns filled directly
- **Graphical forms** — "dumb" PDFs where fields are just colored rectangles, filled via precise text overlay

## How it works

```
PDF in → Extract structure → Map fields → Fill (overlay or AcroForm) → Verify → PDF out
             │                    │                                        │
             │                    ▼                                        │
             │              Load memory ──→ Ask only for                   │
             │              (user-info.json)  missing data                 │
             │                                                             │
             ▼                                                             ▼
        Content stream                                              Visual + programmatic
        parsing + AcroForm                                          alignment check
        field detection                                             (self-correction loop)
```

1. **Extract** — Parses the PDF content stream for text labels, colored rectangles, and checkboxes. Also detects AcroForm interactive fields.
2. **Map** — Associates labels with their input fields using spatial proximity.
3. **Remember** — Loads your previously saved info (name, address, etc.) and only asks for what's missing.
4. **Fill** — Either sets AcroForm field values directly, or creates a reportlab overlay merged onto the original.
5. **Verify** — Reads the output PDF both visually (multimodal) and programmatically to check alignment. Self-corrects up to 3 times.

## Installation

### Prerequisites

```bash
pip install pypdf reportlab
```

### Setup

Clone the repo and create a symlink so `/plume` works from any directory:

```bash
git clone https://github.com/your-username/plume.git ~/Documents/personal/plume
ln -s ~/Documents/personal/plume/.claude/skills/plume ~/.claude/skills/plume
```

### First run

On first use, Plume will ask for your personal info and save it for future forms. You can also pre-populate `memory/user-info.json`:

```json
{
  "personal": {
    "last_name": "MARTIN",
    "first_name": "Sophie",
    "full_name": "MARTIN Sophie"
  },
  "address": {
    "street": "12 rue des Lilas",
    "zip": "75011",
    "city": "Paris"
  }
}
```

## Usage

```
/plume path/to/form.pdf
```

Plume will:
1. Analyze the PDF structure
2. Show you the detected fields
3. Pre-fill from memory and ask for missing data
4. Fill the PDF and save it as `form_filled.pdf`
5. Visually verify the output

## Architecture

```
plume/
├── scripts/
│   ├── extract.py      # PDF structure parser (AcroForm + content stream)
│   ├── fill.py         # Form filler (overlay + AcroForm strategies)
│   ├── verify.py       # Alignment verification
│   └── requirements.txt
├── memory/
│   └── user-info.json  # Your saved data (gitignored in real use)
├── tests/
│   ├── fixtures/       # Sample PDF forms for testing
│   ├── test_extract.py
│   ├── test_fill.py
│   └── test_verify.py
├── .claude/
│   ├── skills/plume/
│   │   ├── SKILL.md    # The /plume slash command
│   │   └── references/
│   │       └── pdf-coordinate-guide.md
│   └── CLAUDE.md
└── assets/
    └── logo.png
```

### Scripts

| Script | Purpose |
|--------|---------|
| `extract.py` | Parses PDF content streams for labels, rectangles, checkboxes. Detects AcroForm fields. Outputs JSON. |
| `fill.py` | Fills PDFs via AcroForm field setting or reportlab text overlay + merge. |
| `verify.py` | Checks filled text positions against target field bounds. Returns pass/fail report. |

### Fill Strategies

| Strategy | When | How |
|----------|------|-----|
| `acroform` | PDF has interactive form fields | Sets field values via pypdf |
| `overlay` | PDF has only graphical fields | Creates reportlab canvas overlay, merges onto original |
| `both` | PDF has some of each | AcroForm fill first, then overlay for graphical fields |
| `auto` | Default | Detects which strategy to use |

## Testing

```bash
pytest tests/ -v
```

The test suite includes ~12 PDF forms from various government sources (IRS, UK HMRC, French CERFA, etc.) covering:
- Interactive AcroForm PDFs with text fields, checkboxes, and dropdowns
- Graphical PDFs with colored rectangles as fields
- Multi-page forms
- Multiple languages (English, French, Spanish, German)

## License

MIT
