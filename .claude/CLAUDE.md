# Plume — PDF Form Filler

This project provides a Claude Code skill (`/plume`) that fills PDF forms automatically.

## Project Layout

- `scripts/extract.py` — Parse PDF structure (AcroForm fields + content stream graphics)
- `scripts/fill.py` — Fill PDFs via AcroForm field values or text overlay
- `scripts/verify.py` — Verify filled text is correctly positioned
- `scripts/requirements.txt` — Python dependencies (pypdf, reportlab)
- `memory/user-info.json` — Persistent user data (remembered across sessions)
- `.claude/skills/plume/SKILL.md` — The `/plume` slash command definition
- `tests/` — Test suite with sample PDF forms

## Conventions

- All scripts accept `--pretty` for human-readable JSON output
- Scripts communicate via JSON on stdout; errors go to stderr
- Fill specs use the format defined in `scripts/fill.py` docstring
- Two fill strategies: `overlay` (graphical PDFs) and `acroform` (interactive form PDFs)
- The `memory/user-info.json` file should never contain real personal data in the repo — it ships with fake example data
