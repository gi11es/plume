#!/usr/bin/env python3
"""Capture a handwritten signature via a GUI drawing canvas.

Opens a tkinter window where the user draws their signature with the mouse.
Saves the result as a PNG with a transparent background (black strokes on alpha).

Usage:
    python capture_signature.py <output.png> [--width 600] [--height 200]

The user can:
  - Draw with mouse (click-and-drag)
  - Click "Clear" to start over
  - Click "Done" to save and exit
  - Close the window to cancel (exits with code 1)
"""

import argparse
import json
import sys
import tkinter as tk

from PIL import Image, ImageDraw


def capture_signature(output_path: str, width: int = 600, height: int = 200) -> bool:
    """Open a signature capture window. Returns True if signature was saved."""
    result = {"saved": False}

    root = tk.Tk()
    root.title("Sign here")
    root.resizable(False, False)

    # White canvas for visual drawing
    canvas = tk.Canvas(root, width=width, height=height, bg="white",
                       cursor="pencil", highlightthickness=1, highlightbackground="#999")
    canvas.pack(padx=10, pady=(10, 5))

    # Parallel PIL image for pixel-accurate export (transparent background)
    pil_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(pil_image)

    # Stroke tracking
    last_point = [None, None]
    stroke_count = [0]

    def on_press(event):
        last_point[0] = event.x
        last_point[1] = event.y

    def on_drag(event):
        x0, y0 = last_point
        x1, y1 = event.x, event.y
        if x0 is not None:
            # Draw on tkinter canvas (visual feedback)
            canvas.create_line(x0, y0, x1, y1, fill="black", width=2, smooth=True,
                               capstyle=tk.ROUND, joinstyle=tk.ROUND)
            # Draw on PIL image (export)
            draw.line([(x0, y0), (x1, y1)], fill=(0, 0, 0, 255), width=2)
        last_point[0] = x1
        last_point[1] = y1
        stroke_count[0] += 1

    def on_release(event):
        last_point[0] = None
        last_point[1] = None

    def clear():
        canvas.delete("all")
        nonlocal pil_image, draw
        pil_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(pil_image)
        stroke_count[0] = 0

    def done():
        if stroke_count[0] < 5:
            # Too few strokes — probably accidental click
            return
        # Crop to bounding box of actual content (with padding)
        bbox = pil_image.getbbox()
        if bbox:
            pad = 10
            x0 = max(0, bbox[0] - pad)
            y0 = max(0, bbox[1] - pad)
            x1 = min(width, bbox[2] + pad)
            y1 = min(height, bbox[3] + pad)
            cropped = pil_image.crop((x0, y0, x1, y1))
            cropped.save(output_path, "PNG")
        else:
            pil_image.save(output_path, "PNG")
        result["saved"] = True
        root.destroy()

    def cancel():
        root.destroy()

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)

    # Buttons
    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=(5, 10))

    tk.Button(btn_frame, text="Clear", command=clear, width=10).pack(side=tk.LEFT, padx=5)
    tk.Button(btn_frame, text="Done", command=done, width=10).pack(side=tk.LEFT, padx=5)
    tk.Button(btn_frame, text="Cancel", command=cancel, width=10).pack(side=tk.LEFT, padx=5)

    # Instruction label
    tk.Label(root, text="Draw your signature above, then click Done",
             fg="#666", font=("Helvetica", 11)).pack(pady=(0, 8))

    root.protocol("WM_DELETE_WINDOW", cancel)
    root.mainloop()

    return result["saved"]


def main():
    parser = argparse.ArgumentParser(description="Capture a handwritten signature")
    parser.add_argument("output", help="Output PNG path")
    parser.add_argument("--width", type=int, default=600, help="Canvas width (default: 600)")
    parser.add_argument("--height", type=int, default=200, help="Canvas height (default: 200)")
    args = parser.parse_args()

    saved = capture_signature(args.output, args.width, args.height)
    if saved:
        print(json.dumps({"status": "saved", "path": args.output}))
    else:
        print(json.dumps({"status": "cancelled"}))
        sys.exit(1)


if __name__ == "__main__":
    main()
