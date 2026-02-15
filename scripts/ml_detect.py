#!/usr/bin/env python3
"""ML-based document layout detection for PDF forms.

Supports three model backends:
1. DocLayout-YOLO — fast YOLO-based document layout detection
2. Florence-2 — Microsoft vision-language model with phrase grounding
3. Grounding DINO — open-set text-prompted object detection

Usage:
    python ml_detect.py assess                          # Show hardware + recommended models
    python ml_detect.py detect <pdf> [--model NAME]     # Detect fields in a PDF
    python ml_detect.py benchmark [--model NAME]        # Benchmark models on test dataset

Install ML dependencies first:  pip install -e ".[ml]"
"""

import argparse
import io
import json
import os
import shutil
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "doclayout-yolo": {
        "pip": "doclayout-yolo",
        "model_id": "juliozhao/DocLayout-YOLO-DocStructBench",
        "size_mb": 100,
        "min_ram_gb": 2,
        "supports": ["gpu", "mps", "cpu"],
        "description": "Fast YOLO-based document layout detection (~12ms/image)",
    },
    "florence2-base": {
        "pip": "transformers accelerate",
        "model_id": "microsoft/Florence-2-base",
        "size_mb": 900,
        "min_ram_gb": 4,
        "supports": ["gpu", "mps", "cpu"],
        "description": "Microsoft vision-language model, 230M params",
    },
    "florence2-large": {
        "pip": "transformers accelerate",
        "model_id": "microsoft/Florence-2-large",
        "size_mb": 2300,
        "min_ram_gb": 8,
        "supports": ["gpu", "mps"],
        "description": "Microsoft vision-language model, 770M params",
    },
    "grounding-dino": {
        "pip": "transformers",
        "model_id": "IDEA-Research/grounding-dino-base",
        "size_mb": 900,
        "min_ram_gb": 4,
        "supports": ["gpu", "mps", "cpu"],
        "description": "Open-set text-prompted object detection",
    },
    "table-transformer": {
        "pip": "transformers",
        "model_id": "microsoft/table-transformer-structure-recognition-v1.1-all",
        "size_mb": 115,
        "min_ram_gb": 2,
        "supports": ["gpu", "mps", "cpu"],
        "description": "DETR-based table structure recognition — rows, columns, cells",
    },
    "yolov8-form": {
        "pip": "ultralytics",
        "model_id": "foduucom/web-form-ui-field-detection",
        "size_mb": 23,
        "min_ram_gb": 2,
        "supports": ["gpu", "mps", "cpu"],
        "description": "YOLOv8s form UI field detection — text fields, buttons, checkboxes",
    },
    "ffdnet-l": {
        "pip": "ultralytics",
        "model_id": "jbarrow/FFDNet-L",
        "size_mb": 100,
        "min_ram_gb": 2,
        "supports": ["gpu", "mps", "cpu"],
        "description": "YOLO11 form field detector — text inputs, checkboxes, signatures",
    },
}

# Categories we detect and their prompt variants per model
DETECTION_PROMPTS = {
    "photo_box": "photo box",
    "checkbox": "checkbox",
    "text_field": "text input field",
    "signature": "signature area",
}

# Normalize ML output labels → canonical types matching ground truth
LABEL_NORMALISATION = {
    # DocLayout-YOLO class names (via CLASS_MAP)
    "photo_box": "photo_box",
    "grid": "grid",
    "text_field": "text_field",
    # Florence-2 / Grounding DINO free-form labels
    "photo box": "photo_box",
    "photo": "photo_box",
    "photograph": "photo_box",
    "portrait": "photo_box",
    "id photo": "photo_box",
    "figure": "photo_box",
    "checkbox": "checkbox",
    "check box": "checkbox",
    "text input field": "text_field",
    "text field": "text_field",
    "text input": "text_field",
    "text": "text_field",
    "signature area": "signature",
    "signature": "signature",
    "signature box": "signature",
    # Table Transformer class names
    "table": "grid",
    "table column": "grid",
    "table row": "grid",
    "table column header": "text_field",
    "table projected row header": "text_field",
    "table spanning cell": "text_field",
    # FFDNet class names
    "textbox": "text_field",
    "choicebutton": "checkbox",
    # YOLOv8 form field class names (approximate — logged on first load)
    "name": "text_field",
    "number": "text_field",
    "email": "text_field",
    "password": "text_field",
    "button": "text_field",
    "radio": "checkbox",
    "radio bullet": "checkbox",
}


def normalise_label(raw_label):
    """Map an ML detection label to a canonical ground-truth type."""
    key = raw_label.strip().lower()
    return LABEL_NORMALISATION.get(key, key)


# ---------------------------------------------------------------------------
# Hardware assessment
# ---------------------------------------------------------------------------

def assess_hardware():
    """Detect hardware capabilities and recommend ML models."""
    result = {
        "cpu_cores": os.cpu_count(),
        "gpu": None,
        "mps": False,
        "ram_gb": 0,
        "disk_free_gb": 0,
        "tier": "cpu",
        "recommended_models": [],
    }

    # RAM
    try:
        import psutil
        mem = psutil.virtual_memory()
        result["ram_gb"] = round(mem.total / (1024 ** 3), 1)
    except ImportError:
        # Fallback: read from sysctl on macOS or /proc/meminfo on Linux
        try:
            import subprocess
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            result["ram_gb"] = round(int(out.strip()) / (1024 ** 3), 1)
        except Exception:
            pass

    # Disk
    try:
        usage = shutil.disk_usage(Path.home())
        result["disk_free_gb"] = round(usage.free / (1024 ** 3), 1)
    except Exception:
        pass

    # GPU (CUDA)
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_gb = round(props.total_mem / (1024 ** 3), 1)
            result["gpu"] = {
                "name": props.name,
                "vram_gb": vram_gb,
                "compute_capability": f"{props.major}.{props.minor}",
            }
            if vram_gb >= 8:
                result["tier"] = "gpu_high"
            else:
                result["tier"] = "gpu_low"
    except ImportError:
        pass

    # Apple Silicon MPS
    try:
        import torch
        if torch.backends.mps.is_available():
            result["mps"] = True
            if result["tier"] == "cpu":
                result["tier"] = "mps"
    except (ImportError, AttributeError):
        pass

    # Recommend models based on tier
    tier = result["tier"]
    ram = result["ram_gb"]

    for name, info in MODEL_REGISTRY.items():
        if ram >= info["min_ram_gb"]:
            if tier in ("gpu_high", "gpu_low") and "gpu" in info["supports"]:
                result["recommended_models"].append(name)
            elif tier == "mps" and "mps" in info["supports"]:
                result["recommended_models"].append(name)
            elif tier == "cpu" and "cpu" in info["supports"]:
                result["recommended_models"].append(name)

    return result


def get_device():
    """Get the best available torch device."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except (ImportError, AttributeError):
        pass
    try:
        import torch
        return torch.device("cpu")
    except ImportError:
        return "cpu"


# ---------------------------------------------------------------------------
# PDF to image conversion
# ---------------------------------------------------------------------------

def pdf_page_to_image(pdf_path, page_num=0, dpi=200):
    """Render a PDF page to a PIL Image."""
    import fitz
    from PIL import Image

    doc = fitz.open(pdf_path)
    page = doc[page_num]
    pix = page.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    page_width = page.rect.width
    page_height = page.rect.height
    doc.close()
    return img, page_width, page_height


def pixel_to_pdf_coords(bbox_px, img_width, img_height, page_width, page_height):
    """Convert pixel bounding box [x0, y0, x1, y1] to PDF coordinates (bottom-left origin).

    Input bbox is in pixel coords (top-left origin).
    Output is in PDF coords (bottom-left origin).
    """
    scale_x = page_width / img_width
    scale_y = page_height / img_height

    x0 = bbox_px[0] * scale_x
    y0 = page_height - bbox_px[1] * scale_y  # flip Y
    x1 = bbox_px[2] * scale_x
    y1 = page_height - bbox_px[3] * scale_y  # flip Y

    # Ensure y0 < y1 in PDF coords (y0 is bottom, y1 is top)
    return [round(x0, 2), round(min(y0, y1), 2), round(x1, 2), round(max(y0, y1), 2)]


# ---------------------------------------------------------------------------
# IoU computation
# ---------------------------------------------------------------------------

def compute_iou(box_a, box_b):
    """Compute Intersection over Union between two boxes [x0, y0, x1, y1]."""
    x0 = max(box_a[0], box_b[0])
    y0 = max(box_a[1], box_b[1])
    x1 = min(box_a[2], box_b[2])
    y1 = min(box_a[3], box_b[3])

    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0
    return intersection / union


# ---------------------------------------------------------------------------
# Model engine: DocLayout-YOLO
# ---------------------------------------------------------------------------

class DocLayoutYOLOEngine:
    """DocLayout-YOLO document layout detection engine."""

    # Map YOLO class names to our field types
    CLASS_MAP = {
        "figure": "photo_box",
        "table": "grid",
        "text": "text_field",
        "title": "text_field",
        "list": "text_field",
        "caption": "text_field",
        "header": "text_field",
        "footer": "text_field",
        "equation": "text_field",
        "reference": "text_field",
    }

    def __init__(self):
        self.model = None

    def load(self):
        """Load the DocLayout-YOLO model."""
        from doclayout_yolo import YOLOv10
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(
            repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
            filename="doclayout_yolo_docstructbench_imgsz1024.pt",
        )
        self.model = YOLOv10(model_path)

    def detect(self, image, prompts=None):
        """Detect document layout elements.

        Args:
            image: PIL Image
            prompts: ignored (YOLO uses fixed classes)

        Returns:
            list of {"label": str, "bbox": [x0,y0,x1,y1], "confidence": float}
            where bbox is in pixel coordinates (top-left origin).
        """
        if self.model is None:
            self.load()

        device = get_device()
        device_str = str(device) if str(device) != "mps" else "cpu"  # YOLO may not support MPS directly

        results = self.model.predict(
            image,
            imgsz=1024,
            conf=0.2,
            device=device_str,
        )

        detections = []
        for r in results:
            boxes = r.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                cls_name = r.names[cls_id] if cls_id in r.names else f"class_{cls_id}"
                field_type = self.CLASS_MAP.get(cls_name, cls_name)
                conf = float(boxes.conf[i])
                xyxy = boxes.xyxy[i].tolist()
                detections.append({
                    "label": field_type,
                    "raw_class": cls_name,
                    "bbox": [round(v, 2) for v in xyxy],
                    "confidence": round(conf, 4),
                })

        return detections


# ---------------------------------------------------------------------------
# Model engine: Florence-2
# ---------------------------------------------------------------------------

class Florence2Engine:
    """Microsoft Florence-2 vision-language model engine.

    Uses the native Florence2ForConditionalGeneration class from transformers.
    The model's HuggingFace checkpoint was originally saved with trust_remote_code,
    but transformers 5.x has native support via Florence2ForConditionalGeneration.
    """

    def __init__(self, variant="base"):
        self.variant = variant
        model_map = {
            "base": "microsoft/Florence-2-base",
            "large": "microsoft/Florence-2-large",
        }
        self.model_id = model_map.get(variant, variant)
        self.model = None
        self.processor = None

    def load(self):
        """Load Florence-2 model and processor.

        Florence-2 requires trust_remote_code=True and transformers ~4.44.
        On non-CUDA systems, we mock flash_attn (required by model code but
        only used for CUDA flash attention — the model falls back to standard
        attention without it).
        """
        import importlib
        import importlib.machinery
        import torch
        import types

        device = get_device()
        dtype = torch.float16 if str(device) == "cuda" else torch.float32

        # Mock flash_attn on non-CUDA systems (required by model code import
        # but not used at runtime — falls back to standard attention)
        if "flash_attn" not in sys.modules:
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                fa = types.ModuleType("flash_attn")
                fa.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None)
                fa.__version__ = "2.5.0"
                bp = types.ModuleType("flash_attn.bert_padding")
                bp.__spec__ = importlib.machinery.ModuleSpec(
                    "flash_attn.bert_padding", None)
                bp.unpad_input = lambda x, y: (x, None, None, None)
                bp.pad_input = lambda x, *a: x
                fa.bert_padding = bp
                sys.modules["flash_attn"] = fa
                sys.modules["flash_attn.bert_padding"] = bp

        from transformers import AutoModelForCausalLM, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=dtype, trust_remote_code=True,
        ).to(device)

        self.model.eval()
        self.device = device
        self.dtype = dtype

    def detect(self, image, prompts=None):
        """Detect elements using phrase grounding.

        Args:
            image: PIL Image
            prompts: list of text phrases to ground (e.g., ["photo box", "checkbox"])

        Returns:
            list of {"label": str, "bbox": [x0,y0,x1,y1], "confidence": float}
        """
        if self.model is None:
            self.load()

        if prompts is None:
            prompts = list(DETECTION_PROMPTS.values())

        prompt_text = ". ".join(prompts) + "."
        task = "<CAPTION_TO_PHRASE_GROUNDING>"
        full_prompt = task + prompt_text

        inputs = self.processor(
            text=full_prompt, images=image, return_tensors="pt"
        )
        # Move tensors to device
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v
                  for k, v in inputs.items()}

        import torch
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs.get("pixel_values", inputs.get("flattened_patches")),
                max_new_tokens=1024,
                num_beams=3,
            )

        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        # Parse the Florence-2 output
        result = self.processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(image.width, image.height),
        )

        detections = []
        if task in result:
            data = result[task]
            bboxes = data.get("bboxes", [])
            labels = data.get("labels", [])
            for bbox, label in zip(bboxes, labels):
                # Florence returns [x0, y0, x1, y1] in pixel coords
                detections.append({
                    "label": label.strip().lower(),
                    "bbox": [round(v, 2) for v in bbox],
                    "confidence": 1.0,  # Florence doesn't provide confidence scores
                })

        return detections


# ---------------------------------------------------------------------------
# Model engine: Grounding DINO
# ---------------------------------------------------------------------------

class GroundingDINOEngine:
    """Grounding DINO open-set text-prompted detection engine."""

    def __init__(self):
        self.model = None
        self.processor = None

    def load(self):
        """Load Grounding DINO from HuggingFace transformers."""
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        model_id = "IDEA-Research/grounding-dino-base"
        device = get_device()
        dtype = torch.float16 if str(device) == "cuda" else torch.float32

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id, torch_dtype=dtype
        ).to(device)
        self.model.eval()
        self.device = device
        self.dtype = dtype

    def detect(self, image, prompts=None):
        """Detect elements using text-prompted detection.

        Args:
            image: PIL Image
            prompts: list of text phrases to detect

        Returns:
            list of {"label": str, "bbox": [x0,y0,x1,y1], "confidence": float}
        """
        if self.model is None:
            self.load()

        if prompts is None:
            prompts = list(DETECTION_PROMPTS.values())

        text_prompt = " . ".join(prompts) + " ."

        inputs = self.processor(
            images=image, text=text_prompt, return_tensors="pt"
        ).to(self.device)

        import torch
        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        # API changed between transformers versions:
        # 4.x uses box_threshold, 5.x uses threshold
        import inspect
        pp_sig = inspect.signature(
            self.processor.post_process_grounded_object_detection)
        pp_kwargs = {
            "target_sizes": target_sizes,
            "text_threshold": 0.2,
        }
        if "box_threshold" in pp_sig.parameters:
            pp_kwargs["box_threshold"] = 0.25
        else:
            pp_kwargs["threshold"] = 0.25
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            **pp_kwargs,
        )[0]

        detections = []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            detections.append({
                "label": label.strip().lower(),
                "bbox": [round(v.item(), 2) for v in box],
                "confidence": round(score.item(), 4),
            })

        return detections


# ---------------------------------------------------------------------------
# Model engine: Table Transformer
# ---------------------------------------------------------------------------

class TableTransformerEngine:
    """Microsoft Table Transformer for table structure recognition."""

    def __init__(self):
        self.model = None
        self.processor = None

    def load(self):
        """Load Table Transformer from HuggingFace transformers."""
        import torch
        from transformers import AutoModelForObjectDetection, AutoImageProcessor

        model_id = "microsoft/table-transformer-structure-recognition-v1.1-all"
        device = get_device()

        # The v1.1 config uses "longest_edge" format which older transformers
        # doesn't handle. Override with explicit size dict.
        self.processor = AutoImageProcessor.from_pretrained(
            model_id, size={"height": 1000, "width": 1000}
        )
        self.model = AutoModelForObjectDetection.from_pretrained(model_id).to(device)
        self.model.eval()
        self.device = device

    def detect(self, image, prompts=None):
        """Detect table structure elements."""
        if self.model is None:
            self.load()

        import torch

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, threshold=0.5, target_sizes=target_sizes
        )[0]

        detections = []
        for score, label_id, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            label_name = self.model.config.id2label[label_id.item()]
            detections.append({
                "label": label_name.strip().lower(),
                "bbox": [round(v.item(), 2) for v in box],
                "confidence": round(score.item(), 4),
            })

        return detections


# ---------------------------------------------------------------------------
# Model engine: YOLOv8 Form Field
# ---------------------------------------------------------------------------

class YOLOv8FormEngine:
    """YOLOv8 web form UI field detection engine."""

    def __init__(self):
        self.model = None

    def load(self):
        """Load YOLOv8 form field detection model."""
        from ultralytics import YOLO
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(
            repo_id="foduucom/web-form-ui-field-detection",
            filename="best.pt",
        )
        self.model = YOLO(model_path)
        print(f"  YOLOv8 form field classes: {self.model.names}", file=sys.stderr)

    def detect(self, image, prompts=None):
        """Detect form UI elements."""
        if self.model is None:
            self.load()

        device = get_device()
        device_str = str(device) if str(device) != "mps" else "cpu"

        results = self.model.predict(
            image,
            conf=0.2,
            device=device_str,
            verbose=False,
        )

        detections = []
        for r in results:
            boxes = r.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                cls_name = r.names.get(cls_id, f"class_{cls_id}")
                conf = float(boxes.conf[i])
                xyxy = boxes.xyxy[i].tolist()
                detections.append({
                    "label": cls_name.strip().lower(),
                    "raw_class": cls_name,
                    "bbox": [round(v, 2) for v in xyxy],
                    "confidence": round(conf, 4),
                })

        return detections


# ---------------------------------------------------------------------------
# Model engine: FFDNet (CommonForms)
# ---------------------------------------------------------------------------

class FFDNetEngine:
    """FFDNet form field detector (CommonForms YOLO11).

    Detects text input fields, choice buttons (checkboxes), and signatures.
    Based on the CommonForms paper (arXiv 2509.16506).
    """

    ID_TO_CLS = {0: "TextBox", 1: "ChoiceButton", 2: "Signature"}

    def __init__(self, variant="L"):
        self.variant = variant
        self.model = None

    def load(self):
        """Load FFDNet model weights via HuggingFace Hub."""
        from ultralytics import YOLO
        from huggingface_hub import hf_hub_download

        repo_id = f"jbarrow/FFDNet-{self.variant}"
        filename = f"FFDNet-{self.variant}.pt"
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        self.model = YOLO(model_path, task="detect")

    def detect(self, image, prompts=None):
        """Detect form fields."""
        if self.model is None:
            self.load()

        device = get_device()
        device_str = str(device) if str(device) != "mps" else "cpu"

        results = self.model.predict(
            image,
            imgsz=1600,
            conf=0.3,
            iou=0.1,
            augment=True,
            device=device_str,
            verbose=False,
        )

        detections = []
        for r in results:
            boxes = r.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                cls_name = self.ID_TO_CLS.get(cls_id, f"class_{cls_id}")
                conf = float(boxes.conf[i])
                xyxy = boxes.xyxy[i].tolist()
                detections.append({
                    "label": cls_name.strip().lower(),
                    "raw_class": cls_name,
                    "bbox": [round(v, 2) for v in xyxy],
                    "confidence": round(conf, 4),
                })

        return detections


# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------

def get_engine(model_name):
    """Create an ML engine for the given model name."""
    if model_name == "doclayout-yolo":
        return DocLayoutYOLOEngine()
    elif model_name == "florence2-base":
        return Florence2Engine("base")
    elif model_name == "florence2-large":
        return Florence2Engine("large")
    elif model_name == "grounding-dino":
        return GroundingDINOEngine()
    elif model_name == "table-transformer":
        return TableTransformerEngine()
    elif model_name == "yolov8-form":
        return YOLOv8FormEngine()
    elif model_name == "ffdnet-l":
        return FFDNetEngine("L")
    else:
        raise ValueError(f"Unknown model: {model_name}. "
                         f"Available: {', '.join(MODEL_REGISTRY.keys())}")


def pick_best_model(hw_info):
    """Pick the best model for the current hardware.

    Based on benchmark results (photo box detection — our primary ML use case):
      1. florence2-large  — 5/7 found, F1=0.59, mIoU=0.89  (needs ≥8 GB RAM, GPU/MPS)
      2. florence2-base   — 3/7 found, F1=0.35, mIoU=0.88  (needs ≥4 GB RAM)
      3. doclayout-yolo   — 2/7 found, F1=0.29, mIoU=0.96  (fast, lightweight)
      4. grounding-dino   — 2/7 found, F1=0.19, mIoU=0.92  (too many false positives)
    """
    recommended = hw_info.get("recommended_models", [])
    tier = hw_info.get("tier", "cpu")
    # florence2-large is the best performer but needs GPU/MPS + 8 GB RAM
    if tier in ("gpu_high", "mps") and "florence2-large" in recommended:
        return "florence2-large"
    # florence2-base is a solid second choice on weaker hardware
    if "florence2-base" in recommended:
        return "florence2-base"
    # doclayout-yolo is the fastest and lightest — good CPU fallback
    if "doclayout-yolo" in recommended:
        return "doclayout-yolo"
    return "doclayout-yolo"  # fallback


# ---------------------------------------------------------------------------
# Detection pipeline
# ---------------------------------------------------------------------------

def detect_pdf(pdf_path, model_name="auto", page_num=0):
    """Run ML detection on a PDF page.

    Returns dict with detections in PDF coordinates.
    """
    if model_name == "auto":
        hw = assess_hardware()
        model_name = pick_best_model(hw)
        print(f"Auto-selected model: {model_name} (tier: {hw['tier']})", file=sys.stderr)

    engine = get_engine(model_name)

    # Check model size and warn if large
    info = MODEL_REGISTRY.get(model_name, {})
    size_mb = info.get("size_mb", 0)
    if size_mb > 1000:
        print(f"Note: {model_name} model is ~{size_mb}MB. "
              f"It will be downloaded on first use.", file=sys.stderr)

    # Render page
    img, page_w, page_h = pdf_page_to_image(pdf_path, page_num, dpi=200)

    # Detect
    t0 = time.time()
    detections_px = engine.detect(img)
    elapsed = time.time() - t0

    # Convert to PDF coordinates
    detections_pdf = []
    for d in detections_px:
        pdf_bbox = pixel_to_pdf_coords(
            d["bbox"], img.width, img.height, page_w, page_h
        )
        detections_pdf.append({
            "label": d["label"],
            "bbox": pdf_bbox,
            "confidence": d["confidence"],
            "raw_class": d.get("raw_class", d["label"]),
        })

    return {
        "model": model_name,
        "page": page_num,
        "page_size": {"width": round(page_w, 1), "height": round(page_h, 1)},
        "inference_time_ms": round(elapsed * 1000, 1),
        "detections": detections_pdf,
    }


# ---------------------------------------------------------------------------
# Ground truth extraction
# ---------------------------------------------------------------------------

def extract_ground_truth(pdf_path):
    """Extract ground truth field locations using the heuristic detector.

    Returns list of {"label": str, "type": str, "bbox": [x0,y0,x1,y1]}
    where bbox is in PDF coordinates.
    """
    # Import the heuristic detector
    scripts_dir = Path(__file__).parent
    sys.path.insert(0, str(scripts_dir))
    from detect_fields import detect_all_pages

    result = detect_all_pages(str(pdf_path), detect_photos=True)

    ground_truth = []
    for page_data in result["pages"]:
        page_num = page_data["page"]
        for field in page_data["fields"]:
            cr = field["cell_rect"]
            ground_truth.append({
                "label": field.get("label", ""),
                "type": field["field_type"],
                "page": page_num,
                "bbox": [cr["x0"], cr["y0"], cr["x1"], cr["y1"]],
            })
        for pb in page_data.get("photo_boxes", []):
            r = pb["rect"]
            ground_truth.append({
                "label": "photo_box",
                "type": "photo_box",
                "page": page_num,
                "bbox": [r["x0"], r["y0"], r["x1"], r["y1"]],
            })

    return ground_truth


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark(model_names=None, pdf_dir=None, verbose=False):
    """Benchmark ML models against heuristic ground truth.

    Runs each model on all graphical test PDFs and computes
    precision, recall, F1, mean IoU, and inference time.
    """
    if pdf_dir is None:
        pdf_dir = Path(__file__).parent.parent / "tests" / "fixtures"
    else:
        pdf_dir = Path(pdf_dir)

    if model_names is None:
        hw = assess_hardware()
        model_names = hw["recommended_models"]
        if not model_names:
            model_names = ["doclayout-yolo"]
        print(f"Hardware tier: {hw['tier']}", file=sys.stderr)
        print(f"Models to benchmark: {', '.join(model_names)}", file=sys.stderr)

    # Find graphical PDFs (the ones ML is most useful for)
    graphical_pdfs = sorted(pdf_dir.glob("*_graphical.pdf"))
    if not graphical_pdfs:
        print(json.dumps({"error": f"No graphical PDFs found in {pdf_dir}"}))
        sys.exit(1)

    print(f"\nBenchmarking {len(model_names)} model(s) on "
          f"{len(graphical_pdfs)} graphical PDFs...\n", file=sys.stderr)

    all_results = {}

    for model_name in model_names:
        print(f"--- {model_name} ---", file=sys.stderr)
        info = MODEL_REGISTRY.get(model_name, {})
        size_mb = info.get("size_mb", 0)
        if size_mb > 1000:
            print(f"  Downloading ~{size_mb}MB model (first time only)...",
                  file=sys.stderr)

        try:
            engine = get_engine(model_name)
        except Exception as e:
            print(f"  Failed to create engine: {e}", file=sys.stderr)
            all_results[model_name] = {"error": str(e)}
            continue

        model_metrics = {
            "total_gt": 0,
            "total_detections": 0,
            "true_positives": 0,
            "ious": [],
            "offsets_pt": [],  # center-point positioning error in PDF points
            "inference_times_ms": [],
            "per_pdf": {},
            "per_type": {},
        }

        for pdf_path in graphical_pdfs:
            pdf_name = pdf_path.stem
            print(f"  {pdf_name}...", file=sys.stderr, end=" ")

            try:
                gt = extract_ground_truth(pdf_path)
            except Exception as e:
                print(f"GT error: {e}", file=sys.stderr)
                continue

            # Run detection on page 0 (primary page)
            try:
                img, page_w, page_h = pdf_page_to_image(str(pdf_path), 0, dpi=200)
                t0 = time.time()
                detections_px = engine.detect(img)
                elapsed_ms = (time.time() - t0) * 1000
            except Exception as e:
                print(f"Detection error: {e}", file=sys.stderr)
                model_metrics["per_pdf"][pdf_name] = {"error": str(e)}
                continue

            # Convert detections to PDF coords, normalise labels
            detections = []
            for d in detections_px:
                pdf_bbox = pixel_to_pdf_coords(
                    d["bbox"], img.width, img.height, page_w, page_h
                )
                detections.append({
                    "label": normalise_label(d["label"]),
                    "raw_label": d["label"],
                    "bbox": pdf_bbox,
                    "confidence": d["confidence"],
                })

            # Match detections against ground truth (page 0 only)
            gt_page0 = [g for g in gt if g["page"] == 0]
            matched_gt = set()
            matched_det = set()
            pair_ious = []

            for di, det in enumerate(detections):
                best_iou = 0
                best_gi = -1
                for gi, g in enumerate(gt_page0):
                    if gi in matched_gt:
                        continue
                    iou = compute_iou(det["bbox"], g["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gi = gi
                if best_iou >= 0.3 and best_gi >= 0:
                    matched_gt.add(best_gi)
                    matched_det.add(di)
                    pair_ious.append(best_iou)

                    # Compute center-point positioning error in PDF points
                    db = det["bbox"]
                    gb = gt_page0[best_gi]["bbox"]
                    det_cx = (db[0] + db[2]) / 2
                    det_cy = (db[1] + db[3]) / 2
                    gt_cx = (gb[0] + gb[2]) / 2
                    gt_cy = (gb[1] + gb[3]) / 2
                    offset_pt = ((det_cx - gt_cx)**2 + (det_cy - gt_cy)**2) ** 0.5

                    # Track per-type using normalised label
                    gt_type = gt_page0[best_gi]["type"]
                    _init = {"gt": 0, "tp": 0, "det": 0, "ious": [], "offsets": []}
                    if gt_type not in model_metrics["per_type"]:
                        model_metrics["per_type"][gt_type] = dict(_init)
                    model_metrics["per_type"][gt_type]["tp"] += 1
                    model_metrics["per_type"][gt_type]["ious"].append(best_iou)
                    model_metrics["per_type"][gt_type]["offsets"].append(offset_pt)
                    model_metrics["offsets_pt"].append(offset_pt)

            # Count GT per type
            _init = {"gt": 0, "tp": 0, "det": 0, "ious": [], "offsets": []}
            for g in gt_page0:
                gt_type = g["type"]
                if gt_type not in model_metrics["per_type"]:
                    model_metrics["per_type"][gt_type] = dict(_init)
                model_metrics["per_type"][gt_type]["gt"] += 1

            # Count detections per normalised type
            for d in detections:
                label = d["label"]
                if label not in model_metrics["per_type"]:
                    model_metrics["per_type"][label] = dict(_init)
                model_metrics["per_type"][label]["det"] += 1

            tp = len(matched_gt)
            model_metrics["total_gt"] += len(gt_page0)
            model_metrics["total_detections"] += len(detections)
            model_metrics["true_positives"] += tp
            model_metrics["ious"].extend(pair_ious)
            model_metrics["inference_times_ms"].append(elapsed_ms)

            pdf_precision = tp / len(detections) if detections else 0
            pdf_recall = tp / len(gt_page0) if gt_page0 else 0

            model_metrics["per_pdf"][pdf_name] = {
                "gt_count": len(gt_page0),
                "detection_count": len(detections),
                "true_positives": tp,
                "precision": round(pdf_precision, 3),
                "recall": round(pdf_recall, 3),
                "mean_iou": round(sum(pair_ious) / len(pair_ious), 3) if pair_ious else 0,
                "inference_ms": round(elapsed_ms, 1),
            }

            print(f"GT={len(gt_page0)} Det={len(detections)} "
                  f"TP={tp} IoU={model_metrics['per_pdf'][pdf_name]['mean_iou']:.3f} "
                  f"({elapsed_ms:.0f}ms)", file=sys.stderr)

            if verbose:
                print(f"    Detections: {json.dumps(detections[:5], indent=2)}", file=sys.stderr)

        # Compute aggregate metrics
        total_tp = model_metrics["true_positives"]
        total_det = model_metrics["total_detections"]
        total_gt = model_metrics["total_gt"]
        times = model_metrics["inference_times_ms"]

        precision = total_tp / total_det if total_det > 0 else 0
        recall = total_tp / total_gt if total_gt > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        mean_iou = sum(model_metrics["ious"]) / len(model_metrics["ious"]) if model_metrics["ious"] else 0
        avg_time = sum(times) / len(times) if times else 0

        offsets = model_metrics["offsets_pt"]
        mean_offset = sum(offsets) / len(offsets) if offsets else 0
        max_offset = max(offsets) if offsets else 0

        all_results[model_name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "mean_iou": round(mean_iou, 4),
            "mean_offset_pt": round(mean_offset, 2),
            "max_offset_pt": round(max_offset, 2),
            "avg_inference_ms": round(avg_time, 1),
            "total_gt": total_gt,
            "total_detections": total_det,
            "true_positives": total_tp,
            "model_size_mb": info.get("size_mb", 0),
            "per_pdf": model_metrics["per_pdf"],
            "per_type": {
                k: {
                    "gt": v["gt"],
                    "detections": v["det"],
                    "true_positives": v["tp"],
                    "mean_iou": round(sum(v["ious"]) / len(v["ious"]), 4) if v["ious"] else 0,
                    "mean_offset_pt": round(sum(v["offsets"]) / len(v["offsets"]), 2) if v["offsets"] else 0,
                }
                for k, v in model_metrics["per_type"].items()
            },
        }

    # Rank models by F1 score
    ranked = sorted(
        [(name, data) for name, data in all_results.items() if "error" not in data],
        key=lambda x: x[1]["f1"],
        reverse=True,
    )

    # Print summary table
    print("\n" + "=" * 95, file=sys.stderr)
    print("BENCHMARK RESULTS", file=sys.stderr)
    print("=" * 95, file=sys.stderr)
    print(f"{'Rank':<5} {'Model':<20} {'F1':<8} {'Prec':<8} {'Recall':<8} "
          f"{'mIoU':<8} {'Err(pt)':<9} {'Time(ms)':<10} {'Size(MB)':<10}", file=sys.stderr)
    print("-" * 95, file=sys.stderr)
    for rank, (name, data) in enumerate(ranked, 1):
        print(f"{rank:<5} {name:<20} {data['f1']:<8.4f} {data['precision']:<8.4f} "
              f"{data['recall']:<8.4f} {data['mean_iou']:<8.4f} "
              f"{data['mean_offset_pt']:<9.2f} "
              f"{data['avg_inference_ms']:<10.1f} {data['model_size_mb']:<10}",
              file=sys.stderr)
    print("=" * 95, file=sys.stderr)

    # Per-field-type breakdown across all models
    print("\n" + "=" * 95, file=sys.stderr)
    print("PER-FIELD-TYPE BREAKDOWN", file=sys.stderr)
    print("=" * 95, file=sys.stderr)
    print(f"{'Model':<20} {'Type':<14} {'GT':<6} {'Det':<6} {'TP':<6} "
          f"{'Recall':<8} {'mIoU':<8} {'Err(pt)':<9}", file=sys.stderr)
    print("-" * 95, file=sys.stderr)
    for name, data in ranked:
        for typ in ["text", "checkbox", "photo_box"]:
            metrics = data["per_type"].get(typ, {})
            gt = metrics.get("gt", 0)
            if gt == 0:
                continue
            det = metrics.get("detections", 0)
            tp = metrics.get("true_positives", 0)
            miou = metrics.get("mean_iou", 0)
            err = metrics.get("mean_offset_pt", 0)
            recall = tp / gt if gt > 0 else 0
            print(f"{name:<20} {typ:<14} {gt:<6} {det:<6} {tp:<6} "
                  f"{recall:<8.4f} {miou:<8.4f} {err:<9.2f}", file=sys.stderr)
    print("=" * 95, file=sys.stderr)

    # Photo-box focused ranking (the primary ML use case)
    print("\n" + "=" * 80, file=sys.stderr)
    print("PHOTO BOX DETECTION (primary ML use case)", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    photo_ranking = []
    for name, data in all_results.items():
        if "error" in data:
            continue
        pt = data["per_type"].get("photo_box", {})
        gt_count = pt.get("gt", 0)
        tp_count = pt.get("true_positives", 0)
        det_count = pt.get("detections", 0)
        miou = pt.get("mean_iou", 0)
        recall = tp_count / gt_count if gt_count > 0 else 0
        prec = tp_count / det_count if det_count > 0 else 0
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0
        photo_ranking.append((name, {
            "gt": gt_count, "detected": det_count, "tp": tp_count,
            "precision": round(prec, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "mean_iou": round(miou, 4),
        }))
    photo_ranking.sort(key=lambda x: (x[1]["f1"], x[1]["mean_iou"]), reverse=True)
    print(f"{'Rank':<5} {'Model':<20} {'Found':<10} {'Prec':<8} {'Recall':<8} "
          f"{'F1':<8} {'mIoU':<8}", file=sys.stderr)
    print("-" * 70, file=sys.stderr)
    for rank, (name, pm) in enumerate(photo_ranking, 1):
        print(f"{rank:<5} {name:<20} {pm['tp']}/{pm['gt']:<8} {pm['precision']:<8.4f} "
              f"{pm['recall']:<8.4f} {pm['f1']:<8.4f} {pm['mean_iou']:<8.4f}",
              file=sys.stderr)
    print("=" * 80, file=sys.stderr)

    # Store photo ranking in output
    output_photo = {name: data for name, data in photo_ranking}

    # Check for errors
    for name, data in all_results.items():
        if "error" in data:
            print(f"\n{name}: FAILED — {data['error']}", file=sys.stderr)

    # Save detailed results
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pdfs_tested": len(graphical_pdfs),
        "models": all_results,
        "ranking": [name for name, _ in ranked],
        "photo_box_ranking": [name for name, _ in photo_ranking],
        "photo_box_metrics": output_photo,
    }

    results_path = Path("/tmp/plume_ml_benchmark.json")
    results_path.write_text(json.dumps(output, indent=2))
    print(f"\nDetailed results saved to {results_path}", file=sys.stderr)

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ML-based document layout detection for PDF forms"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # assess
    assess_parser = subparsers.add_parser("assess", help="Show hardware capabilities")
    assess_parser.add_argument("--pretty", action="store_true")

    # detect
    detect_parser = subparsers.add_parser("detect", help="Detect fields in a PDF")
    detect_parser.add_argument("input_pdf", help="Path to input PDF")
    detect_parser.add_argument("--model", default="auto",
                               help="Model name or 'auto' (default: auto)")
    detect_parser.add_argument("--page", type=int, default=0)
    detect_parser.add_argument("--pretty", action="store_true")

    # benchmark
    bench_parser = subparsers.add_parser("benchmark",
                                         help="Benchmark models on test dataset")
    bench_parser.add_argument("--model", action="append", dest="models",
                              help="Model(s) to benchmark (repeat for multiple). "
                                   "Default: all recommended for hardware.")
    bench_parser.add_argument("--pdf-dir", help="Directory with test PDFs")
    bench_parser.add_argument("--verbose", action="store_true")
    bench_parser.add_argument("--pretty", action="store_true")

    args = parser.parse_args()

    if args.command == "assess":
        hw = assess_hardware()
        indent = 2 if args.pretty else None
        print(json.dumps(hw, indent=indent))

    elif args.command == "detect":
        if not Path(args.input_pdf).exists():
            print(json.dumps({"error": f"PDF not found: {args.input_pdf}"}),
                  file=sys.stderr)
            sys.exit(1)
        result = detect_pdf(args.input_pdf, model_name=args.model, page_num=args.page)
        indent = 2 if args.pretty else None
        print(json.dumps(result, indent=indent))

    elif args.command == "benchmark":
        result = benchmark(
            model_names=args.models,
            pdf_dir=args.pdf_dir,
            verbose=args.verbose,
        )
        indent = 2 if getattr(args, "pretty", False) else None
        print(json.dumps(result, indent=indent))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
