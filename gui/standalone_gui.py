"""
MRAF-Net Brain Tumor Segmentation - Standalone Research GUI
============================================================

Self-contained GUI application for brain tumor segmentation
No external dependencies on src module - all model code included

Author: Anne Nidhusha Nithiyalan (w1985740)
Supervisor: Ms. Mohanadas Jananie
Institution: University of Westminster / IIT

Usage:
    python standalone_gui.py

Then open: http://localhost:7860
"""

import os
import sys
import time
import html
import tempfile
import textwrap
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, List

import numpy as np
import gradio as gr
import nibabel as nib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

# Add parent directory to path to import src modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.mraf_net import MRAFNet, create_model
from src.utils.metrics import (
    compute_dice,
    compute_hausdorff95,
    compute_sensitivity,
    compute_specificity,
    compute_brats_regions,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "colors": {
        0: [0, 0, 0, 0],
        1: [0, 255, 0, 180],    # NCR/NET - Green
        2: [255, 255, 0, 180],  # Edema - Yellow
        4: [255, 0, 0, 180]     # ET - Red
    }
}

REGION_ORDER = [1, 2, 4]
REGION_DETAILS = {
    1: {"short": "NCR/NET", "meaning": "necrotic core", "color_name": "Green"},
    2: {"short": "ED", "meaning": "edema / swelling", "color_name": "Yellow"},
    4: {"short": "ET", "meaning": "enhancing tumor", "color_name": "Red"},
}
MAX_SLICE_INDEX = 100

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


# ============================================================================
# CHECKPOINT DISCOVERY
# ============================================================================

def discover_checkpoints() -> List[str]:
    """Scan experiments/ directory and return paths to all checkpoint files."""
    checkpoints = []
    if EXPERIMENTS_DIR.exists():
        for exp_dir in sorted(EXPERIMENTS_DIR.iterdir()):
            if exp_dir.is_dir():
                ckpt_dir = exp_dir / "checkpoints"
                if ckpt_dir.exists():
                    for ckpt_file in sorted(ckpt_dir.glob("*.pth")):
                        checkpoints.append(str(ckpt_file))
    if not checkpoints:
        # Fallback: scan top-level checkpoints/ folder
        fallback = PROJECT_ROOT / "checkpoints"
        if fallback.exists():
            for ckpt_file in sorted(fallback.glob("*.pth")):
                checkpoints.append(str(ckpt_file))
    return checkpoints


# ============================================================================
# MODEL HANDLER
# ============================================================================

class ModelHandler:
    """Handle model loading and inference."""

    def __init__(self):
        self.model = None
        self.device = CONFIG["device"]
        self.loaded = False
        self.param_count = {}

    def load(self, checkpoint_path: str) -> str:
        """Load model from checkpoint, auto-detecting architecture flags."""
        try:
            if not os.path.exists(checkpoint_path):
                return f"✗ File not found: {checkpoint_path}"

            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            saved_keys = set(checkpoint["model_state_dict"].keys())

            # Auto-detect architecture from saved weights
            has_swin = any("swin_bottleneck" in k for k in saved_keys)

            config = {
                'data': {'in_channels': 4, 'num_classes': 4},
                'model': {
                    'base_features': 32,
                    'deep_supervision': True,
                    'dropout': 0.0,
                    'use_swin_bottleneck': has_swin,
                }
            }

            self.model = create_model(config)
            self.param_count = self.model.get_parameter_count()

            result = self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            missing = [k for k in result.missing_keys if not k.endswith("num_batches_tracked")]
            unexpected = result.unexpected_keys

            self.model.to(self.device)
            self.model.eval()
            self.loaded = True

            metrics = checkpoint.get("metrics", {})
            dice = metrics.get("dice_mean", "N/A")
            dice_str = f"{dice:.4f}" if isinstance(dice, float) else str(dice)
            total_params = self.param_count["total"]

            status = (
                f"✓ Model loaded successfully\n"
                f"▸ Device: {self.device.upper()}\n"
                f"▸ Training Dice: {dice_str}\n"
                f"▸ Parameters: {total_params:,}"
            )
            if missing:
                status += f"\n⚠ Missing keys ({len(missing)}): {', '.join(missing[:3])}{'…' if len(missing)>3 else ''}"
            if unexpected:
                status += f"\n⚠ Unexpected keys ({len(unexpected)}): {', '.join(unexpected[:3])}{'…' if len(unexpected)>3 else ''}"
            return status

        except Exception as e:
            self.loaded = False
            return f"✗ Error: {str(e)}"

    def predict(self, images: np.ndarray, batch_size: int = 16) -> Tuple[np.ndarray, float, float]:
        """Run inference. Returns (segmentation, inference_time_sec, peak_gpu_mb)."""
        if not self.loaded:
            raise RuntimeError("Model not loaded!")

        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)

        t_start = time.perf_counter()

        with torch.no_grad():
            # (C, H, W, D) -> (C, D, H, W)
            images_t = np.transpose(images, (0, 3, 1, 2))
            C, D, H, W = images_t.shape
            pred_np = np.zeros((D, H, W), dtype=np.uint8)

            for start_idx in range(0, D, batch_size):
                end_idx = min(start_idx + batch_size, D)
                batch = images_t[:, start_idx:end_idx, :, :]
                tensor = torch.from_numpy(batch).float().unsqueeze(0).to(self.device)

                output, _ = self.model(tensor)
                if isinstance(output, tuple):
                    output = output[0]
                pred = torch.argmax(F.softmax(output, dim=1), dim=1)
                pred_np[start_idx:end_idx, :, :] = pred.squeeze(0).cpu().numpy()

                if self.device == "cuda":
                    torch.cuda.empty_cache()

            # (D, H, W) -> (H, W, D)
            pred_np = np.transpose(pred_np, (1, 2, 0))
            pred_np[pred_np == 3] = 4  # Restore BraTS label

        inference_time = time.perf_counter() - t_start

        peak_gpu_mb = 0.0
        if self.device == "cuda":
            peak_gpu_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)

        return pred_np, inference_time, peak_gpu_mb


# Global model instance
model = ModelHandler()
stored_data = None


# ============================================================================
# SYSTEM INFO
# ============================================================================

def get_system_info() -> str:
    """Return a formatted hardware/system report."""
    lines = ["## System Information\n"]

    # GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        lines.append(f"**GPU:** {gpu_name}")
        lines.append(f"**VRAM:** {total_vram:.1f} GB")
        lines.append(f"**CUDA Version:** {torch.version.cuda}")
    else:
        lines.append("**GPU:** Not available (running on CPU)")

    # CPU / Python / Torch
    import platform
    lines.append(f"**Platform:** {platform.system()} {platform.release()}")
    lines.append(f"**Python:** {platform.python_version()}")
    lines.append(f"**PyTorch:** {torch.__version__}")

    # Model params (if loaded)
    if model.loaded and model.param_count:
        total = model.param_count["total"]
        lines.append(f"\n**Model Parameters:** {total:,}")

    return "\n\n".join(lines)


# ============================================================================
# DATA PROCESSING
# ============================================================================

def normalize_intensity(images: np.ndarray) -> np.ndarray:
    """Z-score normalization."""
    normalized = np.zeros_like(images)
    for i in range(images.shape[0]):
        img = images[i]
        mask = img > 0
        if mask.sum() > 0:
            mean = img[mask].mean()
            std = img[mask].std() + 1e-8
            normalized[i] = np.where(mask, (img - mean) / std, 0)
    return normalized


def validate_modality_shapes(
    flair_data: np.ndarray,
    t1_data: np.ndarray,
    t1ce_data: np.ndarray,
    t2_data: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
) -> Optional[str]:
    """Validate that uploaded modalities share the same spatial shape."""
    modality_shapes = {
        "FLAIR": tuple(flair_data.shape),
        "T1": tuple(t1_data.shape),
        "T1ce": tuple(t1ce_data.shape),
        "T2": tuple(t2_data.shape),
    }

    unique_shapes = set(modality_shapes.values())
    if len(unique_shapes) != 1:
        lines = [
            "✗ Uploaded MRI modalities do not have the same shape.",
            "",
            "Detected shapes:",
        ]
        for name, shape in modality_shapes.items():
            lines.append(f"- {name}: {shape}")
        lines.extend([
            "",
            "Please upload the 4 modalities from the same case or resample them to a common space first.",
        ])
        return "\n".join(lines)

    if ground_truth is not None and tuple(ground_truth.shape) != next(iter(unique_shapes)):
        lines = [
            "✗ Ground truth shape does not match the uploaded MRI modalities.",
            "",
            f"- MRI modalities: {next(iter(unique_shapes))}",
            f"- Ground truth: {tuple(ground_truth.shape)}",
            "",
            "Please upload a ground-truth mask from the same case and space.",
        ]
        return "\n".join(lines)

    return None


def create_overlay(mri_slice: np.ndarray, seg_slice: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Create MRI with segmentation overlay."""
    mri_norm = (mri_slice - mri_slice.min()) / (mri_slice.max() - mri_slice.min() + 1e-8)
    mri_rgb = np.stack([mri_norm] * 3, axis=-1)

    overlay = np.zeros((*seg_slice.shape, 4), dtype=np.float32)
    for label, color in CONFIG["colors"].items():
        if label > 0:
            mask = seg_slice == label
            overlay[mask] = np.array(color) / 255.0

    for c in range(3):
        mask = overlay[:, :, 3] > 0
        mri_rgb[:, :, c][mask] = (
            mri_rgb[:, :, c][mask] * (1 - alpha * overlay[:, :, 3][mask]) +
            overlay[:, :, c][mask] * alpha * overlay[:, :, 3][mask]
        )

    return (mri_rgb * 255).astype(np.uint8)


def compute_region_percentages(segmentation: np.ndarray) -> Dict[int, float]:
    """Return each tumor region as a share of the predicted tumor volume."""
    unique, counts = np.unique(segmentation, return_counts=True)
    label_counts = dict(zip(unique.astype(int), counts.astype(np.int64)))
    tumor_total = sum(label_counts.get(label, 0) for label in REGION_ORDER)

    if tumor_total == 0:
        return {label: 0.0 for label in REGION_ORDER}

    return {
        label: (label_counts.get(label, 0) / tumor_total) * 100.0
        for label in REGION_ORDER
    }


def build_slice_slider_update(total_slices: int, preferred_slice: int):
    """Clamp the visible slice range for the UI slider."""
    slider_max = min(MAX_SLICE_INDEX, max(0, int(total_slices) - 1))
    slider_value = min(max(0, int(preferred_slice)), slider_max)
    return gr.update(value=slider_value, maximum=slider_max)


def _get_font(size: int):
    """Best-effort font loader with a safe fallback."""
    for font_name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_labeled_output(
    mri_slice: np.ndarray,
    seg_slice: np.ndarray,
    segmentation: np.ndarray,
    view: str,
    slice_idx: int,
    total_slices: int,
    show_overlay: bool,
    alpha: float,
) -> Image.Image:
    """Render the output slice with an embedded legend and region-share note."""
    if show_overlay:
        base_np = create_overlay(mri_slice, seg_slice, alpha)
        title = "Predicted segmentation overlay"
    else:
        mri_norm = (mri_slice - mri_slice.min()) / (mri_slice.max() - mri_slice.min() + 1e-8)
        base_np = (np.stack([mri_norm] * 3, axis=-1) * 255).astype(np.uint8)
        title = "MRI slice"

    region_percentages = compute_region_percentages(segmentation)
    tumor_detected = any(region_percentages[label] > 0 for label in REGION_ORDER)
    note = (
        "Percentages = share of the predicted tumor volume, not model confidence."
        if tumor_detected
        else "No tumor voxels were predicted, so every region share stays at 0.0%."
    )

    canvas_width = max(base_np.shape[1], 430)
    wrap_width = max(36, canvas_width // 8)
    wrapped_note = textwrap.wrap(note, width=wrap_width)
    footer_height = 88 + (len(wrapped_note) * 16)

    base = Image.fromarray(base_np).convert("RGB")
    image_x = (canvas_width - base.width) // 2
    canvas = Image.new("RGB", (canvas_width, base.height + footer_height), color=(10, 10, 10))
    canvas.paste(base, (image_x, 0))

    draw = ImageDraw.Draw(canvas)
    title_font = _get_font(18)
    body_font = _get_font(15)
    small_font = _get_font(12)

    left_badge_right = min(image_x + base.width - 12, image_x + 260)
    draw.rectangle((image_x + 12, 12, left_badge_right, 44), fill=(5, 5, 5), outline=(55, 55, 55))
    draw.text((image_x + 22, 20), title, fill=(245, 245, 245), font=body_font)

    slice_text = f"{view} slice {int(slice_idx) + 1}/{int(total_slices)}"
    try:
        text_width = draw.textbbox((0, 0), slice_text, font=body_font)[2]
    except AttributeError:
        text_width = len(slice_text) * 8
    badge_left = max(image_x + 12, image_x + base.width - text_width - 28)
    draw.rectangle((badge_left, 12, image_x + base.width - 12, 44), fill=(5, 5, 5), outline=(55, 55, 55))
    draw.text((badge_left + 10, 20), slice_text, fill=(245, 245, 245), font=body_font)

    footer_top = base.height
    draw.rectangle((0, footer_top, canvas.width, canvas.height), fill=(14, 14, 14))
    draw.line((0, footer_top, canvas.width, footer_top), fill=(45, 45, 45), width=2)
    draw.text((14, footer_top + 10), "Colors and predicted tumor share", fill=(255, 255, 255), font=title_font)

    column_width = (canvas.width - 28) // 3
    for idx, label in enumerate(REGION_ORDER):
        details = REGION_DETAILS[label]
        color = tuple(CONFIG["colors"][label][:3])
        x_left = 14 + (idx * column_width)
        y_top = footer_top + 42

        draw.rectangle((x_left, y_top + 2, x_left + 18, y_top + 20), fill=color, outline=(230, 230, 230))
        draw.text(
            (x_left + 28, y_top),
            f"{details['short']}: {region_percentages[label]:.1f}%",
            fill=(245, 245, 245),
            font=body_font,
        )
        draw.text(
            (x_left + 28, y_top + 20),
            f"{details['color_name']} = {details['meaning']}",
            fill=(185, 185, 185),
            font=small_font,
        )

    note_y = footer_top + 70
    for line in wrapped_note:
        draw.text((14, note_y), line, fill=(175, 175, 175), font=small_font)
        note_y += 14

    return canvas


def compute_advanced_metrics(
    seg: np.ndarray,
    gt: Optional[np.ndarray],
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    inference_time: float = 0.0,
    peak_gpu_mb: float = 0.0,
) -> Dict:
    """Compute volumes, Dice, HD95, Sensitivity, Specificity."""
    unique, counts = np.unique(seg, return_counts=True)
    vol_dict = dict(zip(unique.astype(int), counts))

    vol_ncr = vol_dict.get(1, 0) / 1000
    vol_ed = vol_dict.get(2, 0) / 1000
    vol_et = vol_dict.get(4, 0) / 1000

    metrics = {
        "volume": {
            "wt": round(vol_ncr + vol_ed + vol_et, 2),
            "tc": round(vol_ncr + vol_et, 2),
            "et": round(vol_et, 2),
        },
        "composition_pct": {
            "ncr_net": round((vol_ncr / (vol_ncr + vol_ed + vol_et)) * 100, 2) if (vol_ncr + vol_ed + vol_et) else 0.0,
            "edema": round((vol_ed / (vol_ncr + vol_ed + vol_et)) * 100, 2) if (vol_ncr + vol_ed + vol_et) else 0.0,
            "enhancing": round((vol_et / (vol_ncr + vol_ed + vol_et)) * 100, 2) if (vol_ncr + vol_ed + vol_et) else 0.0,
        },
        "inference_time": inference_time,
        "peak_gpu_mb": peak_gpu_mb,
    }

    if gt is not None:
        # Convert BraTS label 4 -> 3 for metrics module (which uses 0-3 internally)
        seg_int = seg.copy().astype(np.int64)
        seg_int[seg_int == 4] = 3
        gt_int = gt.copy().astype(np.int64)
        gt_int[gt_int == 4] = 3

        pred_wt, pred_tc, pred_et, tgt_wt, tgt_tc, tgt_et = compute_brats_regions(seg_int, gt_int)

        metrics["dice"] = {
            "wt": round(compute_dice(pred_wt, tgt_wt), 4),
            "tc": round(compute_dice(pred_tc, tgt_tc), 4),
            "et": round(compute_dice(pred_et, tgt_et), 4),
        }
        metrics["dice"]["mean"] = round(
            (metrics["dice"]["wt"] + metrics["dice"]["tc"] + metrics["dice"]["et"]) / 3, 4
        )

        metrics["sensitivity"] = {
            "wt": round(compute_sensitivity(pred_wt, tgt_wt), 4),
            "tc": round(compute_sensitivity(pred_tc, tgt_tc), 4),
            "et": round(compute_sensitivity(pred_et, tgt_et), 4),
        }

        metrics["specificity"] = {
            "wt": round(compute_specificity(pred_wt, tgt_wt), 4),
            "tc": round(compute_specificity(pred_tc, tgt_tc), 4),
            "et": round(compute_specificity(pred_et, tgt_et), 4),
        }

        hd95_wt = compute_hausdorff95(pred_wt, tgt_wt, spacing)
        hd95_tc = compute_hausdorff95(pred_tc, tgt_tc, spacing)
        hd95_et = compute_hausdorff95(pred_et, tgt_et, spacing)
        metrics["hd95"] = {
            "wt": round(hd95_wt, 2) if not np.isinf(hd95_wt) else "N/A",
            "tc": round(hd95_tc, 2) if not np.isinf(hd95_tc) else "N/A",
            "et": round(hd95_et, 2) if not np.isinf(hd95_et) else "N/A",
        }

    return metrics


def render_results_message(title: str, message: str, tone: str = "neutral") -> str:
    """Render a styled empty, warning, or error state for the results panel."""
    safe_title = html.escape(title)
    safe_message = html.escape(message).replace("\n", "<br>")
    return f"""
    <div class="results-shell">
        <div class="results-status results-status--{tone}">
            <div class="results-status__eyebrow">{safe_title}</div>
            <div class="results-status__body">{safe_message}</div>
        </div>
    </div>
    """


def _format_metric_value(value, unit: str = "", decimals: int = 2) -> str:
    """Format numeric values safely for the HTML dashboard."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):.{decimals}f}{unit}"
    return html.escape(str(value))


def _format_metric_pct(value) -> str:
    """Format a 0-1 metric as a human-readable percentage."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value) * 100:.2f}%"
    return html.escape(str(value))


def _render_kpi_card(label: str, code: str, value: str, accent: str, meta: str) -> str:
    """Render a compact KPI card."""
    return f"""
    <div class="results-card results-card--kpi" style="--accent:{accent};">
        <div class="results-card__code">{html.escape(code)}</div>
        <div class="results-card__value">{html.escape(value)}</div>
        <div class="results-card__label">{html.escape(label)}</div>
        <div class="results-card__meta">{html.escape(meta)}</div>
    </div>
    """


def _render_composition_row(label: str, description: str, percentage: float, accent: str) -> str:
    """Render a composition bar row."""
    width = max(0.0, min(100.0, float(percentage)))
    return f"""
    <div class="composition-row">
        <div class="composition-row__meta">
            <span class="composition-row__dot" style="--accent:{accent};"></span>
            <div>
                <div class="composition-row__label">{html.escape(label)}</div>
                <div class="composition-row__description">{html.escape(description)}</div>
            </div>
        </div>
        <div class="composition-row__track">
            <div class="composition-row__fill" style="--accent:{accent}; width:{width:.2f}%;"></div>
        </div>
        <div class="composition-row__value">{width:.2f}%</div>
    </div>
    """


def _render_eval_card(label: str, dice, sensitivity, specificity, hd95, accent: str) -> str:
    """Render evaluation metrics for one tumor region."""
    return f"""
    <div class="results-card results-card--eval" style="--accent:{accent};">
        <div class="results-card__header">
            <div class="results-card__label">{html.escape(label)}</div>
            <div class="results-pill">{_format_metric_pct(dice)}</div>
        </div>
        <div class="results-stat-grid">
            <div class="results-stat">
                <span class="results-stat__label">Sensitivity</span>
                <span class="results-stat__value">{_format_metric_pct(sensitivity)}</span>
            </div>
            <div class="results-stat">
                <span class="results-stat__label">Specificity</span>
                <span class="results-stat__value">{_format_metric_pct(specificity)}</span>
            </div>
            <div class="results-stat">
                <span class="results-stat__label">HD95</span>
                <span class="results-stat__value">{_format_metric_value(hd95, ' mm')}</span>
            </div>
        </div>
    </div>
    """


def format_metrics_text(metrics: Dict) -> str:
    """Format metrics into a card-based HTML dashboard."""
    volume_cards = "".join([
        _render_kpi_card("Whole Tumor", "WT", _format_metric_value(metrics["volume"]["wt"], " ml"), "#f2f2f2", "Combined tumor burden"),
        _render_kpi_card("Tumor Core", "TC", _format_metric_value(metrics["volume"]["tc"], " ml"), "#7cc6ff", "Core sub-region"),
        _render_kpi_card("Enhancing", "ET", _format_metric_value(metrics["volume"]["et"], " ml"), "#ff5a5a", "Active enhancing tissue"),
    ])

    composition_rows = "".join([
        _render_composition_row("NCR/NET", "Green overlay • necrotic core / non-enhancing tumor", metrics["composition_pct"]["ncr_net"], "#33d17a"),
        _render_composition_row("Edema", "Yellow overlay • swelling around tumor", metrics["composition_pct"]["edema"], "#ffd54a"),
        _render_composition_row("Enhancing", "Red overlay • actively enhancing tumor", metrics["composition_pct"]["enhancing"], "#ff5a5a"),
    ])

    performance_cards = [
        _render_kpi_card("Inference Time", "SPEED", _format_metric_value(metrics["inference_time"], " s", decimals=3), "#a8c5ff", "End-to-end model runtime"),
    ]
    if metrics["peak_gpu_mb"] > 0:
        performance_cards.append(
            _render_kpi_card("Peak GPU Memory", "CUDA", _format_metric_value(metrics["peak_gpu_mb"], " MB", decimals=1), "#ffb15a", "Maximum allocated GPU memory")
        )

    hero_pills = [
        f"<div class=\"results-pill\">Inference {_format_metric_value(metrics['inference_time'], ' s', decimals=3)}</div>",
        f"<div class=\"results-pill\">Tumor {_format_metric_value(metrics['volume']['wt'], ' ml')}</div>",
    ]
    evaluation_html = """
    <div class="results-empty-card">
        Upload a ground-truth mask to unlock Dice, Sensitivity, Specificity, and HD95 evaluation cards.
    </div>
    """

    if "dice" in metrics:
        hero_pills.append(f"<div class=\"results-pill results-pill--accent\">Mean Dice {_format_metric_pct(metrics['dice']['mean'])}</div>")
        evaluation_html = f"""
        <div class="results-grid results-grid--eval">
            {_render_eval_card('Whole Tumor', metrics['dice']['wt'], metrics['sensitivity']['wt'], metrics['specificity']['wt'], metrics['hd95']['wt'], '#f2f2f2')}
            {_render_eval_card('Tumor Core', metrics['dice']['tc'], metrics['sensitivity']['tc'], metrics['specificity']['tc'], metrics['hd95']['tc'], '#7cc6ff')}
            {_render_eval_card('Enhancing Tumor', metrics['dice']['et'], metrics['sensitivity']['et'], metrics['specificity']['et'], metrics['hd95']['et'], '#ff5a5a')}
        </div>
        <div class="results-note">
            These percentages compare the prediction against the uploaded ground truth. They are evaluation metrics, not confidence scores.
        </div>
        """

    return f"""
    <div class="results-shell">
        <div class="results-hero">
            <div>
                <div class="results-status__eyebrow">Segmentation Results</div>
                <h2 class="results-hero__title">Card-based tumor summary</h2>
                <p class="results-hero__subtitle">
                    Predicted tumor burden, regional composition, and runtime are grouped into scan-friendly cards for faster review.
                </p>
            </div>
            <div class="results-pill-row">
                {''.join(hero_pills)}
            </div>
        </div>

        <div class="results-section">
            <div class="results-section__head">
                <h3>Tumor Volumes</h3>
                <p>Predicted volume for each clinically relevant region.</p>
            </div>
            <div class="results-grid results-grid--volumes">
                {volume_cards}
            </div>
        </div>

        <div class="results-section">
            <div class="results-section__head">
                <h3>Predicted Tumor Composition</h3>
                <p>How the predicted tumor is split across the overlay colors.</p>
            </div>
            <div class="results-card results-card--composition">
                {composition_rows}
            </div>
            <div class="results-note">
                Percentages above describe region share within the predicted tumor. They are not model confidence scores.
            </div>
        </div>

        <div class="results-section">
            <div class="results-section__head">
                <h3>Performance</h3>
                <p>Runtime and hardware footprint for this prediction.</p>
            </div>
            <div class="results-grid results-grid--performance">
                {''.join(performance_cards)}
            </div>
        </div>

        <div class="results-section">
            <div class="results-section__head">
                <h3>Evaluation</h3>
                <p>Ground-truth comparison appears here when a mask is uploaded.</p>
            </div>
            {evaluation_html}
        </div>
    </div>
    """


# ============================================================================
# GRADIO FUNCTIONS
# ============================================================================

def refresh_checkpoints():
    """Return updated Dropdown choices."""
    choices = discover_checkpoints()
    value = choices[0] if choices else ""
    return gr.update(choices=choices, value=value)


def load_model_fn(path):
    return model.load(path)


def run_segmentation(checkpoint_path, flair, t1, t1ce, t2, gt=None):
    global stored_data

    if not all([flair, t1, t1ce, t2]):
        return None, build_slice_slider_update(MAX_SLICE_INDEX + 1, 50), render_results_message(
            "Missing MRI Inputs",
            "Please upload all 4 MRI modalities before running segmentation.",
            tone="warning",
        )

    if not model.loaded:
        return None, build_slice_slider_update(MAX_SLICE_INDEX + 1, 50), render_results_message(
            "Model Not Loaded",
            "Load a checkpoint first so the app can run segmentation.",
            tone="danger",
        )

    try:
        flair_data = nib.load(flair.name).get_fdata().astype(np.float32)
        affine = nib.load(flair.name).affine
        t1_data = nib.load(t1.name).get_fdata().astype(np.float32)
        t1ce_data = nib.load(t1ce.name).get_fdata().astype(np.float32)
        t2_data = nib.load(t2.name).get_fdata().astype(np.float32)

        ground_truth = None
        if gt:
            ground_truth = nib.load(gt.name).get_fdata().astype(np.int64)

        shape_error = validate_modality_shapes(flair_data, t1_data, t1ce_data, t2_data, ground_truth)
        if shape_error is not None:
            return None, build_slice_slider_update(MAX_SLICE_INDEX + 1, 50), render_results_message(
                "Shape Validation Failed",
                shape_error,
                tone="danger",
            )

        images = np.stack([flair_data, t1_data, t1ce_data, t2_data], axis=0)
        images_norm = normalize_intensity(images)

        segmentation, inference_time, peak_gpu_mb = model.predict(images_norm)

        voxel_vol = float(np.abs(np.linalg.det(affine[:3, :3])))
        spacing = (voxel_vol ** (1 / 3),) * 3

        metrics = compute_advanced_metrics(
            segmentation, ground_truth, spacing, inference_time, peak_gpu_mb
        )

        stored_data = {
            "flair": flair_data,
            "seg": segmentation,
            "gt": ground_truth,
            "affine": affine,
            "metrics": metrics,
        }

        mid = flair_data.shape[2] // 2
        viz = render_labeled_output(
            flair_data[:, :, mid],
            segmentation[:, :, mid],
            segmentation,
            "Axial",
            mid,
            flair_data.shape[2],
            True,
            0.5,
        )

        text = format_metrics_text(metrics)
        return viz, build_slice_slider_update(flair_data.shape[2], mid), text

    except Exception as e:
        return None, build_slice_slider_update(MAX_SLICE_INDEX + 1, 50), render_results_message(
            "Segmentation Failed",
            f"{str(e)}\n\nSee the application logs for the full traceback.",
            tone="danger",
        )


def update_view(slice_idx, view, overlay, alpha):
    global stored_data
    if stored_data is None:
        return None

    flair = stored_data["flair"]
    seg = stored_data["seg"]

    if view == "Axial":
        total_slices = flair.shape[2]
        idx = min(max(0, int(slice_idx)), total_slices - 1)
        mri_s, seg_s = flair[:, :, idx], seg[:, :, idx]
    elif view == "Coronal":
        total_slices = flair.shape[1]
        idx = min(max(0, int(slice_idx)), total_slices - 1)
        mri_s, seg_s = flair[:, idx, :], seg[:, idx, :]
    else:
        total_slices = flair.shape[0]
        idx = min(max(0, int(slice_idx)), total_slices - 1)
        mri_s, seg_s = flair[idx, :, :], seg[idx, :, :]

    return render_labeled_output(
        mri_s,
        seg_s,
        seg,
        view,
        idx,
        total_slices,
        overlay,
        alpha,
    )


def create_3d():
    global stored_data
    if stored_data is None:
        return None

    seg = stored_data["seg"][::3, ::3, ::3]

    fig = plt.figure(figsize=(10, 8), facecolor='#0a0a0a')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0a0a0a')

    for label, color, name in [(1, '#00ff00', 'NCR'), (2, '#ffff00', 'Edema'), (4, '#ff3030', 'ET')]:
        mask = seg == label
        if mask.any():
            coords = np.argwhere(mask)
            if len(coords) > 1500:
                coords = coords[np.random.choice(len(coords), 1500, replace=False)]
            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=color, alpha=0.4, s=1, label=name)

    ax.tick_params(colors='#666666')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333333', labelcolor='#cccccc')
    ax.set_title('3D Tumor Visualization', color='#ffffff')
    return fig


def export_seg():
    global stored_data
    if stored_data is None:
        return None, "⚠ No segmentation available"

    path = tempfile.mktemp(suffix='.nii.gz')
    nii = nib.Nifti1Image(stored_data["seg"].astype(np.int16), stored_data["affine"])
    nib.save(nii, path)
    return path, f"✓ Exported successfully: {path}"


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_app():
    """Create Gradio interface."""

    mri_css = """
    .gradio-container {
        max-width: 100% !important;
        width: 100% !important;
        margin: 0 auto !important;
        padding: 0 40px !important;
        background: #0a0a0a !important;
        color: #e0e0e0 !important;
        box-sizing: border-box !important;
    }
    body, .dark {
        background: #0a0a0a !important;
    }
    .mri-header {
        text-align: center;
        background: radial-gradient(ellipse at center, #1a1a1a 0%, #000000 70%);
        padding: 30px 20px;
        border-radius: 4px;
        color: #ffffff;
        margin-bottom: 20px;
        border: 1px solid #333333;
        box-shadow: 0 0 40px rgba(255,255,255,0.03);
    }
    .mri-header h1 { font-weight:300; letter-spacing:6px; text-transform:uppercase; color:#fff; }
    .mri-header h3 { font-weight:300; letter-spacing:2px; color:#aaa; }
    .mri-header p  { color:#888; font-size:0.9em; }
    .mri-legend {
        background: #111; padding:12px 16px; border-radius:4px;
        border: 1px solid #333; color:#ccc;
    }
    .gr-box, .gr-panel, .gr-form, .gr-input, .gr-padded {
        background: #111111 !important; border-color: #333 !important; color: #e0e0e0 !important;
    }
    .gr-button-primary {
        background: #fff !important; color: #000 !important; border: none !important; font-weight:600;
    }
    .gr-button-primary:hover { background: #ccc !important; }
    .gr-button-secondary {
        background: #1a1a1a !important; color: #e0e0e0 !important; border: 1px solid #444 !important;
    }
    label, .gr-check-radio, .label-wrap { color: #ccc !important; }
    input, textarea, select {
        background: #1a1a1a !important; color: #e0e0e0 !important; border-color: #333 !important;
    }
    .tabs .tab-nav button { color: #888 !important; background: transparent !important; }
    .tabs .tab-nav button.selected { color: #fff !important; border-bottom: 2px solid #fff !important; }
    .prose h1,.prose h2,.prose h3,.prose h4,.prose p,.prose li,.prose td,.prose th { color:#e0e0e0 !important; }
    .prose table, .prose th, .prose td { border-color: #333 !important; }
    .prose code { background:#1a1a1a !important; color:#ccc !important; }
    .prose pre  { background:#0d0d0d !important; }
    .results-shell {
        display: flex;
        flex-direction: column;
        gap: 18px;
        margin: 8px 0 4px;
    }
    .results-hero, .results-status {
        background: linear-gradient(145deg, rgba(26,26,26,0.98), rgba(10,10,10,0.98));
        border: 1px solid #2d2d2d;
        border-radius: 18px;
        padding: 20px 22px;
        box-shadow: 0 18px 40px rgba(0,0,0,0.28);
    }
    .results-hero {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        gap: 16px;
        align-items: flex-start;
    }
    .results-hero__title {
        margin: 6px 0 8px;
        font-size: 1.55rem;
        font-weight: 600;
        color: #ffffff;
    }
    .results-hero__subtitle {
        margin: 0;
        max-width: 720px;
        color: #9f9f9f;
        line-height: 1.6;
    }
    .results-status__eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.18em;
        font-size: 0.74rem;
        color: #8f8f8f;
    }
    .results-status__body {
        margin-top: 12px;
        color: #efefef;
        line-height: 1.7;
    }
    .results-status--warning {
        border-color: rgba(255, 213, 74, 0.35);
        box-shadow: 0 18px 40px rgba(255, 213, 74, 0.08);
    }
    .results-status--danger {
        border-color: rgba(255, 90, 90, 0.35);
        box-shadow: 0 18px 40px rgba(255, 90, 90, 0.08);
    }
    .results-pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        justify-content: flex-end;
    }
    .results-pill {
        display: inline-flex;
        align-items: center;
        min-height: 38px;
        padding: 0 14px;
        border-radius: 999px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        color: #f2f2f2;
        font-size: 0.92rem;
        font-weight: 600;
    }
    .results-pill--accent {
        background: rgba(124, 198, 255, 0.12);
        border-color: rgba(124, 198, 255, 0.28);
    }
    .results-section {
        display: flex;
        flex-direction: column;
        gap: 12px;
    }
    .results-section__head h3 {
        margin: 0;
        color: #ffffff;
        font-size: 1.08rem;
    }
    .results-section__head p {
        margin: 5px 0 0;
        color: #8d8d8d;
        line-height: 1.55;
    }
    .results-grid {
        display: grid;
        gap: 14px;
    }
    .results-grid--volumes {
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    }
    .results-grid--performance {
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    }
    .results-grid--eval {
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    }
    .results-card {
        background: linear-gradient(180deg, rgba(18,18,18,0.98), rgba(8,8,8,0.98));
        border: 1px solid #242424;
        border-radius: 18px;
        padding: 18px;
        position: relative;
        overflow: hidden;
    }
    .results-card::before {
        content: "";
        position: absolute;
        inset: 0 auto auto 0;
        width: 100%;
        height: 3px;
        background: var(--accent, #ffffff);
        opacity: 0.88;
    }
    .results-card__code {
        color: var(--accent, #ffffff);
        letter-spacing: 0.14em;
        text-transform: uppercase;
        font-size: 0.76rem;
        margin-bottom: 14px;
    }
    .results-card__value {
        color: #ffffff;
        font-size: 1.55rem;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .results-card__label {
        color: #d9d9d9;
        font-weight: 600;
    }
    .results-card__meta {
        color: #858585;
        font-size: 0.92rem;
        margin-top: 8px;
        line-height: 1.5;
    }
    .results-card__header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 14px;
    }
    .results-card--composition {
        padding: 20px;
    }
    .composition-row {
        display: grid;
        grid-template-columns: minmax(180px, 240px) 1fr auto;
        gap: 14px;
        align-items: center;
        padding: 12px 0;
    }
    .composition-row + .composition-row {
        border-top: 1px solid rgba(255,255,255,0.06);
    }
    .composition-row__meta {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .composition-row__dot {
        width: 12px;
        height: 12px;
        border-radius: 999px;
        background: var(--accent, #ffffff);
        box-shadow: 0 0 18px rgba(255,255,255,0.16);
        flex: 0 0 auto;
    }
    .composition-row__label {
        color: #f0f0f0;
        font-weight: 600;
    }
    .composition-row__description {
        color: #8c8c8c;
        font-size: 0.9rem;
        line-height: 1.45;
        margin-top: 2px;
    }
    .composition-row__track {
        height: 10px;
        border-radius: 999px;
        background: rgba(255,255,255,0.07);
        overflow: hidden;
    }
    .composition-row__fill {
        height: 100%;
        border-radius: inherit;
        background: var(--accent, #ffffff);
    }
    .composition-row__value {
        color: #ffffff;
        font-weight: 700;
        min-width: 62px;
        text-align: right;
    }
    .results-stat-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 10px;
    }
    .results-stat {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 14px;
        padding: 12px;
        display: flex;
        flex-direction: column;
        gap: 6px;
    }
    .results-stat__label {
        color: #8e8e8e;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .results-stat__value {
        color: #ffffff;
        font-size: 1.02rem;
        font-weight: 600;
    }
    .results-note, .results-empty-card {
        background: rgba(255,255,255,0.03);
        border: 1px dashed rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 14px 16px;
        color: #a4a4a4;
        line-height: 1.6;
    }
    @media (max-width: 900px) {
        .results-pill-row {
            justify-content: flex-start;
        }
        .composition-row {
            grid-template-columns: 1fr;
        }
        .composition-row__value {
            text-align: left;
        }
        .results-stat-grid {
            grid-template-columns: 1fr;
        }
    }
    footer { display: none !important; }
    """

    mri_theme = gr.themes.Base(
        primary_hue=gr.themes.Color(c50="#f5f5f5",c100="#e0e0e0",c200="#ccc",c300="#aaa",c400="#888",c500="#666",c600="#444",c700="#333",c800="#1a1a1a",c900="#0a0a0a",c950="#000"),
        secondary_hue=gr.themes.Color(c50="#f5f5f5",c100="#e0e0e0",c200="#ccc",c300="#aaa",c400="#888",c500="#666",c600="#444",c700="#333",c800="#1a1a1a",c900="#0a0a0a",c950="#000"),
        neutral_hue=gr.themes.Color(c50="#f5f5f5",c100="#e0e0e0",c200="#ccc",c300="#aaa",c400="#888",c500="#666",c600="#444",c700="#333",c800="#1a1a1a",c900="#0a0a0a",c950="#000"),
    )

    initial_checkpoints = discover_checkpoints()
    initial_value = initial_checkpoints[0] if initial_checkpoints else ""

    with gr.Blocks(title="MRAF-Net Brain Tumor Segmentation", css=mri_css, theme=mri_theme) as app:

        # Header
        gr.HTML("""
        <div class="mri-header">
            <h1 style="margin:0 0 8px 0;">⬡ MRAF-Net</h1>
            <h3 style="margin:4px 0;">Multi-Resolution Aligned and Robust Fusion Network</h3>
            <p style="margin:4px 0;">Brain Tumor Segmentation from Multi-Modal MRI</p>
            <hr style="border:none;border-top:1px solid #333;margin:12px auto;width:60%;">
            <p>Anne Nidhusha Nithiyalan &nbsp;|&nbsp; University of Westminster / IIT</p>
        </div>
        """)

        with gr.Row():
            # Left Panel
            with gr.Column(scale=1):
                gr.Markdown("### ⚙ Model Configuration")

                checkpoint_dropdown = gr.Dropdown(
                    label="Checkpoint",
                    choices=initial_checkpoints,
                    value=initial_value,
                    allow_custom_value=True,
                )
                with gr.Row():
                    refresh_btn = gr.Button("⟳ Refresh List", variant="secondary", size="sm")
                    load_btn = gr.Button("⟳ Load Model", variant="primary")

                status = gr.Textbox(label="Status", lines=4, interactive=False)

                gr.Markdown("### ⇪ Upload MRI Scans")
                flair = gr.File(label="FLAIR", file_types=[".nii", ".nii.gz"])
                t1 = gr.File(label="T1", file_types=[".nii", ".nii.gz"])
                t1ce = gr.File(label="T1ce", file_types=[".nii", ".nii.gz"])
                t2 = gr.File(label="T2", file_types=[".nii", ".nii.gz"])
                gt = gr.File(label="Ground Truth (optional)", file_types=[".nii", ".nii.gz"])

                run_btn = gr.Button("▶ Run Segmentation", variant="primary", size="lg")

            # Right Panel
            with gr.Column(scale=2):
                gr.Markdown("### ⊞ Visualization")

                with gr.Row():
                    view = gr.Radio(["Axial", "Coronal", "Sagittal"], value="Axial", label="View")
                    overlay = gr.Checkbox(value=True, label="Overlay")
                    alpha = gr.Slider(0, 1, 0.5, label="Opacity")

                slice_slider = gr.Slider(0, 100, 50, step=1, label="Slice")
                image = gr.Image(label="Result", type="pil", height=450)

                gr.HTML("""
                <div class="mri-legend" style="text-align:center;">
                    <b style="color:#fff;">Legend:</b>&nbsp;&nbsp;
                    <span style="color:#00ff00;border:1px solid #00ff00;padding:2px 8px;border-radius:2px;margin:0 4px;font-size:0.85em;">■ Green = NCR/NET</span>
                    <span style="color:#ffff00;border:1px solid #ffff00;padding:2px 8px;border-radius:2px;margin:0 4px;font-size:0.85em;">■ Yellow = Edema</span>
                    <span style="color:#ff3030;border:1px solid #ff3030;padding:2px 8px;border-radius:2px;margin:0 4px;font-size:0.85em;">■ Red = Enhancing</span>
                </div>
                """)

        # Metrics
        metrics_display = gr.HTML(
            render_results_message(
                "Results Panel",
                "Upload MRI scans and run segmentation to see a card-based summary here.",
            )
        )

        # Tabs
        with gr.Tabs():
            with gr.TabItem("◈ 3D View"):
                plot_btn = gr.Button("⧉ Generate 3D Plot")
                plot = gr.Plot()

            with gr.TabItem("⇓ Export"):
                export_btn = gr.Button("Export Segmentation")
                export_file = gr.File(label="Download")
                export_status = gr.Textbox(label="Status", interactive=False)

            with gr.TabItem("⊡ System Info"):
                sys_refresh_btn = gr.Button("⟳ Refresh System Info")
                sys_info = gr.Markdown(get_system_info())

            with gr.TabItem("ℹ About"):
                gr.Markdown("""
                ## MRAF-Net Architecture

                **Multi-Resolution Aligned and Robust Fusion Network** for brain tumor segmentation.

                ### Key Components:
                - **Modality-Specific Encoders**: Process FLAIR, T1, T1ce, T2 independently
                - **Cross-Modality Fusion**: Integrate complementary information
                - **Swin Transformer Bottleneck**: Global context with window-based self-attention
                - **Attention Gates**: Enhance relevant features
                - **Deep Supervision**: Multi-scale loss computation

                ### BraTS Labels:
                | Label | Region | Description |
                |-------|--------|-------------|
                | 0 | Background | Healthy tissue |
                | 1 | NCR/NET | Necrotic Tumor Core |
                | 2 | ED | Peritumoral Edema |
                | 4 | ET | Enhancing Tumor |

                ### Metrics Reported:
                - **Region-share percentages on the image**: How much of the predicted tumor belongs to each color-coded region
                - **Dice Score**: Overlap similarity with ground truth (higher = better)
                - **HD95**: 95th percentile Hausdorff Distance in mm (lower = better)
                - **Sensitivity**: True positive rate
                - **Specificity**: True negative rate
                - **Inference Time**: Seconds per scan
                - **Peak GPU Memory**: MB allocated during inference

                ---
                **Author**: Anne Nidhusha Nithiyalan (w1985740)
                **Supervisor**: Ms. Mohanadas Jananie
                **Institution**: University of Westminster / IIT
                """)

        # Footer
        gr.HTML("""
        <div style="text-align:center;padding:20px;color:#555;border-top:1px solid #222;margin-top:20px;">
            <p style="letter-spacing:1px;font-size:0.85em;">MRAF-Net © 2026 &nbsp;|&nbsp; Anne Nidhusha Nithiyalan &nbsp;|&nbsp; University of Westminster / IIT</p>
        </div>
        """)

        # Event handlers
        refresh_btn.click(refresh_checkpoints, [], [checkpoint_dropdown])
        load_btn.click(load_model_fn, [checkpoint_dropdown], [status])
        run_btn.click(
            run_segmentation,
            [checkpoint_dropdown, flair, t1, t1ce, t2, gt],
            [image, slice_slider, metrics_display],
        )
        slice_slider.change(update_view, [slice_slider, view, overlay, alpha], [image])
        view.change(update_view, [slice_slider, view, overlay, alpha], [image])
        overlay.change(update_view, [slice_slider, view, overlay, alpha], [image])
        alpha.change(update_view, [slice_slider, view, overlay, alpha], [image])
        plot_btn.click(create_3d, [], [plot])
        export_btn.click(export_seg, [], [export_file, export_status])
        sys_refresh_btn.click(get_system_info, [], [sys_info])

    return app


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  MRAF-Net Brain Tumor Segmentation GUI")
    print("  Anne Nidhusha Nithiyalan (w1985740)")
    print("  University of Westminster / IIT")
    print("="*60)
    if torch.cuda.is_available():
        print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  VRAM: {vram:.1f} GB")
    else:
        print("\n  Device: CPU")
    print("\n  Starting server at http://localhost:7860\n")

    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)
