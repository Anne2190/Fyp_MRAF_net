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
import tempfile
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
from PIL import Image

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


def format_metrics_text(metrics: Dict) -> str:
    """Format metrics dict into Markdown for display."""
    text = "## Results\n\n"

    # Volumes
    text += "### Volumes (ml)\n"
    text += f"| Region | Volume |\n|--------|--------|\n"
    text += f"| Whole Tumor (WT) | {metrics['volume']['wt']:.2f} |\n"
    text += f"| Tumor Core (TC) | {metrics['volume']['tc']:.2f} |\n"
    text += f"| Enhancing (ET) | {metrics['volume']['et']:.2f} |\n\n"

    # Performance
    text += "### Performance\n"
    text += f"- **Inference Time:** {metrics['inference_time']:.3f} s\n"
    if metrics['peak_gpu_mb'] > 0:
        text += f"- **Peak GPU Memory:** {metrics['peak_gpu_mb']:.1f} MB\n"
    text += "\n"

    if "dice" in metrics:
        # Dice
        text += "### Dice Scores\n"
        text += "| Region | Dice | Sensitivity | Specificity | HD95 (mm) |\n"
        text += "|--------|------|-------------|-------------|----------|\n"
        for key, label in [("wt", "Whole Tumor"), ("tc", "Tumor Core"), ("et", "Enhancing")]:
            dice = metrics["dice"][key]
            sens = metrics["sensitivity"][key]
            spec = metrics["specificity"][key]
            hd95 = metrics["hd95"][key]
            text += f"| {label} | {dice:.4f} | {sens:.4f} | {spec:.4f} | {hd95} |\n"
        text += f"\n**Mean Dice: {metrics['dice']['mean']:.4f}**\n"

    return text


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
        return None, 50, 100, "⚠ Upload all 4 modalities"

    if not model.loaded:
        return None, 50, 100, "✗ Load model first"

    try:
        flair_data = nib.load(flair.name).get_fdata().astype(np.float32)
        affine = nib.load(flair.name).affine
        t1_data = nib.load(t1.name).get_fdata().astype(np.float32)
        t1ce_data = nib.load(t1ce.name).get_fdata().astype(np.float32)
        t2_data = nib.load(t2.name).get_fdata().astype(np.float32)

        images = np.stack([flair_data, t1_data, t1ce_data, t2_data], axis=0)
        images_norm = normalize_intensity(images)

        segmentation, inference_time, peak_gpu_mb = model.predict(images_norm)

        ground_truth = None
        if gt:
            ground_truth = nib.load(gt.name).get_fdata().astype(np.int64)

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
        viz = create_overlay(flair_data[:, :, mid], segmentation[:, :, mid])

        text = format_metrics_text(metrics)
        return Image.fromarray(viz), mid, flair_data.shape[2] - 1, text

    except Exception as e:
        import traceback
        return None, 50, 100, f"✗ Error: {e}\n{traceback.format_exc()}"


def update_view(slice_idx, view, overlay, alpha):
    global stored_data
    if stored_data is None:
        return None

    flair = stored_data["flair"]
    seg = stored_data["seg"]
    idx = int(slice_idx)

    if view == "Axial":
        mri_s, seg_s = flair[:, :, idx], seg[:, :, idx]
    elif view == "Coronal":
        mri_s, seg_s = flair[:, idx, :], seg[:, idx, :]
    else:
        mri_s, seg_s = flair[idx, :, :], seg[idx, :, :]

    if overlay:
        viz = create_overlay(mri_s, seg_s, alpha)
    else:
        norm = (mri_s - mri_s.min()) / (mri_s.max() - mri_s.min() + 1e-8)
        viz = (np.stack([norm] * 3, axis=-1) * 255).astype(np.uint8)

    return Image.fromarray(viz)


def create_3d():
    global stored_data
    if stored_data is None:
        return None

    seg = stored_data["seg"][::3, ::3, ::3]

    fig = plt.figure(figsize=(10, 8), facecolor='#0a0a0a')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0a0a0a')

    for label, color, name in [(1, '#ffffff', 'NCR'), (2, '#aaaaaa', 'Edema'), (4, '#555555', 'ET')]:
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
                    <span style="color:#fff;border:1px solid #fff;padding:2px 8px;border-radius:2px;margin:0 4px;font-size:0.85em;">■ NCR/NET</span>
                    <span style="color:#aaa;border:1px solid #aaa;padding:2px 8px;border-radius:2px;margin:0 4px;font-size:0.85em;">■ Edema</span>
                    <span style="color:#666;border:1px solid #666;padding:2px 8px;border-radius:2px;margin:0 4px;font-size:0.85em;">■ Enhancing</span>
                </div>
                """)

        # Metrics
        metrics_display = gr.Markdown("### Run segmentation to see metrics")

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
                - **Dice Score**: Overlap similarity (higher = better)
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
            [image, slice_slider, slice_slider, metrics_display],
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
