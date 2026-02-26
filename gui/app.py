"""
MRAF-Net Brain Tumor Segmentation - Research GUI Application
============================================================

Professional GUI for brain tumor segmentation demonstration
Designed for research presentation and clinical evaluation

Author: Anne Nidhusha Nithiyalan (w1985740)
Supervisor: Ms. Mohanadas Jananie
Institution: University of Westminster / IIT

Usage:
    python app.py

Then open: http://localhost:7860
"""

import os
import sys
import time
import json
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, List

import numpy as np
import gradio as gr
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import torch
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
    "patch_size": [128, 128, 128],
    "num_classes": 4,
    "modalities": ["FLAIR", "T1", "T1ce", "T2"],
    "tumor_labels": {
        0: "Background",
        1: "NCR/NET (Necrotic Core)",
        2: "ED (Peritumoral Edema)",
        4: "ET (Enhancing Tumor)"
    },
    "colors": {
        0: [0, 0, 0, 0],        # Background - transparent
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
        fallback = PROJECT_ROOT / "checkpoints"
        if fallback.exists():
            for ckpt_file in sorted(fallback.glob("*.pth")):
                checkpoints.append(str(ckpt_file))
    return checkpoints


# ============================================================================
# MODEL LOADING
# ============================================================================

class MRAFNetModel:
    """MRAF-Net model wrapper for inference."""

    def __init__(self):
        self.model = None
        self.device = CONFIG["device"]
        self.loaded = False
        self.param_count = {}

    def load(self, checkpoint_path: str) -> str:
        """Load model from checkpoint, auto-detecting architecture flags."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            saved_keys = set(checkpoint["model_state_dict"].keys())

            # Auto-detect architecture from saved weights
            has_swin = any("swin_bottleneck" in k for k in saved_keys)

            config = {
                'data': {
                    'in_channels': 4,
                    'num_classes': CONFIG["num_classes"]
                },
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
            return f"✗ Error loading model: {str(e)}"

    def predict(self, images: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Run prediction on preprocessed images.

        Returns:
            (segmentation, inference_time_sec, peak_gpu_mb)
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded!")

        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)

        t_start = time.perf_counter()

        with torch.no_grad():
            # Convert to tensor: (C, H, W, D) -> (1, C, D, H, W)
            images_t = np.transpose(images, (0, 3, 1, 2))
            tensor = torch.from_numpy(images_t).float().unsqueeze(0).to(self.device)

            output, _ = self.model(tensor)
            if isinstance(output, tuple):
                output = output[0]

            pred = torch.argmax(F.softmax(output, dim=1), dim=1)
            pred_np = pred.squeeze(0).cpu().numpy()

            # Transpose back: (D, H, W) -> (H, W, D)
            pred_np = np.transpose(pred_np, (1, 2, 0))
            pred_np[pred_np == 3] = 4  # Restore BraTS label

        inference_time = time.perf_counter() - t_start

        peak_gpu_mb = 0.0
        if self.device == "cuda":
            peak_gpu_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)

        return pred_np, inference_time, peak_gpu_mb


# Global model instance
model = MRAFNetModel()
stored_data = None


# ============================================================================
# SYSTEM INFO
# ============================================================================

def get_system_info() -> str:
    """Return a formatted hardware/system report."""
    lines = ["## System Information\n"]

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        lines.append(f"**GPU:** {gpu_name}")
        lines.append(f"**VRAM:** {total_vram:.1f} GB")
        lines.append(f"**CUDA Version:** {torch.version.cuda}")
    else:
        lines.append("**GPU:** Not available (running on CPU)")

    import platform
    lines.append(f"**Platform:** {platform.system()} {platform.release()}")
    lines.append(f"**Python:** {platform.python_version()}")
    lines.append(f"**PyTorch:** {torch.__version__}")

    if model.loaded and model.param_count:
        total = model.param_count["total"]
        lines.append(f"\n**Model Parameters:** {total:,}")

    return "\n\n".join(lines)


# ============================================================================
# DATA PROCESSING
# ============================================================================

def load_nifti(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load NIfTI file."""
    nii = nib.load(file_path)
    data = nii.get_fdata().astype(np.float32)
    affine = nii.affine
    return data, affine


def normalize_intensity(images: np.ndarray) -> np.ndarray:
    """Z-score normalization per modality."""
    normalized = np.zeros_like(images)
    for i in range(images.shape[0]):
        img = images[i]
        mask = img > 0
        if mask.sum() > 0:
            mean = img[mask].mean()
            std = img[mask].std() + 1e-8
            normalized[i] = np.where(mask, (img - mean) / std, 0)
    return normalized


def create_overlay_slice(mri_slice: np.ndarray,
                         seg_slice: np.ndarray,
                         alpha: float = 0.5) -> np.ndarray:
    """Create MRI slice with segmentation overlay."""
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


def compute_tumor_metrics(
    segmentation: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    voxel_volume: float = 1.0,
    inference_time: float = 0.0,
    peak_gpu_mb: float = 0.0,
) -> Dict:
    """Compute tumor volumes, Dice, HD95, Sensitivity, and Specificity."""
    metrics = {}

    # Volume statistics
    unique, counts = np.unique(segmentation, return_counts=True)
    vol_dict = dict(zip(unique.astype(int), counts))

    vol_ncr = vol_dict.get(1, 0) * voxel_volume / 1000
    vol_ed = vol_dict.get(2, 0) * voxel_volume / 1000
    vol_et = vol_dict.get(4, 0) * voxel_volume / 1000

    metrics["volume"] = {
        "whole_tumor_ml": round(vol_ncr + vol_ed + vol_et, 2),
        "tumor_core_ml": round(vol_ncr + vol_et, 2),
        "enhancing_ml": round(vol_et, 2),
        "edema_ml": round(vol_ed, 2),
        "necrotic_ml": round(vol_ncr, 2),
    }

    metrics["inference_time"] = inference_time
    metrics["peak_gpu_mb"] = peak_gpu_mb

    if ground_truth is not None:
        # Convert label 4 -> 3 to match metrics module convention
        seg_int = segmentation.copy().astype(np.int64)
        seg_int[seg_int == 4] = 3
        gt_int = ground_truth.copy().astype(np.int64)
        gt_int[gt_int == 4] = 3

        pred_wt, pred_tc, pred_et, tgt_wt, tgt_tc, tgt_et = compute_brats_regions(seg_int, gt_int)

        metrics["dice"] = {
            "whole_tumor": round(compute_dice(pred_wt, tgt_wt), 4),
            "tumor_core": round(compute_dice(pred_tc, tgt_tc), 4),
            "enhancing": round(compute_dice(pred_et, tgt_et), 4),
        }
        metrics["dice"]["mean"] = round(
            (metrics["dice"]["whole_tumor"] + metrics["dice"]["tumor_core"] + metrics["dice"]["enhancing"]) / 3, 4
        )

        metrics["sensitivity"] = {
            "whole_tumor": round(compute_sensitivity(pred_wt, tgt_wt), 4),
            "tumor_core": round(compute_sensitivity(pred_tc, tgt_tc), 4),
            "enhancing": round(compute_sensitivity(pred_et, tgt_et), 4),
        }

        metrics["specificity"] = {
            "whole_tumor": round(compute_specificity(pred_wt, tgt_wt), 4),
            "tumor_core": round(compute_specificity(pred_tc, tgt_tc), 4),
            "enhancing": round(compute_specificity(pred_et, tgt_et), 4),
        }

        spacing = (voxel_volume ** (1 / 3),) * 3
        hd95_wt = compute_hausdorff95(pred_wt, tgt_wt, spacing)
        hd95_tc = compute_hausdorff95(pred_tc, tgt_tc, spacing)
        hd95_et = compute_hausdorff95(pred_et, tgt_et, spacing)
        metrics["hd95"] = {
            "whole_tumor": round(hd95_wt, 2) if not np.isinf(hd95_wt) else "N/A",
            "tumor_core": round(hd95_tc, 2) if not np.isinf(hd95_tc) else "N/A",
            "enhancing": round(hd95_et, 2) if not np.isinf(hd95_et) else "N/A",
        }

    return metrics


# ============================================================================
# GRADIO INTERFACE FUNCTIONS
# ============================================================================

def refresh_checkpoints():
    """Return updated Dropdown choices."""
    choices = discover_checkpoints()
    value = choices[0] if choices else ""
    return gr.update(choices=choices, value=value)


def load_model_handler(checkpoint_path: str) -> str:
    """Handle model loading."""
    if not checkpoint_path:
        return "⚠ Please provide a checkpoint path"
    if not os.path.exists(checkpoint_path):
        return f"✗ File not found: {checkpoint_path}"
    return model.load(checkpoint_path)


def process_mri_files(flair_file, t1_file, t1ce_file, t2_file, gt_file=None):
    """Process uploaded MRI files and run segmentation."""
    if not all([flair_file, t1_file, t1ce_file, t2_file]):
        return None, None, None, "⚠ Please upload all 4 MRI modalities"

    if not model.loaded:
        return None, None, None, "✗ Please load a model first"

    try:
        flair_data, affine = load_nifti(flair_file.name)
        t1_data, _ = load_nifti(t1_file.name)
        t1ce_data, _ = load_nifti(t1ce_file.name)
        t2_data, _ = load_nifti(t2_file.name)

        images = np.stack([flair_data, t1_data, t1ce_data, t2_data], axis=0)
        images_norm = normalize_intensity(images)

        segmentation, inference_time, peak_gpu_mb = model.predict(images_norm)

        ground_truth = None
        if gt_file is not None:
            ground_truth, _ = load_nifti(gt_file.name)

        voxel_vol = float(np.abs(np.linalg.det(affine[:3, :3])))
        metrics = compute_tumor_metrics(
            segmentation, ground_truth, voxel_vol, inference_time, peak_gpu_mb
        )

        global stored_data
        stored_data = {
            "flair": flair_data,
            "segmentation": segmentation,
            "ground_truth": ground_truth,
            "affine": affine,
            "metrics": metrics,
            "shape": flair_data.shape,
        }

        mid_slice = flair_data.shape[2] // 2
        viz_image = create_overlay_slice(
            flair_data[:, :, mid_slice],
            segmentation[:, :, mid_slice]
        )

        metrics_text = format_metrics(metrics)
        return Image.fromarray(viz_image), mid_slice, flair_data.shape[2] - 1, metrics_text

    except Exception as e:
        import traceback
        return None, None, None, f"✗ Error: {str(e)}\n{traceback.format_exc()}"


def update_slice_view(slice_idx: int, view: str, show_overlay: bool, alpha: float):
    """Update slice visualization."""
    global stored_data

    if stored_data is None:
        return None

    flair = stored_data["flair"]
    seg = stored_data["segmentation"]

    if view == "Axial":
        mri_slice = flair[:, :, int(slice_idx)]
        seg_slice = seg[:, :, int(slice_idx)]
    elif view == "Coronal":
        mri_slice = flair[:, int(slice_idx), :]
        seg_slice = seg[:, int(slice_idx), :]
    else:
        mri_slice = flair[int(slice_idx), :, :]
        seg_slice = seg[int(slice_idx), :, :]

    if show_overlay:
        viz_image = create_overlay_slice(mri_slice, seg_slice, alpha)
    else:
        mri_norm = (mri_slice - mri_slice.min()) / (mri_slice.max() - mri_slice.min() + 1e-8)
        viz_image = (np.stack([mri_norm] * 3, axis=-1) * 255).astype(np.uint8)

    return Image.fromarray(viz_image)


def format_metrics(metrics: Dict) -> str:
    """Format metrics for display."""
    text = "## Segmentation Results\n\n"

    text += "### Tumor Volumes\n"
    text += "| Region | Volume (ml) |\n|--------|-------------|\n"
    text += f"| Whole Tumor (WT) | {metrics['volume']['whole_tumor_ml']:.2f} |\n"
    text += f"| Tumor Core (TC) | {metrics['volume']['tumor_core_ml']:.2f} |\n"
    text += f"| Enhancing (ET) | {metrics['volume']['enhancing_ml']:.2f} |\n"
    text += f"| Edema (ED) | {metrics['volume']['edema_ml']:.2f} |\n"
    text += f"| Necrotic (NCR) | {metrics['volume']['necrotic_ml']:.2f} |\n\n"

    text += "### Performance\n"
    text += f"- **Inference Time:** {metrics['inference_time']:.3f} s\n"
    if metrics['peak_gpu_mb'] > 0:
        text += f"- **Peak GPU Memory:** {metrics['peak_gpu_mb']:.1f} MB\n"
    text += "\n"

    if "dice" in metrics:
        text += "### Evaluation Metrics\n"
        text += "| Region | Dice | Sensitivity | Specificity | HD95 (mm) |\n"
        text += "|--------|------|-------------|-------------|----------|\n"
        for key, label in [
            ("whole_tumor", "Whole Tumor"),
            ("tumor_core", "Tumor Core"),
            ("enhancing", "Enhancing"),
        ]:
            dice = metrics["dice"][key]
            sens = metrics["sensitivity"][key]
            spec = metrics["specificity"][key]
            hd95 = metrics["hd95"][key]
            text += f"| {label} | {dice:.4f} | {sens:.4f} | {spec:.4f} | {hd95} |\n"
        text += f"\n**Mean Dice: {metrics['dice']['mean']:.4f}**\n"

    return text


def create_3d_plot():
    """Create 3D visualization plot."""
    global stored_data

    if stored_data is None:
        return None

    seg = stored_data["segmentation"]
    step = 3
    seg_down = seg[::step, ::step, ::step]

    fig = plt.figure(figsize=(10, 8), facecolor='#0a0a0a')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0a0a0a')

    colors = {1: '#ffffff', 2: '#aaaaaa', 4: '#555555'}
    labels = {1: 'NCR/NET', 2: 'Edema', 4: 'Enhancing'}

    for label, color in colors.items():
        mask = seg_down == label
        if np.any(mask):
            coords = np.argwhere(mask)
            if len(coords) > 2000:
                indices = np.random.choice(len(coords), 2000, replace=False)
                coords = coords[indices]
            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                       c=color, alpha=0.4, s=1, label=labels[label])

    ax.set_xlabel('X', color='#aaaaaa')
    ax.set_ylabel('Y', color='#aaaaaa')
    ax.set_zlabel('Z', color='#aaaaaa')
    ax.tick_params(colors='#666666')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333333', labelcolor='#cccccc')
    ax.set_title('3D Tumor Visualization', color='#ffffff')

    plt.tight_layout()
    return fig


def export_segmentation():
    """Export segmentation as NIfTI."""
    global stored_data

    if stored_data is None:
        return None, "⚠ No segmentation to export"

    output_path = tempfile.mktemp(suffix='.nii.gz')
    nii = nib.Nifti1Image(
        stored_data["segmentation"].astype(np.int16),
        stored_data["affine"]
    )
    nib.save(nii, output_path)
    return output_path, f"✓ Saved to: {output_path}"


# ============================================================================
# GRADIO UI DEFINITION
# ============================================================================

def create_interface():
    """Create the Gradio interface."""

    css = """
    /* ── Global Dark Background ── */
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
    /* ── Header ── */
    .header-text {
        text-align: center;
        background: radial-gradient(ellipse at center, #1a1a1a 0%, #000000 70%);
        padding: 30px 20px;
        border-radius: 4px;
        color: #ffffff;
        margin-bottom: 20px;
        border: 1px solid #333333;
        box-shadow: 0 0 40px rgba(255,255,255,0.03);
    }
    .header-text h1 { font-weight: 300; letter-spacing: 6px; text-transform: uppercase; color: #ffffff; }
    .header-text h3 { font-weight: 300; letter-spacing: 2px; color: #aaaaaa; }
    .header-text p  { color: #888888; font-size: 0.9em; }
    /* ── Legend ── */
    .legend-box {
        background: #111111; padding: 12px 16px; border-radius: 4px;
        border: 1px solid #333333; color: #cccccc;
    }
    /* ── Gradio Component Overrides ── */
    .gr-box, .gr-panel, .gr-form, .gr-input, .gr-padded {
        background: #111111 !important; border-color: #333333 !important; color: #e0e0e0 !important;
    }
    .gr-button-primary { background: #ffffff !important; color: #000000 !important; border: none !important; font-weight: 600; }
    .gr-button-primary:hover { background: #cccccc !important; }
    .gr-button-secondary { background: #1a1a1a !important; color: #e0e0e0 !important; border: 1px solid #444444 !important; }
    label, .gr-check-radio, .label-wrap { color: #cccccc !important; }
    input, textarea, select { background: #1a1a1a !important; color: #e0e0e0 !important; border-color: #333333 !important; }
    .tabs .tab-nav button { color: #888888 !important; background: transparent !important; }
    .tabs .tab-nav button.selected { color: #ffffff !important; border-bottom: 2px solid #ffffff !important; }
    .prose h1,.prose h2,.prose h3,.prose h4,.prose p,.prose li,.prose td,.prose th { color: #e0e0e0 !important; }
    .prose table, .prose th, .prose td { border-color: #333333 !important; }
    .prose code { background: #1a1a1a !important; color: #cccccc !important; }
    .prose pre  { background: #0d0d0d !important; }
    footer { display: none !important; }
    """

    initial_checkpoints = discover_checkpoints()
    initial_value = initial_checkpoints[0] if initial_checkpoints else ""

    with gr.Blocks(css=css, title="MRAF-Net Brain Tumor Segmentation", theme=gr.themes.Base(
        primary_hue=gr.themes.Color(c50="#f5f5f5",c100="#e0e0e0",c200="#cccccc",c300="#aaaaaa",c400="#888888",c500="#666666",c600="#444444",c700="#333333",c800="#1a1a1a",c900="#0a0a0a",c950="#000000"),
        secondary_hue=gr.themes.Color(c50="#f5f5f5",c100="#e0e0e0",c200="#cccccc",c300="#aaaaaa",c400="#888888",c500="#666666",c600="#444444",c700="#333333",c800="#1a1a1a",c900="#0a0a0a",c950="#000000"),
        neutral_hue=gr.themes.Color(c50="#f5f5f5",c100="#e0e0e0",c200="#cccccc",c300="#aaaaaa",c400="#888888",c500="#666666",c600="#444444",c700="#333333",c800="#1a1a1a",c900="#0a0a0a",c950="#000000"),
    )) as demo:

        # Header
        gr.HTML("""
        <div class="header-text">
            <h1 style="margin:0 0 8px 0;">⬡ MRAF-Net</h1>
            <h3 style="margin:4px 0;">Multi-Resolution Aligned and Robust Fusion Network</h3>
            <p style="margin:4px 0;">Brain Tumor Segmentation from Multi-Modal MRI</p>
            <hr style="border:none;border-top:1px solid #333;margin:12px auto;width:60%;">
            <p>Anne Nidhusha Nithiyalan (w1985740) &nbsp;|&nbsp; Supervisor: Ms. Mohanadas Jananie</p>
            <p>BEng Software Engineering &nbsp;|&nbsp; University of Westminster / IIT</p>
        </div>
        """)

        with gr.Tabs():
            # Tab 1: Segmentation
            with gr.TabItem("⊕ Segmentation"):
                with gr.Row():
                    # Left column - Inputs
                    with gr.Column(scale=1):
                        gr.Markdown("### ⎙ Load Model")

                        checkpoint_dropdown = gr.Dropdown(
                            label="Checkpoint",
                            choices=initial_checkpoints,
                            value=initial_value,
                            allow_custom_value=True,
                        )
                        with gr.Row():
                            refresh_btn = gr.Button("⟳ Refresh List", variant="secondary", size="sm")
                            load_btn = gr.Button("⟳ Load Model", variant="primary")

                        model_status = gr.Textbox(label="Model Status", lines=4, interactive=False)

                        gr.Markdown("### ⇪ Upload MRI Scans")
                        flair_input = gr.File(label="FLAIR", file_types=[".nii", ".nii.gz"])
                        t1_input = gr.File(label="T1", file_types=[".nii", ".nii.gz"])
                        t1ce_input = gr.File(label="T1ce (Contrast)", file_types=[".nii", ".nii.gz"])
                        t2_input = gr.File(label="T2", file_types=[".nii", ".nii.gz"])
                        gt_input = gr.File(label="Ground Truth (Optional)", file_types=[".nii", ".nii.gz"])

                        run_btn = gr.Button("▶ Run Segmentation", variant="primary", size="lg")

                    # Right column - Visualization
                    with gr.Column(scale=2):
                        gr.Markdown("### ⊞ Visualization")

                        with gr.Row():
                            view_select = gr.Radio(
                                choices=["Axial", "Coronal", "Sagittal"],
                                value="Axial",
                                label="View"
                            )
                            show_overlay = gr.Checkbox(label="Show Overlay", value=True)
                            alpha_slider = gr.Slider(0, 1, value=0.5, label="Overlay Opacity")

                        slice_slider = gr.Slider(0, 100, value=50, step=1, label="Slice")
                        output_image = gr.Image(label="Segmentation Result", type="pil")

                        gr.HTML("""
                        <div class="legend-box">
                            <b style="color:#ffffff;">Legend:</b>&nbsp;&nbsp;
                            <span style="color:#ffffff;border:1px solid #ffffff;padding:2px 8px;border-radius:2px;margin:0 4px;font-size:0.85em;">■ NCR/NET</span>
                            <span style="color:#aaaaaa;border:1px solid #aaaaaa;padding:2px 8px;border-radius:2px;margin:0 4px;font-size:0.85em;">■ Edema</span>
                            <span style="color:#666666;border:1px solid #666666;padding:2px 8px;border-radius:2px;margin:0 4px;font-size:0.85em;">■ Enhancing</span>
                        </div>
                        """)

                with gr.Row():
                    metrics_output = gr.Markdown("### ▸ Upload MRI scans and run segmentation to see metrics")

            # Tab 2: 3D Visualization
            with gr.TabItem("◈ 3D View"):
                gr.Markdown("### 3D Tumor Visualization")
                plot_btn = gr.Button("⧉ Generate 3D Plot")
                plot_output = gr.Plot(label="3D Visualization")

            # Tab 3: Export
            with gr.TabItem("⇓ Export"):
                gr.Markdown("### Export Results")
                export_btn = gr.Button("Export Segmentation (NIfTI)")
                export_file = gr.File(label="Download")
                export_status = gr.Textbox(label="Status", interactive=False)

            # Tab 4: System Info
            with gr.TabItem("⊡ System Info"):
                sys_refresh_btn = gr.Button("⟳ Refresh")
                sys_info = gr.Markdown(get_system_info())

            # Tab 5: About
            with gr.TabItem("ℹ About"):
                gr.Markdown("""
                ## MRAF-Net: Multi-Resolution Aligned and Robust Fusion Network

                ### Overview
                MRAF-Net is an advanced deep learning architecture designed for automatic brain tumor
                segmentation from multi-modal MRI scans. It addresses three key challenges:

                1. **Multi-Resolution Harmonization**: Handles varying MRI resolutions
                2. **Cross-Modality Fusion**: Integrates FLAIR, T1, T1ce, and T2 information
                3. **Edge-Aware Attention**: Improves tumor boundary accuracy

                ### BraTS Challenge Labels
                | Label | Region | Description |
                |-------|--------|-------------|
                | 0 | Background | Healthy tissue |
                | 1 | NCR/NET | Necrotic and Non-Enhancing Tumor Core |
                | 2 | ED | Peritumoral Edema |
                | 4 | ET | GD-Enhancing Tumor |

                ### Evaluation Regions
                - **Whole Tumor (WT)**: Labels 1 + 2 + 4
                - **Tumor Core (TC)**: Labels 1 + 4
                - **Enhancing Tumor (ET)**: Label 4

                ### Metrics Reported
                - **Dice Score**: Overlap similarity (0–1, higher = better)
                - **Sensitivity**: True positive rate
                - **Specificity**: True negative rate
                - **HD95**: 95th percentile Hausdorff Distance in mm (lower = better)
                - **Inference Time**: Seconds per scan
                - **Peak GPU Memory**: MB allocated during inference

                ---

                ### Author Information
                **Student**: Anne Nidhusha Nithiyalan (w1985740)
                **Supervisor**: Ms. Mohanadas Jananie
                **Programme**: BEng (Hons) Software Engineering
                **Institution**: University of Westminster / Informatics Institute of Technology
                **Year**: 2026

                ### Citation
                ```bibtex
                @thesis{nithiyalan2026mrafnet,
                    title={MRAF-Net: Multi-Resolution Aligned and Robust Fusion Network
                           for Brain Tumor Segmentation},
                    author={Nithiyalan, Anne Nidhusha},
                    year={2026},
                    school={University of Westminster / IIT},
                    type={BEng Dissertation}
                }
                ```
                """)

        # Footer
        gr.HTML("""
        <div style="text-align:center;padding:20px;color:#555;border-top:1px solid #222;margin-top:20px;">
            <p style="letter-spacing:1px;font-size:0.85em;">MRAF-Net Brain Tumor Segmentation System &nbsp;|&nbsp; © 2026 Anne Nidhusha Nithiyalan</p>
            <p style="font-size:0.8em;color:#444;">University of Westminster &nbsp;|&nbsp; Informatics Institute of Technology</p>
        </div>
        """)

        # Event handlers
        refresh_btn.click(refresh_checkpoints, [], [checkpoint_dropdown])

        load_btn.click(
            fn=load_model_handler,
            inputs=[checkpoint_dropdown],
            outputs=[model_status]
        )

        run_btn.click(
            fn=process_mri_files,
            inputs=[flair_input, t1_input, t1ce_input, t2_input, gt_input],
            outputs=[output_image, slice_slider, slice_slider, metrics_output]
        )

        slice_slider.change(
            fn=update_slice_view,
            inputs=[slice_slider, view_select, show_overlay, alpha_slider],
            outputs=[output_image]
        )

        view_select.change(
            fn=update_slice_view,
            inputs=[slice_slider, view_select, show_overlay, alpha_slider],
            outputs=[output_image]
        )

        show_overlay.change(
            fn=update_slice_view,
            inputs=[slice_slider, view_select, show_overlay, alpha_slider],
            outputs=[output_image]
        )

        alpha_slider.change(
            fn=update_slice_view,
            inputs=[slice_slider, view_select, show_overlay, alpha_slider],
            outputs=[output_image]
        )

        plot_btn.click(fn=create_3d_plot, outputs=[plot_output])
        export_btn.click(fn=export_segmentation, outputs=[export_file, export_status])
        sys_refresh_btn.click(fn=get_system_info, outputs=[sys_info])

    return demo


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  MRAF-Net Brain Tumor Segmentation GUI")
    print("  Author: Anne Nidhusha Nithiyalan (w1985740)")
    print("  University of Westminster / IIT")
    print("="*60 + "\n")

    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  VRAM: {vram:.1f} GB")
    else:
        print("⚠ CUDA not available, using CPU")

    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )
