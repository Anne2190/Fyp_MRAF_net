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
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict

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

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Use absolute path relative to project root
    "model_path": str(PROJECT_ROOT / "experiments/mraf_net_20260124_130245/checkpoints/best_model.pth"),
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "colors": {
        0: [0, 0, 0, 0],
        1: [0, 255, 0, 180],    # NCR/NET - Green
        2: [255, 255, 0, 180],  # Edema - Yellow
        4: [255, 0, 0, 180]     # ET - Red
    }
}


# ============================================================================
# MODEL HANDLER
# ============================================================================

class ModelHandler:
    """Handle model loading and inference."""
    
    def __init__(self):
        self.model = None
        self.device = CONFIG["device"]
        self.loaded = False
    
    def load(self, checkpoint_path: str) -> str:
        """Load model from checkpoint."""
        try:
            if not os.path.exists(checkpoint_path):
                return f"✗ File not found: {checkpoint_path}"
            
            # Create config to match training setup
            # IMPORTANT: deep_supervision=True to match checkpoint architecture
            config = {
                'data': {
                    'in_channels': 4,
                    'num_classes': 4
                },
                'model': {
                    'base_features': 32,
                    'deep_supervision': True,  # Must match checkpoint
                    'dropout': 0.0
                }
            }
            
            # Create model using the actual architecture
            self.model = create_model(config)
            
            # Load weights
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()
            
            self.loaded = True
            
            metrics = checkpoint.get("metrics", {})
            dice = metrics.get("dice_mean", "N/A")
            dice_str = f"{dice:.4f}" if isinstance(dice, float) else str(dice)
            
            return f"✓ Model loaded successfully\n▸ Device: {self.device}\n▸ Dice Score: {dice_str}"
            
        except Exception as e:
            return f"✗ Error: {str(e)}"
    
    def predict(self, images: np.ndarray, batch_size: int = 16) -> np.ndarray:
        """Run inference with batched processing to avoid OOM errors.
        
        Args:
            images: Input images of shape (C, H, W, D)
            batch_size: Number of slices to process at once (reduce if still OOM)
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded!")
        
        # Clear GPU cache before inference
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        with torch.no_grad():
            # (C, H, W, D) -> (C, D, H, W)
            images_t = np.transpose(images, (0, 3, 1, 2))
            C, D, H, W = images_t.shape
            
            # Initialize output array
            pred_np = np.zeros((D, H, W), dtype=np.uint8)
            
            # Process in batches along depth dimension
            for start_idx in range(0, D, batch_size):
                end_idx = min(start_idx + batch_size, D)
                
                # Get batch: (1, C, batch_D, H, W)
                batch = images_t[:, start_idx:end_idx, :, :]
                tensor = torch.from_numpy(batch).float().unsqueeze(0).to(self.device)
                
                # Forward pass (returns tuple: main_output, ds_outputs)
                output, _ = self.model(tensor)
                if isinstance(output, tuple):
                    output = output[0]
                pred = torch.argmax(F.softmax(output, dim=1), dim=1)
                
                # Move to CPU immediately to free GPU memory
                pred_batch = pred.squeeze(0).cpu().numpy()
                pred_np[start_idx:end_idx, :, :] = pred_batch
                
                # Clear GPU cache after each batch
                if self.device == "cuda":
                    torch.cuda.empty_cache()
            
            # (D, H, W) -> (H, W, D)
            pred_np = np.transpose(pred_np, (1, 2, 0))
            pred_np[pred_np == 3] = 4  # Convert back to BraTS labels
            
            return pred_np


# Global model instance
model = ModelHandler()
stored_data = None


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


def compute_metrics(seg: np.ndarray, gt: Optional[np.ndarray] = None) -> Dict:
    """Compute tumor metrics."""
    unique, counts = np.unique(seg, return_counts=True)
    vol_dict = dict(zip(unique.astype(int), counts))
    
    vol_ncr = vol_dict.get(1, 0) / 1000
    vol_ed = vol_dict.get(2, 0) / 1000
    vol_et = vol_dict.get(4, 0) / 1000
    
    metrics = {
        "volume": {
            "wt": round(vol_ncr + vol_ed + vol_et, 2),
            "tc": round(vol_ncr + vol_et, 2),
            "et": round(vol_et, 2)
        }
    }
    
    if gt is not None:
        def dice(p, g):
            i = np.sum(p & g)
            u = np.sum(p) + np.sum(g)
            return 2 * i / u if u > 0 else 1.0
        
        wt_p = (seg == 1) | (seg == 2) | (seg == 4)
        wt_g = (gt == 1) | (gt == 2) | (gt == 4)
        tc_p = (seg == 1) | (seg == 4)
        tc_g = (gt == 1) | (gt == 4)
        et_p = seg == 4
        et_g = gt == 4
        
        metrics["dice"] = {
            "wt": round(dice(wt_p, wt_g), 4),
            "tc": round(dice(tc_p, tc_g), 4),
            "et": round(dice(et_p, et_g), 4)
        }
        metrics["dice"]["mean"] = round(
            (metrics["dice"]["wt"] + metrics["dice"]["tc"] + metrics["dice"]["et"]) / 3, 4
        )
    
    return metrics


# ============================================================================
# GRADIO FUNCTIONS
# ============================================================================

def load_model_fn(path):
    return model.load(path)


def run_segmentation(flair, t1, t1ce, t2, gt=None):
    global stored_data
    
    if not all([flair, t1, t1ce, t2]):
        return None, 50, 100, "⚠ Upload all 4 modalities"
    
    if not model.loaded:
        return None, 50, 100, "✗ Load model first"
    
    try:
        # Load files
        flair_data, affine = nib.load(flair.name).get_fdata(), nib.load(flair.name).affine
        t1_data = nib.load(t1.name).get_fdata()
        t1ce_data = nib.load(t1ce.name).get_fdata()
        t2_data = nib.load(t2.name).get_fdata()
        
        # Stack and normalize
        images = np.stack([flair_data, t1_data, t1ce_data, t2_data], axis=0).astype(np.float32)
        images_norm = normalize_intensity(images)
        
        # Predict
        segmentation = model.predict(images_norm)
        
        # Ground truth
        ground_truth = None
        if gt:
            ground_truth = nib.load(gt.name).get_fdata().astype(np.int64)
        
        # Compute metrics
        metrics = compute_metrics(segmentation, ground_truth)
        
        # Store
        stored_data = {
            "flair": flair_data,
            "seg": segmentation,
            "gt": ground_truth,
            "affine": affine,
            "metrics": metrics
        }
        
        # Initial view
        mid = flair_data.shape[2] // 2
        viz = create_overlay(flair_data[:, :, mid], segmentation[:, :, mid])
        
        # Format metrics
        text = "## Results\n\n"
        text += "### Volumes (ml)\n"
        text += f"- Whole Tumor: **{metrics['volume']['wt']:.2f}**\n"
        text += f"- Tumor Core: **{metrics['volume']['tc']:.2f}**\n"
        text += f"- Enhancing: **{metrics['volume']['et']:.2f}**\n"
        
        if "dice" in metrics:
            text += "\n### Dice Scores\n"
            text += f"- WT: **{metrics['dice']['wt']:.4f}**\n"
            text += f"- TC: **{metrics['dice']['tc']:.4f}**\n"
            text += f"- ET: **{metrics['dice']['et']:.4f}**\n"
            text += f"- Mean: **{metrics['dice']['mean']:.4f}**\n"
        
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
    
    # Custom CSS — Black & White MRI Theme
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
                model_path = gr.Textbox(label="Checkpoint", value=CONFIG["model_path"])
                load_btn = gr.Button("⟳ Load Model", variant="primary")
                status = gr.Textbox(label="Status", lines=3, interactive=False)
                
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
        
        # Tabs for extra features
        with gr.Tabs():
            with gr.TabItem("◈ 3D View"):
                plot_btn = gr.Button("⧉ Generate 3D Plot")
                plot = gr.Plot()
            
            with gr.TabItem("⇓ Export"):
                export_btn = gr.Button("Export Segmentation")
                export_file = gr.File(label="Download")
                export_status = gr.Textbox(label="Status", interactive=False)
            
            with gr.TabItem("ℹ About"):
                gr.Markdown("""
                ## MRAF-Net Architecture
                
                **Multi-Resolution Aligned and Robust Fusion Network** for brain tumor segmentation.
                
                ### Key Components:
                - **Modality-Specific Encoders**: Process FLAIR, T1, T1ce, T2 independently
                - **Cross-Modality Fusion**: Integrate complementary information
                - **Attention Gates**: Enhance relevant features
                - **Deep Supervision**: Multi-scale loss computation
                
                ### BraTS Labels:
                | Label | Region | Description |
                |-------|--------|-------------|
                | 0 | Background | Healthy tissue |
                | 1 | NCR/NET | Necrotic Tumor Core |
                | 2 | ED | Peritumoral Edema |
                | 4 | ET | Enhancing Tumor |
                
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
        load_btn.click(load_model_fn, [model_path], [status])
        run_btn.click(run_segmentation, [flair, t1, t1ce, t2, gt], [image, slice_slider, slice_slider, metrics_display])
        slice_slider.change(update_view, [slice_slider, view, overlay, alpha], [image])
        view.change(update_view, [slice_slider, view, overlay, alpha], [image])
        overlay.change(update_view, [slice_slider, view, overlay, alpha], [image])
        alpha.change(update_view, [slice_slider, view, overlay, alpha], [image])
        plot_btn.click(create_3d, [], [plot])
        export_btn.click(export_seg, [], [export_file, export_status])
    
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
    print(f"\n  Device: {'CUDA - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("\n  Starting server at http://localhost:7860\n")
    
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)
