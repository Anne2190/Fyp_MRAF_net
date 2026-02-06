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
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.mraf_net import MRAFNet, create_model

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "model_path": "experiments/mraf_net_20260124_130245/checkpoints/best_model.pth",
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
                return f"❌ File not found: {checkpoint_path}"
            
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
            
            return f"✅ Model loaded!\n📍 Device: {self.device}\n📊 Dice: {dice_str}"
            
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def predict(self, images: np.ndarray) -> np.ndarray:
        """Run inference."""
        if not self.loaded:
            raise RuntimeError("Model not loaded!")
        
        with torch.no_grad():
            # (C, H, W, D) -> (1, C, D, H, W)
            images_t = np.transpose(images, (0, 3, 1, 2))
            tensor = torch.from_numpy(images_t).float().unsqueeze(0).to(self.device)
            
            # Forward pass (returns tuple: main_output, ds_outputs)
            output, _ = self.model(tensor)
            if isinstance(output, tuple):
                output = output[0]
            pred = torch.argmax(F.softmax(output, dim=1), dim=1)
            pred_np = pred.squeeze(0).cpu().numpy()
            
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
        return None, 50, 100, "⚠️ Upload all 4 modalities"
    
    if not model.loaded:
        return None, 50, 100, "❌ Load model first"
    
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
        text = "## 📊 Results\n\n"
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
        return None, 50, 100, f"❌ Error: {e}\n{traceback.format_exc()}"


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
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for label, color, name in [(1, 'green', 'NCR'), (2, 'yellow', 'Edema'), (4, 'red', 'ET')]:
        mask = seg == label
        if mask.any():
            coords = np.argwhere(mask)
            if len(coords) > 1500:
                coords = coords[np.random.choice(len(coords), 1500, replace=False)]
            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=color, alpha=0.3, s=1, label=name)
    
    ax.legend()
    ax.set_title('3D Tumor Visualization')
    return fig


def export_seg():
    global stored_data
    if stored_data is None:
        return None, "⚠️ No segmentation"
    
    path = tempfile.mktemp(suffix='.nii.gz')
    nii = nib.Nifti1Image(stored_data["seg"].astype(np.int16), stored_data["affine"])
    nib.save(nii, path)
    return path, f"✅ Exported: {path}"


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_app():
    """Create Gradio interface."""
    
    with gr.Blocks(title="MRAF-Net Brain Tumor Segmentation", theme=gr.themes.Soft()) as app:
        
        # Header
        gr.HTML("""
        <div style="text-align:center; background:linear-gradient(135deg,#667eea,#764ba2); 
                    padding:25px; border-radius:10px; color:white; margin-bottom:20px;">
            <h1 style="margin:0;">🧠 MRAF-Net</h1>
            <h3 style="margin:5px 0;">Multi-Resolution Aligned and Robust Fusion Network</h3>
            <p style="margin:5px 0;">Brain Tumor Segmentation from Multi-Modal MRI</p>
            <p style="font-size:0.9em;">Anne Nidhusha Nithiyalan | University of Westminster / IIT</p>
        </div>
        """)
        
        with gr.Row():
            # Left Panel
            with gr.Column(scale=1):
                gr.Markdown("### 🔧 Model")
                model_path = gr.Textbox(label="Checkpoint", value=CONFIG["model_path"])
                load_btn = gr.Button("Load Model", variant="primary")
                status = gr.Textbox(label="Status", lines=3, interactive=False)
                
                gr.Markdown("### 📤 Upload MRI")
                flair = gr.File(label="FLAIR", file_types=[".nii", ".nii.gz"])
                t1 = gr.File(label="T1", file_types=[".nii", ".nii.gz"])
                t1ce = gr.File(label="T1ce", file_types=[".nii", ".nii.gz"])
                t2 = gr.File(label="T2", file_types=[".nii", ".nii.gz"])
                gt = gr.File(label="Ground Truth (optional)", file_types=[".nii", ".nii.gz"])
                
                run_btn = gr.Button("🧠 Run Segmentation", variant="primary", size="lg")
            
            # Right Panel
            with gr.Column(scale=2):
                gr.Markdown("### 🖼️ Visualization")
                
                with gr.Row():
                    view = gr.Radio(["Axial", "Coronal", "Sagittal"], value="Axial", label="View")
                    overlay = gr.Checkbox(value=True, label="Overlay")
                    alpha = gr.Slider(0, 1, 0.5, label="Opacity")
                
                slice_slider = gr.Slider(0, 100, 50, step=1, label="Slice")
                image = gr.Image(label="Result", type="pil", height=450)
                
                gr.HTML("""
                <div style="background:#f1f5f9;padding:10px;border-radius:8px;text-align:center;">
                    <b>Legend:</b>
                    <span style="color:green;">■ NCR/NET</span> |
                    <span style="color:#DAA520;">■ Edema</span> |
                    <span style="color:red;">■ Enhancing</span>
                </div>
                """)
        
        # Metrics
        metrics_display = gr.Markdown("### 📊 Run segmentation to see metrics")
        
        # Tabs for extra features
        with gr.Tabs():
            with gr.TabItem("🎮 3D View"):
                plot_btn = gr.Button("Generate 3D Plot")
                plot = gr.Plot()
            
            with gr.TabItem("💾 Export"):
                export_btn = gr.Button("Export Segmentation")
                export_file = gr.File(label="Download")
                export_status = gr.Textbox(label="Status", interactive=False)
            
            with gr.TabItem("📖 About"):
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
        <div style="text-align:center;padding:20px;color:#666;border-top:1px solid #eee;margin-top:20px;">
            MRAF-Net © 2026 | Anne Nidhusha Nithiyalan | University of Westminster / IIT
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
