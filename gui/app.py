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


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "model_path": "experiments/mraf_net_20260124_130245/checkpoints/best_model.pth",
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


# ============================================================================
# MODEL LOADING
# ============================================================================

class MRAFNetModel:
    """MRAF-Net model wrapper for inference."""
    
    def __init__(self):
        self.model = None
        self.device = CONFIG["device"]
        self.loaded = False
        
    def load(self, checkpoint_path: str) -> str:
        """Load model from checkpoint."""
        try:
            # Import model architecture
            from src.models.mraf_net import MRAFNet
            
            # Initialize model
            self.model = MRAFNet(
                in_channels=4,
                num_classes=CONFIG["num_classes"],
                base_features=32,
                deep_supervision=False
            )
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()
            
            self.loaded = True
            
            # Get training info
            epoch = checkpoint.get("epoch", "N/A")
            metrics = checkpoint.get("metrics", {})
            dice = metrics.get("dice_mean", "N/A")
            
            return f"✅ Model loaded successfully!\n📍 Device: {self.device}\n📊 Training Dice: {dice:.4f if isinstance(dice, float) else dice}"
            
        except Exception as e:
            self.loaded = False
            return f"❌ Error loading model: {str(e)}"
    
    def predict(self, images: np.ndarray) -> np.ndarray:
        """Run prediction on preprocessed images."""
        if not self.loaded:
            raise RuntimeError("Model not loaded!")
        
        with torch.no_grad():
            # Convert to tensor: (C, H, W, D) -> (1, C, D, H, W)
            images_t = np.transpose(images, (0, 3, 1, 2))
            tensor = torch.from_numpy(images_t).float().unsqueeze(0).to(self.device)
            
            # Forward pass
            output, _ = self.model(tensor)
            
            # Get prediction
            pred = torch.argmax(F.softmax(output, dim=1), dim=1)
            pred_np = pred.squeeze(0).cpu().numpy()
            
            # Transpose back: (D, H, W) -> (H, W, D)
            pred_np = np.transpose(pred_np, (1, 2, 0))
            
            # Convert label 3 back to 4 for BraTS format
            pred_np[pred_np == 3] = 4
            
            return pred_np


# Global model instance
model = MRAFNetModel()


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
    # Normalize MRI slice to 0-255
    mri_norm = (mri_slice - mri_slice.min()) / (mri_slice.max() - mri_slice.min() + 1e-8)
    mri_rgb = np.stack([mri_norm] * 3, axis=-1)
    
    # Create overlay
    overlay = np.zeros((*seg_slice.shape, 4), dtype=np.float32)
    
    for label, color in CONFIG["colors"].items():
        if label > 0:
            mask = seg_slice == label
            overlay[mask] = np.array(color) / 255.0
    
    # Blend
    for c in range(3):
        mask = overlay[:, :, 3] > 0
        mri_rgb[:, :, c][mask] = (
            mri_rgb[:, :, c][mask] * (1 - alpha * overlay[:, :, 3][mask]) +
            overlay[:, :, c][mask] * alpha * overlay[:, :, 3][mask]
        )
    
    return (mri_rgb * 255).astype(np.uint8)


def compute_tumor_metrics(segmentation: np.ndarray, 
                         ground_truth: Optional[np.ndarray] = None,
                         voxel_volume: float = 1.0) -> Dict:
    """Compute tumor metrics."""
    metrics = {}
    
    # Volume statistics
    unique, counts = np.unique(segmentation, return_counts=True)
    vol_dict = dict(zip(unique.astype(int), counts))
    
    # Compute region volumes (in ml, assuming 1mm³ voxels)
    vol_ncr = vol_dict.get(1, 0) * voxel_volume / 1000
    vol_ed = vol_dict.get(2, 0) * voxel_volume / 1000
    vol_et = vol_dict.get(4, 0) * voxel_volume / 1000
    
    vol_wt = vol_ncr + vol_ed + vol_et  # Whole Tumor
    vol_tc = vol_ncr + vol_et            # Tumor Core
    
    metrics["volume"] = {
        "whole_tumor_ml": round(vol_wt, 2),
        "tumor_core_ml": round(vol_tc, 2),
        "enhancing_ml": round(vol_et, 2),
        "edema_ml": round(vol_ed, 2),
        "necrotic_ml": round(vol_ncr, 2)
    }
    
    # Dice scores if ground truth provided
    if ground_truth is not None:
        def dice_score(pred_mask, gt_mask):
            intersection = np.sum(pred_mask & gt_mask)
            union = np.sum(pred_mask) + np.sum(gt_mask)
            if union == 0:
                return 1.0 if intersection == 0 else 0.0
            return 2 * intersection / union
        
        # Whole Tumor (1 + 2 + 4)
        wt_pred = (segmentation == 1) | (segmentation == 2) | (segmentation == 4)
        wt_gt = (ground_truth == 1) | (ground_truth == 2) | (ground_truth == 4)
        
        # Tumor Core (1 + 4)
        tc_pred = (segmentation == 1) | (segmentation == 4)
        tc_gt = (ground_truth == 1) | (ground_truth == 4)
        
        # Enhancing Tumor (4)
        et_pred = segmentation == 4
        et_gt = ground_truth == 4
        
        metrics["dice"] = {
            "whole_tumor": round(dice_score(wt_pred, wt_gt), 4),
            "tumor_core": round(dice_score(tc_pred, tc_gt), 4),
            "enhancing": round(dice_score(et_pred, et_gt), 4),
            "mean": round((dice_score(wt_pred, wt_gt) + 
                          dice_score(tc_pred, tc_gt) + 
                          dice_score(et_pred, et_gt)) / 3, 4)
        }
    
    return metrics


# ============================================================================
# GRADIO INTERFACE FUNCTIONS
# ============================================================================

def load_model_handler(checkpoint_path: str) -> str:
    """Handle model loading."""
    if not checkpoint_path:
        return "⚠️ Please provide a checkpoint path"
    
    if not os.path.exists(checkpoint_path):
        return f"❌ File not found: {checkpoint_path}"
    
    return model.load(checkpoint_path)


def process_mri_files(flair_file, t1_file, t1ce_file, t2_file, gt_file=None):
    """Process uploaded MRI files."""
    if not all([flair_file, t1_file, t1ce_file, t2_file]):
        return None, None, None, "⚠️ Please upload all 4 MRI modalities"
    
    if not model.loaded:
        return None, None, None, "❌ Please load a model first"
    
    try:
        # Load files
        flair_data, affine = load_nifti(flair_file.name)
        t1_data, _ = load_nifti(t1_file.name)
        t1ce_data, _ = load_nifti(t1ce_file.name)
        t2_data, _ = load_nifti(t2_file.name)
        
        # Stack modalities
        images = np.stack([flair_data, t1_data, t1ce_data, t2_data], axis=0)
        
        # Normalize
        images_norm = normalize_intensity(images)
        
        # Run prediction
        segmentation = model.predict(images_norm)
        
        # Load ground truth if provided
        ground_truth = None
        if gt_file is not None:
            ground_truth, _ = load_nifti(gt_file.name)
        
        # Compute metrics
        voxel_vol = np.abs(np.linalg.det(affine[:3, :3]))
        metrics = compute_tumor_metrics(segmentation, ground_truth, voxel_vol)
        
        # Store for visualization
        global stored_data
        stored_data = {
            "flair": flair_data,
            "segmentation": segmentation,
            "ground_truth": ground_truth,
            "affine": affine,
            "metrics": metrics,
            "shape": flair_data.shape
        }
        
        # Create initial visualization
        mid_slice = flair_data.shape[2] // 2
        viz_image = create_overlay_slice(
            flair_data[:, :, mid_slice],
            segmentation[:, :, mid_slice]
        )
        
        # Format metrics
        metrics_text = format_metrics(metrics)
        
        return Image.fromarray(viz_image), mid_slice, flair_data.shape[2] - 1, metrics_text
        
    except Exception as e:
        import traceback
        return None, None, None, f"❌ Error: {str(e)}\n{traceback.format_exc()}"


def update_slice_view(slice_idx: int, view: str, show_overlay: bool, alpha: float):
    """Update slice visualization."""
    global stored_data
    
    if stored_data is None:
        return None
    
    flair = stored_data["flair"]
    seg = stored_data["segmentation"]
    
    # Get slice based on view
    if view == "Axial":
        mri_slice = flair[:, :, int(slice_idx)]
        seg_slice = seg[:, :, int(slice_idx)]
    elif view == "Coronal":
        mri_slice = flair[:, int(slice_idx), :]
        seg_slice = seg[:, int(slice_idx), :]
    else:  # Sagittal
        mri_slice = flair[int(slice_idx), :, :]
        seg_slice = seg[int(slice_idx), :, :]
    
    if show_overlay:
        viz_image = create_overlay_slice(mri_slice, seg_slice, alpha)
    else:
        mri_norm = (mri_slice - mri_slice.min()) / (mri_slice.max() - mri_slice.min() + 1e-8)
        viz_image = (np.stack([mri_norm] * 3, axis=-1) * 255).astype(np.uint8)
    
    return Image.fromarray(viz_image)


def get_max_slice(view: str) -> int:
    """Get maximum slice index for view."""
    global stored_data
    
    if stored_data is None:
        return 100
    
    shape = stored_data["shape"]
    if view == "Axial":
        return shape[2] - 1
    elif view == "Coronal":
        return shape[1] - 1
    else:
        return shape[0] - 1


def format_metrics(metrics: Dict) -> str:
    """Format metrics for display."""
    text = "## 📊 Segmentation Results\n\n"
    
    text += "### 📏 Tumor Volumes\n"
    text += f"| Region | Volume (ml) |\n"
    text += f"|--------|-------------|\n"
    text += f"| Whole Tumor (WT) | {metrics['volume']['whole_tumor_ml']:.2f} |\n"
    text += f"| Tumor Core (TC) | {metrics['volume']['tumor_core_ml']:.2f} |\n"
    text += f"| Enhancing (ET) | {metrics['volume']['enhancing_ml']:.2f} |\n"
    text += f"| Edema (ED) | {metrics['volume']['edema_ml']:.2f} |\n"
    text += f"| Necrotic (NCR) | {metrics['volume']['necrotic_ml']:.2f} |\n\n"
    
    if "dice" in metrics:
        text += "### 🎯 Dice Scores\n"
        text += f"| Region | Dice Score |\n"
        text += f"|--------|------------|\n"
        text += f"| Whole Tumor | {metrics['dice']['whole_tumor']:.4f} |\n"
        text += f"| Tumor Core | {metrics['dice']['tumor_core']:.4f} |\n"
        text += f"| Enhancing | {metrics['dice']['enhancing']:.4f} |\n"
        text += f"| **Mean** | **{metrics['dice']['mean']:.4f}** |\n"
    
    return text


def create_3d_plot():
    """Create 3D visualization plot."""
    global stored_data
    
    if stored_data is None:
        return None
    
    seg = stored_data["segmentation"]
    
    # Downsample for performance
    step = 3
    seg_down = seg[::step, ::step, ::step]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = {1: 'green', 2: 'yellow', 4: 'red'}
    labels = {1: 'NCR/NET', 2: 'Edema', 4: 'Enhancing'}
    
    for label, color in colors.items():
        mask = seg_down == label
        if np.any(mask):
            coords = np.argwhere(mask)
            if len(coords) > 2000:
                indices = np.random.choice(len(coords), 2000, replace=False)
                coords = coords[indices]
            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                      c=color, alpha=0.3, s=1, label=labels[label])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('3D Tumor Visualization')
    
    plt.tight_layout()
    return fig


def export_segmentation():
    """Export segmentation as NIfTI."""
    global stored_data
    
    if stored_data is None:
        return None, "⚠️ No segmentation to export"
    
    # Create temp file
    output_path = tempfile.mktemp(suffix='.nii.gz')
    
    nii = nib.Nifti1Image(
        stored_data["segmentation"].astype(np.int16),
        stored_data["affine"]
    )
    nib.save(nii, output_path)
    
    return output_path, f"✅ Saved to: {output_path}"


# Global storage
stored_data = None


# ============================================================================
# GRADIO UI DEFINITION
# ============================================================================

def create_interface():
    """Create the Gradio interface."""
    
    # Custom CSS
    css = """
    .gradio-container {
        max-width: 1400px !important;
    }
    .header-text {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .metric-box {
        background: #f8fafc;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    .legend-box {
        background: #f1f5f9;
        padding: 10px;
        border-radius: 8px;
    }
    """
    
    with gr.Blocks(css=css, title="MRAF-Net Brain Tumor Segmentation") as demo:
        # Header
        gr.HTML("""
        <div class="header-text">
            <h1>🧠 MRAF-Net</h1>
            <h3>Multi-Resolution Aligned and Robust Fusion Network for Brain Tumor Segmentation</h3>
            <p>BEng Software Engineering | University of Westminster | IIT</p>
            <p>Anne Nidhusha Nithiyalan (w1985740) | Supervisor: Ms. Mohanadas Jananie</p>
        </div>
        """)
        
        with gr.Tabs():
            # Tab 1: Segmentation
            with gr.TabItem("🔬 Segmentation"):
                with gr.Row():
                    # Left column - Inputs
                    with gr.Column(scale=1):
                        gr.Markdown("### 📁 Load Model")
                        checkpoint_input = gr.Textbox(
                            label="Checkpoint Path",
                            value=CONFIG["model_path"],
                            placeholder="Path to model checkpoint"
                        )
                        load_btn = gr.Button("🚀 Load Model", variant="primary")
                        model_status = gr.Textbox(label="Model Status", interactive=False)
                        
                        gr.Markdown("### 📤 Upload MRI Scans")
                        flair_input = gr.File(label="FLAIR", file_types=[".nii", ".nii.gz"])
                        t1_input = gr.File(label="T1", file_types=[".nii", ".nii.gz"])
                        t1ce_input = gr.File(label="T1ce (Contrast)", file_types=[".nii", ".nii.gz"])
                        t2_input = gr.File(label="T2", file_types=[".nii", ".nii.gz"])
                        gt_input = gr.File(label="Ground Truth (Optional)", file_types=[".nii", ".nii.gz"])
                        
                        run_btn = gr.Button("🧠 Run Segmentation", variant="primary", size="lg")
                    
                    # Right column - Visualization
                    with gr.Column(scale=2):
                        gr.Markdown("### 🖼️ Visualization")
                        
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
                        
                        # Legend
                        gr.HTML("""
                        <div class="legend-box">
                            <b>Legend:</b>
                            <span style="color: green;">■ NCR/NET (Necrotic Core)</span> |
                            <span style="color: #DAA520;">■ Edema (Peritumoral)</span> |
                            <span style="color: red;">■ ET (Enhancing Tumor)</span>
                        </div>
                        """)
                
                # Metrics section
                with gr.Row():
                    metrics_output = gr.Markdown("### Upload MRI scans and run segmentation to see metrics")
            
            # Tab 2: 3D Visualization
            with gr.TabItem("🎮 3D View"):
                gr.Markdown("### 3D Tumor Visualization")
                plot_btn = gr.Button("Generate 3D Plot")
                plot_output = gr.Plot(label="3D Visualization")
            
            # Tab 3: Export
            with gr.TabItem("💾 Export"):
                gr.Markdown("### Export Results")
                export_btn = gr.Button("Export Segmentation (NIfTI)")
                export_file = gr.File(label="Download")
                export_status = gr.Textbox(label="Status", interactive=False)
            
            # Tab 4: About
            with gr.TabItem("📖 About"):
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
                
                ### Metrics
                - **Dice Score**: Measures overlap between prediction and ground truth (0-1, higher is better)
                - **Hausdorff Distance 95%**: Measures boundary accuracy in mm (lower is better)
                
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
        <div style="text-align: center; padding: 20px; color: #666; border-top: 1px solid #eee; margin-top: 20px;">
            <p>MRAF-Net Brain Tumor Segmentation System | © 2026 Anne Nidhusha Nithiyalan</p>
            <p>University of Westminster | Informatics Institute of Technology</p>
        </div>
        """)
        
        # Event handlers
        load_btn.click(
            fn=load_model_handler,
            inputs=[checkpoint_input],
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
        
        plot_btn.click(
            fn=create_3d_plot,
            outputs=[plot_output]
        )
        
        export_btn.click(
            fn=export_segmentation,
            outputs=[export_file, export_status]
        )
    
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
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ CUDA not available, using CPU")
    
    # Create and launch interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )
