# 🧠 MRAF-Net Brain Tumor Segmentation GUI

**Multi-Resolution Aligned and Robust Fusion Network for Brain Tumor Segmentation**

A professional GUI application for brain tumor segmentation research demonstration.

---

## 📋 Overview

This GUI application provides an interactive interface for the MRAF-Net deep learning model, enabling:

- 🔬 **Automatic brain tumor segmentation** from multi-modal MRI scans
- 🖼️ **Interactive visualization** with slice navigation and overlay controls
- 📊 **Metrics computation** including Dice scores and tumor volumes
- 🎮 **3D tumor visualization** 
- 💾 **Export functionality** for segmentation results

---

## 🚀 Quick Start

### Option 1: Windows (Recommended)

1. **Copy the `gui` folder** to your MRAF-Net project directory
2. **Double-click `run_gui.bat`**
3. **Open browser** at http://localhost:7860

### Option 2: Manual Setup

```bash
# Install dependencies
pip install gradio nibabel torch numpy matplotlib pillow

# Run the standalone GUI (no src module needed)
python standalone_gui.py

# OR run the full GUI (requires src module)
python app.py
```

---

## 📦 Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU
- Trained MRAF-Net model checkpoint

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Directory Structure

```
mraf_net_fixed/
├── gui/
│   ├── standalone_gui.py    # ⭐ Self-contained GUI (recommended)
│   ├── app.py               # Full GUI (needs src module)
│   ├── requirements.txt     # Dependencies
│   └── run_gui.bat          # Windows launcher
├── experiments/
│   └── mraf_net_XXXXXX/
│       └── checkpoints/
│           └── best_model.pth  # Your trained model
├── src/                     # Model source code
└── ...
```

---

## 🖥️ Usage Guide

### 1️⃣ Load Model

1. Enter the path to your trained model checkpoint
2. Click **"Load Model"**
3. Wait for confirmation message

**Default path:** `experiments/mraf_net_20260124_130245/checkpoints/best_model.pth`

### 2️⃣ Upload MRI Scans

Upload all 4 required MRI modalities:
- **FLAIR** (Fluid Attenuated Inversion Recovery)
- **T1** (T1-weighted)
- **T1ce** (T1-weighted with contrast)
- **T2** (T2-weighted)
- **Ground Truth** (optional, for Dice score computation)

**Supported formats:** `.nii`, `.nii.gz`

### 3️⃣ Run Segmentation

Click **"🧠 Run Segmentation"** to process the MRI scans.

### 4️⃣ Explore Results

- **Slice Navigation:** Use the slider to browse through slices
- **View Selection:** Choose Axial, Coronal, or Sagittal views
- **Overlay Toggle:** Show/hide segmentation overlay
- **Opacity Control:** Adjust overlay transparency

### 5️⃣ Export Results

- Click **"Export Segmentation"** to save as NIfTI file
- Download the segmentation for further analysis

---

## 🎨 Tumor Region Legend

| Color | Label | Region | Description |
|-------|-------|--------|-------------|
| 🟢 Green | 1 | NCR/NET | Necrotic and Non-Enhancing Tumor Core |
| 🟡 Yellow | 2 | ED | Peritumoral Edema |
| 🔴 Red | 4 | ET | GD-Enhancing Tumor |

### Evaluation Regions

| Region | Labels | Description |
|--------|--------|-------------|
| **Whole Tumor (WT)** | 1 + 2 + 4 | Complete tumor extent |
| **Tumor Core (TC)** | 1 + 4 | Core tumor region |
| **Enhancing Tumor (ET)** | 4 | Active tumor enhancement |

---

## 📊 Metrics

### Dice Similarity Coefficient

Measures the overlap between prediction and ground truth:

```
Dice = 2 × |P ∩ G| / (|P| + |G|)
```

- **Range:** 0 to 1 (higher is better)
- **Perfect:** 1.0

### Tumor Volume

Computed in milliliters (ml) based on voxel dimensions.

---

## 🔧 Troubleshooting

### "CUDA out of memory"

- Use CPU mode by setting `device = "cpu"` in the code
- Or reduce input image size

### "Model not found"

- Verify the checkpoint path is correct
- Ensure the file exists and is readable

### "File format not supported"

- Only `.nii` and `.nii.gz` files are supported
- Convert other formats using nibabel

### GUI not opening

- Check if port 7860 is available
- Try: `python standalone_gui.py --port 7861`

---

## 📁 File Descriptions

| File | Description |
|------|-------------|
| `standalone_gui.py` | Self-contained GUI with embedded model architecture |
| `app.py` | Full GUI requiring src module |
| `requirements.txt` | Python dependencies |
| `run_gui.bat` | Windows batch launcher |

---

## 🎓 Research Information

**Project:** MRAF-Net: Multi-Resolution Aligned and Robust Fusion Network for Brain Tumor Segmentation

**Author:** Anne Nidhusha Nithiyalan (w1985740)

**Supervisor:** Ms. Mohanadas Jananie

**Institution:** University of Westminster / Informatics Institute of Technology

**Programme:** BEng (Hons) Software Engineering

**Year:** 2026

---

## 📚 Citation

```bibtex
@thesis{nithiyalan2026mrafnet,
    title={MRAF-Net: Multi-Resolution Aligned and Robust Fusion Network 
           for Brain Tumor Segmentation},
    author={Nithiyalan, Anne Nidhusha},
    year={2026},
    school={University of Westminster / Informatics Institute of Technology},
    type={BEng Dissertation},
    supervisor={Mohanadas, Jananie}
}
```

---

## 📞 Support

For issues or questions:
- Check the troubleshooting section above
- Review the model training logs in `experiments/`
- Verify all MRI files are properly formatted

---

## ⚖️ License

This software is developed for educational and research purposes as part of the BEng Software Engineering dissertation at the University of Westminster.

---

**© 2026 Anne Nidhusha Nithiyalan | University of Westminster | IIT**
