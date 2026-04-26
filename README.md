# MRAF-Net: Multi-Resolution Aligned and Robust Fusion Network for Brain Tumor Segmentation

## 📌 Project Information
- **Author**: Anne Nidhusha Nithiyalan (w1985740)
- **Supervisor**: Ms. Mohanadas Jananie
- **Institution**: Informatics Institute of Technology / University of Westminster
- **Programme**: BEng (Hons) Software Engineering
- **Year**: 2026

---

## 📖 Overview
MRAF-Net is an advanced, highly specialized deep learning architecture designed for automatic brain tumor segmentation from multi-modal MRI scans. 

Brain tumors are highly heterogeneous—varying drastically in shape, spatial location, and anatomical volume. MRAF-Net leverages a 3D U-Net backbone but introduces substantial architectural upgrades (ASPP, Attention Gates, Cross-Modality Fusion) to accurately identify and segment localized tumor sub-regions without needing rigid localization bounding boxes.

---

## 🧠 Dataset & Labeling

### Dataset Origins
The model utilizes the **BraTS 2020 (Brain Tumor Segmentation Challenge)** dataset. It consists of 3D Multimodal Magnetic Resonance Imaging (MRI) scans.
The network fuses 4 distinct MRI sequences to comprehensively profile the tumor:
- **T1** (T1-weighted)
- **T1ce** (T1-weighted contrast-enhanced) - *Best for the active core*
- **T2** (T2-weighted)
- **FLAIR** (Fluid-Attenuated Inversion Recovery) - *Best for peritumoral edema*

### Clinical Labeling
The task is a **Semantic Segmentation** problem. The model outputs a voxel-level 3D mask where every pixel is classified into one of four categories:
1. **Class 0 (Background):** Healthy Brain Tissue / Empty Space
2. **Class 1 (NCR/NET):** Necrotic and Non-enhancing Core (dead tissue inside the tumor)
3. **Class 2 (ED):** Peritumoral Edema (swelling around the tumor)
4. **Class 4 / 3 (ET):** GD-Enhancing Tumor (highly active, growing boundaries)

---

## 🔬 Model Architecture

MRAF-Net relies on state-of-the-art methodology tailored specifically for 3D medical images:

1. **Hybrid Optimization:** Trained using a hybrid `dice_ce` loss function. It perfectly balances spatial overlap evaluation (Dice Loss) with voxel-wise precision (Cross-Entropy Loss) to mitigate severe class imbalance.
2. **Multi-Resolution Harmonization:** A deep encoder extracts hierarchical features across 5 resolution levels (scaling from 32 to 320 channels).
3. **Cross-Modality Fusion:** Iteratively merges the 4 MRI inputs early in the encoder to leverage the structural strengths of each sequence.
4. **ASPP (Atrous Spatial Pyramid Pooling):** Located at the bottleneck with dilation rates of `[6, 12, 18]`. This handles massive variances in tumor size by expanding the network's receptive field organically.
5. **Edge-Aware Attention Gates:** Positioned in the decoder, these drastically suppress background noise (healthy tissue) and enforce gradient updates exclusively on blurry, ambiguous tumor boundaries.
6. **Deep Supervision:** Auxiliary outputs inject gradient signals directly into the intermediate decoder stages, dramatically accelerating network convergence.

---

## 🎨 Visualization & GUI

MRAF-Net comes with a standalone **PyQt5 & Gradio Graphical User Interface (`gui/app.py`)** built for researchers and clinicians to visualize predictions dynamically.

**What do the colors mean?**
The predictions natively map the tumor regions to the following clinical color overlays:
- 🟩 **<span style="color:green">Green</span> (Label 1):** Necrotic Core (NCR/NET) — Dead/inactive tissue at the center of the tumor.
- 🟨 **<span style="color:yellow">Yellow</span> (Label 2):** Peritumoral Edema (ED) — Swelling and fluid accumulation surrounding the tumor.
- 🟥 **<span style="color:red">Red</span> (Label 4):** Enhancing Tumor (ET) — Highly active, fast-growing, and aggressive tissue boundaries.

---

## 💻 System Requirements

### Hardware (Minimum)
- **CPU**: Intel Core i5 or equivalent
- **RAM**: 16 GB (32 GB highly recommended)
- **GPU**: NVIDIA GPU with 6GB+ VRAM (8GB+ recommended)
- **Storage**: 50 GB free space (SSD strongly recommended)

---

## ⚙️ Installation & Setup

### 1. Clone/Create Project Directory
```bash
mkdir mraf_net
cd mraf_net
```

### 2. Create Virtual Environment
**Windows (Command Prompt as Administrator):**
```cmd
python -m venv venv
venv\Scripts\activate
```
**Linux / Mac / PowerShell:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download and Prepare the Dataset
1. Register at: https://www.med.upenn.edu/cbica/brats2020/registration.html
2. Download the Training Data and extract it to: `data/brats2020/MICCAI_BraTS2020_TrainingData/`
3. Run the data preparation script:
```bash
python scripts/prepare_data.py --data_dir data/brats2020/MICCAI_BraTS2020_TrainingData
```

---

## 🚀 Running the Pipeline

All scripts reside under `scripts/` and accept `--help` for comprehensive options.

### 1. Training the Model
We extensively use **Automatic Mixed Precision (AMP)** and **Gradient Checkpointing** to enable memory-efficient training on consumer laptops.
```bash
# Standard mode (uses settings from config/config.yaml)
python scripts/train.py --config config/config.yaml

# Laptop mode (memory-optimized: batch_size=1, patches [64,64,64])
python scripts/train.py --config config/config.yaml --mode laptop
```
*Checkpoints and logs are automatically dumped into the `experiments/` directory.*

### 2. Evaluating a Checkpoint
```bash
python scripts/evaluate.py \
    --checkpoint experiments/<exp_name>/checkpoints/best_model.pth
```

### 3. Running Inference / Prediction
Runs sliding-window inference with Gaussian blending to generate smooth segmentations of entire scans.
```bash
python scripts/predict.py \
    --checkpoint experiments/<exp_name>/checkpoints/best_model.pth \
    --input path/to/case_directory \
    --output predictions/
```
*(The `--input` directory must contain the four modalities named: `flair.nii.gz`, `t1.nii.gz`, `t1ce.nii.gz`, `t2.nii.gz`)*

### 4. Starting the Interface (GUI)
```bash
cd gui
python app.py          # Launch development GUI
```
*The GUI runs on http://localhost:7860.*

---

## 📁 Project Structure
```text
mraf_net/
├── config/config.yaml       # Master training configuration
├── data/                    # Dataset directory containing BraTS files
├── src/                     # Core Deep Learning Source Code
│   ├── models/              # Neural networks (MRAF-Net, U-Net)
│   ├── data/                # Data loaders, normalization, and patch extraction
│   ├── losses/              # Loss functions (Dice, Cross-Entropy)
│   └── utils/               # Helper utilities & Metric computations
├── scripts/                 # Entry points (train.py, evaluate.py, predict.py)
├── gui/                     # Front-end Interface code (Gradio/PyQt5)
├── checkpoints/             # Fallback/Saved weights
├── experiments/             # Live training outputs, models, and tensorboard logs
└── outputs/                 # Inference NIfTI predictions and visualization PNGs
```

---

## 📈 Expected Results

Upon completion of 100-300 epochs, MRAF-Net achieves robust generalization metrics against clinical validation targets:

| Target | Expected Dice Score |
| :--- | :--- |
| **WT (Whole Tumor)** | `0.88 - 0.91` |
| **TC (Tumor Core)** | `0.82 - 0.86` |
| **ET (Enhancing Tumor)** | `0.75 - 0.80` |
| **Mean Dice** | `0.82 - 0.86` |

---

## 🛠️ Troubleshooting

- **CUDA Out of Memory:**
  Edit `config/config.yaml`. Explicitly set `batch_size: 1`, reduce `patch_size` to `[64, 64, 64]`, and ensure `use_amp: true` and `gradient_checkpointing: true` are enabled.
  
- **CUDA Not Detected:**
  Check NVIDIA drivers (`nvidia-smi`), completely uninstall PyTorch, and reinstall strictly with the correct index URL: 
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
  
- **Slow Training Pipeline:**
  Data loading from HDDs can notoriously bottleneck 3D deep learning. Ensure `data_dir` resides on an SSD. Consider increasing `num_workers: 4` if your CPU permits.

---

## 📄 Citation
If utilizing this codebase for academic endeavors, please cite:
```bibtex
@thesis{nithiyalan2026mrafnet,
  author = {Anne Nidhusha Nithiyalan},
  title = {MRAF-Net: Multi-Resolution Aligned and Robust Fusion Network for Brain Tumor Segmentation},
  school = {Informatics Institute of Technology / University of Westminster},
  year = {2026}
}
```

## 📜 License
This project is strictly designated for academic and research purposes only.
