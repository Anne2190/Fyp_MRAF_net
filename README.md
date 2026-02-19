# MRAF-Net: Multi-Resolution Aligned and Robust Fusion Network for Brain Tumor Segmentation

## Project Information
- **Author**: Anne Nidhusha Nithiyalan (w1985740)
- **Supervisor**: Ms. Mohanadas Jananie
- **Institution**: Informatics Institute of Technology / University of Westminster

## Overview
MRAF-Net is a deep learning architecture for automatic brain tumor segmentation from multimodal MRI scans (T1, T1ce, T2, FLAIR). The model addresses:
1. Multi-resolution harmonization for standardizing MRI inputs
2. Cross-modality feature alignment and fusion
3. Edge-aware attention for precise tumor boundary detection

## System Requirements

### Hardware (Minimum)
- **CPU**: Intel Core i5 or better
- **RAM**: 16 GB (32 GB recommended)
- **GPU**: NVIDIA GPU with 6GB+ VRAM (8GB+ recommended)
- **Storage**: 50 GB free space

### Hardware (Your Setup - From Proposal)
- **CPU**: Intel Core i7
- **RAM**: 32 GB
- **Storage**: 1 TB SSD
- **OS**: Windows 10

## Installation & Setup

### 1. Clone/Create Project Directory
```bash
# Create project folder
mkdir mraf_net
cd mraf_net
```

### 2. Create Virtual Environment

#### Windows (Command Prompt as Administrator):
```cmd
python -m venv venv
venv\Scripts\activate
```

#### Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### Linux/Mac:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify CUDA Installation (optional)
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

### 5. Download and Prepare the Dataset
1. Register at: https://www.med.upenn.edu/cbica/brats2020/registration.html
2. Download the Training Data and extract it to:
   `data/brats2020/MICCAI_BraTS2020_TrainingData/`

3. Run the preparation script:
```bash
python scripts/prepare_data.py \
    --data_dir data/brats2020/MICCAI_BraTS2020_TrainingData
```

---

## Running the Application

The repository provides several command‑line entry points. All scripts live under `scripts/` and accept `--help` for a full list of options.

### 1. Training the Model
```bash
# standard mode (uses settings from config/config.yaml)
python scripts/train.py --config config/config.yaml

# laptop mode (smaller batch/patches, mixed‑precision etc.)
python scripts/train.py --config config/config.yaml --mode laptop
```

**Notes:**
- `--mode` is optional; default is `standard`.
- Training results (checkpoints, logs) are saved to a new folder under `experiments/`.
- You can resume from a checkpoint by editing `config/config.yaml` or using the `checkpoint.resume` field.

### 2. Evaluating a Checkpoint
```bash
python scripts/evaluate.py \
    --checkpoint experiments/<exp_name>/checkpoints/best_model.pth
```

Add `--data_dir <path>` or `--output <path>` to override defaults. Use `python scripts/evaluate.py --help` for details.

### 3. Running Inference / Prediction
```bash
python scripts/predict.py \
    --checkpoint experiments/<exp_name>/checkpoints/best_model.pth \
    --input path/to/case_directory \
    --output predictions/
```

- The `--input` directory should contain the four modalities (FLAIR, T1, T1ce, T2) named either `<case>_flair.nii.gz` (or `.nii`) or simply `flair.nii.gz`, etc.
- Predictions are saved as NIfTI files in the `--output` folder.

### 4. Starting the GUI (optional)
A simple PyQt5 interface is provided in the `gui/` folder. After installing the packages listed in `gui/requirements.txt`:
```bash
cd gui
python app.py          # launch development GUI
# or
python standalone_gui.py   # run the bundled single‑file version
```

The GUI allows you to select a checkpoint and case directory from a file dialog and displays the segmentation overlay.

### 5. Other Utilities
- `scripts/prepare_data.py` – preprocess the dataset
- `scripts/test_all.py` – run the unit/integration tests

Each script supports `--help` for command‑line arguments.

---

## Project Structure
```

## Project Structure
```
mraf_net/
├── config/config.yaml       # Training configuration
├── data/                    # Dataset directory
├── src/                     # Source code
│   ├── models/              # Neural network architectures
│   ├── data/                # Data loading and preprocessing
│   ├── losses/              # Loss functions
│   └── utils/               # Helper utilities
├── scripts/                 # Training and evaluation scripts
├── checkpoints/             # Saved models
├── logs/                    # Training logs
└── outputs/                 # Results and visualizations
```

## Training Tips for Local Laptop

### Memory Optimization
1. **Reduce batch size**: Start with batch_size=1
2. **Smaller patch size**: Use 96x96x96 instead of 128x128x128
3. **Enable gradient checkpointing**: Saves memory at cost of speed
4. **Mixed precision training**: Uses FP16 for faster training

### If You Get CUDA Out of Memory Error:
```yaml
# In config/config.yaml, change:
batch_size: 1
patch_size: [96, 96, 96]
use_amp: true
gradient_checkpointing: true
```

## Expected Results
After training for 100-300 epochs:
- **Dice Score (WT)**: ~0.88-0.91
- **Dice Score (TC)**: ~0.82-0.86
- **Dice Score (ET)**: ~0.75-0.80
- **Mean Dice**: ~0.82-0.86

## Troubleshooting

### Issue: CUDA not detected
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Out of Memory
- Reduce `batch_size` to 1
- Reduce `patch_size` to [64, 64, 64]
- Enable `gradient_checkpointing: true`

### Issue: Slow Training
- Enable `use_amp: true` for mixed precision
- Increase `num_workers` in dataloader
- Use SSD storage for data

## Citation
If you use this code, please cite:
```
@thesis{nithiyalan2025mrafnet,
  author = {Anne Nidhusha Nithiyalan},
  title = {MRAF-Net: Multi-Resolution Aligned and Robust Fusion Network for Brain Tumor Segmentation},
  school = {Informatics Institute of Technology / University of Westminster},
  year = {2025}
}
```

## License
This project is for academic purposes only.
