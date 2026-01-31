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

## Quick Start Guide

### Step 1: Clone/Create Project Directory
```bash
# Create project folder
mkdir mraf_net
cd mraf_net
```

### Step 2: Create Virtual Environment

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

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify CUDA Installation
```python
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

### Step 5: Download BraTS 2020 Dataset
1. Register at: https://www.med.upenn.edu/cbica/brats2020/registration.html
2. Download the Training Data
3. Extract to: `data/brats2020/MICCAI_BraTS2020_TrainingData/`

### Step 6: Prepare Data
```bash
python scripts/prepare_data.py --data_dir data/brats2020/MICCAI_BraTS2020_TrainingData
```

### Step 7: Start Training
```bash
# For laptop with limited GPU (6-8GB VRAM)
python scripts/train.py --config config/config.yaml --mode laptop

# For better GPU (12GB+ VRAM)
python scripts/train.py --config config/config.yaml --mode standard
```

### Step 8: Evaluate Model
```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth
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
