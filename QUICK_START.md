# MRAF-Net Quick Start Guide
## Step-by-Step Instructions for Training on Your Local Laptop

---

## 🚀 QUICK START (5 MINUTES)

### Step 1: Create Project Folder
Open Command Prompt (Windows) or Terminal (Mac/Linux):
```bash
mkdir mraf_net
cd mraf_net
```

### Step 2: Copy All Files
Copy all the provided files into the `mraf_net` folder maintaining the folder structure.

### Step 3: Create Virtual Environment

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 4: Install PyTorch with CUDA

**Windows/Linux with NVIDIA GPU:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Mac (CPU only):**
```bash
pip install torch torchvision torchaudio
```

### Step 5: Install Other Dependencies
```bash
pip install -r requirements.txt
```

### Step 6: Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 📥 DOWNLOAD BRATS 2020 DATASET

1. **Register:** Go to https://www.med.upenn.edu/cbica/brats2020/registration.html
2. **Download:** Get `MICCAI_BraTS2020_TrainingData.zip`
3. **Extract:** Extract to `data/brats2020/MICCAI_BraTS2020_TrainingData/`

Your structure should look like:
```
mraf_net/
└── data/
    └── brats2020/
        └── MICCAI_BraTS2020_TrainingData/
            ├── BraTS20_Training_001/
            ├── BraTS20_Training_002/
            └── ...
```

---

## ✅ VERIFY DATA

```bash
python scripts/prepare_data.py --data_dir data/brats2020/MICCAI_BraTS2020_TrainingData
```

---

## 🏋️ START TRAINING

### For Laptops (8GB GPU or less):
```bash
python scripts/train.py --config config/config.yaml --mode laptop
```

### For Better GPUs (12GB+):
```bash
python scripts/train.py --config config/config.yaml --mode standard
```

---

## 📊 MONITOR TRAINING

Open another terminal and run:
```bash
tensorboard --logdir experiments
```
Then open http://localhost:6006 in your browser.

---

## 🔍 EVALUATE MODEL

After training:
```bash
python scripts/evaluate.py --checkpoint experiments/[YOUR_EXPERIMENT]/checkpoints/best_model.pth
```

---

## 🎯 PREDICT ON NEW DATA

```bash
python scripts/predict.py --checkpoint checkpoints/best_model.pth --input path/to/case_folder
```

---

## ⚠️ TROUBLESHOOTING

### Out of Memory Error
Edit `config/config.yaml`:
```yaml
training:
  batch_size: 1
  patch_size: [64, 64, 64]  # Reduce from [96, 96, 96]
  use_amp: true
  gradient_checkpointing: true
```

### CUDA Not Detected
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Slow Training
- Enable mixed precision: `use_amp: true`
- Use SSD for data storage
- Reduce `num_workers` if CPU-bound

---

## 📁 FOLDER STRUCTURE

```
mraf_net/
├── config/
│   └── config.yaml           # Configuration file
├── data/
│   └── brats2020/            # Dataset here
├── src/
│   ├── models/               # Neural network code
│   ├── data/                 # Data loading code
│   ├── losses/               # Loss functions
│   └── utils/                # Utilities
├── scripts/
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   ├── predict.py            # Inference script
│   └── prepare_data.py       # Data preparation
├── checkpoints/              # Saved models
├── logs/                     # Training logs
├── outputs/                  # Predictions
├── experiments/              # Experiment outputs
├── requirements.txt          # Dependencies
├── setup.bat                 # Windows setup
├── setup.sh                  # Linux/Mac setup
└── README.md                 # Full documentation
```

---

## 📈 EXPECTED RESULTS

After training for 100-300 epochs:
| Metric | Expected Range |
|--------|---------------|
| Dice WT | 0.88 - 0.91 |
| Dice TC | 0.82 - 0.86 |
| Dice ET | 0.75 - 0.80 |
| Mean Dice | 0.82 - 0.86 |

---

## 💡 TIPS FOR BEST RESULTS

1. **Train longer:** More epochs = better results
2. **Use all data:** Don't reduce training set size
3. **Enable augmentation:** Already enabled by default
4. **Monitor validation:** Check for overfitting
5. **Use best checkpoint:** `best_model.pth` is saved automatically

---

## 📞 NEED HELP?

1. Check the full `README.md` for detailed information
2. Review error messages carefully
3. Ensure all dependencies are installed correctly
4. Verify dataset structure matches expected format

Good luck with your training! 🎉
