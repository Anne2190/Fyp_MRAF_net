#!/bin/bash
echo "========================================"
echo "MRAF-Net Setup Script for Linux/Mac"
echo "========================================"
echo ""

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 is not installed"
    echo "Please install Python 3.10+"
    exit 1
fi

echo "Step 1: Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create virtual environment"
    exit 1
fi

echo "Step 2: Activating virtual environment..."
source venv/bin/activate

echo "Step 3: Upgrading pip..."
pip install --upgrade pip

echo "Step 4: Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Step 5: Installing other requirements..."
pip install -r requirements.txt

echo "Step 6: Creating necessary directories..."
mkdir -p data/brats2020
mkdir -p checkpoints
mkdir -p logs
mkdir -p outputs

echo "Step 7: Verifying installation..."
python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Download BraTS 2020 dataset"
echo "2. Extract to data/brats2020/MICCAI_BraTS2020_TrainingData/"
echo "3. Run: python scripts/prepare_data.py"
echo "4. Run: python scripts/train.py --config config/config.yaml"
