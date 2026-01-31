# BraTS 2020 Dataset Download Instructions

## Overview
The BraTS (Brain Tumor Segmentation) 2020 dataset contains multimodal MRI scans with expert annotations for brain tumor segmentation.

## Dataset Statistics
- **Training Data**: 369 cases
- **Validation Data**: 125 cases
- **MRI Modalities**: T1, T1ce (contrast-enhanced), T2, FLAIR
- **Resolution**: 240 × 240 × 155 voxels
- **Voxel Size**: 1mm × 1mm × 1mm (isotropic)

## Download Steps

### Step 1: Register
1. Go to: https://www.med.upenn.edu/cbica/brats2020/registration.html
2. Fill out the registration form
3. Accept the data usage agreement
4. You will receive download links via email

### Step 2: Download
Download the following files:
- `MICCAI_BraTS2020_TrainingData.zip` (~5.5 GB)
- `MICCAI_BraTS2020_ValidationData.zip` (~1.8 GB)

### Step 3: Extract
Extract the zip files to this directory:

```
data/
└── brats2020/
    ├── MICCAI_BraTS2020_TrainingData/
    │   ├── BraTS20_Training_001/
    │   │   ├── BraTS20_Training_001_flair.nii.gz
    │   │   ├── BraTS20_Training_001_t1.nii.gz
    │   │   ├── BraTS20_Training_001_t1ce.nii.gz
    │   │   ├── BraTS20_Training_001_t2.nii.gz
    │   │   └── BraTS20_Training_001_seg.nii.gz
    │   ├── BraTS20_Training_002/
    │   │   └── ...
    │   └── ...
    └── MICCAI_BraTS2020_ValidationData/
        └── ...
```

### Step 4: Verify
Run the data preparation script:
```bash
python scripts/prepare_data.py --data_dir data/brats2020/MICCAI_BraTS2020_TrainingData
```

## Segmentation Labels
The segmentation masks contain the following labels:
- **0**: Background
- **1**: Necrotic and Non-Enhancing Tumor Core (NCR/NET)
- **2**: Peritumoral Edema (ED)
- **4**: GD-Enhancing Tumor (ET)

Note: Label 3 is not used in BraTS.

## Evaluation Regions
Models are evaluated on three tumor regions:
1. **Whole Tumor (WT)**: Labels 1 + 2 + 4
2. **Tumor Core (TC)**: Labels 1 + 4
3. **Enhancing Tumor (ET)**: Label 4

## Alternative: Kaggle Dataset
If you have trouble accessing the official dataset, you can use the Kaggle version:
https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation

## File Formats
All MRI volumes are stored in NIfTI format (.nii.gz):
- Compressed gzip format
- Can be loaded with nibabel or SimpleITK libraries

## Citation
```
@article{menze2014multimodal,
  title={The multimodal brain tumor image segmentation benchmark (BRATS)},
  author={Menze, Bjoern H and others},
  journal={IEEE transactions on medical imaging},
  year={2014}
}
```
