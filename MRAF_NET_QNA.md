# MRAF-Net: Comprehensive Project & Architecture Details

This document provides a detailed overview of the MRAF-Net (Multi-Resolution Aligned and Robust Fusion Network) project, covering everything from the dataset, preprocessing, deep learning architecture, training methodology, and accuracy.

## Table of Contents
1. [Dataset Details & Type](#1-dataset-details--type)
2. [Dataset Preprocessing & Augmentation](#2-dataset-preprocessing--augmentation)
3. [Model Architecture (MRAF-Net)](#3-model-architecture-mraf-net)
4. [Training Details & Methodology](#4-training-details--methodology)
5. [Performance & Accuracy Metrics](#5-performance--accuracy-metrics)
6. [Visualizations and Tumor Properties](#6-visualizations-and-tumor-properties)

---

## 1. Dataset Details & Type

### Dataset Origins & Type
- **Dataset:** BraTS 2020 (Brain Tumor Segmentation Challenge).
- **Type:** 3D Multimodal Magnetic Resonance Imaging (MRI) scans.
- **Modalities Used:** 4 distinct sequences per case: 
  - **T1** (T1-weighted)
  - **T1ce** (T1-weighted contrast-enhanced)
  - **T2** (T2-weighted)
  - **FLAIR** (Fluid-Attenuated Inversion Recovery)

### Labeling & Classifications
The task is a **Semantic Segmentation** problem. Medical experts have provided voxel-level manual annotations (assigning a class to every 3D pixel).
- **Class 0:** Background (Healthy Brain Tissue / Empty Space)
- **Class 1:** Necrotic and Non-enhancing Core (NCR/NET - dead tissue inside the tumor)
- **Class 2:** Peritumoral Edema (ED - swelling around the tumor)
- **Class 3:** GD-Enhancing Tumor (ET - the highly active, growing boundaries of the tumor)

---

## 2. Dataset Preprocessing & Augmentation

Handling 3D medical data requires rigorous preprocessing to ensure the model learns optimally.

### Preprocessing Pipeline
- **Verification:** The `prepare_data.py` script rigorously checks each case for corrupted files, missing modalities, and shape mismatches.
- **Normalization:** Intensity statistics (mean and standard deviation) are computed independently per modality to standardize the drastically varying contrast levels found in MRI machinery.
- **Patch Extraction:** Instead of feeding massive, full 3D brain volumes into the model, the data is dynamically cropped into configurable 3D patches (e.g., `96x96x96` or `128x128x128`). This prevents memory crashes and helps the model learn localized textures.

### Data Augmentation (On-the-fly)
To prevent overfitting, heavy spatial and intensity augmentations are applied during training:
- **Spatial:** Random Flips (50% prob) and Random 90-degree Rotations (50% prob).
- **Intensity Shift & Scale:** Random scaling and shifting of voxel intensities (range [-0.1, 0.1]).
- **Noise injection:** Random Gaussian Noise (20% prob) and Gaussian Blur (20% prob) to simulate real-world MRI artifacts.

---

## 3. Model Architecture (MRAF-Net)

MRAF-Net is a specialized, deep 3D Convolutional Neural Network (CNN) built upon the foundation of a 3D U-Net, but heavily upgraded with modern modules.

### Core Architectural Components:
1. **Encoder-Decoder Backbone:** Extracts deep hierarchical features across 5 resolution levels. The feature channels scale aggressively: `32 -> 64 -> 128 -> 256 -> 320`.
2. **Cross-Modality Fusion:** By fusing T1, T1ce, T2, and FLAIR streams early and often, the network comprehensively profiles the tumor (e.g., relying on FLAIR for edema and T1ce for the active core).
3. **ASPP (Atrous Spatial Pyramid Pooling):** Placed at the bottleneck. ASPP uses dilated convolutions with rates of `[6, 12, 18]`. This expands the network's "field of view" without losing resolution, allowing it to capture highly variable tumor sizes (both massive clusters and tiny fragments) simultaneously.
4. **Attention Gates:** Integrated into the decoder pathways. Attention gates proactively suppress irrelevant background noise from healthy brain tissue and force the model to focus gradient updates purely on complex tumor boundaries.
5. **Deep Supervision:** Auxiliary outputs at intermediate decoder levels (`ds_weights: [1.0, 0.5, 0.25, 0.125]`) inject gradient signals directly into the middle layers, accelerating convergence.

---

## 4. Training Details & Methodology

The model employs state-of-the-art training techniques specifically geared for highly imbalanced 3D medical images.

### Target Optimization
- **Loss Function:** Hybrid `dice_ce`. Combines Dice Loss (excellent for spatial overlap evaluation) with Cross-Entropy Loss (excellent for voxel-wise classification precision).
- **Optimizer:** `AdamW` (Adam with decoupled Weight Decay at `1e-5`) for exceptional regularization.
- **LR Scheduler:** Cosine Annealing with a 10-epoch Warmup phase starting at a learning rate of `1e-4`.

### Hardware / Resource Methodology
- **AMP (Automatic Mixed Precision):** Uses computational FP16 (half-precision) where possible to drastically speed up training and save VRAM.
- **Gradient Checkpointing:** Re-computes intermediate activations during the backward pass to drastically lower VRAM usage, enabling training on consumer GPUs like an 8GB laptop GPU.

### Inference & Prediction
- **Sliding Window Inference:** Since the model is trained on patches, sliding window inference with a 50% overlap and Gaussian blending predicts the full volume smoothly without edge artifacts.
- **Test-Time Augmentation (TTA):** Enabled via flip augmentations to boost final prediction robustness.

---

## 5. Performance & Accuracy Metrics

After a robust training cycle of roughly 100 to 300 epochs, MRAF-Net achieves highly competitive segmentation accuracies evaluated across essential clinical tumor groupings:

| Clinical Target | Expected Dice Score Range |
| :--- | :--- |
| **WT (Whole Tumor)** | `0.88 - 0.91` |
| **TC (Tumor Core)** | `0.82 - 0.86` |
| **ET (Enhancing Tumor)**| `0.75 - 0.80` |
| **Mean Dice** | `0.82 - 0.86` |

*Other metrics computed simultaneously include Hausdorff Distance (95%), Sensitivity, and Specificity.*

---

## 6. Visualizations and Tumor Properties

### Visualizing the Segmentation
**Can we color parts of the brain for easier understanding?**
Yes. The configured prediction scripts support `save_visualization: true`. They inherently map the numerical output segments to distinctive RGB color overlays overlaid atop grayscale MRI slices. 

Specifically, the system maps the tumor regions to the following colors for clinical review:
- **<span style="color:green">Green</span> (Label 1):** Necrotic and Non-Enhancing Tumor Core (NCR/NET). This is the dead or inactive tissue at the center of the tumor.
- **<span style="color:yellow">Yellow</span> (Label 2):** Peritumoral Edema (ED). This highlights the swelling and fluid accumulation in the brain tissue surrounding the tumor.
- **<span style="color:red">Red</span> (Label 4 / 3):** GD-Enhancing Tumor (ET). This marks the highly active, rapidly growing, and most aggressive boundaries of the tumor.

The project includes a standalone PyQt5 graphical interface (`gui/app.py`), empowering users to easily load a patient's case, compute the prediction, and scroll visually through the 3D-colored slices dynamically with a built-in color legend.

### Tumor Dimension Constraints
**Are explicit tumor sizes defined artificially?**
No. Brain tumors fluctuate drastically in both morphological shape and anatomical volume. 
MRAF-Net compensates for this intrinsically. By leveraging multi-resolution cascading and the ASPP context module, the model organically learns to "zoom in" on small structural abnormalities, while retaining the macro perspective necessary to encompass massive tumor bodies. Furthermore, patch-based training ensures the network learns *texture and localized context* rather than attempting to memorize static structural coordinates.
