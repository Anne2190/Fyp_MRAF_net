# MRAF-Net Project Overview

## 1. The Core Idea: MRAF-Net
**MRAF-Net** stands for **Multi-Resolution Aligned and Robust Fusion Network**. The central problem this model addresses is how to effectively combine (fuse) information from four different MRI modalities (T1, T1ce, T2, FLAIR) to accurately segment brain tumors.

### Key Innovations
*   **Anatomical & Pathological Alignment (The "Aligned" part)**:
    Instead of blindly stacking all images together, MRAF-Net uses medical knowledge to group them:
    *   **Anatomical Group (T1 + T1ce)**: Provides structural details and highlights the tumor core/enhancing tumor.
    *   **Pathological Group (T2 + FLAIR)**: Best for highlighting edema (fluid) and general lesions.
    This grouping allows the network to learn specialized features for Structure vs. Abnormality before fusing them.

*   **Robust Fusion (The "Robust" part)**:
    Uses a **Cross-Modality Attentional Gate** ([fusion.py](file:///p:/FYP_MRAF-Net/Fyp_MRAF_net/src/models/fusion.py)). It allows the "Anatomical" features to refine the "Pathological" features and vice versa, suppressing noise and enhancing relevant boundaries.

*   **Multi-Resolution (The "Multi-Resolution" part)**:
    The encoder uses **ASPP (Atrous Spatial Pyramid Pooling)** and dilated convolutions to look at the tumor at different scales simultaneously. This improves detection of both tiny tumor sub-regions and large edema areas.

## 2. Implementation Details ("How I done it")
The project is built using **PyTorch** and structured for modularity.

### Architecture (`src/models/`)
*   **`mraf_net.py`**: The main backbone. It processes inputs through 4 separate encoders initially (one per modality) before fusing them.
*   **`fusion.py`**: Contains the custom logic for the "Anatomical vs Pathological" grouping and attention gating.
*   **`attention.py`**: Implements 3D Attention Gates and SE-Blocks used in the decoder to focus on tumor boundaries.
*   **`decoder.py`**: Uses Deep Supervision, meaning it calculates loss at 3 different resolutions during training to ensure the model learns good features at every level.

### Optimization for Hardware ("Laptop Mode")
A dedicated **Laptop Mode** was implemented to allow training on consumer hardware (like your machine):
*   **Patch-based Training**: Instead of the whole brain, it trains on smaller cubic chunks (e.g., 64x64x64).
*   **Mixed Precision (AMP)**: Uses standard `float32` for stability but `float16` for heavy computations to save 50% VRAM.
*   **Gradient Checkpointing**: Trades a bit of computation speed for massive memory savings by re-computing activations instead of storing them.

## 3. The Dataset
*   **Source**: **BraTS 2020** (Brain Tumor Segmentation Challenge 2020).
*   **Input Data**:
    1.  **T1**: Structural MRI.
    2.  **T1ce**: T1 with contrast agent (highlights active tumor).
    3.  **T2**: Structural, highlights some fluids.
    4.  **FLAIR**: Fluid-Attenuated Inversion Recovery (highlights edema).
*   **Ground Truth Labels**:
    *   **Label 1**: Necrotic/Non-enhancing tumor core.
    *   **Label 2**: Peritumoral Edema.
    *   **Label 4**: Enhancing Tumor.
    *   *(Label 0 is background)*.

### Data Preparation (`scripts/prepare_data.py`)
1.  **Verification**: Checks every case folder to ensure all 4 NIfTI files + Segmentation exist.
2.  **Splitting**: Randomly splits valid cases into 80% Training and 20% Validation.
3.  **Normalization**: Computes Mean and Std Dev for every modality to normalize pixel intensities (Z-score normalization).

## 4. Training Process (`scripts/train.py`)
*   **Loss Function**: **Dice Loss + Cross Entropy Loss**.
    *   *Dice Loss* optimizes for overlap (good for segmentation).
    *   *Cross Entropy* optimizes for pixel-wise classification accuracy.
*   **Optimizer**: **AdamW** (Adam with Weight Decay) for stable convergence.
*   **Scheduler**: **Cosine Annealing**. Starts with a high learning rate and smoothly decreases it, which often finds better local minima than step-based drops.
*   **Metrics**:
    *   **Dice Score**: The primary success metric (target > 0.85 for Whole Tumor).
    *   **Hausdorff Distance (HD95)**: Measures how far the predicted boundary is from the real boundary (lower is better).
