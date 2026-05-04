# MRAF-Net: Comprehensive Project & Architecture Details

This document provides a detailed overview of the MRAF-Net (Multi-Resolution Aligned and Robust Fusion Network) project, covering everything from the dataset, preprocessing, deep learning architecture, training methodology, and accuracy.

## Table of Contents
1. [Dataset Details & Type](#1-dataset-details--type)
2. [Dataset Preprocessing & Augmentation](#2-dataset-preprocessing--augmentation)
3. [Model Architecture (MRAF-Net)](#3-model-architecture-mraf-net)
4. [Training Details & Methodology](#4-training-details--methodology)
5. [Performance & Accuracy Metrics](#5-performance--accuracy-metrics)
6. [Visualizations and Tumor Properties](#6-visualizations-and-tumor-properties)
7. [Academic Defense (VIVA) Q&A Bank](#7-academic-defense-viva-qa-bank)
8. [Main Coding Related Q&A](#8-main-coding-related-qa)

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

---

## 7. Academic Defense (VIVA) Q&A Bank

This section contains potential academic defense questions and expertly formulated answers covering the theoretical and practical dimensions of the MRAF-Net project.

### The Problem Space & Dataset
**Q1: Why did you choose the BraTS dataset over other medical imaging datasets?**
**A:** BraTS is the global gold standard for brain tumor segmentation. It provides massive, multi-institutional, multi-modal, and expertly annotated 3D data. The challenge intrinsically forces the model to handle extreme class imbalance and diverse tumor morphologies, making it an excellent benchmark for deep learning robustness.

**Q2: Why do you need 4 different MRI modalities? Couldn't you just use one?**
**A:** No single modality captures the full tumor profile. 
- **T1ce** (contrast-enhanced) highlights the active, enhancing tumor boundary (ET) because the contrast agent gathers where the blood-brain barrier is broken.
- **FLAIR** suppresses the fluid signal but explicitly highlights the entire peritumoral edema (swelling).
- Together, they allow the network to cross-reference structural anomalies to output precise, multi-class segmentations that one scan alone could never provide.

### Architecture (MRAF-Net)
**Q3: Your architecture is based on a 3D U-Net. Why use a 3D CNN instead of slicing the MRI into 2D images and using a 2D CNN?**
**A:** Slicing a 3D brain scan into 2D images completely destroys spatial connectivity along the Z-axis (depth). Tumors are 3D volumetric masses. A 3D CNN preserves this depth continuity natively, significantly reducing false positives and eliminating jagged segmentations between sequential vertical slices.

**Q4: Can you explain ASPP (Atrous Spatial Pyramid Pooling) and why it's critical for this project?**
**A:** Brain tumors exhibit extreme scale variations—ranging from massive connected lobes to tiny scattered fragments. ASPP applies multiple parallel layers using "dilated" (atrous) convolutions (with rates of 6, 12, 18). This artificially expands the network's "field of view" to capture large surrounding contexts without aggressively down-sampling the image and losing resolution. It ensures the model identifies both huge targets and microscopic anomalies simultaneously.

**Q5: What role do Attention Gates play in your decoder?**
**A:** A standard U-Net blindly passes all low-level features (including healthy background tissue) straight across to the decoder via skip connections. Attention Gates act as a spatial filter. Using higher-level semantic features, they calculate an "attention map" that actively suppresses background noise and specifically highlights ambiguous tumor boundaries before merging the features.

### Training & Optimization
**Q6: Why did you use Patch-Based Training instead of passing the entire 3D brain volume at once?**
**A:** Purely due to hardware limitations and dataset heterogeneity. A full 3D scan (`240x240x155` voxels) with 4 modalities requires enormous GPU VRAM (well beyond 24GB). By extracting smaller random patches (e.g., `96x96x96`), we allow the model to train efficiently on consumer-grade hardware (using ≤8GB VRAM) while actually improving the model's ability to learn localized textures instead of lazily memorizing global spatial coordinates.

**Q7: Explain your loss function. Why not just use standard Cross-Entropy Loss?**
**A:** Brain tumor datasets suffer from extreme class imbalance. Healthy background tissue (Class 0) makes up over 98% of the scan, while the tumor classes make up less than 2%. Standard Cross-Entropy would blindly predict "Background" everywhere to passively achieve 98% accuracy while entirely failing to segment the tumor.
We use a **Hybrid `dice_ce` Loss**: 
- **Dice Loss:** Maximizes the volumetric spatial overlap between the prediction and the ground truth (highly robust against background imbalance).
- **Cross-Entropy (CE) Loss:** Ensures rigorous pixel-level classification accuracy is maintained.

**Q8: What is Automatic Mixed Precision (AMP) and why use it?**
**A:** Traditional deep learning uses FP32 (32-bit floating point precision). AMP automatically identifies operations that can safely run in FP16 (16-bit half-precision) without losing mathematical stability. This drastically cuts down GPU Memory usage by almost half and noticeably accelerates matrix multiplications on modern NVIDIA GPUs.

### Evaluation Metrics
**Q9: You've reported Dice Scores, but also HD95 (Hausdorff Distance 95). Why both?**
**A:** 
- The **Dice Score** tells us the volumetric overlap (e.g., "We got 90% of the bulk tumor body correct"). However, Dice is insensitive to structural boundaries.
- **HD95** measures the maximum distance between the true boundary and our predicted boundary in millimeters (ignoring the top 5% extreme outliers to prevent statistical skew). In clinical settings (like surgically targeting radiation therapy), getting the boundary exactly right (low HD95) is just as critical as getting the bulk volume right.

---

## 8. Main Coding Related Q&A

This section focuses on implementation-level questions that may come up during code review, demonstration, or viva discussion.

### GUI Data Flow

**Q10: What is the main flow when the user runs segmentation from the GUI?**
**A:** In `gui/app.py`, the GUI loads the uploaded NIfTI files, validates that all modalities have matching shapes, stacks the four modalities, normalizes intensities, runs model prediction, calculates metrics, stores the result, and renders the visualization.

The important flow is:

```python
images = np.stack([flair_data, t1_data, t1ce_data, t2_data], axis=0)
images_norm = normalize_intensity(images)
segmentation, inference_time, peak_gpu_mb = model.predict(images_norm)
```

Then the predicted segmentation is passed into:

```python
metrics = compute_tumor_metrics(segmentation, ground_truth, voxel_vol, inference_time, peak_gpu_mb)
```

**Q11: Why are the four MRI modalities stacked before prediction?**
**A:** The model expects a multi-channel 3D input. Each modality contributes different clinical information: FLAIR highlights edema, T1ce highlights enhancing tumor, and T1/T2 provide additional anatomical contrast. Stacking creates one tensor where the channel dimension contains `[FLAIR, T1, T1ce, T2]`.

**Q12: Why does the code validate modality shapes before prediction?**
**A:** All four modalities must describe the same patient volume with the same spatial dimensions. If one scan has a different shape, voxel positions no longer align correctly. The model would receive mismatched channels, and the segmentation could become invalid. Shape validation prevents that failure before inference.

### GUI Controls

**Q13: What do the `Axial`, `Coronal`, and `Sagittal` view options do in code?**
**A:** They control which axis of the 3D MRI array is sliced for display.

```python
if view == "Axial":
    mri_slice = flair[:, :, idx]
elif view == "Coronal":
    mri_slice = flair[:, idx, :]
else:
    mri_slice = flair[idx, :, :]
```

`Axial` shows top-to-bottom slices, `Coronal` shows front-to-back slices, and `Sagittal` shows left-to-right side slices.

**Q14: What does the slice slider change?**
**A:** The slice slider changes the index `idx` used to extract a 2D image from the 3D MRI volume. It affects only what the user sees in the visualization panel. It does not change the model prediction or calculated metrics.

**Q15: What does overlay opacity do in code?**
**A:** Opacity controls the blend between the grayscale MRI and the colored tumor mask. The code uses `alpha` to mix MRI intensity with the tumor color:

```python
mri_rgb[:, :, c][mask] = (
    mri_rgb[:, :, c][mask] * (1 - alpha * overlay[:, :, 3][mask]) +
    overlay[:, :, c][mask] * alpha * overlay[:, :, 3][mask]
)
```

An opacity of `0` hides the overlay, `0.5` gives a balanced overlay, and `1` makes the tumor color strongest.

### Labels, Colors, and Segmentation Output

**Q16: What label values does the segmentation output use?**
**A:** The GUI uses BraTS-style tumor labels:

| Label | Meaning | Color |
| --- | --- | --- |
| `0` | Background | Transparent / black |
| `1` | NCR/NET - Necrotic and non-enhancing tumor core | Green |
| `2` | ED - Peritumoral edema | Yellow |
| `4` | ET - Enhancing tumor | Red |

These are configured in the `CONFIG` dictionary in `gui/app.py`.

**Q17: Why is enhancing tumor sometimes label `4` and sometimes converted to label `3`?**
**A:** BraTS ground-truth masks commonly use label `4` for Enhancing Tumor. Some metric helper functions internally expect classes as `0, 1, 2, 3`. Therefore, before metric calculation, the GUI converts label `4` into `3`:

```python
seg_int[seg_int == 4] = 3
gt_int[gt_int == 4] = 3
```

This conversion is only for metric helper compatibility. The displayed and exported segmentation still follows the tumor label convention used by the GUI.

### Tumor Volumes and Composition

**Q18: How does the GUI calculate tumor volumes?**
**A:** The GUI counts how many voxels are predicted for each tumor label, multiplies the count by the physical voxel volume, and converts cubic millimeters to milliliters.

```python
unique, counts = np.unique(segmentation, return_counts=True)
vol_dict = dict(zip(unique.astype(int), counts))

vol_ncr = vol_dict.get(1, 0) * voxel_volume / 1000
vol_ed = vol_dict.get(2, 0) * voxel_volume / 1000
vol_et = vol_dict.get(4, 0) * voxel_volume / 1000
```

The formula is:

```text
volume_ml = voxel_count * voxel_volume_mm3 / 1000
```

**Q19: Where does `voxel_volume` come from?**
**A:** In `gui/app.py`, it is calculated from the NIfTI affine matrix:

```python
voxel_vol = float(np.abs(np.linalg.det(affine[:3, :3])))
```

The affine stores spatial scaling information. Taking the determinant of its 3D spatial part gives the physical volume of one voxel in cubic millimeters.

**Q20: What do Whole Tumor, Tumor Core, Enhancing, Edema, and Necrotic volumes mean?**
**A:** These are clinically meaningful groupings of tumor labels:

| GUI Metric | Formula | Meaning |
| --- | --- | --- |
| Whole Tumor (`WT`) | `NCR/NET + Edema + Enhancing` | Total predicted tumor burden |
| Tumor Core (`TC`) | `NCR/NET + Enhancing` | Core tumor region without edema |
| Enhancing (`ET`) | `Enhancing` only | Active enhancing tissue |
| Edema (`ED`) | `Edema` only | Swelling around the tumor |
| Necrotic (`NCR`) | `NCR/NET` only | Dead or non-enhancing core tissue |

**Q21: What does predicted tumor composition mean?**
**A:** It is the percentage split of the predicted tumor volume. It tells how much of the predicted tumor is NCR/NET, Edema, and Enhancing.

```python
metrics["composition_pct"] = {
    "ncr_net": round((vol_ncr / total_tumor_ml) * 100, 2) if total_tumor_ml else 0.0,
    "edema": round((vol_ed / total_tumor_ml) * 100, 2) if total_tumor_ml else 0.0,
    "enhancing": round((vol_et / total_tumor_ml) * 100, 2) if total_tumor_ml else 0.0,
}
```

This is not model confidence. For example, `Edema = 60%` means 60% of the predicted tumor volume was classified as edema.

**Q22: What happens if the model predicts no tumor voxels?**
**A:** The code avoids division by zero. If total tumor volume is `0`, all composition values are returned as `0.0%`.

### Evaluation Metrics

**Q23: When does the GUI calculate Dice, Sensitivity, Specificity, and HD95?**
**A:** These are calculated only when the user uploads a ground-truth segmentation mask. Without ground truth, the GUI can still show predicted volumes and composition, but it cannot compare the prediction against expert labels.

**Q24: How does the GUI create the BraTS evaluation regions?**
**A:** After converting label `4` to `3` for metric compatibility, the code calls:

```python
pred_wt, pred_tc, pred_et, tgt_wt, tgt_tc, tgt_et = compute_brats_regions(seg_int, gt_int)
```

This creates binary masks for:

| Region | Labels |
| --- | --- |
| Whole Tumor | Tumor labels combined |
| Tumor Core | Core tumor labels |
| Enhancing Tumor | Enhancing tumor only |

**Q25: Why are evaluation percentages different from predicted tumor composition percentages?**
**A:** Predicted tumor composition is a breakdown of the model's own predicted tumor mask. Evaluation metrics compare the model prediction with the uploaded ground truth. Therefore, composition explains the prediction internally, while Dice/Sensitivity/Specificity/HD95 measure correctness against expert annotation.

### Rendering and Export

**Q26: Which function builds the card-based results dashboard?**
**A:** `format_metrics(metrics)` in `gui/app.py` converts the calculated metrics dictionary into HTML cards. It renders tumor volume cards, predicted tumor composition rows, runtime cards, and optional evaluation cards when ground truth is available.

**Q27: What does the 3D view use as input?**
**A:** The 3D view uses the predicted segmentation mask stored after inference. It visualizes tumor label regions in 3D. In the standalone GUI, the segmentation is downsampled using slicing such as `[::3, ::3, ::3]` for faster plotting.

**Q28: What does export save?**
**A:** Export saves the predicted 3D segmentation mask as a NIfTI file. It saves label values, not the colored overlay image. This allows the segmentation to be reused in medical imaging tools.

### Code Design and Reliability

**Q29: Why does the GUI store `stored_data` globally after prediction?**
**A:** The GUI needs access to the latest MRI volume and segmentation after the user changes slice, view, overlay, or opacity controls. Storing `flair`, `segmentation`, `ground_truth`, `affine`, and `metrics` lets callbacks update the visualization without re-running model inference.

**Q30: Which file should be treated as the main source of truth for the GUI?**
**A:** `gui/app.py` should be treated as the main full GUI implementation because it connects to the project source modules. `gui/standalone_gui.py` is useful for demonstrations because it is more self-contained.

**Q31: Is there any difference in volume handling between `gui/app.py` and `gui/standalone_gui.py`?**
**A:** Yes. `gui/app.py` passes the NIfTI-derived `voxel_volume` into `compute_tumor_metrics`, so tumor volume reflects physical voxel size. In `gui/standalone_gui.py`, `compute_advanced_metrics` currently divides voxel counts by `1000` directly, effectively assuming `1 mm3` per voxel for volume cards, while spacing is separately calculated for HD95. For the most accurate physical volume explanation, use the `gui/app.py` logic.

**Q32: Why is normalization done before prediction?**
**A:** MRI intensity values vary widely across scanners, patients, and modalities. Normalization standardizes the input scale so the model receives data closer to what it saw during training, improving stability and prediction quality.

**Q33: Which values affect prediction and which values affect only visualization?**
**A:** Uploaded modalities, model checkpoint, and normalization affect prediction. View, slice, overlay toggle, and opacity affect only visualization after prediction. Tumor volumes and composition are calculated from the predicted segmentation mask, so they do not change when the user changes view or opacity.
