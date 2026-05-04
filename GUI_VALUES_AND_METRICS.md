# GUI Values, Metrics, and Data Flow

This document explains the values shown in the MRAF-Net GUI, what each value means, and how the code gets or calculates it.

## Main GUI Files

| File | Purpose |
| --- | --- |
| `gui/app.py` | Full Gradio GUI connected to the main `src` model code. |
| `gui/standalone_gui.py` | Self-contained GUI version with embedded model support. |
| `scripts/predict.py` | Command-line prediction and visualization helpers. |
| `src/utils/metrics.py` | Core metric functions such as Dice, Sensitivity, Specificity, HD95, and BraTS region grouping. |

## Input Values

| GUI Value | What It Means | How The Value Is Used | Main Code Location |
| --- | --- | --- | --- |
| Model checkpoint path | Path to trained `.pth` weights. | Loaded before prediction so the model can run inference. | `gui/app.py`, model loading section |
| FLAIR file | MRI sequence useful for edema and lesion visibility. | Loaded as a NIfTI volume and stacked with the other modalities. | `gui/app.py`, `run_segmentation` |
| T1 file | T1-weighted MRI sequence. | Loaded as one input channel. | `gui/app.py`, `run_segmentation` |
| T1ce file | Contrast-enhanced T1 sequence, useful for active enhancing tumor. | Loaded as one input channel. | `gui/app.py`, `run_segmentation` |
| T2 file | T2-weighted MRI sequence. | Loaded as one input channel. | `gui/app.py`, `run_segmentation` |
| Ground truth file | Optional expert segmentation mask. | If uploaded, the GUI computes Dice, Sensitivity, Specificity, and HD95. | `gui/app.py`, `compute_tumor_metrics` |

The four MRI modalities are stacked in this order:

```python
images = np.stack([flair_data, t1_data, t1ce_data, t2_data], axis=0)
```

Then they are normalized before prediction:

```python
images_norm = normalize_intensity(images)
segmentation, inference_time, peak_gpu_mb = model.predict(images_norm)
```

## Visualization Controls

| GUI Control | Options / Range | What It Does | How The Value Is Applied |
| --- | --- | --- | --- |
| View | `Axial`, `Coronal`, `Sagittal` | Chooses the anatomical plane used to display a 2D slice from the 3D volume. | `update_slice_view` chooses a different array axis. |
| Slice | `0` to current maximum slice index | Chooses which 2D slice is displayed. | The selected value becomes `idx`, then the MRI and mask are sliced. |
| Show Overlay | `True` or `False` | Shows or hides the colored segmentation mask over the MRI slice. | If enabled, `create_overlay_slice` blends mask colors with the MRI. |
| Overlay Opacity | `0` to `1` | Controls how strong the segmentation colors appear. | Used as `alpha` in the overlay blending formula. |

### View Axis Mapping

| View | Code Slice | Meaning |
| --- | --- | --- |
| Axial | `flair[:, :, idx]` | Top-to-bottom slices through the scan. |
| Coronal | `flair[:, idx, :]` | Front-to-back slices through the scan. |
| Sagittal | `flair[idx, :, :]` | Left-to-right slices through the scan. |

The main function is:

```python
def update_slice_view(slice_idx: int, view: str, show_overlay: bool, alpha: float):
```

In `gui/standalone_gui.py`, the equivalent function is:

```python
def update_view(slice_idx, view, overlay, alpha):
```

## Segmentation Labels and Colors

| Label | Short Name | Full Meaning | GUI Color |
| --- | --- | --- | --- |
| `0` | Background | Healthy tissue or non-tumor area | Transparent / black |
| `1` | NCR/NET | Necrotic and non-enhancing tumor core | Green |
| `2` | ED | Peritumoral edema / swelling | Yellow |
| `4` | ET | Enhancing tumor | Red |

These are configured in `gui/app.py`:

```python
"tumor_labels": {
    0: "Background",
    1: "NCR/NET (Necrotic Core)",
    2: "ED (Peritumoral Edema)",
    4: "ET (Enhancing Tumor)"
}
```

The color map is:

```python
"colors": {
    0: [0, 0, 0, 0],
    1: [0, 255, 0, 180],
    2: [255, 255, 0, 180],
    4: [255, 0, 0, 180]
}
```

## Overlay Opacity Calculation

The overlay is created by blending the grayscale MRI with the colored segmentation mask.

Simplified formula:

```text
display_pixel = MRI_pixel * (1 - alpha * mask_alpha) + color_pixel * (alpha * mask_alpha)
```

Meaning:

| Opacity | Result |
| --- | --- |
| `0` | Only the MRI is visible. |
| `0.5` | MRI and tumor color are both visible. |
| `1` | Tumor overlay color is strongest. |

This affects only visualization. It does not change the model output.

## Tumor Volume Values

Tumor volumes are calculated from the predicted segmentation mask.

The GUI counts how many voxels belong to each tumor label:

```python
unique, counts = np.unique(segmentation, return_counts=True)
vol_dict = dict(zip(unique.astype(int), counts))
```

Then the voxel count is converted into milliliters:

```text
volume_ml = label_voxel_count * voxel_volume_mm3 / 1000
```

The `/ 1000` converts cubic millimeters to milliliters because:

```text
1000 mm3 = 1 ml
```

The voxel volume is taken from the NIfTI affine matrix:

```python
voxel_vol = float(np.abs(np.linalg.det(affine[:3, :3])))
```

### Volume Cards

| GUI Card | Formula | Meaning |
| --- | --- | --- |
| Whole Tumor (`WT`) | `NCR/NET + Edema + Enhancing` | Total predicted tumor burden. |
| Tumor Core (`TC`) | `NCR/NET + Enhancing` | Main tumor core without edema. |
| Enhancing (`ET`) | `Enhancing` only | Active enhancing tumor tissue. |
| Edema (`ED`) | `Edema` only | Swelling around the tumor. |
| Necrotic (`NCR`) | `NCR/NET` only | Dead or non-enhancing core tissue. |

Main function:

```python
def compute_tumor_metrics(segmentation, ground_truth=None, voxel_volume=1.0, inference_time=0.0, peak_gpu_mb=0.0):
```

Important volume lines:

```python
vol_ncr = vol_dict.get(1, 0) * voxel_volume / 1000
vol_ed = vol_dict.get(2, 0) * voxel_volume / 1000
vol_et = vol_dict.get(4, 0) * voxel_volume / 1000
```

## Predicted Tumor Composition

Predicted tumor composition is the percentage split of the predicted tumor mask.

It answers:

```text
Of all voxels predicted as tumor, what percentage is NCR/NET, Edema, and Enhancing?
```

Formula:

```text
composition_percent = region_volume_ml / total_tumor_volume_ml * 100
```

Example:

| Region | Volume |
| --- | --- |
| NCR/NET | `20 ml` |
| Edema | `55 ml` |
| Enhancing | `25 ml` |
| Whole Tumor | `100 ml` |

Composition:

| Region | Composition |
| --- | --- |
| NCR/NET | `20%` |
| Edema | `55%` |
| Enhancing | `25%` |

This is not model confidence. It is a volume share.

Main code:

```python
metrics["composition_pct"] = {
    "ncr_net": round((vol_ncr / total_tumor_ml) * 100, 2) if total_tumor_ml else 0.0,
    "edema": round((vol_ed / total_tumor_ml) * 100, 2) if total_tumor_ml else 0.0,
    "enhancing": round((vol_et / total_tumor_ml) * 100, 2) if total_tumor_ml else 0.0,
}
```

## Slice Legend Percentages

The rendered slice image also shows region-share information under the image.

Function:

```python
def compute_region_percentages(segmentation: np.ndarray) -> Dict[int, float]:
```

This counts label voxels in the full predicted segmentation and returns the share of each tumor label. Like composition, this is not confidence.

## Evaluation Metrics

Evaluation metrics are shown only if the user uploads a ground-truth mask.

| Metric | What It Means | Better Value |
| --- | --- | --- |
| Dice | Overlap between predicted mask and ground truth. | Higher, maximum `1.0`. |
| Sensitivity | How much real tumor was detected. | Higher. |
| Specificity | How well non-tumor voxels were rejected. | Higher. |
| HD95 | Boundary distance error in millimeters, ignoring the worst 5% outliers. | Lower. |

The GUI computes these for BraTS clinical regions:

| Evaluation Region | Labels Included |
| --- | --- |
| Whole Tumor (`WT`) | `1 + 2 + 4` |
| Tumor Core (`TC`) | `1 + 4` |
| Enhancing Tumor (`ET`) | `4` |

Before calling the metric helper, label `4` is converted to `3` because the metric module uses the internal convention `0, 1, 2, 3`:

```python
seg_int = segmentation.copy().astype(np.int64)
seg_int[seg_int == 4] = 3
gt_int = ground_truth.copy().astype(np.int64)
gt_int[gt_int == 4] = 3
```

Then the regions are built:

```python
pred_wt, pred_tc, pred_et, tgt_wt, tgt_tc, tgt_et = compute_brats_regions(seg_int, gt_int)
```

## Runtime Values

| GUI Value | Meaning | How It Is Obtained |
| --- | --- | --- |
| Inference Time | Time taken to run model prediction. | Returned by `model.predict(images_norm)`. |
| Peak GPU Memory | Maximum CUDA memory allocated during prediction. | Returned by `model.predict(images_norm)` when CUDA is available. |

## 3D View

The 3D view uses the predicted segmentation mask to create a 3D visualization of tumor regions.

In the standalone GUI, the segmentation is downsampled for faster plotting:

```python
seg = stored_data["seg"][::3, ::3, ::3]
```

This changes only the 3D visualization performance. It does not change the saved segmentation result.

## Export Value

The export tab saves the predicted segmentation as a NIfTI file. The output is the predicted 3D segmentation mask, not the colored overlay image. The saved file keeps the segmentation labels so it can be opened in medical imaging tools.

## Full GUI Data Flow

```text
User uploads FLAIR, T1, T1ce, T2
        |
        v
NIfTI files are loaded as 3D arrays
        |
        v
Shapes are validated
        |
        v
Modalities are stacked into 4-channel input
        |
        v
Intensity normalization is applied
        |
        v
Model predicts a 3D segmentation mask
        |
        v
Voxel counts, volumes, composition, runtime, and optional ground-truth metrics are calculated
        |
        v
GUI renders slice view, overlay, volume cards, composition rows, evaluation cards, 3D view, and export
```

## Coding Notes

- `gui/app.py` is the best source of truth for the full GUI.
- `gui/standalone_gui.py` has equivalent logic, but its volume calculation assumes `1 mm3` per voxel in `compute_advanced_metrics`, while it separately computes spacing for HD95.
- The GUI composition percentages and evaluation percentages are different:
  - Composition percentages are tumor region shares.
  - Evaluation percentages compare prediction against ground truth.
- Opacity, slice, and view are display controls only. They do not change the predicted segmentation.
