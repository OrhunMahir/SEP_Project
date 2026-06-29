# Swin-Tiny + YOLO Grad-CAM Results

This report documents the reproduced Swin-Tiny experiment, the Grad-CAM
visualization issue found during review, and the corrected `stage3_last_norm1`
outputs. The model was trained on the 4,534 images that were available on the
cluster, using one stratified training/validation split.

> **Dataset scope:** the original manifest contains 5,432 rows, but 898 source
> images were unavailable. These results therefore must not be compared
> directly with runs trained on the complete dataset.

![Corrected Abyssinian Grad-CAM example](panels/stage3/00_abyssinian.jpg)

## What changed after the attribution audit

The first Grad-CAM implementation used `last_block_norm1`, the `norm1` module
inside the final Swin transformer block. Visual inspection showed suspicious
maps: many heatmaps emphasized image borders or background regions instead of
the animal. This was not a reversed-color problem. Red already represented high
positive evidence. The issue was that the chosen final-stage Swin layer created
a systematic outer-ring artifact for this checkpoint.

We audited 143 official-test images with several attribution targets and a
gradient-free occlusion check. The final-stage `last_block_norm1` setting was
clearly the problematic one:

| Attribution target | Spatial grid | Outer-ring peak rate | Mean outer-ring mass | Median edge/inner ratio |
| --- | ---: | ---: | ---: | ---: |
| `stage2_last_norm1` | 28x28 | 16.1% | 12.5% | 0.885 |
| `stage3_last_norm1` | 14x14 | **23.1%** | **30.0%** | **1.103** |
| `last_block_norm1` / final stage | 7x7 | 79.7% | 78.4% | 3.834 |
| `last_block_norm2` | 7x7 | 9.1% | 14.9% | 0.118 |
| `final_norm` | 7x7 | 7.7% | 29.4% | 0.424 |
| Occlusion check | 7x7 | 41.7% | 52.4% | 1.239 |

The project default was therefore changed from `last_block_norm1` to
`stage3_last_norm1`. This target has a 14x14 grid for 224x224 inputs, keeps
more spatial detail than the final 7x7 stage, and no longer shows the severe
outer-ring failure. The visualization code was also updated to use a `turbo`
colormap, activation-weighted overlays, and saved raw `*_heatmap.npy` arrays
for later audits.

The correction does **not** turn Grad-CAM into a perfect segmentation mask.
Some corrected maps still highlight background, bedding, snow, crop edges, or
other contextual cues. That observation is important: after the artifact was
reduced, remaining background activations are more likely to reflect real model
shortcut risk or coarse Grad-CAM localization limits rather than a simple
plotting bug.

Legacy panels generated with the old final-stage target are retained in
`panels/yolo/` and `panels/raw_fallback/` only for reproducibility. The panels
shown below use the corrected `stage3_last_norm1` target.

## Experiment setup

| Component | Setting |
| --- | --- |
| Model | ImageNet-pretrained Swin-Tiny |
| Trainable parameters | 27,535,503 |
| Dataset used | 4,534 available images |
| Split | 3,628 train / 906 validation, stratified, seed 42 |
| Input | 224 × 224 |
| Batch size | 8 |
| Optimizer settings | learning rate `1e-4`, weight decay `1e-4` |
| Regularization | dropout `0.1`, label smoothing `0.1` |
| Schedule | 5 warm-up epochs, 50 maximum epochs |
| Detector preprocessing | YOLOv8n, confidence `0.25`, square crop, `0.10` padding |
| Best checkpoint | epoch 46, selected by validation macro-F1 |
| Training hardware | NVIDIA GeForce RTX 2060, 8 GB |
| Training time | 1,125.76 seconds (18 minutes 45.76 seconds) |

YOLO produced a largest-cat-or-dog crop for 3,557 of the 4,534 images. The
remaining 977 images used the training pipeline's raw-image fallback.

## Validation results

The best checkpoint was evaluated on the 906-image validation split.

| Metric | No confidence threshold | Calibrated threshold `0.50` |
| --- | ---: | ---: |
| Accuracy | 94.26% | **94.59%** |
| Macro precision | 93.08% | **94.28%** |
| Macro recall | **96.30%** | 95.75% |
| Macro-F1 | 94.46% | **94.83%** |
| Weighted-F1 | 94.24% | **94.57%** |
| Reject precision | **98.70%** | 95.92% |
| Reject recall | 88.33% | **91.44%** |
| Reject F1 | 93.22% | **93.63%** |
| False accepts | 30 | **22** |
| False rejects | **3** | 10 |

The threshold was selected by validation accuracy, with macro-F1, reject F1,
false accepts, and false rejects used as tie-breakers. The threshold raises
accuracy by 0.33 percentage points and reduces false accepts by eight, at the
cost of seven additional false rejects.

| Baseline confusion matrix | Threshold `0.50` confusion matrix |
| --- | --- |
| ![Baseline confusion matrix](metrics/confusion_matrix_baseline.png) | ![Calibrated confusion matrix](metrics/confusion_matrix_threshold_050.png) |

Machine-readable results are available in
[`metrics_summary.json`](metrics/metrics_summary.json),
[`threshold_sweep.csv`](metrics/threshold_sweep.csv), and the two confusion
matrix CSV files in this directory.

## Corrected Grad-CAM method

The corrected Grad-CAM panels target `stage3_last_norm1`, the `norm1` module
in the last block of Swin's third transformer stage. Torchvision Swin
activations are channels-last (`[batch, height, width, channels]`), so channel
weights are calculated by averaging gradients across the height and width
dimensions.

One validation image was selected for each of the 20 animal classes. Each panel
shows the original image, the YOLO crop when available, the exact classifier
input, the Grad-CAM heatmap, and the activation-weighted overlay. The
explanation target is the classifier's predicted class.

- 20 class samples requested
- 17 used the normal YOLO-crop path
- 3 used raw fallback because YOLO missed the animal: Sphynx, Boxer, and
  Dalmatian
- all 20 selected samples were classified correctly
- the sample is qualitative and deliberately balanced by class; it is not an
  additional accuracy estimate

## Corrected per-class Grad-CAM gallery

| Class | Corrected `stage3_last_norm1` panel |
| --- | --- |
| Abyssinian | ![Abyssinian](panels/stage3/00_abyssinian.jpg) |
| Bengal | ![Bengal](panels/stage3/01_bengal.jpg) |
| Birman | ![Birman](panels/stage3/02_birman.jpg) |
| Bombay | ![Bombay](panels/stage3/03_bombay.jpg) |
| British Shorthair | ![British Shorthair](panels/stage3/04_british_shorthair.jpg) |
| Maine Coon | ![Maine Coon](panels/stage3/05_maine_coon.jpg) |
| Ragdoll | ![Ragdoll](panels/stage3/06_ragdoll.jpg) |
| Sphynx, raw fallback | ![Sphynx raw fallback](panels/stage3/07_sphynx_raw_fallback.jpg) |
| Tabby | ![Tabby](panels/stage3/08_tabby.jpg) |
| Tiger Cat | ![Tiger Cat](panels/stage3/09_tiger_cat.jpg) |
| Beagle | ![Beagle](panels/stage3/10_beagle.jpg) |
| Pug | ![Pug](panels/stage3/11_pug.jpg) |
| Boxer, raw fallback | ![Boxer raw fallback](panels/stage3/12_boxer_raw_fallback.jpg) |
| Shiba Inu | ![Shiba Inu](panels/stage3/13_shiba_inu.jpg) |
| Samoyed | ![Samoyed](panels/stage3/14_samoyed.jpg) |
| Golden Retriever | ![Golden Retriever](panels/stage3/15_golden_retriever.jpg) |
| German Shepherd | ![German Shepherd](panels/stage3/16_german_shepherd.jpg) |
| Siberian Husky | ![Siberian Husky](panels/stage3/17_siberian_husky.jpg) |
| Dalmatian, raw fallback | ![Dalmatian raw fallback](panels/stage3/18_dalmatian_raw_fallback.jpg) |
| Rottweiler | ![Rottweiler](panels/stage3/19_rottweiler.jpg) |

## Qualitative observations

The corrected maps are more plausible than the legacy final-stage maps: the
dominant artifact is no longer an almost universal high-activation outer ring.
Several panels emphasize animal heads, faces, fur texture, bodies, or
breed-specific regions. However, some panels still place meaningful activation
on context such as bedding, snow, background color, or crop boundaries. This is
expected for a coarse class-discriminative method and may also reveal that the
classifier learned dataset-specific context shortcuts.

The raw-fallback examples are especially useful for separating detector errors
from classifier errors. In the normal YOLO-gated pipeline, Sphynx, Boxer, and
Dalmatian were missed by the detector. When the classifier was run directly on
the raw image, it still predicted the correct breed with high confidence. That
means those failures were detector-gate failures, not Swin classifier failures.

## Official test follow-up

The same corrected `stage3_last_norm1` visualization code was rerun on the 143
labelled official-test images. Aggregate metrics and confusion matrices are
documented in the
[official-test stage3 report](../swin_tiny_official_test_stage3/README.md).

The full private artifact archives produced during this analysis were:

- `swin_official_test_stage3_full.tar.gz`: 143 official-test panels, overlays,
  heatmaps, metadata, and evaluation files.
- `swin_gradcam_stage3_20.tar.gz`: the corrected 20-class qualitative demo
  panels, overlays, heatmaps, metadata, and raw heatmap arrays.

## Limitations

- The 906-image validation split was used both for model selection and
  threshold calibration; there is no independent held-out validation estimate
  here.
- The experiment uses only the 4,534 available images, not the complete
  5,432-row manifest.
- One intentionally selected image per class is useful for inspection but is
  not a representative quantitative Grad-CAM evaluation.
- Grad-CAM is a coarse localization diagnostic, not a causal explanation or
  object segmentation mask.
- Raw fallback changes the preprocessing path and should be reported
  separately from the normal YOLO-gated inference path.

## Reproduction

See the main [Swin-Tiny Grad-CAM guide](../../gradcam_swin.md) for local and
Slurm commands. The corrected outputs were generated with confidence threshold
`0.50`, target class `predicted`, and target layer `stage3_last_norm1`.
