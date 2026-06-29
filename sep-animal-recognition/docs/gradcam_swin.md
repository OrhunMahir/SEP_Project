# Swin-Tiny Grad-CAM

`scripts/gradcam_swin.py` explains a Swin-Tiny checkpoint by combining the
checkpoint's exact deterministic preprocessing with a channels-last Grad-CAM
implementation.

## What the tool explains

By default, the tool targets `norm1` in the final block of Swin's third
transformer stage (`stage3_last_norm1`).

For a 224 × 224 classifier input, this layer produces a 14 × 14 spatial
activation map. Torchvision Swin activations use the channels-last layout
`[batch, height, width, channels]`. Grad-CAM channel weights are therefore
calculated by averaging gradients across the height and width dimensions.

An attribution audit was performed on all 143 official-test images. The
previous `last_block_norm1` target produced a systematic outer-ring artifact:

- 79.7% of final-stage heatmap peaks occurred on the outer ring.
- The median edge-to-inner activation ratio was 3.83.
- The final-stage resolution was only 7 × 7.

The new `stage3_last_norm1` target produced:

- 23.1% of heatmap peaks on the outer ring.
- A median edge-to-inner activation ratio of 1.10.
- A higher 14 × 14 spatial resolution.
- Better qualitative agreement with gradient-free occlusion maps.

For this reason, `stage3_last_norm1` is now the default explanation layer.
The previous `last_block_norm1` option is retained only for reproducing legacy
outputs.

When the checkpoint config contains a `yolo` section, the tool reproduces the
configured largest-cat-or-dog padded crop. If YOLO finds no valid target
animal, the standard pipeline rejects the image before Swin runs. The output
then contains a detector-reject panel rather than a fabricated classifier
heatmap.

## Single-image example

Run from the `sep-animal-recognition` directory:

```bash
PYTHONPATH=src python scripts/gradcam_swin.py \
  --config configs/swin_tiny_pretrained_yolo_crop_padded_50ep.json \
  --checkpoint runs/swin_tiny_pretrained_yolo_crop_padded_50ep/best.pt \
  --image /absolute/path/to/example.jpg \
  --confidence-threshold 0.50 \
  --output-dir runs/swin_gradcam
```

The checkpoint is constructed with `weights=None` before its saved state is
loaded. Grad-CAM therefore does not trigger an unnecessary ImageNet weight
download.

## Validation-manifest example

```bash
PYTHONPATH=src python scripts/gradcam_swin.py \
  --config configs/swin_tiny_pretrained_yolo_crop_padded_50ep.json \
  --checkpoint runs/swin_tiny_pretrained_yolo_crop_padded_50ep/best.pt \
  --manifest splits/val_seed42.csv \
  --dataset-root /absolute/path/to/dataset/all \
  --limit 10 \
  --target-class true \
  --output-dir runs/swin_gradcam_true_classes
```

`--target-class predicted` is the default.

Other accepted values are:

- `true`
- `reject`
- an external class index from `-1` to `19`
- the internal reject index `20`
- a class name such as `Siberian_Husky`

Use the threshold selected by validation calibration with
`--confidence-threshold`. If this argument is omitted, the tool reads
`postprocessing.threshold` from the checkpoint config and falls back to `0.0`.

Use `--no-yolo` for an explicit raw-image diagnostic or ablation. This changes
the preprocessing and must not be presented as the standard YOLO-gated path.

## Target-layer alternatives

- `stage3_last_norm1`: default and recommended 14 × 14 explanation layer.
- `last_block_norm1`: legacy 7 × 7 layer that showed a severe outer-ring
  artifact for this checkpoint.
- `last_block_norm2`: final-stage MLP-branch input; 7 × 7 and strongly
  center-focused in the audit.
- `final_norm`: final normalized spatial representation; also 7 × 7 and
  spatially coarse.

Select a layer explicitly with:

```bash
--target-layer stage3_last_norm1
```

## Color interpretation

The heatmap uses the `turbo` color scale:

- Purple and blue indicate low positive evidence.
- Green and yellow indicate intermediate evidence.
- Orange and red indicate high positive evidence.

The colors are not reversed.

Overlay opacity is proportional to the Grad-CAM activation. Low-activation
pixels therefore no longer tint the entire image with a constant heatmap
opacity.

Grad-CAM applies ReLU and displays positive evidence for the selected class.
It does not display negative evidence against that class.

## Generated artifacts

For every classifier explanation, the output directory contains:

- `*_heatmap.png`: colorized Grad-CAM visualization.
- `*_heatmap.npy`: normalized floating-point activation values.
- `*_overlay.png`: activation-weighted heatmap over the exact classifier input.
- `*_panel.png`: original image, YOLO crop, classifier input, heatmap, and
  overlay.
- `*_metadata.json`: target, prediction, confidence, threshold, crop, layer,
  and artifact paths.

`summary.json` records every requested image, including YOLO-gated rejects.

The displayed raw prediction is the classifier argmax. The final prediction
also applies the configured confidence threshold. When a low-confidence
prediction becomes reject, the heatmap still explains the classifier's
highest-scoring raw class. The confidence threshold is a postprocessing rule,
not a learned visual class.

## Interpretation limits

Grad-CAM is an attribution diagnostic, not proof of causal model behavior.

A red region means that the selected activation layer contributed positive
evidence for the explained class. It does not automatically mean that the
region is the only information used by the model.

Where stronger evidence is required, compare Grad-CAM with a gradient-free
method such as occlusion. The official-test attribution audit used this
comparison when selecting the new default layer.

## Slurm example

Submit from the `sep-animal-recognition` directory:

```bash
sbatch \
  --export=ALL,DATASET_ROOT=/absolute/path/to/dataset/all,CONFIDENCE_THRESHOLD=0.50,LIMIT=20 \
  slurm/gradcam_swin.sbatch
```

The job uses `SLURM_SUBMIT_DIR`, so it does not depend on one user's project
path.

Override the default paths when required:

```bash
sbatch \
  --export=ALL,CONFIG_PATH=runs/available_data/config.json,CHECKPOINT_PATH=runs/experiment/best.pt,MANIFEST_PATH=runs/available_data/samples.csv,DATASET_ROOT=/absolute/path/to/dataset/all,OUTPUT_DIR=runs/experiment/gradcam,CONFIDENCE_THRESHOLD=0.50 \
  slurm/gradcam_swin.sbatch
```

Optional variables include:

- `TARGET_CLASS`
- `TARGET_LAYER`
- `YOLO_DEVICE`
- `LIMIT`
- `NO_YOLO`

The default value of `TARGET_LAYER` is `stage3_last_norm1`.

## Reproduced results

The available-data reproduction, validation metrics, threshold calibration,
confusion matrices, and per-class Grad-CAM examples are collected in the
[Swin-Tiny + YOLO Grad-CAM results report](results/swin_tiny_gradcam_available4534/README.md).

Historical panels generated with `last_block_norm1` must be labelled as legacy
and should be regenerated with `stage3_last_norm1` before drawing localization
conclusions.
