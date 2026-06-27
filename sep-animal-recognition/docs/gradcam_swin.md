# Swin-Tiny Grad-CAM

`scripts/gradcam_swin.py` explains a Swin-Tiny checkpoint by combining the
checkpoint's exact deterministic preprocessing with a channels-last Grad-CAM
implementation.

## What the tool explains

By default the tool targets `norm1` in the final Swin transformer block. Its
activation layout is `[batch, height, width, channels]`, so the Grad-CAM
channel weights are calculated by averaging gradients across the height and
width dimensions.

When the checkpoint config has a `yolo` section, the tool first reproduces the
configured largest-cat-or-dog padded crop. If YOLO finds no valid target
animal, the pipeline rejects the image before Swin runs. The output then
contains a detector-reject panel instead of a fabricated classifier heatmap.

## Single-image example

Run from `sep-animal-recognition`:

```bash
PYTHONPATH=src python scripts/gradcam_swin.py \
  --config configs/swin_tiny_pretrained_yolo_crop_padded_50ep.json \
  --checkpoint runs/swin_tiny_pretrained_yolo_crop_padded_50ep/best.pt \
  --image /absolute/path/to/example.jpg \
  --confidence-threshold 0.30 \
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

`--target-class predicted` is the default. Other accepted values are `true`,
`reject`, an external class index from `-1` to `19`, the internal reject index
`20`, or a class name such as `Siberian_Husky`.

Use the threshold selected by calibration with `--confidence-threshold`. If
this argument is omitted, the tool reads `postprocessing.threshold` from the
checkpoint config and falls back to `0.0`.

Use `--no-yolo` for an explicit raw-image ablation. This changes the
preprocessing and should not be presented as the normal YOLO-cropped model
pipeline.

## Target-layer alternatives

- `last_block_norm1`: default and recommended Swin explanation layer.
- `last_block_norm2`: explains the input to the last block's MLP branch.
- `final_norm`: explains the final normalized spatial representation.

Select one with `--target-layer`.

## Generated artifacts

For every classifier explanation, the output directory contains:

- `*_heatmap.png`: colorized Grad-CAM map.
- `*_overlay.png`: heatmap blended with the exact classifier input.
- `*_panel.png`: original image, YOLO crop, model input, heatmap, and overlay.
- `*_metadata.json`: target, prediction, confidence, threshold, crop, and file paths.

`summary.json` records every requested image, including YOLO-gated rejects.

The displayed raw prediction is the classifier argmax. The final prediction
also applies the configured confidence threshold. When a low-confidence
prediction becomes reject, the heatmap still explains the classifier's raw
highest-scoring class because the threshold itself is a postprocessing rule,
not a learned visual class.

## Slurm example

Submit from the `sep-animal-recognition` directory. The job uses
`SLURM_SUBMIT_DIR`, so it does not depend on one user's project path.

```bash
sbatch \
  --export=ALL,DATASET_ROOT=/absolute/path/to/dataset/all,CONFIDENCE_THRESHOLD=0.50,LIMIT=20 \
  slurm/gradcam_swin.sbatch
```

The default config, checkpoint, and manifest match the pretrained,
YOLO-cropped Swin-Tiny experiment. Override them when needed:

```bash
sbatch \
  --export=ALL,CONFIG_PATH=runs/available_data/config.json,CHECKPOINT_PATH=runs/experiment/best.pt,MANIFEST_PATH=runs/available_data/samples.csv,DATASET_ROOT=/absolute/path/to/dataset/all,OUTPUT_DIR=runs/experiment/gradcam,CONFIDENCE_THRESHOLD=0.50 \
  slurm/gradcam_swin.sbatch
```

Set `NO_YOLO=1` for an explicit raw-image fallback or ablation. Optional
variables also include `TARGET_CLASS`, `TARGET_LAYER`, `YOLO_DEVICE`, and
`LIMIT`.
