# Fine-Grained Animal Recognition with a Reject Class

This repository contains the training, calibration, ensemble evaluation, and
report artifacts for the SEP animal recognition project. The task is a
21-output classification problem: 20 target cat/dog breeds plus a reject class
for non-target or out-of-distribution images.

## Repository Layout

```text
configs/                 Experiment and data-path configurations
scripts/                 Training, threshold calibration, and evaluation scripts
src/animal_recognition/  Reusable dataset, model, metric, and preprocessing code
splits/                  Fixed seed-42 train/validation split
slurm/                   Cluster job wrappers for the main experiments
docs/                    Result tables, Grad-CAM artifacts, and report support files
```

The final documented systems are:

- `pretrained_50ep`: ResNet-18 pretrained + EfficientNet-B0 pretrained +
  Swin-Tiny pretrained, weights `0.35 / 0.35 / 0.30`, threshold `0.30`.
- `scratch_100ep`: Custom CNN 8-conv + ResNet-18 scratch + EfficientNet-B0
  scratch, weights `0.45 / 0.25 / 0.30`, threshold `0.32`.

## Environment

Python 3.10 or newer is recommended. Create a virtual environment from the
repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CUDA training, install the PyTorch build that matches the target machine
before installing the rest of the requirements. For example, on a CUDA 12.1
machine:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

Ultralytics downloads `yolov8n.pt` automatically on first use if it is not
already present.

## Data

The training dataset is expected as a manifest plus image root:

```text
dataset/
  labels.csv
  all/
    <class_or_source_subfolders>/<image files>
```

`labels.csv` must contain:

```text
filename,label
relative/path/to/image.jpg,0
...
```

Labels `0` through `19` are the target breeds in `src/animal_recognition/constants.py`.
The external reject label is `-1`; internally it is converted to output index
`20` during training.

Set the local paths in `configs/data_paths.json` before running scripts:

```json
{
  "train_manifest": "/absolute/path/to/dataset/labels.csv",
  "train_image_root": "/absolute/path/to/dataset/all",
  "official_validation_root": "/absolute/path/to/official_validation_images"
}
```

The repository already includes the fixed split files used for the reported
experiments:

```text
splits/train_seed42.csv
splits/val_seed42.csv
```

To recreate them from the same full manifest:

```bash
python scripts/create_split.py \
  --manifest /absolute/path/to/dataset/labels.csv \
  --output-dir splits \
  --seed 42 \
  --validation-fraction 0.20
```

## Reproducing the Main Pipeline

Use the repository root as the working directory and expose the local package:

```bash
export PYTHONPATH="$PWD/src"
```

### 1. Prepare YOLO Padded Crops

The final experiments use largest cat/dog YOLO crops with square 10% padding and
raw-image fallback. Run crop preparation for each final config family before
training:

```bash
python scripts/prepare_yolo_crops.py \
  --config configs/custom_cnn_yolo_crop_padded_medium_aug_100ep.json \
  --detector yolov8n.pt \
  --detector-device 0 \
  --confidence 0.25 \
  --padding-fraction 0.10 \
  --square-crop
```

Repeat for the other final configs whose `data.image_root` points to a crop
cache, for example:

```bash
python scripts/prepare_yolo_crops.py --config configs/resnet18_yolo_crop_padded_100ep.json --padding-fraction 0.10 --square-crop
python scripts/prepare_yolo_crops.py --config configs/efficientnet_b0_yolo_crop_padded_100ep.json --padding-fraction 0.10 --square-crop
python scripts/prepare_yolo_crops.py --config configs/resnet18_pretrained_yolo_crop_padded_50ep.json --padding-fraction 0.10 --square-crop
python scripts/prepare_yolo_crops.py --config configs/efficientnet_b0_pretrained_yolo_crop_padded_50ep.json --padding-fraction 0.10 --square-crop
python scripts/prepare_yolo_crops.py --config configs/swin_tiny_pretrained_yolo_crop_padded_50ep.json --padding-fraction 0.10 --square-crop
```

### 2. Train the Final Individual Models

```bash
python scripts/train_baseline.py --config configs/custom_cnn_yolo_crop_padded_medium_aug_100ep.json --device cuda
python scripts/train_baseline.py --config configs/resnet18_yolo_crop_padded_100ep.json --device cuda
python scripts/train_baseline.py --config configs/efficientnet_b0_yolo_crop_padded_100ep.json --device cuda

python scripts/train_baseline.py --config configs/resnet18_pretrained_yolo_crop_padded_50ep.json --device cuda
python scripts/train_baseline.py --config configs/efficientnet_b0_pretrained_yolo_crop_padded_50ep.json --device cuda
python scripts/train_baseline.py --config configs/swin_tiny_pretrained_yolo_crop_padded_50ep.json --device cuda
```

Outputs are written under each config's `output_dir`, including:

- `best.pt`
- `history.json`
- `summary.json`
- `validation_per_class_metrics.csv`
- `validation_per_class_metrics.json`

### 3. Calibrate Confidence Thresholds

```bash
python scripts/calibrate_threshold.py \
  --config configs/resnet18_pretrained_yolo_crop_padded_50ep.json \
  --selection-metric accuracy \
  --device cuda
```

Repeat for any individual model where a calibrated threshold table is required.

### 4. Evaluate Internal Validation Ensembles

Scratch 100-epoch ensemble:

```bash
python scripts/evaluate_ensemble.py \
  --config configs/custom_cnn_yolo_crop_padded_medium_aug_100ep.json \
  --config configs/resnet18_yolo_crop_padded_100ep.json \
  --config configs/efficientnet_b0_yolo_crop_padded_100ep.json \
  --weights 0.45,0.25,0.30 \
  --threshold-start 0.32 \
  --threshold-end 0.32 \
  --threshold-step 0.01 \
  --selection-metric accuracy \
  --output-dir runs/ensemble_custom_cnn_resnet18_efficientnet_b0_scratch_padded_100ep \
  --device cuda
```

Pretrained 50-epoch ensemble:

```bash
python scripts/evaluate_ensemble.py \
  --config configs/resnet18_pretrained_yolo_crop_padded_50ep.json \
  --config configs/efficientnet_b0_pretrained_yolo_crop_padded_50ep.json \
  --config configs/swin_tiny_pretrained_yolo_crop_padded_50ep.json \
  --weights 0.35,0.35,0.30 \
  --threshold-start 0.30 \
  --threshold-end 0.30 \
  --threshold-step 0.01 \
  --selection-metric accuracy \
  --output-dir runs/ensemble_resnet18_efficientnet_b0_swin_tiny_pretrained_padded_50ep \
  --device cuda
```

The documented validation results are in:

```text
docs/internal_ensemble_validation_results.md
docs/internal_ensemble_per_class_metrics.md
```

### 5. Evaluate on the Instructor-Provided Labelled Evaluation Set

The official image folder must contain a flat image directory and `labels.csv`.
The repository also provides a course-style root `inference.py` with a `Model`
class whose `forward` method accepts one PIL image and returns `-1` or one of
the 20 class indices. By default it uses the `pretrained_50ep` ensemble; set
`ANIMAL_RECOGNITION_PRESET=scratch_100ep` to use the scratch ensemble.

```bash
python scripts/evaluate_official_ensemble.py \
  --image-folder /absolute/path/to/official_validation_images \
  --preset pretrained_50ep \
  --output-dir runs/official_pretrained_50ep_ensemble_cpu \
  --device cpu \
  --detector-device cpu

python scripts/evaluate_official_ensemble.py \
  --image-folder /absolute/path/to/official_validation_images \
  --preset scratch_100ep \
  --output-dir runs/official_scratch_100ep_ensemble_cpu \
  --device cpu \
  --detector-device cpu
```

The documented official evaluation results are in:

```text
docs/official_test_set_results.md
```

### 6. Generate Custom CNN Grad-CAM Visualizations

Use the same checkpoint/config pair as the final Custom CNN model:

```bash
python gradcam_custom_cnn/generate_custom_cnn_gradcam.py \
  --project-root /absolute/path/to/sep-animal-recognition \
  --checkpoint runs/custom_cnn_yolo_crop_padded_medium_aug_100ep/best.pt \
  --config configs/custom_cnn_yolo_crop_padded_medium_aug_100ep.json \
  --dataset-root /absolute/path/to/dataset/all \
  --labels-csv /absolute/path/to/dataset/labels.csv \
  --output-dir gradcam_outputs/custom_cnn_final_100ep \
  --preprocess yolo-crop
```

For folder-based datasets, omit `--labels-csv` and pass the class-folder root
as `--dataset-root`. The script creates one Grad-CAM image per target breed and
`gradcam_summary.json`; by default it skips the reject class.

## Report Artifacts

- Internal validation aggregate metrics:
  `docs/internal_ensemble_validation_results.md`
- Internal validation per-class metrics:
  `docs/internal_ensemble_per_class_metrics.md`
- Official labelled evaluation metrics:
  `docs/official_test_set_results.md`
- Grad-CAM artifacts:
  `docs/gradcam_outputs/`

## Notes on Large Artifacts

The `.gitignore` excludes generated runs and checkpoints by default. To fully
reproduce the numerical results, place the trained checkpoints at the paths
defined by the configs, or rerun the training commands above. The final
checkpoints used in the report were:

- Scratch ensemble: Custom CNN epoch 84, ResNet-18 epoch 85, EfficientNet-B0
  epoch 94.
- Pretrained ensemble: ResNet-18 epoch 47, EfficientNet-B0 epoch 30, Swin-Tiny
  epoch 29.
