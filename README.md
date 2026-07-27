# Cat and Dog Breed Recognition with a Reject Class

Final submission for the LMU Software Development Practical: Computer Vision &
Deep Learning.

The system predicts 10 cat breeds, 10 dog breeds, or an explicit reject class
for non-target images. It combines a fixed seed-42 split, balanced epoch
sampling, stochastic training augmentation, YOLOv8n localization with raw-image
fallback, confidence calibration, and weighted softmax ensembles.

The final report is included as `Final_Report.pdf`. All commands below are run
from `sep-animal-recognition/`.

## Submission contents

```text
SEP_Final_Submission/
├── Final_Report.pdf
├── README.md
└── sep-animal-recognition/
    ├── configs/                 final and ablation experiment configurations
    ├── data/                    exact manifest, provenance, and SHA-256 list
    ├── docs/                    recorded validation and held-out results
    ├── scripts/                 data, training, calibration, ensemble, and xAI tools
    ├── splits/                  fixed stratified seed-42 split
    ├── src/animal_recognition/  reusable implementation
    ├── tests/                   manifest and Grad-CAM tests
    ├── inference.py             course-style final inference entry point
    ├── environment.yml
    └── requirements.txt
```

There is exactly one README in the submission. Editor state, temporary logs,
machine-specific cluster wrappers, local absolute paths, and cached crops are
excluded. The six final checkpoints are provided separately through the
verified download link in Section 3 because of their size.

## Reported final systems

| System | Models and global weights | Confidence threshold |
|---|---|---:|
| Scratch 100 epochs | Custom CNN 0.45, ResNet18 0.25, EfficientNet-B0 0.30 | 0.32 |
| Pretrained 50 epochs | ResNet18 0.35, EfficientNet-B0 0.35, Swin-Tiny 0.30 | 0.30 |

All weights are global: a model receives the same weight for every class.
Weights were selected on the fixed internal validation set by simplex grid
search with step 0.05. The confidence threshold was searched from 0.00 to 0.95
with step 0.01. The primary selection metric was validation accuracy, followed
by deterministic tie-breakers implemented in `scripts/evaluate_ensemble.py`.

Expected metrics:

| System | Evaluation set | Accuracy | Macro-F1 | Reject-F1 |
|---|---|---:|---:|---:|
| Scratch ensemble | internal validation, 1,085 images | 0.8230 | 0.8384 | 0.7738 |
| Pretrained ensemble | internal validation, 1,085 images | 0.9548 | 0.9595 | 0.9404 |
| Scratch ensemble | isolated held-out set, 143 images | 0.6853 | 0.6396 | 0.7451 |
| Pretrained ensemble | isolated held-out set, 143 images | 0.7902 | 0.7695 | 0.8605 |

The recorded outputs and per-class tables are in `docs/`.

## 1. Environment

Python 3.10 or newer is supported; Python 3.11 is recommended. A CUDA-capable
GPU is strongly recommended for the six full training runs.

```bash
cd sep-animal-recognition
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
```

On an NVIDIA machine, install the PyTorch build matching the installed CUDA
driver first, then install `requirements.txt`. The official selector is:
<https://pytorch.org/get-started/locally/>.

An equivalent Conda setup is:

```bash
conda env create -f environment.yml
conda activate sep-animal-recognition
export PYTHONPATH="$PWD/src"
```

Ultralytics downloads `yolov8n.pt` on first use. Internet access is therefore
needed once unless that detector checkpoint is supplied locally.

Run the preflight at any time:

```bash
python scripts/verify_submission.py
```

Warnings about absent data or six absent final checkpoints are expected before
the data and checkpoint-download or training steps. Structural failures are
not expected.

## 2. Data

### 2.1 Exact composition

`data/labels.csv` is the immutable source of truth:

- 5,432 images;
- 20 target classes plus reject;
- 200 images for most target classes;
- Golden Retriever 150, German Shepherd 152, Siberian Husky 192,
  Dalmatian 186, and Rottweiler 152;
- 1,600 reject images.

The target data comes from Oxford-IIIT Pet, Stanford Dogs, and fixed Wikimedia
Commons URLs. Reject data comes from non-target Stanford Dogs breeds,
Animals-10 non-cat/dog classes, and COCO val2017 scenes. Complete sources,
counts, and licensing notes are in `data/SOURCES.md`.

### 2.2 Download and extract source datasets

Download and extract:

1. Oxford-IIIT Pet images:
   <https://www.robots.ox.ac.uk/~vgg/data/pets/>
2. Stanford Dogs images:
   <http://vision.stanford.edu/aditya86/ImageNetDogs/>
3. Animals-10:
   <https://www.kaggle.com/datasets/alessiocorrado99/animals10>
4. COCO 2017 validation images:
   <https://cocodataset.org/#download>

The exact Wikimedia images do not need to be downloaded manually; their URLs
are recorded in `data/metadata/wikimedia_sources.json`.

### 2.3 Reconstruct the fixed dataset

Pass the extracted image roots to the materialization script:

```bash
python scripts/materialize_dataset.py \
  --oxford-root /path/to/oxford/images \
  --stanford-root /path/to/stanford/Images \
  --animals10-root /path/to/animals10/raw-img \
  --coco-root /path/to/coco/val2017
```

This copies the selected public-source files to `dataset/all/` and downloads
the exact recorded Wikimedia files. It never modifies the source datasets.

If the complete project dataset was provided directly, place it as:

```text
dataset/
└── all/
    ├── Abyssinian/
    ├── ...
    └── reject/
```

Then verify the layout and every image digest:

```bash
python scripts/materialize_dataset.py --verify-only
python scripts/generate_dataset_checksums.py \
  --image-root dataset/all \
  --verify
python scripts/verify_submission.py --require-data
```

For a dataset stored elsewhere, change `train_image_root` in
`configs/data_paths.json` to an absolute path. The included default is
`dataset/all`.

### 2.4 Fixed split

The included split is stratified 80/20 with seed 42:

```text
splits/train_seed42.csv     4,347 images
splits/val_seed42.csv       1,085 images
```

It can be regenerated from the exact manifest:

```bash
python scripts/create_split.py \
  --manifest data/labels.csv \
  --output-dir splits \
  --seed 42 \
  --validation-fraction 0.20
```

Training uses `BalancedEpochSampler`. The smallest training class contains 120
images, so each epoch samples 120 images from every one of the 21 classes
(2,520 samples per epoch). This prevents the 1,280 training reject images from
dominating optimization.

## 3. Checkpoints and full reproduction

### 3.1 Download the six final checkpoints

All checkpoints used by the scratch and pretrained final ensembles are
available in one external archive:

<https://syncandshare.lrz.de/dl/fiXZG7pDei98oRGVCKkaCZ/Budakci_Erdag_Degirmenci_Mahirogullari_SEP_Checkpoints.zip>

Archive SHA-256:
`22c2d68cd4c6fa755c765ae2c19a8990cf0608cf650f3c7499f9ae20daa17e38`.

Download and extract the archive directly inside `sep-animal-recognition/`:

```bash
curl -L \
  "https://syncandshare.lrz.de/dl/fiXZG7pDei98oRGVCKkaCZ/Budakci_Erdag_Degirmenci_Mahirogullari_SEP_Checkpoints.zip" \
  -o Budakci_Erdag_Degirmenci_Mahirogullari_SEP_Checkpoints.zip
unzip Budakci_Erdag_Degirmenci_Mahirogullari_SEP_Checkpoints.zip -d .
shasum -a 256 -c SHA256SUMS
python scripts/verify_submission.py --require-checkpoints
```

The extracted `runs/` tree matches the paths used by all final configs.
Checkpoint loading does not redownload ImageNet initialization weights.

### 3.2 Retrain all six final models

The full pipeline creates one shared YOLO crop cache, trains six final models,
searches both ensemble weights and thresholds, and optionally evaluates the
isolated course test set.

```bash
chmod +x scripts/reproduce_final.sh
DEVICE=cuda DETECTOR_DEVICE=0 ./scripts/reproduce_final.sh
```

To include the instructor-provided 143-image folder:

```bash
OFFICIAL_TEST_ROOT=/path/to/official_test \
DEVICE=cuda \
DETECTOR_DEVICE=0 \
./scripts/reproduce_final.sh
```

This is a full training reproduction, not a short smoke test. Runtime depends
on GPU model and cluster load. The script recreates all six checkpoints under
`runs/`; downloading the supplied checkpoints is not required for retraining.

## 4. Manual reproduction

### 4.1 Create the shared YOLO cache

YOLOv8n selects the largest cat/dog detection at confidence 0.25. The crop is
made square with 10% context padding. If no valid detection exists, the
unchanged raw image is copied instead of forcing reject.

```bash
python scripts/prepare_yolo_crops.py \
  --config configs/custom_cnn_yolo_crop_padded_medium_aug_100ep.json \
  --detector yolov8n.pt \
  --detector-device 0 \
  --confidence 0.25 \
  --padding-fraction 0.10 \
  --square-crop
```

All six final configs point to the same `runs/yolo_crops_final` cache.

### 4.2 Train the scratch models

```bash
python scripts/train_baseline.py \
  --config configs/custom_cnn_yolo_crop_padded_medium_aug_100ep.json \
  --device cuda
python scripts/train_baseline.py \
  --config configs/resnet18_yolo_crop_padded_100ep.json \
  --device cuda
python scripts/train_baseline.py \
  --config configs/efficientnet_b0_yolo_crop_padded_100ep.json \
  --device cuda
```

### 4.3 Train the pretrained models

```bash
python scripts/train_baseline.py \
  --config configs/resnet18_pretrained_yolo_crop_padded_50ep.json \
  --device cuda
python scripts/train_baseline.py \
  --config configs/efficientnet_b0_pretrained_yolo_crop_padded_50ep.json \
  --device cuda
python scripts/train_baseline.py \
  --config configs/swin_tiny_pretrained_yolo_crop_padded_50ep.json \
  --device cuda
```

Each training run writes:

- `best.pt`, selected by validation macro-F1;
- `config.json`;
- `history.json`;
- `summary.json`;
- validation per-class metrics.

The recorded best epochs were Custom CNN 84, scratch ResNet18 85, scratch
EfficientNet-B0 94, pretrained ResNet18 47, pretrained EfficientNet-B0 30,
and pretrained Swin-Tiny 29. Small numerical differences may occur across
PyTorch/CUDA versions even with the deterministic seed settings.

### 4.4 Reproduce ensemble selection

Scratch:

```bash
python scripts/evaluate_ensemble.py \
  --config configs/custom_cnn_yolo_crop_padded_medium_aug_100ep.json \
  --config configs/resnet18_yolo_crop_padded_100ep.json \
  --config configs/efficientnet_b0_yolo_crop_padded_100ep.json \
  --weight-step 0.05 \
  --threshold-start 0.00 \
  --threshold-end 0.95 \
  --threshold-step 0.01 \
  --selection-metric accuracy \
  --output-dir runs/reproduced_scratch_100ep_ensemble \
  --device cuda
```

Pretrained:

```bash
python scripts/evaluate_ensemble.py \
  --config configs/resnet18_pretrained_yolo_crop_padded_50ep.json \
  --config configs/efficientnet_b0_pretrained_yolo_crop_padded_50ep.json \
  --config configs/swin_tiny_pretrained_yolo_crop_padded_50ep.json \
  --weight-step 0.05 \
  --threshold-start 0.00 \
  --threshold-end 0.95 \
  --threshold-step 0.01 \
  --selection-metric accuracy \
  --output-dir runs/reproduced_pretrained_50ep_ensemble \
  --device cuda
```

The expected selected vectors, in the order shown in each command, are
`0.45,0.25,0.30` at threshold `0.32` and `0.35,0.35,0.30` at threshold
`0.30`.

### 4.5 Evaluate the isolated held-out set

The official folder contains flat image files and `labels.csv`. It was not
used for model, hyperparameter, threshold, or ensemble selection.

```bash
python scripts/evaluate_official_ensemble.py \
  --image-folder /path/to/official_test \
  --preset scratch_100ep \
  --output-dir runs/official_scratch_100ep_ensemble_cpu \
  --device cpu \
  --detector-device cpu

python scripts/evaluate_official_ensemble.py \
  --image-folder /path/to/official_test \
  --preset pretrained_50ep \
  --output-dir runs/official_pretrained_50ep_ensemble_cpu \
  --device cpu \
  --detector-device cpu
```

The observed average CPU inference times were 0.134 s/image for scratch and
0.159 s/image for pretrained; both were below the five-second requirement.

## 5. Inference

After the six checkpoints exist, the course-style `Model` class accepts one
PIL image and returns `-1` or a target label `0`–`19`.

```python
from PIL import Image
from inference import Model

model = Model(preset="pretrained_50ep", device="cpu")
with Image.open("example.jpg") as image:
    prediction = model(image)
print(prediction)
```

Folder evaluation:

```bash
python inference.py \
  --image-folder /path/to/images \
  --preset pretrained_50ep \
  --device cpu
```

Set `ANIMAL_RECOGNITION_PRESET=scratch_100ep` to select the scratch ensemble.

## 6. Grad-CAM

Custom CNN:

```bash
python gradcam_custom_cnn/generate_custom_cnn_gradcam.py \
  --project-root "$PWD" \
  --checkpoint runs/custom_cnn_yolo_crop_padded_medium_aug_100ep/best.pt \
  --config configs/custom_cnn_yolo_crop_padded_medium_aug_100ep.json \
  --dataset-root dataset/all \
  --labels-csv data/labels.csv \
  --output-dir runs/gradcam_custom_cnn \
  --preprocess yolo-crop
```

Swin-Tiny:

```bash
python scripts/gradcam_swin.py \
  --config configs/swin_tiny_pretrained_yolo_crop_padded_50ep.json \
  --checkpoint runs/swin_tiny_pretrained_yolo_crop_padded_50ep/best.pt \
  --manifest splits/val_seed42.csv \
  --dataset-root dataset/all \
  --target-class predicted \
  --target-layer stage3_last_norm1 \
  --output-dir runs/gradcam_swin \
  --device cuda
```

The Swin implementation uses the last normalization layer of stage 3 by
default, producing a 14×14 activation map. The heatmap is normalized, resized
bilinearly to the classifier-input size, colorized with the `turbo` colormap,
and alpha-composited over the image. A YOLO miss follows the same raw-image
fallback used by final inference.

## 7. Reproducing selected ablations

The unified trainer also supports the principal architecture ablations
described in the report:

```bash
python scripts/train_baseline.py --config configs/custom_cnn_baseline.json --device cuda
python scripts/train_baseline.py --config configs/ablations/custom_cnn_8conv.json --device cuda
python scripts/train_baseline.py --config configs/ablations/custom_cnn_12conv.json --device cuda
python scripts/train_baseline.py --config configs/ablations/custom_cnn_8conv_k5.json --device cuda
python scripts/train_baseline.py --config configs/ablations/custom_cnn_8conv_wide_medium_aug.json --device cuda
python scripts/train_baseline.py --config configs/ablations/custom_cnn_8conv_se_medium_aug.json --device cuda
python scripts/train_baseline.py --config configs/ablations/resnet34_scratch.json --device cuda
python scripts/train_baseline.py --config configs/resnet50_scratch.json --device cuda
python scripts/train_baseline.py --config configs/swin_tiny_scratch.json --device cuda
```

Additional YOLO confidence, padding, image-size, dropout, training-length, and
pretraining configurations remain in `configs/`.

Auxiliary analysis tools:

- `scripts/audit_data.py` checks manifest paths, class counts, and missing or
  duplicate files without modifying the dataset.
- `scripts/calibrate_threshold.py` reproduces a single model's reject-threshold
  sweep on the fixed internal validation split.
- `scripts/evaluate_yolo_crop.py` compares raw-image and YOLO-cropped
  validation inference for a fixed classifier.
- `scripts/inference_yolo_swin.py` reproduces the standalone YOLO + Swin-Tiny
  validation pipeline and its recorded metrics.

## 8. Tests

The manifest tests use only the Python standard library:

```bash
python -m unittest tests/test_manifest_integrity.py
```

After installing the ML environment:

```bash
python -m unittest discover -s tests -p "test_*.py"
python -m compileall -q src scripts inference.py gradcam_custom_cnn
```

## 9. Reproducibility clarification

Report Table 3 contains per-class metrics for the 1,085-image internal
validation set. It is not the 143-image held-out table. The held-out target
classes contain only about 4–6 images each, so their individual class
estimates are much less stable.

The held-out accuracy and macro-F1 values in the report match
`docs/official_test_set_results.md`. The same results document also records
reject-class metrics, false accepts, false rejects, and timing measurements.
