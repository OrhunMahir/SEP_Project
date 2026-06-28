# Custom CNN Grad-CAM

Standalone Grad-CAM helper for the final Custom CNN checkpoint. It does not edit
the project training, inference, or dataset files.

## Recommended Usage

Use the same checkpoint/config pair as the final Custom CNN model:

```bash
cd /Users/orhun/Desktop/SEP_Project/sep-animal-recognition

python gradcam_custom_cnn/generate_custom_cnn_gradcam.py \
  --project-root /Users/orhun/Desktop/SEP_Project/sep-animal-recognition \
  --checkpoint runs/custom_cnn_yolo_crop_padded_medium_aug_100ep/best.pt \
  --config configs/custom_cnn_yolo_crop_padded_medium_aug_100ep.json \
  --dataset-root /ABS/PATH/TO/EXTERNAL/DATASET/all \
  --labels-csv /ABS/PATH/TO/EXTERNAL/DATASET/labels.csv \
  --output-dir gradcam_outputs/custom_cnn_final_100ep \
  --preprocess yolo-crop
```

The `labels.csv` file should contain:

```csv
filename,label
image_001.jpg,0
image_002.jpg,1
```

The script creates one Grad-CAM image per breed and a `gradcam_summary.json`.
By default, it skips the reject class and generates the 20 breed visualizations.

## Folder-Based Dataset

If the external dataset is organized as one folder per class, `labels.csv` is not
needed:

```bash
python gradcam_custom_cnn/generate_custom_cnn_gradcam.py \
  --project-root /Users/orhun/Desktop/SEP_Project/sep-animal-recognition \
  --checkpoint runs/custom_cnn_yolo_crop_padded_medium_aug_100ep/best.pt \
  --config configs/custom_cnn_yolo_crop_padded_medium_aug_100ep.json \
  --dataset-root /ABS/PATH/TO/EXTERNAL/DATASET \
  --output-dir gradcam_outputs/custom_cnn_final_100ep \
  --preprocess yolo-crop
```

Folder names can use underscores or spaces, for example `British_Shorthair` or
`British Shorthair`.

## Notes

- Target Grad-CAM layer is `features.3.layers.3`, the last convolution in the
  final Custom CNN block.
- The script chooses a representative image for each class. It prefers correctly
  predicted images with high target-class confidence, then falls back to the
  highest target-class confidence example if no correct prediction is found.
- If the final model was trained on YOLO cropped images, the most faithful
  explanation is to run Grad-CAM on similarly cropped inputs.
- `--preprocess yolo-crop` uses the same largest cat/dog crop idea as the
  training pipeline. If YOLO cannot find an animal, the image is skipped by
  default. Use `--on-yolo-miss center-crop` only if you explicitly want fallback
  images included.
