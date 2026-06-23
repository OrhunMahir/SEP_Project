"""Evaluation harness for the Fine-grained Animal Recognition project.

We run this script on the held-out test set, so do not change the interface.
Implement your solution as the `Model` below: an `nn.Module` whose `forward`
takes a PIL image and returns a predicted class index, an integer in
{-1, 0, ..., 19}, where -1 means "reject", i.e. no target species is present.
Inside `forward` you are free to do anything you like: run an off-the-shelf
detector, find bounding boxes, crop the largest animal, classify the crop,
decide when to return -1, and so on.

The script reads `labels.csv` from the image folder, with columns
`filename,label`, where `label` is the integer class index from CLASSES (or -1
for confounders / images with no target species). The images themselves are a
flat, numbered set (0001.jpg, 0002.jpg, ...) sitting next to `labels.csv`. The
script runs your model on every image and prints the standard classification
metrics.

    python inference.py --image-folder <folder>
"""

import argparse
import os
from pathlib import Path
import sys

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "sep-animal-recognition" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from animal_recognition.data import evaluation_transform
from animal_recognition.models import build_model

REJECT = -1

# Official class mapping fixed by the chair (index -> species). Train your
# classifier against this exact order so your labels match our evaluation.
CLASSES = [
    "Abyssinian",         #  0
    "Bengal",             #  1
    "Birman",             #  2
    "Bombay",             #  3
    "British_Shorthair",  #  4
    "Maine_Coon",         #  5
    "Ragdoll",            #  6
    "Sphynx",             #  7
    "Tabby",              #  8
    "Tiger_Cat",          #  9
    "Beagle",             # 10
    "Pug",                # 11
    "Boxer",              # 12
    "Shiba_Inu",          # 13
    "Samoyed",            # 14
    "Golden_Retriever",   # 15
    "German_Shepherd",    # 16
    "Siberian_Husky",     # 17
    "Dalmatian",          # 18
    "Rottweiler",         # 19
]
NUM_CLASSES = len(CLASSES)


class Model(nn.Module):
    """ResNet-18 inference with the internal-validation confidence threshold."""

    def __init__(self) -> None:
        super().__init__()
        checkpoint_text = os.environ.get(
            "ANIMAL_RECOGNITION_CHECKPOINT",
            str(PROJECT_ROOT / "checkpoints" / "resnet18_scratch_best.pt"),
        )
        checkpoint_path = Path(checkpoint_text)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                "ResNet-18 checkpoint was not found. Set ANIMAL_RECOGNITION_CHECKPOINT "
                f"or place the file at {PROJECT_ROOT / 'checkpoints' / 'resnet18_scratch_best.pt'}."
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        checkpoint_config = checkpoint["config"]
        if checkpoint.get("model_name") != "resnet18":
            raise ValueError("The final inference harness requires the scratch ResNet-18 checkpoint.")
        self.classifier = build_model(checkpoint_config["model"]).to(self.device)
        self.classifier.load_state_dict(checkpoint["model_state_dict"])
        self.classifier.eval()
        self.transform = evaluation_transform(int(checkpoint_config["data"]["image_size"]))
        self.confidence_threshold = 0.30

    @torch.inference_mode()
    def forward(self, image: Image.Image) -> int:
        """Return a target-class index or -1 when the image should be rejected."""
        image_tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        probabilities = torch.softmax(self.classifier(image_tensor), dim=1)
        confidence, internal_label = probabilities.max(dim=1)
        if float(confidence.item()) < self.confidence_threshold:
            return REJECT
        label = int(internal_label.item())
        return REJECT if label == NUM_CLASSES else label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-folder", type=Path, default="images")
    args = parser.parse_args()

    df = pd.read_csv(args.image_folder / "labels.csv")
    model = Model().eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for filename, label in tqdm(zip(df["filename"], df["label"]), total=len(df)):
            image = Image.open(args.image_folder / filename).convert("RGB")
            pred = model(image)
            y_true.append(int(label))
            y_pred.append(int(pred))

    labels = [REJECT] + list(range(NUM_CLASSES))
    target_names = ["reject(-1)"] + CLASSES
    print(f"\nAccuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, labels=labels,
                                target_names=target_names, digits=3,
                                zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred, labels=labels))
