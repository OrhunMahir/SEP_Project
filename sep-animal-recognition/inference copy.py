# Review notes before changing the baseline:
#
# - This file is still the original evaluation wrapper with a random placeholder
#   model. The trained CustomCNN is not loaded here yet, so running this script
#   would not evaluate the baseline we trained.
# - The filename should probably be changed back to `inference.py` for the final
#   hand-in, unless the provided course script is kept separately and this file is
#   only a local copy.
# - The model checkpoint path, preprocessing transform, and internal reject label
#   conversion still need to be wired into `Model.forward`.
# - The training code uses internal label 20 for reject and converts it back to
#   -1 elsewhere. Inference must do the same conversion before returning a value.
# - Some split rows currently look suspicious: a few `reject_stanford__...`
#   examples include target dog breeds such as beagle, pug, boxer, and Samoyed
#   but are labeled as -1. These should be checked before trusting reject metrics.
# - `configs/data_paths.json` contains local machine paths. That is fine for a
#   quick run, but final code should use documented arguments or project-relative
#   paths so another machine can reproduce the results.
# - No trained weights or baseline summary files are present in `runs/` in the
#   current checkout. Before submission, keep the selected checkpoint and the
#   validation history together with the code.
# - The current metrics are useful for model selection, but the final report will
#   also need per-class precision, recall, and F1 tables, not just the aggregate
#   macro/weighted scores.
#
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
import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

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
    """TODO (students): replace this with your own model.

    Contract: given a PIL image, return a class index in {-1, 0, ..., 19}.
    The placeholder below is a uniform random guesser so the script runs.
    """

    def forward(self, image: Image.Image) -> int:
        return random.randint(-1, NUM_CLASSES - 1)


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
