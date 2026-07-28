# Included Animals-10 subset

This directory contains the exact 296 Animals-10 reject images used by the
reported experiments: 37 images each from Butterfly, Chicken, Cow, Elephant,
Horse, Sheep, Spider, and Squirrel.

The images were selected from the cleaned Rapidata/Animals-10 dataset:

https://huggingface.co/datasets/Rapidata/Animals-10

Rapidata identifies the original source as:

https://www.kaggle.com/datasets/alessiocorrado99/animals10

Both dataset pages identify the dataset license as GPL-2.0. A copy of the
license is included as `COPYING.GPL-2.0`.

The files are bundled because the preparation pipeline converted the selected
images to RGB JPEG files and assigned deterministic names. Including this small
subset avoids ambiguity from upstream mirrors, file names, or JPEG encoder
versions and permits byte-exact reconstruction of the reported dataset.

Every included file is covered by the expected digest in
`data/metadata/image_sha256.csv`. The normal dataset materialization and
checksum commands verify all 296 files after copying them to `dataset/all`.
