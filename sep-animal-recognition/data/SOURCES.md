# Dataset sources and exact composition

The fixed manifest in `data/labels.csv` contains 5,432 image paths and is the
source of truth for every reported experiment. Labels `0`–`19` are the target
classes defined in `src/animal_recognition/constants.py`; `-1` is the external
reject label.

## Target classes

- Oxford-IIIT Pet: 200 images each for Abyssinian, Bengal, Birman, Bombay,
  British Shorthair, Maine Coon, Ragdoll, Sphynx, Beagle, Pug, Boxer,
  Shiba Inu, and Samoyed.
  <https://www.robots.ox.ac.uk/~vgg/data/pets/>
- Stanford Dogs: Golden Retriever (150), German Shepherd (152),
  Siberian Husky (192), and Rottweiler (152).
  <http://vision.stanford.edu/aditya86/ImageNetDogs/>
- Wikimedia Commons: Tabby (200), Tiger Cat (200), and Dalmatian (186).
  The exact source URL for every selected image is stored in
  `data/metadata/wikimedia_sources.json`.

## Reject class

The 1,600 reject images combine:

- non-target breeds from Stanford Dogs;
- non-cat/dog classes from Animals-10;
- generic scenes from COCO val2017.

Animals-10: <https://www.kaggle.com/datasets/alessiocorrado99/animals10>

COCO: <https://cocodataset.org/#download>

## Provenance files

- `data/labels.csv`: exact relative path and class label for all 5,432 images.
- `data/metadata/manifest_summary.csv`: count and source for every class.
- `data/metadata/class_counts.json`: expected class counts.
- `data/metadata/wikimedia_sources.json`: exact selected Wikimedia URLs.
- `data/metadata/image_sha256.csv`: expected SHA-256 digest of each final image.

`scripts/materialize_dataset.py` reconstructs the expected `dataset/all`
layout from extracted source datasets and downloads the exact Wikimedia files.
The script copies source images; it does not alter the original datasets.

The public datasets retain their original licenses. Wikimedia files have
per-file licenses; use their source pages for attribution when redistributing
the images.
