# Data pipeline overview

## Manifest-driven loading

`labels.csv` is the only source of truth for training images and labels. The code does not discover images by scanning folders, which keeps experiments reproducible.

## Reject label conversion

The submission interface requires `-1` for reject. During training, `-1` is converted to output index `20` because PyTorch cross-entropy losses require non-negative class indices. Inference converts `20` back to `-1`.

## Fixed validation split

All models use the same stratified 80/20 split generated with seed `42`. This makes model comparisons fair because every model is evaluated on the same images.

## Balanced training sampler

The reject class has 1,600 images, whereas target classes have roughly 150-200 images. The balanced sampler selects an equal number of examples from each class per epoch so reject samples do not dominate training.

## Deterministic validation

Training images use controlled augmentation. Validation and inference use deterministic resize, center crop, and normalization so evaluation metrics remain stable.
