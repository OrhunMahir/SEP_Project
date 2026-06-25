# Custom CNN Image Size Sweep Results

This experiment compares the Custom CNN baseline with three different input image sizes: 160, 224, and 288.

The goal was to observe whether increasing the image resolution improves validation performance. Larger images may preserve more visual detail, but they also increase computation time and may not help if the model capacity is limited.

## Experimental Setup

All runs used the same Custom CNN architecture and 25 training epochs. Only the input image size was changed.

| Run | Image Size | Batch Size | Epochs |
|---|---:|---:|---:|
| custom_cnn_image_160 | 160 | 32 | 25 |
| custom_cnn_image_224 | 224 | 32 | 25 |
| custom_cnn_image_288 | 288 | 32 | 25 |

The runs were executed on the university GPU cluster through Slurm using the `NvidiaAll` partition.

## Results

| Image Size | Accuracy | Macro F1 | Weighted F1 | Reject Precision | Reject Recall | Reject F1 | False Accepts | False Rejects | Time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 160 | 0.3505 | 0.3596 | 0.3126 | 0.7576 | 0.0965 | 0.1712 | 234 | 8 | 329.8 s |
| 224 | 0.3133 | 0.3216 | 0.2741 | 0.8000 | 0.0772 | 0.1408 | 239 | 5 | 342.2 s |
| 288 | 0.2892 | 0.2939 | 0.2493 | 0.9444 | 0.0656 | 0.1227 | 242 | 1 | 377.6 s |

## Observations

The smallest image size, 160, achieved the best validation performance across the main metrics. It had the highest accuracy, macro F1, weighted F1, and reject F1.

Increasing the image size from 160 to 224 reduced the overall performance. Increasing it further to 288 reduced the performance even more and also increased runtime.

Although image size 288 achieved the highest reject precision, its reject recall was the lowest. This means that when the model predicted reject, it was usually correct, but it missed many reject samples.

## Conclusion

Image size 160 performed best among the tested settings. For this Custom CNN baseline, larger input images did not improve performance and made training slower.

This suggests that the current Custom CNN architecture benefits more from faster, lower-resolution inputs than from larger images. A larger image size may be more useful for stronger architectures, but for this baseline model, 160 is the best candidate among the tested image sizes.

## Note

These results were produced on the available filtered dataset subset on the cluster. The original split referenced some missing image files, so the train and validation splits were filtered to include only images present on the cluster.
