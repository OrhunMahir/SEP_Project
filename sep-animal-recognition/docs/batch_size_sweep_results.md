# Custom CNN Batch Size Sweep Results

This experiment compares the Custom CNN baseline with three different batch sizes: 16, 32, and 64.

The goal was to observe how batch size affects validation performance, training time, and reject-class behavior. Batch size controls how many images are processed before each model update.

## Experimental Setup

All runs used the same Custom CNN architecture and 25 training epochs. Only the batch size was changed.

| Run | Batch Size | Epochs |
|---|---:|---:|
| custom_cnn_batch_16 | 16 | 25 |
| custom_cnn_batch_32 | 32 | 25 |
| custom_cnn_batch_64 | 64 | 25 |

The runs were executed on the university GPU cluster through Slurm using the `NvidiaAll` partition.

## Results

| Batch Size | Accuracy | Macro F1 | Weighted F1 | Reject Precision | Reject Recall | Reject F1 | False Accepts | False Rejects | Time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.3165 | 0.3214 | 0.2811 | 0.8000 | 0.0927 | 0.1661 | 235 | 6 | 361.9 s |
| 32 | 0.3133 | 0.3216 | 0.2741 | 0.8000 | 0.0772 | 0.1408 | 239 | 5 | 349.0 s |
| 64 | 0.2979 | 0.2939 | 0.2608 | 0.8571 | 0.0927 | 0.1672 | 235 | 4 | 380.8 s |

## Observations

Batch sizes 16 and 32 produced very similar overall performance. Batch size 32 achieved the highest macro F1 score, but only by a very small margin.

Batch size 16 achieved the best weighted F1 score and also reduced false accepts compared with batch size 32. This suggests that the smaller batch may provide a slightly better balance between overall classification and reject-class behavior.

Batch size 64 achieved the highest reject F1 score, but its overall macro F1 and weighted F1 were lower than the other two settings. It was also the slowest run in this experiment.

## Conclusion

Batch size 16 appears to be the most balanced setting among the tested values. It achieved the best weighted F1, a reject F1 very close to batch size 64, and fewer false accepts than batch size 32.

Batch size 32 remains a reasonable default because it had nearly identical macro F1 and slightly faster runtime. However, for reject-class reliability, batch size 16 is a better candidate for later experiments.

## Note

These results were produced on the available filtered dataset subset on the cluster. The original split referenced some missing image files, so the train and validation splits were filtered to include only images present on the cluster.
