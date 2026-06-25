# Custom CNN Epoch Sweep Results

This experiment compares the Custom CNN baseline with different maximum epoch counts: 10, 25, and 50 epochs.

The goal was to observe whether longer training improves validation performance, especially for the reject class. This is important for our project because the model should not only classify known cat and dog breeds, but also learn when an input should be rejected.

## Experimental Setup

All runs used the same Custom CNN baseline architecture and the same filtered dataset split available on the cluster.

Only the number of training epochs was changed:

| Run | Max Epochs |
|---|---:|
| custom_cnn_epochs_10 | 10 |
| custom_cnn_epochs_25 | 25 |
| custom_cnn_epochs_50 | 50 |

The runs were executed on the university GPU cluster through Slurm using the `NvidiaAll` partition.

## Results

| Epochs | Accuracy | Macro F1 | Weighted F1 | Reject Precision | Reject Recall | Reject F1 | False Accepts | False Rejects |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.2125 | 0.2046 | 0.1623 | 1.0000 | 0.0193 | 0.0379 | 254 | 0 |
| 25 | 0.3133 | 0.3216 | 0.2741 | 0.8000 | 0.0772 | 0.1408 | 239 | 5 |
| 50 | 0.4074 | 0.4231 | 0.3822 | 0.8723 | 0.1583 | 0.2680 | 218 | 6 |

## Observations

Increasing the number of epochs consistently improved the validation performance.

The overall accuracy increased from 0.2125 at 10 epochs to 0.4074 at 50 epochs. Macro F1 also improved from 0.2046 to 0.4231, which indicates that longer training helped across classes rather than only improving the most frequent classes.

The reject class also improved. Reject F1 increased from 0.0379 to 0.2680. This means the model became better at identifying samples that should be rejected, although the reject recall is still relatively low.

False accepts decreased from 254 to 218 when training was increased from 10 to 50 epochs. This is important because false accepts are risky in a reject-class setting: they mean the model incorrectly assigns an unknown or reject sample to one of the known breed classes.

## Conclusion

The 50-epoch run performed best among the tested settings. Longer training improved both general classification metrics and reject-class behavior.

For the next experiments, 50 epochs can be used as a stronger Custom CNN baseline. However, reject recall is still limited, so additional improvements such as threshold calibration, stronger architectures, or data balancing should still be considered.

## Note

These results were produced on the available filtered dataset subset on the cluster. The original split referenced some missing image files, so the train and validation splits were filtered to include only images present on the cluster.
