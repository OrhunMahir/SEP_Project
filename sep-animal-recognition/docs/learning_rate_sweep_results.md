# Custom CNN Learning Rate Sweep Results

This experiment compares the Custom CNN baseline with three different learning rates: 0.0005, 0.001, and 0.002.

The goal was to observe how the learning rate affects validation performance and reject-class behavior. The learning rate controls how strongly the model updates its weights after each batch.

## Experimental Setup

All runs used the same Custom CNN architecture and 25 training epochs. Only the learning rate was changed.

| Run | Learning Rate | Image Size | Batch Size | Epochs |
|---|---:|---:|---:|---:|
| custom_cnn_lr_0005 | 0.0005 | 224 | 32 | 25 |
| custom_cnn_lr_001 | 0.001 | 224 | 32 | 25 |
| custom_cnn_lr_002 | 0.002 | 224 | 32 | 25 |

The runs were executed on the university GPU cluster through Slurm using the `NvidiaAll` partition.

## Results

| Learning Rate | Accuracy | Macro F1 | Weighted F1 | Reject Precision | Reject Recall | Reject F1 | False Accepts | False Rejects | Time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0005 | 0.2793 | 0.2822 | 0.2355 | 0.7368 | 0.0541 | 0.1007 | 245 | 5 | 381.8 s |
| 0.0010 | 0.3133 | 0.3216 | 0.2741 | 0.8000 | 0.0772 | 0.1408 | 239 | 5 | 387.2 s |
| 0.0020 | 0.3406 | 0.3466 | 0.3105 | 0.8333 | 0.1158 | 0.2034 | 229 | 6 | 375.4 s |

## Observations

The highest tested learning rate, 0.002, achieved the best validation performance across the main metrics. Accuracy, macro F1, weighted F1, and reject F1 all improved as the learning rate increased.

The reject class also benefited from the higher learning rate. Reject F1 improved from 0.1007 at learning rate 0.0005 to 0.2034 at learning rate 0.002. False accepts also decreased from 245 to 229.

Within this tested range, the higher learning rate did not make training unstable. Instead, it helped the Custom CNN learn faster and reach better validation performance within 25 epochs.

## Conclusion

Learning rate 0.002 performed best among the tested values. It achieved the best overall validation metrics and the best reject-class F1.

For later Custom CNN experiments, 0.002 is a stronger candidate than the original 0.001 baseline. However, values higher than 0.002 were not tested here, so further tuning could explore whether performance continues to improve or starts to become unstable.

## Note

These results were produced on the available filtered dataset subset on the cluster. The original split referenced some missing image files, so the train and validation splits were filtered to include only images present on the cluster.
