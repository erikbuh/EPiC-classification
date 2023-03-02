# EPiC-classification

Packages to install are:
- pytorch
- numpy
- matplotlib
- sklearn
- comet_ml (for logging)

---

Run the training for 10 epochs via:
```bash
PARAMS=(
    --epochs 10
    --dataset_train TRAINING_SET
    --dataset_val VALIDATION_SET
    --dataset_test TEST_SET

)

python train.py "${PARAMS[@]}"
```

---

Find additional settings `python train.py --help` or in the [config.py](config.py) file.

---
The [dataloader](dataloader.py) expects the DATA_SETS to be in an `.npz` format with the `data` stored in the format `[EVENTS, POINTS, FEATURES]` (using zero-padding for variable-sized point clouds) and `labels` (0 or 1).
