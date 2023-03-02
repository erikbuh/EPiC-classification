# EPiC-classification

Permutation-equivariant classification network for variable-sized point clouds based on the discriminator used in the EPiC-GAN. 

Reference: *EPiC-GAN - Equivariant Point Cloud Generation for Particle Jets* ([arXiv:2301:08128]( http://arxiv.org/abs/2301.08128))

---
Packages used:
- pytorch
- numpy
- matplotlib
- sklearn
- comet_ml (for online logging)

---
Run the training for 10 epochs via:
```bash
#!/bin/bash
PARAMS=(
    --epochs 10
    --dataset_train TRAINING_SET
    --dataset_val VALIDATION_SET
    --dataset_test TEST_SET
    --logdir ./
)
python train.py "${PARAMS[@]}"
```

---
Find additional settings via `python train.py --help` or in the [config.py](config.py) file.

---
The [dataloader](dataloader.py) expects the DATA_SETS to be in an `.npz` format with the `data` stored in the format `[EVENTS, POINTS, FEATURES]` (using zero-padding for variable-sized point clouds) and `labels` (0 or 1).
