# Fog Training v4 for Long‑Tail CIFAR-100

This repository contains the implementation of **Fog Training v4**, a two-stage entropy‑regularized framework for long‑tail learning on CIFAR‑100‑LT. Stage‑A stabilizes early learning by leveraging predictive uncertainty, and Stage‑B fine‑tunes with adaptive margins and dynamic sample/class weighting.

## Repository Structure

```
fog-training-v4/
├── model.py                      # WRN32x8 architecture definition
├── data_loader.py                # Imbalanced CIFAR data loader
├── evaluation.py                 # Metrics: accuracy, head/mid/tail splits
├── first-stage-training.py       # Stage‑A (Fog) training script
├── second-stage-training.py      # Stage‑B (Fine‑tuning) training script
└── README.md                     # This documentation file
```

## Prerequisites

- Python 3.7+
- PyTorch 1.10+
- torchvision 0.11+
- tqdm 4.64+

Install dependencies via:

```bash
pip install torch torchvision tqdm
```

## Usage

### 1. Stage‑A: Fog Training

This stage performs cross‑entropy with logit adjustment, fog‑temperature scaling, and prior annealing. The checkpoint produced here will be used for Stage‑B.

```bash
python first-stage-training.py \
  --device_id 0 \
  --dataset CIFAR100_LT \
  --bs 64 \
  --lr 0.1 \
  --num_workers 4 \
  --momentum 0.9 \
  --weight_decay 5e-3 \
  --num_classes 100 \
  --num_epochs 200 \
  --imb_type exp \
  --imb_factor 0.01
```

- **Output**: `./exp/stage_1/model.pth` (naïve fog‑phase model)

### 2. Stage‑B: IWB‑Style Fine‑Tuning

Loads the Stage‑A checkpoint, then applies adaptive margin and dynamic weighting.

```bash
python second-stage-training.py \
  --device_id 0 \
  --dataset CIFAR100_LT \
  --bs 64 \
  --lr 0.1 \
  --num_workers 4 \
  --momentum 0.9 \
  --weight_decay 5e-3 \
  --num_classes 100 \
  --num_epochs 80 \
  --pretrain True \
  --imb_type exp \
  --imb_factor 0.01
```

- **Output**: `./exp/stage_2/model.pth` (final IWB‑style model)

## Arguments Reference

All scripts support the following arguments:

```text
--device_id         CUDA device index (default: "0")
--dataset           Dataset name (CIFAR100_LT or CIFAR10_LT)
--bs                Mini‑batch size (default: 64)
--lr                Learning rate (default: 0.1)
--num_workers       DataLoader workers (default: 4)
--momentum          SGD momentum (default: 0.9)
--weight_decay      Weight decay (default: 5e-3)
--num_classes       Number of classes (default: 100)
--num_epochs        Number of epochs (default: 200 for Stage‑A; 80 for Stage‑B)
--pretrain          Load Stage‑A model in Stage‑B (True/False)
--imb_type          Imbalance type: 'exp' or 'none'
--imb_factor        Imbalance factor (default: 1.0)
```

## Contact

For questions or collaboration, please open an issue or contact the author.

