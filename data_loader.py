import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from collections import defaultdict

def get_dataloaders(dataset: str, imb_type: str, imb_factor: float,
                    bs: int, num_workers: int, seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    if dataset.upper().startswith("CIFAR100"):
        num_classes, max_per = 100, 500
        ds_cls = datasets.CIFAR100
        norm    = ((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))
    elif dataset.upper().startswith("CIFAR10"):
        num_classes, max_per = 10, 5000
        ds_cls = datasets.CIFAR10
        norm    = ((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))
    else:
        raise ValueError(f"Unsupported dataset {dataset}")

    # imbalanced 샘플 수 계산
    if imb_type == "exp":
        img_num = [int(max_per * (imb_factor**(i/(num_classes-1)))) for i in range(num_classes)]
    else:
        img_num = [max_per] * num_classes

    # transforms
    train_tf = transforms.Compose([
        transforms.RandomCrop(32,4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(*norm)
    ])
    test_tf  = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(*norm)
    ])

    base = ds_cls("./data", train=True, download=True, transform=train_tf)
    test = ds_cls("./data", train=False, download=True, transform=test_tf)

    cls_idx = defaultdict(list)
    for idx, (_, label) in enumerate(base):
        cls_idx[label].append(idx)
    sel = [idx for c,n in enumerate(img_num) for idx in random.sample(cls_idx[c], n)]

    train_loader = DataLoader(Subset(base, sel), batch_size=bs, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(test,               batch_size=bs, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    cls_freq  = torch.tensor(img_num, dtype=torch.float32)
    prior_log = torch.log(cls_freq / cls_freq.sum())

    # head/mid/tail split (CIFAR-100만)
    head = mid = tail = []
    if num_classes > 10:
        sorted_idx = torch.argsort(cls_freq, descending=True)
        third = num_classes // 3
        head  = sorted_idx[:third].tolist()
        mid   = sorted_idx[third:2*third].tolist()
        tail  = sorted_idx[2*third:].tolist()

    return train_loader, val_loader, cls_freq, prior_log, head, mid, tail
