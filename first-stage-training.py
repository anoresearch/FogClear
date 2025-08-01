#!/usr/bin/env python
import os, math, random, argparse
import numpy as np
import torch, torch.nn.functional as F
from data_loader import get_dataloaders
from model import WRN32x8
from evaluation import evaluate_acc

def main():
    parser = argparse.ArgumentParser()
    # Hardware
    parser.add_argument("--device_id",    type=str,   default="0")
    # Directories & Dataset
    parser.add_argument("--dataset",      type=str,   default="CIFAR100_LT")
    # Hyper parameters
    parser.add_argument("--bs",           type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=0.1)
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--momentum",      type=float, default=0.9)
    parser.add_argument("--num_classes",  type=int,   default=100)
    parser.add_argument("--encoder_layers", type=int, default=34,
                        help="(unused for WRN32x8)")
    parser.add_argument("--print_fr",     type=int,   default=1)
    parser.add_argument("--num_epochs",   type=int,   default=200)
    parser.add_argument("--pretrain",     type=lambda x: x.lower()=="true",
                        default=False)
    parser.add_argument("--imb_type",     type=str,   default="exp")
    parser.add_argument("--imb_factor",   type=float, default=1.0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # DataLoaders
    train_loader, val_loader, cls_freq, prior_log, *_ = \
        get_dataloaders(args.dataset, args.imb_type, args.imb_factor,
                        args.bs, args.num_workers)
    prior_log = prior_log.to(device)

    # Fog parameters
    MAX_EPOCH = args.num_epochs
    THR_ABS   = 0.10 * math.log(args.num_classes)
    EMA_BETA  = 0.95
    T_KAPPA, H_TARGET = 0.4, 0.8

    # Model & Optim
    model = WRN32x8(num_classes=args.num_classes).to(device)
    bb, fc = [], []
    for n,p in model.named_parameters():
        (fc if "fc" in n else bb).append(p)
    opt_bb = torch.optim.SGD(bb, lr=args.lr, momentum=args.momentum,
                             weight_decay=args.weight_decay)
    opt_fc = torch.optim.SGD(fc, lr=args.lr, momentum=args.momentum,
                             weight_decay=args.weight_decay)
    sched_bb = torch.optim.lr_scheduler.CosineAnnealingLR(opt_bb, MAX_EPOCH)
    sched_fc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_fc, MAX_EPOCH)

    EMA_fog = fog_base = None
    os.makedirs("./exp/stage_1", exist_ok=True)
    for ep in range(1, MAX_EPOCH+1):
        model.train()
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            raw = model(x)
            logits_la = raw + prior_log
            p = F.softmax(logits_la,1)
            Hb = (-(p*(p+1e-7).log()).sum(1).mean().item())
            EMA_fog = Hb if EMA_fog is None else EMA_BETA*EMA_fog + (1-EMA_BETA)*Hb
            if fog_base is None: fog_base = EMA_fog
            T   = 1 + T_KAPPA * max(0,(EMA_fog-H_TARGET)/H_TARGET)
            rho = min(1.0, EMA_fog/fog_base)
            adj = (raw + (1-rho)*prior_log) / T
            loss = F.cross_entropy(adj, y)
            opt_bb.zero_grad(); opt_fc.zero_grad()
            loss.backward()
            opt_bb.step(); opt_fc.step()
        sched_bb.step(); sched_fc.step()

        if ep % args.print_fr == 0:
            val_acc = evaluate_acc(model, val_loader, prior_log, device)
            print(f"[A {ep:03}] Fog={EMA_fog:.3f}|Val={val_acc*100:.2f}%")

    torch.save(model.state_dict(), "./exp/stage_1/model.pth")
    print("✔ Stage-A complete. Model saved to ./exp/stage_1/model.pth")

if __name__=="__main__":
    main()
