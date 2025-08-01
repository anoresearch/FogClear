#!/usr/bin/env python
import os, math, argparse
import torch, torch.nn.functional as F
from data_loader import get_dataloaders
from model       import WRN32x8
from evaluation  import evaluate_acc
from utils       import set_seed, save_checkpoint
import config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id",   type=str,   default=config.DEVICE_ID)
    parser.add_argument("--dataset",     type=str,   default=config.DATASET)
    parser.add_argument("--bs",          type=int,   default=config.BATCH_SIZE)
    parser.add_argument("--lr",          type=float, default=config.LR)
    parser.add_argument("--num_workers", type=int,   default=config.NUM_WORKERS)
    parser.add_argument("--momentum",    type=float, default=config.MOMENTUM)
    parser.add_argument("--weight_decay",type=float, default=config.WEIGHT_DECAY)
    parser.add_argument("--num_classes", type=int,   default=config.NUM_CLASSES)
    parser.add_argument("--num_epochs",  type=int,   default=config.NUM_EPOCHS_STAGE_A)
    parser.add_argument("--imb_type",    type=str,   default=config.IMB_TYPE)
    parser.add_argument("--imb_factor",  type=float, default=config.IMB_FACTOR)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed()

    # 데이터로더
    train_loader, val_loader, cls_freq, prior_log, *_ = \
        get_dataloaders(args.dataset, args.imb_type, args.imb_factor,
                        args.bs, args.num_workers)
    prior_log = prior_log.to(device)

    # 모델·옵티마이저·스케줄러
    model = WRN32x8(num_classes=args.num_classes).to(device)
    bb, fc = [], []
    for n,p in model.named_parameters():
        (bb if "fc" not in n else fc).append(p)
    opt_bb = torch.optim.SGD(bb, lr=args.lr, momentum=args.momentum,
                             weight_decay=args.weight_decay)
    opt_fc = torch.optim.SGD(fc, lr=args.lr, momentum=args.momentum,
                             weight_decay=args.weight_decay)
    sched_bb = torch.optim.lr_scheduler.CosineAnnealingLR(opt_bb, args.num_epochs)
    sched_fc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_fc, args.num_epochs)

    ema_fog = fog_base = None
    os.makedirs(config.STAGE1_DIR, exist_ok=True)
    for ep in range(1, args.num_epochs+1):
        model.train()
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            raw    = model(x)
            logits = raw + prior_log
            p      = F.softmax(logits,1)
            Hb     = (-(p*(p+1e-7).log()).sum(1).mean().item())
            ema_fog = Hb if ema_fog is None else \
                      config.FOG_EMA_BETA*ema_fog + (1-config.FOG_EMA_BETA)*Hb
            if fog_base is None: fog_base = ema_fog

            T   = 1 + config.T_KAPPA * max(0,(ema_fog-config.H_TARGET)/config.H_TARGET)
            rho = min(1.0, ema_fog/fog_base)
            adj = (raw + (1-rho)*prior_log) / T

            loss = F.cross_entropy(adj, y)
            opt_bb.zero_grad(); opt_fc.zero_grad()
            loss.backward()
            opt_bb.step(); opt_fc.step()

        sched_bb.step(); sched_fc.step()
        if ep % 10 == 0 or ep==1:
            acc = evaluate_acc(model, val_loader, prior_log, device)
            print(f"[A {ep:03}] Fog={ema_fog:.3f}|Val={acc*100:.2f}%")

    save_checkpoint(model, os.path.join(config.STAGE1_DIR, "model.pth"))
    print("✔ Stage-A 완료:", config.STAGE1_DIR+"/model.pth")

if __name__ == "__main__":
    main()
