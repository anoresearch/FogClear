#!/usr/bin/env python
import os, math, random, argparse
import numpy as np
import torch, torch.nn.functional as F
from tqdm import tqdm
from data_loader import get_dataloaders
from model import WRN32x8
from evaluation import evaluate

def main():
    parser = argparse.ArgumentParser()
    # Hardware & Dataset
    parser.add_argument("--device_id",   type=str,   default="0")
    parser.add_argument("--dataset",     type=str,   default="CIFAR100_LT")
    # Hyper params
    parser.add_argument("--bs",          type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=0.1)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--weight_decay",type=float, default=5e-3)
    parser.add_argument("--momentum",     type=float, default=0.9)
    parser.add_argument("--num_classes", type=int,   default=100)
    parser.add_argument("--print_fr",    type=int,   default=1)
    parser.add_argument("--num_epochs",  type=int,   default=200)
    parser.add_argument("--pretrain",    type=lambda x: x.lower()=="true",
                        default=False)
    parser.add_argument("--imb_type",    type=str,   default="exp")
    parser.add_argument("--imb_factor",  type=float, default=1.0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Data
    train_loader, val_loader, cls_freq, prior_log, head, mid, tail = \
        get_dataloaders(args.dataset, args.imb_type, args.imb_factor,
                        args.bs, args.num_workers)
    prior_log = prior_log.to(device)

    # Stage-B params
    LR_BB_FINE, LR_FC = 0.01, args.lr
    FINE_EPOCHS, CLEAR_EPOCHS = 10, 80
    ALPHA_S,BETA_S,GAMMA_S = 1.0,1.0,0.0
    FOG_IMB_ALPHA, DELTA_MAX = 0.5,0.5
    LCONF, TAU = 0.2,0.7
    CLIP_W, EMA_UC = 3.0, 0.95

    # Model
    model = WRN32x8(num_classes=args.num_classes).to(device)
    if args.pretrain:
        model.load_state_dict(torch.load("./exp/stage_1/model.pth", map_location=device))

    bb_params = [p for n,p in model.named_parameters() if "fc" not in n]
    opt_bb = torch.optim.SGD(bb_params, LR_BB_FINE, momentum=args.momentum,
                             weight_decay=args.weight_decay)
    sched_bb = torch.optim.lr_scheduler.CosineAnnealingLR(opt_bb, FINE_EPOCHS)
    opt_fc = torch.optim.SGD(model.fc.parameters(), args.lr,
                             momentum=args.momentum, weight_decay=args.weight_decay)
    sched_fc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_fc, CLEAR_EPOCHS)

    os.makedirs("./exp/stage_2", exist_ok=True)
    for ep in range(1, CLEAR_EPOCHS+1):
        model.train()
        for x,y in tqdm(train_loader, desc=f"[B {ep:02}]"):
            x,y = x.to(device), y.to(device)
            logits = model(x) + prior_log

            # margin
            with torch.no_grad():
                pk = logits.softmax(1)
                md = (pk.topk(2,1).values[:,0] - pk.topk(2,1).values[:,1]).clamp(0,1)
                delta = (DELTA_MAX * md).unsqueeze(1)
                oh = F.one_hot(y, args.num_classes).float()
            logits_m = logits - delta*oh
            logp = F.log_softmax(logits_m,1)

            # sample weight
            with torch.no_grad():
                ps = (logits_m / 1.0).softmax(1)
                H = (-(ps*(ps+1e-7).log()).sum(1) / math.log(args.num_classes))
                E = -torch.logsumexp(logits_m.detach(),1)
                E_n = (E - E.mean())/(E.std()+1e-6)
                w_s = torch.clamp(ALPHA_S*H + BETA_S*E_n, 0, CLIP_W)

            # class-fog weight
            # ... (기존 EMA_UC 업데이트 로직 유지) ...

            loss = (w_s * F.nll_loss(logp, y, reduction="none")).mean()
            loss += LCONF * F.relu(md - TAU).mean()

            if ep <= FINE_EPOCHS:
                opt_bb.zero_grad()
            opt_fc.zero_grad()
            loss.backward()
            if ep <= FINE_EPOCHS: opt_bb.step()
            opt_fc.step()

        sched_fc.step()
        if ep <= FINE_EPOCHS: sched_bb.step()

        if ep % args.print_fr == 0:
            te,h_acc,m_acc,t_acc = evaluate(model, val_loader,
                                            prior_log, head, mid, tail, device)
            print(f"[B {ep:02}] Test={te*100:.2f}%|Head={h_acc*100:.2f}%|"
                  f"Mid={m_acc*100:.2f}%|Tail={t_acc*100:.2f}%")

    torch.save(model.state_dict(), "./exp/stage_2/model.pth")
    print("✔ Stage-B complete. Model saved to ./exp/stage_2/model.pth")

if __name__=="__main__":
    main()
