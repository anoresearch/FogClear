#!/usr/bin/env python
import os, math, argparse
import torch, torch.nn.functional as F
from tqdm import tqdm
from data_loader import get_dataloaders
from model       import WRN32x8
from evaluation  import evaluate
from utils       import set_seed, load_checkpoint, save_checkpoint
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
    parser.add_argument("--num_epochs",  type=int,   default=config.NUM_EPOCHS_STAGE_B)
    parser.add_argument("--pretrain",    type=lambda x: x.lower()=="true",
                        default=config.PRETRAIN)
    parser.add_argument("--imb_type",    type=str,   default=config.IMB_TYPE)
    parser.add_argument("--imb_factor",  type=float, default=config.IMB_FACTOR)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed()

    # 데이터
    train_loader, val_loader, cls_freq, prior_log, head, mid, tail = \
        get_dataloaders(args.dataset, args.imb_type, args.imb_factor,
                        args.bs, args.num_workers)
    prior_log = prior_log.to(device)

    # 모델 로드
    model = WRN32x8(num_classes=args.num_classes).to(device)
    if args.pretrain:
        load_checkpoint(model, os.path.join(config.STAGE1_DIR, "model.pth"), device)

    bb_params = [p for n,p in model.named_parameters() if "fc" not in n]
    opt_bb = torch.optim.SGD(bb_params, LR=0.01, momentum=args.momentum,
                             weight_decay=args.weight_decay)
    sched_bb = torch.optim.lr_scheduler.CosineAnnealingLR(opt_bb, config.FINE_EPOCHS)
    opt_fc = torch.optim.SGD(model.fc.parameters(), args.lr,
                             momentum=args.momentum, weight_decay=args.weight_decay)
    sched_fc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_fc, args.num_epochs)

    os.makedirs(config.STAGE2_DIR, exist_ok=True)
    U_cls  = torch.zeros(args.num_classes)
    imb_ratio = (cls_freq.max()/cls_freq).to(device)

    for ep in range(1, args.num_epochs+1):
        model.train()
        for x,y in tqdm(train_loader, desc=f"[B {ep:02}]"):
            x,y = x.to(device), y.to(device)
            logits = model(x) + prior_log

            # adaptive margin
            with torch.no_grad():
                pk     = logits.softmax(1)
                md     = (pk.topk(2,1).values[:,0] - pk.topk(2,1).values[:,1]).clamp(0,1)
                delta  = (config.DELTA_MAX * md).unsqueeze(1)
                onehot = F.one_hot(y, args.num_classes).float()
            logits_m = logits - delta * onehot
            logp     = F.log_softmax(logits_m, 1)

            # sample weight
            with torch.no_grad():
                ps = (logits_m / 1.0).softmax(1)
                H  = (-(ps*(ps+1e-7).log()).sum(1) / math.log(args.num_classes))
                E  = -torch.logsumexp(logits_m.detach(),1)
                En = (E - E.mean())/(E.std()+1e-6)
                w_s= torch.clamp(config.ALPHA_S*H + config.BETA_S*En, 0, config.CLIP_W)

            # class-fog weight
            U_cls[y.cpu()] = config.EMA_UC * U_cls[y.cpu()] + (1-config.EMA_UC)*(H.cpu()*math.log(args.num_classes))
            fog_ratio = U_cls / U_cls.mean()
            w_c_all   = (imb_ratio**config.FOG_IMB_ALPHA) * (fog_ratio.to(device)**(1-config.FOG_IMB_ALPHA))
            w_cls     = w_c_all[y] / w_c_all.mean()

            loss = (w_s * w_cls * F.nll_loss(logp, y, reduction="none")).mean()
            loss += config.LCONF * F.relu((md - config.TAU)).mean()

            if ep <= config.FINE_EPOCHS:
                opt_bb.zero_grad()
            opt_fc.zero_grad()
            loss.backward()
            if ep <= config.FINE_EPOCHS: opt_bb.step()
            opt_fc.step()

        sched_fc.step()
        if ep <= config.FINE_EPOCHS: sched_bb.step()

        if ep % 10 == 0 or ep==1:
            te,h,m,t = evaluate(model, val_loader, prior_log, head, mid, tail, device)
            print(f"[B {ep:02}] Test={te*100:.2f}% | Head={h*100:.2f}% | Mid={m*100:.2f}% | Tail={t*100:.2f}%")

    save_checkpoint(model, os.path.join(config.STAGE2_DIR, "model.pth"))
    print("✔ Stage-B 완료:", config.STAGE2_DIR+"/model.pth")

if __name__ == "__main__":
    main()
