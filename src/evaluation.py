import torch
import torch.nn.functional as F

def evaluate_acc(model, val_loader, prior_log, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x) + prior_log.to(device)
            preds  = logits.argmax(1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    return correct / total

def evaluate(model, val_loader, prior_log, head, mid, tail, device):
    model.eval()
    corr = tot = 0
    h_corr = m_corr = t_corr = 0
    h_tot = m_tot = t_tot = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x) + prior_log.to(device)
            preds  = logits.argmax(1)
            for p, lbl in zip(preds, y):
                tot += 1
                corr += (p == lbl).item()
                if lbl.item() in head:
                    h_tot += 1; h_corr += (p == lbl).item()
                elif lbl.item() in mid:
                    m_tot += 1; m_corr += (p == lbl).item()
                else:
                    t_tot += 1; t_corr += (p == lbl).item()
    return (corr/tot, h_corr/h_tot, m_corr/m_tot, t_corr/t_tot)
