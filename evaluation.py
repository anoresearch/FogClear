import torch
import torch.nn.functional as F

def evaluate_acc(model, val_loader, prior_log_full, device):
    """Val‐set overall accuracy (used in Stage-A)."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x) + prior_log_full
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

def evaluate(model, val_loader, prior_log_full, head_classes, mid_classes, tail_classes, device):
    """Test‐set overall + head/mid/tail acc (used in Stage-B)."""
    model.eval()
    total = corr = 0
    h_tot = h_corr = 0
    m_tot = m_corr = 0
    t_tot = t_corr = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x) + prior_log_full
            preds = logits.argmax(1)
            for p, label in zip(preds, y):
                total += 1
                corr += (p == label).item()
                if label.item() in head_classes:
                    h_tot += 1
                    h_corr += (p == label).item()
                elif label.item() in mid_classes:
                    m_tot += 1
                    m_corr += (p == label).item()
                else:
                    t_tot += 1
                    t_corr += (p == label).item()
    return (corr/total, h_corr/h_tot, m_corr/m_tot, t_corr/t_tot)
