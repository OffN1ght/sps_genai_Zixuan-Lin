import torch
import torch.nn as nn

@torch.no_grad()
def evaluate_model(model, data_loader, criterion, device='cpu'):
    model.to(device)
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0

    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)

        if isinstance(outputs, tuple):  # VAE
            recon, mu, logvar = outputs
            recon_loss = criterion(recon, inputs)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / inputs.size(0)
            loss = recon_loss + kl
            total_loss += loss.item()
        else:  # 分类
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total += targets.size(0)

    if total > 0:
        acc = total_correct / total
        print(f"[Eval] Loss: {total_loss/len(data_loader):.4f}  Acc: {acc*100:.2f}%")
        return total_loss/len(data_loader), acc
    else:
        print(f"[Eval] Loss: {total_loss/len(data_loader):.4f}")
        return total_loss/len(data_loader), None
