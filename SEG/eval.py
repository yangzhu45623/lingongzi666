import torch
from dice_loss import dice_coeff
import numpy as np

def eval_net(net, loader, device, mode):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    dice_loss = []
    total_acc = []
    for batch in loader:
        imgs, true_masks = batch['image'], batch['mask']
        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            mask_pred = net(imgs)
            if mode.startswith("deeplabv3_resnet50"):
                mask_pred = mask_pred["out"]
            pred = torch.sigmoid(mask_pred)
            pred = (pred > 0.5).float()
            dice_loss.append(dice_coeff(pred, true_masks).item())
            pred_mask = (pred > 0.5).long()
            res = pred_mask == true_masks
            acc = np.mean(res.cpu().numpy())
            total_acc.append(acc)

    return np.mean(np.array(dice_loss)),np.mean(np.array(total_acc))
