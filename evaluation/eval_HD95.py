import torch
from tqdm import tqdm
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def _surface_distances(mask_a, mask_b):
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)

    if mask_a.sum() == 0 or mask_b.sum() == 0:
        return np.array([], dtype=np.float32)

    structure = np.ones((3, 3), dtype=bool)
    surface_a = mask_a ^ binary_erosion(mask_a, structure=structure, border_value=0)
    surface_b = mask_b ^ binary_erosion(mask_b, structure=structure, border_value=0)

    if not surface_a.any() or not surface_b.any():
        return np.array([], dtype=np.float32)

    distance_map = distance_transform_edt(~surface_b)
    return distance_map[surface_a].astype(np.float32)

def compute_hd95(pred, target):
    """
    Compute 95th percentile Hausdorff Distance (HD95) for binary masks.
    """
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    if pred.ndim == 4:
        pred = pred[:, 0, :, :]
        target = target[:, 0, :, :]
    elif pred.ndim == 2:
        pred = pred[None, ...]
        target = target[None, ...]

    fallback_hd95 = float(np.hypot(pred.shape[-2] - 1, pred.shape[-1] - 1))
    hd95_values = []

    for i in range(pred.shape[0]):
        pred_mask = pred[i] > 0.5
        target_mask = target[i] > 0.5

        if pred_mask.sum() == 0 and target_mask.sum() == 0:
            hd95_values.append(0.0)
            continue

        if pred_mask.sum() == 0 or target_mask.sum() == 0:
            hd95_values.append(fallback_hd95)
            continue

        distances = np.concatenate([
            _surface_distances(pred_mask, target_mask),
            _surface_distances(target_mask, pred_mask),
        ])

        if distances.size == 0:
            hd95_values.append(fallback_hd95)
        else:
            hd95_values.append(float(np.percentile(distances, 95)))

    return float(np.nanpercentile(hd95_values, 95))


def eval_net_hd95(net, loader, device):
    """Evaluate network only using HD95"""
    net.eval()
    mask_type = torch.float32
    n_val = len(loader)
    total_hd95 = 0

    with tqdm(total=n_val, desc='Validation HD95', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['label']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)
                if isinstance(mask_pred, (list, tuple)):
                    mask_pred = sum(mask_pred)
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()

            hd95 = compute_hd95(pred, true_masks)
            total_hd95 += hd95

            pbar.update()

    net.train()

    return {'hd95': total_hd95 / max(n_val, 1)}
