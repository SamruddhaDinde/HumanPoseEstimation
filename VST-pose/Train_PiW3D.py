"""Training script for VST-Pose on the Person-in-WiFi 3D dataset.

Run from VST-Pose/VST-Pose/:
    python Train_PiW3D.py --config_file config/config_piw3d.yaml

Single-person mode
------------------
When ``single_person_only: true`` is set in the config (and ``max_persons: 1``),
the dataloader filters out multi-person samples.  This restricts VST-Pose to
single-person scenarios, enabling a direct comparison against the
Person-in-WiFi 3D paper's 1-person benchmark of 91.7 mm MPJPE.

Multi-person handling
---------------------
Each batch contains padded keypoint tensors of shape (B, max_persons, 14, 3)
and a float mask of shape (B, max_persons) where 1.0 = real person, 0.0 =
padding.  The masked MSE loss averages only over real-person slots so that
padded rows do not contribute to the gradient.

Speed supervision
-----------------
alpha in the config controls the speed-loss weight.  It is set to 0.0 by
default because Person-in-WiFi 3D provides a single per-segment ground truth
pose rather than a temporal sequence of poses, making a velocity target
unavailable.  Set alpha > 0.0 only if you extend the dataloader to supply
consecutive-frame pose pairs.

Evaluation
----------
MPJPE and PA-MPJPE are computed in millimetres (ground-truth coordinates are
stored in metres and multiplied by 1000 before the metric call).  PCK uses a
joint-order-independent bbox-diagonal scale factor — see ``utils.py``.
"""

alpha = 0.0   # speed-loss weight — keep 0.0 until velocity GT is available

import os
import argparse
import yaml
import torch
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import time
import random

from Feeder.person_in_wifi3d import make_piw3d_dataloader
from Model.PiW3D.conv_STFormer_PiW3D import PiW3DModel
from utils import calulate_error, compute_pck_pckh


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def setup_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """MSE loss averaged only over real (non-padded) person slots.

    Args:
        pred:   (B, max_persons, 14, 3)
        target: (B, max_persons, 14, 3)
        mask:   (B, max_persons)  – 1.0 real, 0.0 padding
    """
    diff = ((pred - target) ** 2).mean(dim=(-1, -2))   # (B, max_persons)
    loss = (diff * mask).sum() / (mask.sum().clamp(min=1e-8))
    return loss


def extract_valid_persons(pred_np, gt_np, mask_np):
    """Gather valid (non-padded) person poses from a batch.

    Args:
        pred_np:  (B, max_persons, 14, 3) numpy array
        gt_np:    (B, max_persons, 14, 3) numpy array
        mask_np:  (B, max_persons) numpy bool/float array

    Returns:
        valid_pred: (N_valid, 14, 3)
        valid_gt:   (N_valid, 14, 3)
    """
    preds_list, gts_list = [], []
    B = pred_np.shape[0]
    for b in range(B):
        valid = mask_np[b].astype(bool)
        if valid.any():
            preds_list.append(pred_np[b][valid])
            gts_list.append(gt_np[b][valid])
    if preds_list:
        return np.concatenate(preds_list, axis=0), np.concatenate(gts_list, axis=0)
    # Edge case: entire batch is padding (should never happen in practice)
    return np.zeros((0, 14, 3), dtype=np.float32), np.zeros((0, 14, 3), dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('VST-Pose  →  Person-in-WiFi 3D')

    parser = argparse.ArgumentParser(description='VST-Pose training on Person-in-WiFi 3D')
    parser.add_argument('--config_file', type=str, default='config/config_piw3d.yaml')
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    setup_seed(config.get('init_rand_seed', 42))

    dataset_root       = config['dataset_root']
    max_persons        = config.get('max_persons', 3)
    num_packets        = config.get('num_packets', 20)
    feature_dim        = config.get('feature_dim', 14)
    dim                = config.get('dim', 128)
    single_person_only = config.get('single_person_only', False)

    if single_person_only:
        print(f"Mode: SINGLE-PERSON ONLY (max_persons={max_persons})")
    else:
        print(f"Mode: MULTI-PERSON (max_persons={max_persons})")

    # ── Dataloaders ──────────────────────────────────────────────────
    train_loader = make_piw3d_dataloader(
        data_root          = dataset_root,
        split              = 'train',
        batch_size         = config['train_loader']['batch_size'],
        num_workers        = config['train_loader'].get('num_workers', 4),
        max_persons        = max_persons,
        single_person_only = single_person_only,
    )
    test_loader = make_piw3d_dataloader(
        data_root          = dataset_root,
        split              = 'test',
        batch_size         = config['test_loader']['batch_size'],
        num_workers        = config['test_loader'].get('num_workers', 4),
        max_persons        = max_persons,
        shuffle            = False,
        single_person_only = single_person_only,
    )

    # ── Device ───────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Model ────────────────────────────────────────────────────────
    model = PiW3DModel(
        num_packets = num_packets,
        feature_dim = feature_dim,
        dim         = dim,
        max_persons = max_persons,
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr    = config['base_learning_rate'],
        betas = (0.9, 0.999),
    )
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.75)

    # ── Output paths ─────────────────────────────────────────────────
    weights_path = os.path.join(config['save_path'], 'person-in-wifi-3d')
    os.makedirs(weights_path, exist_ok=True)
    writer = SummaryWriter('logs/PiW3D_Training')

    num_epochs     = config['total_epoch']
    best_val_loss  = float('inf')
    best_mpjpe     = float('inf')
    best_pampjpe   = float('inf')
    best_pck       = [0.0] * 5   # PCK @ 50/40/30/20/10 %

    for epoch in range(num_epochs):
        t0 = time.time()
        print(f'\nEpoch {epoch + 1}/{num_epochs}')

        # ── Training ─────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Train {epoch + 1}', ncols=100)

        for batch in pbar:
            csi       = batch['csi'].to(device)          # (B, 20, 3, 30, 3)
            kp_target = batch['keypoints'].to(device)    # (B, max_persons, 14, 3)
            mask      = batch['mask'].to(device)         # (B, max_persons)

            optimizer.zero_grad()
            pose_pred, speed_pred = model(csi)           # (B, max_persons, 14, 3) each

            pose_loss = masked_mse_loss(pose_pred, kp_target, mask)

            if alpha > 0.0:
                # Speed target: requires consecutive-frame GT – not available
                # by default.  Only enable if your dataloader returns velocity.
                speed_target = batch.get('speed_target')
                if speed_target is not None:
                    speed_target = speed_target.to(device)
                    speed_loss = masked_mse_loss(speed_pred, speed_target, mask)
                    loss = (1 - alpha) * pose_loss + alpha * speed_loss
                else:
                    loss = pose_loss
            else:
                loss = pose_loss

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        writer.add_scalar('Loss/Train', train_loss, epoch + 1)
        scheduler.step()
        print(f'LR: {scheduler.get_last_lr()[0]:.6f}')

        # ── Validation ───────────────────────────────────────────────
        model.eval()
        val_loss    = 0.0
        mpjpe_list  = []
        pampjpe_list = []
        pck_iter    = [[] for _ in range(5)]

        pbar_val = tqdm(test_loader, desc=f'Test  {epoch + 1}', ncols=100)
        with torch.no_grad():
            for batch in pbar_val:
                csi       = batch['csi'].to(device)
                kp_target = batch['keypoints'].to(device)
                mask      = batch['mask'].to(device)

                pose_pred, _ = model(csi)
                loss = masked_mse_loss(pose_pred, kp_target, mask)
                val_loss += loss.item()

                # ── Per-person metrics ────────────────────────────────
                pred_np = pose_pred.cpu().numpy()   # (B, max_persons, 14, 3)
                gt_np   = kp_target.cpu().numpy()
                mask_np = mask.cpu().numpy()

                valid_pred, valid_gt = extract_valid_persons(pred_np, gt_np, mask_np)

                if valid_pred.shape[0] > 0:
                    # Convert metres → millimetres for reporting.
                    # Ground-truth coordinates in PiW3D are stored in metres.
                    valid_pred_mm = valid_pred * 1000.0
                    valid_gt_mm   = valid_gt   * 1000.0
                    mpjpe, pampjpe, _, _ = calulate_error(valid_pred_mm, valid_gt_mm, align=False)
                    mpjpe_list  += mpjpe.tolist()
                    pampjpe_list += pampjpe.tolist()

                    # PCK is scale-invariant — use original (metre) values.
                    # PCK expects (N, 3, 14) — permute last two dims.
                    vp_pck = valid_pred.transpose(0, 2, 1)   # (N, 3, 14)
                    vg_pck = valid_gt.transpose(0, 2, 1)
                    for i, thr in enumerate([0.5, 0.4, 0.3, 0.2, 0.1]):
                        pck_iter[i].append(
                            compute_pck_pckh(vp_pck, vg_pck, thr,
                                             align=False, dataset='person-in-wifi-3d')
                        )

        val_loss    /= len(test_loader)
        avg_mpjpe   = float(np.mean(mpjpe_list))   if mpjpe_list   else float('nan')
        avg_pampjpe = float(np.mean(pampjpe_list)) if pampjpe_list else float('nan')

        writer.add_scalar('Loss/Validation', val_loss,    epoch + 1)
        writer.add_scalar('Metric/MPJPE',    avg_mpjpe,   epoch + 1)
        writer.add_scalar('Metric/PAMPJPE',  avg_pampjpe, epoch + 1)

        # PCK overall = index 14 (total over all joints) per utils.py convention
        pck_overall = [float(np.mean(pck_iter[i], axis=0)[14]) for i in range(5)]
        for i, pct in enumerate([50, 40, 30, 20, 10]):
            writer.add_scalar(f'PCK/pck{pct}', pck_overall[i], epoch + 1)

        elapsed = time.time() - t0
        print(f'Epoch {epoch + 1} — {elapsed:.1f}s')
        print(f'  Train loss: {train_loss:.4f}   Val loss: {val_loss:.4f}')
        print(f'  MPJPE: {avg_mpjpe:.2f} mm   PA-MPJPE: {avg_pampjpe:.2f} mm')
        print(f'  PCK@50:{pck_overall[0]:.2f}  @40:{pck_overall[1]:.2f}'
              f'  @30:{pck_overall[2]:.2f}  @20:{pck_overall[3]:.2f}'
              f'  @10:{pck_overall[4]:.2f}')

        # ── Save best ────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       os.path.join(weights_path, 'best_val_loss.pth'))
            print(f'  [saved] best val_loss {best_val_loss:.4f}')

        if avg_mpjpe < best_mpjpe:
            best_mpjpe = avg_mpjpe
            torch.save(model.state_dict(),
                       os.path.join(weights_path, 'best_mpjpe.pth'))
            print(f'  [saved] best MPJPE {best_mpjpe:.2f} mm')

        if avg_pampjpe < best_pampjpe:
            best_pampjpe = avg_pampjpe
            torch.save(model.state_dict(),
                       os.path.join(weights_path, 'best_pampjpe.pth'))
            print(f'  [saved] best PA-MPJPE {best_pampjpe:.2f} mm')

        for i, pck_val in enumerate(pck_overall):
            if pck_val > best_pck[i]:
                best_pck[i] = pck_val

        # ── Result summary ───────────────────────────────────────────
        summary_path = os.path.join(weights_path, 'result_summary.txt')
        with open(summary_path, 'w') as f:
            f.write('Best Results Summary\n')
            f.write(f'Best MPJPE:    {best_mpjpe:.4f} mm\n')
            f.write(f'Best PA-MPJPE: {best_pampjpe:.4f} mm\n')
            for i, pct in enumerate([50, 40, 30, 20, 10]):
                f.write(f'Best PCK@{pct}: {best_pck[i]:.4f}\n')

    writer.close()
    print(f'\nTraining complete. Results saved to {weights_path}')
