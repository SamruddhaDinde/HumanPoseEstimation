"""Training script for VST-Pose on the Person-in-WiFi 3D dataset.

Run from VST-Pose/VST-Pose/:
    python Train_PiW3D.py --config_file config/config_piw3d.yaml

Train / Val / Test protocol
---------------------------
PiW3D ships with only train and test splits.  We carve a deterministic 10%
slice of the train list as a held-out validation set (controlled by
``val_fraction`` and ``val_seed`` in the config).  Training optimises on the
remaining ~90%; the validation set is used to select the best checkpoint.
The test set is NEVER touched during training — it is reserved for a single
final evaluation run, performed separately by ``Eval_PiW3D.py``.

Single-person mode
------------------
When ``single_person_only: true`` is set in the config (and ``max_persons: 1``),
the dataloader filters out multi-person samples.  This restricts VST-Pose to
single-person scenarios, enabling a direct comparison against the
Person-in-WiFi 3D paper's 1-person benchmark of 91.7 mm MPJPE.

Logging
-------
Three artefacts are written to ``{save_path}/person-in-wifi-3d/``:
  * metrics_per_epoch.csv  — one row per epoch, all metrics, for plotting
  * metrics_per_epoch.txt  — same content, human-readable layout
  * result_summary.txt     — best metrics seen across the whole run
TensorBoard logs land in ``logs/PiW3D_Training/``.

Speed supervision
-----------------
``alpha`` (top-of-file constant) controls the speed-loss weight.  It is set
to 0.0 because Person-in-WiFi 3D provides a single per-segment pose ground
truth, so a velocity target is unavailable.  Set > 0 only if you extend the
dataloader to supply consecutive-frame pose pairs.

Evaluation units
----------------
MPJPE / PA-MPJPE / per-axis errors are reported in millimetres.  PiW3D
ground-truth coordinates are stored in metres, so we multiply by 1000
before the metric calls.  PCK uses a joint-order-independent bbox-diagonal
scale factor (see ``utils.py``) — values are NOT directly comparable to
papers that use a torso scale.
"""

alpha = 0.0   # speed-loss weight — keep 0.0 until velocity GT is available

import os
import argparse
import csv
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
from utils import calulate_error, compute_pck_pckh, calculate_per_axis_error


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
    """MSE loss averaged only over real (non-padded) person slots."""
    diff = ((pred - target) ** 2).mean(dim=(-1, -2))   # (B, max_persons)
    loss = (diff * mask).sum() / (mask.sum().clamp(min=1e-8))
    return loss


def extract_valid_persons(pred_np, gt_np, mask_np):
    """Gather valid (non-padded) person poses from a batch."""
    preds_list, gts_list = [], []
    B = pred_np.shape[0]
    for b in range(B):
        valid = mask_np[b].astype(bool)
        if valid.any():
            preds_list.append(pred_np[b][valid])
            gts_list.append(gt_np[b][valid])
    if preds_list:
        return np.concatenate(preds_list, axis=0), np.concatenate(gts_list, axis=0)
    return np.zeros((0, 14, 3), dtype=np.float32), np.zeros((0, 14, 3), dtype=np.float32)


def evaluate(model, loader, device, desc: str):
    """Run one evaluation pass and return a dict of metrics."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    mpjpe_list = []
    pampjpe_list = []
    per_axis_preds, per_axis_gts = [], []
    pck_iter = [[] for _ in range(5)]

    #pbar = tqdm(loader, desc=desc, ncols=100)
    with torch.no_grad():
        for batch in loader:
            csi       = batch['csi'].to(device)
            kp_target = batch['keypoints'].to(device)
            mask      = batch['mask'].to(device)

            pose_pred, _ = model(csi)
            loss = masked_mse_loss(pose_pred, kp_target, mask)
            total_loss += loss.item()
            n_batches += 1

            pred_np = pose_pred.cpu().numpy()
            gt_np   = kp_target.cpu().numpy()
            mask_np = mask.cpu().numpy()

            valid_pred, valid_gt = extract_valid_persons(pred_np, gt_np, mask_np)
            if valid_pred.shape[0] == 0:
                continue

            valid_pred_mm = valid_pred * 1000.0
            valid_gt_mm   = valid_gt   * 1000.0

            mpjpe, pampjpe, _, _ = calulate_error(valid_pred_mm, valid_gt_mm, align=False)
            mpjpe_list   += mpjpe.tolist()
            pampjpe_list += pampjpe.tolist()

            per_axis_preds.append(valid_pred_mm)
            per_axis_gts.append(valid_gt_mm)

            vp_pck = valid_pred.transpose(0, 2, 1)
            vg_pck = valid_gt.transpose(0, 2, 1)
            for i, thr in enumerate([0.5, 0.4, 0.3, 0.2, 0.1]):
                pck_iter[i].append(
                    compute_pck_pckh(vp_pck, vg_pck, thr,
                                     align=False, dataset='person-in-wifi-3d')
                )

    avg_loss    = total_loss / max(n_batches, 1)
    avg_mpjpe   = float(np.mean(mpjpe_list))   if mpjpe_list   else float('nan')
    avg_pampjpe = float(np.mean(pampjpe_list)) if pampjpe_list else float('nan')

    if per_axis_preds:
        all_p = np.concatenate(per_axis_preds, axis=0)
        all_g = np.concatenate(per_axis_gts, axis=0)
        # axis labels: x → horizontal (h), y → vertical (v), z → depth (d).
        # If your dataset uses a different convention, edit axis_order below.
        axis_errs = calculate_per_axis_error(all_p, all_g, axis_order=('h', 'v', 'd'))
    else:
        axis_errs = {'mpjpe_h': float('nan'),
                     'mpjpe_v': float('nan'),
                     'mpjpe_d': float('nan')}

    pck_overall = [float(np.mean(pck_iter[i], axis=0)[14]) if pck_iter[i] else float('nan')
                   for i in range(5)]

    return {
        'loss':     avg_loss,
        'mpjpe':    avg_mpjpe,
        'pampjpe':  avg_pampjpe,
        'mpjpe_h':  axis_errs['mpjpe_h'],
        'mpjpe_v':  axis_errs['mpjpe_v'],
        'mpjpe_d':  axis_errs['mpjpe_d'],
        'pck50':    pck_overall[0],
        'pck40':    pck_overall[1],
        'pck30':    pck_overall[2],
        'pck20':    pck_overall[3],
        'pck10':    pck_overall[4],
    }


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
    val_fraction       = config.get('val_fraction', 0.10)
    val_seed           = config.get('val_seed', 42)

    if single_person_only:
        print(f"Mode: SINGLE-PERSON ONLY (max_persons={max_persons})")
    else:
        print(f"Mode: MULTI-PERSON (max_persons={max_persons})")
    print(f"Validation split: {val_fraction*100:.0f}% of train (seed={val_seed})")

    # ── Dataloaders (train + val only — test reserved for Eval_PiW3D.py) ──
    train_loader = make_piw3d_dataloader(
        data_root          = dataset_root,
        split              = 'train',
        batch_size         = config['train_loader']['batch_size'],
        num_workers        = config['train_loader'].get('num_workers', 4),
        max_persons        = max_persons,
        single_person_only = single_person_only,
        val_fraction       = val_fraction,
        val_seed           = val_seed,
    )
    val_cfg = config.get('val_loader',
                         config.get('test_loader', {'batch_size': 32, 'num_workers': 4}))
    val_loader = make_piw3d_dataloader(
        data_root          = dataset_root,
        split              = 'val',
        batch_size         = val_cfg['batch_size'],
        num_workers        = val_cfg.get('num_workers', 4),
        max_persons        = max_persons,
        shuffle            = False,
        single_person_only = single_person_only,
        val_fraction       = val_fraction,
        val_seed           = val_seed,
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
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.75)

    # ── Output paths ─────────────────────────────────────────────────
    weights_path = os.path.join(config['save_path'], 'person-in-wifi-3d')
    os.makedirs(weights_path, exist_ok=True)
    writer = SummaryWriter('logs/PiW3D_Training')

    csv_path = os.path.join(weights_path, 'metrics_per_epoch.csv')
    txt_path = os.path.join(weights_path, 'metrics_per_epoch.txt')

    # CSV header — columns chosen for plotting convenience
    csv_columns = [
        'epoch', 'lr', 'train_loss',
        'val_loss', 'val_mpjpe', 'val_pampjpe',
        'val_mpjpe_h', 'val_mpjpe_v', 'val_mpjpe_d',
        'val_pck50', 'val_pck40', 'val_pck30', 'val_pck20', 'val_pck10',
        'epoch_seconds',
    ]
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(csv_columns)

    with open(txt_path, 'w') as f:
        f.write("VST-Pose on Person-in-WiFi-3D — per-epoch metrics\n")
        f.write("=" * 70 + "\n\n")

    # ── Training loop ────────────────────────────────────────────────
    num_epochs    = config['total_epoch']
    best_val_loss = float('inf')
    best_mpjpe    = float('inf')
    best_pampjpe  = float('inf')
    best_mpjpe_h  = float('inf')
    best_mpjpe_v  = float('inf')
    best_mpjpe_d  = float('inf')
    best_pck      = [0.0] * 5

    for epoch in range(num_epochs):
        t0 = time.time()
        print(f'\nEpoch {epoch + 1}/{num_epochs}')

        # ── Training ─────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        n_batches = 0
        #pbar = tqdm(train_loader, desc=f'Train {epoch + 1}', ncols=100)

        for batch in train_loader:
            csi       = batch['csi'].to(device)
            kp_target = batch['keypoints'].to(device)
            mask      = batch['mask'].to(device)

            optimizer.zero_grad()
            pose_pred, _ = model(csi)
            loss = masked_mse_loss(pose_pred, kp_target, mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(n_batches, 1)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # ── Validation ───────────────────────────────────────────────
        m = evaluate(model, val_loader, device, desc=f'Val   {epoch + 1}')

        # ── Console output ───────────────────────────────────────────
        elapsed = time.time() - t0
        print(f'Epoch {epoch + 1} — {elapsed:.1f}s   LR: {current_lr:.6f}')
        print(f'  Train loss: {train_loss:.4f}   Val loss: {m["loss"]:.4f}')
        print(f'  MPJPE: {m["mpjpe"]:.2f} mm   PA-MPJPE: {m["pampjpe"]:.2f} mm')
        print(f'  MPJPE-H: {m["mpjpe_h"]:.2f} mm  '
              f'MPJPE-V: {m["mpjpe_v"]:.2f} mm  '
              f'MPJPE-D: {m["mpjpe_d"]:.2f} mm')
        print(f'  PCK@50:{m["pck50"]:.2f}  @40:{m["pck40"]:.2f}'
              f'  @30:{m["pck30"]:.2f}  @20:{m["pck20"]:.2f}'
              f'  @10:{m["pck10"]:.2f}')

        # ── TensorBoard ──────────────────────────────────────────────
        writer.add_scalar('Loss/Train',      train_loss,   epoch + 1)
        writer.add_scalar('Loss/Validation', m['loss'],    epoch + 1)
        writer.add_scalar('Metric/MPJPE',    m['mpjpe'],   epoch + 1)
        writer.add_scalar('Metric/PAMPJPE',  m['pampjpe'], epoch + 1)
        writer.add_scalar('Metric/MPJPE_h',  m['mpjpe_h'], epoch + 1)
        writer.add_scalar('Metric/MPJPE_v',  m['mpjpe_v'], epoch + 1)
        writer.add_scalar('Metric/MPJPE_d',  m['mpjpe_d'], epoch + 1)
        for k in ('pck50', 'pck40', 'pck30', 'pck20', 'pck10'):
            writer.add_scalar(f'PCK/{k}', m[k], epoch + 1)

        # ── CSV row ──────────────────────────────────────────────────
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch + 1, current_lr, train_loss,
                m['loss'], m['mpjpe'], m['pampjpe'],
                m['mpjpe_h'], m['mpjpe_v'], m['mpjpe_d'],
                m['pck50'], m['pck40'], m['pck30'], m['pck20'], m['pck10'],
                elapsed,
            ])

        # ── Per-epoch TXT (human-readable) ───────────────────────────
        with open(txt_path, 'a') as f:
            f.write(f"Epoch {epoch + 1:>3}/{num_epochs}  "
                    f"({elapsed:.1f}s, lr={current_lr:.6f})\n")
            f.write(f"  Train loss : {train_loss:.4f}\n")
            f.write(f"  Val loss   : {m['loss']:.4f}\n")
            f.write(f"  MPJPE      : {m['mpjpe']:.2f} mm\n")
            f.write(f"  PA-MPJPE   : {m['pampjpe']:.2f} mm\n")
            f.write(f"  MPJPE-H/V/D: "
                    f"{m['mpjpe_h']:.2f} / {m['mpjpe_v']:.2f} / {m['mpjpe_d']:.2f} mm\n")
            f.write(f"  PCK@50/40/30/20/10: "
                    f"{m['pck50']:.2f} / {m['pck40']:.2f} / {m['pck30']:.2f} / "
                    f"{m['pck20']:.2f} / {m['pck10']:.2f}\n\n")

        # ── Save best ────────────────────────────────────────────────
        # We track three "best" checkpoints; for reporting use best_mpjpe.pth.
        if m['loss'] < best_val_loss:
            best_val_loss = m['loss']
            torch.save(model.state_dict(),
                       os.path.join(weights_path, 'best_val_loss.pth'))
            print(f'  [saved] best val_loss {best_val_loss:.4f}')

        if m['mpjpe'] < best_mpjpe:
            best_mpjpe = m['mpjpe']
            torch.save(model.state_dict(),
                       os.path.join(weights_path, 'best_mpjpe.pth'))
            print(f'  [saved] best MPJPE {best_mpjpe:.2f} mm')

        if m['pampjpe'] < best_pampjpe:
            best_pampjpe = m['pampjpe']
            torch.save(model.state_dict(),
                       os.path.join(weights_path, 'best_pampjpe.pth'))
            print(f'  [saved] best PA-MPJPE {best_pampjpe:.2f} mm')

        # Track best per-axis & PCK independently
        if m['mpjpe_h'] < best_mpjpe_h: best_mpjpe_h = m['mpjpe_h']
        if m['mpjpe_v'] < best_mpjpe_v: best_mpjpe_v = m['mpjpe_v']
        if m['mpjpe_d'] < best_mpjpe_d: best_mpjpe_d = m['mpjpe_d']
        for i, key in enumerate(['pck50', 'pck40', 'pck30', 'pck20', 'pck10']):
            if m[key] > best_pck[i]:
                best_pck[i] = m[key]

        # ── Result summary (best-ever values; rewritten each epoch) ──
        summary_path = os.path.join(weights_path, 'result_summary.txt')
        with open(summary_path, 'w') as f:
            f.write('Best Results Summary (validation set)\n')
            f.write('=' * 50 + '\n')
            f.write(f'Best val_loss:  {best_val_loss:.4f}\n')
            f.write(f'Best MPJPE:     {best_mpjpe:.4f} mm\n')
            f.write(f'Best PA-MPJPE:  {best_pampjpe:.4f} mm\n')
            f.write(f'Best MPJPE-H:   {best_mpjpe_h:.4f} mm\n')
            f.write(f'Best MPJPE-V:   {best_mpjpe_v:.4f} mm\n')
            f.write(f'Best MPJPE-D:   {best_mpjpe_d:.4f} mm\n')
            for i, pct in enumerate([50, 40, 30, 20, 10]):
                f.write(f'Best PCK@{pct}:   {best_pck[i]:.4f}\n')
            f.write('\nNote: numbers above are best per-metric across all\n')
            f.write('epochs on the held-out VALIDATION split.  Final\n')
            f.write('test-set numbers come from running Eval_PiW3D.py\n')
            f.write('against best_mpjpe.pth.\n')

    writer.close()
    print(f'\nTraining complete.')
    print(f'  Per-epoch CSV: {csv_path}')
    print(f'  Per-epoch TXT: {txt_path}')
    print(f'  Best summary:  {os.path.join(weights_path, "result_summary.txt")}')
    print(f'\nNext step: run Eval_PiW3D.py to get the final test-set number.')