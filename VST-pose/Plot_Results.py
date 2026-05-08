"""Plot training curves from metrics_per_epoch.csv.

Run after training:
    python Plot_Results.py --csv pose_weights/person-in-wifi-3d/metrics_per_epoch.csv

Produces four PNG figures next to the CSV:
  * loss_curves.png       — train + val loss vs. epoch
  * mpjpe_curves.png      — MPJPE + PA-MPJPE vs. epoch
  * mpjpe_per_axis.png    — MPJPE-H / V / D vs. epoch
  * pck_curves.png        — PCK at 5 thresholds vs. epoch

Designed to be re-runnable: re-execute after each training run to regenerate
plots without retraining.
"""

import os
import argparse
import csv
import matplotlib.pyplot as plt


def load_csv(path: str):
    """Load metrics CSV into a dict of column → list-of-floats."""
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    cols = {k: [] for k in rows[0].keys()}
    for row in rows:
        for k, v in row.items():
            try:
                cols[k].append(float(v))
            except (ValueError, TypeError):
                cols[k].append(float('nan'))
    return cols


def save_fig(fig, path: str):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  wrote {path}")
    plt.close(fig)


def plot_losses(cols, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cols['epoch'], cols['train_loss'], label='Train loss', linewidth=2)
    ax.plot(cols['epoch'], cols['val_loss'],   label='Val loss',   linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE loss (metres²)')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, os.path.join(out_dir, 'loss_curves.png'))


def plot_mpjpe(cols, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cols['epoch'], cols['val_mpjpe'],   label='MPJPE',    linewidth=2)
    ax.plot(cols['epoch'], cols['val_pampjpe'], label='PA-MPJPE', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error (mm)')
    ax.set_title('Validation MPJPE and PA-MPJPE')
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, os.path.join(out_dir, 'mpjpe_curves.png'))


def plot_per_axis(cols, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cols['epoch'], cols['val_mpjpe_h'], label='MPJPE-H (horizontal)', linewidth=2)
    ax.plot(cols['epoch'], cols['val_mpjpe_v'], label='MPJPE-V (vertical)',   linewidth=2)
    ax.plot(cols['epoch'], cols['val_mpjpe_d'], label='MPJPE-D (depth)',      linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Per-axis L1 error (mm)')
    ax.set_title('Validation Per-Axis MPJPE')
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, os.path.join(out_dir, 'mpjpe_per_axis.png'))


def plot_pck(cols, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for k in ('val_pck50', 'val_pck40', 'val_pck30', 'val_pck20', 'val_pck10'):
        thr = k.replace('val_pck', '@')
        ax.plot(cols['epoch'], cols[k], label=f'PCK{thr}', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PCK (%)')
    ax.set_title('Validation PCK at Multiple Thresholds')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 105)
    save_fig(fig, os.path.join(out_dir, 'pck_curves.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot training curves from CSV.')
    parser.add_argument('--csv', type=str,
                        default='pose_weights/person-in-wifi-3d/metrics_per_epoch.csv')
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    cols = load_csv(args.csv)
    print(f"Loaded {len(cols['epoch'])} epochs from {args.csv}")

    out_dir = os.path.dirname(os.path.abspath(args.csv))

    plot_losses(cols,   out_dir)
    plot_mpjpe(cols,    out_dir)
    plot_per_axis(cols, out_dir)
    plot_pck(cols,      out_dir)

    print(f"\nAll plots saved to: {out_dir}")