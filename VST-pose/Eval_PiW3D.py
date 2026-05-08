"""Final test-set evaluation for VST-Pose on Person-in-WiFi 3D.

Run AFTER training completes:
    python Eval_PiW3D.py --config_file config/config_piw3d.yaml \
                         --checkpoint pose_weights/person-in-wifi-3d/best_mpjpe.pth

Why a separate script?
----------------------
During training, ``Train_PiW3D.py`` evaluates each epoch against a held-out
validation set carved from the train list.  The TEST set is never touched
during training — that ensures the headline number we report is unbiased.

This script loads a saved checkpoint, evaluates it once on the test set,
and writes the result to ``test_results.txt`` next to the checkpoint.
The number from this script is what should go in your report.
"""

import os
import argparse
import yaml
import torch
import numpy as np

from Feeder.person_in_wifi3d import make_piw3d_dataloader
from Model.PiW3D.conv_STFormer_PiW3D import PiW3DModel

# Re-use the evaluate() function from training so metrics are computed
# identically on the test set.
from Train_PiW3D import evaluate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Final test-set evaluation for VST-Pose.')
    parser.add_argument('--config_file', type=str, default='config/config_piw3d.yaml')
    parser.add_argument('--checkpoint',  type=str,
                        default='pose_weights/person-in-wifi-3d/best_mpjpe.pth',
                        help='Path to the .pth state-dict to evaluate.')
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}\n"
            f"Did you run Train_PiW3D.py first?"
        )

    with open(args.config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dataset_root       = config['dataset_root']
    max_persons        = config.get('max_persons', 3)
    num_packets        = config.get('num_packets', 20)
    feature_dim        = config.get('feature_dim', 14)
    dim                = config.get('dim', 128)
    single_person_only = config.get('single_person_only', False)

    print(f"Evaluating: {args.checkpoint}")
    print(f"Mode: {'SINGLE-PERSON' if single_person_only else 'MULTI-PERSON'} "
          f"(max_persons={max_persons})")

    # Test loader — note: NO val_fraction here, we want the full test set.
    test_loader = make_piw3d_dataloader(
        data_root          = dataset_root,
        split              = 'test',
        batch_size         = config.get('test_loader', {}).get('batch_size', 32),
        num_workers        = config.get('test_loader', {}).get('num_workers', 4),
        max_persons        = max_persons,
        shuffle            = False,
        single_person_only = single_person_only,
        val_fraction       = 0.0,    # do not split the test list
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model = PiW3DModel(
        num_packets = num_packets,
        feature_dim = feature_dim,
        dim         = dim,
        max_persons = max_persons,
    ).to(device)

    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded weights from {args.checkpoint}")

    # ── Single evaluation pass on the test set ──────────────────────
    metrics = evaluate(model, test_loader, device, desc='Test')

    # ── Console summary ─────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("FINAL TEST-SET RESULTS")
    print("=" * 50)
    print(f"  Test loss   : {metrics['loss']:.4f}")
    print(f"  MPJPE       : {metrics['mpjpe']:.2f} mm")
    print(f"  PA-MPJPE    : {metrics['pampjpe']:.2f} mm")
    print(f"  MPJPE-H/V/D : "
          f"{metrics['mpjpe_h']:.2f} / "
          f"{metrics['mpjpe_v']:.2f} / "
          f"{metrics['mpjpe_d']:.2f} mm")
    print(f"  PCK@50/40/30/20/10: "
          f"{metrics['pck50']:.2f} / {metrics['pck40']:.2f} / "
          f"{metrics['pck30']:.2f} / {metrics['pck20']:.2f} / "
          f"{metrics['pck10']:.2f}")

    # ── Persist to disk next to the checkpoint ──────────────────────
    out_dir = os.path.dirname(args.checkpoint)
    out_path = os.path.join(out_dir, 'test_results.txt')
    with open(out_path, 'w') as f:
        f.write("VST-Pose on Person-in-WiFi-3D — Final Test-Set Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Checkpoint  : {args.checkpoint}\n")
        f.write(f"Config file : {args.config_file}\n")
        f.write(f"Mode        : {'SINGLE-PERSON' if single_person_only else 'MULTI-PERSON'}\n")
        f.write(f"Test samples: {len(test_loader.dataset)}\n")
        f.write("-" * 60 + "\n")
        f.write(f"Test loss     : {metrics['loss']:.4f}\n")
        f.write(f"MPJPE         : {metrics['mpjpe']:.4f} mm\n")
        f.write(f"PA-MPJPE      : {metrics['pampjpe']:.4f} mm\n")
        f.write(f"MPJPE-H       : {metrics['mpjpe_h']:.4f} mm\n")
        f.write(f"MPJPE-V       : {metrics['mpjpe_v']:.4f} mm\n")
        f.write(f"MPJPE-D       : {metrics['mpjpe_d']:.4f} mm\n")
        f.write(f"PCK@50        : {metrics['pck50']:.4f}\n")
        f.write(f"PCK@40        : {metrics['pck40']:.4f}\n")
        f.write(f"PCK@30        : {metrics['pck30']:.4f}\n")
        f.write(f"PCK@20        : {metrics['pck20']:.4f}\n")
        f.write(f"PCK@10        : {metrics['pck10']:.4f}\n")
        f.write("\nNotes:\n")
        f.write(" - PCK uses bbox-diagonal scale (joint-order independent),\n")
        f.write("   not directly comparable to papers that use torso scale.\n")
        f.write(" - Numbers above are computed on the full PiW3D test list\n")
        f.write("   (after single_person filtering, if enabled).\n")

    print(f"\nResults written to: {out_path}")