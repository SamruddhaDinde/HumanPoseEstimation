"""VST-Pose adapted for the Person-in-WiFi 3D dataset.

Key differences from the MM-Fi version (Model/Speed/conv_STFormer_Speed.py):

  CSIEncoder   – input (B, T=20, 3, 30, 3) instead of (B, T, 3, 114, 10)
                 Pools only along the subcarrier axis (H) to preserve the
                 TX dimension (W=3); fc output size changes accordingly.

  KPDecoder    – outputs (B, max_persons, 14, 3) instead of (B, T, 17, 3).
                 Temporal pooling (mean over T) collapses the packet axis
                 because Person-in-WiFi 3D provides one ground-truth pose
                 per segment, not one per packet.

  SpeedDecoder – same shape change as KPDecoder.  Speed is estimated from
                 the difference between the last and first packet's feature
                 vector (ts_mid_feature[:, -1] − ts_mid_feature[:, 0]).
                 The speed loss is weighted by alpha=0.0 in the default
                 config because per-segment ground truth cannot supply a
                 velocity target; set alpha > 0 only if you build a loader
                 that returns consecutive-frame pose pairs.

All shared Transformer blocks (MLP, Former, TSBlock, Speedformer,
TemporalModel) are imported from the MM-Fi model file unchanged – they are
parameterised by `channel` and `dim` so no edits are needed there.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-use the generic Transformer backbone from the MM-Fi model.
# TemporalModel internally uses DropPath, MLP, Former, TSBlock, Speedformer,
# and trunc_normal_ – they are imported inside that module, not here.
from Model.Speed.conv_STFormer_Speed import TemporalModel

__all__ = ['PiW3DModel']


# ══════════════════════════════════════════════════════════════════════
# 1. Encoder
# ══════════════════════════════════════════════════════════════════════

class CSIEncoder(nn.Module):
    """CNN encoder for Person-in-WiFi 3D CSI segments.

    Input shape per call:  (B, T=20, C=3, H=30, W=3)
      T  = 20 WiFi packets (treated as temporal frames)
      C  = 3  RX antennas  (channels)
      H  = 30 OFDM subcarriers
      W  = 3  TX antennas

    Spatial pooling strategy:
      pool1  MaxPool2d((2,1)) → collapses H (30→15), keeps W (3)
      pool2  MaxPool2d((2,1)) → collapses H (15→7),  keeps W (3)
      After conv3: feature map (feature_dim, 7, 3)
      fc1 input size: 7 × 3 = 21

    Output: (B, T, feature_dim, dim)
    """

    def __init__(self, feature_dim: int = 14, dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(3,  16,          kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1))   # (H=30→15, W=3)

        self.conv2 = nn.Conv2d(16, 32,          kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1))   # (H=15→7,  W=3)

        self.conv3 = nn.Conv2d(32, feature_dim, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(feature_dim)
        # feature map after conv3: (feature_dim, 7, 3)

        self.fc1 = nn.Linear(7 * 3, dim)  # 21 → dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 3, 30, 3)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        x = self.pool1(F.leaky_relu(self.bn1(self.conv1(x))))  # (B*T, 16, 15, 3)
        x = self.pool2(F.leaky_relu(self.bn2(self.conv2(x))))  # (B*T, 32,  7, 3)
        x = F.leaky_relu(self.bn3(self.conv3(x)))              # (B*T, fd,  7, 3)

        bt, fd, h, w = x.shape
        x = x.view(bt, fd, h * w)   # (B*T, fd, 21)
        x = self.fc1(x)              # (B*T, fd, dim)
        x = x.view(B, T, fd, -1)    # (B,   T,  fd, dim)
        return x


# ══════════════════════════════════════════════════════════════════════
# 2. Decoders
# ══════════════════════════════════════════════════════════════════════

class KPDecoder(nn.Module):
    """Decode temporal features into multi-person 3-D keypoints.

    Temporal pooling (mean over the packet axis) collapses T before the
    linear projection because Person-in-WiFi 3D supplies a single pose
    ground truth per segment, not one per packet.

    Output: (B, max_persons, 14, 3)
    """

    def __init__(self, channel: int = 14, dim: int = 128, max_persons: int = 3):
        super().__init__()
        self.max_persons = max_persons
        self.fc1 = nn.Linear(channel * dim, 256)
        self.fc2 = nn.Linear(256, max_persons * 14 * 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, channel, dim)
        B = x.shape[0]
        x = x.mean(dim=1)        # temporal pool → (B, channel, dim)
        x = x.view(B, -1)        # (B, channel*dim)
        x = F.relu(self.fc1(x))  # (B, 256)
        x = self.fc2(x)          # (B, max_persons*14*3)
        return x.view(B, self.max_persons, 14, 3)


class SpeedDecoder(nn.Module):
    """Decode speed (pose displacement over the packet window).

    Uses ts_mid_feature[:, -1] − ts_mid_feature[:, 0] as the speed proxy,
    which captures the change in the learned representation across the 20
    temporal packets within each CSI segment.

    Output: (B, max_persons, 14, 3)
    """

    def __init__(self, channel: int = 14, dim: int = 128, max_persons: int = 3):
        super().__init__()
        self.max_persons = max_persons
        self.fc1 = nn.Linear(channel * dim, 256)
        self.fc2 = nn.Linear(256, max_persons * 14 * 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, channel, dim) — ts_mid_feature from TemporalModel
        B = x.shape[0]
        speed = x[:, -1] - x[:, 0]   # (B, channel, dim)
        speed = speed.view(B, -1)     # (B, channel*dim)
        speed = F.relu(self.fc1(speed))
        speed = self.fc2(speed)       # (B, max_persons*14*3)
        return speed.view(B, self.max_persons, 14, 3)


# ══════════════════════════════════════════════════════════════════════
# 3. Full model
# ══════════════════════════════════════════════════════════════════════

class PiW3DModel(nn.Module):
    """VST-Pose for Person-in-WiFi 3D (multi-person, 14 joints).

    Args:
        num_packets:   Number of WiFi packets per CSI segment (T dimension).
                       Must match the dataset value of 20.
        feature_dim:   Encoder output channels; also the joint-proxy
                       channel count fed into the Transformer.  Default 14
                       (equal to the number of skeleton joints – mirrors the
                       MM-Fi convention where this was 17).
        dim:           Feature dimension per channel inside the Transformer.
        max_persons:   Maximum number of persons per sample.  Decoder
                       always outputs this many pose slots; real ones are
                       selected via the per-sample mask during training.
    """

    def __init__(
        self,
        num_packets: int = 20,
        feature_dim: int = 14,
        dim: int = 128,
        max_persons: int = 3,
    ):
        super().__init__()
        self.num_packets  = num_packets
        self.max_persons  = max_persons

        self.encoder = CSIEncoder(feature_dim=feature_dim, dim=dim)

        self.temporal_model = TemporalModel(
            Thead=4, Shead=4,
            channel=feature_dim,
            dim=dim,
            mlp_ratio=1,
            drop=0.3,
            drop_path=0.,
            att_fuse=True,
            maxlen=200,   # ≥ num_packets; generous upper bound
        )

        self.kp_decoder    = KPDecoder(channel=feature_dim, dim=dim, max_persons=max_persons)
        self.speed_decoder = SpeedDecoder(channel=feature_dim, dim=dim, max_persons=max_persons)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: CSI tensor of shape (B, T=20, 3, 30, 3)

        Returns:
            pose:  (B, max_persons, 14, 3) – predicted 3-D keypoints
            speed: (B, max_persons, 14, 3) – predicted pose displacement
        """
        _, T, _, _, _ = x.shape
        if T != self.num_packets:
            raise ValueError(
                f"Expected {self.num_packets} packets per segment, got {T}."
            )

        x = self.encoder(x)                           # (B, T, feature_dim, dim)
        x, speed_feat = self.temporal_model(x, alpha=0.5)

        pose  = self.kp_decoder(x)                    # (B, max_persons, 14, 3)
        speed = self.speed_decoder(speed_feat)         # (B, max_persons, 14, 3)
        return pose, speed


# ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    csi = torch.randn(4, 20, 3, 30, 3)   # batch=4, T=20 packets
    model = PiW3DModel()
    pose, speed = model(csi)
    print('pose: ', pose.shape)    # (4, 3, 14, 3)
    print('speed:', speed.shape)   # (4, 3, 14, 3)
