import os
import numpy as np
import scipy.io as scio
import torch
from torch.utils.data import Dataset, DataLoader


def _to_amplitude(arr: np.ndarray) -> np.ndarray:
    """Convert a CSI array to real-valued amplitude.

    Handles three cases that appear in practice:
      1. Already real (float) – returned as-is.
      2. Standard complex dtype (np.complex64/128) – np.abs applied.
      3. Structured dtype with 'real'/'imag' fields – the form h5py uses
         when MATLAB saves complex data as separate real/imag components.
         dtype example: [('real', '<f8'), ('imag', '<f8')]
    """
    if arr.dtype.names and 'real' in arr.dtype.names and 'imag' in arr.dtype.names:
        # Structured complex: reconstruct and take magnitude
        return np.sqrt(arr['real'].astype(np.float64) ** 2 +
                       arr['imag'].astype(np.float64) ** 2)
    if np.iscomplexobj(arr):
        return np.abs(arr)
    return arr


def _load_csi(path):
    """Load CSI amplitude from a .mat file, supporting both v5 and v7.3 formats.

    Returns an array of shape (TX=3, RX=3, subcarriers=30, packets=20) —
    the canonical MATLAB layout — regardless of which reader was used.

    MATLAB v7.3 files are HDF5-based and require h5py.  h5py reads arrays in
    C order, which reverses all dimensions compared to MATLAB's Fortran order,
    so we transpose back after loading.
    """
    # ── Try scipy first (handles .mat v5 / v6) ───────────────────────────
    try:
        mat = scio.loadmat(path)
        for key in ('csi', 'CSI', 'csiData', 'data'):
            if key in mat:
                arr = mat[key]
                break
        else:
            keys = [k for k in mat if not k.startswith('_')]
            if not keys:
                raise KeyError(f"No data key found in {path}.")
            arr = mat[keys[0]]
        # scipy returns shape as stored in MATLAB: (3, 3, 30, 20) ✓
        if np.iscomplexobj(arr):
            arr = np.abs(arr)
        return arr.astype(np.float32)

    except NotImplementedError:
        pass  # v7.3 HDF5 file — fall through to h5py

    # ── Fall back to h5py for MATLAB v7.3 (HDF5) files ──────────────────
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "The .mat files are MATLAB v7.3 (HDF5) format.  "
            "Install h5py to read them:  pip install h5py"
        )

    with h5py.File(path, 'r') as f:
        for key in ('csi', 'CSI', 'csiData', 'data'):
            if key in f:
                arr = f[key][:]   # h5py shape is REVERSED vs MATLAB
                break
        else:
            keys = [k for k in f.keys()]
            if not keys:
                raise KeyError(f"No data key found in {path}.")
            arr = f[keys[0]][:]

    # h5py gives (pkt=20, sub=30, RX=3, TX=3) when MATLAB shape is (3,3,30,20).
    # Reverse all axes to restore the canonical MATLAB layout (3, 3, 30, 20).
    arr = arr.transpose()          # (TX=3, RX=3, sub=30, pkt=20)
    arr = _to_amplitude(arr)
    return arr.astype(np.float32)


class PersonInWiFi3DDataset(Dataset):
    """PyTorch Dataset for the Person-in-WiFi 3D benchmark.

    Directory layout expected at ``data_root``:
        Person-in-WiFi-3D/
          train_data/
            train_data_list.txt
            csi/        *.mat   – CSI amplitude, shape (3, 3, 30, 20)
            keypoint/   *.npy   – 3-D skeleton,  shape (n_persons, 14, 3)
          test_data/
            test_data_list.txt
            csi/
            keypoint/

    File naming convention: ``S{scene}{n_people}_{seq}_{frame}``
    e.g. ``S11_01_10`` → scene 1, 1 person, sequence 1, frame 10.

    Each CSI sample has shape (TX=3, RX=3, subcarriers=30, packets=20).
    The loader permutes this to (T=20, C=3, H=30, W=3) so the CNN encoder
    can treat packets as the temporal dimension, RX antennas as channels,
    subcarriers as height, and TX antennas as width.

    Keypoints are zero-padded to ``max_persons`` rows.  A float mask of
    shape (max_persons,) marks which rows are real (1.0) vs padding (0.0).

    If ``single_person_only`` is True, samples with more than one person are
    filtered out at construction time (used for VST-Pose comparison runs
    where the model is restricted to single-person scenarios).
    """

    NUM_JOINTS = 14
    # (TX, RX, subcarriers, packets) – native shape in the .mat files
    CSI_NATIVE_SHAPE = (3, 3, 30, 20)

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        max_persons: int = 3,
        single_person_only: bool = False,
    ):
        assert split in ('train', 'test'), "split must be 'train' or 'test'"
        self.data_root = data_root
        self.split = split
        self.max_persons = max_persons
        self.single_person_only = single_person_only

        split_dir = 'train_data' if split == 'train' else 'test_data'
        self.split_path = os.path.join(data_root, split_dir)

        list_file = os.path.join(self.split_path, f'{split}_data_list.txt')
        with open(list_file, 'r') as f:
            self.sample_names = [ln.strip() for ln in f if ln.strip()]

        # ── Optional single-person filter ──────────────────────────────
        if self.single_person_only:
            before = len(self.sample_names)
            self.sample_names = [
                n for n in self.sample_names if self._parse_n_persons(n) == 1
            ]
            after = len(self.sample_names)
            print(
                f"[{split}] single_person_only filter: kept {after} / {before} "
                f"samples ({100.0 * after / max(before, 1):.1f}%)"
            )

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.sample_names)

    def _parse_n_persons(self, name: str) -> int:
        """Extract declared person count from filename character index 2.

        Example: 'S11_01_10' → int('1') = 1 person.
        """
        return int(name[2])

    def __getitem__(self, idx: int):
        name = self.sample_names[idx]

        # ── CSI ────────────────────────────────────────────────────────
        csi_path = os.path.join(self.split_path, 'csi', name + '.mat')
        csi = _load_csi(csi_path)  # (3, 3, 30, 20)

        # Permute: (TX, RX, sub, pkt) → (pkt, RX, sub, TX)
        #          = (T=20, C=3, H=30, W=3)
        csi = np.transpose(csi, (3, 1, 2, 0))

        # Normalise to [0, 1] per sample
        lo, hi = csi.min(), csi.max()
        if hi > lo:
            csi = (csi - lo) / (hi - lo)

        # ── Keypoints ──────────────────────────────────────────────────
        kp_path = os.path.join(self.split_path, 'keypoint', name + '.npy')
        keypoints = np.load(kp_path).astype(np.float32)  # (n, 14, 3)
        n_persons = keypoints.shape[0]

        # Pad / truncate to max_persons
        padded_kp = np.zeros((self.max_persons, self.NUM_JOINTS, 3), dtype=np.float32)
        n_valid = min(n_persons, self.max_persons)
        padded_kp[:n_valid] = keypoints[:n_valid]

        # 1.0 for real persons, 0.0 for padding slots
        mask = np.zeros(self.max_persons, dtype=np.float32)
        mask[:n_valid] = 1.0

        return {
            'csi':       torch.from_numpy(csi),               # (20, 3, 30, 3)
            'keypoints': torch.from_numpy(padded_kp),         # (max_persons, 14, 3)
            'mask':      torch.from_numpy(mask),              # (max_persons,)
            'n_persons': n_valid,
            'name':      name,
        }


# ──────────────────────────────────────────────────────────────────────
def make_piw3d_dataloader(
    data_root: str,
    split: str,
    batch_size: int,
    num_workers: int = 4,
    max_persons: int = 3,
    shuffle=None,
    single_person_only: bool = False,
) -> DataLoader:
    """Build a DataLoader for Person-in-WiFi 3D.

    Args:
        data_root:          Path to the ``Person-in-WiFi-3D`` root directory.
        split:              ``'train'`` or ``'test'``.
        batch_size:         Samples per batch.
        num_workers:        Worker processes for data loading.
                            On Windows set to 0 if multiprocessing errors occur.
        max_persons:        Maximum persons per sample (pads shorter ones).
        shuffle:            Defaults to True for train, False for test.
        single_person_only: If True, only samples with exactly 1 person are
                            included.  Used for restricted VST-Pose runs.
    """
    dataset = PersonInWiFi3DDataset(
        data_root,
        split=split,
        max_persons=max_persons,
        single_person_only=single_person_only,
    )
    if shuffle is None:
        shuffle = (split == 'train')
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=(split == 'train'),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader