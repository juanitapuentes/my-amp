import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class AmpDataset(Dataset):
    """
    PyTorch Dataset for Antimicrobial Peptides (AMPs).
    Returns tuples of (sequence_tensor, distance_map_tensor, label_tensor).
    Sequence tensor is produced via a user-provided transform (e.g., token IDs).
    Distance maps are loaded from .npy files named '{Hash}.npy'.
    Labels correspond to defined antimicrobial activities.
    """
    LABEL_COLUMNS = [
        "Antibacterial", "Antifungal", "Antiviral",
        "Antiparasitic", "Antimicrobial"
    ]

    def __init__(
        self,
        csv_file: str,
        maps_dir: str,
        seq_transform=None,
        split_file: str=None,
        args=None
    ):
        """
        Args:
            csv_file: Path to CSV with columns 'Sequence', 'Hash', and LABEL_COLUMNS.
            maps_dir: Directory containing distance map .npy files named '{Hash}.npy'.
            seq_transform: Callable mapping sequence string to tensor.
            split_file: Optional CSV listing sequences for this split.
        """
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found at {csv_file}")
        df = pd.read_csv(csv_file)
        if split_file:
            splits = pd.read_csv(split_file)['Sequence'].tolist()
            df = df.set_index('Sequence').loc[splits].reset_index()
        df['Hash'] = df['Hash'].fillna('')
        df[self.LABEL_COLUMNS] = df[self.LABEL_COLUMNS].fillna(0.0)
        self.df = df
        self.maps_dir = maps_dir
        self.seq_transform = seq_transform
        self.args = args

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sequence = row['Sequence']
        if self.seq_transform:
            sequence = self.seq_transform(sequence)

        hash_str = row['Hash']
        path = os.path.join(self.maps_dir, f"{hash_str}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Distance map not found at {path}")
        raw_map = np.load(path).astype(np.float32)
        tensor_map = torch.from_numpy(raw_map).unsqueeze(0).unsqueeze(0)
        L = self.args.dist_max_len
        resized = F.interpolate(
            tensor_map,
            size=(L, L),
            mode='bilinear',
            align_corners=False
        )
        distance_map = resized.squeeze(0).squeeze(0).float()

        labels = torch.tensor(
            row[self.LABEL_COLUMNS].values.astype(np.float32)
        )

        return sequence, distance_map, labels
    
# dataset.py

class AmpDatasetWithImages(Dataset):
    """
    PyTorch Dataset for AMPs returning:
      - seq_ids:   LongTensor (max_len,)
      - img:       FloatTensor (1, img_size, img_size)
      - labels:    FloatTensor (num_classes,)
    Accepts an `args` object so you can pass through global_min/global_max, etc.
    """
    LABEL_COLUMNS = [
        "Antibacterial", "Antifungal", "Antiviral",
        "Antiparasitic", "Antimicrobial"
    ]

    def __init__(
        self,
        csv_file: str,
        maps_dir: str,
        seq_transform=None,
        split_file: str = None,
        global_min: float = None,
        global_max: float = None,
        img_size: int = 224,
        args=None,                    # ← new
    ):
        import pandas as pd
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV not found: {csv_file}")
        df = pd.read_csv(csv_file)
        if split_file:
            splits = pd.read_csv(split_file)['Sequence'].tolist()
            df = df.set_index('Sequence').loc[splits].reset_index()
        df['Hash'] = df['Hash'].fillna('')
        df[self.LABEL_COLUMNS] = df[self.LABEL_COLUMNS].fillna(0.0)
        self.df = df
        self.maps_dir   = maps_dir
        self.seq_transform = seq_transform
        self.global_min = global_min
        self.global_max = global_max
        self.img_size   = img_size
        self.args       = args         # store it if you need elsewhere

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --- sequence IDs ---
        seq = row['Sequence']
        if self.seq_transform:
            seq_ids = self.seq_transform(seq)
        else:
            raise ValueError("You must provide a seq_transform")

        # --- load distance map ---
        npy_path = os.path.join(self.maps_dir, f"{row['Hash']}.npy")
        mat = np.load(npy_path).astype(np.float32)

        # --- normalize globally ---
        mat = (mat - self.global_min) / (self.global_max - self.global_min + 1e-8)
        mat = np.clip(mat, 0.0, 1.0)

        # --- resize to square image ---
        x = torch.from_numpy(mat).unsqueeze(0)  # (1, H, W)
        x = F.interpolate(
            x.unsqueeze(0),
            size=(self.img_size, self.img_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # now (1, img_size, img_size)

        # --- labels ---
        labels = torch.tensor(
            row[self.LABEL_COLUMNS].values.astype(np.float32),
            dtype=torch.float32
        )

        return seq_ids, x, labels