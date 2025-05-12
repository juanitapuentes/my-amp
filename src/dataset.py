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
    

class AmpDatasetWithImages(Dataset):
    """
    PyTorch Dataset for Antimicrobial Peptides (AMPs).
    Returns tuples of (sequence_tensor, image_tensor, label_tensor).
    Sequence tensor is produced via a user-provided transform.
    Distance maps (.npy) are now treated as images and fed through a ViT pipeline.
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

        # transform para ViT: convierte la matriz 2D normalizada a PIL y luego a tensor 3×H×W
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(mode='F'),
            transforms.Resize((self.args.dist_max_len, self.args.dist_max_len)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5],
                                 std=[0.5]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # secuencia
        seq = row['Sequence']
        if self.seq_transform:
            seq = self.seq_transform(seq)

        # carga .npy y normaliza a [0,1]
        hash_str = row['Hash']
        npy_path = os.path.join(self.maps_dir, f"{hash_str}.npy")
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"Distance map not found at {npy_path}")
        mat = np.load(npy_path).astype(np.float32)
        mn, mx = mat.min(), mat.max()
        mat = (mat - mn) / (mx - mn + 1e-8)

        # procesa como imagen para ViT → tensor (3, H, W)
        img = self.image_transform(mat)

        # etiquetas
        labels = torch.tensor(
            row[self.LABEL_COLUMNS].values.astype(np.float32),
            dtype=torch.float32
        )

        return seq, img, labels