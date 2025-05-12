#!/usr/bin/env python3
"""
pretrain_mae_maps.py

Self-supervised masked autoencoder (MAE) pretraining on protein distance maps.
Generates and saves pretrained encoder weights for downstream multimodal AMP classification.
"""

import argparse
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def compute_global_stats(maps_dir):
    """
    Scan all .npy files under maps_dir to compute global min and max.
    """
    paths = glob.glob(os.path.join(maps_dir, "*.npy"))
    if not paths:
        raise FileNotFoundError(f"No .npy files found in {maps_dir}")
    gmin, gmax = float("inf"), -float("inf")
    for p in paths:
        mat = np.load(p)
        m, M = float(mat.min()), float(mat.max())
        if m < gmin: gmin = m
        if M > gmax: gmax = M
    return gmin, gmax, paths


class MapDataset(Dataset):
    """
    Dataset of single-channel distance maps for MAE pretraining.
    Returns: img_tensor of shape (1, img_size, img_size)
    """
    def __init__(self, maps_dir, global_min, global_max, img_size):
        self.paths = glob.glob(os.path.join(maps_dir, "*.npy"))
        if not self.paths:
            raise FileNotFoundError(f"No .npy files found in {maps_dir}")
        self.global_min = global_min
        self.global_max = global_max
        self.img_size = img_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        mat = np.load(self.paths[idx]).astype(np.float32)
        # global normalize and clip
        mat = (mat - self.global_min) / (self.global_max - self.global_min + 1e-8)
        mat = np.clip(mat, 0.0, 1.0)
        # to torch tensor and resize
        x = torch.from_numpy(mat).unsqueeze(0)  # (1, H, W)
        x = F.interpolate(
            x.unsqueeze(0),
            size=(self.img_size, self.img_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # (1, img_size, img_size)
        return x


class MAEForMaps(nn.Module):
    """
    Masked Autoencoder for 1-channel distance maps.
    """
    def __init__(
        self,
        img_size:   int = 224,
        patch_size: int = 16,
        d_model:    int = 256,
        n_heads:    int = 8,
        enc_layers: int = 6,
        dec_layers: int = 4,
        mask_ratio: float = 0.75,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        
        # patch embedding
        self.patch_embed = nn.Conv2d(1, d_model, patch_size, patch_size)
        num_patches = (img_size // patch_size) ** 2
        
        # encoder tokens & pos emb
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.enc_pos   = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, enc_layers)
        
        # decoder tokens & pos emb
        self.dec_pos    = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        dec_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 4, batch_first=True
        )
        self.decoder = nn.TransformerEncoder(dec_layer, dec_layers)
        self.reconstruct = nn.Linear(d_model, patch_size * patch_size)

    def random_masking(self, x):
        """
        x: (B, N, D) patch tokens without CLS
        returns: x_vis, ids_keep, ids_restore
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - self.mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_vis = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        return x_vis, ids_keep, ids_restore

    def forward(self, img):
        """
        img: (B, 1, H, W)
        returns: scalar MAE loss
        """
        B = img.size(0)
        # embed patches
        patches = self.patch_embed(img)               # (B, D, P, P)
        patches = patches.flatten(2).transpose(1, 2)  # (B, N, D)
        
        # mask
        x_vis, ids_keep, ids_restore = self.random_masking(patches)

        # encoder input
        cls = self.cls_token.expand(B, -1, -1)       # (B,1,D)
        enc_input = torch.cat([cls, x_vis], dim=1)   # (B,1+len_keep,D)
        enc_input = enc_input + self.enc_pos[:, :enc_input.size(1), :]
        enc_out = self.encoder(enc_input)            # (B,1+len_keep,D)

        # prepare patches for decoder
        N, D = patches.size(1), patches.size(2)
        patch_tokens = torch.zeros(B, N, D, device=img.device)
        patch_tokens.scatter_(
            1,
            ids_keep.unsqueeze(-1).expand(-1, -1, D),
            enc_out[:, 1:]
        )

        # decoder input (CLS + all patch tokens)
        dec_input = torch.cat([enc_out[:, :1], patch_tokens], dim=1)  # (B,N+1,D)
        dec_input = dec_input + self.dec_pos
        dec_out = self.decoder(dec_input)                             # (B,N+1,D)

        # reconstruct only patch pixels (skip CLS)
        rec_tokens = dec_out[:, 1:]                                   # (B,N,D)
        rec_patches = self.reconstruct(rec_tokens)                    # (B,N,P*P)
        target = patches.reshape(B, N, -1)                            # (B,N,P*P)

        # loss on masked patches
        mask = torch.ones(B, N, device=img.device)
        mask.scatter_(1, ids_keep, 0)  # 0 for visible, 1 for masked
        loss = ((rec_patches - target) ** 2 * mask.unsqueeze(-1)).sum()
        loss = loss / mask.sum()
        return loss


def main():
    parser = argparse.ArgumentParser("MAE Pretraining on distance maps")
    parser.add_argument("--maps_dir",   type=str, required=True)
    parser.add_argument("--img_size",   type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--d_model",    type=int, default=256)
    parser.add_argument("--n_heads",    type=int, default=8)
    parser.add_argument("--enc_layers", type=int, default=6)
    parser.add_argument("--dec_layers", type=int, default=4)
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--device",     type=str, default="cuda")
    parser.add_argument("--out_path",   type=str, default="mae_struct_encoder.pth")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("Computing global min/max...")
    global_min, global_max, _ = compute_global_stats(args.maps_dir)
    print(f"Global range: [{global_min:.4f}, {global_max:.4f}]")

    # Data
    ds = MapDataset(
        maps_dir= args.maps_dir,
        global_min=global_min,
        global_max=global_max,
        img_size= args.img_size
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # MAE model
    mae = MAEForMaps(
        img_size   = args.img_size,
        patch_size = args.patch_size,
        d_model    = args.d_model,
        n_heads    = args.n_heads,
        enc_layers = args.enc_layers,
        dec_layers = args.dec_layers,
        mask_ratio = args.mask_ratio
    ).to(device)

    optimizer = torch.optim.AdamW(mae.parameters(), lr=args.lr, weight_decay=0.05)

    # Train
    for epoch in range(1, args.epochs + 1):
        mae.train()
        total_loss = 0.0
        # add tqdm if you want
        print(f"Epoch {epoch:02d} | Training...")
        for img in tqdm(loader, desc="Training", leave=False):
            img = img.to(device)
            optimizer.zero_grad()
            loss = mae(img)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch:02d} | MAE Loss: {avg_loss:.4f}")

    # Save encoder
    torch.save(mae.encoder.state_dict(), args.out_path)
    print(f"Saved encoder weights to {args.out_path}")


if __name__ == "__main__":
    main()
