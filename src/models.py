import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, MultiheadAttention
import timm


class SequenceTransformer(nn.Module):
    """
    Transformer encoder for peptide sequences.
    Incorporates token and positional embeddings and a [CLS] token.
    """
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        args=None
    ):
        super().__init__()
        self.args = args
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len+1, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*4, batch_first=True
        )
        self.encoder = TransformerEncoder(layer, num_layers=n_layers)

        if self.args.mode == 'sequence':
            self.classifier = nn.Linear(d_model, args.num_classes)

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        """
        Args:
            x: Tensor shape (batch_size, seq_len) of token IDs.
        Returns:
            Tensor shape (batch_size, seq_len+1, d_model) of encoded features.
        """
        B, S = x.size()
        tok = self.token_emb(x)
        pos_ids = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
        pos = self.pos_emb(pos_ids)
        seq = tok + pos
        cls = self.cls_token.expand(B, -1, -1)
        inp = torch.cat([cls, seq], dim=1)
        
        if self.args.mode == 'sequence':
            cls_out = self.encoder(inp)[:, 0, :]
            logits = self.classifier(cls_out)
            return logits
        
        return self.encoder(inp)

class DistanceTransformer(nn.Module):
    """
    Transformer encoder for distance maps.
    Projects each row of the SxS distance matrix as a token.
    """
    def __init__(
        self,
        max_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        args=None
    ):
        super().__init__()
        self.args = args
        self.patch_embed = nn.Linear(max_len, d_model)
        self.pos_emb = nn.Embedding(max_len+1, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*4, batch_first=True
        )
        self.encoder = TransformerEncoder(layer, num_layers=n_layers)

        if self.args.mode == 'distance':
            self.classifier = nn.Linear(d_model, args.num_classes)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: Tensor shape (batch_size, S, S) distance matrices.
        Returns:
            Tensor shape (batch_size, S+1, d_model) of encoded features.
        """
        B, S, _ = x.size()
        tokens = self.patch_embed(x)
        pos_ids = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
        pos = self.pos_emb(pos_ids)
        seq = tokens + pos
        cls = self.cls_token.expand(B, -1, -1)
        inp = torch.cat([cls, seq], dim=1)

        if self.args.mode == 'distance':
            cls_out = self.encoder(inp)[:, 0, :]
            logits = self.classifier(cls_out)
            return logits

        return self.encoder(inp)

class CrossAttentionModel(nn.Module):
    """
    Combines sequence and distance transformers via cross-attention.
    Uses [CLS] token from sequence as query over distance tokens.
    """
    def __init__(
        self,
        seq_model: SequenceTransformer,
        dist_model: DistanceTransformer,
        d_model: int,
        n_heads: int,
        num_classes: int
    ):
        super().__init__()
        self.seq_model = seq_model
        self.dist_model = dist_model
        self.cross_attn = MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(
        self,
        seq_ids: torch.LongTensor,
        dist_map: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Args:
            seq_ids: Tensor (batch_size, seq_len)
            dist_map: Tensor (batch_size, S, S)
        Returns:
            logits: Tensor (batch_size, num_classes)
        """
        seq_out = self.seq_model(seq_ids)
        dist_out = self.dist_model(dist_map)
        q = seq_out[:, :1, :]
        k = dist_out
        v = dist_out
        attn_out, _ = self.cross_attn(q, k, v)
        cls_feat = attn_out.squeeze(1)
        return self.classifier(cls_feat)

class ClassifierTransformer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, num_layers=4, num_classes=None, vocab_size=None, max_len=None):
        super().__init__()
        self.cls_token_id = vocab_size
        self.token_emb = nn.Embedding(vocab_size + 1, d_model, padding_idx=0)
        self.pos_emb   = nn.Embedding(max_len + 1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4,
                                                   dropout=0.1, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Linear(d_model, num_classes) if num_classes else None

    def forward_get_cls(self, x):
        bsz, seq_len = x.size()
        cls_tokens = torch.full((bsz, 1), self.cls_token_id,
                                dtype=torch.long, device=x.device)
        x = torch.cat([cls_tokens, x], dim=1)
        positions = torch.arange(seq_len + 1, device=x.device).unsqueeze(0)
        h = self.token_emb(x) + self.pos_emb(positions)
        h = self.encoder(h)
        return h[:, 0, :]

class ImageViTEncoder(nn.Module):
    #def __init__(self, model_name='vit_tiny_patch16_224', pretrained=True):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        super().__init__()
        # now accepts 1-channel images instead of 3
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=1)
        self.vit.head = nn.Identity()  # eliminar capa de clasificación

    def forward_get_cls(self, x):

        return self.vit(x)

class BidirectionalCrossAttention(nn.Module):
    def __init__(self, dim_seq, dim_img, num_heads):
        super().__init__()
        self.cross_seq_to_img = MultiheadAttention(embed_dim=dim_seq,
                                                   kdim=dim_img, vdim=dim_img,
                                                   num_heads=num_heads,
                                                   batch_first=True)
        self.cross_img_to_seq = MultiheadAttention(embed_dim=dim_img,
                                                   kdim=dim_seq, vdim=dim_seq,
                                                   num_heads=num_heads,
                                                   batch_first=True)

    def forward(self, cls_seq, cls_img):
        q_seq = cls_seq.unsqueeze(1)
        q_img = cls_img.unsqueeze(1)
        attn_seq, _ = self.cross_seq_to_img(q_seq, q_img, q_img)
        attn_img, _ = self.cross_img_to_seq(q_img, q_seq, q_seq)
        return attn_seq.squeeze(1), attn_img.squeeze(1)

# models.py

class StructTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        img_channels: int = 1,             # ← ensure default is 1
    ):
        super().__init__()
        # <— in_channels must match your Dataset’s output (1)
        self.patch_embed = nn.Conv2d(
            in_channels=img_channels,      # was 2 before
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size
        )
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb   = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = TransformerEncoder(layer, num_layers=n_layers)

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, img_size, img_size)
        B = x.size(0)
        patches = self.patch_embed(x)                   # (B, d_model, P, P)
        patches = patches.flatten(2).transpose(1, 2)    # (B, num_patches, d_model)
        cls = self.cls_token.expand(B, -1, -1)          # (B, 1, d_model)
        x = torch.cat([cls, patches], dim=1)            # (B, num_patches+1, d_model)
        x = x + self.pos_emb                            # add positional embeddings
        out = self.encoder(x)                           # (B, num_patches+1, d_model)
        return out[:, 0]                                # return CLS token
    
# models.py (continued)

class MultiModalClassifier(nn.Module):
    def __init__(
        self,
        seq_d_model:    int    = 256,
        struct_d_model: int    = 256,
        n_heads:        int    = 8,
        num_layers:     int    = 4,
        num_classes:    int    = 5,
        vocab_size:     int    = None,
        max_len_seq:    int    = 200,
        img_size:       int    = 224,
        patch_size:     int    = 16,
        img_channels:   int    = 1,       # <— pass 1 here
    ):
        super().__init__()
        # sequence branch (unchanged)
        self.seq_encoder = ClassifierTransformer(
            d_model=seq_d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            num_classes=None,
            vocab_size=vocab_size,
            max_len=max_len_seq
        )

        # structure branch now single-channel
        self.struct_encoder = StructTransformer(
            img_size=img_size,
            patch_size=patch_size,
            d_model=struct_d_model,
            n_heads=n_heads,
            n_layers=num_layers,
            img_channels=img_channels
        )

        # fusion
        self.cross_attn = BidirectionalCrossAttention(
            dim_seq=seq_d_model,
            dim_img=struct_d_model,
            num_heads=n_heads
        )
        self.dropout    = nn.Dropout(0.2)
        self.classifier = nn.Linear(seq_d_model + struct_d_model, num_classes)

    def forward(self, seq_ids, img_tensor):
        # seq_ids:   (B, S)
        # img_tensor:(B, 1, img_size, img_size)
        cls_seq = self.seq_encoder.forward_get_cls(seq_ids)  # (B, seq_d_model)
        cls_img = self.struct_encoder(img_tensor)            # (B, struct_d_model)
        seq_att, img_att = self.cross_attn(cls_seq, cls_img) # (B, seq_d_model), (B, struct_d_model)
        h = torch.cat([seq_att, img_att], dim=1)             # (B, seq_d_model + struct_d_model)
        return self.classifier(self.dropout(h))