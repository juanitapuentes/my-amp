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
    def __init__(self, model_name='vit_tiny_patch16_224', pretrained=True):
    #def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
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


class GatedBidirectionalCrossAttention(nn.Module):
    def __init__(self, dim_seq, dim_img, num_heads):
        super().__init__()
        # your original cross-attention modules
        self.cross_seq_to_img = MultiheadAttention(embed_dim=dim_seq,
                                                   kdim=dim_img, vdim=dim_img,
                                                   num_heads=num_heads,
                                                   batch_first=True)
        self.cross_img_to_seq = MultiheadAttention(embed_dim=dim_img,
                                                   kdim=dim_seq, vdim=dim_seq,
                                                   num_heads=num_heads,
                                                   batch_first=True)
        # two scalar gates (initialized to 0 → sigmoid(0)=0.5)
        self.alpha_seq = nn.Parameter(torch.tensor(0.0))
        self.alpha_img = nn.Parameter(torch.tensor(0.0))
        self.sigmoid = nn.Sigmoid()

    def forward(self, cls_seq, cls_img):
        # cls_seq: (B, D_seq), cls_img: (B, D_img)
        q_seq = cls_seq.unsqueeze(1)    # (B,1,D_seq)
        k_img = cls_img.unsqueeze(1)    # (B,1,D_img)
        v_img = k_img
        attn_seq, _ = self.cross_seq_to_img(q_seq, k_img, v_img)
        attn_seq = attn_seq.squeeze(1)  # (B,D_seq)

        # gate between original and attended sequence
        γ_seq = self.sigmoid(self.alpha_seq)             # scalar in (0,1)
        fused_seq = γ_seq * attn_seq + (1-γ_seq) * cls_seq

        q_img = cls_img.unsqueeze(1)
        k_seq = cls_seq.unsqueeze(1)
        v_seq = k_seq
        attn_img, _ = self.cross_img_to_seq(q_img, k_seq, v_seq)
        attn_img = attn_img.squeeze(1)  # (B,D_img)

        γ_img = self.sigmoid(self.alpha_img)
        fused_img = γ_img * attn_img + (1-γ_img) * cls_img

        return fused_seq, fused_img
    
class MultiModalClassifier(nn.Module):
    def __init__(self, seq_d_model=256, vit_out_dim=768, n_heads=8,
                 num_layers=4, num_classes=5, vocab_size=None, max_len_seq=200):
        super().__init__()
        self.seq_encoder = ClassifierTransformer(
            d_model=seq_d_model, n_heads=n_heads,
            num_layers=num_layers, num_classes=None,
            vocab_size=vocab_size, max_len=max_len_seq
        )
        self.struct_encoder = ImageViTEncoder()
        self.cross_attn = BidirectionalCrossAttention(
            seq_d_model, vit_out_dim, n_heads
        )
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(seq_d_model + vit_out_dim, num_classes)

    def forward(self, seq_ids, struct_tensor):
        cls_seq = self.seq_encoder.forward_get_cls(seq_ids)
        cls_img = self.struct_encoder.forward_get_cls(struct_tensor)
        cls_seq_att, cls_img_att = self.cross_attn(cls_seq, cls_img)
        h = torch.cat([cls_seq_att, cls_img_att], dim=1)
        return self.classifier(self.dropout(h))



class MultiModalClassifierGate(nn.Module):
    def __init__(self, seq_d_model=256, vit_out_dim=192, n_heads=8,
                 num_layers=4, num_classes=5, vocab_size=None, max_len_seq=200):
        super().__init__()
        self.seq_encoder = ClassifierTransformer(
            d_model=seq_d_model, n_heads=n_heads,
            num_layers=num_layers, num_classes=None,
            vocab_size=vocab_size, max_len=max_len_seq
        )
        self.struct_encoder = ImageViTEncoder()
        # use the gated version here
        self.cross_attn = GatedBidirectionalCrossAttention(
            dim_seq=seq_d_model,
            dim_img=vit_out_dim,
            num_heads=n_heads
        )
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(seq_d_model + vit_out_dim, num_classes)

    def forward(self, seq_ids, struct_tensor):
        cls_seq = self.seq_encoder.forward_get_cls(seq_ids)
        cls_img = self.struct_encoder.forward_get_cls(struct_tensor)
        # this now returns gated fusions
        fused_seq, fused_img = self.cross_attn(cls_seq, cls_img)
        h = torch.cat([fused_seq, fused_img], dim=1)
        return self.classifier(self.dropout(h))