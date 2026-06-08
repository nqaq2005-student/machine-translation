import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadAttention


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Sử dụng hàm kích hoạt GELU để hội tụ tốt hơn
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # Pre-LN: Norm -> Attention -> Dropout -> Add
        norm_x = self.layer_norm1(x)
        attn_out = self.self_attn(query=norm_x, key=norm_x, value=norm_x, mask=src_mask)
        x = x + self.dropout(attn_out)

        # Pre-LN: Norm -> FFN -> Dropout -> Add
        norm_x2 = self.layer_norm2(x)
        ff_out = self.feed_forward(norm_x2)
        x = x + self.dropout(ff_out)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        # Cross-Attention
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # 1. Masked Self-Attention (Chỉ được nhìn các từ trong quá khứ)
        norm_x = self.layer_norm1(x)
        self_attn_out = self.self_attn(query=norm_x, key=norm_x, value=norm_x, mask=tgt_mask)
        x = x + self.dropout(self_attn_out)

        # 2. Cross-Attention (Query từ Decoder, Key & Value từ Encoder)
        norm_x2 = self.layer_norm2(x)
        cross_attn_out = self.cross_attn(query=norm_x2, key=encoder_output, value=encoder_output, mask=src_mask)
        x = x + self.dropout(cross_attn_out)

        # 3. Feed Forward
        norm_x3 = self.layer_norm3(x)
        ff_out = self.feed_forward(norm_x3)
        x = x + self.dropout(ff_out)

        return x