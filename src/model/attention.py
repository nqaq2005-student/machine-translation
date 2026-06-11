import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention (nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads     # kich thuoc moi head

        # khoi tao trong so
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Lớp chiếu đầu ra (Output projection)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, query, key, value, mask = None):
        batch_size = query.size(0)

        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            query=Q,
            key=K,
            value=V,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0
        )

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.d_model)

        return self.W_o(attn_output)

