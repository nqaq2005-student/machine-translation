import torch
import torch.nn as nn
import math

from .layers import EncoderLayer, DecoderLayer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Khởi tạo ma trận vị trí kích thước (max_seq_len, d_model) chứa các số 0
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # Áp dụng hàm Sin và Cosine
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Các vị trí chẵn dùng Sin
        pe[:, 1::2] = torch.cos(position * div_term)  # Các vị trí lẻ dùng Cos

        # Thêm một chiều batch (1, max_seq_len, d_model) để cộng dễ dàng
        pe = pe.unsqueeze(0)

        # Register buffer để pe không bị xem là tham số cần train (requires_grad=False)
        # nhưng vẫn được lưu vào file checkpoint và tự động chuyển sang GPU
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        # Cộng Positional Encoding vào Input Embedding
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        # Stack nhiều EncoderLayer lên nhau bằng nn.ModuleList
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask):
        # src là các index của từ: (batch_size, seq_len)
        x = self.emb(src) * math.sqrt(self.d_model)  # Scale embedding
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, encoder_output, src_mask, tgt_mask):
        x = self.emb(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=1024, dropout=0.1):
        super().__init__()

        # Vì dịch 2 chiều dùng chung tập từ vựng (Joint Vocabulary),
        # ta sử dụng chung cấu hình vocab_size cho cả Encoder và Decoder
        self.encoder = Encoder(vocab_size, d_model, num_layers, num_heads, d_ff, dropout)
        self.decoder = Decoder(vocab_size, d_model, num_layers, num_heads, d_ff, dropout)

        # Lớp Linear cuối cùng để ánh xạ vector về lại không gian từ vựng
        self.generator = nn.Linear(d_model, vocab_size, bias=False)

        # --- THỦ THUẬT QUAN TRỌNG: Weight Tying ---
        # Chia sẻ trọng số giữa lớp Embedding của Encoder, Decoder và lớp Generator cuối cùng.
        # Điều này giúp giảm khoảng 32 triệu tham số thừa, giữ model ở mức ~50M.
        self.decoder.emb.weight = self.encoder.emb.weight
        self.generator.weight = self.encoder.emb.weight

        # Khởi tạo trọng số với phân phối chuẩn (chuẩn mực của Transformer)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # 1. Mã hóa câu nguồn
        encoder_output = self.encoder(src, src_mask)

        # 2. Giải mã để sinh câu đích
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)

        # 3. Tính toán xác suất từ vựng (Logits)
        return self.generator(decoder_output)