# Machine Translation EN ↔ VI

Mô hình dịch máy hai chiều Anh–Việt xây dựng từ đầu bằng PyTorch, dựa trên kiến trúc Transformer với các kỹ thuật hiện đại: **Pre-LayerNorm**, **Weight Tying**, **Mixed Precision**, và **Beam Search** có length penalty.

---

## Tính năng nổi bật

- **Dịch 2 chiều** EN → VI và VI → EN với một mô hình duy nhất, dùng direction token `<2vi>` / `<2en>` thay cho BOS
- **Pre-LayerNorm Transformer** — hội tụ ổn định hơn kiến trúc Post-LN gốc
- **Weight Tying** — chia sẻ trọng số giữa Encoder embedding, Decoder embedding và Generator, giảm ~32M tham số
- **GELU activation** thay ReLU trong Feed-Forward
- **Adaptive Mixed Precision** — tự phát hiện GPU: `bfloat16` trên Ampere (A100, RTX 30/40), `float16 + GradScaler` trên Turing/Volta (T4, V100)
- **Multi-GPU** với PyTorch DDP qua `torchrun`
- **Beam Search** (beam=4, length penalty α=0.6) cho inference; **Greedy** cho tính BLEU nhanh trong evaluate
- **BPE Tokenizer** vocab 32k dùng chung cho cả hai ngôn ngữ
- **Flash Attention** qua `F.scaled_dot_product_attention` (PyTorch 2.0+)

---

## Cấu trúc dự án

```
machine-translation/
├── configs/
│   └── config.yaml              # Siêu tham số mô hình & huấn luyện
├── data/
│   ├── raw/                     # corpus.en, corpus.vi (sinh tự động)
│   └── processed/               # tokenizer-envi.json, train/val .jsonl
├── src/
│   ├── model/
│   │   ├── attention.py         # MultiHeadAttention (Flash Attention qua SDPA)
│   │   ├── layers.py            # EncoderLayer, DecoderLayer (Pre-LN)
│   │   └── transformer.py       # Transformer, Encoder, Decoder, PositionalEncoding
│   ├── data_pipeline/
│   │   ├── prepare_data.py      # Tải dataset HuggingFace, tạo file JSONL
│   │   ├── train_bpe.py         # Huấn luyện BPE Tokenizer
│   │   └── dataset.py           # BilingualDataset (PyTorch Dataset)
│   └── utils/
│       ├── translate.py         # Greedy & Beam Search decoding
│       ├── metrics.py           # Tính BLEU (sacrebleu, batched)
│       └── helpers.py           # Load config, LR scheduler, checkpoint utils
├── train.py                     # Entry point — tự chọn single/multi GPU
├── train_single_gpu.py          # Training loop single GPU
├── train_multi_gpu.py           # Training loop multi-GPU (DDP)
├── evaluate.py                  # Đánh giá toàn bộ checkpoint → CSV
├── inference.py                 # Giao diện dịch tương tác
└── requirements.txt
```

---

## Kiến trúc mô hình

| Tham số | Giá trị |
|---|---|
| d_model | 512 |
| num_heads | 8 |
| num_layers | 6 encoder + 6 decoder |
| d_ff | 1024 |
| vocab_size | 32 000 (dùng chung EN + VI) |
| max_seq_len | 128 |
| dropout | 0.1 |
| Tổng tham số | ~50M |

**Luồng forward:**

```
src → Encoder Embedding × √d_model + PositionalEncoding
    → 6× EncoderLayer [Pre-LN → Self-Attention → Add, Pre-LN → FFN(GELU) → Add]
    → encoder_output

tgt → Decoder Embedding × √d_model + PositionalEncoding
    → 6× DecoderLayer [Pre-LN → Masked Self-Attention → Add,
                        Pre-LN → Cross-Attention(encoder_output) → Add,
                        Pre-LN → FFN(GELU) → Add]
    → Generator (Linear, weight tied với Encoder emb) → logits
```

**Weight Tying:**

```python
self.decoder.emb.weight = self.encoder.emb.weight   # Decoder chia sẻ embedding với Encoder
self.generator.weight   = self.encoder.emb.weight   # Generator chia sẻ luôn
```

Ba ma trận này cùng trỏ đến một tensor duy nhất, giảm ~32M tham số so với tách riêng.

---

## Cài đặt

```bash
git clone https://github.com/nqaq2005-student/machine-translation.git
cd machine-translation
pip install -r requirements.txt
```

**Yêu cầu:** Python ≥ 3.10, PyTorch ≥ 2.0, CUDA (khuyến nghị)

---

## Hướng dẫn sử dụng

### Bước 1 — Chuẩn bị dữ liệu

Tải dataset `KietReal/Vietnamese-English-translation` từ HuggingFace, lọc câu, tạo file JSONL và xuất raw text để train tokenizer:

```bash
python src/data_pipeline/prepare_data.py
```

Kết quả:

```
data/processed/train_data.jsonl   # ~8M cặp câu (en2vi + vi2en)
data/processed/val_data.jsonl
data/raw/corpus.en
data/raw/corpus.vi
```

### Bước 2 — Huấn luyện BPE Tokenizer

```bash
python src/data_pipeline/train_bpe.py
```

Kết quả: `data/processed/tokenizer-envi.json` (vocab 32k, dùng chung cho cả hai ngôn ngữ)

### Bước 3 — Huấn luyện mô hình

**Single GPU:**

```bash
python train_single_gpu.py
```

**Multi-GPU (ví dụ 2 GPU):**

```bash
torchrun --nproc_per_node=2 train_multi_gpu.py
```

Script tự phát hiện compute capability GPU và chọn precision phù hợp. Checkpoint được lưu vào `checkpoints/` mỗi 10 000 bước.

**Cấu hình training** (`configs/config.yaml`):

| Tham số | Giá trị mặc định |
|---|---|
| batch_size | 64 |
| epochs | 15 |
| learning_rate | 5e-4 |
| warmup_steps | 4 000 |
| label_smoothing | 0.1 |
| weight_decay | 0.01 |
| save_checkpoint_freq | 10 000 bước |

LR scheduler theo công thức gốc Transformer: tăng tuyến tính trong `warmup_steps` bước, sau đó giảm theo `1/√step`.

### Bước 4 — Đánh giá

Quét toàn bộ checkpoint trong `checkpoints/`, tính Val Loss và BLEU Score, ghi ra CSV:

```bash
python evaluate.py
```

Kết quả: `evaluation_results.csv`

```
Step,   Train_Loss, Val_Loss, BLEU_Score
10000,  3.2145,     3.1820,   12.45
20000,  2.8932,     2.8640,   18.73
...
```

> BLEU được tính bằng **Greedy decoding** (`beam_size=1`) để đảm bảo tốc độ trong quá trình đánh giá nhiều checkpoint. Điểm thực tế khi dùng Beam Search sẽ cao hơn khoảng 2–5 điểm tuyệt đối.

### Bước 5 — Inference

Giao diện dịch tương tác, dùng **Beam Search** (beam=4) cho chất lượng tốt nhất:

```bash
python inference.py
```

```
══════════════════════════════════════════════════
      HỆ THỐNG DỊCH MÁY EN <-> VI TRANSFORMER
══════════════════════════════════════════════════
Vui lòng chọn chiều dịch:
  [1] Anh -> Việt (en2vi)
  [2] Việt -> Anh (vi2en)
Nhập lựa chọn: 1

📝 Nhập câu (EN): The cat sat on the mat.
✨ Bản dịch (VI): Con mèo ngồi trên tấm thảm.
```

---

## Cơ chế Beam Search

Inference dùng Beam Search (mặc định `beam_size=4`, `length_penalty=0.6`) để tìm câu dịch có xác suất tổng cao nhất thay vì chọn tham lam từng từ.

**Công thức tính điểm:**

```
điểm_cuối = Σ log(pᵢ) / (độ_dài ^ α)
```

Tại mỗi bước, mỗi beam mở rộng ra toàn bộ vocab, tạo `beam_size × vocab_size` ứng viên. Toàn bộ ứng viên được xếp hạng **chung một bảng** và chỉ giữ lại `beam_size` nhánh tốt nhất — không phải top-k riêng từng beam. Length penalty (α=0.6) bù cho việc câu dài bị thiệt thòi khi cộng log-prob âm dần theo bước.

**Tại sao Greedy cho BLEU, Beam cho inference:**

| | Greedy (`evaluate.py`) | Beam Search (`inference.py`) |
|---|---|---|
| Tốc độ | Toàn batch song song, nhanh | Tuần tự từng câu, chậm hơn ~4–6× |
| Chất lượng | Thấp hơn 2–5 BLEU | Câu mượt và chính xác hơn |
| Mục đích | So sánh checkpoint trong training | Bản dịch cuối cho người dùng |

Thứ tự xếp hạng checkpoint theo Greedy BLEU vẫn nhất quán với Beam BLEU — checkpoint tốt nhất theo Greedy vẫn là tốt nhất theo Beam.

**Điều chỉnh tham số:**

```python
from src.utils.translate import translate_sentence, translate_batch

# Inference 1 câu — Beam Search (mặc định beam=4)
result = translate_sentence(
    model, tokenizer, "Hello world",
    direction="en2vi", max_len=128, device=device,
    beam_size=4, length_penalty=0.6
)

# Tính BLEU — Greedy (mặc định beam=1, nhanh)
from src.utils.metrics import calculate_bleu
bleu = calculate_bleu(
    model, tokenizer, val_data,
    max_len=128, device=device,
    compute_dtype=torch.float16,
    batch_size=64          # beam_size=1 theo mặc định
)

# Tính BLEU chính xác bằng Beam (dùng cho báo cáo kết quả cuối)
bleu_final = calculate_bleu(..., batch_size=16, beam_size=4)
```

---

## Yêu cầu phần cứng

| Chế độ | GPU | VRAM |
|---|---|---|
| Single GPU (float16) | NVIDIA T4 / V100 | ≥ 16 GB |
| Single GPU (bfloat16) | NVIDIA A100 / RTX 30xx / 40xx | ≥ 16 GB |
| Multi-GPU | 2× T4 hoặc tốt hơn | 2× 16 GB |
| CPU only | — | RAM ≥ 16 GB (rất chậm) |

---

## Phụ thuộc

```
torch>=2.0.0
tokenizers>=0.13.0
datasets>=2.14.0
PyYAML>=6.0
tqdm>=4.65.0
sacrebleu>=2.3.0
```

---

## License

MIT