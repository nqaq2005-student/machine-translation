# 🌍 Machine Translation (Anh-Việt)

Dự án dịch máy song ngữ Anh ↔ Việt bằng kiến trúc Transformer xây dựng từ đầu với PyTorch.


## ✨ Tính năng chính

- 🧠 Mô hình Transformer custom với encoder/decoder chuẩn
- ⚡ Adapt `float16`/`bfloat16` tùy theo GPU
- 🖧 Hỗ trợ chạy DataParallel nếu có nhiều GPU
- 🔄 Tự động phục hồi checkpoint khi huấn luyện gián đoạn
- 🔤 Tokenizer HuggingFace BPE cho cả hai ngôn ngữ

## 📂 Cấu trúc Thư mục (Project Structure)

```text
machine-translation/
├── configs/
│   └── config.yaml           # Cấu hình siêu tham số (Model, Training)
├── data/
│   ├── raw/                  # File text thô (chưa xử lý)
│   └── processed/            # File .jsonl đã chia split và thư mục Tokenizer
├── checkpoints/              # Thư mục lưu trọng số (.pt) theo từng step và epoch
├── src/
│   ├── model/                # Mã nguồn kiến trúc Transformer
│   │   ├── attention.py      # Multi-Head Attention
│   │   ├── layers.py         # Các lớp Encoder/Decoder
│   │   └── transformer.py    # Lắp ráp mô hình Transformer
│   ├── data_pipeline/        # Kịch bản xử lý dữ liệu
│   │   ├── dataset.py        # Kế thừa PyTorch Dataset, tạo mask tự động
│   │   ├── prepare_data.py   # Làm sạch, chia tập train/val và lưu dạng JSONL
│   │   └── train_bpe.py      # Kịch bản huấn luyện Tokenizer
│   └── utils/                # Các hàm tiện ích hỗ trợ
│       ├── metrics.py        # Tính toán điểm BLEU
│       └── helpers.py        # Load data, thiết lập Random Seed, LR Scheduler
├── train.py                  # Kịch bản huấn luyện chính (Hỗ trợ Auto-Resume)
├── evaluate.py               # Kịch bản đánh giá, tìm Checkpoint tốt nhất
├── translate.py              # Kịch bản dịch trực tiếp (Inference với Greedy Search)
└── requirements.txt          # Danh sách thư viện phụ thuộc
```

## ⚙️ Setup

1. Clone repository:

```bash
git clone <repository-url>
cd machine-translation
```

2. Tạo và kích hoạt môi trường ảo:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

3. Cài đặt thư viện:

```bash
pip install -r requirements.txt
```

## 🛠️ Hướng dẫn sử dụng

### 1. Chuẩn bị dữ liệu

Thay đổi đường dẫn dữ liệu trong `prepare_data.py` nếu cần:

```python
def main(dataset_name="your-link-dataset"):
```

Chạy script tiền xử lý dữ liệu để tạo file `jsonl` phù hợp:

```bash
python src/data_pipeline/prepare_data.py
```

### 2. Huấn luyện tokenizer BPE

Huấn luyện tokenizer song ngữ và lưu file kết quả vào `data/processed/`:

```bash
python src/data_pipeline/train_bpe.py
```

### 3. Huấn luyện mô hình

Chạy script huấn luyện. `train.py` sẽ tự động phát hiện GPU và chọn chiến lược mixed precision phù hợp:

```bash
python train.py
```

### 4. Đánh giá mô hình

Sử dụng `evaluate.py` để đánh giá checkpoint trên tập validation/test và tính BLEU:

```bash
python evaluate.py
```

### 5. Dịch thử nghiệm

Dịch câu mẫu với mô hình đã lưu trong `checkpoints/`:

```bash
python translate.py
```

## ⚙️ Cấu hình

Các thông số chính nằm trong `configs/config.yaml`.

Ví dụ:

```yaml
model:
  vocab_size: 32000
  d_model: 512
  num_heads: 8
  num_layers: 6
  d_ff: 2048
  dropout: 0.1
  max_seq_len: 128

training:
  batch_size: 256
  epochs: 15
  learning_rate: 5e-4
  label_smoothing: 0.1
```


## ✅ Yêu cầu

- 🐍 Python 3.10+
- 🔥 torch>=2.0.0
- 🧩 tokenizers>=0.13.0
- 📦 datasets>=2.14.0
- 📊 sacrebleu>=2.3.0

## 📜 Licenses

Project được cấp phép theo MIT License.

---

