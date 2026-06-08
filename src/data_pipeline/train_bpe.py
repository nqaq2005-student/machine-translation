import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


def train_tokenizer(data_files, vocab_size=32000, save_path="data/processed/tokenizer-envi.json"):
    print(f"Bắt đầu huấn luyện BPE Tokenizer với {len(data_files)} file...")

    # 1. Khởi tạo mô hình BPE cơ bản
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    # 2. Tiền xử lý: Tách từ sơ bộ bằng khoảng trắng (Whitespace)
    tokenizer.pre_tokenizer = Whitespace()

    # 3. Cấu hình Trainer
    # Đưa các token đặc biệt lên đầu để cố định ID của chúng (0, 1, 2, 3, 4, 5)
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]", "<2en>", "<2vi>"]
    )

    # 4. Bắt đầu train trên danh sách file (.txt)
    tokenizer.train(data_files, trainer)

    # 5. Lưu lại model thành file JSON để dùng cho Dataset và quá trình Dịch
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)
    print(f"Hoàn tất! Tokenizer đã được lưu tại: {save_path}")


if __name__ == "__main__":
    # Giả sử bạn đã gom chung tất cả câu tiếng Anh vào train.en và tiếng Việt vào train.vi
    files = ["data/raw/train.en", "data/raw/train.vi"]
    train_tokenizer(files)