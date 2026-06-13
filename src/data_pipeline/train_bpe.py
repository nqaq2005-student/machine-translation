import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.decoders import WordPiece  

def train_tokenizer(data_files, vocab_size=32000, save_path="data/processed/tokenizer-envi.json"):
    print(f"Bắt đầu huấn luyện BPE Tokenizer với {len(data_files)} file...")

    tokenizer = Tokenizer(BPE(unk_token="[UNK]", continuing_subword_prefix="##"))

    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]", "<2en>", "<2vi>"],
        continuing_subword_prefix="##"
    )

    tokenizer.train(data_files, trainer)

    tokenizer.decoder = WordPiece(prefix="##")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)
    print(f"✅ Hoàn tất! Tokenizer đã được lưu tại: {save_path}")

if __name__ == "__main__":
    files = [
        "data/raw/corpus.en",
        "data/raw/corpus.vi"
    ]
    for f in files:
        if not os.path.exists(f):
            print(f"LỖI: Không tìm thấy file {f}")
            exit(1)

    train_tokenizer(files)