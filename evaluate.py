import os
import glob
import torch
import yaml
import sacrebleu
from tqdm import tqdm
from tokenizers import Tokenizer

# Import kiến trúc model và hàm dịch từ các file đã viết
from src.model.transformer import Transformer
from translate import translate_sentence
from src.utils.helpers import load_jsonl_dataset

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)



def evaluate_checkpoint(model, tokenizer, checkpoint_path, sources, references, direction, max_len, device):
    """Tải trọng số và tính BLEU Score cho 1 checkpoint"""
    # 1. Tải trọng số
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    predictions = []

    # 2. Dịch từng câu trong tập Validation
    print(f"\nĐang đánh giá: {os.path.basename(checkpoint_path)}")
    for src in tqdm(sources, desc="Translating", leave=False):
        pred_text = translate_sentence(
            model=model,
            tokenizer=tokenizer,
            sentence=src,
            direction=direction,
            max_len=max_len,
            device=device
        )
        predictions.append(pred_text)

    # 3. Tính điểm BLEU bằng SacreBLEU
    # SacreBLEU yêu cầu references phải nằm trong list of lists: [ [ref1, ref2], [ref1, ref2] ]
    bleu = sacrebleu.corpus_bleu(predictions, [references])

    return bleu.score


def main():
    cfg = load_config()
    model_cfg = cfg['model']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Đang khởi tạo môi trường đánh giá...")
    tokenizer = Tokenizer.from_file("data/processed/tokenizer-envi.json")

    model = Transformer(
        vocab_size=model_cfg['vocab_size'],
        d_model=model_cfg['d_model'],
        num_heads=model_cfg['num_heads'],
        num_layers=model_cfg['num_layers'],
        d_ff=model_cfg['d_ff']
    ).to(device)

    # Lấy danh sách toàn bộ các file .pt trong thư mục checkpoints, sắp xếp theo thời gian tạo
    checkpoint_files = sorted(glob.glob("checkpoints/*.pt"), key=os.path.getmtime)

    if not checkpoint_files:
        print("Không tìm thấy file checkpoint nào trong thư mục 'checkpoints/'.")
        return

    # Lấy 500 câu Validation theo chiều Anh -> Việt để test
    val_file = "data/processed/val_data.jsonl"  # Nhớ tạo file này nhé
    direction_to_test = "en2vi"

    print(f"Đang tải dữ liệu Validation từ {val_file}...")
    sources, references = load_jsonl_dataset(val_file, direction=direction_to_test, limit=500)

    print(f"Số lượng câu đánh giá: {len(sources)} câu ({direction_to_test})")
    print("-" * 50)

    # Từ điển lưu kết quả
    results = {}

    # Quét qua từng checkpoint để chấm điểm
    for ckpt in checkpoint_files:
        score = evaluate_checkpoint(
            model=model,
            tokenizer=tokenizer,
            checkpoint_path=ckpt,
            sources=sources,
            references=references,
            direction=direction_to_test,
            max_len=model_cfg['max_seq_len'],
            device=device
        )
        results[os.path.basename(ckpt)] = score
        print(f"-> Điểm BLEU: {score:.2f}")

    # In Bảng tổng sắp
    print("\n" + "=" * 40)
    print(" BẢNG TỔNG SẮP BLEU SCORE".center(40))
    print("=" * 40)

    best_ckpt = None
    best_score = -1

    for ckpt, score in results.items():
        print(f"{ckpt:<30} : {score:>5.2f}")
        if score > best_score:
            best_score = score
            best_ckpt = ckpt

    print("-" * 40)
    print(f"🏆 Checkpoint tốt nhất: {best_ckpt} (BLEU: {best_score:.2f})")


if __name__ == "__main__":
    main()