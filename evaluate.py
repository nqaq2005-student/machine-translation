import os
import glob
import torch
import yaml
import sacrebleu
from tqdm import tqdm
from tokenizers import Tokenizer

from src.model.transformer import Transformer
from translate import translate_sentence
from src.utils.helpers import load_jsonl_dataset
from src.utils.helpers import load_config


def evaluate_checkpoint(model, tokenizer, checkpoint_path, sources, references, direction, max_len, device,
                        compute_dtype):
    """Tải trọng số và tính BLEU Score cho 1 checkpoint"""
    # 1. Tải trọng số
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']

    # --- SỬA LỖI KEY DATAPARALLEL KHI LOAD TRỌNG SỐ TỪ FILE ---
    from collections import OrderedDict
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.eval()

    predictions = []

    # 2. Dịch từng câu trong tập Validation
    print(f"\nĐang đánh giá: {os.path.basename(checkpoint_path)}")
    for src in tqdm(sources, desc="Translating", leave=False):
        # ⚡ Truyền compute_dtype vào hàm dịch
        pred_text = translate_sentence(
            model=model,
            tokenizer=tokenizer,
            sentence=src,
            direction=direction,
            max_len=max_len,
            device=device,
            compute_dtype=compute_dtype
        )
        predictions.append(pred_text)

    # 3. Tính điểm BLEU bằng SacreBLEU
    if not predictions:
        return 0.0
    bleu = sacrebleu.corpus_bleu(predictions, [references])

    return bleu.score


def main():
    cfg = load_config()
    model_cfg = cfg['model']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    compute_dtype = torch.float32
    if torch.cuda.is_available():
        if torch.cuda.get_device_capability(0)[0] >= 8:
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = torch.float16

    print("Đang khởi tạo môi trường đánh giá...")
    tokenizer = Tokenizer.from_file("data/processed/tokenizer-envi.json")

    # ⚡ Bổ sung tham số dropout
    model = Transformer(
        vocab_size=model_cfg['vocab_size'],
        d_model=model_cfg['d_model'],
        num_heads=model_cfg['num_heads'],
        num_layers=model_cfg['num_layers'],
        d_ff=model_cfg['d_ff'],
        dropout=model_cfg['dropout']
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
            device=device,
            compute_dtype=compute_dtype  # ⚡ Truyền biến vào hàm
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