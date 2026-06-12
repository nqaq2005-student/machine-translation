import os
import glob
import torch
import sacrebleu
import re
import csv
from tqdm import tqdm
from tokenizers import Tokenizer

from src.model.transformer import Transformer
from translate import translate_sentence
from src.utils.helpers import load_jsonl_dataset, load_config

def clean_detokenize(text):
    """Dọn dẹp khoảng trắng thừa trước dấu câu để chuẩn hóa đánh giá SacreBLEU."""
    return re.sub(r'\s+([?.!,:;])', r'\1', text)

def evaluate_checkpoint(model, tokenizer, checkpoint_path, sources, references, direction, max_len, device, compute_dtype):
    """Tải trọng số và tính BLEU Score cho 1 checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint) 

    # Cởi bỏ áo DDP 'module.' một cách chuyên nghiệp
    new_state_dict = {k.removeprefix('module.'): v for k, v in state_dict.items()}

    model.load_state_dict(new_state_dict)
    model.eval()

    predictions = []

    print(f"  Đang dịch: {os.path.basename(checkpoint_path)}...")
    with torch.no_grad():
        for src in tqdm(sources, desc="  Tiến độ", leave=False):
            pred_text = translate_sentence(
                model=model,
                tokenizer=tokenizer,
                sentence=src,
                direction=direction,
                max_len=max_len,
                device=device,
                compute_dtype=compute_dtype
            )
            predictions.append(clean_detokenize(pred_text))

    if not predictions:
        return 0.0
        
    cleaned_references = [clean_detokenize(ref) for ref in references]
    bleu = sacrebleu.corpus_bleu(predictions, [cleaned_references])
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

    model = Transformer(
        vocab_size=model_cfg['vocab_size'],
        d_model=model_cfg['d_model'],
        num_heads=model_cfg['num_heads'],
        num_layers=model_cfg['num_layers'],
        d_ff=model_cfg['d_ff'],
        dropout=model_cfg['dropout']
    ).to(device)

    checkpoint_files = sorted(glob.glob("checkpoints/*.pt"), key=os.path.getmtime)
    if not checkpoint_files:
        print("❌ Không tìm thấy file checkpoint nào trong thư mục 'checkpoints/'.")
        return

    val_file = "data/processed/val_data.jsonl" 
    
    # DANH SÁCH 2 CHIỀU CẦN ĐÁNH GIÁ
    directions_to_test = ["en2vi", "vi2en"]

    for direction in directions_to_test:
        print("\n" + "=" * 60)
        print(f"🚀 BẮT ĐẦU ĐÁNH GIÁ CHIỀU: {direction.upper()} ".center(60))
        print("=" * 60)

        # ⚡ BỎ GIỚI HẠN: Không dùng biến limit nữa để nạp full tập dữ liệu
        print(f"Đang nạp toàn bộ dữ liệu Validation ({direction})...")
        sources, references = load_jsonl_dataset(val_file, direction=direction)
        print(f"-> Đã nạp xong {len(sources)} câu.")
        print("-" * 60)

        results = {}
        best_ckpt = None
        best_score = -1

        # Quét qua từng checkpoint
        for ckpt in checkpoint_files:
            score = evaluate_checkpoint(
                model=model,
                tokenizer=tokenizer,
                checkpoint_path=ckpt,
                sources=sources,
                references=references,
                direction=direction,
                max_len=model_cfg['max_seq_len'],
                device=device,
                compute_dtype=compute_dtype 
            )
            ckpt_name = os.path.basename(ckpt)
            results[ckpt_name] = score
            print(f"  -> Điểm BLEU: {score:.2f}\n")
            
            if score > best_score:
                best_score = score
                best_ckpt = ckpt_name

        # --- LƯU KẾT QUẢ RA FILE CSV ---
        output_filename = f"bleu_results_{direction}.csv"
        # Mở file với utf-8 đề phòng tên file có ký tự lạ
        with open(output_filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Checkpoint', 'BLEU_Score']) # Viết Header
            for ckpt_name, score in results.items():
                writer.writerow([ckpt_name, f"{score:.2f}"])
                
        print("-" * 60)
        print(f"🏆 [TỔNG KẾT {direction.upper()}] Checkpoint tốt nhất: {best_ckpt} (BLEU: {best_score:.2f})")
        print(f"💾 Đã lưu toàn bộ điểm số vào file: {output_filename}")

if __name__ == "__main__":
    main()