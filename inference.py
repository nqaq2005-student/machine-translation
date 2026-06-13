import torch
from tokenizers import Tokenizer
import os
from src.model.transformer import Transformer
from src.utils.helpers import load_config, clean_detokenize, get_latest_checkpoint
from src.utils.translate import translate_sentence




def main():
    print("="*50)
    print("HỆ THỐNG DỊCH MÁY EN <-> VI TRANSFORMER".center(50))
    print("="*50)

    # 1. Khởi tạo cấu hình và thiết bị
    cfg = load_config()
    model_cfg = cfg['model']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    compute_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    ckpt_dir = "checkpoints"
    # 2. Nạp Checkpoint mới nhất
    ckpt_path = get_latest_checkpoint(ckpt_dir)
    if not ckpt_path:
        print("❌ Lỗi: Không tìm thấy file checkpoint nào. Vui lòng train mô hình trước!")
        return

    print(f"⏳ Đang nạp Tokenizer...")
    tokenizer = Tokenizer.from_file("data/processed/tokenizer-envi.json")

    print(f"⏳ Đang nạp Mô hình từ: {os.path.basename(ckpt_path)}...")
    model = Transformer(
        vocab_size=model_cfg['vocab_size'],
        d_model=model_cfg['d_model'],
        num_heads=model_cfg['num_heads'],
        num_layers=model_cfg['num_layers'],
        d_ff=model_cfg['d_ff'],
        dropout=model_cfg['dropout']
    ).to(device)

    # Load trọng số và cởi bỏ lớp áo DDP 'module.'
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    clean_state_dict = {k.removeprefix('module.'): v for k, v in state_dict.items()}
    
    model.load_state_dict(clean_state_dict)
    model.eval()
    print("Hệ thống đã sẵn sàng!\n")

    # 3. Chọn chiều dịch
    print("Vui lòng chọn chiều dịch:")
    print("  [1] Anh -> Việt (en2vi)")
    print("  [2] Việt -> Anh (vi2en)")
    choice = input("Nhập lựa chọn (1 hoặc 2): ").strip()
    
    direction = "en2vi" if choice == "1" else "vi2en"
    dir_label = "EN -> VI" if direction == "en2vi" else "VI -> EN"
    
    print(f"\nĐã kích hoạt chế độ dịch: {dir_label}")
    print("(Gõ 'q' hoặc 'quit' để thoát chương trình)\n")

    # 4. Vòng lặp Chat tương tác
    with torch.no_grad():
        while True:
            # Nhập câu cần dịch
            source_text = input(f"📝 Nhập câu ({dir_label[:2]}): ").strip()
            
            # Xử lý lệnh thoát
            if source_text.lower() in ['q', 'quit', 'exit']:
                print("\n👋 Cảm ơn bạn đã sử dụng hệ thống. Hẹn gặp lại!")
                break

            if not source_text:
                continue

            pred_text = translate_sentence(
                model=model,
                tokenizer=tokenizer,
                sentence=source_text,
                direction=direction,
                max_len=model_cfg['max_seq_len'],
                device=device,
                compute_dtype=compute_dtype
            )

            final_translation = clean_detokenize(pred_text)
            print(f"✨ Bản dịch ({dir_label[-2:]}): {final_translation}\n")
            print("-" * 50)

if __name__ == "__main__":
    main()
    