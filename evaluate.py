import os
import csv
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from tqdm import tqdm  # ⚡ THÊM THƯ VIỆN THANH TIẾN TRÌNH Ở ĐÂY
from src.model.transformer import Transformer
from src.utils.metrics import calculate_bleu
from src.utils.helpers import get_step_number, load_jsonl_data, load_config
from src.data_pipeline import BilingualDataset


def compute_val_loss(model, val_dataloader, criterion, device, compute_dtype=torch.float16):
    """Quét qua DataLoader để tính Validation Loss trung bình."""
    model.eval()
    total_loss = 0.0
    total_batches = 0

    # ⚡ Bọc val_dataloader bằng tqdm và đặt leave=False để thanh tiến trình biến mất gọn gàng sau khi tính xong
    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=compute_dtype):
        for batch in tqdm(val_dataloader, desc="📉 Đang tính Loss", leave=False):
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            labels = batch['label'].to(device)

            output = model(encoder_input, decoder_input, encoder_mask, decoder_mask)

            # Ép tensor về đúng chiều để tính CrossEntropy (2D logits, 1D labels)
            loss = criterion(output.view(-1, output.size(-1)), labels.view(-1))

            total_loss += loss.item()
            total_batches += 1

    return total_loss / total_batches if total_batches > 0 else 0.0


def evaluate_all_checkpoints(ckpt_dir, output_csv, model, tokenizer, val_dataloader, val_data_list, max_len, batch_size, device):
    """Quét toàn bộ thư mục, nạp từng checkpoint, đánh giá và ghi ra CSV."""

    # 1. Tìm và sắp xếp file checkpoint theo thứ tự thời gian (step)
    files = glob.glob(os.path.join(ckpt_dir, "*.pt"))
    valid_files = [f for f in files if get_step_number(f) != -1]
    sorted_ckpts = sorted(valid_files, key=get_step_number)

    if not sorted_ckpts:
        print(f"❌ Khẩn cấp: Không tìm thấy file checkpoint nào trong thư mục '{ckpt_dir}'.")
        return

    print(f"🔍 Tìm thấy {len(sorted_ckpts)} checkpoints hợp lệ. Bắt đầu đánh giá...")

    # 2. Cài đặt hàm Loss (Chỉ định bỏ qua việc tính loss cho token [PAD])
    pad_id = tokenizer.token_to_id("[PAD]")
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    # 3. Mở file CSV để ghi kết quả liên tục
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Step', 'Train_Loss', 'Val_Loss', 'BLEU_Score'])

        # 4. Lặp qua từng file Checkpoint (Có thể bọc thêm tqdm ở vòng lặp lớn nếu muốn theo dõi tiến độ tổng thể)
        for ckpt_path in sorted_ckpts:
            step = get_step_number(ckpt_path)
            print(f"\n" + "=" * 50)
            print(f"🔄 Đang nạp Checkpoint Step: {step}...")

            # Nạp trọng số an toàn vào thiết bị hiện tại (tránh lỗi lệch GPU)
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            train_loss = checkpoint.get('loss', "N/A")
            if isinstance(train_loss, float):
                train_loss_str = f"{train_loss:.4f}"
            else:
                train_loss_str = str(train_loss)

            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

            # Tự động gỡ bỏ tiền tố 'module.' nếu file được train bằng Multi-GPU (DDP)
            clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(clean_state_dict)

            # Tính điểm Loss
            val_loss = compute_val_loss(model, val_dataloader, criterion, device)
            print(f"Val loss: {val_loss}\n")
            # Tính điểm BLEU
            print("🌐 Đang tính BLEU Score (Batched Inference)...")
            bleu_score = calculate_bleu(
                model=model,
                tokenizer=tokenizer,
                val_data=val_data_list,
                max_len=max_len,
                device=device,
                compute_dtype=torch.float16,
                batch_size=batch_size
            )

            # Ghi kết quả vào CSV
            writer.writerow([step, train_loss_str, f"{val_loss:.4f}", f"{bleu_score:.2f}"])
            f.flush()
            print(
                f"✅ Kết quả Step {step} -> Train Loss: {train_loss_str} | Val Loss: {val_loss:.4f} | BLEU: {bleu_score:.2f}")


    print(f"\n🎉 HOÀN TẤT TẤT CẢ! Dữ liệu đã được lưu an toàn tại: {output_csv}")

def main():
    # --- 1. Thiết lập chung ---
    cfg = load_config()
    model_cfg = cfg['model']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = "checkpoints"
    output_csv = "evaluation_results.csv"
    MAX_LEN = 128

    print(f"🚀 Khởi động Evaluation Script trên thiết bị: {device}")

    # --- 2. Khởi tạo Tokenizer ---
    tokenizer = Tokenizer.from_file("data/processed/tokenizer-envi.json")
    # --- 3. Khởi tạo Model ---
    model = Transformer(
        vocab_size=model_cfg['vocab_size'],
        d_model=model_cfg['d_model'],
        num_heads=model_cfg['num_heads'],
        num_layers=model_cfg['num_layers'],
        d_ff=model_cfg['d_ff'],
        dropout=model_cfg['dropout']
    ).to(device)

    val_data_list = load_jsonl_data("data/processed/val_data.jsonl")

    val_dataset = BilingualDataset(
        data_list=val_data_list,
        tokenizer=tokenizer,
        max_seq_len=MAX_LEN
    )

    # Khởi tạo DataLoader để đóng gói thành Batch cho model
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )

    evaluate_all_checkpoints(
        ckpt_dir=ckpt_dir,
        output_csv=output_csv,
        model=model,
        tokenizer=tokenizer,
        val_dataloader=val_dataloader,
        val_data_list=val_data_list,
        max_len=MAX_LEN,
        batch_size = 64,
        device=device
    )

if __name__ == "__main__":
    main()