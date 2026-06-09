import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
import yaml
from tqdm import tqdm
import os
import glob
import sacrebleu
import json
from src.model.transformer import Transformer
from src.data_pipeline.dataset import BilingualDataset
from src.utils.metrics import calculate_bleu
from src.utils.helpers import load_mixed_validation, get_lr_scheduler # Thêm get_lr_scheduler

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    # 1. Load Cấu hình
    cfg = load_config()
    model_cfg = cfg['model']
    train_cfg = cfg['training']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang huấn luyện trên thiết bị: {device}")

    # 2. Khởi tạo Tokenizer và Dữ liệu
    tokenizer = Tokenizer.from_file("data/processed/tokenizer-envi.json")

    print("Đang tải dữ liệu huấn luyện vào RAM...")
    data_list = []
    with open("data/processed/train_data.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data_list.append(json.loads(line))

    print(f"Tổng số cặp câu huấn luyện: {len(data_list)}")

    dataset = BilingualDataset(data_list, tokenizer, model_cfg['max_seq_len'])
    dataloader = DataLoader(dataset, batch_size=train_cfg['batch_size'], shuffle=True, num_workers=2)

    # 3. Khởi tạo Mô hình
    model = Transformer(
        vocab_size=model_cfg['vocab_size'],
        d_model=model_cfg['d_model'],
        num_heads=model_cfg['num_heads'],
        num_layers=model_cfg['num_layers'],
        d_ff=model_cfg['d_ff'],
        dropout=model_cfg['dropout']
    ).to(device)

    # 4. Tối ưu hóa (Optimizer & Loss Function)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg['learning_rate'], eps=1e-9)
    lr_scheduler = get_lr_scheduler(optimizer, warmup_steps=4000, d_model=model_cfg['d_model'])

    loss_fn = nn.CrossEntropyLoss(ignore_index=dataset.pad_id, label_smoothing=train_cfg['label_smoothing'])

    start_epoch = 0
    global_step = 0

    checkpoint_files = glob.glob("checkpoints/*.pt")

    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        print(f"🔄 Tìm thấy Checkpoint: {latest_checkpoint}")
        print("Đang tiến hành khôi phục trạng thái...")

        checkpoint = torch.load(latest_checkpoint, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint.get('epoch', 0)
        global_step = checkpoint.get('global_step', 0)

        if 'scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            print(f" Phát hiện Checkpoint cũ. Đang đồng bộ hóa Scheduler tới bước {global_step}...")
            for _ in range(global_step):
                lr_scheduler.step()

        # Nếu checkpoint cũ thu được từ việc kết thúc trọn vẹn một epoch, ta tăng epoch tiếp theo lên 1
        if 'epoch_' in os.path.basename(latest_checkpoint):
            start_epoch += 1

        print(f"✅ Khôi phục thành công! Sẽ tiếp tục từ Epoch {start_epoch + 1}, Step {global_step}.")
    else:
        print("✨ Không có checkpoint cũ. Bắt đầu huấn luyện từ đầu (From scratch).")

    print("Đang tải Mini Validation Set (150 en2vi + 150 vi2en)...")
    val_data = load_mixed_validation("data/processed/val_data.jsonl", limit_per_direction=50)

    # 5. VÒNG LẶP HUẤN LUYỆN CHÍNH
    model.train()
    save_step_freq = 2000

    for epoch in range(start_epoch, train_cfg['epochs']):
        total_loss = 0
        batches_to_skip = global_step % len(dataloader) if epoch == start_epoch else 0
        batch_iterator = tqdm(dataloader, desc=f"Epoch {epoch + 1:02d}/{train_cfg['epochs']}", leave=True)

        for step_idx, batch in enumerate(batch_iterator):
            global_step += 1

            if epoch == start_epoch and step_idx < batches_to_skip:
                batch_iterator.set_postfix({"Trạng thái": f"Tua nhanh qua {batches_to_skip} steps cũ..."})
                continue

            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(encoder_input, decoder_input, encoder_mask, decoder_mask)
                loss = loss_fn(logits.view(-1, model_cfg['vocab_size']), label.view(-1))


            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            total_loss += loss.item()
            batch_iterator.set_postfix({"loss": f"{loss.item():.4f}"})

            # Logic lưu checkpoint theo Step
            if global_step % save_step_freq == 0:
                current_bleu = 0.0

                if val_data:
                    tqdm.write(f"-> Đang tính BLEU cho {len(val_data)} câu (2 chiều) tại Step {global_step}...")
                    current_bleu = calculate_bleu(model, tokenizer, val_data, model_cfg['max_seq_len'], device)

                batch_iterator.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "bleu": f"{current_bleu:.2f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.6f}"
                })

                checkpoint_path = f"checkpoints/transformer_step_{global_step}.pt"

                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    'loss': loss.item(),
                    'bleu_score': current_bleu,
                }, checkpoint_path)

                tqdm.write(
                    f"[Lưu] Step {global_step} | Loss: {loss.item():.4f} | BLEU: {current_bleu:.2f} -> {checkpoint_path}")

        # Tính Loss trung bình của Epoch
        avg_loss = total_loss / len(dataloader)

        # Tính điểm BLEU chốt sổ cuối Epoch
        epoch_bleu = 0.0
        if val_data:
            tqdm.write(f"-> Đang tính BLEU tổng kết cuối Epoch {epoch + 1}...")
            epoch_bleu = calculate_bleu(model, tokenizer, val_data, model_cfg['max_seq_len'], device)

        tqdm.write(f"--- KẾT THÚC EPOCH {epoch + 1} | Avg Loss: {avg_loss:.4f} | BLEU: {epoch_bleu:.2f} ---")

        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'loss': avg_loss,
            'bleu_score': epoch_bleu,
        }, f"checkpoints/transformer_epoch_{epoch + 1}.pt")

if __name__ == "__main__":
    main()