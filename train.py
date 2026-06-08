import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
import yaml
from tqdm import tqdm
import os
import glob
import sacrebleu
from translate import translate_sentence # Import hàm dịch từ file bạn đã viết
import json
from src.model.transformer import Transformer
from src.data_pipeline.dataset import BilingualDataset
from src.utils.helpers import load_jsonl_dataset

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Tải Mini Validation Set (dùng hàm helper chung)
print("Đang tải Mini Validation Set...")
val_sources, val_references = load_jsonl_dataset("data/processed/val_data.jsonl", limit=30)

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

    # [SỬA LỖI 4]: Chuyển num_workers=-1 thành 4 (hoặc 2)
    dataset = BilingualDataset(data_list, tokenizer, model_cfg['max_seq_len'])
    dataloader = DataLoader(dataset, batch_size=train_cfg['batch_size'], shuffle=True, num_workers=4)

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
    loss_fn = nn.CrossEntropyLoss(ignore_index=dataset.pad_id, label_smoothing=train_cfg['label_smoothing'])

    # =================================================================
    # TÍNH NĂNG AUTO-RESUME
    # =================================================================
    start_epoch = 0
    global_step = 0

    checkpoint_files = glob.glob("checkpoints/*.pt")

    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        print(f"🔄 Tìm thấy Checkpoint: {latest_checkpoint}")
        print("Đang tiến hành khôi phục trạng thái...")

        checkpoint = torch.load(latest_checkpoint, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint.get('epoch', 0)
        global_step = checkpoint.get('global_step', 0)

        if 'epoch_' in latest_checkpoint:
            start_epoch += 1

        print(f"✅ Khôi phục thành công! Sẽ tiếp tục từ Epoch {start_epoch + 1}, Step {global_step}.")
    else:
        print("✨ Không có checkpoint cũ. Bắt đầu huấn luyện từ đầu (From scratch).")
    # =================================================================

    # [SỬA LỖI 2]: Khởi tạo tập Validation trước khi vào vòng lặp
    print("Đang tải Mini Validation Set...")
    val_sources, val_references = load_mini_validation(limit=30)

    # [SỬA LỖI 1]: Đưa vòng lặp ra cùng cấp thụt lề với if/else ở trên
    # 5. VÒNG LẶP HUẤN LUYỆN CHÍNH
    model.train()

    # [SỬA LỖI 3]: Xóa dòng `global_step = 0` ở đây để không đè lên giá trị Resume
    save_step_freq = 200

    for epoch in range(start_epoch, train_cfg['epochs']):
        total_loss = 0

        batch_iterator = tqdm(dataloader, desc=f"Epoch {epoch + 1:02d}/{train_cfg['epochs']}", leave=True)

        for batch in batch_iterator:
            global_step += 1

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

            total_loss += loss.item()
            batch_iterator.set_postfix({"loss": f"{loss.item():.4f}"})

            # Logic lưu checkpoint theo Step
            if global_step % save_step_freq == 0:
                current_bleu = 0.0

                model.eval()

                if val_sources:
                    predictions = []
                    for src in tqdm(val_sources, desc="Tính BLEU nhanh", leave=False):
                        pred_text = translate_sentence(
                            model=model,
                            tokenizer=tokenizer,
                            sentence=src,
                            direction="en2vi",
                            max_len=model_cfg['max_seq_len'],
                            device=device
                        )
                        predictions.append(pred_text)

                    bleu_result = sacrebleu.corpus_bleu(predictions, [val_references])
                    current_bleu = bleu_result.score

                model.train()

                batch_iterator.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "bleu": f"{current_bleu:.2f}"
                })

                checkpoint_path = f"checkpoints/transformer_step_{global_step}.pt"
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    'bleu_score': current_bleu,
                }, checkpoint_path)

                tqdm.write(
                    f"[Lưu] Step {global_step} | Loss: {loss.item():.4f} | BLEU: {current_bleu:.2f} -> {checkpoint_path}")

        # Tính Loss trung bình của Epoch (đã xóa đoạn bị lặp double)
        avg_loss = total_loss / len(dataloader)

        # Tính điểm BLEU chốt sổ cuối Epoch
        epoch_bleu = 0.0
        if val_sources:
            model.eval()
            predictions = []

            for src in tqdm(val_sources, desc=f"Tính BLEU cuối Epoch {epoch + 1}", leave=False):
                pred_text = translate_sentence(
                    model=model,
                    tokenizer=tokenizer,
                    sentence=src,
                    direction="en2vi",
                    max_len=model_cfg['max_seq_len'],
                    device=device
                )
                predictions.append(pred_text)

            epoch_bleu = sacrebleu.corpus_bleu(predictions, [val_references]).score
            model.train()

        print(f"--- KẾT THÚC EPOCH {epoch + 1} | Avg Loss: {avg_loss:.4f} | BLEU: {epoch_bleu:.2f} ---")

        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'bleu_score': epoch_bleu,
        }, f"checkpoints/transformer_epoch_{epoch + 1}.pt")

if __name__ == "__main__":
    main()