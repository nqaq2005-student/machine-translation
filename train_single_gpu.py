import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
import json
from tqdm import tqdm

from src.model.transformer import Transformer
from src.data_pipeline.dataset import BilingualDataset
from src.utils.metrics import calculate_bleu
from src.utils.helpers import load_jsonl_data, get_lr_scheduler, get_latest_checkpoint
from src.utils.helpers import load_config, get_compute_dtype


def main():
    # 1. LOAD CONFIG VÀ KIỂM TRA PHẦN CỨNG (T4 vs A100)
    cfg = load_config()
    model_cfg = cfg['model']
    train_cfg = cfg['training']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Đang khởi động trên thiết bị: {device.type} (SINGLE GPU)")

    # Tối ưu hóa phần cứng động (Dynamic Hardware Optimization)
    compute_dtype, use_scaler = get_compute_dtype()


    # 2. KHỞI TẠO DỮ LIỆU
    tokenizer = Tokenizer.from_file("data/processed/tokenizer-envi.json")
    print("⏳ Đang tải dữ liệu huấn luyện vào RAM...")
    data_list = []
    with open("data/processed/train_data.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data_list.append(json.loads(line))
            if len(data_list) >= 8000000:
                break

    print(f"📊 Tổng số cặp câu huấn luyện: {len(data_list)}")
    dataset = BilingualDataset(data_list, tokenizer, model_cfg['max_seq_len'])
    dataloader = DataLoader(dataset, batch_size=train_cfg['batch_size'], shuffle=True, num_workers=2)

    # 3. KHỞI TẠO MÔ HÌNH (THUẦN SINGLE GPU)
    model = Transformer(
        vocab_size=model_cfg['vocab_size'],
        d_model=model_cfg['d_model'],
        num_heads=model_cfg['num_heads'],
        num_layers=model_cfg['num_layers'],
        d_ff=model_cfg['d_ff'],
        dropout=model_cfg['dropout']
    ).to(device)

    # 4. KHỞI TẠO BỘ TỐI ƯU VÀ SCALER
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg['learning_rate'], eps=1e-9)
    lr_scheduler = get_lr_scheduler(optimizer, warmup_steps=4000)
    loss_fn = nn.CrossEntropyLoss(ignore_index=dataset.pad_id, label_smoothing=train_cfg['label_smoothing'])

    scaler = torch.amp.GradScaler(enabled=use_scaler)

    # 5. KHÔI PHỤC CHECKPOINT
    start_epoch = 0
    global_step = 0
    checkpoint_dir = "checkpoints"

    if checkpoint_dir:
        latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            print(f"🔄 Tìm thấy Checkpoint: {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)

            # ⚡ Tự động lột vỏ "module." nếu lỡ load phải file checkpoint cũ của Đa GPU
            state_dict = checkpoint['model_state_dict']
            clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

            model.load_state_dict(clean_state_dict)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            start_epoch = checkpoint.get('epoch', 0)
            global_step = checkpoint.get('global_step', 0)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

            print(f"✅ Khôi phục thành công! Sẽ tiếp tục từ Epoch {start_epoch + 1}, Step {global_step}.")
        else:
            print("✨ Bắt đầu huấn luyện từ đầu (From scratch).")
    else:
        print("✨ Bắt đầu huấn luyện từ đầu (From scratch).")

    val_data = load_jsonl_data("data/processed/val_data.jsonl", limit=320)

    # 6. VÒNG LẶP HUẤN LUYỆN
    model.train()
    save_step_freq = 10000

    for epoch in range(start_epoch, train_cfg['epochs']):
        total_loss = 0
        batches_to_skip = global_step % len(dataloader) if epoch == start_epoch else 0

        batch_iterator = tqdm(
            dataloader,
            desc=f"🚀 Epoch {epoch + 1:02d}/{train_cfg['epochs']}",
            leave=True,
            file=sys.stdout,
            colour='cyan',
            bar_format="{desc} {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        )

        for step_idx, batch in enumerate(batch_iterator):
            if epoch == start_epoch and step_idx < batches_to_skip:
                batch_iterator.set_postfix_str(f"⏭️ Đang tua nhanh qua {batches_to_skip} steps cũ...")
                continue

            global_step += 1

            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type='cuda', dtype=compute_dtype):
                logits = model(encoder_input, decoder_input, encoder_mask, decoder_mask)
                loss = loss_fn(logits.view(-1, model_cfg['vocab_size']), label.view(-1))

            scaler.scale(loss).backward() #type: ignore
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            total_loss += loss.item()

            batch_iterator.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
            })

            # Lưu Checkpoint theo Step
            if global_step % save_step_freq == 0:
                checkpoint_path = f"checkpoints/transformer_step_{global_step}.pt"

                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'loss': loss.item(),
                }, checkpoint_path)

                tqdm.write(f"💾 [Lưu] Step {global_step} | Loss: {loss.item():.4f}  -> {checkpoint_path}")

        avg_loss = total_loss / len(dataloader)
        epoch_bleu = 0.0

        if val_data:
            tqdm.write(f"\n-> Đang tính BLEU tổng kết cuối Epoch {epoch + 1}...")
            epoch_bleu = calculate_bleu(model, tokenizer, val_data, model_cfg['max_seq_len'], device, compute_dtype=compute_dtype)

        tqdm.write(f"\n🏁 --- KẾT THÚC EPOCH {epoch + 1} | Avg Loss: {avg_loss:.4f} | BLEU: {epoch_bleu:.2f} ---")

        # Lưu Checkpoint theo Epoch
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'loss': avg_loss,
            'bleu_score': epoch_bleu,
        }, f"checkpoints/transformer_epoch_{epoch + 1}.pt")


if __name__ == "__main__":
    main()