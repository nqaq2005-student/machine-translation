import os
import json
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from tokenizers import Tokenizer
from tqdm import tqdm
import sys
from datasets import load_dataset

from src.model.transformer import Transformer
from src.data_pipeline.dataset import BilingualDataset
from src.utils.metrics import calculate_bleu
from src.utils.helpers import load_mixed_validation, get_lr_scheduler, load_config


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def main():
    # 1. Tải cấu hình và thiết lập DDP cho đa GPU
    cfg = load_config()
    model_cfg = cfg['model']
    train_cfg = cfg['training']

    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    setup(rank, world_size)

    compute_dtype = torch.float16
    use_scaler = True

    # 2. Khởi tạo dữ liệu

    tokenizer = Tokenizer.from_file("data/processed/tokenizer-envi.json")

    dataset_hf = load_dataset("json", data_files="data/processed/train_data.jsonl", split="train")
    num_rows_to_take = min(train_cfg['num_rows'], len(dataset_hf))
    dataset_hf = dataset_hf.select(range(num_rows_to_take))
    

    dataset = BilingualDataset(dataset_hf, tokenizer, model_cfg['max_seq_len'])
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset, 
        batch_size=train_cfg['batch_size'], 
        sampler=sampler, 
        num_workers=2,       # Chia đều 4 core cho 2 tiến trình GPU
        pin_memory=True,     # Giữ nguyên True: Ép RAM đẩy dữ liệu thẳng vào VRAM cực nhanh
        drop_last=True       # Nên thêm cờ này nếu kích thước batch cuối cùng không đều
    )

    # 3. Khởi tạo mô hình và DDP
    model = Transformer(
        vocab_size=model_cfg['vocab_size'],
        d_model=model_cfg['d_model'],
        num_heads=model_cfg['num_heads'],
        num_layers=model_cfg['num_layers'],
        d_ff=model_cfg['d_ff'],
        dropout=model_cfg['dropout']
    ).to(rank)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])


    # 4. KHỞI TẠO BỘ TỐI ƯU VÀ SCALER

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg['learning_rate'], eps=1e-9)
    lr_scheduler = get_lr_scheduler(optimizer, warmup_steps=4000)
    loss_fn = nn.CrossEntropyLoss(ignore_index=dataset.pad_id, label_smoothing=train_cfg['label_smoothing'])

    scaler = torch.amp.GradScaler(device='cuda', enabled=use_scaler)

    # 5. KHÔI PHỤC CHECKPOINT
    start_epoch = 0
    global_step = 0
    checkpoint_files = glob.glob("checkpoints/*.pt")
    
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        if rank == 0:
            print(f"🔄 Tìm thấy Checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=f'cuda:{rank}')
        model.module.load_state_dict(checkpoint['model_state_dict'])
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']

        if rank == 0:
            print(f"Đã khôi phục thành công! Tiếp tục từ epoch {start_epoch+1}, global step {global_step}.")
    
    val_data = load_mixed_validation("data/processed/val_data.jsonl", limit_per_direction=150)

    # 6. Vòng lặp huấn luyện chính
    model.train()
    save_step_freq = train_cfg['save_checkpoint_freq']

    for epoch in range(start_epoch, train_cfg['epochs']):
        total_loss = 0
        dataloader.sampler.set_epoch(epoch)
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{train_cfg['epochs']}", 
            disable=(rank != 0),
            bar_format="{desc} {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
            file=sys.stdout,      # Ép ghi thẳng ra luồng chuẩn
            mininterval=1.0,     # Chỉ in cập nhật log mỗi 1 giây (tránh làm file log quá dài)
            ascii=True
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Lấy dữ liệu từ dictionary và đẩy lên GPU
            encoder_input = batch['encoder_input'].to(rank)
            decoder_input = batch['decoder_input'].to(rank)
            encoder_mask = batch['encoder_mask'].to(rank)
            decoder_mask = batch['decoder_mask'].to(rank)
            label = batch['label'].to(rank)
    
            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda', dtype=compute_dtype, enabled=use_scaler):
                output = model(encoder_input, decoder_input, encoder_mask, decoder_mask)
                loss = loss_fn(output.reshape(-1, output.shape[-1]), label.reshape(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            total_loss += loss.item()
            global_step += 1

            if rank == 0:
                progress_bar.set_postfix({
                    "loss": f"{total_loss/(batch_idx+1):.4f}", 
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
                })

                if global_step % save_step_freq == 0:
                    checkpoint_path = f"checkpoints/checkpoint_step{global_step}.pt"
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'loss': loss.item(),
                    }, checkpoint_path)
                    print(f"\nĐã lưu checkpoint tại {checkpoint_path}")

        avg_loss = total_loss / len(dataloader)
        epoch_bleu = 0.0    
        if rank == 0:
            if val_data:
                tqdm.write(f"\n-> Đang tính BLEU tổng kết cuối Epoch {epoch + 1}...")
                epoch_bleu = calculate_bleu(
                    model, 
                    tokenizer, 
                    val_data, 
                    model_cfg['max_seq_len'], 
                    device=f'cuda:{rank}', 
                    compute_dtype=torch.float16
                )
                
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | BLEU: {epoch_bleu:.2f}")
            
            checkpoint_path = f"checkpoints/checkpoint_epoch{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': avg_loss,
                'bleu': epoch_bleu
            }, checkpoint_path)
            print(f"\nĐã lưu checkpoint tại {checkpoint_path}")

        dist.barrier() # Đồng bộ hóa tất cả các tiến trình trước khi tiếp tục sang epoch tiếp theo

    cleanup()


if __name__ == "__main__":
    main()