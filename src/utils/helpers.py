import json
import torch
import random
import numpy as np

def load_jsonl_dataset(file_path, direction="en2vi", limit=None):
    sources, references = [], []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if item.get("direction") == direction:
                    sources.append(item["src"])
                    references.append(item["tgt"])
                    if limit and len(sources) >= limit: break
        return sources, references
    except Exception as e:
        print(f"Lỗi đọc {file_path}: {e}")
        return [], []

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(state, filename):
    torch.save(state, filename)

def load_checkpoint(filename, device):
    return torch.load(filename, map_location=device)

def get_lr_scheduler(optimizer, warmup_steps=4000, d_model=512):
    # Trả về một hàm lambda đơn giản cho learning rate scheduler
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: (d_model**-0.5) * min((step+1)**-0.5, (step+1) * warmup_steps**-1.5)
    )

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_mixed_validation(file_path, limit_per_direction=150):
    """
    Tải tập validation chứa cả 2 chiều dịch.
    Trả về danh sách các dictionary: [{"src": "...", "tgt": "...", "direction": "..."}]
    """
    val_data = []
    count_en2vi = 0
    count_vi2en = 0

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                import json
                item = json.loads(line)

                if item.get("direction") == "en2vi" and count_en2vi < limit_per_direction:
                    val_data.append(item)
                    count_en2vi += 1
                elif item.get("direction") == "vi2en" and count_vi2en < limit_per_direction:
                    val_data.append(item)
                    count_vi2en += 1

                # Dừng đọc nếu đã đủ 300 câu (150 mỗi chiều)
                if count_en2vi >= limit_per_direction and count_vi2en >= limit_per_direction:
                    break

        return val_data
    except Exception as e:
        print(f"Lỗi đọc {file_path}: {e}")
        return []