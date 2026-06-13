import torch
import yaml
import re
import os
import glob
import json
from typing import List, Dict, Optional

def get_step_number(filepath : str) -> int:

    filename = os.path.basename(filepath)

    match = re.search(r'step(\d+)', filename)
    if match:
        return int(match.group(1))  # Ép về số nguyên (VD: 20000)
    return -1  # Nếu là file rác không có số, cho điểm -1 để nó xếp bét


def get_latest_checkpoint(ckpt_dir="checkpoints"):
    """Tự động tìm file checkpoint mới nhất trong thư mục."""
    files = glob.glob(os.path.join(ckpt_dir, "*.pt"))
    if not files:
        return None

    return max(files, key=get_step_number)


def get_lr_scheduler(optimizer, warmup_steps=4000):

    def lr_lambda(step):
        current_step = step + 1
        if current_step <= warmup_steps:
            return current_step / warmup_steps
        return (warmup_steps / current_step) ** 0.5
        
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def clean_detokenize(text):
    """Dọn dẹp khoảng trắng thừa trước dấu câu để văn bản tự nhiên hơn."""
    return re.sub(r'\s+([?.!,:;\'"])', r'\1', text)


def load_jsonl_data(file_path: str, limit: Optional[int] = None, direction: Optional[str] = None) -> List[Dict]:
    """Đọc dữ liệu JSONL, hỗ trợ lọc theo chiều dịch và giới hạn số lượng."""
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())

                if direction is None or item.get("direction") == direction:
                    data.append(item)
                    if limit and len(data) >= limit:
                        break

        return data

    except Exception as e:
        print(f"❌ Lỗi đọc file {file_path}: {e}")
        return []