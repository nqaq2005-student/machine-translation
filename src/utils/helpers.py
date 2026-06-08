import json

def load_jsonl_dataset(file_path, direction="en2vi", limit=None):
    """
    Hàm tiện ích dùng chung để đọc dữ liệu từ file JSON Lines.
    Áp dụng cho cả Train, Mini Validation và Evaluation để tránh lặp code.
    """
    sources = []
    references = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if item.get("direction") == direction:
                    sources.append(item["src"])
                    references.append(item["tgt"])

                    # Nếu có giới hạn limit và đã đạt đủ số lượng thì dừng
                    if limit and len(sources) >= limit:
                        break
        return sources, references
    except Exception as e:
        print(f"Lỗi khi đọc file {file_path}: {e}")
        return [], []