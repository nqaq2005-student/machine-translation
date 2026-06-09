import json
import os
from datasets import load_dataset
from tqdm import tqdm


def process_split(dataset, split_name, output_file):
    """Hàm dùng chung để làm sạch và lưu dữ liệu cho bất kỳ split nào (train/dev/test)"""
    print(f"\nĐang xử lý tập [{split_name.upper()}] và lưu vào {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    processed_count = 0
    skipped_count = 0

    with open(output_file, 'w', encoding='utf-8') as f:
        # Lấy đúng phần dữ liệu của split tương ứng
        for item in tqdm(dataset[split_name]):
            translation = item.get('translation', item)
            en_text = translation.get('English', '').strip()
            vi_text = translation.get('Vietnamese', '').strip()

            # 1. Bỏ qua các câu rỗng
            if not en_text or not vi_text:
                skipped_count += 1
                continue

            # 2. Bỏ qua các câu quá dài (max_seq_len = 128)
            if len(en_text.split()) > 100 or len(vi_text.split()) > 100:
                skipped_count += 1
                continue

            # Tạo cặp EN -> VI
            en2vi_record = {"src": en_text, "tgt": vi_text, "direction": "en2vi"}
            f.write(json.dumps(en2vi_record, ensure_ascii=False) + '\n')

            # Tạo cặp VI -> EN
            vi2en_record = {"src": vi_text, "tgt": en_text, "direction": "vi2en"}
            f.write(json.dumps(vi2en_record, ensure_ascii=False) + '\n')

            processed_count += 2

    print(f"-> Hoàn tất tập {split_name}: Đã lưu {processed_count} bản ghi (bỏ qua {skipped_count} câu lỗi/dài).")


def create_txt_for_tokenizer(jsonl_file):
    """Trích xuất text để train BPE Tokenizer (Chỉ nên dùng tập Train)"""
    print("\nĐang xuất file raw text cho Tokenizer (chỉ dùng dữ liệu Train)...")
    en_path = "data/raw/corpus.en"
    vi_path = "data/raw/corpus.vi"

    os.makedirs("data/raw", exist_ok=True)

    with open(jsonl_file, 'r', encoding='utf-8') as f, \
            open(en_path, 'w', encoding='utf-8') as f_en, \
            open(vi_path, 'w', encoding='utf-8') as f_vi:

        for line in tqdm(f):
            data = json.loads(line)
            # Chỉ lấy chiều en2vi để tránh lặp từ vựng
            if data['direction'] == 'en2vi':
                f_en.write(data['src'] + '\n')
                f_vi.write(data['tgt'] + '\n')

    print(f"Đã xuất xong text cho Tokenizer tại {en_path} và {vi_path}")


def main(dataset_name="KietReal/Vietnamese-English-translation"):
    print(f"Đang tải toàn bộ dataset {dataset_name} từ Hugging Face...")
    # Không truyền tham số split, thư viện sẽ tải về dạng một DatasetDict chứa cả train, dev, test
    dataset = load_dataset(dataset_name)

    # 1. Xử lý tập Train (Huấn luyện)
    process_split(dataset, "train", "data/processed/train_data.jsonl")

    # 2. Xử lý tập Dev (Dùng làm Validation tính BLEU)
    # File này sẽ tự động ăn khớp với đường dẫn val_file trong evaluate.py của chúng ta
    process_split(dataset, "validation", "data/processed/val_data.jsonl")

    # 3. Tạo data cho Tokenizer từ tập Train
    create_txt_for_tokenizer("data/processed/train_data.jsonl")

    print("\nToàn bộ quy trình tiền xử lý dữ liệu đã thành công!")


if __name__ == "__main__":
    main()