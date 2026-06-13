import torch
import sacrebleu
from tqdm import tqdm
from .translate import translate_batch
from .helpers import clean_detokenize


def calculate_bleu(model, tokenizer, val_data, max_len, device, compute_dtype, batch_size=32):
    """
    Tính điểm BLEU cho danh sách hỗn hợp các câu en2vi và vi2en bằng Batch Inference.
    """
    model.eval()
    predictions = []
    references = []

    # Cắt dữ liệu thành từng khối (chunk) có độ lớn bằng batch_size
    with torch.no_grad():
        for i in tqdm(range(0, len(val_data), batch_size), desc="Đang dịch (Batched)"):
            # Trích xuất 1 batch dữ liệu
            batch = val_data[i:i + batch_size]

            src_texts = [item["src"] for item in batch]
            ref_texts = [item["tgt"] for item in batch]
            directions = [item["direction"] for item in batch]

            # Gọi hàm dịch theo batch
            pred_texts = translate_batch(
                model=model,
                tokenizer=tokenizer,
                sentences=src_texts,
                directions=directions,
                max_len=max_len,
                device=device,
                compute_dtype=compute_dtype
            )

            # Dọn dẹp khoảng trắng cho toàn bộ batch
            clean_preds = [clean_detokenize(text) for text in pred_texts]
            clean_refs = [clean_detokenize(text) for text in ref_texts]

            predictions.extend(clean_preds)
            references.extend(clean_refs)

    model.train()

    if not predictions:
        return 0.0

    bleu_result = sacrebleu.corpus_bleu(predictions, [references])
    return bleu_result.score