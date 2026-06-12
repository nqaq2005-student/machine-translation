import torch
import sacrebleu
from translate import translate_sentence


def calculate_bleu(model, tokenizer, val_data, max_len, device,                      ):
    """
    Tính điểm BLEU cho danh sách hỗn hợp các câu en2vi và vi2en.
    """
    model.eval()
    predictions = []
    references = []

    # Sử dụng torch.no_grad() để tránh tràn VRAM
    with torch.no_grad():
        for item in val_data:
            src_text = item["src"]
            ref_text = item["tgt"]
            direction = item["direction"]  # Lấy chiều dịch (en2vi hoặc vi2en) từ dữ liệu

            pred_text = translate_sentence(
                model=model,
                tokenizer=tokenizer,
                sentence=src_text,
                direction=direction,
                max_len=max_len,
                device=device,
                compute_dtype=compute_dtype
            )

            predictions.append(pred_text)
            references.append(ref_text)

    model.train()

    # sacrebleu yêu cầu references phải nằm trong list của list
    if not predictions:
        return 0.0

    bleu_result = sacrebleu.corpus_bleu(predictions, [references])
    return bleu_result.score