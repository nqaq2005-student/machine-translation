import torch
import yaml
from tokenizers import Tokenizer
import os
import sys

from src.model.transformer import Transformer
from src.utils.helpers import load_config


def generate_causal_mask(size, device):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int) == 0
    return mask.to(device)

def translate_sentence(model, tokenizer, sentence, direction, max_len, device, compute_dtype = torch.float16):
    model.eval()

    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    pad_id = tokenizer.token_to_id("[PAD]")

    direction_token = tokenizer.token_to_id("<2vi>") if direction == 'en2vi' else tokenizer.token_to_id("<2en>")
    src_tokens = tokenizer.encode(sentence).ids

    src_input = [direction_token] + src_tokens + [eos_id]
    src_tensor = torch.tensor(src_input, dtype=torch.int64).unsqueeze(0).to(device)
    encoder_mask = (src_tensor != pad_id).unsqueeze(0).unsqueeze(0).to(device)

    # Lột vỏ bọc DataParallel nếu model vẫn đang bị bọc
    actual_model = model.module if hasattr(model, 'module') else model
    full_causal_mask = generate_causal_mask(max_len, device)
    # Kích hoạt ép xung phần cứng (Autocast) để dịch nhanh gấp đôi
    with torch.no_grad(),   torch.amp.autocast(device_type='cuda', dtype=compute_dtype):
        encoder_output = actual_model.encoder(src_tensor, encoder_mask)
        decoder_input = torch.tensor([[bos_id]], dtype=torch.int64).to(device)

        for _ in range(max_len):
            seq_len = decoder_input.size(1)
            causal_mask = full_causal_mask[:, :seq_len, :seq_len]
            decoder_pad_mask = (decoder_input != pad_id).unsqueeze(0).unsqueeze(0).to(device)
            decoder_mask = decoder_pad_mask & causal_mask

            decoder_output = actual_model.decoder(decoder_input, encoder_output, encoder_mask, decoder_mask)
            logits = actual_model.generator(decoder_output)

            next_word_logits = logits[:, -1, :]
            next_word_id = next_word_logits.argmax(dim=-1).item()

            if next_word_id == eos_id:
                break

            next_word_tensor = torch.tensor([[next_word_id]], dtype=torch.int64).to(device)
            decoder_input = torch.cat([decoder_input, next_word_tensor], dim=1)

    output_ids = decoder_input.squeeze(0).tolist()[1:]
    return tokenizer.decode(output_ids)


if __name__ == "__main__":
    cfg = load_config()
    model_cfg = cfg['model']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dò tìm phần cứng để tối ưu tốc độ sinh từ (Inference)
    compute_dtype = torch.float32
    if torch.cuda.is_available():
        if torch.cuda.get_device_capability(0)[0] >= 8:
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = torch.float16

    print(f"🚀 Chế độ dịch đang chạy trên thiết bị: {device} (Định dạng: {compute_dtype})")
    tokenizer = Tokenizer.from_file("data/processed/tokenizer-envi.json")

    model = Transformer(
        vocab_size=model_cfg['vocab_size'],
        d_model=model_cfg['d_model'],
        num_heads=model_cfg['num_heads'],
        num_layers=model_cfg['num_layers'],
        d_ff=model_cfg['d_ff'],
        dropout=model_cfg['dropout']
    ).to(device)

    checkpoint_dir = "checkpoints"
    import glob

    checkpoint_files = glob.glob(f"{checkpoint_dir}/*.pt")

    if not checkpoint_files:
        print("❌ Lỗi: Không tìm thấy file checkpoint nào!")
        sys.exit()

    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    print(f"🔄 Đang nạp Checkpoint: {latest_checkpoint}")

    checkpoint = torch.load(latest_checkpoint, map_location=device)
    state_dict = checkpoint['model_state_dict']

    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    print("✅ Tải mô hình thành công!\n" + "-" * 50)

    sentences_to_translate = [
        {"text": "The attention mechanism allows the model to focus on relevant parts of the input.", "dir": "en2vi"},
        {"text": "Mô hình ngôn ngữ lớn đang thay đổi thế giới công nghệ.", "dir": "vi2en"}
    ]

    for item in sentences_to_translate:
        result = translate_sentence(
            model=model,
            tokenizer=tokenizer,
            sentence=item["text"],
            direction=item["dir"],
            max_len=model_cfg['max_seq_len'],
            device=device,
            compute_dtype=compute_dtype
        )
        print(f"[{item['dir'].upper()}] Nguồn: {item['text']}")
        print(f"         Dịch : {result}\n")