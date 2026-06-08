import torch
import yaml
from tokenizers import Tokenizer

# Import kiến trúc model
from src.model.transformer import Transformer


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_causal_mask(size, device):
    """Tạo ma trận tam giác dưới để che lấp các từ tương lai trong quá trình sinh từ"""
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int) == 0
    return mask.to(device)


def translate_sentence(model, tokenizer, sentence, direction, max_len, device):
    model.eval()  # Bật chế độ đánh giá (tắt Dropout)

    # Lấy ID của các token đặc biệt
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    pad_id = tokenizer.token_to_id("[PAD]")

    # 1. TIỀN XỬ LÝ ĐẦU VÀO (ENCODER)
    direction_token = tokenizer.token_to_id("<2vi>") if direction == 'en2vi' else tokenizer.token_to_id("<2en>")
    src_tokens = tokenizer.encode(sentence).ids

    # Ghép token định hướng vào đầu, [EOS] vào cuối
    src_input = [direction_token] + src_tokens + [eos_id]
    src_tensor = torch.tensor(src_input, dtype=torch.int64).unsqueeze(0).to(device)  # Shape: (1, seq_len)

    # Encoder Mask (Chỉ chứa 1 câu không có PAD nên mask toàn True)
    encoder_mask = (src_tensor != pad_id).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():  # Tắt tính toán gradient để tiết kiệm VRAM và tăng tốc
        # 2. CHẠY ENCODER MỘT LẦN DUY NHẤT
        encoder_output = model.encoder(src_tensor, encoder_mask)

        # 3. QUÁ TRÌNH SINH TỪ (DECODER - AUTOREGRESSIVE)
        # Bắt đầu với token [BOS]
        decoder_input = torch.tensor([[bos_id]], dtype=torch.int64).to(device)

        for _ in range(max_len):
            # Tạo mask cho chuỗi hiện tại của decoder
            seq_len = decoder_input.size(1)
            causal_mask = generate_causal_mask(seq_len, device)

            # Pad mask cho decoder (tương tự encoder)
            decoder_pad_mask = (decoder_input != pad_id).unsqueeze(0).unsqueeze(0).to(device)
            decoder_mask = decoder_pad_mask & causal_mask

            # Chạy Decoder
            decoder_output = model.decoder(decoder_input, encoder_output, encoder_mask, decoder_mask)

            # Đưa qua lớp Linear (Generator) để lấy xác suất từ vựng
            logits = model.generator(decoder_output)

            # --- GREEDY SEARCH ---
            # Chỉ lấy logits của token ở vị trí cuối cùng (bước thời gian hiện tại)
            next_word_logits = logits[:, -1, :]
            next_word_id = next_word_logits.argmax(dim=-1).item()

            # Nếu dự đoán ra token [EOS] -> Kết thúc câu
            if next_word_id == eos_id:
                break

            # Nếu chưa kết thúc, nối token vừa dự đoán vào chuỗi decoder_input để làm đầu vào cho bước sau
            next_word_tensor = torch.tensor([[next_word_id]], dtype=torch.int64).to(device)
            decoder_input = torch.cat([decoder_input, next_word_tensor], dim=1)

    # 4. GIẢI MÃ KẾT QUẢ (IDs -> Text)
    # Bỏ token [BOS] ở vị trí số 0
    output_ids = decoder_input.squeeze(0).tolist()[1:]
    translated_text = tokenizer.decode(output_ids)

    return translated_text


if __name__ == "__main__":
    # Load cấu hình và thiết bị
    cfg = load_config()
    model_cfg = cfg['model']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Đang tải Tokenizer và Model...")
    tokenizer = Tokenizer.from_file("data/processed/tokenizer-envi.json")

    # Khởi tạo khung mô hình
    model = Transformer(
        vocab_size=model_cfg['vocab_size'],
        d_model=model_cfg['d_model'],
        num_heads=model_cfg['num_heads'],
        num_layers=model_cfg['num_layers'],
        d_ff=model_cfg['d_ff']
    ).to(device)

    # Load trọng số đã huấn luyện (Giả sử bạn muốn test ở Epoch 15)
    # Thay đổi đường dẫn này trỏ tới file checkpoint thực tế của bạn
    checkpoint = torch.load("checkpoints/transformer_epoch_15.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Tải mô hình thành công!\n")
    print("-" * 50)

    # Test thử trực tiếp
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
            device=device
        )
        print(f"[{item['dir'].upper()}] Nguồn: {item['text']}")
        print(f"         Dịch : {result}\n")