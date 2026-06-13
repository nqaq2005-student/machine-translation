import torch


def generate_causal_mask(size, device):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int) == 0
    return mask.to(device)

def translate_batch(model, tokenizer, sentences, directions, max_len, device, compute_dtype=torch.float16):
    """
    Dịch một mảng (batch) các câu cùng lúc để tối đa hóa công suất GPU.
    - sentences: Danh sách các câu gốc (List[str])
    - directions: Danh sách chiều dịch tương ứng (List[str])
    """
    model.eval()

    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    pad_id = tokenizer.token_to_id("[PAD]")

    batch_size = len(sentences)

    src_tokens_list = []
    max_src_len = 0

    for sentence, direction in zip(sentences, directions):
        dir_token = tokenizer.token_to_id("<2vi>") if direction == 'en2vi' else tokenizer.token_to_id("<2en>")
        tokens = tokenizer.encode(sentence).ids
        seq = [dir_token] + tokens + [eos_id]
        src_tokens_list.append(seq)
        if len(seq) > max_src_len:
            max_src_len = len(seq)

    # Lót Padding (đệm [PAD]) để mọi câu dài bằng nhau (thành ma trận chữ nhật)
    padded_src = [seq + [pad_id] * (max_src_len - len(seq)) for seq in src_tokens_list]
    src_tensor = torch.tensor(padded_src, dtype=torch.int64).to(device)  # [Batch, Seq_len]

    encoder_mask = (src_tensor != pad_id).unsqueeze(1).unsqueeze(2).to(device)

    actual_model = model.module if hasattr(model, 'module') else model
    full_causal_mask = generate_causal_mask(max_len, device)

    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=compute_dtype):
        # Chạy Encoder 1 lần duy nhất cho cả batch
        encoder_output = actual_model.encoder(src_tensor, encoder_mask)

        # Khởi tạo ma trận đầu vào của Decoder chứa toàn [BOS], kích thước [Batch, 1]
        decoder_input = torch.full((batch_size, 1), bos_id, dtype=torch.int64, device=device)

        # Cờ theo dõi: Đánh dấu True cho các câu chưa dịch xong
        unfinished_sents = torch.ones(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            seq_len = decoder_input.size(1)
            causal_mask = full_causal_mask[:, :seq_len, :seq_len]
            decoder_pad_mask = (decoder_input != pad_id).unsqueeze(1).unsqueeze(2).to(device)
            decoder_mask = decoder_pad_mask & causal_mask

            decoder_output = actual_model.decoder(decoder_input, encoder_output, encoder_mask, decoder_mask)
            logits = actual_model.generator(decoder_output)

            # Lấy xác suất của từ cuối cùng cho toàn bộ Batch [Batch, Vocab]
            next_word_logits = logits[:, -1, :]
            next_word_ids = next_word_logits.argmax(dim=-1)  # [Batch]

            # Nếu câu nào đã dịch xong (unfinished_sents == False), ép từ tiếp theo của nó thành [PAD]
            next_word_ids = torch.where(unfinished_sents, next_word_ids, torch.tensor(pad_id, device=device))

            # Cập nhật lại cờ: Nếu có câu nào vừa dự đoán ra [EOS], đánh dấu nó là False
            unfinished_sents = unfinished_sents & (next_word_ids != eos_id)

            # Nối từ mới vào ma trận
            decoder_input = torch.cat([decoder_input, next_word_ids.unsqueeze(1)], dim=1)

            # Nếu TẤT CẢ các câu trong batch đều đã False (gặp EOS), dừng vòng lặp sớm
            if unfinished_sents.max() == 0:
                break

    # Giải mã (Decode) ma trận số thành văn bản
    output_texts = []
    for i in range(batch_size):
        # Cắt bỏ token [BOS] ở vị trí số 0
        ids = decoder_input[i].tolist()[1:]
        clean_ids = []
        for token_id in ids:
            if token_id == eos_id:  # Gặp EOS là chặt đứt đuôi luôn
                break
            if token_id != pad_id:  # Bỏ qua các token lót
                clean_ids.append(token_id)

        output_texts.append(tokenizer.decode(clean_ids))

    return output_texts


def translate_sentence(model, tokenizer, sentence, direction, max_len, device, compute_dtype=torch.float16):
    """
    Hàm dịch 1 câu (Wrapper).
    """
    
    sentences = [sentence]
    directions = [direction]

    results = translate_batch(
        model=model,
        tokenizer=tokenizer,
        sentences=sentences,
        directions=directions,
        max_len=max_len,
        device=device,
        compute_dtype=compute_dtype
    )

    return results[0]