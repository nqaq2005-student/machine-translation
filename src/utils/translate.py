import torch
import torch.nn.functional as F
from typing import List


def generate_causal_mask(size: int, device: torch.device) -> torch.Tensor:
    """Tạo causal mask hình tam giác dưới, shape (1, size, size)."""
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int) == 0
    return mask.to(device)


def _encode_sources(
    tokenizer,
    sentences: List[str],
    directions: List[str],
    pad_id: int,
    eos_id: int,
    device: torch.device,
):
    """
    Tokenize và pad toàn bộ câu nguồn thành tensor.
    Trả về (src_tensor, encoder_mask).
    """
    src_tokens_list = []
    for sentence, direction in zip(sentences, directions):
        dir_token = tokenizer.token_to_id("<2vi>") if direction == "en2vi" \
                    else tokenizer.token_to_id("<2en>")
        tokens = tokenizer.encode(sentence).ids
        seq = [dir_token] + tokens + [eos_id]
        src_tokens_list.append(seq)

    max_src_len = max(len(s) for s in src_tokens_list)
    padded = [s + [pad_id] * (max_src_len - len(s)) for s in src_tokens_list]
    src_tensor = torch.tensor(padded, dtype=torch.int64, device=device)
    encoder_mask = (src_tensor != pad_id).unsqueeze(1).unsqueeze(2)  # (B,1,1,L)
    return src_tensor, encoder_mask


# Greedy decoding  (giữ lại để dùng khi beam_size=1 hoặc tính BLEU nhanh)

def _greedy_decode(
    actual_model,
    encoder_output: torch.Tensor,
    encoder_mask: torch.Tensor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Greedy decoding cho một batch.
    Trả về decoder_input tensor (B, T) bao gồm cả token [BOS] đầu.
    """
    batch_size = encoder_output.size(0)
    full_causal_mask = generate_causal_mask(max_len, device)

    decoder_input = torch.full((batch_size, 1), bos_id, dtype=torch.int64, device=device)
    unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        seq_len = decoder_input.size(1)
        causal_mask = full_causal_mask[:, :seq_len, :seq_len]
        dec_pad_mask = (decoder_input != pad_id).unsqueeze(1).unsqueeze(2)
        dec_mask = dec_pad_mask & causal_mask

        dec_out = actual_model.decoder(decoder_input, encoder_output, encoder_mask, dec_mask)
        logits = actual_model.generator(dec_out)                  # (B, T, V)
        next_ids = logits[:, -1, :].argmax(dim=-1)                # (B,)

        next_ids = torch.where(unfinished, next_ids, torch.tensor(pad_id, device=device))
        unfinished = unfinished & (next_ids != eos_id)
        decoder_input = torch.cat([decoder_input, next_ids.unsqueeze(1)], dim=1)

        if not unfinished.any():
            break

    return decoder_input



# Beam search decoding


def _beam_search_decode(
    actual_model,
    encoder_output: torch.Tensor,
    encoder_mask: torch.Tensor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_len: int,
    beam_size: int,
    length_penalty: float,
    device: torch.device,
) -> List[List[int]]:
    """
    Beam search cho từng câu trong batch (xử lý lần lượt).

    Tại sao xử lý lần lượt (not batched across beams)?
    - Batch beam search song song rất phức tạp do câu kết thúc ở các bước khác nhau.
    - Với beam_size nhỏ (4-5) và max_len=128, vòng lặp tuần tự vẫn rất nhanh trên GPU.

    Trả về: List[List[int]] — danh sách token ids (không bao gồm BOS, EOS, PAD)
              cho từng câu trong batch.
    """
    batch_size = encoder_output.size(0)
    full_causal_mask = generate_causal_mask(max_len, device)
    results = []

    for i in range(batch_size):
        # Lấy encoder output của câu thứ i, expand thành beam_size bản sao
        # enc_out_i: (1, src_len, d_model) -> (beam_size, src_len, d_model)
        enc_out_i = encoder_output[i].unsqueeze(0).expand(beam_size, -1, -1)
        enc_mask_i = encoder_mask[i].unsqueeze(0).expand(beam_size, -1, -1, -1)

        # --- Trạng thái của các beam ---
        # sequences: (beam_size, cur_len)  — các token đã sinh kể cả [BOS]
        sequences = torch.full((beam_size, 1), bos_id, dtype=torch.int64, device=device)
        # scores: log-prob tích lũy, shape (beam_size,)
        scores = torch.zeros(beam_size, device=device)
        # Đánh dấu beam nào đã gặp [EOS]
        done = torch.zeros(beam_size, dtype=torch.bool, device=device)
        # Lưu trữ các beam đã hoàn thành (token list, score)
        completed: List[tuple] = []

        vocab_size = None  # sẽ xác định ở bước đầu tiên

        for step in range(max_len - 1):
            # Nếu tất cả beam đã xong, dừng sớm
            if done.all():
                break

            seq_len = sequences.size(1)
            causal_mask = full_causal_mask[:, :seq_len, :seq_len]   # (1, T, T)
            dec_pad_mask = (sequences != pad_id).unsqueeze(1).unsqueeze(2)
            dec_mask = dec_pad_mask & causal_mask                    # (B, 1, T, T)

            dec_out = actual_model.decoder(sequences, enc_out_i, enc_mask_i, dec_mask)
            logits = actual_model.generator(dec_out)                 # (beam, T, V)
            next_logits = logits[:, -1, :]                           # (beam, V)

            if vocab_size is None:
                vocab_size = next_logits.size(-1)

            # Log-softmax để có log-prob của từng từ
            log_probs = F.log_softmax(next_logits, dim=-1)           # (beam, V)

            # Với beam đã done, chỉ cho phép sinh [PAD] (không mở rộng thêm)
            # Trick: đặt log_probs của tất cả từ = -inf, chỉ để [PAD] = 0
            for b in range(beam_size):
                if done[b]:
                    log_probs[b] = float("-inf")
                    log_probs[b, pad_id] = 0.0

            # Cộng với điểm tích lũy: (beam, 1) + (beam, V) = (beam, V)
            total_scores = scores.unsqueeze(1) + log_probs           # (beam, V)

            # Flatten để chọn top beam_size cặp (beam_idx, word_idx) tốt nhất
            # Chỉ expand từ beam đang sống (not done) ở bước đầu, sau đó top-k toàn cục
            flat_scores = total_scores.view(-1)                      # (beam*V,)

            # Lấy top beam_size (số lượng cần giữ)
            topk_scores, topk_flat_ids = flat_scores.topk(beam_size, dim=0, sorted=True)

            # Giải mã flat index thành (beam_idx, word_idx)
            topk_beam_ids = topk_flat_ids // vocab_size              # (beam_size,)
            topk_word_ids = topk_flat_ids % vocab_size               # (beam_size,)

            # Xây dựng sequences mới từ top-k kết quả
            new_sequences = torch.cat(
                [sequences[topk_beam_ids], topk_word_ids.unsqueeze(1)], dim=1
            )                                                        # (beam, T+1)
            new_scores = topk_scores                                  # (beam,)
            new_done = done[topk_beam_ids].clone()

            # Cập nhật cờ done: beam nào vừa sinh [EOS] thì đánh dấu xong
            just_finished = (topk_word_ids == eos_id) & (~new_done)
            for b in range(beam_size):
                if just_finished[b]:
                    # Lưu câu hoàn chỉnh (không gồm BOS và EOS)
                    token_ids = new_sequences[b, 1:].tolist()
                    # Cắt tại EOS nếu còn sót
                    try:
                        eos_pos = token_ids.index(eos_id)
                        token_ids = token_ids[:eos_pos]
                    except ValueError:
                        pass
                    # Áp dụng length penalty: điểm / (độ dài câu ^ alpha)
                    length = max(len(token_ids), 1)
                    penalized = new_scores[b].item() / (length ** length_penalty)
                    completed.append((token_ids, penalized))
                    new_done[b] = True

            sequences = new_sequences
            scores = new_scores
            done = new_done

        # Nếu chưa có câu hoàn thành nào, lấy beam tốt nhất còn đang sống
        if not completed:
            best_ids = sequences[0, 1:].tolist()
            # Bỏ PAD và EOS ở cuối
            clean = []
            for t in best_ids:
                if t == eos_id or t == pad_id:
                    break
                clean.append(t)
            completed.append((clean, scores[0].item()))

        # Chọn câu có penalized score cao nhất
        best_tokens, _ = max(completed, key=lambda x: x[1])
        results.append(best_tokens)

    return results



# Public API


def translate_batch(
    model,
    tokenizer,
    sentences: List[str],
    directions: List[str],
    max_len: int,
    device,
    compute_dtype=torch.float16,
    beam_size: int = 1,
    length_penalty: float = 0.6,
) -> List[str]:
    """
    Dịch một batch câu. Mặc định dùng Greedy (beam_size=1) để tính BLEU nhanh.

    Tham số:
        beam_size     — 1 = greedy (mặc định, dùng cho evaluate/BLEU).
                        4-5 = beam search (dùng cho production inference).
        length_penalty— alpha trong (len ^ alpha): chỉ có tác dụng khi beam_size > 1.
                        0.6 là giá trị chuẩn của Google Neural MT.

    Gọi từ metrics.py (BLEU): beam_size=1  → greedy, nhanh, nhất quán
    Gọi từ inference.py (1 câu): beam_size=4 → beam search, chất lượng cao hơn
    """
    model.eval()

    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    pad_id = tokenizer.token_to_id("[PAD]")

    actual_model = model.module if hasattr(model, "module") else model

    # Xác định device_type cho autocast (chỉ cuda hỗ trợ fp16/bf16)
    if isinstance(device, torch.device):
        device_type = device.type
    else:
        device_type = str(device).split(":")[0]

    use_autocast = device_type == "cuda"

    src_tensor, encoder_mask = _encode_sources(
        tokenizer, sentences, directions, pad_id, eos_id, device
    )

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=compute_dtype)
        if use_autocast
        else torch.amp.autocast(device_type="cpu", enabled=False)
    )

    with torch.no_grad(), autocast_ctx:
        # Encode 1 lần duy nhất cho cả batch
        encoder_output = actual_model.encoder(src_tensor, encoder_mask)

        if beam_size <= 1:
            # Greedy path: nhanh hơn khi chỉ cần tốc độ (debug, BLEU nhanh)
            dec_matrix = _greedy_decode(
                actual_model, encoder_output, encoder_mask,
                bos_id, eos_id, pad_id, max_len, device
            )
            output_texts = []
            for i in range(len(sentences)):
                ids = dec_matrix[i, 1:].tolist()
                clean = []
                for t in ids:
                    if t == eos_id:
                        break
                    if t != pad_id:
                        clean.append(t)
                output_texts.append(tokenizer.decode(clean))
        else:
            # Beam search path
            token_lists = _beam_search_decode(
                actual_model, encoder_output, encoder_mask,
                bos_id, eos_id, pad_id,
                max_len, beam_size, length_penalty, device
            )
            output_texts = [tokenizer.decode(ids) for ids in token_lists]

    return output_texts


def translate_sentence(
    model,
    tokenizer,
    sentence: str,
    direction: str,
    max_len: int,
    device,
    compute_dtype=torch.float16,
    beam_size: int = 4,
    length_penalty: float = 0.6,
) -> str:
    """
    Dịch 1 câu cho production inference. Mặc định dùng Beam Search (beam_size=4).
    Khác với translate_batch (greedy mặc định để tính BLEU nhanh),
    hàm này ưu tiên chất lượng bản dịch hơn tốc độ.
    """
    results = translate_batch(
        model=model,
        tokenizer=tokenizer,
        sentences=[sentence],
        directions=[direction],
        max_len=max_len,
        device=device,
        compute_dtype=compute_dtype,
        beam_size=beam_size,
        length_penalty=length_penalty,
    )
    return results[0]