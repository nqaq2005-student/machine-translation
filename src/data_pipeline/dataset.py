import torch
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_seq_len):
        """
        data_list: Một list chứa các dictionary dạng:
                   [{"src": "Hello", "tgt": "Xin chào", "direction": "en2vi"}, ...]
        tokenizer: Đối tượng tokenizer đã được huấn luyện (HuggingFace tokenizers)
        """
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # Trích xuất ID của các token đặc biệt từ Tokenizer
        self.pad_id = tokenizer.token_to_id("[PAD]")
        self.bos_id = tokenizer.token_to_id("[BOS]")  # Bắt đầu câu (Begin of Sentence)
        self.eos_id = tokenizer.token_to_id("[EOS]")  # Kết thúc câu (End of Sentence)

        # Token định hướng ngôn ngữ (dùng thay thế [BOS] ở Encoder)
        self.vi2en_id = tokenizer.token_to_id("<2en>")
        self.en2vi_id = tokenizer.token_to_id("<2vi>")

        # Mặt nạ tránh nhìn trước tương lai
        self.causal_mask = (torch.triu(
            torch.ones((1, max_seq_len, max_seq_len)), diagonal=1
        ).type(torch.int) == 0)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        src_text = item['src']
        tgt_text = item['tgt']
        direction = item['direction']

        # 1. Tokenize văn bản thành mảng các số nguyên (IDs)
        src_tokens = self.tokenizer.encode(src_text).ids
        tgt_tokens = self.tokenizer.encode(tgt_text).ids

        # 2. Chọn token điều hướng làm mốc bắt đầu cho Encoder
        direction_token = self.en2vi_id if direction == 'en2vi' else self.vi2en_id

        # 3. Cắt bớt nếu câu dài vượt mức max_seq_len (trừ hao không gian cho BOS/EOS)
        src_tokens = src_tokens[:self.max_seq_len - 2]
        tgt_tokens = tgt_tokens[:self.max_seq_len - 1]

        # 4. Tính toán số lượng token [PAD] cần đắp thêm vào cuối
        src_padding_len = self.max_seq_len - len(src_tokens) - 2
        tgt_padding_len = self.max_seq_len - len(tgt_tokens) - 1

        # XÂY DỰNG TENSORS ĐẦU VÀO VÀ NHÃN (LABELS)

        # Encoder Input: <Direction> + Source Tokens + <EOS> + <PAD>...
        encoder_input = torch.tensor(
            [direction_token] + src_tokens + [self.eos_id] + [self.pad_id] * src_padding_len,
            dtype=torch.int64
        )

        # Decoder Input: <BOS> + Target Tokens + <PAD>...
        decoder_input = torch.tensor(
            [self.bos_id] + tgt_tokens + [self.pad_id] * tgt_padding_len,
            dtype=torch.int64
        )

        # Label (Dùng để tính Loss): Target Tokens + <EOS> + <PAD>...
        label = torch.tensor(
            tgt_tokens + [self.eos_id] + [self.pad_id] * tgt_padding_len,
            dtype=torch.int64
        )

        # TẠO ATTENTION MASKS (Che lấp thông tin)

        # Đánh dấu True ở những vị trí CÓ TỪ (khác PAD), False ở vị trí PAD.
        # Shape: (1, 1, max_seq_len) để chuẩn bị broadcast trên nhiều Attention Heads
        encoder_mask = (encoder_input != self.pad_id).unsqueeze(0).unsqueeze(0)

        decoder_pad_mask = (decoder_input != self.pad_id).unsqueeze(0).unsqueeze(0)

        # Kết hợp hai Mask của Decoder bằng phép AND (Chỉ True khi thỏa mãn cả 2 điều kiện)
        # Shape: (1, max_seq_len, max_seq_len)
        decoder_mask = decoder_pad_mask & self.causal_mask

        return {
            "encoder_input": encoder_input,  # (max_seq_len)
            "decoder_input": decoder_input,  # (max_seq_len)
            "encoder_mask":  encoder_mask,   # (1, 1, max_seq_len)
            "decoder_mask":  decoder_mask,   # (1, max_seq_len, max_seq_len)
            "label":         label,          # (max_seq_len)
            "src_text":      src_text,       # Lưu lại text gốc để debug/tính BLEU
            "tgt_text":      tgt_text
        }