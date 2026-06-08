# machine-translation

Mẫu cấu hình để build model dịch máy 2 chiều **Vi ↔ En** bằng kiến trúc **Transformer** với kích thước khoảng **50M tham số**.

## Cách dùng nhanh

```bash
python /tmp/workspace/nqaq2005-student/machine-translation/translation_transformer.py
```

Lệnh trên sẽ in ra cấu hình mặc định và tổng tham số ước lượng.

## Thiết kế chính

- 1 model duy nhất cho cả 2 chiều dịch: `vi-en` và `en-vi`
- Dùng token điều hướng:
  - `<2en>` cho Vi -> En
  - `<2vi>` cho En -> Vi
- Cấu hình mặc định:
  - `d_model=512`
  - `num_encoder_layers=6`
  - `num_decoder_layers=6`
  - `dim_feedforward=1536`
  - `vocab_size=30000`
  - chia sẻ embedding + buộc output projection theo embedding để giảm số tham số