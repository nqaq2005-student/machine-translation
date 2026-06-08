# machine-translation

A compact PyTorch-based machine translation project skeleton with Transformer architecture, BPE tokenization support, training / evaluation / translation scripts, and a simple project layout.

## Structure

- `configs/config.yaml` — configuration for model hyperparameters, training, tokenization, and paths
- `data/raw/` — raw bilingual text files used for training / validation / test
- `data/processed/` — tokenizer and processed artifacts
- `src/model/` — Transformer model implementation
- `src/data_pipeline/` — tokenizer training and dataset pipeline
- `src/utils/` — helper utilities and metrics
- `checkpoints/` — saved model weights
- `train.py` — training entry point
- `evaluate.py` — evaluation entry point
- `translate.py` — inference and beam search

## Setup

1. Create a Python virtual environment and activate it.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Prepare your raw bilingual files under `data/raw/`.
4. Train the BPE tokenizer:

```bash
python src/data_pipeline/train_bpe.py --config configs/config.yaml
```

5. Train the model:

```bash
python train.py --config configs/config.yaml
```

6. Evaluate or translate using the saved checkpoint:

```bash
python evaluate.py --config configs/config.yaml --checkpoint checkpoints/transformer.pt
python translate.py --config configs/config.yaml --checkpoint checkpoints/transformer.pt --sentence "Hello world"
```
