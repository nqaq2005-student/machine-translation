from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple

Direction = Literal["vi-en", "en-vi"]


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int = 30_000
    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 1_536
    max_position_embeddings: int = 512
    dropout: float = 0.1
    share_token_embeddings: bool = True
    tie_output_projection: bool = True
    direction_tokens: Tuple[str, str] = ("<2en>", "<2vi>")

    def validate(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size phải > 0")
        if self.d_model <= 0 or self.dim_feedforward <= 0:
            raise ValueError("d_model và dim_feedforward phải > 0")
        if self.nhead <= 0 or self.d_model % self.nhead != 0:
            raise ValueError("nhead phải > 0 và d_model phải chia hết cho nhead")
        if self.num_encoder_layers <= 0 or self.num_decoder_layers <= 0:
            raise ValueError("Số layer encoder/decoder phải > 0")
        if self.max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings phải > 0")
        if self.tie_output_projection and not self.share_token_embeddings:
            raise ValueError(
                "tie_output_projection=True yêu cầu share_token_embeddings=True"
            )
        if len(self.direction_tokens) != 2:
            raise ValueError("direction_tokens phải có đúng 2 token")

    def estimated_parameters(self) -> int:
        return estimate_transformer_parameters(self)


def _encoder_layer_params(d_model: int, dim_feedforward: int) -> int:
    # self-attention (Q, K, V, O) + bias
    attention = 4 * d_model * d_model + 4 * d_model
    # feed-forward 2 lớp + bias
    feed_forward = (
        2 * d_model * dim_feedforward + dim_feedforward + d_model
    )
    # 2 LayerNorm (weight + bias)
    norms = 4 * d_model
    return attention + feed_forward + norms


def _decoder_layer_params(d_model: int, dim_feedforward: int) -> int:
    # self-attention + cross-attention (mỗi loại: Q, K, V, O + bias)
    attentions = 8 * d_model * d_model + 8 * d_model
    feed_forward = (
        2 * d_model * dim_feedforward + dim_feedforward + d_model
    )
    # 3 LayerNorm (weight + bias)
    norms = 6 * d_model
    return attentions + feed_forward + norms


def estimate_transformer_parameters(config: TransformerConfig) -> int:
    config.validate()

    token_embeddings = config.vocab_size * config.d_model
    position_embeddings = config.max_position_embeddings * config.d_model
    embedding_block = token_embeddings + position_embeddings
    if not config.share_token_embeddings:
        embedding_block *= 2

    encoder = _encoder_layer_params(
        config.d_model, config.dim_feedforward
    ) * config.num_encoder_layers
    decoder = _decoder_layer_params(
        config.d_model, config.dim_feedforward
    ) * config.num_decoder_layers

    output_projection = 0
    if not config.tie_output_projection:
        output_projection = config.vocab_size * config.d_model + config.vocab_size

    return embedding_block + encoder + decoder + output_projection


class BidirectionalTranslationTransformer:
    def __init__(self, config: TransformerConfig | None = None) -> None:
        self.config = config or TransformerConfig()
        self.config.validate()
        self.supported_directions: Tuple[Direction, Direction] = ("vi-en", "en-vi")
        self._direction_to_token: Dict[str, str] = {
            "vi-en": self.config.direction_tokens[0],
            "en-vi": self.config.direction_tokens[1],
        }

    def get_direction_token(self, direction: Direction) -> str:
        if direction not in self._direction_to_token:
            raise ValueError(
                f"Direction không hợp lệ: {direction}. Hỗ trợ: {self.supported_directions}"
            )
        return self._direction_to_token[direction]

    def build_training_sample(
        self, source_text: str, target_text: str, direction: Direction
    ) -> Dict[str, str]:
        direction_token = self.get_direction_token(direction)
        return {
            "direction": direction,
            "source": f"{direction_token} {source_text.strip()}".strip(),
            "target": target_text.strip(),
        }

    def summary(self) -> Dict[str, object]:
        return {
            "supported_directions": self.supported_directions,
            "direction_tokens": self._direction_to_token.copy(),
            "estimated_parameters": self.config.estimated_parameters(),
            "config": self.config,
        }


if __name__ == "__main__":
    model = BidirectionalTranslationTransformer()
    summary = model.summary()
    print("Bidirectional Transformer (Vi↔En)")
    print(f"Supported directions: {summary['supported_directions']}")
    print(f"Direction tokens: {summary['direction_tokens']}")
    print(f"Estimated parameters: {summary['estimated_parameters']:,}")
