"""TinyLlama-1.1B-Chat-v1.0 model configuration + TP layout for 2 chips."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TinyLlamaConfig:
    # Model dimensions (exact values from HF config.json)
    num_layers: int = 22
    hidden_size: int = 2048               # D
    intermediate_size: int = 5632         # F
    num_attention_heads: int = 32
    num_key_value_heads: int = 4          # GQA: 4 KV heads, 32 Q heads
    head_dim: int = 64
    vocab_size: int = 32000
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5

    # TP-derived sizes (for 2 chips)
    @property
    def num_chips(self) -> int:
        return 2

    @property
    def heads_per_chip(self) -> int:
        return self.num_attention_heads // self.num_chips         # 16

    @property
    def kv_heads_per_chip(self) -> int:
        return self.num_key_value_heads // self.num_chips         # 2

    @property
    def d_per_chip(self) -> int:
        return self.hidden_size // self.num_chips                  # 1024

    @property
    def f_per_chip(self) -> int:
        return self.intermediate_size // self.num_chips            # 2816

    @property
    def gqa_group_size(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads  # 8

    def validate(self):
        """Sanity checks on partitionability."""
        assert self.num_attention_heads % self.num_chips == 0, \
            "num_attention_heads must be divisible by num_chips"
        assert self.num_key_value_heads % self.num_chips == 0, \
            "num_key_value_heads must be divisible by num_chips"
        assert self.hidden_size % self.num_chips == 0
        assert self.intermediate_size % self.num_chips == 0
        # TILE_LAYOUT requirement (32-tile alignment)
        for v in (self.d_per_chip, self.f_per_chip, self.kv_heads_per_chip * self.head_dim):
            assert v % 32 == 0, f"per-chip shape {v} not tile-aligned"
