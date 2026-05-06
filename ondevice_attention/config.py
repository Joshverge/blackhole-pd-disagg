"""Llama-architecture model config.

Two modes:
  - Default constructor: TinyLlama-1.1B-Chat-v1.0 dimensions (back-compat).
  - LlamaConfig.from_hf_dir(model_dir): load any Llama-arch HF model
    (TinyLlama-1.1B, Llama-3.2-1B-Instruct, Llama-2-7B, etc.) from its
    config.json. Standard RoPE only — Llama-3.2's long-context frequency
    scaling is not implemented (fine for prompts under ~8K tokens).
"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LlamaConfig:
    # Defaults match TinyLlama-1.1B so existing call sites (LlamaConfig())
    # behave the same as the old TinyLlamaConfig().
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
    tie_word_embeddings: bool = False
    model_name: str = "TinyLlama-1.1B-Chat-v1.0"

    # ---- TP-derived (we use 2 chips for the disagg setup) ----
    @property
    def num_chips(self) -> int:
        return 2

    @property
    def heads_per_chip(self) -> int:
        return self.num_attention_heads // self.num_chips

    @property
    def kv_heads_per_chip(self) -> int:
        return self.num_key_value_heads // self.num_chips

    @property
    def d_per_chip(self) -> int:
        return self.hidden_size // self.num_chips

    @property
    def f_per_chip(self) -> int:
        return self.intermediate_size // self.num_chips

    @property
    def gqa_group_size(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def validate(self):
        """Sanity checks on partitionability + tile alignment."""
        assert self.num_attention_heads % self.num_chips == 0, \
            f"num_attention_heads ({self.num_attention_heads}) not divisible by num_chips ({self.num_chips})"
        assert self.num_key_value_heads % self.num_chips == 0, \
            f"num_key_value_heads ({self.num_key_value_heads}) not divisible by num_chips ({self.num_chips})"
        assert self.hidden_size % self.num_chips == 0
        assert self.intermediate_size % self.num_chips == 0
        # TILE_LAYOUT requires inner dims to be 32-aligned
        for v, name in (
            (self.d_per_chip, "d_per_chip"),
            (self.f_per_chip, "f_per_chip"),
            (self.kv_heads_per_chip * self.head_dim, "kv_heads_per_chip*head_dim"),
            (self.hidden_size, "hidden_size"),
            (self.intermediate_size, "intermediate_size"),
            (self.vocab_size, "vocab_size"),
        ):
            assert v % 32 == 0, f"{name}={v} not 32-aligned for TILE_LAYOUT"

    @classmethod
    def from_hf_dir(cls, model_dir: str) -> "LlamaConfig":
        """Load config from a HuggingFace-format model directory's config.json."""
        path = Path(model_dir) / "config.json"
        assert path.exists(), f"config.json not found at {path}"
        with open(path) as f:
            d = json.load(f)

        model_type = d.get("model_type", "")
        if model_type not in ("llama",):
            print(f"WARNING: model_type={model_type!r} - this loader expects 'llama'-arch. "
                  "May still work if shapes match, but proceed with caution.")

        cfg = cls(
            num_layers=d["num_hidden_layers"],
            hidden_size=d["hidden_size"],
            intermediate_size=d["intermediate_size"],
            num_attention_heads=d["num_attention_heads"],
            num_key_value_heads=d.get("num_key_value_heads", d["num_attention_heads"]),
            head_dim=d.get("head_dim", d["hidden_size"] // d["num_attention_heads"]),
            vocab_size=d["vocab_size"],
            max_position_embeddings=d.get("max_position_embeddings", 2048),
            rope_theta=d.get("rope_theta", 10000.0),
            rms_norm_eps=d.get("rms_norm_eps", 1e-5),
            tie_word_embeddings=d.get("tie_word_embeddings", False),
            model_name=Path(model_dir).name,
        )
        cfg.validate()
        return cfg

    def summary(self) -> str:
        return (
            f"{self.model_name}: layers={self.num_layers}, "
            f"D={self.hidden_size}, F={self.intermediate_size}, "
            f"Hq={self.num_attention_heads}, Hkv={self.num_key_value_heads}, "
            f"Dh={self.head_dim}, vocab={self.vocab_size}, "
            f"rope_theta={self.rope_theta}, "
            f"tie_emb={self.tie_word_embeddings}"
        )


# Back-compat alias so existing imports (`from config import TinyLlamaConfig`)
# keep working. The default constructor still produces TinyLlama dimensions.
TinyLlamaConfig = LlamaConfig
