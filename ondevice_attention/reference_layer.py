"""
reference_layer.py — pure torch (fp32) reference implementation of one
TinyLlama transformer layer. Ground truth for PCC checks against the
ttnn 2-chip TP version.

Includes RoPE. KV cache still deferred to checkpoint 3.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from config import TinyLlamaConfig
from rope import apply_rope_torch


@dataclass
class LayerWeights:
    """One transformer layer's worth of weights, on host in fp32."""
    # RMSNorm weights (D,)
    attn_norm: torch.Tensor
    ffn_norm: torch.Tensor
    # Attention projections
    wq: torch.Tensor       # (D, num_q_heads * head_dim) = (D, D)
    wk: torch.Tensor       # (D, num_kv_heads * head_dim)
    wv: torch.Tensor       # (D, num_kv_heads * head_dim)
    wo: torch.Tensor       # (D, D)
    # SwiGLU MLP
    w_gate: torch.Tensor   # (D, F)
    w_up: torch.Tensor     # (D, F)
    w_down: torch.Tensor   # (F, D)


def make_random_weights(cfg: TinyLlamaConfig, seed: int = 0) -> LayerWeights:
    """Realistic-magnitude random weights."""
    g = torch.Generator().manual_seed(seed)
    D = cfg.hidden_size
    F = cfg.intermediate_size
    Hq = cfg.num_attention_heads * cfg.head_dim    # = D for TinyLlama
    Hkv = cfg.num_key_value_heads * cfg.head_dim   # = 256 for TinyLlama

    return LayerWeights(
        attn_norm=torch.ones(D),
        ffn_norm=torch.ones(D),
        wq=torch.randn(D, Hq, generator=g) / (D ** 0.5),
        wk=torch.randn(D, Hkv, generator=g) / (D ** 0.5),
        wv=torch.randn(D, Hkv, generator=g) / (D ** 0.5),
        wo=torch.randn(D, D, generator=g) / (D ** 0.5),
        w_gate=torch.randn(D, F, generator=g) / (D ** 0.5),
        w_up=torch.randn(D, F, generator=g) / (D ** 0.5),
        w_down=torch.randn(F, D, generator=g) / (F ** 0.5),
    )


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm (Llama variant)."""
    # x: (..., D)
    var = x.pow(2).mean(-1, keepdim=True)
    x_norm = x * torch.rsqrt(var + eps)
    return x_norm * weight


def attention_forward(
    x: torch.Tensor,                # (B, S, D)
    w: LayerWeights,
    cfg: TinyLlamaConfig,
    cos: Optional[torch.Tensor] = None,    # (S, Dh) — RoPE table sliced to S positions
    sin: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """One MHA/GQA forward pass with optional RoPE."""
    B, S, D = x.shape
    Hq = cfg.num_attention_heads
    Hkv = cfg.num_key_value_heads
    Dh = cfg.head_dim
    G = cfg.gqa_group_size

    q = x @ w.wq
    k = x @ w.wk
    v = x @ w.wv

    q = q.view(B, S, Hq, Dh).transpose(1, 2)        # (B, Hq, S, Dh)
    k = k.view(B, S, Hkv, Dh).transpose(1, 2)       # (B, Hkv, S, Dh)
    v = v.view(B, S, Hkv, Dh).transpose(1, 2)

    # RoPE on Q and K (V is not rotated)
    if cos is not None and sin is not None:
        q = apply_rope_torch(q, cos, sin)
        k = apply_rope_torch(k, cos, sin)

    # GQA: repeat each KV head G times to match Q heads
    k = k.repeat_interleave(G, dim=1)
    v = v.repeat_interleave(G, dim=1)

    is_causal = S > 1
    out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    out = out.transpose(1, 2).contiguous().view(B, S, Hq * Dh)
    return out @ w.wo


def mlp_forward(x: torch.Tensor, w: LayerWeights) -> torch.Tensor:
    """SwiGLU MLP: down(silu(gate(x)) * up(x))."""
    g = F.silu(x @ w.w_gate)        # (B, S, F)
    u = x @ w.w_up                  # (B, S, F)
    h = g * u                       # gated
    return h @ w.w_down             # (B, S, D)


def transformer_layer(
    x: torch.Tensor,                # (B, S, D), fp32
    w: LayerWeights,
    cfg: TinyLlamaConfig,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """One full transformer layer: pre-norm -> attn -> resid -> pre-norm -> mlp -> resid."""
    h = rms_norm(x, w.attn_norm, cfg.rms_norm_eps)
    a = attention_forward(h, w, cfg, cos=cos, sin=sin)
    x = x + a

    h = rms_norm(x, w.ffn_norm, cfg.rms_norm_eps)
    m = mlp_forward(h, w)
    x = x + m
    return x
