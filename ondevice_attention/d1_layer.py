"""
d1_layer.py — single transformer layer with ON-DEVICE attention math.


Difference from path_c_real_eth/mesh_layer.py:
  mesh_layer.py: project on device, PULL TO HOST, attention math on
                 CPU, push back, output projection on device. ~88 PCIe
                 roundtrips per decode token.
  d1_layer.py:   project on device, RoPE on host (Q+K only), attention
                 math on device, output projection on device. ~44 PCIe
                 roundtrips per decode token (RoPE only). Decode rate
                 should approximately double.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import ttnn

from config import TinyLlamaConfig
from rope import apply_rope_torch


@dataclass
class D1LayerWeights:
    """One transformer layer's weights, replicated on a single device.

    K and V projections are PRE-EXPANDED on host (`expand_kv_for_gqa`)
    so each chip stores them at full Hq head count, not Hkv. This means
    SDPA can run as plain MHA on device with no GQA bookkeeping at
    runtime. Costs G×=8× more weight memory for K, V (~1 MB extra for
    TinyLlama, fine).
    """
    attn_norm: ttnn.Tensor       # (1, D)
    ffn_norm: ttnn.Tensor        # (1, D)
    wq: ttnn.Tensor              # (D, Hq*Dh)
    wk: ttnn.Tensor              # (D, Hq*Dh)  — expanded
    wv: ttnn.Tensor              # (D, Hq*Dh)  — expanded
    wo: ttnn.Tensor              # (Hq*Dh, D)
    w_gate: ttnn.Tensor          # (D, F)
    w_up: ttnn.Tensor            # (D, F)
    w_down: ttnn.Tensor          # (F, D)


def expand_kv_for_gqa(w_kv: torch.Tensor, cfg: TinyLlamaConfig) -> torch.Tensor:
    """Pre-expand a K or V projection weight on host so each KV head is
    replicated G times. After expansion shape goes from (D, Hkv*Dh) to
    (D, Hq*Dh). Mathematically identical to repeat_interleave on the
    head dim at runtime, just done once at upload time.
    """
    D = w_kv.shape[0]
    Dh = cfg.head_dim
    Hkv = cfg.num_key_value_heads
    G = cfg.gqa_group_size
    assert w_kv.shape == (D, Hkv * Dh), f"unexpected w_kv shape {w_kv.shape}"
    w = w_kv.reshape(D, Hkv, Dh)
    w = w.repeat_interleave(G, dim=1)
    return w.reshape(D, Hkv * G * Dh)   # = (D, Hq * Dh)


def upload_layer_weights(layer_w, device, cfg) -> D1LayerWeights:
    """Place one layer's weights on the device, K/V pre-expanded."""
    def _to_dev(t: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    wk_expanded = expand_kv_for_gqa(layer_w.wk, cfg)
    wv_expanded = expand_kv_for_gqa(layer_w.wv, cfg)

    return D1LayerWeights(
        attn_norm=_to_dev(layer_w.attn_norm.reshape(1, -1)),
        ffn_norm=_to_dev(layer_w.ffn_norm.reshape(1, -1)),
        wq=_to_dev(layer_w.wq),
        wk=_to_dev(wk_expanded),
        wv=_to_dev(wv_expanded),
        wo=_to_dev(layer_w.wo),
        w_gate=_to_dev(layer_w.w_gate),
        w_up=_to_dev(layer_w.w_up),
        w_down=_to_dev(layer_w.w_down),
    )


def attention_ondevice_prefill(
    x_tt: ttnn.Tensor,
    w: D1LayerWeights,
    cfg: TinyLlamaConfig,
    cos_host: torch.Tensor,    # (S, Dh) host fp32
    sin_host: torch.Tensor,
    device,
) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
    """Prefill-mode attention with on-device SDPA. No KV cache.
    Returns (output_tt, full_K_tt, full_V_tt) so the caller can stash
    K and V somewhere (host or device) for later decode.

    """
    B, S, D = x_tt.shape
    Hq = cfg.num_attention_heads
    Dh = cfg.head_dim

    # 1. Project Q, K, V on device. K, V come out at full Hq head count
    #    because the weights were pre-expanded.
    q = ttnn.matmul(x_tt, w.wq)    # (B, S, Hq*Dh)
    k = ttnn.matmul(x_tt, w.wk)
    v = ttnn.matmul(x_tt, w.wv)

    # 2. Reshape to per-head form on device.
    q = ttnn.reshape(q, (B, S, Hq, Dh))
    k = ttnn.reshape(k, (B, S, Hq, Dh))
    v = ttnn.reshape(v, (B, S, Hq, Dh))
    q = ttnn.transpose(q, 1, 2)        # (B, Hq, S, Dh)
    k = ttnn.transpose(k, 1, 2)
    v = ttnn.transpose(v, 1, 2)

    # 3. RoPE — kept on host for D1 (transitional, costs 2 PCIe ops).
    #    Slice the precomputed cos/sin tables to just the positions we need.
    cos_slice = cos_host[:S]
    sin_slice = sin_host[:S]
    q_host = ttnn.to_torch(q).to(torch.float32)
    k_host = ttnn.to_torch(k).to(torch.float32)
    q_host = apply_rope_torch(q_host, cos_slice, sin_slice)
    k_host = apply_rope_torch(k_host, cos_slice, sin_slice)
    q = ttnn.from_torch(
        q_host.to(torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
    )
    k = ttnn.from_torch(
        k_host.to(torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
    )

    # 4. SDPA on device — direct port of tp_layer.py:189-209.
    k_t = ttnn.transpose(k, -2, -1)                    # (B, Hq, Dh, S)
    scores = ttnn.matmul(q, k_t)                       # (B, Hq, S, S)
    scores = ttnn.multiply(scores, 1.0 / math.sqrt(Dh))

    # Causal mask for prefill.
    if S > 1:
        mask_t = torch.full((S, S), float("-inf"), dtype=torch.bfloat16)
        mask_t = torch.triu(mask_t, diagonal=1).reshape(1, 1, S, S)
        mask = ttnn.from_torch(
            mask_t,
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
        )
        scores = ttnn.add(scores, mask)

    attn_w = ttnn.softmax(scores, dim=-1)              # (B, Hq, S, S)
    attn_out = ttnn.matmul(attn_w, v)                  # (B, Hq, S, Dh)

    # 5. Re-assemble heads, output projection.
    attn_out = ttnn.transpose(attn_out, 1, 2)          # (B, S, Hq, Dh)
    attn_out = ttnn.reshape(attn_out, (B, S, Hq * Dh))
    output_tt = ttnn.matmul(attn_out, w.wo)            # (B, S, D)

    return output_tt, k, v


def mlp_ondevice(x_tt, w, cfg) -> ttnn.Tensor:
    """SwiGLU MLP on device. Same as tp_layer.py:tp_mlp but no all_reduce
    because weights are replicated, not row-sharded."""
    g = ttnn.matmul(x_tt, w.w_gate)
    u = ttnn.matmul(x_tt, w.w_up)
    h = ttnn.multiply(ttnn.silu(g), u)
    return ttnn.matmul(h, w.w_down)


def transformer_layer_prefill(
    x_tt: ttnn.Tensor,
    w: D1LayerWeights,
    cfg: TinyLlamaConfig,
    cos_host: torch.Tensor,
    sin_host: torch.Tensor,
    device,
) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
    """One full transformer layer with on-device math (prefill mode).
    Returns (x_out, k_for_cache, v_for_cache).
    """
    h = ttnn.rms_norm(x_tt, weight=w.attn_norm, epsilon=cfg.rms_norm_eps)
    a, k_full, v_full = attention_ondevice_prefill(
        h, w, cfg, cos_host, sin_host, device
    )
    x_tt = ttnn.add(x_tt, a)

    h = ttnn.rms_norm(x_tt, weight=w.ffn_norm, epsilon=cfg.rms_norm_eps)
    m = mlp_ondevice(h, w, cfg)
    x_tt = ttnn.add(x_tt, m)
    return x_tt, k_full, v_full
