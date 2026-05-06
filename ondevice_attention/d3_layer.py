"""
d3_layer.py — On-device attention with fixed-shape pre-allocated KV cache.

Addresses JIT-compile thrashing. Cache is a fixed-shape buffer of
(1, Hq, MAX_S, Dh) per layer. Each decode step:
  - Cache update via repeat → mul (one-hot mask) → add at cur_pos
    (D5 replaces this with paged_update_cache; see d5_layer.py).
  - SDPA against the fixed-shape cache.
  - Position mask sets positions > cur_pos to -inf before softmax.

All decode-step kernel shapes are stable across tokens, so the ttnn
JIT compiles each kernel once and reuses it on subsequent steps.

References:
  - tt-metal/tests/ttnn/nightly/.../test_paged_update_cache.py
  - tt-metal/models/demos/qwen3_vl/tt/attention.py:472
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import ttnn

from config import TinyLlamaConfig
from rope import apply_rope_torch, apply_rope_ondevice
from d2_layer import (
    D2LayerWeights,
    upload_layer_weights_to_submesh,
    _from_torch_replicated,
    _to_torch_local,
)


# Reuse D2's layer weight container — same weights, different cache strategy
D3LayerWeights = D2LayerWeights
upload_d3_layer_weights = upload_layer_weights_to_submesh


def allocate_kv_cache_on_submesh(cfg: TinyLlamaConfig, max_s: int, submesh):
    """Allocate a fixed-shape KV cache buffer per layer on the submesh.
    Returns (K_caches, V_caches) — each is a list of 22 ttnn.Tensors,
    each of shape (1, Hq, MAX_S, Dh) bf16 zero-initialized.

    These buffers persist across all decode steps. paged_update_cache
    writes new positions in place; the kernel cache stays warm.
    """
    Hq = cfg.num_attention_heads
    Dh = cfg.head_dim
    cache_shape = (1, Hq, max_s, Dh)

    K_caches = []
    V_caches = []
    for _ in range(cfg.num_layers):
        K_zero = torch.zeros(*cache_shape, dtype=torch.bfloat16)
        V_zero = torch.zeros(*cache_shape, dtype=torch.bfloat16)
        K_caches.append(_from_torch_replicated(K_zero.to(torch.float32), submesh))
        V_caches.append(_from_torch_replicated(V_zero.to(torch.float32), submesh))
    return K_caches, V_caches


def fill_cache_from_prefill(K_caches, V_caches, prefill_K_list, prefill_V_list,
                            real_len: int, max_s: int, submesh):
    """After prefill produces variable-length K, V per layer, copy them
    into the fixed-shape cache buffers. Done ONCE at the prefill→decode
    transition, so dynamic shape happens only here.

    Approach: pull the prefill K, V to host, place at positions [0:real_len]
    in a zero-padded MAX_S buffer, upload back. Variable-shape ops
    contained to host; on-device decode loop sees only fixed shapes.
    """
    Hq, Dh = prefill_K_list[0].shape[1], prefill_K_list[0].shape[3]

    # Pull prefill caches to host, pad, push back to submesh as fixed-shape.
    new_K_caches = []
    new_V_caches = []
    for K_tt, V_tt in zip(prefill_K_list, prefill_V_list):
        K_host = _to_torch_local(K_tt)  # shape (1, Hq, S_padded, Dh)
        V_host = _to_torch_local(V_tt)

        # Trim to real_len and pad to MAX_S
        K_padded = torch.zeros(1, Hq, max_s, Dh, dtype=K_host.dtype)
        V_padded = torch.zeros(1, Hq, max_s, Dh, dtype=V_host.dtype)
        K_padded[:, :, :real_len, :] = K_host[:, :, :real_len, :]
        V_padded[:, :, :real_len, :] = V_host[:, :, :real_len, :]

        new_K_caches.append(_from_torch_replicated(K_padded, submesh))
        new_V_caches.append(_from_torch_replicated(V_padded, submesh))
    return new_K_caches, new_V_caches


def attention_d3_decode(
    x_tt: ttnn.Tensor,
    w: D3LayerWeights,
    cfg: TinyLlamaConfig,
    cos_tt: ttnn.Tensor,         # (1, 1, 1, Dh) device tensor for cur_pos
    sin_tt: ttnn.Tensor,         # (1, 1, 1, Dh) device tensor for cur_pos
    trans_mat_tt: ttnn.Tensor,   # (1, 1, Dh, Dh) device tensor (constant)
    cur_pos: int,
    K_cache_tt: ttnn.Tensor,    # fixed-shape (1, Hq, MAX_S, Dh)
    V_cache_tt: ttnn.Tensor,
    max_s: int,
    submesh,
) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
    """Decode-mode attention with FIXED-SHAPE on-device KV cache.

    Single-token path (S_new=1). Returns (output, K_cache_new, V_cache_new).
    Caller MUST write the new caches back so the next decode step sees them.
    """
    B = 1
    S_new = 1
    Hq = cfg.num_attention_heads
    Dh = cfg.head_dim

    # 1. Project Q, K_new, V_new on device.
    q = ttnn.matmul(x_tt, w.wq)
    k_new = ttnn.matmul(x_tt, w.wk)
    v_new = ttnn.matmul(x_tt, w.wv)

    # 2. Reshape to per-head form on device.
    q = ttnn.reshape(q, (B, S_new, Hq, Dh))
    k_new = ttnn.reshape(k_new, (B, S_new, Hq, Dh))
    v_new = ttnn.reshape(v_new, (B, S_new, Hq, Dh))
    q = ttnn.transpose(q, 1, 2)            # (1, Hq, 1, Dh)
    k_new = ttnn.transpose(k_new, 1, 2)
    v_new = ttnn.transpose(v_new, 1, 2)

    # 3. ROPE on device. Was a host roundtrip (pull Q,K, apply_rope_torch,
    # push back); now matmul + multiply + multiply + add, no PCIe transfer.
    # Saves 2 host roundtrips per layer × 22 layers = 44 per decode token.
    q = apply_rope_ondevice(q, cos_tt, sin_tt, trans_mat_tt)
    k_new = apply_rope_ondevice(k_new, cos_tt, sin_tt, trans_mat_tt)

    # 4. Update fixed-shape cache via broadcast + masked-add.
    # paged_update_cache requires L1-sharded memory for inputs (see
    # d5_layer.py for that path). This formulation uses only basic ttnn ops
    # without the sharding plumbing:
    #   - Broadcast K_new (B, Hq, 1, Dh) to (B, Hq, MAX_S, Dh) via ttnn.repeat.
    #   - Multiply by a one-hot position mask (1.0 at cur_pos, 0 elsewhere).
    #   - Add to K_cache (which is zero at cur_pos because prefill only
    #     filled [0:real_len] and decode writes each position once).
    # All ops are fixed-shape — JIT compiles once.
    pos_mask_host = torch.zeros(1, 1, max_s, 1, dtype=torch.float32)
    pos_mask_host[0, 0, cur_pos, 0] = 1.0
    pos_mask_tt = _from_torch_replicated(pos_mask_host, submesh)

    # Broadcast K_new (B, Hq, 1, Dh) → (B, Hq, MAX_S, Dh) by repeating along dim 2
    k_broadcast = ttnn.repeat(k_new, ttnn.Shape([1, 1, max_s, 1]))
    v_broadcast = ttnn.repeat(v_new, ttnn.Shape([1, 1, max_s, 1]))

    # Multiply by position mask: zeros everywhere except cur_pos
    k_update = ttnn.multiply(k_broadcast, pos_mask_tt)
    v_update = ttnn.multiply(v_broadcast, pos_mask_tt)

    # Add to existing cache (cache[cur_pos] is 0 before this step)
    K_cache_tt = ttnn.add(K_cache_tt, k_update)
    V_cache_tt = ttnn.add(V_cache_tt, v_update)

    # 5. SDPA against fixed-shape cache (1, Hq, MAX_S, Dh).
    # All matmul shapes are stable across tokens — kernel cache hits forever.
    K_t = ttnn.transpose(K_cache_tt, -2, -1)      # (1, Hq, Dh, MAX_S)
    scores = ttnn.matmul(q, K_t)                   # (1, Hq, 1, MAX_S) FIXED
    scores = ttnn.multiply(scores, 1.0 / math.sqrt(Dh))

    # 6. Position mask: positions > cur_pos get -inf.
    # Mask shape (1, 1, 1, MAX_S) — broadcasts against scores.
    mask_t = torch.full((max_s,), float("-inf"), dtype=torch.bfloat16)
    mask_t[: cur_pos + 1] = 0.0
    mask_t = mask_t.reshape(1, 1, 1, max_s).to(torch.float32)
    mask_tt = _from_torch_replicated(mask_t, submesh)
    scores = ttnn.add(scores, mask_tt)

    attn_w = ttnn.softmax(scores, dim=-1)          # (1, Hq, 1, MAX_S) FIXED
    attn_out = ttnn.matmul(attn_w, V_cache_tt)     # (1, Hq, 1, Dh) FIXED

    # 7. Reassemble + output projection.
    attn_out = ttnn.transpose(attn_out, 1, 2)      # (1, 1, Hq, Dh)
    attn_out = ttnn.reshape(attn_out, (1, 1, Hq * Dh))
    output = ttnn.matmul(attn_out, w.wo)
    return output, K_cache_tt, V_cache_tt


def mlp_ondevice(x_tt, w) -> ttnn.Tensor:
    g = ttnn.matmul(x_tt, w.w_gate)
    u = ttnn.matmul(x_tt, w.w_up)
    h = ttnn.multiply(ttnn.silu(g), u)
    return ttnn.matmul(h, w.w_down)


def transformer_layer_d3_decode(
    x_tt: ttnn.Tensor,
    w: D3LayerWeights,
    cfg: TinyLlamaConfig,
    cos_tt: ttnn.Tensor,
    sin_tt: ttnn.Tensor,
    trans_mat_tt: ttnn.Tensor,
    cur_pos: int,
    K_cache_tt: ttnn.Tensor,
    V_cache_tt: ttnn.Tensor,
    max_s: int,
    submesh,
) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
    """One full transformer layer in decode mode with fixed-shape cache.
    Returns (x_out, K_cache_new, V_cache_new). Caller writes caches back.
    """
    h = ttnn.rms_norm(x_tt, weight=w.attn_norm, epsilon=cfg.rms_norm_eps)
    a, K_cache_tt, V_cache_tt = attention_d3_decode(
        h, w, cfg, cos_tt, sin_tt, trans_mat_tt, cur_pos,
        K_cache_tt, V_cache_tt, max_s, submesh,
    )
    x_tt = ttnn.add(x_tt, a)

    h = ttnn.rms_norm(x_tt, weight=w.ffn_norm, epsilon=cfg.rms_norm_eps)
    m = mlp_ondevice(h, w)
    x_tt = ttnn.add(x_tt, m)
    return x_tt, K_cache_tt, V_cache_tt
