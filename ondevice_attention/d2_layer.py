"""
d2_layer.py — Layer with on-device SDPA, on-device KV cache, and submesh support.

Extends d1_layer.py with:
  - decode-mode attention (S=1, attend over full cache, no causal mask)
  - on-device KV cache: cached_K, cached_V are ttnn.Tensors that live on the
    submesh, growing one position per decode step via ttnn.concat
  - submesh-friendly placement via mesh_mapper=ReplicateTensorToMesh(submesh)
    (works the same on a 1×1 submesh as on a regular Device)

"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import ttnn

from config import TinyLlamaConfig
from rope import apply_rope_torch


@dataclass
class D2LayerWeights:
    """Same as D1LayerWeights but uploaded to a submesh (or any MeshDevice)
    via ReplicateTensorToMesh. K, V projections are pre-expanded on host
    so each chip stores them at full Hq head count — see d1_layer.py for
    the rationale.
    """
    attn_norm: ttnn.Tensor
    ffn_norm: ttnn.Tensor
    wq: ttnn.Tensor
    wk: ttnn.Tensor
    wv: ttnn.Tensor
    wo: ttnn.Tensor
    w_gate: ttnn.Tensor
    w_up: ttnn.Tensor
    w_down: ttnn.Tensor


def _expand_kv_for_gqa(w_kv: torch.Tensor, cfg: TinyLlamaConfig) -> torch.Tensor:
    D = w_kv.shape[0]
    Dh = cfg.head_dim
    Hkv = cfg.num_key_value_heads
    G = cfg.gqa_group_size
    assert w_kv.shape == (D, Hkv * Dh), f"unexpected w_kv shape {w_kv.shape}"
    w = w_kv.reshape(D, Hkv, Dh)
    w = w.repeat_interleave(G, dim=1)
    return w.reshape(D, Hkv * G * Dh)


def upload_layer_weights_to_submesh(layer_w, submesh, cfg) -> D2LayerWeights:
    """Place one layer's weights on the submesh (replicated across all
    devices in the submesh — for 1×1 submeshes that's just one chip)."""
    def _to_sub(t: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=submesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
        )

    wk_expanded = _expand_kv_for_gqa(layer_w.wk, cfg)
    wv_expanded = _expand_kv_for_gqa(layer_w.wv, cfg)

    return D2LayerWeights(
        attn_norm=_to_sub(layer_w.attn_norm.reshape(1, -1)),
        ffn_norm=_to_sub(layer_w.ffn_norm.reshape(1, -1)),
        wq=_to_sub(layer_w.wq),
        wk=_to_sub(wk_expanded),
        wv=_to_sub(wv_expanded),
        wo=_to_sub(layer_w.wo),
        w_gate=_to_sub(layer_w.w_gate),
        w_up=_to_sub(layer_w.w_up),
        w_down=_to_sub(layer_w.w_down),
    )


def _from_torch_replicated(t_host: torch.Tensor, submesh) -> ttnn.Tensor:
    return ttnn.from_torch(
        t_host.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )


def _to_torch_local(t_tt: ttnn.Tensor) -> torch.Tensor:
    """Pull a single chip's view to host as fp32. For 1×1 submeshes,
    chip 0's view IS the full tensor."""
    shards = ttnn.get_device_tensors(t_tt)
    return ttnn.to_torch(shards[0]).to(torch.float32)


def attention_ondevice(
    x_tt: ttnn.Tensor,
    w: D2LayerWeights,
    cfg: TinyLlamaConfig,
    cos_host: torch.Tensor,
    sin_host: torch.Tensor,
    cur_pos: int,
    cached_K_tt: Optional[ttnn.Tensor],
    cached_V_tt: Optional[ttnn.Tensor],
    submesh,
) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
    """On-device attention with KV cache.
      cached_K, cached_V = None  → prefill mode (causal mask, no concat)
      cached_K, cached_V provided → decode mode (no mask, concat with cache)

    Returns (output_tt, full_K_tt, full_V_tt) where full_K/V can be
    written back into the cache for the next decode step.
    """
    B = x_tt.shape[0]
    S_new = x_tt.shape[1]
    Hq = cfg.num_attention_heads
    Dh = cfg.head_dim

    # 1. Project Q, K, V on device.
    q = ttnn.matmul(x_tt, w.wq)
    k_new = ttnn.matmul(x_tt, w.wk)
    v_new = ttnn.matmul(x_tt, w.wv)

    # 2. Reshape to per-head form on device.
    q = ttnn.reshape(q, (B, S_new, Hq, Dh))
    k_new = ttnn.reshape(k_new, (B, S_new, Hq, Dh))
    v_new = ttnn.reshape(v_new, (B, S_new, Hq, Dh))
    q = ttnn.transpose(q, 1, 2)        # (B, Hq, S_new, Dh)
    k_new = ttnn.transpose(k_new, 1, 2)
    v_new = ttnn.transpose(v_new, 1, 2)

    # 3. RoPE on host (transitional — costs 2 PCIe roundtrips).
    cos_slice = cos_host[cur_pos : cur_pos + S_new]
    sin_slice = sin_host[cur_pos : cur_pos + S_new]
    q_h = _to_torch_local(q)
    k_h = _to_torch_local(k_new)
    q_h = apply_rope_torch(q_h, cos_slice, sin_slice)
    k_h = apply_rope_torch(k_h, cos_slice, sin_slice)
    q = _from_torch_replicated(q_h, submesh)
    k_new = _from_torch_replicated(k_h, submesh)

    # 4. Concat with cache if present (decode mode).
    if cached_K_tt is not None:
        full_K = ttnn.concat([cached_K_tt, k_new], dim=2)
        full_V = ttnn.concat([cached_V_tt, v_new], dim=2)
    else:
        full_K = k_new
        full_V = v_new

    # 5. SDPA on device — same pattern as D1, with mask only in prefill.
    k_t = ttnn.transpose(full_K, -2, -1)
    scores = ttnn.matmul(q, k_t)
    scores = ttnn.multiply(scores, 1.0 / math.sqrt(Dh))

    if cached_K_tt is None and S_new > 1:
        # Prefill: causal mask.
        S = S_new
        mask_t = torch.full((S, S), float("-inf"), dtype=torch.bfloat16)
        mask_t = torch.triu(mask_t, diagonal=1).reshape(1, 1, S, S)
        mask = _from_torch_replicated(mask_t.to(torch.float32), submesh)
        scores = ttnn.add(scores, mask)
    # Decode: no mask. The new token attends over the full cache freely.

    attn_w = ttnn.softmax(scores, dim=-1)
    attn_out = ttnn.matmul(attn_w, full_V)

    # 6. Reassemble + output projection.
    attn_out = ttnn.transpose(attn_out, 1, 2)
    attn_out = ttnn.reshape(attn_out, (B, S_new, Hq * Dh))
    output = ttnn.matmul(attn_out, w.wo)

    return output, full_K, full_V


def mlp_ondevice(x_tt, w) -> ttnn.Tensor:
    """SwiGLU MLP on device."""
    g = ttnn.matmul(x_tt, w.w_gate)
    u = ttnn.matmul(x_tt, w.w_up)
    h = ttnn.multiply(ttnn.silu(g), u)
    return ttnn.matmul(h, w.w_down)


def transformer_layer_with_cache(
    x_tt: ttnn.Tensor,
    w: D2LayerWeights,
    cfg: TinyLlamaConfig,
    cos_host: torch.Tensor,
    sin_host: torch.Tensor,
    cur_pos: int,
    cached_K_tt: Optional[ttnn.Tensor],
    cached_V_tt: Optional[ttnn.Tensor],
    submesh,
) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
    """One full transformer layer with on-device math + on-device KV cache."""
    h = ttnn.rms_norm(x_tt, weight=w.attn_norm, epsilon=cfg.rms_norm_eps)
    a, full_K, full_V = attention_ondevice(
        h, w, cfg, cos_host, sin_host, cur_pos, cached_K_tt, cached_V_tt, submesh
    )
    x_tt = ttnn.add(x_tt, a)

    h = ttnn.rms_norm(x_tt, weight=w.ffn_norm, epsilon=cfg.rms_norm_eps)
    m = mlp_ondevice(h, w)
    x_tt = ttnn.add(x_tt, m)
    return x_tt, full_K, full_V
