"""
tp_layer.py — 2-chip tensor-parallel implementation of one TinyLlama
transformer layer in ttnn.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import ttnn

from config import TinyLlamaConfig
from reference_layer import LayerWeights
from rope import apply_rope_via_host


@dataclass
class TPLayerWeights:
    """One transformer layer's weights, sharded across the mesh."""
    attn_norm: ttnn.Tensor      # replicated, (D,)
    ffn_norm: ttnn.Tensor       # replicated, (D,)
    # Attention projections (column-shard for Q/K/V, row-shard for O)
    wq: ttnn.Tensor             # col-shard on dim=1: (D, D/N) per chip
    wk: ttnn.Tensor             # col-shard on dim=1: (D, kv_heads_per_chip*Dh)
    wv: ttnn.Tensor             # col-shard on dim=1
    wo: ttnn.Tensor             # row-shard on dim=0: (D/N, D)
    # SwiGLU MLP
    w_gate: ttnn.Tensor         # col-shard on dim=1: (D, F/N)
    w_up: ttnn.Tensor           # col-shard on dim=1
    w_down: ttnn.Tensor         # row-shard on dim=0: (F/N, D)


def expand_kv_for_gqa(w_kv: torch.Tensor, cfg: TinyLlamaConfig) -> torch.Tensor:
    """
    Pre-expand a K or V projection weight on host so each KV head is replicated
    G times along the output dim. After expansion, the projection has shape
    (D, num_q_heads * head_dim) — same as Q — and column-sharding it naturally
    aligns each chip's K/V heads with its Q heads.

    This sidesteps GQA at runtime entirely. Mathematically identical to
    repeat_interleave on the head dim, just done once at upload time.

    Input  shape: (D, num_kv_heads * head_dim)
    Output shape: (D, num_q_heads  * head_dim)
    """
    D = w_kv.shape[0]
    Dh = cfg.head_dim
    Hkv = cfg.num_key_value_heads
    G = cfg.gqa_group_size
    assert w_kv.shape == (D, Hkv * Dh), f"unexpected w_kv shape {w_kv.shape}"

    # (D, Hkv, Dh) → repeat_interleave each head G times along Hkv → (D, Hkv*G, Dh)
    w = w_kv.reshape(D, Hkv, Dh)
    w = w.repeat_interleave(G, dim=1)
    return w.reshape(D, Hkv * G * Dh)   # = (D, num_q_heads * Dh)


def upload_weights(
    weights: LayerWeights,
    mesh: ttnn.MeshDevice,
    cfg: TinyLlamaConfig,
) -> TPLayerWeights:
    """Take host fp32 weights, cast to bf16, place on the mesh with TP layout.
    GQA is handled by pre-expanding K/V on host so each chip ends up with
    num_q_heads_per_chip K/V heads (matching its Q heads exactly). No runtime
    repeat needed in attention."""
    def _replicate(t: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    def _col_shard(t: torch.Tensor) -> ttnn.Tensor:
        """Shard along last dim (the output features dim)."""
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
        )

    def _row_shard(t: torch.Tensor) -> ttnn.Tensor:
        """Shard along first dim (the input features dim)."""
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )

    # RMSNorm weights are 1D (D,). For TILE_LAYOUT we need 2D, so reshape to (1, D).
    attn_norm_2d = weights.attn_norm.reshape(1, -1)
    ffn_norm_2d = weights.ffn_norm.reshape(1, -1)

    # GQA: expand K and V on host to match Q's head count
    wk_expanded = expand_kv_for_gqa(weights.wk, cfg)
    wv_expanded = expand_kv_for_gqa(weights.wv, cfg)

    return TPLayerWeights(
        attn_norm=_replicate(attn_norm_2d),
        ffn_norm=_replicate(ffn_norm_2d),
        wq=_col_shard(weights.wq),
        wk=_col_shard(wk_expanded),
        wv=_col_shard(wv_expanded),
        wo=_row_shard(weights.wo),
        w_gate=_col_shard(weights.w_gate),
        w_up=_col_shard(weights.w_up),
        w_down=_row_shard(weights.w_down),
    )


def all_reduce_via_gather(t: ttnn.Tensor, mesh: ttnn.MeshDevice) -> ttnn.Tensor:
    """Approximate all_reduce as all_gather + sum, since true all_reduce
    isn't available. Each chip ends up with the same summed result."""
    # Insert a leading device-axis by all_gather on dim=0; each chip then holds
    # (n_chips, ...). We sum along that axis on host, re-upload, replicated.
    # That's expensive but matches the math. A faster path: do all_gather on
    # a new dim, then ttnn.sum on-device. Try the on-device path first.
    gathered = ttnn.all_gather(t, dim=0, topology=ttnn.Topology.Linear)
    # gathered shape per chip: (n_chips, ...) where n_chips=2 along dim 0
    # Sum across that axis on-device.
    summed = ttnn.sum(gathered, dim=0, keepdim=False)
    return summed


def tp_attention(
    x: ttnn.Tensor,                 # replicated (B, S, D)
    w: TPLayerWeights,
    mesh: ttnn.MeshDevice,
    cfg: TinyLlamaConfig,
    cos: Optional[torch.Tensor] = None,    # (S, Dh) host fp32 — RoPE
    sin: Optional[torch.Tensor] = None,
) -> ttnn.Tensor:
    """
    Tensor-parallel attention with optional RoPE.

    Heads are split across chips. Each chip has Hq_per Q heads AND Hq_per K/V
    heads (because we expanded K/V on host to match Q's head count — GQA is
    pre-applied at upload time, not at runtime). Each chip computes SDPA on
    its own subset of heads, then row-sharded output projection produces
    partial sums that we all_reduce.

    RoPE (when cos/sin provided) is applied to Q and K via a host roundtrip
    — slow but unambiguously correct. Optimize in checkpoint 4 if needed.

    Inputs / outputs shape: (B, S, D) replicated on each chip.
    """
    B, S, D = x.shape
    Hq_per = cfg.heads_per_chip       # 16 — same head count for Q, K, V per chip
    Dh = cfg.head_dim

    # Project Q/K/V — column-sharded weights, output is column-sharded per chip.
    q = ttnn.matmul(x, w.wq)
    k = ttnn.matmul(x, w.wk)
    v = ttnn.matmul(x, w.wv)

    # Reshape to (B, S, Hq_per, Dh) and transpose to heads-major
    q = ttnn.reshape(q, (B, S, Hq_per, Dh))
    k = ttnn.reshape(k, (B, S, Hq_per, Dh))
    v = ttnn.reshape(v, (B, S, Hq_per, Dh))
    q = ttnn.transpose(q, 1, 2)        # (B, Hq_per, S, Dh)
    k = ttnn.transpose(k, 1, 2)
    v = ttnn.transpose(v, 1, 2)

    # RoPE on Q and K (host roundtrip — correct but slow)
    if cos is not None and sin is not None:
        q, k = apply_rope_via_host(q, k, cos, sin, mesh)

    # SDPA: scores = Q @ K^T / sqrt(Dh), causal mask, softmax, @ V
    k_t = ttnn.transpose(k, -2, -1)                  # (B, Hq_per, Dh, S)
    scores = ttnn.matmul(q, k_t)                     # (B, Hq_per, S, S)
    scores = ttnn.multiply(scores, 1.0 / math.sqrt(Dh))

    # Causal mask (only when S > 1)
    if S > 1:
        # Build a host-side causal mask, replicated on the mesh
        mask_t = torch.full((S, S), float("-inf"), dtype=torch.bfloat16)
        mask_t = torch.triu(mask_t, diagonal=1).reshape(1, 1, S, S)
        mask = ttnn.from_torch(
            mask_t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        scores = ttnn.add(scores, mask)

    attn_w = ttnn.softmax(scores, dim=-1)             # (B, Hq_per, S, S)
    attn_out = ttnn.matmul(attn_w, v)                  # (B, Hq_per, S, Dh)

    # Re-assemble heads into (B, S, Hq_per*Dh)
    attn_out = ttnn.transpose(attn_out, 1, 2)          # (B, S, Hq_per, Dh)
    attn_out = ttnn.reshape(attn_out, (B, S, Hq_per * Dh))

    # Output projection — row-sharded W_O. Each chip produces a partial-sum
    # contribution to the full (B, S, D) output.
    y_partial = ttnn.matmul(attn_out, w.wo)            # (B, S, D), partial

    # All-reduce across chips
    y = all_reduce_via_gather(y_partial, mesh)
    return y


def tp_mlp(
    x: ttnn.Tensor,                  # replicated (B, S, D)
    w: TPLayerWeights,
    mesh: ttnn.MeshDevice,
    cfg: TinyLlamaConfig,
) -> ttnn.Tensor:
    """SwiGLU MLP with TP. gate/up column-sharded, down row-sharded + all_reduce."""
    g = ttnn.matmul(x, w.w_gate)         # (B, S, F/N) per chip
    u = ttnn.matmul(x, w.w_up)           # (B, S, F/N) per chip
    h = ttnn.multiply(ttnn.silu(g), u)   # (B, S, F/N) gated, per chip
    y_partial = ttnn.matmul(h, w.w_down) # (B, S, D), partial
    return all_reduce_via_gather(y_partial, mesh)


def tp_transformer_layer(
    x: ttnn.Tensor,                  # replicated (B, S, D)
    w: TPLayerWeights,
    mesh: ttnn.MeshDevice,
    cfg: TinyLlamaConfig,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
) -> ttnn.Tensor:
    """One full transformer layer: pre-norm -> attn -> resid -> pre-norm -> mlp -> resid."""
    h = ttnn.rms_norm(x, weight=w.attn_norm, epsilon=cfg.rms_norm_eps)
    a = tp_attention(h, w, mesh, cfg, cos=cos, sin=sin)
    x = ttnn.add(x, a)

    h = ttnn.rms_norm(x, weight=w.ffn_norm, epsilon=cfg.rms_norm_eps)
    m = tp_mlp(h, w, mesh, cfg)
    x = ttnn.add(x, m)
    return x
