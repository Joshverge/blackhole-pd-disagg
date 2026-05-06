"""
d5_layer.py — D4 + paged_update_cache for the cache write.

Differences from d3_layer.py (D4):
  - Cache write was: ttnn.repeat → ttnn.multiply (one-hot mask) → ttnn.add.
    Touches MAX_S × Hq × Dh elements per (K|V) per layer.
  - Cache write is now: ttnn.to_memory_config (DRAM → L1 sharded) →
    ttnn.experimental.paged_update_cache.
    Touches B × Hq × Dh elements per (K|V) per layer.

For B=1, Hq=32, Dh=64, MAX_S=128 that's 128× less data moved per
cache write per layer.

"""

import math
from typing import Tuple

import torch
import ttnn

from config import TinyLlamaConfig
from rope import apply_rope_ondevice
from d2_layer import (
    D2LayerWeights,
    upload_layer_weights_to_submesh,
    _from_torch_replicated,
    _to_torch_local,
)
# Reuse D3's cache allocator and prefill→decode bridge — unchanged.
from d3_layer import (
    allocate_kv_cache_on_submesh,
    fill_cache_from_prefill,
)


# Reuse the same layer weight container.
D5LayerWeights = D2LayerWeights
upload_d5_layer_weights = upload_layer_weights_to_submesh


def build_kv_input_mem_config(submesh, B: int, Hq: int, Dh: int) -> ttnn.MemoryConfig:
    """Pre-build the L1 HEIGHT_SHARDED memory config for K_new / V_new.
    Done once at model init; the same config is reused on every decode step
    for every layer. Validated by d5_smoke.py.

    For B=1, Hq=32, Dh=64: one core, shard shape [Hq, Dh] = [32, 64].
    """
    compute_grid = submesh.compute_with_storage_grid_size()
    shard_grid = ttnn.num_cores_to_corerangeset(B, compute_grid, True)
    # Input shape (1, B, Hq, Dh). With B users on B cores, each shard is
    # one (Hq, Dh) tile-row.
    shard_shape = [Hq, Dh]
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        shard_spec,
    )


def attention_d5_decode(
    x_tt: ttnn.Tensor,
    w: D5LayerWeights,
    cfg: TinyLlamaConfig,
    cos_tt: ttnn.Tensor,
    sin_tt: ttnn.Tensor,
    trans_mat_tt: ttnn.Tensor,
    cur_pos_tt: ttnn.Tensor,         # int32 (B,) on device — uploaded once per step
    cur_pos: int,                    # also pass int for the SDPA position mask
    K_cache_tt: ttnn.Tensor,
    V_cache_tt: ttnn.Tensor,
    kv_input_mem_config: ttnn.MemoryConfig,
    max_s: int,
    submesh,
) -> ttnn.Tensor:
    """Decode-mode attention with paged_update_cache cache write.

    Key differences from d3:
      - K_new, V_new stay in (1, B, Hq, Dh) shape (no transpose to
        (1, Hq, B, Dh)) — that's the shape paged_update_cache wants.
      - Cache write is one fused op instead of repeat+mask+add.
      - paged_update_cache modifies the cache in place — no return values.
        We don't need the reference-passing dance from D3.

    Q is still transposed to (1, Hq, 1, Dh) for the SDPA matmul.
    """
    B = 1
    S_new = 1
    Hq = cfg.num_attention_heads
    Dh = cfg.head_dim

    # 1. Project Q, K_new, V_new on device.
    q = ttnn.matmul(x_tt, w.wq)
    k_new = ttnn.matmul(x_tt, w.wk)
    v_new = ttnn.matmul(x_tt, w.wv)

    # 2. Reshape to (1, S=1, Hq, Dh). For paged_update_cache, K_new and
    # V_new want this shape directly (B is dim 1, Hq is dim 2). Q still
    # gets transposed to (1, Hq, 1, Dh) for the SDPA matmul.
    q = ttnn.reshape(q, (B, S_new, Hq, Dh))
    k_new = ttnn.reshape(k_new, (B, S_new, Hq, Dh))   # (1, 1, Hq, Dh)
    v_new = ttnn.reshape(v_new, (B, S_new, Hq, Dh))   # (1, 1, Hq, Dh)
    q = ttnn.transpose(q, 1, 2)                        # (1, Hq, 1, Dh)

    # 3. ROPE on device. apply_rope_ondevice operates on the last dim and
    # broadcasts cos/sin/trans_mat over leading dims, so it works on both
    # (1, Hq, 1, Dh) Q and (1, 1, Hq, Dh) K_new.
    q = apply_rope_ondevice(q, cos_tt, sin_tt, trans_mat_tt)
    k_new = apply_rope_ondevice(k_new, cos_tt, sin_tt, trans_mat_tt)

    # 4. Update fixed-shape cache via two separate paged_update_cache calls.
    # paged_fused_update_cache requires K_new and V_new to be sharded on
    # disjoint cores (validator: "input_tensor1 and input_tensor2 must not
    # overlap"). With B=1 and a single-core shard, both inputs land on core
    # (0,0) and collide. Two separate calls avoid the constraint — each
    # takes one input. tt_transformers/attention.py:739-744 uses this same
    # non-fused path when use_qk_fused=False.
    k_new_sh = ttnn.to_memory_config(k_new, kv_input_mem_config)
    ttnn.experimental.paged_update_cache(
        K_cache_tt, k_new_sh,
        update_idxs_tensor=cur_pos_tt,
        share_cache=False,
    )
    ttnn.deallocate(k_new_sh)
    ttnn.deallocate(k_new)

    v_new_sh = ttnn.to_memory_config(v_new, kv_input_mem_config)
    ttnn.experimental.paged_update_cache(
        V_cache_tt, v_new_sh,
        update_idxs_tensor=cur_pos_tt,
        share_cache=False,
    )
    ttnn.deallocate(v_new_sh)
    ttnn.deallocate(v_new)

    # 5. SDPA against fixed-shape cache (1, Hq, MAX_S, Dh) — unchanged from D3/D4.
    K_t = ttnn.transpose(K_cache_tt, -2, -1)
    scores = ttnn.matmul(q, K_t)
    scores = ttnn.multiply(scores, 1.0 / math.sqrt(Dh))

    # 6. Position mask: positions > cur_pos get -inf.
    mask_t = torch.full((max_s,), float("-inf"), dtype=torch.bfloat16)
    mask_t[: cur_pos + 1] = 0.0
    mask_t = mask_t.reshape(1, 1, 1, max_s).to(torch.float32)
    mask_tt = _from_torch_replicated(mask_t, submesh)
    scores = ttnn.add(scores, mask_tt)

    attn_w = ttnn.softmax(scores, dim=-1)
    attn_out = ttnn.matmul(attn_w, V_cache_tt)

    # 7. Reassemble + output projection.
    attn_out = ttnn.transpose(attn_out, 1, 2)
    attn_out = ttnn.reshape(attn_out, (1, 1, Hq * Dh))
    output = ttnn.matmul(attn_out, w.wo)
    return output


def mlp_ondevice(x_tt, w) -> ttnn.Tensor:
    g = ttnn.matmul(x_tt, w.w_gate)
    u = ttnn.matmul(x_tt, w.w_up)
    h = ttnn.multiply(ttnn.silu(g), u)
    return ttnn.matmul(h, w.w_down)


def transformer_layer_d5_decode(
    x_tt: ttnn.Tensor,
    w: D5LayerWeights,
    cfg: TinyLlamaConfig,
    cos_tt: ttnn.Tensor,
    sin_tt: ttnn.Tensor,
    trans_mat_tt: ttnn.Tensor,
    cur_pos_tt: ttnn.Tensor,
    cur_pos: int,
    K_cache_tt: ttnn.Tensor,
    V_cache_tt: ttnn.Tensor,
    kv_input_mem_config: ttnn.MemoryConfig,
    max_s: int,
    submesh,
) -> ttnn.Tensor:
    """One full transformer layer in decode mode. Returns x_out only —
    paged_fused_update_cache modified the caches in place, so no reference
    passing needed."""
    h = ttnn.rms_norm(x_tt, weight=w.attn_norm, epsilon=cfg.rms_norm_eps)
    a = attention_d5_decode(
        h, w, cfg, cos_tt, sin_tt, trans_mat_tt,
        cur_pos_tt, cur_pos,
        K_cache_tt, V_cache_tt,
        kv_input_mem_config, max_s, submesh,
    )
    x_tt = ttnn.add(x_tt, a)

    h = ttnn.rms_norm(x_tt, weight=w.ffn_norm, epsilon=cfg.rms_norm_eps)
    m = mlp_ondevice(h, w)
    x_tt = ttnn.add(x_tt, m)
    return x_tt
