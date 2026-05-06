"""
d5_model.py — TinyLlama with D5 decode (paged_update_cache cache write).

Same structure as d3_model.py. Differences:
  - Builds the L1 sharded memory config for K_new/V_new ONCE at init.
  - At each decode_step, uploads cur_pos as int32 (B,) ONCE per step
    (not per layer), shared across all 22 layers.
  - Calls transformer_layer_d5_decode (which uses paged_fused_update_cache
    and modifies caches in place — no return-value plumbing for caches).

Prefill side is same as D3 (D3PrefillModel).
"""

from typing import List

import torch
import ttnn

from config import TinyLlamaConfig
from d2_layer import (
    upload_layer_weights_to_submesh,
    _from_torch_replicated,
    _to_torch_local,
)
from d5_layer import (
    build_kv_input_mem_config,
    transformer_layer_d5_decode,
    D5LayerWeights,
)
from d3_layer import (
    allocate_kv_cache_on_submesh,
    fill_cache_from_prefill,
)
from d3_model import D3PrefillModel  # prefill is unchanged
from rope import precompute_rope, precompute_rope_transform_matrix
from weight_loader import FullModelWeights


D5PrefillModel = D3PrefillModel


class D5DecodeModel:
    """Decode-only on decode_sub. Uses paged_fused_update_cache for cache
    write — fewer ops, fewer bytes moved per layer than D3's broadcast+mask.
    """

    def __init__(self, fmw: FullModelWeights, submesh, cfg: TinyLlamaConfig,
                 max_s: int):
        self.cfg = cfg
        self.submesh = submesh
        self.max_s = max_s
        self.embed_tokens_host = fmw.embed_tokens

        self.layers: List[D5LayerWeights] = [
            upload_layer_weights_to_submesh(lw, submesh, cfg) for lw in fmw.layers
        ]

        self.final_norm = ttnn.from_torch(
            fmw.final_norm.reshape(1, -1).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=submesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
        )
        self.lm_head = ttnn.from_torch(
            fmw.lm_head.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=submesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
        )
        # Cap precompute at 8192 positions. Llama-3.2 reports 131072 in its
        # config which would build a 64MB table on host. Standard RoPE only
        # (Llama-3.2 long-context frequency scaling not implemented).
        rope_max = min(cfg.max_position_embeddings, 8192)
        self.cos, self.sin = precompute_rope(
            max_pos=rope_max,
            head_dim=cfg.head_dim,
            theta=cfg.rope_theta,
        )

        # ROPE transformation matrix — uploaded once, shared across all layers/steps.
        T_host = precompute_rope_transform_matrix(cfg.head_dim).reshape(
            1, 1, cfg.head_dim, cfg.head_dim
        )
        self.trans_mat_tt = _from_torch_replicated(T_host, submesh)

        # L1 sharded memory config for K_new / V_new — built once, reused
        # on every layer of every decode step.
        self.kv_input_mem_config = build_kv_input_mem_config(
            submesh, B=1, Hq=cfg.num_attention_heads, Dh=cfg.head_dim
        )

        # Pre-allocate fixed-shape cache buffers — one per layer.
        self.K_caches, self.V_caches = allocate_kv_cache_on_submesh(
            cfg, max_s, submesh
        )
        self.cur_pos = 0

    def _embed(self, input_ids: torch.Tensor) -> ttnn.Tensor:
        x_host = self.embed_tokens_host[input_ids].to(torch.bfloat16)
        return _from_torch_replicated(x_host, self.submesh)

    def populate_cache_from_prefill(self, prefill_K_list, prefill_V_list,
                                    real_len: int):
        """Same one-shot copy as D3."""
        self.K_caches, self.V_caches = fill_cache_from_prefill(
            self.K_caches, self.V_caches,
            prefill_K_list, prefill_V_list,
            real_len, self.max_s, self.submesh,
        )
        self.cur_pos = real_len

    def decode_step(self, token_id: int) -> torch.Tensor:
        """One decode step. Uploads cos/sin/cur_pos ONCE per step, calls
        all 22 layers with paged_fused_update_cache for the cache write."""
        cur_pos = self.cur_pos
        token_t = torch.tensor([[token_id]], dtype=torch.int64)
        x_tt = self._embed(token_t)

        # Upload cos[cur_pos] and sin[cur_pos] once per step.
        cos_slice = self.cos[cur_pos : cur_pos + 1].reshape(
            1, 1, 1, self.cfg.head_dim
        ).to(torch.float32)
        sin_slice = self.sin[cur_pos : cur_pos + 1].reshape(
            1, 1, 1, self.cfg.head_dim
        ).to(torch.float32)
        cos_tt = _from_torch_replicated(cos_slice, self.submesh)
        sin_tt = _from_torch_replicated(sin_slice, self.submesh)

        # Upload cur_pos as int32 (B=1,) — shared across all 22 layers.
        cur_pos_host = torch.tensor([cur_pos], dtype=torch.int32)
        cur_pos_tt = ttnn.from_torch(
            cur_pos_host,
            dtype=ttnn.int32,
            device=self.submesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.submesh),
        )

        for layer_idx, layer_w in enumerate(self.layers):
            x_tt = transformer_layer_d5_decode(
                x_tt, layer_w, self.cfg,
                cos_tt, sin_tt, self.trans_mat_tt,
                cur_pos_tt, cur_pos,
                self.K_caches[layer_idx], self.V_caches[layer_idx],
                self.kv_input_mem_config,
                self.max_s, self.submesh,
            )
            # paged_fused_update_cache modified the caches in place;
            # nothing to write back.

        x_tt = ttnn.rms_norm(x_tt, weight=self.final_norm,
                             epsilon=self.cfg.rms_norm_eps)
        logits_tt = ttnn.matmul(x_tt, self.lm_head)
        ttnn.synchronize_device(self.submesh)
        logits = _to_torch_local(logits_tt)

        self.cur_pos = cur_pos + 1
        return logits
