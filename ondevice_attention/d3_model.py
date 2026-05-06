"""
d3_model.py — TinyLlama with prefill via variable-shape path and
decode via fixed-shape on-device cache.

Strategy:
  - Prefill: run once per request, variable-shape KV is acceptable because
    the cost is amortized across all generated tokens.
  - Cache transition: after prefill (or after socket migration on the
    decode side), pad the variable-shape cache to MAX_S and copy into a
    pre-allocated fixed-shape buffer. One-time host roundtrip.
  - Decode: every token uses fixed-shape kernels. JIT compiles each kernel
    once on the first token, then runs from cache.

Avoids JIT-compile thrashing while keeping prefill
code unchanged. The decode loop is the per-token-latency path; that's the
one made fixed-shape.
"""

from typing import List, Tuple

import torch
import ttnn

from config import TinyLlamaConfig
from d2_layer import (
    upload_layer_weights_to_submesh,
    transformer_layer_with_cache,
    _from_torch_replicated,
    _to_torch_local,
)
from d3_layer import (
    allocate_kv_cache_on_submesh,
    fill_cache_from_prefill,
    transformer_layer_d3_decode,
    D3LayerWeights,
)
from rope import precompute_rope, precompute_rope_transform_matrix
from weight_loader import FullModelWeights


class D3PrefillModel:
    """Prefill-only on prefill_sub. Variable
    shape is fine because prefill runs once per request)."""

    def __init__(self, fmw: FullModelWeights, submesh, cfg: TinyLlamaConfig):
        self.cfg = cfg
        self.submesh = submesh
        self.embed_tokens_host = fmw.embed_tokens

        self.layers: List[D3LayerWeights] = [
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

    def _embed(self, input_ids: torch.Tensor) -> ttnn.Tensor:
        x_host = self.embed_tokens_host[input_ids].to(torch.bfloat16)
        return _from_torch_replicated(x_host, self.submesh)

    def prefill(self, input_ids: torch.Tensor):
        """Run prefill on prefill_sub with variable-shape KV. Returns
        (logits_host, K_list, V_list) where the K, V are ttnn.Tensors on
        the submesh, one per layer, shape (1, Hq, S_padded, Dh).
        """
        K_list, V_list = [], []
        x_tt = self._embed(input_ids)
        for layer_w in self.layers:
            x_tt, full_K, full_V = transformer_layer_with_cache(
                x_tt, layer_w, self.cfg,
                self.cos, self.sin,
                cur_pos=0,
                cached_K_tt=None, cached_V_tt=None,
                submesh=self.submesh,
            )
            K_list.append(full_K)
            V_list.append(full_V)

        x_tt = ttnn.rms_norm(x_tt, weight=self.final_norm,
                             epsilon=self.cfg.rms_norm_eps)
        logits_tt = ttnn.matmul(x_tt, self.lm_head)
        ttnn.synchronize_device(self.submesh)
        logits = _to_torch_local(logits_tt)
        return logits, K_list, V_list


class D3DecodeModel:
    """Decode-only on decode_sub. Pre-allocates fixed-shape KV buffers.
    All decode-step kernel shapes are stable; JIT compile happens once.
    """

    def __init__(self, fmw: FullModelWeights, submesh, cfg: TinyLlamaConfig,
                 max_s: int):
        self.cfg = cfg
        self.submesh = submesh
        self.max_s = max_s
        self.embed_tokens_host = fmw.embed_tokens

        self.layers: List[D3LayerWeights] = [
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

        # Upload the ROPE transformation matrix once. Shared across all 22 layers
        # and all decode steps — constant device tensor.
        T_host = precompute_rope_transform_matrix(cfg.head_dim).reshape(
            1, 1, cfg.head_dim, cfg.head_dim
        )
        self.trans_mat_tt = _from_torch_replicated(T_host, submesh)

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
        """One-shot copy from prefill's variable-shape K, V into
        fixed-shape buffers. Called at the prefill→decode transition,
        after migration to decode_sub.
        """
        self.K_caches, self.V_caches = fill_cache_from_prefill(
            self.K_caches, self.V_caches,
            prefill_K_list, prefill_V_list,
            real_len, self.max_s, self.submesh,
        )
        self.cur_pos = real_len

    def decode_step(self, token_id: int) -> torch.Tensor:
        """One decode step. All kernel shapes stable across calls.
        Updates cache via repeat+mask+add. Returns logits.
        """
        cur_pos = self.cur_pos
        token_t = torch.tensor([[token_id]], dtype=torch.int64)
        x_tt = self._embed(token_t)

        # Upload cos[cur_pos] and sin[cur_pos] ONCE per decode step.
        # Shape (1, 1, 1, Dh). Reused across all 22 layers — replaces the
        # 44 per-token PCIe roundtrips that the host-RoPE path was doing.
        cos_slice = self.cos[cur_pos : cur_pos + 1].reshape(
            1, 1, 1, self.cfg.head_dim
        ).to(torch.float32)
        sin_slice = self.sin[cur_pos : cur_pos + 1].reshape(
            1, 1, 1, self.cfg.head_dim
        ).to(torch.float32)
        cos_tt = _from_torch_replicated(cos_slice, self.submesh)
        sin_tt = _from_torch_replicated(sin_slice, self.submesh)

        for layer_idx, layer_w in enumerate(self.layers):
            x_tt, K_new, V_new = transformer_layer_d3_decode(
                x_tt, layer_w, self.cfg,
                cos_tt, sin_tt, self.trans_mat_tt,
                cur_pos,
                self.K_caches[layer_idx], self.V_caches[layer_idx],
                self.max_s, self.submesh,
            )
            # CRITICAL: write the updated caches back so the NEXT decode token
            # sees this token's K, V at cur_pos. ttnn.add returns a new tensor,
            # so without this, every step starts from the prefill-only cache.
            self.K_caches[layer_idx] = K_new
            self.V_caches[layer_idx] = V_new

        x_tt = ttnn.rms_norm(x_tt, weight=self.final_norm,
                             epsilon=self.cfg.rms_norm_eps)
        logits_tt = ttnn.matmul(x_tt, self.lm_head)
        ttnn.synchronize_device(self.submesh)
        logits = _to_torch_local(logits_tt)

        self.cur_pos = cur_pos + 1
        return logits
