"""
d2_model.py — Full TinyLlama on a submesh with ON DEVICE KV cache.
"""

from typing import List, Tuple

import torch
import ttnn

from config import TinyLlamaConfig
from d2_layer import (
    D2LayerWeights,
    upload_layer_weights_to_submesh,
    transformer_layer_with_cache,
    _from_torch_replicated,
    _to_torch_local,
)
from rope import precompute_rope
from weight_loader import FullModelWeights


class D2OnDeviceCache:
    """KV cache stored on a single submesh as ttnn.Tensors.
    K[i], V[i] each have shape (B, Hq, cur_pos, Dh) bf16 on device.
    """
    def __init__(self):
        self.K: List[ttnn.Tensor] = []
        self.V: List[ttnn.Tensor] = []
        self.cur_pos: int = 0

    def num_layers(self) -> int:
        return len(self.K)


class D2Model:
    """Single-submesh TinyLlama with replicated weights and on-device math.

    Holds:
      - weights (replicated on the submesh)
      - cos/sin RoPE tables on host
      - the embedding table on host (we look up tokens then upload)
    """

    def __init__(self, fmw: FullModelWeights, submesh, cfg: TinyLlamaConfig):
        self.cfg = cfg
        self.submesh = submesh
        self.embed_tokens_host = fmw.embed_tokens

        self.layers: List[D2LayerWeights] = [
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

        self.cos, self.sin = precompute_rope(
            max_pos=cfg.max_position_embeddings,
            head_dim=cfg.head_dim,
            theta=cfg.rope_theta,
        )

    def _embed(self, input_ids: torch.Tensor) -> ttnn.Tensor:
        """Look up embeddings on host (it's a small int→vector lookup),
        then upload the activation to the submesh."""
        x_host = self.embed_tokens_host[input_ids].to(torch.bfloat16)
        return _from_torch_replicated(x_host, self.submesh)

    def _final_logits(self, x_tt: ttnn.Tensor) -> torch.Tensor:
        """Final RMSNorm + lm_head on device, then pull back to host
        (we pick next token on host with argmax)."""
        x_tt = ttnn.rms_norm(x_tt, weight=self.final_norm,
                             epsilon=self.cfg.rms_norm_eps)
        logits_tt = ttnn.matmul(x_tt, self.lm_head)
        ttnn.synchronize_device(self.submesh)
        return _to_torch_local(logits_tt)

    def prefill(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, D2OnDeviceCache]:
        """Run prefill on the submesh. Returns (logits_host, cache_on_device).
        The cache's K[i], V[i] are ttnn.Tensors that live on this submesh's
        device DRAM. Subsequent decode steps grow them via ttnn.concat.
        """
        cache = D2OnDeviceCache()
        x_tt = self._embed(input_ids)
        for layer_w in self.layers:
            x_tt, full_K, full_V = transformer_layer_with_cache(
                x_tt, layer_w, self.cfg,
                self.cos, self.sin,
                cur_pos=0,
                cached_K_tt=None, cached_V_tt=None,
                submesh=self.submesh,
            )
            cache.K.append(full_K)
            cache.V.append(full_V)
        cache.cur_pos = input_ids.shape[-1]
        logits = self._final_logits(x_tt)
        return logits, cache

    def decode_step(self, token_id: int,
                    cache: D2OnDeviceCache) -> Tuple[torch.Tensor, D2OnDeviceCache]:
        """One decode step. Reads/writes the on-device KV cache."""
        cur_pos = cache.cur_pos
        token_t = torch.tensor([[token_id]], dtype=torch.int64)
        x_tt = self._embed(token_t)

        for layer_idx, layer_w in enumerate(self.layers):
            x_tt, full_K, full_V = transformer_layer_with_cache(
                x_tt, layer_w, self.cfg,
                self.cos, self.sin,
                cur_pos=cur_pos,
                cached_K_tt=cache.K[layer_idx],
                cached_V_tt=cache.V[layer_idx],
                submesh=self.submesh,
            )
            cache.K[layer_idx] = full_K
            cache.V[layer_idx] = full_V

        cache.cur_pos = cur_pos + 1
        logits = self._final_logits(x_tt)
        return logits, cache
