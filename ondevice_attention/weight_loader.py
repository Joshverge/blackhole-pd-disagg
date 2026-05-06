"""
weight_loader.py — load TinyLlama-1.1B-Chat-v1.0 HF weights into our
LayerWeights / Model dataclasses.

HF stores Linear weights as (out, in) and applies as `x @ W.T`. Our
reference / TP code uses `x @ W` directly, so every Linear weight is
transposed at load time.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch

from config import TinyLlamaConfig
from reference_layer import LayerWeights


@dataclass
class FullModelWeights:
    """All weights for a full TinyLlama forward, in fp32 on host."""
    embed_tokens: torch.Tensor    # (vocab_size, D)
    layers: List[LayerWeights]    # 22 entries
    final_norm: torch.Tensor      # (D,)
    lm_head: torch.Tensor         # (D, vocab_size)


def _load_state_dict(model_dir: Path) -> dict:
    """Load all weights into a single fp32 state dict on host."""
    sd_paths = sorted(model_dir.glob("model-*.safetensors"))
    bin_path = model_dir / "pytorch_model.bin"
    single_st = model_dir / "model.safetensors"

    if sd_paths:
        from safetensors.torch import load_file
        state = {}
        for p in sd_paths:
            state.update(load_file(str(p)))
        return state
    if single_st.exists():
        from safetensors.torch import load_file
        return load_file(str(single_st))
    if bin_path.exists():
        return torch.load(str(bin_path), map_location="cpu")
    raise FileNotFoundError(
        f"No weights found in {model_dir} (expected model.safetensors, "
        f"model-*.safetensors, or pytorch_model.bin)"
    )


def load_tinyllama(model_dir: str, cfg: TinyLlamaConfig) -> FullModelWeights:
    """Load TinyLlama HF weights from a local directory.

    Returns FullModelWeights in fp32 on host. The transpose to our `x @ W`
    convention is applied here so callers don't have to worry about it.
    """
    p = Path(model_dir)
    assert p.exists(), f"model dir not found: {p}"

    state = _load_state_dict(p)
    fp32 = lambda t: t.detach().to(torch.float32)

    # Embedding (vocab_size, D), used as embed[ids] — no transpose needed
    embed = fp32(state["model.embed_tokens.weight"])
    assert embed.shape == (cfg.vocab_size, cfg.hidden_size), \
        f"embed shape {embed.shape} != ({cfg.vocab_size}, {cfg.hidden_size})"

    final_norm = fp32(state["model.norm.weight"])
    assert final_norm.shape == (cfg.hidden_size,)

    # LM head (vocab, D) — applied as x @ W.T in HF, so our lm_head = LM_head.T
    lm_head_hf = fp32(state["lm_head.weight"])
    assert lm_head_hf.shape == (cfg.vocab_size, cfg.hidden_size), \
        f"lm_head shape {lm_head_hf.shape}"
    lm_head = lm_head_hf.T.contiguous()                  # (D, vocab)

    layers = []
    for i in range(cfg.num_layers):
        prefix = f"model.layers.{i}."

        attn_norm = fp32(state[prefix + "input_layernorm.weight"])
        ffn_norm = fp32(state[prefix + "post_attention_layernorm.weight"])

        # Attention projections: HF stores (out, in), we need (in, out)
        wq = fp32(state[prefix + "self_attn.q_proj.weight"]).T.contiguous()  # (D, Hq*Dh)
        wk = fp32(state[prefix + "self_attn.k_proj.weight"]).T.contiguous()  # (D, Hkv*Dh)
        wv = fp32(state[prefix + "self_attn.v_proj.weight"]).T.contiguous()  # (D, Hkv*Dh)
        wo = fp32(state[prefix + "self_attn.o_proj.weight"]).T.contiguous()  # (Hq*Dh, D)

        w_gate = fp32(state[prefix + "mlp.gate_proj.weight"]).T.contiguous()  # (D, F)
        w_up = fp32(state[prefix + "mlp.up_proj.weight"]).T.contiguous()      # (D, F)
        w_down = fp32(state[prefix + "mlp.down_proj.weight"]).T.contiguous()  # (F, D)

        # Sanity-check shapes
        D = cfg.hidden_size
        F = cfg.intermediate_size
        Hq = cfg.num_attention_heads * cfg.head_dim
        Hkv = cfg.num_key_value_heads * cfg.head_dim
        assert wq.shape == (D, Hq), f"layer {i} wq shape {wq.shape}"
        assert wk.shape == (D, Hkv), f"layer {i} wk shape {wk.shape}"
        assert wv.shape == (D, Hkv), f"layer {i} wv shape {wv.shape}"
        assert wo.shape == (Hq, D), f"layer {i} wo shape {wo.shape}"
        assert w_gate.shape == (D, F), f"layer {i} w_gate shape {w_gate.shape}"
        assert w_up.shape == (D, F), f"layer {i} w_up shape {w_up.shape}"
        assert w_down.shape == (F, D), f"layer {i} w_down shape {w_down.shape}"

        layers.append(LayerWeights(
            attn_norm=attn_norm,
            ffn_norm=ffn_norm,
            wq=wq, wk=wk, wv=wv, wo=wo,
            w_gate=w_gate, w_up=w_up, w_down=w_down,
        ))

    return FullModelWeights(
        embed_tokens=embed,
        layers=layers,
        final_norm=final_norm,
        lm_head=lm_head,
    )
