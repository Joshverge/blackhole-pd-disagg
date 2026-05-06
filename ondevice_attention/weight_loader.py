"""
weight_loader.py - load any Llama-architecture HF model into our format.

Supports:
  - TinyLlama-1.1B-Chat-v1.0  (separate lm_head)
  - Llama-3.2-1B-Instruct     (tied embeddings: lm_head == embed_tokens)
  - Llama-2-7B / Llama-3-8B   (separate lm_head)

HF stores Linear weights as (out, in) and applies as `x @ W.T`. Our code
uses `x @ W` directly, so every Linear weight is transposed at load time.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch

from config import LlamaConfig
from reference_layer import LayerWeights


@dataclass
class FullModelWeights:
    """All weights for a full forward, in fp32 on host."""
    embed_tokens: torch.Tensor    # (vocab_size, D)
    layers: List[LayerWeights]
    final_norm: torch.Tensor      # (D,)
    lm_head: torch.Tensor         # (D, vocab_size)


def _load_state_dict(model_dir: Path) -> dict:
    """Load all weights into a single state dict on host."""
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


def load_llama(model_dir: str, cfg: LlamaConfig) -> FullModelWeights:
    """Load any Llama-arch HF model. Handles tied embeddings."""
    p = Path(model_dir)
    assert p.exists(), f"model dir not found: {p}"

    state = _load_state_dict(p)
    fp32 = lambda t: t.detach().to(torch.float32)

    # ---- Embedding (vocab_size, D) ----
    embed = fp32(state["model.embed_tokens.weight"])
    assert embed.shape == (cfg.vocab_size, cfg.hidden_size), \
        f"embed shape {tuple(embed.shape)} != ({cfg.vocab_size}, {cfg.hidden_size})"

    # ---- Final norm ----
    final_norm = fp32(state["model.norm.weight"])
    assert final_norm.shape == (cfg.hidden_size,)

    # ---- LM head (with tied-embeddings handling) ----
    if "lm_head.weight" in state:
        lm_head_hf = fp32(state["lm_head.weight"])
        assert lm_head_hf.shape == (cfg.vocab_size, cfg.hidden_size), \
            f"lm_head shape {tuple(lm_head_hf.shape)}"
        lm_head = lm_head_hf.T.contiguous()                 # (D, vocab)
    elif cfg.tie_word_embeddings:
        # lm_head == embed_tokens. Transpose for our (D, vocab) convention.
        lm_head = embed.T.contiguous()
        print("[weight_loader] tied embeddings: using model.embed_tokens.weight as lm_head")
    else:
        raise KeyError(
            "lm_head.weight is missing AND tie_word_embeddings=False - "
            "model directory may be incomplete"
        )

    # ---- Layers ----
    D = cfg.hidden_size
    F = cfg.intermediate_size
    Hq = cfg.num_attention_heads * cfg.head_dim
    Hkv = cfg.num_key_value_heads * cfg.head_dim

    layers = []
    for i in range(cfg.num_layers):
        prefix = f"model.layers.{i}."

        attn_norm = fp32(state[prefix + "input_layernorm.weight"])
        ffn_norm = fp32(state[prefix + "post_attention_layernorm.weight"])

        wq = fp32(state[prefix + "self_attn.q_proj.weight"]).T.contiguous()
        wk = fp32(state[prefix + "self_attn.k_proj.weight"]).T.contiguous()
        wv = fp32(state[prefix + "self_attn.v_proj.weight"]).T.contiguous()
        wo = fp32(state[prefix + "self_attn.o_proj.weight"]).T.contiguous()

        w_gate = fp32(state[prefix + "mlp.gate_proj.weight"]).T.contiguous()
        w_up = fp32(state[prefix + "mlp.up_proj.weight"]).T.contiguous()
        w_down = fp32(state[prefix + "mlp.down_proj.weight"]).T.contiguous()

        assert wq.shape == (D, Hq), f"layer {i} wq shape {tuple(wq.shape)} != ({D}, {Hq})"
        assert wk.shape == (D, Hkv), f"layer {i} wk shape {tuple(wk.shape)}"
        assert wv.shape == (D, Hkv), f"layer {i} wv shape {tuple(wv.shape)}"
        assert wo.shape == (Hq, D), f"layer {i} wo shape {tuple(wo.shape)}"
        assert w_gate.shape == (D, F), f"layer {i} w_gate shape {tuple(w_gate.shape)}"
        assert w_up.shape == (D, F), f"layer {i} w_up shape {tuple(w_up.shape)}"
        assert w_down.shape == (F, D), f"layer {i} w_down shape {tuple(w_down.shape)}"

        layers.append(LayerWeights(
            attn_norm=attn_norm, ffn_norm=ffn_norm,
            wq=wq, wk=wk, wv=wv, wo=wo,
            w_gate=w_gate, w_up=w_up, w_down=w_down,
        ))

    return FullModelWeights(
        embed_tokens=embed,
        layers=layers,
        final_norm=final_norm,
        lm_head=lm_head,
    )


# Back-compat alias for existing call sites.
load_tinyllama = load_llama
