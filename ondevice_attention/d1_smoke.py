#!/usr/bin/env python3
"""
d1_smoke.py .

Run:
    HF_MODEL=$HOME/models/TinyLlama-1.1B-Chat-v1.0 \\
        python d1_smoke.py
"""

import os
import sys

os.environ.setdefault("TT_VISIBLE_DEVICES", "0")  # any single chip

import torch
import torch.nn.functional as F
import ttnn

from config import TinyLlamaConfig
from weight_loader import load_tinyllama
from rope import precompute_rope, apply_rope_torch
from d1_layer import upload_layer_weights, transformer_layer_prefill


# Test config — small prompt, single layer
SEQ_LEN = 32        # 32-token prefill, tile-aligned
LAYER_IDX = 0       # test layer 0; if other layers diverge, expand later


def reference_layer_host_sdpa(
    x_torch: torch.Tensor,
    layer_w,
    cfg: TinyLlamaConfig,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Reference: same forward as path_c_real_eth/mesh_layer.py would do
    in prefill mode, but without all the ttnn dispatch — just torch.

    This is the ground truth D1 must PCC-match against. The numerical
    quirks (bf16 cast on weights, bf16 cast on activations) are
    preserved so PCC is meaningful.
    """
    Hq = cfg.num_attention_heads
    Hkv = cfg.num_key_value_heads
    Dh = cfg.head_dim
    G = cfg.gqa_group_size

    # bf16 cast everything to match what the device would do
    x = x_torch.to(torch.bfloat16).to(torch.float32)
    wq = layer_w.wq.to(torch.bfloat16).to(torch.float32)
    wk = layer_w.wk.to(torch.bfloat16).to(torch.float32)
    wv = layer_w.wv.to(torch.bfloat16).to(torch.float32)
    wo = layer_w.wo.to(torch.bfloat16).to(torch.float32)
    w_gate = layer_w.w_gate.to(torch.bfloat16).to(torch.float32)
    w_up = layer_w.w_up.to(torch.bfloat16).to(torch.float32)
    w_down = layer_w.w_down.to(torch.bfloat16).to(torch.float32)
    attn_norm = layer_w.attn_norm.to(torch.bfloat16).to(torch.float32)
    ffn_norm = layer_w.ffn_norm.to(torch.bfloat16).to(torch.float32)

    B, S, D = x.shape

    # RMSNorm pre-attention
    rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps)
    h = x * rms * attn_norm

    # QKV — note K, V at Hkv heads, Q at Hq heads
    q = h @ wq                                    # (B, S, Hq*Dh)
    k = h @ wk                                    # (B, S, Hkv*Dh)
    v = h @ wv

    q = q.view(B, S, Hq, Dh).transpose(1, 2)      # (B, Hq, S, Dh)
    k = k.view(B, S, Hkv, Dh).transpose(1, 2)     # (B, Hkv, S, Dh)
    v = v.view(B, S, Hkv, Dh).transpose(1, 2)

    # RoPE on Q and K
    q = apply_rope_torch(q, cos[:S], sin[:S])
    k = apply_rope_torch(k, cos[:S], sin[:S])

    # GQA expand for SDPA
    K_exp = k.repeat_interleave(G, dim=1)         # (B, Hq, S, Dh)
    V_exp = v.repeat_interleave(G, dim=1)

    # Causal SDPA (matches F.scaled_dot_product_attention with is_causal=True)
    attn_out = F.scaled_dot_product_attention(q, K_exp, V_exp, is_causal=(S > 1))
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, Hq * Dh)

    # Output projection + residual
    a = attn_out @ wo
    x = x + a

    # MLP
    rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps)
    h = x * rms * ffn_norm
    g = h @ w_gate
    u = h @ w_up
    silu_g = g * torch.sigmoid(g)                 # SiLU = x * sigmoid(x)
    m = (silu_g * u) @ w_down
    x = x + m
    return x


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    am, bm = a.mean(), b.mean()
    num = ((a - am) * (b - bm)).sum()
    den = (((a - am) ** 2).sum() * ((b - bm) ** 2).sum()).sqrt()
    return (num / den).item() if den.item() != 0 else 0.0


def main():
    visible = os.environ.get("TT_VISIBLE_DEVICES", "<unset>")
    print(f"TT_VISIBLE_DEVICES = {visible}")

    model_dir = os.environ.get("HF_MODEL")
    if model_dir is None:
        print("ERROR: set HF_MODEL=/path/to/TinyLlama-1.1B-Chat-v1.0", file=sys.stderr)
        return 1

    cfg = TinyLlamaConfig()
    cfg.validate()

    print(f"\n[1/5] Loading TinyLlama weights to host...")
    fmw = load_tinyllama(model_dir, cfg)
    layer_w = fmw.layers[LAYER_IDX]
    print(f"  loaded {len(fmw.layers)} layers, testing layer {LAYER_IDX}")

    print(f"\n[2/5] Building random input tensor (B=1, S={SEQ_LEN}, D={cfg.hidden_size})...")
    torch.manual_seed(0)
    x_torch = torch.randn(1, SEQ_LEN, cfg.hidden_size, dtype=torch.float32)

    cos, sin = precompute_rope(
        max_pos=cfg.max_position_embeddings,
        head_dim=cfg.head_dim,
        theta=cfg.rope_theta,
    )

    print(f"\n[3/5] Computing reference output (host SDPA, fp32 simulation)...")
    y_ref = reference_layer_host_sdpa(x_torch, layer_w, cfg, cos, sin)
    print(f"  ref shape: {tuple(y_ref.shape)}  mean={y_ref.mean().item():+.4e}  "
          f"std={y_ref.std().item():.4e}")

    print(f"\n[4/5] Running on-device path...")
    device = ttnn.open_device(device_id=0)
    try:
        # Upload layer weights
        d1_w = upload_layer_weights(layer_w, device, cfg)
        print(f"  layer weights uploaded to device 0")

        # Upload input
        x_tt = ttnn.from_torch(
            x_torch.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        # Forward
        y_tt, _, _ = transformer_layer_prefill(x_tt, d1_w, cfg, cos, sin, device)
        ttnn.synchronize_device(device)

        # Pull back
        y_dev = ttnn.to_torch(y_tt).to(torch.float32)
    finally:
        ttnn.close_device(device)

    print(f"  dev shape: {tuple(y_dev.shape)}  mean={y_dev.mean().item():+.4e}  "
          f"std={y_dev.std().item():.4e}")

    print(f"\n[5/5] PCC comparison...")
    if y_dev.shape != y_ref.shape:
        print(f"  reshaping y_dev from {tuple(y_dev.shape)} to {tuple(y_ref.shape)}")
        y_dev = y_dev.reshape(y_ref.shape)

    p = pcc(y_ref, y_dev)
    diff = (y_ref - y_dev).abs()
    rel = diff.mean().item() / y_ref.std().item() * 100

    print(f"  PCC:                  {p:.6f}")
    print(f"  mean abs diff:        {diff.mean().item():.4e}")
    print(f"  mean rel diff (% of std): {rel:.2f}%")

    PCC_TOL = 0.998
    ok = p > PCC_TOL
    print()
    print(f"=== D1 {'PASS' if ok else 'FAIL'} ===  (PCC tol = {PCC_TOL})")
    if ok:
        print("On-device SDPA is bit-faithful to the host-SDPA reference.")
    else:
        print("On-device path diverges from host reference.")
        print("Common causes: bf16 casting differences, GQA expansion order,")
        print("RoPE position offset, missing causal mask. Inspect intermediates.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
