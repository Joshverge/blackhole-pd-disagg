"""
rope.py — rotary position embeddings (RoPE).

D2 we apply RoPE via a host roundtrip: the Q/K matmul output
gets pulled off the mesh, rotated on host (fp32), and sent back with the
same column-shard layout. This is slow (extra ETH transfers per layer) but
correct. We can swap to a device-side rotate_half in checkpoint 4 if
latency is a concern.

RoPE math (Llama variant):
    cos, sin: shape (S, Dh)  derived from positions and base frequencies
    rotate_half(x): split last dim in two; output = [-x2, x1]
    apply: y = x * cos + rotate_half(x) * sin
"""

from typing import Tuple

import torch
import ttnn


def precompute_rope(
    max_pos: int,
    head_dim: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for all positions in [0, max_pos).

    Returns (cos, sin) each of shape (max_pos, head_dim), fp32.
    """
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)                    # (max_pos, half)
    emb = torch.cat([freqs, freqs], dim=-1)             # (max_pos, head_dim)
    return emb.cos(), emb.sin()


def rotate_half_torch(x: torch.Tensor) -> torch.Tensor:
    """Llama RoPE rotate_half: [-x2, x1]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope_torch(
    x: torch.Tensor,                # (..., S, Dh)
    cos: torch.Tensor,              # (S, Dh) — broadcasts over leading dims
    sin: torch.Tensor,              # (S, Dh)
) -> torch.Tensor:
    """RoPE on a single tensor (Q or K). cos/sin must already be sliced
    to the right sequence length."""
    return x * cos + rotate_half_torch(x) * sin


def precompute_rope_transform_matrix(head_dim: int) -> torch.Tensor:
    """Build a (head_dim, head_dim) matrix T such that  x @ T == rotate_half(x).

    rotate_half splits the last dim in two and returns concat([-x2, x1]).
    Equivalently, output[j] = -x[j+half]  for j < half
                  output[j] = +x[j-half]  for j >= half

    So T[i][j] = -1 when i >= half and j == i - half
                = +1 when i <  half and j == i + half
                = 0  otherwise.

    Using this matrix lets us express ROPE as a matmul (no slicing/concat):
        x_rot = x @ T
        y     = x * cos + x_rot * sin

    All decode-step shapes are stable, so JIT compiles once.
    """
    half = head_dim // 2
    T = torch.zeros(head_dim, head_dim, dtype=torch.float32)
    for i in range(half):
        T[i, i + half] = 1.0
    for i in range(half, head_dim):
        T[i, i - half] = -1.0
    return T


def apply_rope_ondevice(
    x_tt: ttnn.Tensor,          # (B, Hq, S, Dh)  — typically (1, Hq, 1, Dh) at decode
    cos_tt: ttnn.Tensor,        # (1, 1, 1, Dh) — broadcasts over batch+head
    sin_tt: ttnn.Tensor,        # (1, 1, 1, Dh)
    trans_mat_tt: ttnn.Tensor,  # (1, 1, Dh, Dh)
) -> ttnn.Tensor:
    """ROPE on device. Uses a transformation matrix to express rotate_half
    as a matmul, avoiding the per-layer host roundtrip used by D2/D3.

    All inputs are device tensors with stable shapes — JIT compiles once.
    """
    x_rot = ttnn.matmul(x_tt, trans_mat_tt)
    return ttnn.add(
        ttnn.multiply(x_tt, cos_tt),
        ttnn.multiply(x_rot, sin_tt),
    )


def apply_rope_via_host(
    q_tt: ttnn.Tensor,
    k_tt: ttnn.Tensor,
    cos: torch.Tensor,              # (S, Dh) host fp32
    sin: torch.Tensor,              # (S, Dh) host fp32
    mesh: ttnn.MeshDevice,
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    """Apply RoPE by bringing Q and K to host, rotating, sending back.

    Args:
        q_tt, k_tt: column-sharded across mesh, shape (B, H_per_chip, S, Dh) per chip
        cos, sin: host fp32, broadcastable over batch and head dims

    Returns:
        (q_rotated, k_rotated) — same TP layout as inputs
    """
    def _rope_one(t_tt: ttnn.Tensor) -> ttnn.Tensor:
        shards = ttnn.get_device_tensors(t_tt)
        rotated = []
        for shard in shards:
            x = ttnn.to_torch(shard).to(torch.float32)
            rot = apply_rope_torch(x, cos, sin)
            rotated.append(rot)

        # Concat per-chip shards along the head axis (dim=1) and re-shard.
        full = torch.cat(rotated, dim=1)
        return ttnn.from_torch(
            full.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
        )

    return _rope_one(q_tt), _rope_one(k_tt)
