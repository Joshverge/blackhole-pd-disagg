"""
d5_smoke.py — Isolated smoke test for ttnn.experimental.paged_update_cache.

Run on a single chip — no submesh, no model:
    TT_VISIBLE_DEVICES=0 python path_d_ondevice/d5_smoke.py
"""

import os
import sys

# Single chip, no submesh — just need a device.
os.environ.setdefault("TT_VISIBLE_DEVICES", "0")

import torch
import ttnn


def main():
    print("=" * 70)
    print("D5 smoke — paged_update_cache recipe validation")
    print("=" * 70)

    # Match TinyLlama decode shapes.
    B = 1
    Hq = 32
    Dh = 64
    MAX_S = 128

    # Open device.
    device = ttnn.open_device(device_id=0)
    print(f"opened device {device.id()}")

    try:
        run_test(device, B, Hq, Dh, MAX_S)
    finally:
        ttnn.close_device(device)
        print("device closed")


def run_test(device, B, Hq, Dh, MAX_S):
    print(f"shapes: cache=({B}, {Hq}, {MAX_S}, {Dh})  "
          f"input=(1, {B}, {Hq}, {Dh})")

    # ---- 1. Cache allocation: DRAM-interleaved, zero-initialized.
    cache_host = torch.zeros(B, Hq, MAX_S, Dh, dtype=torch.bfloat16)
    cache_tt = ttnn.from_torch(
        cache_host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    print("  cache allocated in DRAM")

    # ---- 2. Build the sharding spec for K_new input. Done ONCE; reusable.
    compute_grid = device.compute_with_storage_grid_size()
    print(f"  compute grid: {compute_grid}")
    shard_grid = ttnn.num_cores_to_corerangeset(B, compute_grid, True)
    # For a (1, B, Hq, Dh) bf16 TILE input:
    #   volume = 1 * B * Hq * Dh
    #   padded_shape[-1] = Dh
    #   shard_shape = [volume / Dh / B, Dh] = [Hq, Dh] = one tile-row per user
    shard_shape = [Hq, Dh]
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        shard_spec,
    )
    print(f"  shard config: HEIGHT_SHARDED L1, shape={shard_shape}, "
          f"{B} core(s)")

    # ---- 3. Test sequence: write at three positions, verify each.
    write_positions = [0, 12, 47]
    written_data = {}

    for cur_pos in write_positions:
        # Make a recognizable input: all values = cur_pos+1 (so 0 isn't ambiguous)
        x_value = float(cur_pos + 1)
        x_host = torch.full((1, B, Hq, Dh), x_value, dtype=torch.bfloat16)
        written_data[cur_pos] = x_host.clone()

        # Upload + shard.
        x_tt_dram = ttnn.from_torch(
            x_host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x_tt_sharded = ttnn.to_memory_config(x_tt_dram, input_mem_config)

        # update_idxs: int32 (B,) on device.
        idxs_host = torch.tensor([cur_pos], dtype=torch.int32)
        idxs_tt = ttnn.from_torch(idxs_host, dtype=ttnn.int32, device=device)

        # Call paged_update_cache. In-place modification of cache_tt.
        ttnn.experimental.paged_update_cache(
            cache_tt,
            x_tt_sharded,
            update_idxs_tensor=idxs_tt,
            share_cache=False,
        )
        print(f"  wrote x={x_value:.0f} at cur_pos={cur_pos}")

        # Cleanup ephemeral tensors.
        ttnn.deallocate(x_tt_dram)
        ttnn.deallocate(x_tt_sharded)
        ttnn.deallocate(idxs_tt)

    # ---- 4. Pull cache back, verify.
    cache_back = ttnn.to_torch(cache_tt).to(torch.float32)
    print(f"\n  cache shape after writes: {tuple(cache_back.shape)}")

    failures = 0

    # Check each written position has the right value.
    for cur_pos in write_positions:
        expected_val = float(cur_pos + 1)
        # cache shape (B, Hq, MAX_S, Dh) — slot at cur_pos
        slot = cache_back[0, :, cur_pos, :]   # (Hq, Dh)
        actual_val = slot.mean().item()
        if abs(actual_val - expected_val) < 0.01:
            print(f"  ✓ cur_pos={cur_pos:3d}: mean={actual_val:.3f} "
                  f"(expected {expected_val:.0f})")
        else:
            print(f"  ✗ cur_pos={cur_pos:3d}: mean={actual_val:.3f} "
                  f"(expected {expected_val:.0f}) — FAIL")
            failures += 1

    # Check unwritten positions are still zero.
    unwritten = [1, 5, 50, 100, 127]
    for pos in unwritten:
        if pos in write_positions:
            continue
        slot = cache_back[0, :, pos, :]
        actual_val = slot.mean().item()
        if abs(actual_val) < 0.01:
            print(f"  ✓ pos={pos:3d} (unwritten): mean={actual_val:.4f}")
        else:
            print(f"  ✗ pos={pos:3d} (unwritten): mean={actual_val:.4f} "
                  f"— should be 0, FAIL")
            failures += 1

    print()
    if failures == 0:
        print("  ALL CHECKS PASSED — paged_update_cache recipe is correct")
        print("  ready to integrate into d3_layer.py as D5")
    else:
        print(f"  FAILED ({failures} check(s) failed)")
        sys.exit(1)


if __name__ == "__main__":
    main()
