#!/usr/bin/env python3
"""
d5_pd_full.py — Integrated PD disagg with paged_update_cache.

Differences from d3_pd_full.py:
  - D3/D4: cache write was repeat → multiply (one-hot mask) → add.
    Touched MAX_S × Hq × Dh elements per (K|V) per layer per token.
  - D5: cache write is to_memory_config → paged_update_cache.
    Touches only B × Hq × Dh elements per (K|V) per layer per token.

For B=1, Hq=32, Dh=64, MAX_S=128 that's 128× less data moved per
cache write per layer.

Run:
    HF_MODEL=$HOME/models/TinyLlama-1.1B-Chat-v1.0 \\
        python d5_pd_full.py --chat --prompt "What is your favorite condiment?" \\
            --max_new_tokens 100
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("TT_VISIBLE_DEVICES", "2,3")

import torch
import ttnn

from config import TinyLlamaConfig
from d5_model import D5PrefillModel, D5DecodeModel
from weight_loader import load_tinyllama


SOCKET_STORAGE = ttnn.BufferType.L1
SOCKET_FIFO_SIZE = 10 * 1024
SENDER_CORE_COORD = ttnn.CoreCoord(0, 0)
RECEIVER_CORE_COORD = ttnn.CoreCoord(0, 1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--chat", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=50)
    p.add_argument("--max_total_seq", type=int, default=128,
                   help="MAX_S for the fixed-shape KV cache.")
    p.add_argument("--output_text", default="/tmp/path_d5_text.txt")
    p.add_argument("--summary_json", default="/tmp/path_d5_summary.json")
    return p.parse_args()


def format_chat_prompt(prompt: str) -> str:
    return f"<|user|>\n{prompt}</s>\n<|assistant|>\n"


def next_tile_len(n: int) -> int:
    return ((n + 31) // 32) * 32


def fmt_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def migrate_kv_via_socket_per_layer(
    src_K_list, src_V_list,
    send_socket, recv_socket,
    prefill_sub, decode_sub,
):
    """Same per-layer socket migration as D3."""
    n_layers = len(src_K_list)
    dst_K_list = [
        ttnn.allocate_tensor_on_device(src_K_list[i].spec, decode_sub)
        for i in range(n_layers)
    ]
    dst_V_list = [
        ttnn.allocate_tensor_on_device(src_V_list[i].spec, decode_sub)
        for i in range(n_layers)
    ]
    total_bytes = 0
    t0 = time.perf_counter()
    for i in range(n_layers):
        ttnn.experimental.send_async(src_K_list[i], send_socket)
        ttnn.experimental.recv_async(dst_K_list[i], recv_socket)
        ttnn.experimental.send_async(src_V_list[i], send_socket)
        ttnn.experimental.recv_async(dst_V_list[i], recv_socket)
        shape = src_K_list[i].shape
        n_elem = 1
        for d in shape:
            n_elem *= d
        total_bytes += 2 * n_elem * 2
    ttnn.synchronize_device(prefill_sub)
    ttnn.synchronize_device(decode_sub)
    t_total = time.perf_counter() - t0
    return dst_K_list, dst_V_list, t_total, total_bytes


def main():
    args = parse_args()
    visible = os.environ["TT_VISIBLE_DEVICES"]
    model_dir = os.environ.get("HF_MODEL")
    if model_dir is None:
        print("ERROR: set HF_MODEL=/path/to/TinyLlama-1.1B-Chat-v1.0", file=sys.stderr)
        return 1

    print("=" * 70)
    print("INTEGRATED ETH-disagg with paged_update_cache")
    print("=" * 70)
    print(f"TT_VISIBLE_DEVICES = {visible}")
    print(f"HF_MODEL           = {model_dir}")
    print(f"prompt             = {args.prompt!r}")
    print(f"max_new_tokens     = {args.max_new_tokens}")
    print(f"max_total_seq      = {args.max_total_seq}")

    cfg = TinyLlamaConfig()
    cfg.validate()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    prompt_text = format_chat_prompt(args.prompt) if args.chat else args.prompt
    prompt_ids = tok(prompt_text, return_tensors="pt").input_ids
    real_len = prompt_ids.shape[-1]
    target = next_tile_len(real_len)
    if target > real_len:
        pad = torch.full((1, target - real_len), pad_id, dtype=prompt_ids.dtype)
        prompt_ids = torch.cat([prompt_ids, pad], dim=-1)
    print(f"\nprompt tokens:     real_len={real_len}  padded_len={prompt_ids.shape[-1]}")

    if real_len + args.max_new_tokens > args.max_total_seq:
        print(f"ERROR: max_total_seq ({args.max_total_seq}) < real_len + max_new_tokens "
              f"({real_len + args.max_new_tokens}). Increase --max_total_seq.",
              file=sys.stderr)
        return 1

    print("\nLoading TinyLlama weights to host...")
    t0 = time.perf_counter()
    fmw = load_tinyllama(model_dir, cfg)
    t_load = time.perf_counter() - t0
    print(f"  weight load (host): {t_load:.2f}s")

    print("\nOpening parent 1x2 mesh + carving submeshes...")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    try:
        parent = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(1, 2),
            physical_device_ids=[0, 1],
        )
        try:
            prefill_sub = parent.create_submesh(
                ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 0)
            )
            decode_sub = parent.create_submesh(
                ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 1)
            )
            print(f"  prefill_sub physical chip: "
                  f"{prefill_sub.get_device_id(ttnn.MeshCoordinate([0, 0]))}")
            print(f"  decode_sub  physical chip: "
                  f"{decode_sub.get_device_id(ttnn.MeshCoordinate([0, 0]))}")

            print("\nBuilding socket pair...")
            socket_connections = [
                ttnn.SocketConnection(
                    ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), SENDER_CORE_COORD),
                    ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), RECEIVER_CORE_COORD),
                ),
            ]
            socket_mem_config = ttnn.SocketMemoryConfig(SOCKET_STORAGE, SOCKET_FIFO_SIZE)
            socket_config = ttnn.SocketConfig(socket_connections, socket_mem_config)
            send_socket, recv_socket = ttnn.create_socket_pair(
                prefill_sub, decode_sub, socket_config,
            )
            print("  socket pair created")

            print("\nBuilding prefill model on prefill_sub...")
            t0 = time.perf_counter()
            prefill_model = D5PrefillModel(fmw, prefill_sub, cfg)
            t_p_build = time.perf_counter() - t0
            print(f"  prefill_sub upload: {t_p_build:.2f}s")

            print("\nPrefill on prefill_sub (variable-shape, on-device math)...")
            t0 = time.perf_counter()
            logits, prefill_K, prefill_V = prefill_model.prefill(prompt_ids)
            t_prefill = time.perf_counter() - t0
            print(f"  prefill ({prompt_ids.shape[-1]} tokens): {t_prefill*1000:.1f} ms")

            first_token = logits[0, real_len - 1].argmax().item()
            print(f"  first decode token: {first_token} ({tok.decode([first_token])!r})")

            print(f"\nBuilding decode model on decode_sub (MAX_S={args.max_total_seq})...")
            t0 = time.perf_counter()
            decode_model = D5DecodeModel(fmw, decode_sub, cfg, args.max_total_seq)
            t_d_build = time.perf_counter() - t0
            print(f"  decode_sub upload + cache alloc: {t_d_build:.2f}s")

            print("\nMigrating KV chip-to-chip via socket...")
            dst_K_list, dst_V_list, t_migrate, migrate_bytes = (
                migrate_kv_via_socket_per_layer(
                    prefill_K, prefill_V, send_socket, recv_socket,
                    prefill_sub, decode_sub,
                )
            )
            eff_gbps = migrate_bytes / t_migrate / 1e9 if t_migrate > 0 else 0.0
            print(f"  migrated 44 tensors ({fmt_bytes(migrate_bytes)} bf16) "
                  f"in {t_migrate*1000:.2f} ms  ({eff_gbps:.3f} GB/s)")

            print("Populating fixed-shape cache from migrated tensors...")
            t0 = time.perf_counter()
            decode_model.populate_cache_from_prefill(dst_K_list, dst_V_list, real_len)
            t_populate = time.perf_counter() - t0
            print(f"  cache population: {t_populate*1000:.2f} ms")

            print("\nDecode on decode_sub (paged_update_cache cache write)...")
            generated = list(prompt_ids[0, :real_len].tolist()) + [first_token]
            cur_token = first_token
            print("  ↳ " + tok.decode([cur_token]), end="", flush=True)

            per_token_times = []
            for step in range(args.max_new_tokens - 1):
                t0 = time.perf_counter()
                logits = decode_model.decode_step(cur_token)
                per_token_times.append(time.perf_counter() - t0)
                next_id = logits[0, 0].argmax().item()
                print(tok.decode([next_id]), end="", flush=True)
                generated.append(next_id)
                cur_token = next_id
                if next_id == eos_id:
                    print("  [EOS]", end="", flush=True)
                    break
            print()

            t_decode_total = sum(per_token_times)
            n_new = len(generated) - real_len
            avg_decode_ms = (t_decode_total / max(len(per_token_times), 1)) * 1000
            tps = len(per_token_times) / max(t_decode_total, 1e-9)
            print(f"  decode {n_new} new tokens: {t_decode_total*1000:.1f} ms "
                  f"({avg_decode_ms:.1f} ms/tok = {tps:.2f} tok/s)")

            if len(per_token_times) > 5:
                steady_times = per_token_times[3:]
                steady_avg_ms = sum(steady_times) / len(steady_times) * 1000
                steady_tps = len(steady_times) / sum(steady_times)
                print(f"  steady-state (after 3 warmup tokens): "
                      f"{steady_avg_ms:.1f} ms/tok = {steady_tps:.2f} tok/s")
            else:
                steady_avg_ms = avg_decode_ms
                steady_tps = tps

            full_text = tok.decode(generated, skip_special_tokens=False)
            with open(args.output_text, "w") as f:
                f.write(full_text)

        finally:
            ttnn.close_mesh_device(parent)
    finally:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    per_request = t_prefill + t_migrate + t_populate + t_decode_total

    print()
    print("=" * 70)
    print("Output")
    print("=" * 70)
    print(full_text)

    print()
    print("=" * 70)
    print("Breakdown (submesh + paged_update_cache + on-device ROPE + ETH socket)")
    print("=" * 70)
    print(f"PREFILL on prefill_sub (variable-shape):  {t_prefill*1000:>9.1f} ms")
    print(f"KV migration (44 socket calls):           {t_migrate*1000:>9.2f} ms"
          f"  ({eff_gbps:.3f} GB/s)")
    print(f"Cache population (one-time padding):      {t_populate*1000:>9.2f} ms")
    print(f"DECODE (paged_update_cache + on-dev ROPE):"
          f"{t_decode_total*1000:>9.1f} ms ({avg_decode_ms:.1f} ms/tok = {tps:.2f} tok/s)")
    if len(per_token_times) > 5:
        print(f"  steady-state (after warmup):            "
              f"{steady_avg_ms:.1f} ms/tok = {steady_tps:.2f} tok/s")
    print()
    print(f"Per-request latency (excl. weight load):  {per_request*1000:.1f} ms")
    print()
    print("Comparison:")
    print(f"  (host-SDPA):              ~31 tok/s")
    print(f"  (broadcast+mask, host ROPE):     25.07 tok/s")
    print(f"  (broadcast+mask, on-dev ROPE):   39.51 tok/s")
    print(f"  (paged_update_cache, this run): {steady_tps:>5.2f} tok/s steady-state")

    summary = {
        "args": vars(args),
        "tt_visible_devices": visible,
        "real_len": real_len,
        "padded_len": int(prompt_ids.shape[-1]),
        "max_total_seq": args.max_total_seq,
        "n_new": n_new,
        "first_decode_token": first_token,
        "kv_migration_bytes_bf16": migrate_bytes,
        "t_load_weights_s": t_load,
        "t_prefill_sub_build_s": t_p_build,
        "t_decode_sub_build_s": t_d_build,
        "t_prefill_s": t_prefill,
        "t_migrate_s": t_migrate,
        "t_populate_cache_s": t_populate,
        "t_decode_total_s": t_decode_total,
        "per_token_avg_ms": avg_decode_ms,
        "per_token_steady_ms": steady_avg_ms,
        "tokens_per_s": tps,
        "tokens_per_s_steady": steady_tps,
        "per_request_latency_s": per_request,
        "output_text": full_text,
        "per_token_times_s": per_token_times,
    }
    with open(args.summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull summary JSON: {args.summary_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
