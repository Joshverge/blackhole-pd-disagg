#!/usr/bin/env python3
"""
d5_pd_full.py - Integrated PD disagg with paged_update_cache.

This version supports request-level pipelining across multiple prompts:
the prefill chip can prefill prompt N+1 while the decode chip decodes
prompt N. Decode is still single-request at a time; this is pipeline
batching, not true tensor batched decode.

Run:
    HF_MODEL=$HOME/models/TinyLlama-1.1B-Chat-v1.0 \
        python d5_pd_full.py --chat \
            --prompt "What is your favorite condiment?" \
            --prompt "Write one sentence about Tenstorrent." \
            --max_new_tokens 100
"""

import argparse
import json
import os
import queue
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from typing import List, Optional

os.environ.setdefault("TT_VISIBLE_DEVICES", "2,3")

import torch
import ttnn

from config import LlamaConfig
from d5_model import D5PrefillModel, D5DecodeModel
from weight_loader import load_llama


SOCKET_STORAGE = ttnn.BufferType.L1
SOCKET_FIFO_SIZE = 10 * 1024
SENDER_CORE_COORD = ttnn.CoreCoord(0, 0)
RECEIVER_CORE_COORD = ttnn.CoreCoord(0, 1)


@dataclass
class PromptRequest:
    index: int
    raw_prompt: str
    prompt_text: str
    prompt_ids: torch.Tensor
    real_len: int
    padded_len: int


@dataclass
class DecodeReadyRequest:
    request: PromptRequest
    dst_K_list: List[ttnn.Tensor]
    dst_V_list: List[ttnn.Tensor]
    first_token: int
    t_prefill_s: float
    t_migrate_s: float
    migrate_bytes: int
    prefill_started_s: float
    prefill_finished_s: float
    migrate_started_s: float
    migrate_finished_s: float


@dataclass
class RequestResult:
    request_index: int
    prompt: str
    prompt_text: str
    real_len: int
    padded_len: int
    first_decode_token: int
    generated_ids: List[int]
    output_text: str
    n_new: int
    eos_reached: bool
    kv_migration_bytes_bf16: int
    t_prefill_s: float
    t_migrate_s: float
    t_populate_cache_s: float
    t_decode_total_s: float
    per_token_avg_ms: float
    per_token_steady_ms: float
    tokens_per_s: float
    tokens_per_s_steady: float
    per_request_latency_s: float
    per_token_times_s: List[float]
    prefill_started_s: float
    prefill_finished_s: float
    migrate_started_s: float
    migrate_finished_s: float
    decode_started_s: float
    decode_finished_s: float


class PipelineWorkerError(Exception):
    """Wrap an exception raised in a worker thread with its traceback."""

    def __init__(self, worker_name: str, exc: BaseException):
        self.worker_name = worker_name
        self.exc = exc
        self.trace = traceback.format_exc()
        super().__init__(f"{worker_name} failed: {exc}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", action="append", required=True,
                   help="Input prompt. Repeat this flag for pipeline batching.")
    p.add_argument("--chat", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=50)
    p.add_argument("--max_total_seq", type=int, default=128,
                   help="MAX_S for the fixed-shape KV cache.")
    p.add_argument("--output_text", default="/tmp/path_d5_text.txt")
    p.add_argument("--summary_json", default="/tmp/path_d5_summary.json")
    return p.parse_args()


def format_chat_prompt(prompt: str, tok) -> str:
    """Build a chat prompt from the tokenizer template when available."""
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"<|user|>\n{prompt}</s>\n<|assistant|>\n"


def next_tile_len(n: int) -> int:
    return ((n + 31) // 32) * 32


def fmt_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def deallocate_tensors(tensors):
    for tensor in tensors:
        ttnn.deallocate(tensor)


def prepare_request(index: int, raw_prompt: str, tok, pad_id: int,
                    chat: bool, max_new_tokens: int,
                    max_total_seq: int) -> PromptRequest:
    prompt_text = format_chat_prompt(raw_prompt, tok) if chat else raw_prompt
    prompt_ids = tok(prompt_text, return_tensors="pt").input_ids
    real_len = prompt_ids.shape[-1]
    target = next_tile_len(real_len)
    if target > real_len:
        pad = torch.full((1, target - real_len), pad_id, dtype=prompt_ids.dtype)
        prompt_ids = torch.cat([prompt_ids, pad], dim=-1)

    if real_len + max_new_tokens > max_total_seq:
        raise ValueError(
            f"request {index}: max_total_seq ({max_total_seq}) < "
            f"real_len + max_new_tokens ({real_len + max_new_tokens}); "
            "increase --max_total_seq"
        )

    return PromptRequest(
        index=index,
        raw_prompt=raw_prompt,
        prompt_text=prompt_text,
        prompt_ids=prompt_ids,
        real_len=real_len,
        padded_len=int(prompt_ids.shape[-1]),
    )


def migrate_kv_via_socket_per_layer(
    src_K_list, src_V_list,
    send_socket, recv_socket,
    prefill_sub, decode_sub,
):
    """Per-layer socket migration from prefill_sub to decode_sub."""
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


def run_prefill(request: PromptRequest, prefill_model, tok) -> tuple:
    print(f"\n[request {request.index}] Prefill on prefill_sub "
          f"(real_len={request.real_len}, padded_len={request.padded_len})...")
    t0 = time.perf_counter()
    logits, prefill_K, prefill_V = prefill_model.prefill(request.prompt_ids)
    t_prefill = time.perf_counter() - t0
    first_token = logits[0, request.real_len - 1].argmax().item()
    print(f"[request {request.index}] prefill: {t_prefill*1000:.1f} ms; "
          f"first token {first_token} ({tok.decode([first_token])!r})")
    return prefill_K, prefill_V, first_token, t_prefill


def run_decode_request(
    ready: DecodeReadyRequest,
    decode_model,
    tok,
    eos_id: Optional[int],
    max_new_tokens: int,
) -> RequestResult:
    request = ready.request
    print(f"\n[request {request.index}] Populating fixed-shape cache...")
    t0 = time.perf_counter()
    decode_model.populate_cache_from_prefill(
        ready.dst_K_list, ready.dst_V_list, request.real_len
    )
    t_populate = time.perf_counter() - t0
    deallocate_tensors(ready.dst_K_list + ready.dst_V_list)
    print(f"[request {request.index}] cache population: {t_populate*1000:.2f} ms")

    print(f"[request {request.index}] Decode on decode_sub...")
    decode_started_s = time.perf_counter()
    generated = list(request.prompt_ids[0, :request.real_len].tolist())
    generated.append(ready.first_token)
    cur_token = ready.first_token
    print(f"[request {request.index}] -> " + tok.decode([cur_token]), end="", flush=True)

    eos_reached = False
    per_token_times = []
    for _ in range(max_new_tokens - 1):
        t0 = time.perf_counter()
        logits = decode_model.decode_step(cur_token)
        per_token_times.append(time.perf_counter() - t0)
        next_id = logits[0, 0].argmax().item()
        print(tok.decode([next_id]), end="", flush=True)
        generated.append(next_id)
        cur_token = next_id
        if eos_id is not None and next_id == eos_id:
            eos_reached = True
            print("  [EOS]", end="", flush=True)
            break
    print()
    decode_finished_s = time.perf_counter()

    t_decode_total = sum(per_token_times)
    n_new = len(generated) - request.real_len
    avg_decode_ms = (t_decode_total / max(len(per_token_times), 1)) * 1000
    tps = len(per_token_times) / max(t_decode_total, 1e-9)
    if len(per_token_times) > 5:
        steady_times = per_token_times[3:]
        steady_avg_ms = sum(steady_times) / len(steady_times) * 1000
        steady_tps = len(steady_times) / sum(steady_times)
    else:
        steady_avg_ms = avg_decode_ms
        steady_tps = tps

    print(f"[request {request.index}] decode {n_new} new tokens: "
          f"{t_decode_total*1000:.1f} ms "
          f"({avg_decode_ms:.1f} ms/tok = {tps:.2f} tok/s)")

    full_text = tok.decode(generated, skip_special_tokens=False)
    per_request_latency = (
        ready.t_prefill_s + ready.t_migrate_s + t_populate + t_decode_total
    )

    return RequestResult(
        request_index=request.index,
        prompt=request.raw_prompt,
        prompt_text=request.prompt_text,
        real_len=request.real_len,
        padded_len=request.padded_len,
        first_decode_token=ready.first_token,
        generated_ids=generated,
        output_text=full_text,
        n_new=n_new,
        eos_reached=eos_reached,
        kv_migration_bytes_bf16=ready.migrate_bytes,
        t_prefill_s=ready.t_prefill_s,
        t_migrate_s=ready.t_migrate_s,
        t_populate_cache_s=t_populate,
        t_decode_total_s=t_decode_total,
        per_token_avg_ms=avg_decode_ms,
        per_token_steady_ms=steady_avg_ms,
        tokens_per_s=tps,
        tokens_per_s_steady=steady_tps,
        per_request_latency_s=per_request_latency,
        per_token_times_s=per_token_times,
        prefill_started_s=ready.prefill_started_s,
        prefill_finished_s=ready.prefill_finished_s,
        migrate_started_s=ready.migrate_started_s,
        migrate_finished_s=ready.migrate_finished_s,
        decode_started_s=decode_started_s,
        decode_finished_s=decode_finished_s,
    )


def run_pipeline(
    requests: List[PromptRequest],
    prefill_model,
    decode_model,
    tok,
    eos_id: Optional[int],
    max_new_tokens: int,
    send_socket,
    recv_socket,
    prefill_sub,
    decode_sub,
) -> List[RequestResult]:
    ready_q = queue.Queue(maxsize=1)
    decode_lock = threading.Lock()
    stop_event = threading.Event()
    results: List[Optional[RequestResult]] = [None] * len(requests)
    finished = object()

    def put_pipeline_item(item) -> bool:
        while not stop_event.is_set():
            try:
                ready_q.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def prefill_worker():
        try:
            for request in requests:
                if stop_event.is_set():
                    break
                prefill_started_s = time.perf_counter()
                prefill_K, prefill_V, first_token, t_prefill = run_prefill(
                    request, prefill_model, tok
                )
                prefill_finished_s = time.perf_counter()

                print(f"[request {request.index}] Waiting for decode_sub to migrate KV...")
                with decode_lock:
                    migrate_started_s = time.perf_counter()
                    dst_K, dst_V, t_migrate, migrate_bytes = (
                        migrate_kv_via_socket_per_layer(
                            prefill_K, prefill_V, send_socket, recv_socket,
                            prefill_sub, decode_sub,
                        )
                    )
                    migrate_finished_s = time.perf_counter()

                eff_gbps = migrate_bytes / t_migrate / 1e9 if t_migrate > 0 else 0.0
                print(f"[request {request.index}] migrated "
                      f"{len(dst_K) * 2} tensors ({fmt_bytes(migrate_bytes)} bf16) "
                      f"in {t_migrate*1000:.2f} ms ({eff_gbps:.3f} GB/s)")
                deallocate_tensors(prefill_K + prefill_V)

                if not put_pipeline_item(DecodeReadyRequest(
                    request=request,
                    dst_K_list=dst_K,
                    dst_V_list=dst_V,
                    first_token=first_token,
                    t_prefill_s=t_prefill,
                    t_migrate_s=t_migrate,
                    migrate_bytes=migrate_bytes,
                    prefill_started_s=prefill_started_s,
                    prefill_finished_s=prefill_finished_s,
                    migrate_started_s=migrate_started_s,
                    migrate_finished_s=migrate_finished_s,
                )):
                    break
                del prefill_K, prefill_V
            put_pipeline_item(finished)
        except BaseException as exc:
            put_pipeline_item(PipelineWorkerError("prefill worker", exc))
            put_pipeline_item(finished)

    def decode_worker():
        while True:
            item = ready_q.get()
            if item is finished:
                break
            if isinstance(item, PipelineWorkerError):
                raise item
            try:
                with decode_lock:
                    result = run_decode_request(
                        item, decode_model, tok, eos_id, max_new_tokens
                    )
                results[result.request_index] = result
            except BaseException as exc:
                raise PipelineWorkerError("decode worker", exc) from exc

    worker_exc: List[BaseException] = []

    def decode_worker_guarded():
        try:
            decode_worker()
        except BaseException as exc:
            worker_exc.append(exc)
            stop_event.set()

    t_prefill = threading.Thread(target=prefill_worker, name="prefill-migrate")
    t_decode = threading.Thread(target=decode_worker_guarded, name="decode")
    t_decode.start()
    t_prefill.start()
    t_prefill.join()
    t_decode.join()

    if worker_exc:
        raise worker_exc[0]
    missing = [i for i, result in enumerate(results) if result is None]
    if missing:
        raise RuntimeError(f"pipeline finished without results for requests {missing}")
    return [result for result in results if result is not None]


def result_to_dict(result: RequestResult) -> dict:
    return {
        "request_index": result.request_index,
        "prompt": result.prompt,
        "prompt_text": result.prompt_text,
        "real_len": result.real_len,
        "padded_len": result.padded_len,
        "first_decode_token": result.first_decode_token,
        "generated_ids": result.generated_ids,
        "output_text": result.output_text,
        "n_new": result.n_new,
        "eos_reached": result.eos_reached,
        "kv_migration_bytes_bf16": result.kv_migration_bytes_bf16,
        "t_prefill_s": result.t_prefill_s,
        "t_migrate_s": result.t_migrate_s,
        "t_populate_cache_s": result.t_populate_cache_s,
        "t_decode_total_s": result.t_decode_total_s,
        "per_token_avg_ms": result.per_token_avg_ms,
        "per_token_steady_ms": result.per_token_steady_ms,
        "tokens_per_s": result.tokens_per_s,
        "tokens_per_s_steady": result.tokens_per_s_steady,
        "per_request_latency_s": result.per_request_latency_s,
        "per_token_times_s": result.per_token_times_s,
        "prefill_started_s": result.prefill_started_s,
        "prefill_finished_s": result.prefill_finished_s,
        "migrate_started_s": result.migrate_started_s,
        "migrate_finished_s": result.migrate_finished_s,
        "decode_started_s": result.decode_started_s,
        "decode_finished_s": result.decode_finished_s,
    }


def write_summary(args, visible: str, cfg: LlamaConfig, results: List[RequestResult],
                  t_load: float, t_p_build: float, t_d_build: float,
                  pipeline_wall_s: float):
    total_new_tokens = sum(r.n_new for r in results)
    total_decode_steps = sum(len(r.per_token_times_s) for r in results)
    total_decode_s = sum(r.t_decode_total_s for r in results)
    aggregate_tps = total_decode_steps / max(total_decode_s, 1e-9)
    pipeline_tps = total_new_tokens / max(pipeline_wall_s, 1e-9)

    summary = {
        "args": vars(args),
        "tt_visible_devices": visible,
        "model": cfg.summary(),
        "num_requests": len(results),
        "max_total_seq": args.max_total_seq,
        "max_new_tokens": args.max_new_tokens,
        "t_load_weights_s": t_load,
        "t_prefill_sub_build_s": t_p_build,
        "t_decode_sub_build_s": t_d_build,
        "pipeline_wall_s": pipeline_wall_s,
        "total_new_tokens": total_new_tokens,
        "decode_step_tokens_per_s": aggregate_tps,
        "pipeline_new_tokens_per_s": pipeline_tps,
        "requests": [result_to_dict(r) for r in results],
    }
    with open(args.summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def print_final_report(results: List[RequestResult], pipeline_wall_s: float):
    print()
    print("=" * 70)
    print("Outputs")
    print("=" * 70)
    for result in results:
        print(f"\n[request {result.request_index}] {result.output_text}")

    print()
    print("=" * 70)
    print("Pipeline Breakdown")
    print("=" * 70)
    for result in results:
        eff_gbps = (
            result.kv_migration_bytes_bf16 / result.t_migrate_s / 1e9
            if result.t_migrate_s > 0 else 0.0
        )
        print(f"[request {result.request_index}] prefill: "
              f"{result.t_prefill_s*1000:>9.1f} ms")
        print(f"[request {result.request_index}] KV migration: "
              f"{result.t_migrate_s*1000:>9.2f} ms "
              f"({eff_gbps:.3f} GB/s)")
        print(f"[request {result.request_index}] cache population: "
              f"{result.t_populate_cache_s*1000:>9.2f} ms")
        print(f"[request {result.request_index}] decode: "
              f"{result.t_decode_total_s*1000:>9.1f} ms "
              f"({result.per_token_avg_ms:.1f} ms/tok = "
              f"{result.tokens_per_s:.2f} tok/s)")
        print(f"[request {result.request_index}] per-request latency excl. load: "
              f"{result.per_request_latency_s*1000:.1f} ms")
    print()
    print(f"Pipeline wall time excl. weight/model build: {pipeline_wall_s*1000:.1f} ms")


def main():
    args = parse_args()
    visible = os.environ["TT_VISIBLE_DEVICES"]
    model_dir = os.environ.get("HF_MODEL")
    if model_dir is None:
        print("ERROR: set HF_MODEL=/path/to/Llama-format-model", file=sys.stderr)
        return 1

    print("=" * 70)
    print("PIPELINED ETH-disagg with paged_update_cache")
    print("=" * 70)
    print(f"TT_VISIBLE_DEVICES = {visible}")
    print(f"HF_MODEL           = {model_dir}")
    print(f"num_prompts        = {len(args.prompt)}")
    print(f"max_new_tokens     = {args.max_new_tokens}")
    print(f"max_total_seq      = {args.max_total_seq}")

    cfg = LlamaConfig.from_hf_dir(model_dir)
    print(f"\nmodel: {cfg.summary()}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    try:
        requests = [
            prepare_request(
                i, prompt, tok, pad_id, args.chat,
                args.max_new_tokens, args.max_total_seq,
            )
            for i, prompt in enumerate(args.prompt)
        ]
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("\nPrepared prompts:")
    for request in requests:
        print(f"  request {request.index}: real_len={request.real_len} "
              f"padded_len={request.padded_len} prompt={request.raw_prompt!r}")

    print(f"\nLoading {cfg.model_name} weights to host...")
    t0 = time.perf_counter()
    fmw = load_llama(model_dir, cfg)
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

            print(f"\nBuilding decode model on decode_sub "
                  f"(MAX_S={args.max_total_seq})...")
            t0 = time.perf_counter()
            decode_model = D5DecodeModel(fmw, decode_sub, cfg, args.max_total_seq)
            t_d_build = time.perf_counter() - t0
            print(f"  decode_sub upload + cache alloc: {t_d_build:.2f}s")

            print("\nStarting request pipeline...")
            pipeline_t0 = time.perf_counter()
            results = run_pipeline(
                requests=requests,
                prefill_model=prefill_model,
                decode_model=decode_model,
                tok=tok,
                eos_id=eos_id,
                max_new_tokens=args.max_new_tokens,
                send_socket=send_socket,
                recv_socket=recv_socket,
                prefill_sub=prefill_sub,
                decode_sub=decode_sub,
            )
            pipeline_wall_s = time.perf_counter() - pipeline_t0

            if len(results) == 1:
                with open(args.output_text, "w") as f:
                    f.write(results[0].output_text)

        finally:
            ttnn.close_mesh_device(parent)
    finally:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    print_final_report(results, pipeline_wall_s)
    write_summary(
        args=args,
        visible=visible,
        cfg=cfg,
        results=results,
        t_load=t_load,
        t_p_build=t_p_build,
        t_d_build=t_d_build,
        pipeline_wall_s=pipeline_wall_s,
    )
    print(f"\nFull summary JSON: {args.summary_json}")
    if len(results) == 1:
        print(f"Single-request output text: {args.output_text}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
