# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""Break down the first-token cost of disaggregated GPT-OSS serving and try cheaper KV hand-offs.

One compile, then the same two-request scenario is replayed once per hand-off mode:
request A is prefilled into slot 0 and decoded, then request B joins in slot 1 while A is
still decoding. Every phase is timed on its own (wire call, submit, wait), and the tokens
of each mode are compared with the upstream behaviour (``every_step``).

modes:
    every_step  wire all KV rows, inputs and RetainedState outputs, on every decode step (upstream example)
    all_once    wire all KV rows, inputs and outputs, only on the step where a request joins
    new_inout   wire only the joining slot's row, inputs and outputs
    new_in      wire only the joining slot's row, inputs only
    prompt_ctx  like new_in, but the buffers cover only the prompt's positions, not the full context

usage (from the efficient-transformers checkout, on a host with the cards):
    python3 examples/disagg_serving/gpt_oss_ttft_breakdown.py --num-hidden-layers 8 --mxfp6
"""

import argparse
import json
from statistics import median
from time import perf_counter

import numpy as np
from transformers import AutoConfig, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.generation._kv_dma_handoff import qaicrt
from QEfficient.generation.cloud_infer import QAICInferenceSession

NUM_CORES = 16
MOE_PREFILL_PACKED_CHUNK_SIZE = 256
TEXT_A = "The quick brown fox jumps over the lazy dog near the quiet river bank. "
TEXT_B = "A small boat drifts slowly across the cold lake while the sun goes down. "
MODES = ("every_step", "all_once", "new_inout", "new_in", "prompt_ctx")


def compile_sessions(qeff_model, args):
    common = dict(
        ctx_len=args.ctx_len,
        full_batch_size=args.full_batch_size,
        num_cores=NUM_CORES,
        mos=1,
        aic_enable_depth_first=True,
        num_speculative_tokens=None,
        split_retained_state_io=True,
        retain_full_kv=True,
        use_onnx_subfunctions=True,
        mxfp6_matmul=args.mxfp6,
        mxint8_kv_cache=args.mxint8_kv,
    )
    decode_kwargs = {"node_precision_info": args.non_subfunc_npi} if args.non_subfunc_npi else {}
    prefill_kwargs = {"node_precision_info": args.subfunc_npi} if args.subfunc_npi else {}
    decode_qpc = qeff_model.compile(
        prefill_seq_len=1, num_devices=args.decode_num_devices, offload_pt_weights=False, **common, **decode_kwargs
    )
    prefill_qpc = qeff_model.compile(
        prefill_seq_len=args.prefill_seq_len,
        num_devices=args.prefill_num_devices,
        mdp_num_partitions=args.stages,
        qaic_config={"moe_config": {"expert_parallel_chunk_size": MOE_PREFILL_PACKED_CHUNK_SIZE}},
        prefill_only=True,
        enable_chunking=True,
        **common,
        **prefill_kwargs,
    )
    session_kwargs = dict(kv_dma_share=True, full_batch_size=args.full_batch_size)
    prefill = QAICInferenceSession(
        prefill_qpc, device_ids=args.prefill_device_ids, stages=args.stages, cluster_id="prefill", **session_kwargs
    )
    decode = QAICInferenceSession(decode_qpc, device_ids=args.decode_device_ids, cluster_id="decode", **session_kwargs)
    return prefill, decode


def build_prompt(tokenizer, text, num_tokens, seq_len):
    """Exactly ``num_tokens`` real tokens, padded to a multiple of ``seq_len``."""
    ids = tokenizer(text * (num_tokens // 8 + 1), add_special_tokens=False)["input_ids"][:num_tokens]
    if len(ids) < num_tokens:
        raise ValueError(f"filler text gave {len(ids)} tokens, need {num_tokens}")
    padded_len = -(num_tokens // -seq_len) * seq_len
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    mask = np.zeros(padded_len, dtype=bool)
    if tokenizer.padding_side == "left":
        mask[padded_len - num_tokens :] = True
    else:
        mask[:num_tokens] = True
    input_ids = np.full((1, padded_len), pad_id, dtype=np.int64)
    input_ids[0, mask] = ids
    position_ids = np.where(mask, np.arange(padded_len), -1)[None, :].astype(np.int64)
    return input_ids, position_ids


def wire(session, exec_idx, buff_map, buffers, slot):
    """Point KV bindings of one execObj at host buffers. Returns the call time in ms."""
    slices = [(binding_index, buf) for (_, binding_index), buf in zip(buff_map, buffers)]
    start = perf_counter()
    status, _ = session.execObj[exec_idx].setDataWithSlices(
        slices, session._kv_dma.kv_slicing_spec_handle, [("batch_index", slot), ("ctx_start", 0)]
    )
    if status != qaicrt.QStatus.QS_SUCCESS:
        raise RuntimeError("setDataWithSlices failed")
    return 1e3 * (perf_counter() - start)


def run_prefill(prefill, prompt, slot, handoff_buffers, seq_len):
    """Chunked prefill of one prompt. Returns (first_token, next_pos, timings in ms)."""
    input_ids, position_ids = prompt
    kv = prefill._kv_dma
    buff_map = kv.decode_rs_full_buff_map if kv.is_hybrid_kv else kv.kv_only_buff_map
    inputs = {"batch_index": np.array([[slot]], dtype=np.int64)}
    num_chunks = input_ids.shape[1] // seq_len
    times = {"chunk_ms": [], "prefill_wire_ms": 0.0}
    exec_idx = None
    for i in range(num_chunks):
        inputs["input_ids"] = np.ascontiguousarray(input_ids[:, i * seq_len : (i + 1) * seq_len])
        inputs["position_ids"] = np.ascontiguousarray(position_ids[:, i * seq_len : (i + 1) * seq_len])
        start = perf_counter()
        exec_idx = kv.prefill_available_exec_objs.get()
        try:
            if i == num_chunks - 1:
                times["prefill_wire_ms"] = wire(prefill, exec_idx, buff_map, handoff_buffers, slot)
            if prefill.execObj[exec_idx].setData(kv._tuple_list_with_outputs(inputs, exec_idx)) != (
                qaicrt.QStatus.QS_SUCCESS
            ):
                raise RuntimeError("prefill setData failed")
            if prefill.queue.enqueue(prefill.execObj[exec_idx]) != qaicrt.QStatus.QS_SUCCESS:
                raise RuntimeError("prefill enqueue failed")
        except Exception:
            kv.prefill_available_exec_objs.put(exec_idx)
            raise
        prefill.complete_inf(exec_idx, is_prefill=True)
        times["chunk_ms"].append(1e3 * (perf_counter() - start))
    token = int(np.argmax(prefill.get_outputs(index=exec_idx)["logits"]))
    return token, int(position_ids.max()) + 1, times


def decode_wiring(decode, mode, kv_caches, slot, small):
    """Buffers and binding map to wire on the decode side when ``slot`` joins."""
    inputs_map, outputs_map = decode.decode_buff_map, decode.decode_rs_kv_only_buff_map
    if mode in ("every_step", "all_once"):
        return inputs_map + outputs_map, kv_caches + kv_caches, 0
    rows = [kv[slot : slot + 1] for kv in kv_caches]
    if mode == "new_inout":
        return inputs_map + outputs_map, rows + rows, slot
    if mode == "new_in":
        return inputs_map, rows, slot
    return inputs_map, small, slot


def decode_step(decode, state, wiring=None):
    """One decode step for every active slot. Returns (wire_ms, run_ms)."""
    n = len(state["active"])
    inputs = {name: np.full((n, 1), -1, dtype=np.int64) for name in ("input_ids", "position_ids", "batch_index")}
    for slot in range(n):
        if state["active"][slot]:
            inputs["input_ids"][slot, 0] = state["token"][slot]
            inputs["position_ids"][slot, 0] = state["pos"][slot]
            inputs["batch_index"][slot, 0] = slot
    wire_ms = wire(decode, decode.decode_execObj_idx, *wiring) if wiring else 0.0
    start = perf_counter()
    exec_idx = decode.np_run(inputs, is_prefill=False)
    decode.complete_inf(exec_idx, is_prefill=False)
    logits = decode.get_outputs(index=exec_idx)["logits"]
    run_ms = 1e3 * (perf_counter() - start)
    next_tokens = np.argmax(logits.reshape(n, -1, logits.shape[-1])[:, -1, :], axis=-1)
    for slot in range(n):
        if state["active"][slot]:
            state["token"][slot] = int(next_tokens[slot])
            state["pos"][slot] += 1
            state["out"][slot].append(state["token"][slot])
    return wire_ms, run_ms


def scenario(prefill, decode, kv_caches, prompts, mode, args):
    """A alone in slot 0 for ``--steps`` steps, then B joins in slot 1 for ``--steps`` more."""
    n = args.full_batch_size
    state = {"active": [False] * n, "token": [0] * n, "pos": [0] * n, "out": [[] for _ in range(n)]}
    result = {}
    for slot, prompt in enumerate(prompts):
        small = None
        handoff = [kv[slot : slot + 1] for kv in kv_caches]
        if mode == "prompt_ctx":
            small = [np.zeros((1, kv.shape[1], prompt[0].shape[1], kv.shape[3]), dtype=kv.dtype) for kv in kv_caches]
            handoff = small
        token, pos, times = run_prefill(prefill, prompt, slot, handoff, args.prefill_seq_len)
        state["active"][slot], state["token"][slot], state["pos"][slot] = True, token, pos
        state["out"][slot].append(token)
        wiring = decode_wiring(decode, mode, kv_caches, slot, small)
        steps = [decode_step(decode, state, wiring)]
        for _ in range(args.steps - 1):
            steps.append(decode_step(decode, state, wiring if mode == "every_step" else None))
        tag = "a" if slot == 0 else "b"
        result[f"{tag}_prefill_ms"] = sum(times["chunk_ms"])
        result[f"{tag}_prefill_wire_ms"] = times["prefill_wire_ms"]
        result[f"{tag}_decode_wire_ms"] = steps[0][0]
        result[f"{tag}_first_step_ms"] = steps[0][1]
        result[f"{tag}_steady_step_ms"] = median(wire_ms + run_ms for wire_ms, run_ms in steps[1:])
    result["tokens_a"], result["tokens_b"] = state["out"][0], state["out"][1]
    return result


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    config = AutoConfig.from_pretrained(args.model_id)
    full_layers = config.num_hidden_layers
    if args.num_hidden_layers:
        config.num_hidden_layers = args.num_hidden_layers
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(args.model_id, continuous_batching=True, config=config)
    prefill, decode = compile_sessions(qeff_model, args)
    kv_caches = [np.zeros(shape, dtype=dtype) for (shape, dtype) in decode.kv_cache_info]
    kv_gb_per_slot = sum(kv[0:1].nbytes for kv in kv_caches) / 1e9
    prompts = [
        build_prompt(tokenizer, TEXT_A, args.prompt_tokens, args.prefill_seq_len),
        build_prompt(tokenizer, TEXT_B, args.prompt_tokens, args.prefill_seq_len),
    ]

    cols = ["a_prefill_ms", "a_prefill_wire_ms", "a_decode_wire_ms", "a_first_step_ms", "a_steady_step_ms",
            "b_decode_wire_ms", "b_first_step_ms", "b_steady_step_ms"]  # fmt: skip
    print(f"\nctx_len={args.ctx_len} slots={args.full_batch_size} prompt={args.prompt_tokens} tokens "
          f"kv_gb_per_slot={kv_gb_per_slot:.3f}")  # fmt: skip
    print(f"layers compiled: {config.num_hidden_layers} of {full_layers}")
    print("b_first_step_ms is also the stall request A sees when request B joins.")
    print(f"{'mode':>11}  " + "  ".join(f"{c:>18}" for c in cols) + "  same_tokens_a  same_tokens_b", flush=True)

    # Each mode is printed and saved as soon as it finishes, so a crash in a later mode loses nothing.
    rows = []
    reference = None
    for mode in args.modes:
        row = {"mode": mode}
        try:
            samples = [scenario(prefill, decode, kv_caches, prompts, mode, args) for _ in range(args.repeats + 1)][1:]
            for key in samples[0]:
                row[key] = samples[-1][key] if key.startswith("tokens") else median(s[key] for s in samples)
        except Exception as exc:  # report the failing mode and keep going with the others
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        if "error" in row:
            print(f"{mode:>11}  FAILED: {row['error']}", flush=True)
        else:
            if mode == "every_step":
                reference = row
            same = ["-", "-"]
            if reference is not None:
                same = [str(row[k] == reference[k]) for k in ("tokens_a", "tokens_b")]
            timings = "  ".join(f"{row[c]:>18.2f}" for c in cols)
            print(f"{mode:>11}  {timings}  {same[0]:>13}  {same[1]:>13}", flush=True)
        if args.out:
            with open(args.out, "w") as f:
                json.dump({"args": vars(args), "kv_gb_per_slot": kv_gb_per_slot, "rows": rows}, f, indent=2)
    if args.out:
        print(f"wrote {args.out}")
    prefill.deactivate()
    decode.deactivate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-id", default="openai/gpt-oss-20b")
    parser.add_argument("--ctx-len", type=int, default=32768)
    parser.add_argument("--prefill-seq-len", type=int, default=512)
    parser.add_argument("--prompt-tokens", type=int, default=100)
    parser.add_argument("--num-hidden-layers", type=int, default=None, help="reduce depth for fast compiles")
    parser.add_argument("--full-batch-size", type=int, default=4, help="decode slots (KV rows), at least 2")
    parser.add_argument("--stages", type=int, default=4, help="prefill pipeline depth (mdp_num_partitions)")
    parser.add_argument("--prefill-num-devices", type=int, default=8)
    parser.add_argument("--decode-num-devices", type=int, default=4)
    parser.add_argument("--prefill-device-ids", type=int, nargs="+", default=None)
    parser.add_argument("--decode-device-ids", type=int, nargs="+", default=None)
    parser.add_argument("--mxfp6", action="store_true", help="compile with mxfp6_matmul")
    parser.add_argument("--mxint8-kv", action="store_true", help="compile with mxint8_kv_cache")
    parser.add_argument("--subfunc-npi", default=None, help="prefill NPI yaml (120B)")
    parser.add_argument("--non-subfunc-npi", default=None, help="decode NPI yaml (120B)")
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--steps", type=int, default=8, help="decode steps before and after request B joins")
    parser.add_argument("--out", default="gpt_oss_ttft_breakdown.json")
    main(parser.parse_args())
