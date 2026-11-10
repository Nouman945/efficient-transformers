"""Where does the fixed first-token cost go in disaggregated GPT-OSS serving?

Six runs: two compiled context lengths x three prompt lengths (1, 2 and 3 chunks). The context length
changes only the size of the prefill -> decode KV hand-off; the prompt length
changes only the number of prefill chunks. Timers sit around every prefill chunk,
the last chunk (which carries the hand-off) and the first decode steps.

Based on examples/disagg_serving/gpt_oss_disagg_mode_cb_chunking_with_kv_share.py.

usage (run from the efficient-transformers checkout, on a host with the cards):
    python3 examples/disagg_serving/gpt_oss_ttft_handoff_test.py --num-hidden-layers 4
    python3 examples/disagg_serving/gpt_oss_ttft_handoff_test.py --model-id openai/gpt-oss-120b --num-hidden-layers 4 \
        --subfunc-npi examples/disagg_serving/subfunction_120b_npi.yaml \
        --non-subfunc-npi examples/disagg_serving/non_subfunction_120b_npi.yaml
"""

import argparse
import json
from statistics import median
from time import perf_counter

import numpy as np
from transformers import AutoConfig, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession

NUM_CORES = 16
MOE_PREFILL_PACKED_CHUNK_SIZE = 256
FILLER_TEXT = "The quick brown fox jumps over the lazy dog near the quiet river bank. "


def compile_sessions(qeff_model, args, ctx_len):
    """Compile the decode and prefill QPCs for one context length and open both sessions."""
    common = dict(
        ctx_len=ctx_len,
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
        prefill_seq_len=1,
        num_devices=args.decode_num_devices,
        offload_pt_weights=False,
        **common,
        **decode_kwargs,
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
    prefill = QAICInferenceSession(
        prefill_qpc,
        device_ids=args.prefill_device_ids,
        kv_dma_share=True,
        stages=args.stages,
        full_batch_size=args.full_batch_size,
        cluster_id="prefill",
    )
    decode = QAICInferenceSession(
        decode_qpc,
        device_ids=args.decode_device_ids,
        kv_dma_share=True,
        full_batch_size=args.full_batch_size,
        cluster_id="decode",
    )
    return prefill, decode


def build_prompt(tokenizer, num_tokens, prefill_seq_len):
    """Exactly ``num_tokens`` real tokens, padded to a multiple of ``prefill_seq_len``."""
    ids = tokenizer(FILLER_TEXT * (num_tokens // 8 + 1), add_special_tokens=False)["input_ids"][:num_tokens]
    if len(ids) < num_tokens:
        raise ValueError(f"filler text gave {len(ids)} tokens, need {num_tokens}")
    num_chunks = -(num_tokens // -prefill_seq_len)
    padded_len = num_chunks * prefill_seq_len
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    mask = np.zeros(padded_len, dtype=bool)
    if tokenizer.padding_side == "left":
        mask[padded_len - num_tokens :] = True
    else:
        mask[:num_tokens] = True
    input_ids = np.full((1, padded_len), pad_id, dtype=np.int64)
    input_ids[0, mask] = ids
    position_ids = np.where(mask, np.arange(padded_len), -1)[None, :].astype(np.int64)
    return input_ids, position_ids, num_chunks


def time_request(prefill, decode, kv_caches, decode_kv_map, prompt, args):
    """Run one request in slot 0. Returns seconds per prefill chunk and per decode step."""
    input_ids, position_ids, num_chunks = prompt
    seq = args.prefill_seq_len
    slot_view = [kv[0:1] for kv in kv_caches]
    chunk_inputs = {"batch_index": np.array([[0]], dtype=np.int64)}
    chunk_s = []
    exec_idx = None
    for i in range(num_chunks):
        chunk_inputs["input_ids"] = input_ids[:, i * seq : (i + 1) * seq]
        chunk_inputs["position_ids"] = position_ids[:, i * seq : (i + 1) * seq]
        last = i == num_chunks - 1
        start = perf_counter()
        exec_idx = prefill.np_run_pipeline(chunk_inputs, last_chunk=last, kv_cache_buffers=slot_view if last else None)
        prefill.complete_inf(exec_idx, is_prefill=True)
        chunk_s.append(perf_counter() - start)
    token = int(np.argmax(prefill.get_outputs(index=exec_idx)["logits"]))
    pos = int(position_ids.max()) + 1

    n = args.full_batch_size
    decode_s = []
    tokens = [token]
    for step in range(args.decode_steps):
        inputs = {name: np.full((n, 1), -1, dtype=np.int64) for name in ("input_ids", "position_ids", "batch_index")}
        inputs["input_ids"][0, 0], inputs["position_ids"][0, 0], inputs["batch_index"][0, 0] = token, pos, 0
        start = perf_counter()
        if step == 0 or not args.handoff_once:
            decode.set_data_for_kv_handoff(
                kv_caches + kv_caches,
                [("batch_index", 0), ("ctx_start", 0)],
                index=decode.decode_execObj_idx,
                buff_map=decode_kv_map,
            )
        exec_idx = decode.np_run(inputs, is_prefill=False)
        decode.complete_inf(exec_idx, is_prefill=False)
        logits = decode.get_outputs(index=exec_idx)["logits"]
        decode_s.append(perf_counter() - start)
        token = int(np.argmax(logits.reshape(n, -1, logits.shape[-1])[0, -1]))
        tokens.append(token)
        pos += 1
    return chunk_s, decode_s, tokens


def summarise(samples):
    """Median over repeats of the pieces we care about, in milliseconds."""
    chunks = [s[0] for s in samples]
    decodes = [s[1] for s in samples]
    other = [c for run in chunks for c in run[:-1]]
    return {
        "other_chunk_ms": 1e3 * median(other) if other else None,
        "last_chunk_ms": 1e3 * median(run[-1] for run in chunks),
        "first_token_ms": 1e3 * median(sum(run) for run in chunks),
        "first_decode_ms": 1e3 * median(run[0] for run in decodes),
        "steady_decode_ms": 1e3 * median(d for run in decodes for d in run[1:]),
    }


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    config = AutoConfig.from_pretrained(args.model_id)
    full_layers = config.num_hidden_layers
    if args.num_hidden_layers:
        config.num_hidden_layers = args.num_hidden_layers
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(args.model_id, continuous_batching=True, config=config)

    prompt_lens = [i * args.prefill_seq_len + args.short_prompt for i in range(3)]
    rows = []
    for ctx_len in args.ctx_lens:
        prefill, decode = compile_sessions(qeff_model, args, ctx_len)
        kv_caches = [np.zeros(shape, dtype=dtype) for (shape, dtype) in decode.kv_cache_info]
        decode_kv_map = decode.decode_buff_map + decode.decode_rs_kv_only_buff_map
        kv_gb = sum(kv[0:1].nbytes for kv in kv_caches) / 1e9
        for prompt_len in prompt_lens:
            prompt = build_prompt(tokenizer, prompt_len, args.prefill_seq_len)
            samples = [
                time_request(prefill, decode, kv_caches, decode_kv_map, prompt, args) for _ in range(args.repeats + 1)
            ][1:]  # first request is warm-up
            row = {"ctx_len": ctx_len, "prompt_tokens": prompt_len, "chunks": prompt[2], "kv_gb_per_request": kv_gb}
            row.update(summarise(samples))
            row["tokens"] = samples[-1][2]
            rows.append(row)
        prefill.deactivate()
        decode.deactivate()

    cols = ["ctx_len", "prompt_tokens", "chunks", "kv_gb_per_request", "other_chunk_ms", "last_chunk_ms",
            "first_token_ms", "first_decode_ms", "steady_decode_ms"]  # fmt: skip
    print("\n" + "  ".join(f"{c:>17}" for c in cols))
    for row in rows:
        cells = (row[c] for c in cols)
        print("  ".join(f"{'-' if v is None else round(v, 3) if isinstance(v, float) else v:>17}" for v in cells))

    layers = config.num_hidden_layers
    print(f"\nlayers compiled: {layers} of {full_layers}. KV size and hand-off time scale with layers "
          f"(x{full_layers / layers:.1f} for the full model).")  # fmt: skip
    by_key = {(r["ctx_len"], r["chunks"]): r for r in rows}
    small, large = min(args.ctx_lens), max(args.ctx_lens)
    if small != large:
        last = by_key[(large, 1)]["last_chunk_ms"] - by_key[(small, 1)]["last_chunk_ms"]
        first = by_key[(large, 1)]["first_decode_ms"] - by_key[(small, 1)]["first_decode_ms"]
        print(f"hand-off write, ctx {small} -> {large}: last prefill chunk grows by {last:.1f} ms")
        print(f"hand-off read,  ctx {small} -> {large}: first decode step grows by {first:.1f} ms")
    two = by_key[(small, 2)]
    print(f"one prefill chunk without hand-off (ctx {small}): {two['other_chunk_ms']:.1f} ms; "
          f"same work with hand-off: {two['last_chunk_ms']:.1f} ms")  # fmt: skip

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "rows": rows}, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-id", default="openai/gpt-oss-20b")
    parser.add_argument("--ctx-lens", type=int, nargs="+", default=[4096, 32768])
    parser.add_argument("--prefill-seq-len", type=int, default=512)
    parser.add_argument("--short-prompt", type=int, default=100, help="tokens in the short prompt")
    parser.add_argument("--num-hidden-layers", type=int, default=None, help="reduce depth for fast compiles")
    parser.add_argument("--full-batch-size", type=int, default=4)
    parser.add_argument("--stages", type=int, default=4, help="prefill pipeline depth (mdp_num_partitions)")
    parser.add_argument("--prefill-num-devices", type=int, default=8)
    parser.add_argument("--decode-num-devices", type=int, default=4)
    parser.add_argument("--prefill-device-ids", type=int, nargs="+", default=None)
    parser.add_argument("--decode-device-ids", type=int, nargs="+", default=None)
    parser.add_argument("--mxfp6", action="store_true", help="compile with mxfp6_matmul")
    parser.add_argument("--mxint8-kv", action="store_true", help="compile with mxint8_kv_cache")
    parser.add_argument("--subfunc-npi", default=None, help="prefill NPI yaml (120B)")
    parser.add_argument("--non-subfunc-npi", default=None, help="decode NPI yaml (120B)")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument(
        "--handoff-once",
        action="store_true",
        help="wire the KV buffers on the first decode step only, not on every step",
    )
    parser.add_argument("--out", default="gpt_oss_ttft_handoff.json")
    main(parser.parse_args())
