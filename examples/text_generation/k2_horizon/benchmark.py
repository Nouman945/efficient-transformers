# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Benchmark a K2 Horizon model on Cloud AI 100 over a matrix of configurations.

For every entry in the matrix file: compile (cached by QEfficient), run a warm-up,
then `--repeats` timed runs while sampling board power with qaic-util, and append
one CSV row with TTFT, decode tokens/s, total tokens/s, board power and
tokens/s per watt.

    python benchmark.py --matrix benchmark_matrix.json --out results.csv
    python benchmark.py --matrix benchmark_matrix.json --only mx_4dev_ctx4k --repeats 5
"""

import argparse
import csv
import json
import re
import statistics
import subprocess
import threading
import time
from pathlib import Path

from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM

QAIC_UTIL = "/opt/qti-aic/tools/qaic-util"
CSV_FIELDS = [
    "id",
    "model",
    "devices",
    "num_devices",
    "mxfp6",
    "mxint8_kv_cache",
    "prefill_seq_len",
    "ctx_len",
    "input_len",
    "generation_len",
    "batch_size",
    "repeats",
    "ttft_s",
    "decode_tok_s",
    "total_tok_s",
    "e2e_s",
    "decode_tok_s_per_stream",
    "idle_board_w",
    "decode_board_w",
    "tok_s_per_w",
    "tok_s_per_dollar",
    "qpc",
    "timestamp",
]


def parse_board_power(text):
    """Map QID -> Board power(Watts) from `qaic-util -q` output."""
    powers = {}
    qid = None
    for line in text.splitlines():
        m = re.match(r"\s*QID\s+(\d+)", line)
        if m:
            qid = int(m.group(1))
            continue
        m = re.match(r"\s*Board power\(Watts\):\s*([\d.]+)", line)
        if m and qid is not None:
            powers[qid] = float(m.group(1))
    return powers


def read_board_power(devices):
    try:
        out = subprocess.run([QAIC_UTIL, "-q"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    powers = parse_board_power(out)
    values = [powers[d] for d in devices if d in powers]
    # every device on one card reports the whole board, so take the max rather than the sum
    return max(values) if values else None


class PowerSampler:
    """Samples board power on a thread until stopped."""

    def __init__(self, devices, interval_s=0.5):
        self.devices = devices
        self.interval_s = interval_s
        self.samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            value = read_board_power(self.devices)
            if value is not None:
                self.samples.append(value)
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=15)

    def peak(self):
        return max(self.samples) if self.samples else None


def build_prompt(tokenizer, input_len):
    """A prompt that tokenizes to exactly input_len tokens including the BOS token."""
    base = "The quick brown fox jumps over the lazy dog. "
    ids = tokenizer(base * (input_len // 4 + 2), add_special_tokens=False)["input_ids"][: input_len - 1]
    return tokenizer.decode(ids)


def run_case(case, args, tokenizer):
    devices = case["devices"]
    batch_size = case.get("batch_size", 1)
    continuous_batching = batch_size > 1

    model = QEFFAutoModelForCausalLM.from_pretrained(
        args.model_name, trust_remote_code=True, continuous_batching=continuous_batching
    )
    compile_kwargs = dict(
        prefill_seq_len=case["prefill_seq_len"],
        ctx_len=case["ctx_len"],
        num_cores=args.num_cores,
        num_devices=len(devices),
        mxfp6_matmul=case.get("mxfp6", False),
        mxint8_kv_cache=case.get("mxint8_kv_cache", False),
        use_onnx_subfunctions=True,
        aic_enable_depth_first=True,
        mos=1,
    )
    if continuous_batching:
        compile_kwargs["full_batch_size"] = batch_size
    if args.node_precision_info and case.get("mxfp6", False):
        compile_kwargs["node_precision_info"] = args.node_precision_info
    qpc_path = model.compile(**compile_kwargs)

    prompt = build_prompt(tokenizer, case["input_len"])
    prompts = [prompt] * batch_size
    generate = lambda: model.generate(  # noqa: E731
        tokenizer=tokenizer, prompts=prompts, device_ids=devices, generation_len=case["generation_len"]
    )

    generate()  # warm-up, not timed
    idle_w = read_board_power(devices)

    metrics = []
    with PowerSampler(devices) as sampler:
        for _ in range(args.repeats):
            metrics.append(generate().perf_metrics)
    decode_w = sampler.peak()

    ttft = statistics.mean(float(m.prefill_time) for m in metrics)
    decode = statistics.mean(float(m.decode_perf) for m in metrics)
    total = statistics.mean(float(m.total_perf) for m in metrics)
    e2e = statistics.mean(float(m.total_time) for m in metrics)
    return {
        "id": case["id"],
        "model": args.model_name,
        "devices": " ".join(str(d) for d in devices),
        "num_devices": len(devices),
        "mxfp6": case.get("mxfp6", False),
        "mxint8_kv_cache": case.get("mxint8_kv_cache", False),
        "prefill_seq_len": case["prefill_seq_len"],
        "ctx_len": case["ctx_len"],
        "input_len": case["input_len"],
        "generation_len": case["generation_len"],
        "batch_size": batch_size,
        "repeats": args.repeats,
        "ttft_s": round(ttft, 3),
        "decode_tok_s": round(decode, 2),
        "total_tok_s": round(total, 2),
        "e2e_s": round(e2e, 2),
        "decode_tok_s_per_stream": round(decode / batch_size, 2),
        "idle_board_w": idle_w,
        "decode_board_w": decode_w,
        "tok_s_per_w": round(decode / decode_w, 4) if decode_w else "",
        "tok_s_per_dollar": round(decode / args.card_price_usd, 4) if args.card_price_usd else "",
        "qpc": str(qpc_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def main():
    parser = argparse.ArgumentParser(description="K2 Horizon benchmark matrix on Cloud AI 100")
    parser.add_argument("--model-name", default="IFM/K2-Horizon-7B")
    parser.add_argument("--matrix", default=str(Path(__file__).with_name("benchmark_matrix.json")))
    parser.add_argument("--out", default="k2_horizon_benchmark.csv")
    parser.add_argument("--only", nargs="*", help="Run only these matrix ids")
    parser.add_argument("--repeats", type=int, default=3, help="Timed runs per case, after one warm-up")
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument("--card-price-usd", type=float, default=None, help="Card price for tokens/s per dollar")
    parser.add_argument(
        "--node-precision-info", default=None, help="Compiler NPI yaml applied to every MXFP6 case (see configs/)"
    )
    args = parser.parse_args()

    cases = json.loads(Path(args.matrix).read_text())
    if args.only:
        cases = [c for c in cases if c["id"] in args.only]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    out = Path(args.out)
    write_header = not out.exists()
    for case in cases:
        print(f"\n=== {case['id']} ===", flush=True)
        try:
            row = run_case(case, args, tokenizer)
        except Exception as exc:  # keep going, one failed case must not lose the others
            print(f"{case['id']} failed: {exc}", flush=True)
            continue
        with out.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if write_header:
                writer.writeheader()
                write_header = False
            writer.writerow(row)
        print(json.dumps(row, indent=1), flush=True)
    print(f"\nResults in {out}")


if __name__ == "__main__":
    main()
