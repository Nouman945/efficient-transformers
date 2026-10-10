# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
End-to-end benchmark of the K2 Horizon sizes on one Cloud AI 100 Ultra card.

One command does the whole flow: cache variables, idle-card check, SDK and commit
record, the matrix per model through benchmark.py, one CSV per model and a
Markdown report with every table.

    python run_benchmark.py                                   # 7B, 3.7B, 0.9B, default matrices
    python run_benchmark.py --models IFM/K2-Horizon-7B --only mx_4dev_ctx4k --repeats 5
    python run_benchmark.py --workload serving                # 1024-token prompts, long outputs, up to 32 users
    python run_benchmark.py --dry-run                         # checks and plan only, no compile
    python run_benchmark.py --report-only                     # rebuild REPORT.md from the CSVs

Results land in <workspace>/k2_horizon_benchmark/ (workspace defaults to
/local/mnt/workspace/$USER). Run it from the repo's venv.
"""

import argparse
import csv
import getpass
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
QAIC_TOOLS = Path("/opt/qti-aic/tools")
DEFAULT_MODELS = ["IFM/K2-Horizon-7B", "IFM/K2-Horizon-3.7B", "IFM/K2-Horizon-0.9B"]
# the 0.9B degrades with MXFP6 weights, its matrices use fp16 + MXINT8 instead
# "serving": 1024-token prompts, long outputs, 1 to 32 users (128 for the 0.9B), the shapes in the customer's reference sheet
WORKLOADS = {
    "default": {"IFM/K2-Horizon-0.9B": "benchmark_matrix_0_9b.json", None: "benchmark_matrix.json"},
    "serving": {"IFM/K2-Horizon-0.9B": "benchmark_matrix_serving_0_9b.json", None: "benchmark_matrix_serving.json"},
}
MAX_CTX_LEN = 65536  # K2_HORIZON_MAX_POSITION_EMBEDDINGS

REPORT_COLUMNS = [
    ("id", "Case"),
    ("num_devices", "Devices"),
    ("precision", "Precision"),
    ("input_len", "Input"),
    ("generation_len", "Output"),
    ("batch_size", "Batch"),
    ("ttft_s", "TTFT s"),
    ("decode_tok_s", "Decode tok/s"),
    ("decode_tok_s_per_stream", "Per stream"),
    ("itl_ms", "ITL ms"),
    ("decode_board_w", "Board W"),
    ("tok_s_per_w", "tok/s per W"),
    ("tok_s_per_dollar", "tok/s per $"),
]


def model_slug(model_name):
    return model_name.replace("/", "__")


def run(cmd, timeout=30):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def parse_qaic_status(text):
    """QID -> {"status": str, "active": int} from `qaic-util -q` output."""
    cards = {}
    qid = None
    for line in text.splitlines():
        m = re.match(r"\s*QID\s+(\d+)", line)
        if m:
            qid = int(m.group(1))
            cards[qid] = {"status": "", "active": 0}
            continue
        if qid is None:
            continue
        m = re.match(r"\s*Status\s*:\s*(\S+)", line)
        if m:
            cards[qid]["status"] = m.group(1)
        m = re.match(r"\s*Networks Active\s*:\s*(\d+)", line)
        if m:
            cards[qid]["active"] = int(m.group(1))
    return cards


def setup_environment(workspace):
    """Caches on the local disk, never on the NFS home (quota). Existing values win."""
    defaults = {
        "HF_HOME": workspace / "hf",
        "QEFF_HOME": workspace / "qeff_cache",
        "QEFF_LOG_PATH": workspace / "qeff_logs",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, str(value))
    for key in ("HF_HOME", "QEFF_HOME", "QEFF_LOG_PATH"):
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
    return {key: os.environ[key] for key in defaults}


def check_cards(devices, dry_run):
    """Every device we will use must be Ready with no network loaded, or the numbers are off."""
    problems = []
    if not (QAIC_TOOLS / "qaic-util").exists():
        problems.append(f"{QAIC_TOOLS / 'qaic-util'} not found, is the Cloud AI SDK installed on this host?")
    else:
        cards = parse_qaic_status(run([str(QAIC_TOOLS / "qaic-util"), "-q"]))
        for d in sorted(devices):
            if d not in cards:
                problems.append(f"device {d} is not listed by qaic-util")
            elif cards[d]["status"].lower() != "ready":
                problems.append(f"device {d} status is {cards[d]['status'] or 'unknown'}, expected Ready")
            elif cards[d]["active"]:
                problems.append(f"device {d} has {cards[d]['active']} network(s) active, something else is using it")
    for p in problems:
        print(("warning: " if dry_run else "error: ") + p)
    return not problems or dry_run


def load_matrix(path, only):
    cases = json.loads(path.read_text())
    if only:
        unknown = set(only) - {c["id"] for c in cases}
        if unknown:
            raise SystemExit(f"unknown case id(s) {sorted(unknown)} in {path.name}")
        cases = [c for c in cases if c["id"] in only]
    for c in cases:
        if c["ctx_len"] > MAX_CTX_LEN:
            raise SystemExit(f"{c['id']}: ctx_len {c['ctx_len']} is above the rotary cap {MAX_CTX_LEN}")
        if c["prefill_seq_len"] > c["ctx_len"]:
            raise SystemExit(f"{c['id']}: prefill_seq_len is larger than ctx_len")
        if c["input_len"] + c["generation_len"] > c["ctx_len"]:
            raise SystemExit(f"{c['id']}: input_len + generation_len does not fit in ctx_len")
        if not c["devices"]:
            raise SystemExit(f"{c['id']}: empty device list")
    return cases


def collect_metadata(env):
    commit = run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"]).strip()
    branch = run(["git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "HEAD"]).strip()
    sdk = run([str(QAIC_TOOLS / "qaic-version-util"), "--apps"]).strip()
    return {
        "host": socket.gethostname(),
        "user": getpass.getuser(),
        "python": platform.python_version(),
        "qefficient_commit": commit or "unknown",
        "qefficient_branch": branch or "unknown",
        "qaic_sdk": sdk.splitlines()[0] if sdk else "unknown",
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        **env,
    }


def benchmark_model(model_name, cases, args, out_csv):
    """Runs every case through benchmark.py and appends the rows to out_csv."""
    # imported here so the cache variables are in the environment before QEfficient loads
    sys.path.insert(0, str(HERE))
    import benchmark
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    case_args = argparse.Namespace(
        model_name=model_name,
        repeats=args.repeats,
        num_cores=args.num_cores,
        card_price_usd=args.card_price_usd,
    )
    failures = []
    if out_csv.exists():
        with out_csv.open(newline="") as f:
            header = next(csv.reader(f), [])
        if header != benchmark.CSV_FIELDS:  # columns changed since that file was written, keep it aside
            out_csv.rename(out_csv.with_suffix(".old.csv"))
    write_header = not out_csv.exists()
    for case in cases:
        print(f"\n=== {model_name} / {case['id']} ===", flush=True)
        try:
            row = benchmark.run_case(case, case_args, tokenizer)
        except Exception as exc:  # one failed case must not lose the others
            print(f"{case['id']} failed: {exc}", flush=True)
            failures.append({"id": case["id"], "error": str(exc)})
            continue
        with out_csv.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=benchmark.CSV_FIELDS)
            if write_header:
                writer.writeheader()
                write_header = False
            writer.writerow(row)
        print(json.dumps(row, indent=1), flush=True)
    return failures


def precision_label(row):
    weights = "MXFP6" if row["mxfp6"] == "True" else "fp16"
    return f"{weights} + MXINT8 KV" if row["mxint8_kv_cache"] == "True" else weights


def latest_rows(csv_path):
    """One row per case id, the last one written wins (reruns append)."""
    rows = {}
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows[row["id"]] = row
    return list(rows.values())


def write_report(out_dir, models, failures):
    meta_path = out_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    lines = ["# K2 Horizon benchmark on Cloud AI 100 Ultra", ""]
    if meta:
        lines += [
            f"Host {meta.get('host')}, SDK {meta.get('qaic_sdk')}, QEfficient {meta.get('qefficient_commit')} "
            f"({meta.get('qefficient_branch')}), started {meta.get('started')}.",
            "",
        ]
    lines += [
        "Decode tok/s is the sum over all streams; per stream is what one user sees. Board W is the peak board "
        "power sampled during the timed runs (one card, 4 devices, 150 W TDP). tok/s per W = decode tok/s / board W. ITL is the mean time between two tokens of one stream.",
        "",
    ]
    columns = REPORT_COLUMNS
    for model_name in models:
        csv_path = out_dir / f"{model_slug(model_name)}.csv"
        lines += [f"## {model_name}", ""]
        if not csv_path.exists():
            lines += ["No results.", ""]
            continue
        rows = latest_rows(csv_path)
        cols = columns if any(r.get("tok_s_per_dollar") for r in rows) else columns[:-1]
        lines.append("| " + " | ".join(label for _, label in cols) + " |")
        lines.append("|" + "---|" * len(cols))
        for row in rows:
            row = {**row, "precision": precision_label(row)}
            lines.append("| " + " | ".join(str(row.get(key, "")) for key, _ in cols) + " |")
        lines.append("")
        failed = failures.get(model_name, [])
        if failed:
            lines += ["Failed cases:", ""] + [f"- `{f['id']}`: {f['error']}" for f in failed] + [""]
        lines += [
            f"Settings for every row: `prefill_seq_len` and `ctx_len` per case in the matrix, `num_cores` "
            f"{meta.get('num_cores', 16)}, repeats {meta.get('repeats', 3)} after one warm-up, "
            f"`mos=1`, `aic_enable_depth_first`, ONNX subfunctions. Full columns in `{csv_path.name}`.",
            "",
        ]
    report = out_dir / "REPORT.md"
    report.write_text("\n".join(lines))
    return report


def main():
    parser = argparse.ArgumentParser(description="K2 Horizon end-to-end benchmark on Cloud AI 100 Ultra")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--workload", choices=sorted(WORKLOADS), default="default", help="which matrix set to run")
    parser.add_argument("--matrix", help="matrix file for every model (overrides --workload)")
    parser.add_argument("--only", nargs="*", help="run only these case ids")
    parser.add_argument("--repeats", type=int, default=3, help="timed runs per case after one warm-up")
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument("--card-price-usd", type=float, help="card price, fills tok/s per dollar")
    parser.add_argument(
        "--workspace", default=f"/local/mnt/workspace/{getpass.getuser()}", help="local disk for caches and results"
    )
    parser.add_argument("--out-dir", help="results folder (default: <workspace>/k2_horizon_benchmark[_<workload>])")
    parser.add_argument("--dry-run", action="store_true", help="checks and plan only, no compile or run")
    parser.add_argument("--report-only", action="store_true", help="rebuild REPORT.md from existing CSVs")
    args = parser.parse_args()

    workspace = Path(args.workspace)
    suffix = "" if args.workload == "default" else f"_{args.workload}"
    out_dir = Path(args.out_dir) if args.out_dir else workspace / f"k2_horizon_benchmark{suffix}"
    if args.report_only:
        report = write_report(out_dir, args.models, {})
        print(report)
        return

    env = {}
    if not args.dry_run:
        try:
            env = setup_environment(workspace)
        except PermissionError as exc:
            raise SystemExit(f"{exc}: pass --workspace with a writable local path (see SERVER_SETUP.md)")

    plan = []
    for model_name in args.models:
        matrices = WORKLOADS[args.workload]
        matrix = Path(args.matrix) if args.matrix else HERE / matrices.get(model_name, matrices[None])
        plan.append((model_name, matrix, load_matrix(matrix, args.only)))
    devices = {d for _, _, cases in plan for c in cases for d in c["devices"]}

    print(f"workspace {workspace}, results in {out_dir}")
    for model_name, matrix, cases in plan:
        print(f"{model_name}: {len(cases)} cases from {matrix.name} ({', '.join(c['id'] for c in cases)})")
    if not check_cards(devices, args.dry_run):
        raise SystemExit(1)
    if "VIRTUAL_ENV" not in os.environ:
        print("warning: no venv active, QEfficient may not import (source qeff_env/bin/activate.csh first)")
    if args.dry_run:
        print("dry run, nothing compiled or run")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = collect_metadata(env)
    meta.update({"num_cores": args.num_cores, "repeats": args.repeats, "card_price_usd": args.card_price_usd})
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1), flush=True)

    failures = {}
    for model_name, _, cases in plan:
        failures[model_name] = benchmark_model(model_name, cases, args, out_dir / f"{model_slug(model_name)}.csv")

    report = write_report(out_dir, args.models, failures)
    print(f"\nreport: {report}")
    for model_name, failed in failures.items():
        for f in failed:
            print(f"failed: {model_name} / {f['id']}: {f['error']}")
    if any(failures.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
