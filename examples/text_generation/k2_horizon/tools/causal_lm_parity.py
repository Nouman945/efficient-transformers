"""
CPU parity check for one causal LM: HF PyTorch -> QEff KV PyTorch -> ONNX export -> ONNX Runtime,
with a token match at every stage. Compile and run on the card is Step 1 of the README.

Usage: python causal_lm_parity.py IFM/K2-Horizon-7B --trust-remote-code [--prompt "..."] [--prompt-len 8] [--ctx-len 32]
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.utils.run_utils import ApiRunner

RESULTS_DIR = Path(__file__).parent / "results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name")
    parser.add_argument("--prompt", default="My name is")
    parser.add_argument("--prompt-len", type=int, default=8)
    parser.add_argument("--ctx-len", type=int, default=32)
    parser.add_argument(
        "--num-hidden-layers",
        type=int,
        default=-1,
        help="Load only the first N layers (for models too big for this host)",
    )
    parser.add_argument("--skip-ort", action="store_true", help="Export only, skip the ONNX Runtime leg")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--skip-export", action="store_true", help="Stop after the QEff PyTorch stage")
    args = parser.parse_args()

    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    model_kwargs = dict(
        attn_implementation="eager",
        low_cpu_mem_usage=False,
        dtype=getattr(torch, args.dtype),
        trust_remote_code=args.trust_remote_code,
    )
    if args.num_hidden_layers > 0:
        model_kwargs["num_hidden_layers"] = args.num_hidden_layers
    model_hf = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model_hf.eval()

    runner = ApiRunner(
        1,
        tokenizer,
        model_hf.config,
        [args.prompt],
        args.prompt_len,
        args.ctx_len,
        dtype=getattr(torch, args.dtype),
    )

    result = {
        "model": args.model_name,
        "architecture": model_hf.__class__.__name__,
        "params": sum(p.numel() for p in model_hf.parameters()),
        "num_hidden_layers": args.num_hidden_layers,
        "prompt": args.prompt,
        "prompt_len": args.prompt_len,
        "ctx_len": args.ctx_len,
        "dtype": args.dtype,
    }

    # Greedy loop without EOS stopping, so all gen_len tokens are compared.
    hf_tokens = np.asarray(runner.run_hf_model_on_pytorch_CB(model_hf)[0]).reshape(-1)
    # The QEff transforms edit the model in place, so wrap it only after the HF reference run.
    qeff_model = QEFFAutoModelForCausalLM(model_hf, pretrained_model_name_or_path=args.model_name)
    kv_tokens = np.asarray(runner.run_kv_model_on_pytorch(qeff_model.model)).reshape(-1)
    n = min(len(hf_tokens), len(kv_tokens))
    result["hf_tokens"] = hf_tokens.tolist()
    result["hf_vs_kv_match"] = bool(n > 0 and (hf_tokens[:n] == kv_tokens[:n]).all())
    result["compared_tokens"] = n
    result["hf_text"] = tokenizer.decode(hf_tokens)
    if args.skip_export:
        write_result(result, args.model_name)

    start = time.time()
    onnx_path = qeff_model.export()
    result["onnx_path"] = str(onnx_path)
    result["export_seconds"] = round(time.time() - start, 1)

    # ORT loads the ONNX weights again, so drop the PyTorch copy first to stay inside RAM.
    del qeff_model, model_hf
    gc.collect()

    if not args.skip_ort:
        ort_tokens = np.asarray(runner.run_kv_model_on_ort(str(onnx_path))).reshape(-1)
        n = min(len(hf_tokens), len(ort_tokens))
        result["hf_vs_ort_match"] = bool(n > 0 and (hf_tokens[:n] == ort_tokens[:n]).all())
        result["kv_vs_ort_match"] = bool((kv_tokens[: len(ort_tokens)] == ort_tokens[: len(kv_tokens)]).all())

    write_result(result, args.model_name)


def write_result(result, model_name):
    RESULTS_DIR.mkdir(exist_ok=True)
    suffix = "" if result["num_hidden_layers"] == -1 else f"__{result['num_hidden_layers']}L"
    out = RESULTS_DIR / (model_name.replace("/", "__") + suffix + ".json")
    out.write_text(json.dumps(result, indent=2))
    print("RESULT", json.dumps({k: v for k, v in result.items() if k != "hf_tokens"}))
    sys.exit(0 if all(v for k, v in result.items() if k.endswith("_match")) else 1)


if __name__ == "__main__":
    main()
