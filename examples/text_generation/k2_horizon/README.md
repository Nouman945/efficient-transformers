# K2 Horizon on Cloud AI 100 Ultra

How to onboard, validate and benchmark the dense K2 Horizon models (`IFM/K2-Horizon-0.9B`, `IFM/K2-Horizon-3.7B`, `IFM/K2-Horizon-7B`) on a server with Cloud AI 100 Ultra cards. Every step below is run on the server; the commands are given for all three sizes.

## Status

All three sizes are onboarded: CPU parity (HF PyTorch, QEff PyTorch, ONNX Runtime) passes, the card output matches the CPU reference, the repo tests pass on AI 100, and the benchmark matrix has been run. One Ultra card, MXFP6 weights + MXINT8 KV cache (fp16 weights for the 0.9B), 128-token prompt, 256 generated tokens:

| Model | 1 device | 4 devices, single user | 4 devices, 16 users (aggregate / per user) | tok/s per W at 16 users |
|---|---|---|---|---|
| `IFM/K2-Horizon-7B` | 14.1 tok/s | 48.6 tok/s, TTFT 0.06 s | 452 / 28 tok/s | 3.2 |
| `IFM/K2-Horizon-3.7B` | 23.4 tok/s | 65.9 tok/s, TTFT 0.04 s | 385 / 24 tok/s | 2.9 |
| `IFM/K2-Horizon-0.9B` | 45.8 tok/s | 134.9 tok/s, TTFT 0.03 s | 1070 / 67 tok/s | 9.1 |

The repo tests on AI 100 pass for the dummy, continuous-batching and per-PR fp16/fp32 lanes; the bf16 lane targets AI 200 and cannot compile on an AI 100-only host.

| Model | Layers | Hidden | Weights fp16 | Weights MXFP6 | Precision to use | Notes |
|---|---|---|---|---|---|---|
| `IFM/K2-Horizon-0.9B` | 28 | 1536 | 2.2 GB | not used | fp16 + MXINT8 KV | MXFP6 degrades the output (the first four layers are the sensitive ones), so run it without `--mxfp6` |
| `IFM/K2-Horizon-3.7B` | 36 | 2560 | 10 GB | 4 GB | MXFP6 + MXINT8 KV | same architecture as the 7B, the 250k vocabulary is most of the weight |
| `IFM/K2-Horizon-7B` | 36 | 4096 | 18 GB | 7 GB | MXFP6 + MXINT8 KV | |

The wrapper covers the dense sizes; 0.9B, 3.7B and 7B are validated, the 32B has the same architecture but has not been run. The 36B and 375B MoVA sizes have routed value experts and a sparse MoE that are not mapped here.

## Files in this folder

| File | Purpose |
|---|---|
| `README.md` | this guide: onboarding, validation and benchmark steps for the three sizes |
| `SERVER_SETUP.md` | first-time setup on the shared server: csh, home quota, caches on the local disk |
| `k2_horizon_inference.py` | compile and run one model on the card, prints the output and `perf_metrics` (Steps 1 and 3) |
| `run_benchmark.py` | the benchmark: runs every case for every model, writes CSVs and `REPORT.md` (Step 4) |
| `benchmark.py` | single-matrix engine used by `run_benchmark.py`; can be run on its own |
| `benchmark_matrix.json`, `benchmark_matrix_0_9b.json` | default cases, 128-token prompt (MXFP6 for 7B and 3.7B, fp16 for the 0.9B) |
| `benchmark_matrix_serving.json`, `benchmark_matrix_serving_0_9b.json` | 1024-token prompts, long outputs, 1 to 32 users (128 for the 0.9B) |
| `tools/causal_lm_parity.py` | CPU parity check, HF PyTorch vs QEff PyTorch vs ONNX Runtime (Step 0) |
| `tools/k2_tiny_check.py` | the same three stages on a tiny random K2 config, seconds to run, for wrapper changes |
| `tools/mxfp6_probe.py` | simulates MXFP6 weights on CPU per layer or weight group to find what degrades a model |

The wrapper itself is in `QEfficient/transformers/models/k2_horizon/` (see the last section).

## Prerequisites

First time on a shared server (csh, home quota, local disk)? Do [SERVER_SETUP.md](SERVER_SETUP.md) first.

- A server with Cloud AI 100 Ultra cards, Platform SDK and Apps SDK installed (`/opt/qti-aic` exists). Check the cards with `/opt/qti-aic/tools/qaic-util -q`.
- Python 3.10 to 3.12. The repo pins `transformers==5.5.4`.
- Internet access to Hugging Face. The models are public, no token needed.
- Disk on the local drive, not the NFS home: 7B needs about 60 GB (bf16 checkpoint, fp32 ONNX, QPCs), 3.7B about 35 GB, 0.9B about 8 GB.

One AI 100 Ultra card is 4 devices of 32 GB. Every size fits on one device; use 4 devices for throughput.

## Install

csh or tcsh (the shell on these servers):

```csh
git clone -b k2-horizon-server-docs https://github.com/Nouman945/efficient-transformers.git
cd efficient-transformers
python3 -m venv qeff_env
source qeff_env/bin/activate.csh
pip install -U pip
pip install -e ".[test]"

mkdir -p /local/mnt/workspace/$user/hf /local/mnt/workspace/$user/qeff_cache /local/mnt/workspace/$user/qeff_logs
setenv HF_HOME /local/mnt/workspace/$user/hf
setenv QEFF_HOME /local/mnt/workspace/$user/qeff_cache
setenv QEFF_LOG_PATH /local/mnt/workspace/$user/qeff_logs
setenv HF_HUB_ENABLE_HF_TRANSFER 1
```

bash: same commands with `source qeff_env/bin/activate` and `export NAME=value`.

Every command below quotes the device list, `'[0]'`. csh, tcsh and zsh treat a bare `[0]` as a file pattern and fail with `python: No match.` The quotes are harmless in bash.

The models are remote code, so every load passes `trust_remote_code`. Without it transformers stops at an interactive prompt.

## Step 0: CPU parity (onboarding check, no card needed)

`tools/causal_lm_parity.py` runs HF PyTorch, QEff PyTorch and ONNX Runtime on the same prompt and checks that the tokens match. This is the proof that the wrapper is correct before anything is compiled. It writes `tools/results/<model>.json`.

```csh
cd examples/text_generation/k2_horizon
python tools/causal_lm_parity.py IFM/K2-Horizon-0.9B --trust-remote-code --prompt-len 8 --ctx-len 32
python tools/causal_lm_parity.py IFM/K2-Horizon-3.7B --trust-remote-code --prompt-len 8 --ctx-len 32
python tools/causal_lm_parity.py IFM/K2-Horizon-7B --trust-remote-code --prompt-len 8 --ctx-len 32
cd ../../..
```

The ONNX leg runs in fp32 and needs about 36 GB of host RAM for the 7B. On a smaller host use `--num-hidden-layers 8` (first 8 layers with real weights, enough to catch a wrapper bug) or check the PyTorch legs on the full model with `--dtype bfloat16 --skip-export`. Expected: `RESULT {... "match": true ...}` for every stage.

## Step 1: compile and run on the card

Start without MXFP6 so the first run is a precision check, not a performance run. One command per model:

```csh
python examples/text_generation/k2_horizon/k2_horizon_inference.py --model-name IFM/K2-Horizon-0.9B \
    --prompt "The capital of France is" --prefill-seq-len 128 --ctx-len 4096 --num-cores 16 --device-group '[0]'

python examples/text_generation/k2_horizon/k2_horizon_inference.py --model-name IFM/K2-Horizon-3.7B \
    --prompt "The capital of France is" --prefill-seq-len 128 --ctx-len 4096 --num-cores 16 --device-group '[0]'

python examples/text_generation/k2_horizon/k2_horizon_inference.py --model-name IFM/K2-Horizon-7B \
    --prompt "The capital of France is" --prefill-seq-len 128 --ctx-len 4096 --num-cores 16 --device-group '[0]'
```

The same thing through the CLI (replace the model name as needed):

```csh
python -m QEfficient.cloud.infer --model_name IFM/K2-Horizon-7B --trust_remote_code \
    --batch_size 1 --prompt_len 128 --ctx_len 4096 --num_cores 16 --device_group '[0]' \
    --prompt "The capital of France is" --mos 1 --aic_enable_depth_first
```

Expected output for this prompt, greedy, from the 7B: ` Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain is Madrid. The`. The 3.7B and 0.9B also answer ` Paris` and continue sensibly; compare them with Step 2.

The first run exports the ONNX (a few minutes) and compiles the QPC (longer). Both are cached under `QEFF_HOME`, so later runs with the same settings skip straight to generation. Add `--use-onnx-subfunctions` to cut export and compile time.

## Step 2: token match against Hugging Face (repo tests)

The repo tests build a reduced random model of each configuration, run HF PyTorch, QEff PyTorch, ONNX Runtime and the card, and assert the tokens match. Run from the repo root, with the same `HF_HOME` and `QEFF_HOME` as above, and without `-n auto` (the card tests must run one at a time):

```csh
setenv QEFF_REGENERATE_GOLDEN 1
pytest tests/transformers/models/causal_lm_models/test_causal_lm_models.py -k "K2-Horizon-7B and dummy" -v
pytest tests/transformers/models/causal_lm_models/test_causal_lm_models.py -k "K2-Horizon-0.9B and dummy" -v
pytest tests/transformers/models/causal_lm_models/test_causal_lm_models.py -k "k2_horizon_text" -v
```

- `K2-Horizon-7B and dummy`: the 7B configuration (head_dim 128, gate and QK norm). The 3.7B has the same architecture, so this test covers it too.
- `K2-Horizon-0.9B and dummy`: the 0.9B configuration (head_dim 64, YaRN rope).
- `k2_horizon_text`: the per-PR lanes on the 7B: continuous batching, ONNX subfunctions, CCL, fp16 and fp32 export. The speculative-decoding case is an expected xfail; the bf16 lane fails on an AI 100-only host for every model.

`QEFF_REGENERATE_GOLDEN=1` writes the HF reference tokens into `tests/golden_outputs/goldens.json`. If the file changed, commit it, it is part of the PR:

```csh
git add tests/golden_outputs/goldens.json
git commit -s -m "K2 Horizon: update goldens"
```

For the real weights, compare the card output of Step 1 with a greedy Hugging Face run of the same prompt:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "IFM/K2-Horizon-7B"  # or IFM/K2-Horizon-3.7B, IFM/K2-Horizon-0.9B
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, dtype=torch.bfloat16)
inputs = tokenizer("The capital of France is", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=24, do_sample=False)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:]))
```

Compare the first 20 or so tokens. fp16 on the card and bf16 on CPU can drift on a long tail, which is normal. A different first token, or repeated junk, is not.

## Step 3: performance precision

Switch on the production precision and read the metrics that `generate()` prints. The 7B and 3.7B take MXFP6 weights; the 0.9B keeps fp16 weights:

```csh
python examples/text_generation/k2_horizon/k2_horizon_inference.py --model-name IFM/K2-Horizon-7B \
    --prefill-seq-len 128 --ctx-len 4096 --generation-len 256 --num-cores 16 --device-group '[0,1,2,3]' \
    --mxfp6 --mxint8-kv-cache --use-onnx-subfunctions

python examples/text_generation/k2_horizon/k2_horizon_inference.py --model-name IFM/K2-Horizon-3.7B \
    --prefill-seq-len 128 --ctx-len 4096 --generation-len 256 --num-cores 16 --device-group '[0,1,2,3]' \
    --mxfp6 --mxint8-kv-cache --use-onnx-subfunctions

python examples/text_generation/k2_horizon/k2_horizon_inference.py --model-name IFM/K2-Horizon-0.9B \
    --prefill-seq-len 128 --ctx-len 4096 --generation-len 256 --num-cores 16 --device-group '[0,1,2,3]' \
    --mxint8-kv-cache --use-onnx-subfunctions
```

The output must still read correctly at this precision; if it does not, go back to Step 1 settings and compare. `perf_metrics` gives prefill time (time to first token), decode tokens per second, and total throughput. Batching is set at compile time (`batch_size` or continuous batching with `full_batch_size` in `compile()`), see `examples/text_generation/continuous_batching.py`; the benchmark in Step 4 does this for you.

Precision on AI 100 Ultra: fp16 compute, MXFP6 weights (`--mxfp6`), MXINT8 KV cache (`--mxint8-kv-cache`). bf16 and FP8 are not AI 100 features, so use the bf16 Hub repo and let the compiler quantize; skip the `-FP8` repos. If an MXFP6 run degrades a model, `tools/mxfp6_probe.py` simulates MXFP6 on CPU per weight group or layer to find the sensitive part (that is how the 0.9B result was found).

## Step 4: Benchmark (throughput, TTFT, ITL, Performance per Watt and per Dollar)

`run_benchmark.py` does the whole sweep in one command: sets the cache variables under `/local/mnt/workspace/$USER`, refuses to start if a device is busy, records the SDK version and QEfficient commit, runs every case (compile, one warm-up, `--repeats` timed runs with board power sampled), and writes one CSV per model plus `REPORT.md` with the tables.

```csh
python examples/text_generation/k2_horizon/run_benchmark.py --dry-run             # checks and plan, nothing compiled
python examples/text_generation/k2_horizon/run_benchmark.py                       # all three sizes, default matrix
python examples/text_generation/k2_horizon/run_benchmark.py --workload serving    # all three sizes, 1024-token prompts, long outputs, 1 to 32 users

python examples/text_generation/k2_horizon/run_benchmark.py --models IFM/K2-Horizon-7B      # one size
python examples/text_generation/k2_horizon/run_benchmark.py --models IFM/K2-Horizon-3.7B
python examples/text_generation/k2_horizon/run_benchmark.py --models IFM/K2-Horizon-0.9B
python examples/text_generation/k2_horizon/run_benchmark.py --models IFM/K2-Horizon-7B --only mx_4dev_ctx4k mx_4dev_ctx4k_cb16 --repeats 5
python examples/text_generation/k2_horizon/run_benchmark.py --card-price-usd 8000  # fills tok/s per dollar
python examples/text_generation/k2_horizon/run_benchmark.py --report-only         # rebuild REPORT.md from the CSVs
```

Results: `/local/mnt/workspace/$USER/k2_horizon_benchmark[_serving]/{REPORT.md, meta.json, IFM__K2-Horizon-7B.csv, ...}`. A rerun appends to the CSV and the report keeps the newest row per case; a failed case is listed under the table and the script exits 1.

Matrices (`benchmark_matrix*.json`, one compile per distinct devices / precision / prefill / ctx / batch):

| File | Models | Cases |
|---|---|---|
| `benchmark_matrix.json` | 7B, 3.7B | 128-token prompt, 256 out: fp16 1 device, MXFP6 + MXINT8 on 1 and 4 devices, 1k and 4k prompts, 4 / 8 / 16 users |
| `benchmark_matrix_0_9b.json` | 0.9B | same shapes, fp16 + MXINT8 (MXFP6 degrades the 0.9B output) |
| `benchmark_matrix_serving.json` | 7B, 3.7B | 1024 in / 1024 out, 1024 in / 10240 out, 10240 in / 1024 out, each at 1, 2, 4, 8, 16, 32 users |
| `benchmark_matrix_serving_0_9b.json` | 0.9B | 1024 in / 128 out at 1 to 128 users, 1024 in / 1024 out at 1 to 32 users |

Fields per case: `devices`, `mxfp6`, `mxint8_kv_cache`, `prefill_seq_len`, `ctx_len` (65536 max), `input_len`, `generation_len`, `batch_size` (more than 1 = continuous batching with `full_batch_size`). The 10240-output cases take about 6 minutes per repeat per case.

Columns in the report:

| Column | Meaning |
|---|---|
| `ttft_s` | Time to first token, prefill of the whole prompt |
| `decode_tok_s` | Decode tokens per second summed over all streams |
| `decode_tok_s_per_stream` | What one user sees, `decode_tok_s / batch_size` |
| `itl_ms` | Inter-token latency of one stream, `1000 / decode_tok_s_per_stream` (a mean, not a P50) |
| `decode_board_w` | Peak board power during the timed runs, from `qaic-util -q` (one card, 4 devices, 150 W TDP; every device reports the whole board, so the max is taken, not the sum) |
| `tok_s_per_w` | Performance per Watt, `decode_tok_s / decode_board_w` |
| `tok_s_per_dollar` | `decode_tok_s / --card-price-usd`; the price is an input, agree it with the customer and say whether it is list price or hourly cost |

Next to every number, state prompt tokens, generated tokens, batch size, devices, precision, `prefill_seq_len`, `ctx_len`, the SDK version and the QEfficient commit (all in `meta.json`). Nothing else may run on the card during a benchmark. When comparing with a serving run elsewhere, note that ours is a static batch of identical prompts through `generate()` and our ITL is a mean.

## Troubleshooting

- `Do you wish to run the custom code? [y/N]`: `trust_remote_code` was not passed.
- The rotary tables are capped at 65536 positions (`K2_HORIZON_MAX_POSITION_EMBEDDINGS`), so keep `ctx_len` within that.
- Out of memory during export: the export runs in fp32 and needs about 36 GB of host RAM for the 7B.
- Compile takes long: add `--use-onnx-subfunctions`, and keep `ctx_len` to what the benchmark needs.
- `[ERROR] cache_position is part of K2HorizonModel.forward's signature, but not documented`: printed by transformers when it loads the remote code. Harmless.

## What the wrapper changes

- `QEfficient/transformers/models/k2_horizon/modeling_k2_horizon.py`: KV cache attention (with the optional gate and QK norm used by larger sizes), grouped RMSNorm on the compiler custom op, static rotary tables.
- `QEfficient/transformers/models/pytorch_transforms.py`: the K2 classes are mapped by name through `KVCacheExternalModuleMapperTransform`.
- `QEfficient/transformers/modeling_utils.py`: `K2HorizonConfig` added to the external class mapping.
- `tests/configs/causal_model_configs.json`, `QEfficient/utils/test_utils.py`: dummy-layer test config.
