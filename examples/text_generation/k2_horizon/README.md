# K2 Horizon on Cloud AI 100 Ultra

How to onboard and validate the dense K2 Horizon models (`IFM/K2-Horizon-0.9B`, `IFM/K2-Horizon-3.7B`, `IFM/K2-Horizon-7B`, `IFM/K2-Horizon-32B`) on a server with Cloud AI 100 Ultra cards. Every step is run on the server; the commands are given for all four sizes. Benchmarking is a separate job and lives in its own repo, [Nouman945/qaic-llm-benchmark](https://github.com/Nouman945/qaic-llm-benchmark), not in QEfficient.

## Status

A size is onboarded when all four gates hold. Checked on one Ultra card (4 devices of 32 GB):

| Gate | 0.9B | 3.7B | 7B | 32B |
|---|---|---|---|---|
| 1. CPU parity, HF PyTorch == QEff PyTorch == ONNX Runtime (Step 0) | pass, full model | pass, full model | pass, full model | pass, reduced config and first layer; full model pending |
| 2. Card output == Hugging Face greedy, fp16 (Steps 1, 3b) | pass, 57/64 tokens | pass, 64/64 | pass, 64/64 | pass, 64/64, 4 devices |
| 3. Repo tests on the card (Step 2) | dummy pass; few pass; full: one token flips after 5 (under review) | dummy: one near-tie token flips (under review); few: 2 token mismatches (under review); full pass | dummy, few, full, per-PR pass | dummy, few, full pass |
| 4. Production precision reads right (Steps 3, 3b) | fp16 + MXINT8 KV, 57/64 (MXFP6 degrades it) | MXFP6 + MXINT8 KV, 64/64 | MXFP6 + MXINT8 KV, 49/64 | MXFP6 + MXINT8 KV: text right, forks from HF at token 1 on a near-tie; top-5 check pending |
| Long context, needle at 32K and 64K (Step 3b) | pending | pass, pass | pass, pass | pass; 64K compile hung |

Single user, 128-token prompt, 256 generated tokens, at the precision in gate 4 (`perf_metrics` from Step 3):

| Model | 1 device | 4 devices |
|---|---|---|
| `IFM/K2-Horizon-32B` | does not fit | 11.7 tok/s MXFP6 (5.6 tok/s, TTFT 0.7 s in fp16) |
| `IFM/K2-Horizon-7B` | 14.1 tok/s | 48.6 tok/s, TTFT 0.06 s |
| `IFM/K2-Horizon-3.7B` | 23.4 tok/s | 66.4 tok/s, TTFT 0.04 s |
| `IFM/K2-Horizon-0.9B` | 45.8 tok/s | 131.5 tok/s, TTFT 0.03 s |

The per-PR lanes pass for fp16 and fp32; the bf16 lane targets AI 200 and cannot compile on an AI 100-only host. In the few lane the two MDP tests fail for every model (upstream test bug, see Step 2). The three single-token flips in gate 3 are fp32-CPU vs fp16-card exact-match tests on near-tie logits; the shipped-shape runs in gates 2 and 4 passed for those sizes, and the flips are being looked at before they are called accepted.

| Model | Params (Hub) | Layers | Hidden | Heads / KV | Norm groups | Weights fp16 | Weights MXFP6 | KV per token (MXINT8) | Notes |
|---|---|---|---|---|---|---|---|---|---|
| `IFM/K2-Horizon-0.9B` | 1.08B | 28 | 1536 | 32 / 8, head_dim 64 | 1 | 2.2 GB | not used | 28 KB | YaRN rope, 64k vocabulary; MXFP6 degrades the output (the first four layers are the sensitive ones) |
| `IFM/K2-Horizon-3.7B` | 5.06B | 36 | 2560 | 32 / 8, head_dim 128 | 2 | 10 GB | 4 GB | 72 KB | same architecture as the 7B, the 250k vocabulary is a quarter of the weight |
| `IFM/K2-Horizon-7B` | 9.00B | 36 | 4096 | 32 / 8, head_dim 128 | 4 | 18 GB | 7 GB | 72 KB | |
| `IFM/K2-Horizon-32B` | 34.8B | 64 | 5120 | 64 / 8, head_dim 128 | 4 | 70 GB | 27 GB | 128 KB | `hidden_size != heads * head_dim`; needs all 4 devices, the fp32 export needs about 140 GB of host RAM |

The names count non-embedding parameters; the 250k-token vocabulary adds 1.3 to 2.6B on top, so the "7B" is a 9B model in memory and at the lm_head. All four are Apache-2.0, bf16 safetensors, not gated, pretrained from scratch by IFM (Qwen3-style decoder: GQA, RMSNorm, SwiGLU, RoPE) with a chat template; `eos_token_id` is `[1, 250019]` for the 3.7B, 7B and 32B and `1` for the 0.9B. No QK norm, no attention gate and no experts in any of them.

The wrapper covers these four dense sizes only; a config with experts is refused at load.

## Files in this folder

| File | Purpose |
|---|---|
| `README.md` | this guide: onboarding and validation steps for the four sizes |
| `SERVER_SETUP.md` | first-time setup on the shared server: csh, home quota, caches on the local disk |
| `k2_horizon_inference.py` | compile and run one model on the card, prints the output and `perf_metrics` (Steps 1 and 3) |
| `tools/causal_lm_parity.py` | the same three stages on the real weights (Step 0) |
| `tools/mxfp6_probe.py` | simulates MXFP6 weights on CPU per layer or weight group to find what degrades a model (Step 3) |
| `scripts/debug/k2_horizon_qpc_vs_hf.py` (PR branch) | compiled QPC vs the Hugging Face model, token by token, at the shipped shapes and at long context (Step 3b) |

The wrapper itself is in `QEfficient/transformers/models/k2_horizon/` (see the last section).

## Prerequisites

First time on a shared server (csh, home quota, local disk)? Do [SERVER_SETUP.md](SERVER_SETUP.md) first.

- A server with Cloud AI 100 Ultra cards, Platform SDK and Apps SDK installed (`/opt/qti-aic` exists). Check the cards with `/opt/qti-aic/tools/qaic-util -q`.
- Python 3.10 to 3.12. The repo pins `transformers==5.5.4`.
- Internet access to Hugging Face. The models are public, no token needed.
- Disk on the local drive, not the NFS home: 7B needs about 60 GB (bf16 checkpoint, fp32 ONNX, QPCs), 3.7B about 35 GB, 0.9B about 8 GB, 32B about 250 GB.
- Host RAM for the fp32 export: about 36 GB for the 7B, about 140 GB for the 32B (`free -g` to check).

One AI 100 Ultra card is 4 devices of 32 GB. The 0.9B, 3.7B and 7B fit on one device; use 4 devices for throughput. The 32B needs all 4 devices.

## Install

csh or tcsh (the shell on these servers):

```csh
git clone -b k2-horizon-onboarding-docs https://github.com/Nouman945/efficient-transformers.git
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

Later visits: `git fetch fork; git reset --hard fork/k2-horizon-onboarding-docs`.

## Step 0: CPU parity (onboarding check, no card needed)

First the quickcheck case: each size's real config reduced to 2 layers, random weights, HF PyTorch -> QEff PyTorch -> ONNX Runtime, token match. A minute per size; run it after every wrapper change:

```csh
pytest tests/unit_test/models/test_model_quickcheck.py -k k2_horizon -v
```

Expected: 4 passed. A KV mismatch is the wrapper; an ORT mismatch is the export.

Then the real weights. `tools/causal_lm_parity.py` runs the same three stages on the checkpoint and writes `tools/results/<model>.json`:

```csh
cd examples/text_generation/k2_horizon
python3 tools/causal_lm_parity.py IFM/K2-Horizon-0.9B --trust-remote-code --prompt-len 8 --ctx-len 32
python3 tools/causal_lm_parity.py IFM/K2-Horizon-3.7B --trust-remote-code --prompt-len 8 --ctx-len 32
python3 tools/causal_lm_parity.py IFM/K2-Horizon-7B --trust-remote-code --prompt-len 8 --ctx-len 32
python3 tools/causal_lm_parity.py IFM/K2-Horizon-32B --trust-remote-code --prompt-len 8 --ctx-len 32 --num-hidden-layers 8
cd ../../..
```

The ONNX leg runs in fp32 and needs about 36 GB of host RAM for the 7B and about 140 GB for the 32B. On a smaller host use `--num-hidden-layers 8` (first 8 layers with real weights, enough to catch a wrapper bug; the 32B command above does this) or check the PyTorch legs on the full model with `--dtype bfloat16 --skip-export`. Expected: `RESULT {... "match": true ...}` for every stage.

## Step 1: compile and run on the card

Start without MXFP6 so the first run is a precision check, not a performance run. One command per model:

```csh
python3 examples/text_generation/k2_horizon/k2_horizon_inference.py --model-name IFM/K2-Horizon-0.9B \
    --prompt "The capital of France is" --prefill-seq-len 128 --ctx-len 4096 --num-cores 16 --device-group '[0]'

python3 examples/text_generation/k2_horizon/k2_horizon_inference.py --model-name IFM/K2-Horizon-3.7B \
    --prompt "The capital of France is" --prefill-seq-len 128 --ctx-len 4096 --num-cores 16 --device-group '[0]'

python3 examples/text_generation/k2_horizon/k2_horizon_inference.py --model-name IFM/K2-Horizon-7B \
    --prompt "The capital of France is" --prefill-seq-len 128 --ctx-len 4096 --num-cores 16 --device-group '[0]'

python3 examples/text_generation/k2_horizon/k2_horizon_inference.py --model-name IFM/K2-Horizon-32B \
    --prompt "The capital of France is" --prefill-seq-len 128 --ctx-len 4096 --num-cores 16 --device-group '[0,1,2,3]' \
    --use-onnx-subfunctions
```

The 32B is 70 GB in fp16, so it runs on all four devices from the start; the other sizes fit on one. The same thing through the CLI (replace the model name as needed; add `--generation_len 256`, otherwise it decodes up to `ctx_len`):

```csh
python3 -m QEfficient.cloud.infer --model_name IFM/K2-Horizon-7B --trust_remote_code \
    --batch_size 1 --prompt_len 128 --ctx_len 4096 --generation_len 256 --num_cores 16 --device_group '[0]' \
    --prompt "The capital of France is" --mos 1 --aic_enable_depth_first
```

Expected output for this prompt, greedy, from the 7B: ` Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain is Madrid. The`. The 32B answers ` Paris, a city known for its rich history, culture, and iconic landmarks.` The 3.7B and 0.9B also answer ` Paris` and continue sensibly. Compare each with the Hugging Face run at the end of Step 2.

The first run exports the ONNX (a few minutes) and compiles the QPC (longer). Both are cached under `QEFF_HOME`, so later runs with the same settings skip straight to generation. Add `--use-onnx-subfunctions` to cut export and compile time. Keep the `PerfMetrics(...)` line, the log path and the `Model compiled to:` path for the record.

## Step 2: token match against Hugging Face (repo tests)

The repo tests run the same model four ways, HF PyTorch, QEff PyTorch, ONNX Runtime and the card, and assert the generated tokens are identical. There are three lanes per model. Run from the repo root, with the same `HF_HOME` and `QEFF_HOME` as above, and without `-n auto` (the card tests must run one at a time):

| Lane | `-k` | Weights | What it proves |
|---|---|---|---|
| dummy | `"K2-Horizon-<size> and dummy"` | random, 2 layers, real config features | the wrapper's math (grouped norm, rope, GQA, CB) |
| few | `"K2-Horizon-<size> and few"` | real checkpoint, first layer only | real weights load and match through the whole pipeline; 4 tests: plain, CB, and a 2-partition MDP compile with and without ONNX subfunctions |
| full | `"K2-Horizon-<size> and full"` | real checkpoint, all layers, fp32, 4 devices | the full model on the card equals Hugging Face token for token; 2 tests: plain and CB |

All four sizes are registered (`ModelConfig.EXTERNAL_MODELS` and `tests/configs/causal_model_configs.json`). Dummy and few lanes for every size:

```csh
foreach size (0.9B 3.7B 7B 32B)
    pytest tests/transformers/models/causal_lm_models/test_causal_lm_models.py -k "K2-Horizon-$size and dummy" -v
    pytest tests/transformers/models/causal_lm_models/test_causal_lm_models.py -k "K2-Horizon-$size and few" -v
end
```

Full lanes, one size at a time. Each loads the checkpoint in fp32 on the host twice (HF reference plus the export copy), so the 0.9B, 3.7B and 7B need up to 80 GB of RAM and the 32B about 300 GB. Skip the 32B full lane if the host cannot hold it; the few lane plus the Step 1 text check stand in for it.

```csh
pytest tests/transformers/models/causal_lm_models/test_causal_lm_models.py -k "K2-Horizon-7B and full" -v
```

The per-PR lanes (continuous batching, ONNX subfunctions, CCL, fp16 and fp32 export) run on the 7B dummy shape:

```csh
pytest tests/transformers/models/causal_lm_models/test_causal_lm_models.py -k "k2_horizon_text" -v
```

Expected: dummy 2 passed, 1 skipped; few 2 passed and 2 failed (`test_few_causal_lm_onnx_mdp_compile_only` asks for 2 MDP partitions without `num_devices=2`, an upstream test bug that hits every model); full 2 passed; per-PR 4 passed, 1 xfail, 1 failed. The xfail is the speculative-decoding case. The failure is the bf16 case: it compiles for AI 200 (the test picks the target from the dtype, `test_causal_lm_models.py:123`) and fails on an AI 100-only host for every model.

Each lane takes its Hugging Face reference tokens from `tests/golden_outputs/goldens.json`. A lane with no entry computes the reference live and warns but never writes the file: the first run of a new lane, and any run after a change that alters the model output, needs `setenv QEFF_REGENERATE_GOLDEN 1`. `QEFF_RUN_CPU_REFERENCES 1` also runs the QEff PyTorch and ONNX Runtime legs live. The dummy and per-PR entries are in the PR already (made on the laptop, no weights needed); the few and full entries are optional, upstream computes them live for most models, and the server cannot push anyway, so leave them where they are. If a dummy or per-PR entry ever changes, that is the commit:

```csh
git add tests/golden_outputs/goldens.json
git commit -s -m "K2 Horizon: goldens for the 3.7B and 32B lanes"
```

For a quick real-weight read on any size, compare the card output of Step 1 with a greedy Hugging Face run of the same prompt:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "IFM/K2-Horizon-7B"  # or IFM/K2-Horizon-3.7B, IFM/K2-Horizon-0.9B, IFM/K2-Horizon-32B (needs about 70 GB of host RAM)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, dtype=torch.bfloat16)
inputs = tokenizer("The capital of France is", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=24, do_sample=False)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:]))
```

Compare the first 20 or so tokens. fp16 on the card and bf16 on CPU can drift on a long tail, which is normal. A different first token, or repeated junk, is not.

## Step 3: production precision

Switch on the precision customers run and read the metrics that `generate()` prints. The 32B, 7B and 3.7B take MXFP6 weights; the 0.9B keeps fp16 weights:

```csh
python3 examples/text_generation/k2_horizon/k2_horizon_inference.py --model-name IFM/K2-Horizon-32B \
    --prefill-seq-len 128 --ctx-len 4096 --generation-len 256 --num-cores 16 --device-group '[0,1,2,3]' \
    --mxfp6 --mxint8-kv-cache --use-onnx-subfunctions

python3 examples/text_generation/k2_horizon/k2_horizon_inference.py --model-name IFM/K2-Horizon-7B \
    --prefill-seq-len 128 --ctx-len 4096 --generation-len 256 --num-cores 16 --device-group '[0,1,2,3]' \
    --mxfp6 --mxint8-kv-cache --use-onnx-subfunctions

python3 examples/text_generation/k2_horizon/k2_horizon_inference.py --model-name IFM/K2-Horizon-3.7B \
    --prefill-seq-len 128 --ctx-len 4096 --generation-len 256 --num-cores 16 --device-group '[0,1,2,3]' \
    --mxfp6 --mxint8-kv-cache --use-onnx-subfunctions

python3 examples/text_generation/k2_horizon/k2_horizon_inference.py --model-name IFM/K2-Horizon-0.9B \
    --prefill-seq-len 128 --ctx-len 4096 --generation-len 256 --num-cores 16 --device-group '[0,1,2,3]' \
    --mxint8-kv-cache --use-onnx-subfunctions
```

The output must still read correctly at this precision; if it does not, go back to Step 1 settings and compare. `perf_metrics` gives prefill time (time to first token), decode tokens per second and total throughput for one user. These are the numbers in the Status table. Multi-user numbers, power, cost per token and the customer's shapes are the job of [qaic-llm-benchmark](https://github.com/Nouman945/qaic-llm-benchmark).

Precision on AI 100 Ultra: fp16 compute, MXFP6 weights (`--mxfp6`), MXINT8 KV cache (`--mxint8-kv-cache`). bf16 and FP8 are not AI 100 features, so use the bf16 Hub repo and let the compiler quantize; skip the `-FP8`, `-NVFP4` and `-GGUF` repos. If an MXFP6 run degrades a model, `tools/mxfp6_probe.py` simulates MXFP6 on CPU per weight group or layer to find the sensitive part (that is how the 0.9B result was found).

## Step 3b: QPC against Hugging Face at the shipped shapes, and at long context

Steps 1 to 3 compare by eye and the repo lanes compare at 32 tokens. `scripts/debug/k2_horizon_qpc_vs_hf.py` compares a compiled QPC with the original model token by token at the shape you ship, and checks long range. Compiles are cached under `QEFF_HOME`, so a QPC that already exists from Step 1 or 3 is reused.

Prompt mode, greedy on both sides, reports the first token where they differ. In fp16, pass = the first 16 tokens identical (`--min-match`; fp16 vs bf16 usually agree for 20 or more, then drift, which is normal; a different first token is not). At MXFP6 pass `--hf-topk 5` instead: the QPC's own tokens are fed back to HF and each must be in HF's top-5 at its position, because a quantized run can fork on a near-tie at the first token (the 32B picks `Paris.` where HF picks `Paris,`) and still be right:

```csh
python3 scripts/debug/k2_horizon_qpc_vs_hf.py --model-name IFM/K2-Horizon-7B --prefill-seq-len 128 --ctx-len 4096 --device-group '[0]'
python3 scripts/debug/k2_horizon_qpc_vs_hf.py --model-name IFM/K2-Horizon-7B --prefill-seq-len 128 --ctx-len 4096 --device-group '[0,1,2,3]' \
    --mxfp6 --mxint8-kv-cache --use-onnx-subfunctions --hf-topk 5
```

Needle mode, the long-context check: a prose prompt of `--needle-tokens` tokens with a code word planted near the start and a question at the end; pass = the QPC answers with the word. This is the only step that exercises rope and the cache past a few thousand positions (the 0.9B's YaRN scaling is identity below 8192, so nothing before this touches it). Run it at 32K and at the 64K cap, on 4 devices, at the production precision; `--skip-hf` leaves out the CPU run, which would take hours at this length:

```csh
foreach size (0.9B 3.7B 7B 32B)
    python3 scripts/debug/k2_horizon_qpc_vs_hf.py --model-name IFM/K2-Horizon-$size --needle-tokens 30000 --prefill-seq-len 1024 --ctx-len 32768 \
        --device-group '[0,1,2,3]' --mxfp6 --mxint8-kv-cache --use-onnx-subfunctions --skip-hf
    python3 scripts/debug/k2_horizon_qpc_vs_hf.py --model-name IFM/K2-Horizon-$size --needle-tokens 60000 --prefill-seq-len 1024 --ctx-len 65536 \
        --device-group '[0,1,2,3]' --mxfp6 --mxint8-kv-cache --use-onnx-subfunctions --skip-hf
end
```

Drop `--mxfp6` for the 0.9B. Each is one compile (the 64K one is long) plus a prefill of 60K tokens. Results go to `scripts/debug/results/qpc_vs_hf_<model>_<tag>.json`; the `pass` field and the two texts are what to keep. A fail at 64K and a pass at 32K points at the rotary tables or the cache indexing, not the weights. The 32B at 64K with `--prefill-seq-len 1024` hung the compiler (one core, no log output for 40 hours, while its 32K case compiled in 15 minutes); kill such a run, retry once with `--prefill-seq-len 256`, and if it hangs again record the 32B's validated context as 32K.

## Troubleshooting

- `Do you wish to run the custom code? [y/N]`: `trust_remote_code` was not passed.
- The rotary tables are capped at 65536 positions (`K2_HORIZON_MAX_POSITION_EMBEDDINGS`), so keep `ctx_len` within that; `get_specializations` rejects more.
- Out of memory during export: the export runs in fp32 and needs about 36 GB of host RAM for the 7B, about 140 GB for the 32B.
- 32B on one device fails to compile: 70 GB fp16 / 27 GB MXFP6 plus the KV cache do not fit 32 GB; use `--device-group '[0,1,2,3]'`.
- Compile takes long: add `--use-onnx-subfunctions`, and keep `ctx_len` to what the run needs.
- `[ERROR] cache_position is part of K2HorizonModel.forward's signature, but not documented`: printed by transformers when it loads the remote code. Harmless.
- Yellow `aten::index_put_` / `aten::index` warnings during export: normal for every QEfficient export.
- The output repeats after a few thousand tokens: greedy decoding on a bare prompt does that; keep `--generation-len` at 256 for checks.

## What the wrapper changes

The upstream PR is branch `k2-horizon-onboarding`; this folder is not part of it. The PR touches:

- `QEfficient/transformers/models/k2_horizon/modeling_k2_horizon.py` (+ `__init__.py`): KV cache attention with the optional gate and QK norm, grouped RMSNorm on the compiler custom op with a unit-weight buffer built once, static rotary tables capped at 65536 positions, `get_specializations` that validates `prefill_seq_len` and `ctx_len` against the baked table, and refusal of configs with experts, partial rotary or sliding window.
- `QEfficient/transformers/models/pytorch_transforms.py`: the five K2 classes mapped by name through `KVCacheExternalModuleMapperTransform` (remote code, so no class import).
- `QEfficient/transformers/modeling_utils.py`: `K2HorizonConfig` in the external class mapping.
- `QEfficient/utils/constants.py`: `K2_HORIZON_MAX_POSITION_EMBEDDINGS`.
- `QEfficient/utils/test_utils.py`, `tests/configs/causal_model_configs.json`, `tests/golden_outputs/goldens.json`: the four sizes as external test models, a dummy-layer config per size and the per-PR `k2_horizon_text` entry, with their HF goldens.
- `tests/transformers/models/causal_lm_models/check_causal_models.py`: external models run the HF reference leg unless their entry sets `skip_hf_reference` (grok-1 keeps the skip).
- `tests/unit_test/{utils/test_modeling_registry.py, models/test_new_arch_accuracy.py, transforms/test_transform_accuracy.py}`: registry and mapping checks, the `ctx_len` guard against the baked table, and the grouped RMSNorm against a PyTorch reference.
- `docs/source/validate.md`, `examples/text_generation/README.md`: the K2 Horizon rows and run command.
