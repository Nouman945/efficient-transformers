# K2 Horizon on Cloud AI 100 Ultra

How to compile and run `IFM/K2-Horizon-7B` on a server with Cloud AI 100 Ultra cards, check that the card output matches Hugging Face, and collect the numbers for the benchmark.

## Status

| Stage | Where | Result |
|---|---|---|
| HF PyTorch vs QEff PyTorch, full 36 layers (bf16) | CPU | Tokens match over 24 generated tokens |
| HF vs QEff PyTorch vs ONNX Runtime, first 8 layers (fp32) | CPU | Tokens match |
| ONNX export, full model (fp32) | Server | Done |
| Compile and run on AI 100 Ultra, fp16, 1 device, ctx 4096 | Server | Output matches the CPU reference. TTFT 0.2 s, decode 6.19 tokens/s |
| MXFP6 + MXINT8, 1 device | Server | TTFT 0.14 s, decode 14.4 tokens/s, board power 70 W |
| MXFP6 + MXINT8, 4 devices | Server | TTFT 0.05 s, decode 51.8 tokens/s, board power about 127 W, 0.41 tokens/s per W |
| Repo tests on AI 100 (dummy, CB, per-PR fp16/fp32 lanes) | Server | Passed. The bf16 lane targets AI 200 and cannot compile on an AI 100-only host |

The wrapper covers the dense sizes (0.9B, 3.7B, 7B, 32B). The 36B and 375B MoVA sizes have routed value experts and a sparse MoE that are not mapped yet.

## Prerequisites

First time on a shared server (csh, home quota, local disk)? Do [SERVER_SETUP.md](SERVER_SETUP.md) first.

Onboarding a different model? Start from [ONBOARDING_PLAYBOOK.md](ONBOARDING_PLAYBOOK.md), the full step-by-step record of this onboarding, with the CPU parity scripts in `tools/`.

- A server with Cloud AI 100 Ultra cards, Platform SDK and Apps SDK installed (`/opt/qti-aic` exists). Check the cards with `/opt/qti-aic/tools/qaic-util -q`.
- Python 3.10 to 3.12. The repo pins `transformers==5.5.4`.
- Internet access to Hugging Face. The model is public, no token needed.
- Disk: about 18 GB for the bf16 checkpoint, about 36 GB for the fp32 ONNX, plus the QPC.

One AI 100 Ultra card is 4 devices of 32 GB. The 7B needs about 18 GB in fp16 and about 7 GB with MXFP6 weights, so one device is enough. Use more devices only for throughput.

## Install

bash:

```bash
git clone -b k2-horizon-server-docs https://github.com/Nouman945/efficient-transformers.git
cd efficient-transformers
python3 -m venv qeff_env && source qeff_env/bin/activate
pip install -U pip && pip install -e ".[test]"

export HF_HUB_ENABLE_HF_TRANSFER=1
export QEFF_HOME=/path/with/space/qeff_cache   # ONNX and QPC files go here
```

csh or tcsh:

```csh
git clone -b k2-horizon-server-docs https://github.com/Nouman945/efficient-transformers.git
cd efficient-transformers
python3 -m venv qeff_env
source qeff_env/bin/activate.csh
pip install -U pip
pip install -e ".[test]"

setenv HF_HUB_ENABLE_HF_TRANSFER 1
setenv QEFF_HOME /path/with/space/qeff_cache   # a real directory on a disk with space, for example $HOME/qeff_cache
```

If you use the SDK's own environment instead: `source /opt/qti-aic/dev/python/qeff/bin/activate` (bash) or `source /opt/qti-aic/dev/python/qeff/bin/activate.csh` (csh), then `pip install -e .` from the checkout.

Every command below quotes the device list, `'[0]'`. csh, tcsh and zsh treat a bare `[0]` as a file pattern and fail with `python: No match.` The quotes are harmless in bash.

The model is remote code, so every load passes `trust_remote_code`. Without it transformers stops at an interactive prompt.

## Step 1: compile and run

Start without MXFP6 so the first run is a precision check, not a performance run.

```bash
python examples/text_generation/k2_horizon/k2_horizon_inference.py \
    --model-name IFM/K2-Horizon-7B \
    --prompt "The capital of France is" \
    --prefill-seq-len 128 --ctx-len 4096 \
    --num-cores 16 --device-group '[0]'
```

The same thing through the CLI:

```bash
python -m QEfficient.cloud.infer --model_name IFM/K2-Horizon-7B --trust_remote_code \
    --batch_size 1 --prompt_len 128 --ctx_len 4096 --num_cores 16 --device_group '[0]' \
    --prompt "The capital of France is" --mos 1 --aic_enable_depth_first
```

Expected output for this prompt, greedy: ` Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain is Madrid. The`

The first run exports the ONNX (a few minutes) and compiles the QPC (longer). Both are cached under `QEFF_HOME`, so later runs with the same settings skip straight to generation. Add `--use-onnx-subfunctions` to cut export and compile time.

## Step 2: token match against Hugging Face

The repo tests do this for a reduced random model, on the card. Run both from the repo root, with the same `HF_HOME` and `QEFF_HOME` as the run above:

```csh
setenv QEFF_REGENERATE_GOLDEN 1
pytest tests/transformers/models/causal_lm_models/test_causal_lm_models.py -k "K2-Horizon and dummy" -v
pytest tests/transformers/models/causal_lm_models/test_causal_lm_models.py -k "k2_horizon_text" -v
```

- The first builds a tiny random K2 model, runs HF PyTorch, QEff PyTorch, ONNX Runtime and the card, and asserts the tokens match. `QEFF_REGENERATE_GOLDEN=1` makes it write the HF reference tokens into `tests/golden_outputs/goldens.json`.
- The second runs the per-PR lanes: continuous batching, ONNX subfunctions, CCL, fp16 and bf16 export. The speculative-decoding case is an expected xfail for this model.

Commit the golden file afterwards, it is part of the PR:

```csh
git status
git add tests/golden_outputs/goldens.json
git commit -s -m "K2 Horizon: add dummy-layer golden"
git push fork k2-horizon-server-docs
```

For the real weights, compare the card output with a greedy Hugging Face run of the same prompt:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "IFM/K2-Horizon-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, dtype=torch.bfloat16)
inputs = tokenizer("The capital of France is", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=24, do_sample=False)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:]))
```

Compare the first 20 or so tokens. fp16 on the card and bf16 on CPU can drift on a long tail, which is normal. A different first token, or repeated junk, is not.

## Step 3: performance runs

Switch on the production precision and read the metrics that `generate()` prints:

```bash
python examples/text_generation/k2_horizon/k2_horizon_inference.py \
    --prefill-seq-len 128 --ctx-len 4096 --generation-len 256 \
    --num-cores 16 --device-group '[0]' --mxfp6 --mxint8-kv-cache
```

`perf_metrics` gives prefill time (time to first token), decode tokens per second, and total throughput. Repeat for the context lengths and batch sizes the customer cares about. Batching is set at compile time (`batch_size` or continuous batching with `full_batch_size` in `compile()`), see `examples/text_generation/continuous_batching.py`.

Precision on AI 100 Ultra: fp16 compute, MXFP6 weights (`--mxfp6`), MXINT8 KV cache (`--mxint8-kv-cache`). bf16 and FP8 are not AI 100 features, so use the bf16 Hub repo and let the compiler quantize; skip the `-FP8` repos.

## Step 4: Performance per Watt and per Dollar

For the full sweep (devices, precision, context lengths, batch sizes) use [BENCHMARK.md](BENCHMARK.md) and `benchmark.py`; they write one CSV with all the metrics below.

- Watts: `/opt/qti-aic/tools/qaic-util -q` prints `Board power(Watts)` and `SOC power(Watts)` per device (board TDP cap is 150 W on the Ultra, SoC cap 31 W). Sample it while a decode run is in progress, for example `watch -n 1 /opt/qti-aic/tools/qaic-util -q` in a second terminal, and record idle power as well. Use board power for the per-card number.
- Performance per Watt = decode tokens per second divided by the measured card power during decode.
- Performance per Dollar = decode tokens per second divided by the card price (or hourly cost) used for the comparison. The price is an input, agree it with the customer.

Report the compile settings next to every number: prompt length, context length, batch size, cores, devices, MXFP6, MXINT8.

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
