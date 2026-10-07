# K2-Horizon-0.9B and 3.7B: end-to-end onboarding on the server

Same wrapper, same flow as the 7B. The dense K2 Horizon wrapper already covers both sizes; what is left is the evidence: CPU parity, card parity, repo tests, benchmark, recorded results. This guide is written to be handed to Codex one model at a time. The Codex prompts are at the end.

What differs between the sizes:

| | 0.9B | 3.7B | 7B (done) |
|---|---|---|---|
| Params | 1.08B | 5.06B | 9.0B |
| Rope | YaRN (factor 16, attention_factor 1.277) | default | default |
| head_dim / norm groups | 64 / 1 | 128 / 2 | 128 / 4 |
| Vocab | 64256 | 250624 | 250624 |
| bf16 weights | 2.2 GB | 10 GB | 18 GB |
| fp32 host RAM for full-size export + ORT | about 10 GB | about 45 GB | about 75 GB |
| Repo dummy test entry | yes, covers the YaRN path | not needed (same path as 7B) | yes |

Already done before this guide: tiny random parity on the 0.9B config (YaRN path) passes HF = QEff = ONNX Runtime; the full-size 0.9B passes all three CPU stages (24 tokens, text " the capital of France.\n* The capital of France is Paris.\n* The capital of France is not Berlin."); dummy-test goldens for the 0.9B are committed. Step 1 for the 0.9B is therefore a re-check on the server; the 3.7B has no CPU evidence yet.

Prerequisites: [SERVER_SETUP.md](SERVER_SETUP.md) done, branch `k2-horizon-server-docs` checked out and up to date (`git fetch; git reset --hard origin/k2-horizon-server-docs` if the history moved), caches on `/local/mnt/workspace/<user>`, `free -g` noted.

Replace `<M>` below with `IFM/K2-Horizon-0.9B` or `IFM/K2-Horizon-3.7B`, and `<size>` with `0.9B` or `3.7B`.

## Step 1: CPU parity on the server host

Three stages, HF PyTorch = QEff PyTorch = ONNX Runtime, token for token. The script needs no card.

```csh
set T = examples/text_generation/k2_horizon/tools
python $T/causal_lm_parity.py <M> --trust-remote-code --prompt "The capital of France is"
```

- 0.9B: run at full size as written (about 10 GB of host RAM).
- 3.7B: full size needs about 45 GB of free host RAM (`free -g`). If there is less, run two commands instead: `--num-hidden-layers 8` (three stages, reduced) and then `--dtype bfloat16 --skip-export` (full model, HF vs QEff PyTorch only).

Pass = the printed `RESULT` line has every `*_match` field `true` and `compared_tokens` 24. The text after "The capital of France is" should read as a real continuation; one repeated token is normal only for the reduced-layer run. Results land in `$T/results/IFM__K2-Horizon-<size>.json`.

## Step 2: card run, precision check

```csh
python examples/text_generation/k2_horizon/k2_horizon_inference.py --model-name <M> \
    --prompt "The capital of France is" --prefill-seq-len 128 --ctx-len 4096 \
    --num-cores 16 --device-group '[0]'
```

Then the HF reference for the same prompt, on the host CPU:

```csh
python -c "import torch; from transformers import AutoModelForCausalLM, AutoTokenizer; m='<M>'; t=AutoTokenizer.from_pretrained(m, trust_remote_code=True); mod=AutoModelForCausalLM.from_pretrained(m, trust_remote_code=True, dtype=torch.bfloat16); i=t('The capital of France is', return_tensors='pt'); o=mod.generate(**i, max_new_tokens=32, do_sample=False); print(t.decode(o[0][i['input_ids'].shape[1]:]))"
```

Pass = the first 20 tokens agree. Late drift between fp16 on the card and bf16 on the host is normal; a different first token or repeated junk is not. Record the card's `Performance Stats` block.

Then the production precision, on 1 device and on the whole card:

```csh
python examples/text_generation/k2_horizon/k2_horizon_inference.py --model-name <M> \
    --prompt "The capital of France is" --prefill-seq-len 128 --ctx-len 4096 --generation-len 128 \
    --num-cores 16 --device-group '[0]' --mxfp6 --mxint8-kv-cache
python examples/text_generation/k2_horizon/k2_horizon_inference.py --model-name <M> \
    --prompt "The capital of France is" --prefill-seq-len 128 --ctx-len 4096 --generation-len 128 \
    --num-cores 16 --device-group '[0,1,2,3]' --mxfp6 --mxint8-kv-cache
```

The MXFP6 text should still read the same at the start.

## Step 3: repo tests on the card

0.9B only (the 3.7B shares the 7B's code path and test entry):

```csh
pytest tests/transformers/models/causal_lm_models/test_causal_lm_models.py -k "K2-Horizon-0.9B and dummy" -v
```

No `-n`. Expected: 2 passed (HF vs QEff vs ORT vs card, and the CB variant), 1 skipped. The goldens are already committed, so do not set `QEFF_REGENERATE_GOLDEN`; the test compares the card against them. If it reports no golden for a variant and runs the HF reference live, that is still a valid pass, but say so.

## Step 4: benchmark

```csh
python examples/text_generation/k2_horizon/benchmark.py --model-name <M> \
    --matrix examples/text_generation/k2_horizon/benchmark_matrix.json \
    --out /local/mnt/workspace/$user/k2_<size>_benchmark.csv --repeats 3
```

Same 8-case matrix as the 7B (see [BENCHMARK.md](BENCHMARK.md)). First pass compiles 8 QPCs. Make sure `qaic-util -q` shows `Networks Active:0` on every device before starting. Rerun any case whose `decode_board_w` reached 150 with `--only <id> --repeats 5`.

## Step 5: record and push

1. Create `examples/text_generation/k2_horizon/RESULTS_<size>.md` with: the Step 1 `RESULT` line(s), the Step 2 card text and the HF text side by side with the `Performance Stats` for fp16, MXFP6 1 device and MXFP6 4 devices, the Step 3 pytest summary line, the Step 4 CSV as a table, idle board power, and the SDK version from `/opt/qti-aic/tools/qaic-version-util --apps`.
2. Add a row per size to the status table at the top of [README.md](README.md).
3. Commit and push on the docs branch:
   ```csh
   git add examples/text_generation/k2_horizon/RESULTS_<size>.md examples/text_generation/k2_horizon/README.md
   git commit -s -m "K2 Horizon <size>: card parity, repo tests and benchmark results"
   git push origin k2-horizon-server-docs
   ```
   If the push is refused for credentials, see SERVER_SETUP.md on `gh auth login`, or leave the commit local and report it.

## 0.9B and MXFP6: known issue and the fix to test

On the card, the 0.9B with `--mxfp6` produced a looping output ("1.#. The capital of France is 1. ...") while fp16 and fp16 + MXINT8 were correct. A CPU simulation of MXFP6 (E2M3, 32-element blocks; `tools/mxfp6_probe.py`) reproduces the sensitivity and locates it:

| Weights in MXFP6 | Perplexity (fp32 = 9.28) | Greedy continuation |
|---|---|---|
| all | 9.70 | off topic after 5 tokens |
| all except layers 0 to 3 | 9.49 | correct |
| all except MLP of layers 0 to 3 | 9.44 | `* * * *` loop |
| all except attention of layers 0 to 3 | 9.78 | off topic |
| all except layers 24 to 27 | 9.70 | off topic |

No single weight is an outlier (quantization error is a uniform 3% everywhere); the model is simply sensitive in its first four layers. Keeping the 28 weight matmuls of layers 0 to 3 in full precision costs about 14% of the MXFP6 saving and restores correct output in the simulation.

`configs/k2_horizon_0_9b_npi_layers0-3.yaml` lists those 28 nodes (ONNX output names of `q/k/v/o_proj` and `gate/up/down_proj` MatMuls in `/model/layers.0..3/`). Test it on the card:

```csh
python examples/text_generation/k2_horizon/k2_horizon_inference.py --model-name IFM/K2-Horizon-0.9B \
    --prompt "The capital of France is" --prefill-seq-len 128 --ctx-len 4096 --generation-len 64 \
    --num-cores 16 --device-group '[0]' --mxfp6 --mxint8-kv-cache \
    --node-precision-info examples/text_generation/k2_horizon/configs/k2_horizon_0_9b_npi_layers0-3.yaml
```

- Output coherent and starting like the fp16 run: the NPI is the production setting for the 0.9B with MXFP6; pass `--node-precision-info` to `benchmark.py` for the MXFP6 cases and record it in RESULTS_0.9B.md.
- Still broken: widen the file to layers 0 to 5 (regenerate with `tools/mxfp6_probe.py`'s method, or edit the yaml by copying the layer blocks) and try once more. If that also fails, the validated setting for the 0.9B is fp16 + MXINT8 KV cache, no MXFP6; write that in RESULTS_0.9B.md and benchmark with `--only fp16_1dev_ctx4k` plus a copy of the MXFP6 cases with `"mxfp6": false`.

Either way the 0.9B's intended platform is mobile; on AI 100 its 2 GB of fp16 weights need no compression.

## What a failure means

| Where | Symptom | Likely cause |
|---|---|---|
| Step 1 | `hf_vs_kv_match` false | wrapper bug on this size's path (YaRN tables, norm groups). Stop, report the RESULT line and the two completions |
| Step 1 | `hf_vs_ort_match` false, kv true | export issue; report the ONNX path |
| Step 1 | killed, no traceback | host RAM; use the reduced-layer plus bf16 pair |
| Step 2 | compile error | report the full `Compiler stderr` block |
| Step 2 | different first token vs HF | precision or wrapper issue on the card; report both texts |
| Step 3 | assertion on tokens | card vs committed golden differ; report the test output |
| Step 4 | a case prints `failed:` | reported and skipped by the script; include the message |

## Codex prompts

0.9B:

> In ~/efficient-transformers (branch k2-horizon-server-docs, local disk checkout), follow examples/text_generation/k2_horizon/K2_SMALL_SIZES.md for IFM/K2-Horizon-0.9B, Steps 1 to 5 in order, and for the MXFP6 runs apply the section "0.9B and MXFP6" (test the NPI yaml first; use it in the benchmark if it works, otherwise fall back as described). Set the cache variables from SERVER_SETUP.md first and confirm `qaic-util -q` shows Networks Active:0. Run pytest without -n. Report every RESULT line, both generated texts in Step 2 with the Performance Stats blocks, the pytest summary, the benchmark CSV, idle board power and the SDK version. Do not change source files other than creating RESULTS_0.9B.md and editing the README status table. Commit with -s and push; if the push fails, report the commit hash.

3.7B:

> In ~/efficient-transformers (branch k2-horizon-server-docs, local disk checkout), follow examples/text_generation/k2_horizon/K2_SMALL_SIZES.md for IFM/K2-Horizon-3.7B, Steps 1, 2, 4 and 5 (Step 3 does not apply). Check `free -g` first and pick the Step 1 variant the guide prescribes for the available RAM. Set the cache variables from SERVER_SETUP.md and confirm Networks Active:0 before card runs. Report every RESULT line, both generated texts in Step 2 with the Performance Stats blocks, the benchmark CSV, idle board power and the SDK version. Do not change source files other than creating RESULTS_3.7B.md and editing the README status table. Commit with -s and push; if the push fails, report the commit hash.
