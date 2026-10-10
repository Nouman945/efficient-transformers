# Benchmarking K2-Horizon-7B on Cloud AI 100 Ultra

How to produce the benchmark numbers for `IFM/K2-Horizon-7B`: throughput, latency, Performance per Watt and Performance per Dollar, over a fixed matrix of configurations. Everything runs from `benchmark.py` in this folder and lands in one CSV.

Do [SERVER_SETUP.md](SERVER_SETUP.md) first (caches on the local disk, csh notes). The model must already have passed [README.md](README.md) Step 1 on this host.

## What is measured

All numbers come from `QEfficient`'s own `perf_metrics` after `generate()`, plus `qaic-util -q` for power.

| Metric | Meaning |
|---|---|
| `ttft_s` | Time to first token: prefill of the whole prompt, seconds |
| `decode_tok_s` | Decode throughput, tokens per second, summed over all streams in the batch |
| `decode_tok_s_per_stream` | `decode_tok_s / batch_size`, what one user sees |
| `total_tok_s` | Generated tokens divided by prefill plus decode time |
| `e2e_s` | Wall time of one `generate()` call |
| `idle_board_w` | Board power before the timed runs |
| `decode_board_w` | Peak board power sampled every 0.5 s during the timed runs |
| `tok_s_per_w` | `decode_tok_s / decode_board_w` (Performance per Watt) |
| `tok_s_per_dollar` | `decode_tok_s / card price`, only when `--card-price-usd` is given |

Every device on one Ultra card reports the power of the whole board, so the sampler takes the max across the devices in use, not the sum. One card, 4 devices, TDP cap 150 W.

Each case does one warm-up `generate()` (not timed) and then `--repeats` timed runs (default 3), reported as the mean.

## The matrix

`benchmark_matrix.json` has 8 cases. Edit it, or copy it, to change the sweep.

| id | Devices | Precision | Prefill / ctx | Input tokens | Batch | Purpose |
|---|---|---|---|---|---|---|
| `fp16_1dev_ctx4k` | 1 | fp16 | 128 / 4096 | 128 | 1 | Baseline, precision check |
| `mx_1dev_ctx4k` | 1 | MXFP6 + MXINT8 | 128 / 4096 | 128 | 1 | Production precision, one device |
| `mx_4dev_ctx4k` | 4 | MXFP6 + MXINT8 | 128 / 4096 | 128 | 1 | Whole card, single stream |
| `mx_4dev_ctx4k_in1k` | 4 | MXFP6 + MXINT8 | 1024 / 4096 | 1024 | 1 | Long prompt, TTFT at 1k input |
| `mx_4dev_ctx8k_in4k` | 4 | MXFP6 + MXINT8 | 1024 / 8192 | 4096 | 1 | Long context, 4k input |
| `mx_4dev_ctx4k_cb4` | 4 | MXFP6 + MXINT8 | 128 / 4096 | 128 | 4 | Continuous batching, 4 users |
| `mx_4dev_ctx4k_cb8` | 4 | MXFP6 + MXINT8 | 128 / 4096 | 128 | 8 | 8 users |
| `mx_4dev_ctx4k_cb16` | 4 | MXFP6 + MXINT8 | 128 / 4096 | 128 | 16 | 16 users |

Fields per case: `devices` (list of device ids), `mxfp6`, `mxint8_kv_cache`, `prefill_seq_len`, `ctx_len`, `input_len` (prompt tokens, a multiple of `prefill_seq_len` avoids a padded last chunk), `generation_len`, `batch_size` (1 = plain, more = continuous batching with `full_batch_size`).

Every distinct combination of devices, precision, prefill, ctx and batch size is a separate compile. The first pass over the matrix compiles 8 QPCs, tens of minutes each at 4096 and 8192 context; later passes reuse them from `QEFF_HOME`. `ctx_len` must stay at or below 65536 (rotary table cap).

## Run it

One command, `run_benchmark.py`. It sets `HF_HOME`, `QEFF_HOME` and `QEFF_LOG_PATH` under `/local/mnt/workspace/$USER` (values already in the environment win), refuses to start if a device is not `Ready` or has a network active, records the SDK version and the QEfficient commit, runs the matrix for each model (the 0.9B gets `benchmark_matrix_0_9b.json`, fp16 + MXINT8), and writes one CSV per model plus `REPORT.md` with the tables.

```csh
cd ~/efficient-transformers
source qeff_env/bin/activate.csh

python examples/text_generation/k2_horizon/run_benchmark.py --dry-run   # checks and plan, nothing compiled
python examples/text_generation/k2_horizon/run_benchmark.py             # 7B, 3.7B, 0.9B, full matrices
```

Results: `/local/mnt/workspace/$USER/k2_horizon_benchmark/{REPORT.md, meta.json, IFM__K2-Horizon-7B.csv, ...}`.

Useful variants:

```csh
# one model, one case, more repeats
python examples/text_generation/k2_horizon/run_benchmark.py --models IFM/K2-Horizon-7B --only mx_4dev_ctx4k --repeats 5

# with a card price so tokens/s per dollar is filled in (price is an input, agree it with the customer)
python examples/text_generation/k2_horizon/run_benchmark.py --card-price-usd 8000

# rebuild REPORT.md from the CSVs without running anything
python examples/text_generation/k2_horizon/run_benchmark.py --report-only
```

A rerun appends to the CSV and the report keeps the newest row per case. A case that fails (compile error, out of memory) is listed under the table and the script exits 1; the others still run. `benchmark.py` underneath is the single-matrix engine and can still be called on its own.

## Report

Use the CSV as is, or this table. Reference run of the full matrix on 6 October 2026 (256 generated tokens, 3 repeats, idle 46 to 48 W):

| Case | Input | Batch | TTFT | Decode tok/s | Per stream | Board W | tok/s per W |
|---|---|---|---|---|---|---|---|
| fp16, 1 dev, ctx 4k | 128 | 1 | 0.58 s | 5.9 | 5.9 | 65 | 0.09 |
| MXFP6+MXINT8, 1 dev, ctx 4k | 128 | 1 | 0.14 s | 14.1 | 14.1 | 71 | 0.20 |
| MXFP6+MXINT8, 4 dev, ctx 4k | 128 | 1 | 0.06 s | 48.6 | 48.6 | 131 | 0.37 |
| MXFP6+MXINT8, 4 dev, ctx 4k | 1024 | 1 | 0.44 s | 48.1 | 48.1 | 139 | 0.35 |
| MXFP6+MXINT8, 4 dev, ctx 8k | 4096 | 1 | 1.90 s | 44.8 | 44.8 | 140 | 0.32 |
| MXFP6+MXINT8, 4 dev, ctx 4k, CB | 128 | 4 | 0.06 s | 100 | 25.0 | 150 | 0.67 |
| MXFP6+MXINT8, 4 dev, ctx 4k, CB | 128 | 8 | 0.06 s | 284 | 35.5 | 133 | 2.14 |
| MXFP6+MXINT8, 4 dev, ctx 4k, CB | 128 | 16 | 0.06 s | 456 | 28.5 | 140 | 3.26 |

The batch-4 row hit the 150 W TDP cap and its per-stream rate is below the batch-8 row, so treat it as throttled and rerun the CB cases with `--repeats 5` before publishing.

Next to every number, state: prompt tokens, generated tokens, batch size, devices, precision, `prefill_seq_len`, `ctx_len`, SDK version (`/opt/qti-aic/tools/qaic-version-util --apps`) and the QEfficient commit. Idle power (45 W on this card) goes in the report too.

For Performance per Dollar: `decode_tok_s / price`. Decide with the customer whether price means list price of the card or an hourly cost, and say which one in the report.

## Interpreting the results

- Decode on one device is memory-bandwidth bound: 18 GB of fp16 weights read per token gives about 6 tok/s; MXFP6 cuts the bytes and roughly doubles it. 4 devices split the weights and multiply again.
- Continuous batching raises `decode_tok_s` (aggregate) while `decode_tok_s_per_stream` falls slowly; the sweet spot is where per-stream stays acceptable. Performance per Watt improves with batching because the board power rises much less than throughput.
- TTFT grows with input length; `mx_4dev_ctx4k_in1k` vs `mx_4dev_ctx4k` shows the cost per prompt token. `prefill_seq_len` larger than the prompt wastes compute; smaller means more chunks.
- Output text should still read correctly in the MXFP6 cases. The warm-up output is not printed by the script; run README Step 1 with `--mxfp6` once if you want to eyeball it.

## Codex prompt

> In ~/efficient-transformers on branch k2-horizon-server-docs, activate qeff_env (csh) and run `python examples/text_generation/k2_horizon/run_benchmark.py --dry-run`. If it reports no problems, run it again without `--dry-run`. Report the contents of /local/mnt/workspace/$USER/k2_horizon_benchmark/REPORT.md and meta.json, and any failed case with its error. Do not change source files, do not push.
