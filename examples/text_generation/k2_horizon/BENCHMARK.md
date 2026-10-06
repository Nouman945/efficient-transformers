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

```csh
cd ~/efficient-transformers
source qeff_env/bin/activate.csh
setenv HF_HOME /local/mnt/workspace/nrasheed/hf
setenv QEFF_HOME /local/mnt/workspace/nrasheed/qeff_cache
setenv QEFF_LOG_PATH /local/mnt/workspace/nrasheed/qeff_logs

python examples/text_generation/k2_horizon/benchmark.py \
    --matrix examples/text_generation/k2_horizon/benchmark_matrix.json \
    --out /local/mnt/workspace/nrasheed/k2_horizon_benchmark.csv \
    --repeats 3
```

Useful variants:

```csh
# one case only, more repeats
python examples/text_generation/k2_horizon/benchmark.py --only mx_4dev_ctx4k --repeats 5 --out /local/mnt/workspace/nrasheed/k2.csv

# with a card price so tokens/s per dollar is filled in (price is an input, agree it with the customer)
python examples/text_generation/k2_horizon/benchmark.py --card-price-usd 8000 --out /local/mnt/workspace/nrasheed/k2.csv
```

The script appends one row per case and prints the row as JSON. A case that fails (compile error, out of memory) is printed and skipped; the others still run. Nothing else must be running on the card during the timed runs, or the power and throughput numbers are off.

## Report

Use the CSV as is, or this table (fill from the CSV). Reference numbers measured on 6 October 2026, single stream:

| Case | TTFT | Decode tok/s | Board W | tok/s per W |
|---|---|---|---|---|
| fp16, 1 device, ctx 4k | 0.20 s | 6.2 | 70 | 0.09 |
| MXFP6+MXINT8, 1 device, ctx 4k | 0.14 s | 14.4 | 70 | 0.21 |
| MXFP6+MXINT8, 4 devices, ctx 4k | 0.05 s | 51.8 | 127 | 0.41 |

Next to every number, state: prompt tokens, generated tokens, batch size, devices, precision, `prefill_seq_len`, `ctx_len`, SDK version (`/opt/qti-aic/tools/qaic-version-util --apps`) and the QEfficient commit. Idle power (45 W on this card) goes in the report too.

For Performance per Dollar: `decode_tok_s / price`. Decide with the customer whether price means list price of the card or an hourly cost, and say which one in the report.

## Interpreting the results

- Decode on one device is memory-bandwidth bound: 18 GB of fp16 weights read per token gives about 6 tok/s; MXFP6 cuts the bytes and roughly doubles it. 4 devices split the weights and multiply again.
- Continuous batching raises `decode_tok_s` (aggregate) while `decode_tok_s_per_stream` falls slowly; the sweet spot is where per-stream stays acceptable. Performance per Watt improves with batching because the board power rises much less than throughput.
- TTFT grows with input length; `mx_4dev_ctx4k_in1k` vs `mx_4dev_ctx4k` shows the cost per prompt token. `prefill_seq_len` larger than the prompt wastes compute; smaller means more chunks.
- Output text should still read correctly in the MXFP6 cases. The warm-up output is not printed by the script; run README Step 1 with `--mxfp6` once if you want to eyeball it.

## Codex prompt

> In ~/efficient-transformers on branch k2-horizon-server-docs, follow examples/text_generation/k2_horizon/BENCHMARK.md. Set the three cache variables as in SERVER_SETUP.md, then run benchmark.py over the full matrix with `--repeats 3` and `--out /local/mnt/workspace/nrasheed/k2_horizon_benchmark.csv`. Make sure nothing else is using the cards first (`qaic-util -q` shows Networks Active:0). Report the CSV contents, any case that failed with its error, and the SDK version from `qaic-version-util --apps`. Do not change source files, do not push.
