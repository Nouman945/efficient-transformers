# Cloud AI 100 model onboarding playbook

One document for the whole job: taking a customer's Hugging Face models from "which of these can run on Cloud AI 100 Ultra" to a merged QEfficient PR with benchmark numbers. Written from the MBZUAI catalog analysis and the `IFM/K2-Horizon-7B` onboarding (October 2026). Every command here was run; every pitfall listed was hit.

Scripts referenced as `tools/...` live next to this file. The worked example lives in `examples/text_generation/k2_horizon/` on this branch.

## Contents

| Part | What it covers | Where it runs |
|---|---|---|
| A | Environments: laptop, server, checks | both |
| B | Catalog analysis: classify a customer's repos, verify, report | laptop |
| C | Onboarding one model: read the code, write the wrapper, CPU parity | laptop |
| D | Repo tests, docs, review | laptop |
| E | Card validation: first run, repo tests on the card | server |
| F | Benchmarking: matrix, metrics, Perf/Watt, Perf/Dollar | server |
| G | Reporting: what to produce and how | laptop |
| H | Submission: branches, PR, rules | work system |
| I | Checklists and troubleshooting | |

Timeline for reference: catalog of 109 repos classified in one day; 7B wrapper written, CPU-validated, reviewed, card-validated and benchmarked in two more.

---

## Part A: environments

### A1. Laptop (CPU only, everything up to ONNX Runtime)

1. Clone and install. The repo pins `transformers==5.5.4` and CPU torch 2.7; do not reuse an env with another transformers.
   ```bash
   git clone https://github.com/quic/efficient-transformers.git
   cd efficient-transformers
   python3.11 -m venv qeff_env && source qeff_env/bin/activate
   pip install -U pip && pip install -e ".[test]"
   ```
2. Caches on a disk with space (`tools/env.sh`):
   ```bash
   export HF_HUB_CACHE=/path/hf_cache HF_HUB_ENABLE_HF_TRANSFER=1 QEFF_HOME=/path/qeff_cache
   ```
   Budget about 3x the bf16 checkpoint size per model (weights + fp32 ONNX). Delete weight caches of models that are done; the ONNX stays useful.
3. A stored Hugging Face login that has expired makes every public download fail with `401`. `hf auth login` again, or `export HF_HUB_DISABLE_IMPLICIT_TOKEN=1` while working on public repos. Gated repos need a valid login with access approved.
4. RAM sets the limit. fp32 is 4 bytes per parameter and export plus ONNX Runtime each hold a copy: on 31 GB, full-size three-stage parity works to about 2B parameters. Above that: reduced layers for the three stages and bf16 for the PyTorch-only stage (Part C4).

### A2. Server with AI 100 Ultra cards

Shell is csh/tcsh on our server. The three differences from bash: `setenv NAME value`, `source env/bin/activate.csh`, and quote any `[0]` (`--device_group '[0]'`), or you get `python: No match.`

1. Cards and SDK:
   ```csh
   /opt/qti-aic/tools/qaic-util -q
   /opt/qti-aic/tools/qaic-version-util --apps
   ```
   One Ultra card shows as 4 devices (QID 0 to 3). Note `Board power(Watts)` at idle (45 to 48 W here) and `Board TDP Cap(Watts)` (150). Write down the SDK version; every report carries it.
2. The home directory is NFS with a per-user quota. `df -h ~` shows the export, not your quota, so writes fail with `Disk quota exceeded` while `df` says 99 GB free. Put all three caches on the local disk; on our server that is `/local/mnt/workspace` (`drwxrwxrwt`; `/local/mnt` itself is root-owned):
   ```csh
   set user = `whoami`
   mkdir -p /local/mnt/workspace/$user/hf /local/mnt/workspace/$user/qeff_cache /local/mnt/workspace/$user/qeff_logs
   setenv HF_HOME /local/mnt/workspace/$user/hf
   setenv QEFF_HOME /local/mnt/workspace/$user/qeff_cache
   setenv QEFF_LOG_PATH /local/mnt/workspace/$user/qeff_logs
   setenv HF_HUB_ENABLE_HF_TRANSFER 1
   touch $HF_HOME/.w && rm $HF_HOME/.w && echo writable
   ```
   `tools/env.csh` has this. Put the `setenv` lines in `~/.cshrc` with the real path. `QEFF_LOG_PATH` matters: the log is the first write that fails on a full quota, and it takes `git pull` down with it.
3. If a run already died on the quota, clean home: `rm -rf ~/.cache/huggingface/hub/models--<org>--<model> ~/.cache/qefficient_logs`.
4. Python env: `python3 -m venv qeff_env; source qeff_env/bin/activate.csh; pip install -e ".[test]"`, or the SDK's own `source /opt/qti-aic/dev/python/qeff/bin/activate.csh` then `pip install -e .`.
5. The server has no GitHub credentials. Push from the laptop; regenerate goldens on the laptop (Part D4) instead of copying files off the server.
6. Driving the server through an agent (we used Codex) works well with this rule set: follow the named guide file, run pytest without `-n`, report every command output, do not change source files, do not push.

---

## Part B: catalog analysis

Goal: for an org with many repos, say which run today, which need work, which cannot run, and in what order to do them. Output: an analysis report and an effort-ranked plan (Part G).

### B1. Pull the catalog
```bash
python tools/catalog_fetch.py MBZUAI
```
One line per repo with `model_type`, `architectures`, size, and flags: gated, remote_code, adapter, no_config. Takes a minute for 100 repos, downloads nothing.

### B2. Classify into tiers
Rule per repo, in this order:

| Tier | Rule | Example |
|---|---|---|
| Not onboardable | LoRA adapter only, GGUF build, no `config.json` (research checkpoint), ONNX-only | Video-ChatGPT, ArTST, `*-GGUF` |
| Ready | `model_type` is mapped in the library (`pytorch_transforms.py`) and listed in `docs/source/validate.md` | GPT-2, Llama, Qwen3-VL, Qwen2.5-VL, LLaVA-HF |
| New wrapper | class exists in the pinned transformers but the library has no wrapper, or remote code with a plain decoder | Qwen2-VL, GPT-Neo, T5 seq2seq, SpeechT5, MobiLlama |
| Convert first | saved from a research codebase in a non-HF layout (original LLaVA, GLaMM, PALO), or carries extra heads (SAM decoder) | `LlavaLlamaForCausalLM`, `GLaMMForCausalLM` |

"Ready" from config alone is a hypothesis. Four of thirty MBZUAI "ready" repos were wrong: two had no `lm_head.weight` and a lost `tie_word_embeddings` flag, one had a broken `image_token_index`, two had wrong special-token ids.

### B3. Verify every "ready" repo properly
Run one agent per model with `tools/support_check_prompt.md`. The checks that matter, all without downloading weights:
- library mapping with file:line evidence, not a guess;
- safetensors headers vs a meta-device model built from the config: missing keys, unexpected keys, shape mismatches, after accounting for tied heads and transformers' legacy renames;
- transform dry run on the meta model (attention modules must become QEff classes);
- config compared with the library's validated reference for that family;
- sizing in 32 GB devices.

Verdicts: supported, supported with caveats, not supported, blocked (gated). Collect the caveats: they become the fixes list. Three kinds showed up: fix-at-load (override a wrong config value; no library change), partial use case (video models validated for images only), and hardware (multi-device, multi-card).

### B4. Local checks for the small ones
For anything that fits in RAM, run the three-stage parity now (Part C4) with `tools/causal_lm_parity.py` or `tools/vlm_parity.py`. Seven GPT-2 repos, one Qwen3 and one Qwen3-VL passed this way in an afternoon; that evidence goes in the report.

### B5. Rank by effort
Group A (no code), B (new wrapper), C (conversion or new pipeline). Inside each, rank by engineering time for the whole work item (fine-tunes share the work), and put Hub downloads next to it so value can be weighed against effort. Sizes: XS under a day, S 1 to 3 days, M a week, L two weeks, XL three weeks or more. Say plainly these are estimates.

---

## Part C: onboarding one model

Skip C2 and C3 if the class is already supported.

### C1. Read the upstream code
1. Get the files: in-tree, `transformers/models/<type>/modeling_<type>.py` in the venv; remote code, `curl -sL https://huggingface.co/<org>/<model>/resolve/main/modeling_<type>.py`.
2. Check its imports resolve under the pinned transformers: `python -c "import transformers.utils.output_capturing, transformers.masking_utils"`.
3. List the classes (Attention, DecoderLayer, Model, ForCausalLM, norm, rotary, MLP, MoE) and write down everything that differs from Llama. K2 Horizon: grouped RMSNorm, optional attention gate, optional QK norm, `rope_head_dim`, YaRN on one size, MoVA value experts and sparse MoE on the large sizes, 250k vocab, 524288 positions.
4. Make the per-size feature table. It decides what to support now and what to refuse.
5. Choose the wrapper to copy from `skills_studio/skills/qeff-model-onboarding/references/model-family-map.md`: Llama for dense GQA, Qwen3-MoE or GPT-OSS for MoE, Grok-1 for remote-code wiring, Gemma3 for hybrid cache.

### C2. Write the wrapper
`QEfficient/transformers/models/<family>/modeling_<family>.py` plus `__init__.py` with the license header.

In-tree class: subclass (`class QEffXAttention(XAttention)`) and map class to class in `KVCacheTransform._module_mapping`.

Remote-code class: plain `nn.Module` classes whose `forward` and optional `__qeff_init__` get bound by name onto the live modules.
1. Write `QEffXRMSNorm`, `QEffXAttention`, `QEffXDecoderLayer`, `QEffXModel`, `QEffXForCausalLM`. Use only attributes the upstream `__init__` creates.
2. Register under `KVCacheExternalModuleMapperTransform._match_string_replace_method` in `pytorch_transforms.py`, keyed by upstream class name, with `"__qeff_init__"` where needed.
3. Add `"<X>Config": "QEFFAutoModelForCausalLM"` to `EXTERNAL_MODEL_CLASS_MAPPING` in `modeling_utils.py`.

What goes where (the Llama pattern):
- Attention: q, k, v projections; the model's QK norm before the head split if any; `qeff_apply_rotary_pos_emb` with the passed `cos_cached`/`sin_cached`; `past_key_value_update(module=self, ...)`; `eager_attention_forward` (boolean mask, True = masked); the gate if any; `o_proj`.
- Decoder layer: two residual blocks; unpack `(hidden, router_logits)` if the MLP returns a tuple.
- Model `__qeff_init__`: `cos_cached`/`sin_cached` as `nn.Parameter` from `rotary_emb.inv_freq` and `attention_scaling` (covers YaRN), length capped by a constant in `QEfficient/utils/constants.py`, dtype from `self.embed_tokens.weight.dtype`.
- Model `forward`: `QEffDynamicCache.from_legacy_cache`, `_create_causal_mask(position_ids, target_length=past_seen_tokens)`, table slices by `position_ids`, layer loop, final norm, `BaseModelOutputWithPast`.
- CausalLM `forward`: pop `output_router_logits`, `logits_to_keep`, `labels`; logits at `position_ids.to(int32).argmax`; `get_submodules_for_export` returns the decoder layer class.
- Norm: `select_interface(CustomRMSNormFunc.apply, torch.ops.qefficient.rms_norm)`; grouped norm = reshape to groups, op with a ones weight, reshape back, full weight.

Rules learned the hard way:
- Refuse unsupported configs in `__qeff_init__` (MoE/MoVA, partial rotary, sliding window). Otherwise the 36B loads 72 GB and dies in `forward` with a `TypeError`.
- No blocked-attention branch in a remote-code wrapper: `BlockingAttentionTransform` only tags in-tree classes, so it is dead code.
- Ops without a bf16 kernel (`F.softplus`) in float32, or the bf16 compile lane fails with `Kernel lookup failed for: libjit_element_select_splat_kernel_f`.
- Static shapes only; no `x[..., :t.shape[-1]]`, no `.sum(keepdim=True)` on a runtime axis.
- `pyproject.toml` is read-only. `ruff check` and `ruff format` on every changed file.

### C3. Fix-at-load models (no wrapper change)
When Part B found a wrong config value, the fix is an override at load, never an edit to the customer's Hub repo: `from_pretrained(..., tie_word_embeddings=True)`, or a corrected `config` object passed in. Record the override in the model's recipe; it goes in the run guide and the PR body.

### C4. CPU parity, three stages
HF PyTorch = QEff PyTorch = ONNX Runtime, token for token. Export success alone proves nothing.
1. Tiny random model with every optional feature on (`tools/k2_tiny_check.py`, adapt the config fields). Fast, catches wiring.
2. Real weights, reduced layers: `tools/causal_lm_parity.py <model> --num-hidden-layers 2 --trust-remote-code`, then 8. Expected to match; the text is usually one repeated token, so weak evidence alone.
3. Full model, PyTorch only, bf16: `--dtype bfloat16 --skip-export`, prompt with a known answer, read the text. This one shows the wrapper is right.
4. Vision-language: `tools/vlm_parity.py` runs HF, exports both ONNX graphs (`kv_offload=True`) and drives them in ONNX Runtime with a loop mirroring `kv_offload_generate`; `--reduced` for big models. The repo's generic VLM ORT runner does not know every family's inputs.

---

## Part D: repo tests, docs, review

1. Dummy-layer config in `tests/configs/causal_model_configs.json` (`causal_lm_models`): shrunk dims in `additional_params`, optional features switched on, real `vocab_size`.
2. Per-PR entry (`per_pr_causal_text_models`): `id`, `model_name`, `model_type`, `is_moe`, `supports_blocking`, `num_hidden_layers`, `config_overrides`, `known_*` fields for lanes that cannot pass (speculative decoding is class-keyed and raises for remote code).
3. Remote code: add to `ModelConfig.EXTERNAL_MODELS` in `QEfficient/utils/test_utils.py`. Only set `skip_hf_reference` if the model cannot run on CPU; otherwise the test has no reference and compares the card against nothing.
4. Goldens, generated on CPU and committed:
   ```bash
   QEFF_REGENERATE_GOLDEN=1 pytest tests/transformers/models/causal_lm_models/test_causal_lm_models.py -k "<model> and dummy" -q
   QEFF_REGENERATE_GOLDEN=1 pytest tests/transformers/models/causal_lm_models/test_causal_lm_models.py -k "<per_pr_id> and not speculative" -q
   ```
   Without the SDK the tests fail at compile, after writing `tests/golden_outputs/goldens.json`. The variant keys matched the server's exactly.
5. Docs: row in `docs/source/validate.md` (marker ② on the family cell for remote code); a section in `examples/text_generation/README.md` with the `basic_inference.py --trust-remote-code` command. Site-specific guides stay out of the PR.
6. Two independent reviews before the card: one with the repo's `skills_studio/skills/qeff-pr-reviewer/SKILL.md` checklist, one adversarial against the upstream code with tiny CPU experiments allowed. Ours found: no HF reference in the dummy test, docs that could not ship, a dead branch, 512 MB rotary tables, an MoE failure only at forward time, untested optional features. All fixed before stage 4.

---

## Part E: card validation

### E1. First run (precision check, no MXFP6)
```csh
python examples/text_generation/basic_inference.py --model-name <org>/<model> --trust-remote-code \
    --prompt "The capital of France is" --prefill-seq-len 128 --ctx-len 4096 --num-cores 16 --device-group '[0]'
```
Order of events: config and remote code download (two harmless `[ERROR] cache_position ... not documented` lines from transformers), weights (18 GB for the 7B), ONNX export (minutes, host RAM about 36 GB for fp32), `qaic-compile` (tens of minutes, silent), then output and `Performance Stats`. Everything is cached under `QEFF_HOME`; later runs with the same settings skip to generation. `ctx_len` must stay within the rotary cap (65536 for K2).

Compare the text with a bf16 HF `generate()` of the same prompt. The first 20 tokens must agree; late drift between fp16 and bf16 is normal, a different first token is not. K2-7B gave " Paris. The capital of Germany is Berlin. The capital of Italy is Rome. ..." on both.

Then `--mxfp6` and `--mxint8-kv-cache`, on 1 device and on `'[0,1,2,3]'`. Precision on AI 100: fp16 compute, MXFP6 weights, MXINT8 KV cache. bf16 and FP8 are not AI 100 features; use the bf16 Hub repo and let the compiler quantize.

### E2. Repo tests on the card
```csh
setenv QEFF_REGENERATE_GOLDEN 1
pytest tests/transformers/models/causal_lm_models/test_causal_lm_models.py -k "<model> and dummy" -v
pytest tests/transformers/models/causal_lm_models/test_causal_lm_models.py -k "<per_pr_id>" -v
```
- Never `-n auto`: each worker needs its own card, 64 workers on 4 cards errors at setup. Serial or `-n 4`.
- Expected: dummy 2 passed, 1 skipped (KV-repeat feature test); per-PR all passed except registered xfails and the bf16 lane.
- The bf16 lane compiles for `aic-hw-version=ai200`. On an AI 100-only host it fails for every model with `libjit_element_select_splat_kernel_f`; prove it once with `-k "llama_text and bf16"` and leave it to the upstream CI.

---

## Part F: benchmarking

`tools/benchmark.py` with `tools/benchmark_matrix.json`: per case it compiles (cached), warms up, runs `--repeats` timed `generate()` calls while sampling `qaic-util -q` every 0.5 s, and appends a CSV row.

```csh
python tools/benchmark.py --model-name <org>/<model> --matrix tools/benchmark_matrix.json \
    --out /local/mnt/workspace/$user/<model>_benchmark.csv --repeats 3 [--card-price-usd 8000]
```

Metrics: `ttft_s`, `decode_tok_s` (aggregate over the batch), `decode_tok_s_per_stream`, `total_tok_s`, `e2e_s`, `idle_board_w`, `decode_board_w` (peak), `tok_s_per_w`, `tok_s_per_dollar`. Each device on a card reports the whole board's power; the sampler takes the max, not the sum.

The default matrix: fp16 and MXFP6 on 1 device; MXFP6 on 4 devices at 128, 1k and 4k input tokens; continuous batching at 4, 8 and 16 users. Every distinct devices/precision/prefill/ctx/batch is its own compile; the first pass over 8 cases takes about an hour, later passes reuse the QPCs.

Reading the results (K2-7B, one Ultra card): one device fp16 6 tok/s, MXFP6 14; four devices 48; 16 users 456 aggregate at 28 per user; Perf/Watt from 0.09 (fp16, 1 device) to 3.3 tok/s per W (16 users). Decode on one device is bandwidth bound, MXFP6 roughly doubles it, devices multiply it, batching lifts Perf/Watt because board power barely moves. A case that hits the 150 W cap may be throttled (our batch-4 row was below batch-8 per stream); rerun it with more repeats. Nothing else may use the cards during a run.

Performance per Dollar is `decode_tok_s / price`; agree with the customer whether price is list price or hourly cost and say which.

---

## Part G: reporting

Three documents came out of this work; build them from data, not by hand, so a rerun refreshes them.

1. **Catalog analysis report** (`tools/reports/build_analysis_report.py`): tiers with counts, per-family tables, local validation results, the four-stage flow, server commands, risks first. Generated from the catalog JSON and the parity result files; PDF through headless Chrome.
2. **Effort-ranked plan** (`tools/reports/build_ranked_plan.py`): groups A/B/C, rank, effort size, downloads, why this rank, what is left, suggested order. The rows are curated by hand in the script; the counts are checked against the catalog.
3. **Benchmark report**: the CSV plus a table with every number next to its settings (prompt tokens, generated tokens, batch, devices, precision, prefill, ctx, SDK version, library commit), idle power, and the Perf/Watt and Perf/Dollar formulas.

Rules that kept the reports honest: lead with the verdict and the risks; mark every estimate as an estimate; never claim parity from export alone; never claim a card result from a CPU run; when a finding changes (the DFJ repos moved tiers twice), say so in the next version rather than silently editing. Theme: company colors, no logo, footer saying it is a working document.

The PR body is the fourth document: `.github/PULL_REQUEST_TEMPLATE/pr_template.md` with a Testing section listing every stage with its result and every failing lane with its reason.

---

## Part H: submission

1. Fork, feature branch from upstream `main`. Two branches when there are site-specific docs: the PR branch (wrapper, transforms, constants, tests, goldens, `validate.md`, README section) and a docs branch rebased on it (run guide, server setup, benchmark, playbook). Only the first goes upstream. Rebase the docs branch after every PR-branch change and force-push it; tell the server to `git fetch; git reset --hard` on it.
2. Commits `git commit -s`, author set to the submitter, message says what, not how.
3. Repo rules: a human opens the PR, reviews every line, discloses AI assistance, lists the exact test commands. Agents prepare, humans submit.
4. Open as draft with the body file; `gh pr ready` once the card evidence is in:
   ```bash
   gh pr create --repo quic/efficient-transformers --base main --head <user>:<branch> --draft --title "..." --body-file pr_body.md
   ```
5. Expect questions on hash plumbing, the validated-models table, feature coverage in the test config, and anything site-specific in the diff.

---

## Part I: checklists and troubleshooting

### Checklist
- [ ] Laptop env: pinned transformers, caches on a big disk, HF login valid
- [ ] Server env: cards listed, SDK version noted, caches on local disk, `writable` printed
- [ ] Catalog fetched, tiers assigned, every "ready" repo verified with keys and a transform dry run
- [ ] Small models parity-checked on CPU; reports built from data
- [ ] Upstream code read, per-size feature table, imports resolve under the pin
- [ ] Wrapper written and registered; refuses unsupported configs; rotary capped; dtype from weights; no dead branches; bf16-unsafe ops in float32
- [ ] Tiny random, reduced-layer and full-bf16 parity all match
- [ ] Dummy and per-PR configs, `EXTERNAL_MODELS`, goldens committed, `validate.md` row, README section, ruff clean
- [ ] Two reviews done, findings fixed
- [ ] Card: fp16 text matches HF; MXFP6 on 1 and 4 devices
- [ ] Repo tests on the card pass; bf16 lane explained
- [ ] Benchmark CSV with Perf/Watt and SDK version; throttled rows rerun
- [ ] Reports built; PR body complete; PR opened by a human

### Troubleshooting
| Symptom | Cause | Fix |
|---|---|---|
| `401` on a public repo | expired stored HF token | `hf auth login` or `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` |
| `python: No match.` | csh expands `[0]` | quote it: `'[0]'` |
| `Disk quota exceeded`, even for `git pull` | NFS home quota full | caches to `/local/mnt/workspace/<user>`, delete partial downloads in `~/.cache` |
| `PermissionError ... /local/mnt/<user>` | `/local/mnt` is root-owned | use `/local/mnt/workspace/<user>`, check with `touch` |
| `Do you wish to run the custom code? [y/N]` | remote code without the flag | `--trust-remote-code` / `trust_remote_code=True` |
| `[ERROR] cache_position ... not documented` | transformers docstring check on the remote code | harmless |
| `No transforms applied to model` | class not mapped (or, for dual-QPC VLMs, the second wrapper on an already transformed model) | check the mapping; for VLMs it is expected once |
| `pytest.UsageError: QAIC worker gw6 has no exclusive 1-device slice` | `-n auto` on a 4-card host | run serial or `-n 4` |
| `Kernel lookup failed for: libjit_element_select_splat_kernel_f` in the bf16 lane | AI 200 target on an AI 100-only SDK, or a bf16-unsafe op | prove with Llama; compute the op in float32 |
| `RuntimeError: No reference tokens available` in the dummy test | model in `EXTERNAL_MODELS` with the HF leg skipped | remove `skip_hf_reference`, commit goldens |
| `lm_head.weight MISSING` on load, garbage output | checkpoint relies on tying but the config lost the flag | `tie_word_embeddings=True` at load |
| Image ignored by a LLaVA fine-tune | wrong `image_token_index` in the repo config | override at load |
| ONNX 0.5 GB bigger than the weights | rotary tables for 524288 positions | cap with a constant |
| 36B loads then `TypeError ... position_embeddings` | MoE layers not mapped | refuse in `__qeff_init__` |
| Export killed on the laptop | fp32 model + export copy over RAM | reduced layers, or export on the server |
| Benchmark row below a smaller batch | board at the 150 W cap, throttled | rerun with more repeats, note it |
