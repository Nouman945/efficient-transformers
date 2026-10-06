# Playbook: onboarding a Hugging Face model to Cloud AI 100 with QEfficient

Step by step, written from the `IFM/K2-Horizon-7B` onboarding (October 2026). It covers a model whose class is not in the library yet, including remote-code models. If the model's class is already supported, skip Phases 3 and 4 and go from Phase 2 straight to Phase 5.

The whole flow, and what each phase proves:

| Phase | Where | Proves |
|---|---|---|
| 1. Classify | laptop | whether the library already supports the class, and what is missing |
| 2. Read the upstream code | laptop | which features the wrapper must handle |
| 3. Write the wrapper | laptop | code exists and is registered |
| 4. CPU parity | laptop | HF PyTorch = QEff PyTorch = ONNX Runtime, token for token |
| 5. Tests, docs, review | laptop | the repo's own tests carry the model; two reviews pass |
| 6. Card run | server | ONNX compiles and the card output matches HF |
| 7. Repo tests on the card | server | dummy, continuous batching and per-PR lanes pass |
| 8. Benchmark | server | TTFT, tokens/s, Perf/Watt, Perf/Dollar |
| 9. Pull request | work system | upstream submission with all evidence |

Time for K2-Horizon-7B: about two working days, with the card runs driven remotely.

## Phase 0: environment on the laptop

The laptop has no card and no SDK. It can do everything up to ONNX Runtime.

1. Clone the repo and make a venv. The repo pins `transformers==5.5.4` and CPU torch 2.7, so do not reuse a conda env with another transformers.
   ```bash
   git clone https://github.com/quic/efficient-transformers.git
   cd efficient-transformers
   python3.11 -m venv qeff_env && source qeff_env/bin/activate
   pip install -U pip && pip install -e ".[test]"
   ```
2. Set the caches on a disk with space. Weights plus fp32 ONNX is about 3x the bf16 checkpoint size.
   ```bash
   export HF_HUB_CACHE=/path/hf_cache HF_HUB_ENABLE_HF_TRANSFER=1 QEFF_HOME=/path/qeff_cache
   ```
   `tools/env.sh` in this folder is the version we used.
3. If downloads fail with `401` on a public repo, a stored Hugging Face login has expired. `hf auth login` again, or `export HF_HUB_DISABLE_IMPLICIT_TOKEN=1` for public repos.
4. RAM decides what you can run at full size: fp32 needs 4 bytes per parameter, and the ONNX export plus ONNX Runtime each need a copy. On 31 GB, full-size three-stage parity works up to about 2B parameters; above that use reduced layers (Phase 4) and bf16 for the PyTorch-only stage.

## Phase 1: classify the model

Goal: decide between "already supported", "new wrapper", "convert the checkpoint first", "not onboardable".

1. Pull the facts from the Hub API, no download:
   ```bash
   curl -s https://huggingface.co/api/models/<org>/<model> | python3 -m json.tool | head -60
   curl -sL https://huggingface.co/<org>/<model>/resolve/main/config.json
   ```
   Record: `architectures`, `model_type`, `auto_map` (remote code), `transformers_version`, gated flag, weight format (safetensors or `.bin`), parameter count, license.
2. Check whether the class is already wired in the library. All three must be true for "supported":
   - the HF class appears in `QEfficient/transformers/models/pytorch_transforms.py` (`KVCacheTransform._module_mapping` for in-tree classes, `KVCacheExternalModuleMapperTransform._match_string_replace_method` for remote code);
   - `docs/source/validate.md` lists the family;
   - the config has no feature the wrapper ignores (sliding window, MoE, partial rotary, unusual norm).
3. Check the checkpoint fits the class without downloading weights. Read the safetensors headers and compare with a meta-device model:
   ```python
   from huggingface_hub import HfApi
   import torch
   from transformers import AutoConfig, AutoModelForCausalLM
   meta = HfApi().get_safetensors_metadata("<org>/<model>")
   keys = {k: tuple(t.shape) for fm in meta.files_metadata.values() for k, t in fm.tensors.items()}
   cfg = AutoConfig.from_pretrained("<org>/<model>", trust_remote_code=True)
   with torch.device("meta"):
       model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
   expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
   print("missing", sorted(set(expected) - set(keys))[:10])
   print("unexpected", sorted(set(keys) - set(expected))[:10])
   ```
   Account for benign differences (tied `lm_head`, keys transformers renames on load). This check is what catches repos with no `lm_head.weight` or a lost `tie_word_embeddings` flag. For `.bin`-only repos it is not possible; say so.
4. Dry-run the transforms on the meta model: `QEFFAutoModelForCausalLM(model, pretrained_model_name_or_path="<org>/<model>")`. The attention modules must become QEff classes and there must be no `No transforms applied` warning. If the class is unmapped, you are in "new wrapper".
5. Size it: fp16 bytes = params x 2. One Ultra card is 4 devices of 32 GB. Up to about 8B fits one device; 70B needs 2 cards. MXFP6 weights take about a third of fp16.
6. When classifying many repos at once, run one agent per model with the checklist above (we did 30 at a time). Insist on verdicts with file:line evidence; config-only reasoning was wrong for 4 of 30 repos.

## Phase 2: read the upstream model code

For a new wrapper, read the whole modeling file before writing anything.

1. Get the files. In-tree classes: `transformers/models/<type>/modeling_<type>.py` in the venv. Remote code: download `modeling_*.py` and `configuration_*.py` from the repo.
   ```bash
   curl -sL https://huggingface.co/<org>/<model>/resolve/main/modeling_<type>.py -o modeling_<type>.py
   ```
2. Confirm the remote code imports resolve under the pinned transformers (newer repos import helpers that may not exist in 5.5.4):
   ```bash
   python -c "import transformers.utils.output_capturing, transformers.masking_utils"
   ```
3. List the classes: Attention, DecoderLayer, Model, ForCausalLM, the norm, the rotary embedding, the MLP, any MoE block. For each, write down what differs from Llama. For K2 Horizon: grouped RMSNorm (`layernorm_num_groups`), optional attention gate (`attention_gate_func`), optional QK norm, `rope_head_dim` (partial rotary), YaRN on one size, MoVA value experts and sparse MoE on the large sizes, 250k vocab, 524288 positions.
4. Pull the per-size config flags into a table (which size uses which feature). That decides what the wrapper must support now and what it can refuse.
5. Pick the closest existing wrapper to copy from, using `skills_studio/skills/qeff-model-onboarding/references/model-family-map.md` in the repo: Llama for dense GQA, Qwen3-MoE or GPT-OSS for MoE, Grok-1 for the remote-code wiring pattern, Gemma3 for hybrid cache.

## Phase 3: write the wrapper

Layout: `QEfficient/transformers/models/<family>/modeling_<family>.py` plus an empty `__init__.py` with the license header.

### In-tree class
Subclass the HF classes (`class QEffXAttention(XAttention)`) and map class to class in `KVCacheTransform._module_mapping`. Copy `models/llama/modeling_llama.py`.

### Remote-code class
You cannot import the class, so write plain `nn.Module` classes whose `forward` (and optional `__qeff_init__`) get bound onto the live modules by name:

1. Classes: `QEffXRMSNorm`, `QEffXAttention`, `QEffXDecoderLayer`, `QEffXModel`, `QEffXForCausalLM`. Each `forward` takes `self` and uses the attributes the upstream `__init__` created (`q_proj`, `layer_idx`, `num_key_value_groups`, `scaling`, ...). Check every attribute you use exists in the upstream `__init__`.
2. Register in `pytorch_transforms.py` under `KVCacheExternalModuleMapperTransform._match_string_replace_method`, keyed by upstream class name. Include `"__qeff_init__"` in the dict when you need it; it runs right after binding.
3. Add `"<X>Config": "QEFFAutoModelForCausalLM"` to `EXTERNAL_MODEL_CLASS_MAPPING` in `QEfficient/transformers/modeling_utils.py`.

### What goes in each forward (Llama pattern)
- Attention: project q, k, v; apply the model's QK norm if any (before the head split, as upstream does); `qeff_apply_rotary_pos_emb` with the `cos_cached`/`sin_cached` passed in; `past_key_value_update(module=self, ...)` for the KV cache; `eager_attention_forward` (boolean mask, True = masked); the gate if any; `o_proj`.
- Decoder layer: the two residual blocks; unpack a tuple if the MLP returns `(hidden, router_logits)`.
- Model `__qeff_init__`: build `cos_cached`/`sin_cached` as `nn.Parameter` from `rotary_emb.inv_freq` and `rotary_emb.attention_scaling` (this covers YaRN); cap the length with a constant in `QEfficient/utils/constants.py` (we used 65536; 524288 positions were 512 MB of ONNX); take the dtype from `self.embed_tokens.weight.dtype`, not `config.torch_dtype`.
- Model `forward`: `QEffDynamicCache.from_legacy_cache`, `_create_causal_mask(position_ids, target_length=past_seen_tokens)`, slice the tables with `position_ids`, loop the layers, final norm, return `BaseModelOutputWithPast`.
- CausalLM `forward`: pop kwargs the export glue may pass (`output_router_logits`, `logits_to_keep`, `labels`); select logits at `position_ids.argmax` (cast to int32 first); `get_submodules_for_export` returns the decoder layer class (for remote code: `type(self.model.layers[0])`).
- Norm: `select_interface(CustomRMSNormFunc.apply, torch.ops.qefficient.rms_norm)`. For a grouped norm: reshape to `(..., n_groups, group)`, run the op with a ones weight, reshape back, multiply by the full weight.

### Rules learned the hard way
- Refuse what you do not support, at transform time, in `__qeff_init__`: MoE/MoVA configs, partial rotary, sliding window. Otherwise the 36B loads 72 GB and dies in `forward` with a confusing `TypeError`.
- Do not copy the blocked-attention branch from Llama into a remote-code wrapper. `BlockingAttentionTransform` only tags in-tree classes, so the branch is dead code.
- Compute `F.softplus` (and anything else with no bf16 kernel) in float32: the bf16 compile lane fails with `Kernel lookup failed for: libjit_element_select_splat_kernel_f` otherwise. Qwen3.5's wrapper does the same.
- Keep every shape static and every reduction axis constant; no `x[..., :t.shape[-1]]`, no `.sum(keepdim=True)` on a runtime axis (subfunction export breaks).
- `pyproject.toml` is read-only. Run `ruff check` and `ruff format` on every changed file.

## Phase 4: CPU parity, three stages

The acceptance rule of the repo: HF PyTorch, QEff PyTorch and ONNX Runtime must produce the same tokens. Export success alone proves nothing.

1. **Tiny random model first** (`tools/k2_tiny_check.py`, adapt the config fields): shrink hidden size, layers, heads, keep the real vocab, turn on every optional feature (QK norm, gate, norm groups), build with `from_config(trust_remote_code=True)`, run all three stages. Fast and catches wiring errors. Must match.
2. **Real weights, reduced layers** (`tools/causal_lm_parity.py <model> --num-hidden-layers 2 --trust-remote-code`, then 8 layers). Matches are expected, but the text is usually one repeated token, so this is weak evidence on its own.
3. **Full model, PyTorch only, bf16** (`--dtype bfloat16 --skip-export`). Fits 9B in 31 GB RAM. Use a prompt with a known answer ("The capital of France is") and read the text. This is the stage that shows the wrapper is right on the real model.
4. The parity script frees the PyTorch model before the ONNX Runtime stage; without that the 8-layer run does not fit in RAM.
5. For vision-language models the repo's generic ORT runner does not know every family's inputs. We wrote `onboarding/vlm_parity.py` with a dual-ONNX loop that mirrors `kv_offload_generate`; reuse it, and remember `ApiRunnerVlm.setup_ort_session` for the INT32_MAX sentinel the ORT needs rewritten.

## Phase 5: tests, docs, review

1. **Dummy-layer test config** in `tests/configs/causal_model_configs.json`, `causal_lm_models` list: `model_name`, `model_type`, `additional_params` with the shrunk dims and the optional features switched on. Keep the real `vocab_size` (the prompt's token ids must exist).
2. **Per-PR entry** in `per_pr_causal_text_models`: `id`, `model_name`, `model_type`, `is_moe`, `supports_blocking`, `num_hidden_layers`, `config_overrides`, and `known_*` fields for lanes that cannot pass (we registered `known_speculative_export_or_compile_issue`; speculative decoding is class-keyed and raises for remote code).
3. **Remote code**: add the model to `ModelConfig.EXTERNAL_MODELS` in `QEfficient/utils/test_utils.py` so the tests pass `trust_remote_code`. The check helper skips the HF reference for entries that set `skip_hf_reference`; do not set it unless the model cannot run on CPU.
4. **Goldens**: the tests compare the card against a committed HF reference in `tests/golden_outputs/goldens.json`. Generate them on CPU:
   ```bash
   QEFF_REGENERATE_GOLDEN=1 pytest tests/transformers/models/causal_lm_models/test_causal_lm_models.py -k "<model> and dummy" -q
   QEFF_REGENERATE_GOLDEN=1 pytest tests/transformers/models/causal_lm_models/test_causal_lm_models.py -k "<per_pr_id> and not speculative" -q
   ```
   On a host without the SDK the tests fail at the compile step, after the golden is written; that is fine. Commit `goldens.json`. The variant keys must match the ones the server produces (they did for us: same config, same prompts).
5. **Docs**: add the row to `docs/source/validate.md` (footnote marker ② on the family cell for remote code), and a short section in `examples/text_generation/README.md` with the `basic_inference.py --trust-remote-code` command. Site-specific run guides do not go in the PR.
6. **Review before the PR**, with two independent readers: one following the repo's `skills_studio/skills/qeff-pr-reviewer/SKILL.md` checklist, one adversarial, line by line against the upstream code, allowed to run tiny CPU experiments. Our round found: no HF reference in the dummy test for external models, docs that could not ship, a dead code branch, the 512 MB rotary tables, an unreachable MoE failure, missing feature coverage in the test config. Fix all of it before the card.

## Phase 6: first run on the server

Full details for this server: [SERVER_SETUP.md](SERVER_SETUP.md) and [README.md](README.md).

1. `qaic-util -q`: cards present, SDK working, note idle `Board power(Watts)`.
2. The home directory is NFS with a quota. Put `HF_HOME`, `QEFF_HOME` and `QEFF_LOG_PATH` on the local disk (`/local/mnt/workspace/<user>`) and check with `touch`. The shell is csh: `setenv`, `activate.csh`, and quote device lists `'[0]'`.
3. Clone the fork branch, make the venv, `pip install -e ".[test]"`.
4. First run without MXFP6, so it is a precision check:
   ```csh
   python examples/text_generation/basic_inference.py --model-name <org>/<model> --trust-remote-code \
       --prompt "The capital of France is" --prefill-seq-len 128 --ctx-len 4096 --num-cores 16 --device-group '[0]'
   ```
   Export takes minutes, compile tens of minutes with nothing printed. Compare the text with a bf16 HF `generate()` of the same prompt; the first 20 tokens should agree.
5. Then the same with `--mxfp6`, and with `'[0,1,2,3]'`. Read the `Performance Stats` block each time.

## Phase 7: repo tests on the card

```csh
setenv QEFF_REGENERATE_GOLDEN 1
pytest tests/transformers/models/causal_lm_models/test_causal_lm_models.py -k "<model> and dummy" -v
pytest tests/transformers/models/causal_lm_models/test_causal_lm_models.py -k "<per_pr_id>" -v
```

- No `-n auto`: the conftest gives each worker its own card, and 64 workers on 4 cards errors at setup. Serial or `-n 4`.
- The bf16 per-PR lane compiles for `aic-hw-version=ai200`. On an AI 100-only host it fails for every model (Llama included) with the `libjit_element_select_splat_kernel_f` error. Prove that with the Llama lane once and move on; the upstream CI has the AI 200 toolchain.
- Expected: dummy 2 passed and 1 skipped (KV-repeat feature test), per-PR all passed except registered xfails and the bf16 lane.
- If the server cannot push (no GitHub credentials), regenerate the goldens on the laptop (Phase 5.4) instead of transcribing them.

## Phase 8: benchmark

[BENCHMARK.md](BENCHMARK.md) and `benchmark.py`: a matrix of devices, precision, prompt length, context length and batch size; each case compiles once, warms up, runs 3 timed repeats, samples board power, and appends a CSV row with TTFT, decode tok/s (aggregate and per stream), board watts and tok/s per W. Pass `--card-price-usd` for tok/s per dollar.

- Every device on a card reports the whole board's power; use the max, not the sum.
- A case that reaches the 150 W TDP cap may be throttled; rerun it with more repeats.
- Report every number with its settings and the SDK version (`qaic-version-util --apps`).

## Phase 9: pull request

1. Work on a fork, feature branch from upstream `main`. Keep two branches if you have site-specific docs: the PR branch (library code, tests, goldens, `validate.md`, a README section) and a docs branch rebased on it (run guides, server setup, benchmark). Only the first goes upstream.
2. Commits signed off (`git commit -s`), author set to the submitter. Message says what, not how.
3. The contribution rules: a human submits the PR, reviews every line, discloses AI assistance and lists the exact test commands in the description. Agents may prepare everything but must not open the PR.
4. PR description follows `.github/PULL_REQUEST_TEMPLATE/pr_template.md`: description, type of change, testing (CPU stages, card runs with numbers, test lanes with pass/fail and the reason for any failure), checklist, model card link. Open as draft until the card evidence is in.
5. Expect maintainer questions on: hash plumbing (none needed if no new graph-affecting kwargs), the validated-models table, test coverage of optional features, and anything site-specific that slipped into the diff.

## Checklist

- [ ] Hub facts recorded: class, remote code, transformers version, weight format, size, license, gated
- [ ] Checkpoint keys match the class (safetensors headers vs meta model)
- [ ] Transform dry run on a meta model applies QEff classes
- [ ] Upstream code read, per-size feature table written, imports resolve under the pinned transformers
- [ ] Wrapper written, registered, refuses unsupported configs at transform time
- [ ] Rotary tables capped, dtype from weights, no dead blocking branch, bf16-unsafe ops in float32
- [ ] Tiny random model: HF = QEff = ORT
- [ ] Real weights reduced layers: HF = QEff = ORT
- [ ] Full model bf16: HF = QEff, sensible text
- [ ] Dummy and per-PR test configs, EXTERNAL_MODELS entry, goldens committed
- [ ] validate.md row, README section, ruff clean
- [ ] Two reviews done, findings fixed
- [ ] Card: fp16 output matches HF; MXFP6 1 and 4 devices run
- [ ] Repo tests on the card pass (bf16 lane explained)
- [ ] Benchmark CSV with Perf/Watt, SDK version noted
- [ ] PR opened by a human with the full testing section
