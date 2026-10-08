**Description**

Adds support for the dense K2 Horizon models (`IFM/K2-Horizon-0.9B`, `3.7B`, `7B`, `32B`, class `K2HorizonForCausalLM`, remote code). The dense sizes are Llama-style GQA decoders with a grouped RMSNorm and an optional post-attention gate, so the wrapper follows the Llama one and is wired by class name through `KVCacheExternalModuleMapperTransform`.

- `QEfficient/transformers/models/k2_horizon/modeling_k2_horizon.py`: KV cache attention (optional silu/softplus gate and QK norm), grouped RMSNorm on `CustomRMSNorm`, static rotary tables, model and causal LM forwards.
- `pytorch_transforms.py`: name-based mapping for `K2HorizonForCausalLM`, `K2HorizonModel`, `K2HorizonDecoderLayer`, `K2HorizonAttention`, `K2HorizonRMSNorm`.
- `modeling_utils.py`: `K2HorizonConfig` in `EXTERNAL_MODEL_CLASS_MAPPING`.
- `tests/configs/causal_model_configs.json`, `QEfficient/utils/test_utils.py`: dummy-layer config (QK norm and gate enabled) for the causal LM test, plus a per-PR entry `k2_horizon_text`.
- `check_causal_models.py`: external models run the HF reference leg unless their entry sets `skip_hf_reference` (grok-1 keeps the skip). Without this the K2 dummy test had no reference tokens.
- `constants.py`: `K2_HORIZON_MAX_POSITION_EMBEDDINGS = 65536` caps the baked rotary tables (configs declare 524288).
- `examples/text_generation/README.md`: run command with `basic_inference.py --trust-remote-code`.

Not covered, refused at transform time with `NotImplementedError`: the MoVA/MoE sizes (`36B-A4B`, `375B-A23B`), partial rotary (`rope_head_dim != head_dim`), sliding window. Blocking is not supported for this remote-code class (the blocking transform is class-keyed), and speculative decoding raises through `SpDTransform` as for other external models. Weight-free export does not work for remote-code models because `_build_meta_model` does not pass `trust_remote_code` (pre-existing, same for grok-1).

**Related Issue**

None.

**Type of Change**

- New feature (non-breaking change which adds functionality)

**Testing**

On CPU:

- Tiny random `K2HorizonForCausalLM` (2 layers, 4 heads, 2 KV heads, 4 norm groups): HF PyTorch, QEff PyTorch and ONNX Runtime give the same 16 tokens.
- `IFM/K2-Horizon-7B`, first 2 and first 8 layers with real weights, fp32: HF PyTorch, QEff PyTorch and ONNX Runtime give the same 24 tokens.
- `IFM/K2-Horizon-7B`, all 36 layers, bf16: HF PyTorch and QEff PyTorch give the same 24 tokens for "The capital of France is" (" Paris. The capital of Germany is Berlin. ...").
- `ruff check` and `ruff format --check` clean.

On AI 100 Ultra (one card, SDK on an AI 100-only host):

- `IFM/K2-Horizon-7B`, fp16, 1 device, prompt 128, ctx 4096: card output matches the CPU reference word for word. TTFT 0.2 s, decode 6.2 tokens/s.
- Same with MXFP6 weights and MXINT8 KV cache: 1 device 14.4 tokens/s; 4 devices TTFT 0.05 s, 51.8 tokens/s.
- `pytest tests/transformers/models/causal_lm_models/test_causal_lm_models.py -k "K2-Horizon and dummy"`: 2 passed (HF vs QEff PyTorch vs ONNX Runtime vs AI 100, and the CB variant), 1 skipped (KV-repeat feature test, not applicable).
- `pytest ... -k "k2_horizon_text"`: 4 passed (fp16 subfunction CB, prefix caching, CCL, fp32-export/fp16-compile CCL), 1 xfail (speculative decoding, registered), 1 failed: `test_per_pr_causal_bf16_subfunction_cb_ccl_compile_only`. That lane compiles for `aic-hw-version=ai200` and fails on this host for every model (`llama_text` fails identically with `Kernel lookup failed for: libjit_element_select_splat_kernel_f`), so it needs the CI's AI 200 toolchain. The softplus gate is computed in float32 to keep that lane clean of a bf16 Softplus.
- Goldens for the dummy and per-PR variants are committed in `tests/golden_outputs/goldens.json`.
- `IFM/K2-Horizon-3.7B`: CPU parity (HF PyTorch, QEff PyTorch, ONNX Runtime) passes on the full model; fp16 card output matches the CPU reference; MXFP6 + MXINT8 output is unchanged. Decode 23.3 tokens/s on 1 device, 65.6 on 4 devices, 375 aggregate at `full_batch_size=16`.
- `IFM/K2-Horizon-0.9B`: CPU parity passes; fp16 + MXINT8 card output matches the CPU reference; dummy causal LM tests 2 passed. Decode 45 tokens/s on 1 device, 131 on 4 devices, 1032 aggregate at `full_batch_size=16`. MXFP6 weights degrade this size's output (the first four layers are the sensitive ones in a CPU MXFP6 simulation), so `validate.md` lists it without MXFP6.
- `IFM/K2-Horizon-7B` benchmark with MXFP6 + MXINT8 on 4 devices: 48.6 tokens/s single stream, 456 aggregate at `full_batch_size=16`.

The earlier commands, for reference:

```bash
pytest -n auto tests/transformers/models/causal_lm_models/test_causal_lm_models.py -k "K2-Horizon and dummy"
python -m QEfficient.cloud.infer --model_name IFM/K2-Horizon-7B --trust_remote_code --batch_size 1 --prompt_len 128 --ctx_len 4096 --num_cores 16 --device_group [0] --prompt "The capital of France is" --mos 1 --aic_enable_depth_first
```

**Checklist**

- [x] My code follows the style guidelines of this project
- [x] I have performed a self-review of my own code
- [x] I have commented my code, particularly in hard-to-understand areas
- [x] I have made corresponding changes to the documentation
- [x] My changes generate no new warnings
- [x] I have added tests that prove my fix is effective or that my feature works
- [x] New and existing unit tests pass locally with my changes (on an AI 100 Ultra host; the bf16/AI 200 lane needs CI)
- [x] Any dependent changes have been merged and published in downstream modules

**Additional Context**

AI assistance was used for the wrapper, tests and run guide; I reviewed every line and ran the validation above. The CPU parity checks were run through small parity scripts outside the repo (HF vs QEff PyTorch vs ONNX Runtime token match using `QEfficient.utils.run_utils.ApiRunner`).

Model cards: https://huggingface.co/IFM/K2-Horizon-7B
