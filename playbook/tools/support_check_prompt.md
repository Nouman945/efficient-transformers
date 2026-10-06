# Per-model support check: agent prompt template

Run one agent per model with this prompt (we used 30 in parallel through a workflow). Fill in MODEL, FAMILY, AUTO_CLASS and KNOWN. Ask for structured output with the fields listed at the end.

---

You are verifying whether ONE Hugging Face model is really supported by the QEfficient library (quic/efficient-transformers) for Qualcomm Cloud AI 100. Be skeptical: an earlier pass assumed support from config.json alone and was wrong for some repos (their checkpoints had no lm_head weight). Confirm or refute support with evidence.

MODEL: {MODEL}
Assumed family: {FAMILY}
Assumed QEff auto class: {AUTO_CLASS}
What is already known: {KNOWN}

ENVIRONMENT
- Library checkout: {CHECKOUT}. Read-only. Do not edit anything.
- Before any python command run: source {ENV_SH}
- Put helper scripts in {SCRATCH}/{MODEL_SLUG}/ only.
- No AI 100 card here; little RAM; other agents run at the same time.

HARD RULES
- Do NOT download model weights. Do not call from_pretrained on the model. Config, tokenizer and processor files are fine.
- Do NOT run ONNX export, compile, generation or pytest.
- Build models only on the meta device.
- If the repo is gated or returns 401/403, report verdict "blocked" with what the public API shows.

CHECKS, in order
1. Hub metadata: https://huggingface.co/api/models/{MODEL}?blobs=true and config.json. Record architectures, model_type, transformers_version, gated, auto_map (remote code), file list and sizes, weight format, tokenizer / chat template / preprocessor presence.
2. Library mapping: find where this exact HF class is wired (QEfficient/transformers/modeling_utils.py, QEfficient/transformers/models/pytorch_transforms.py, QEfficient/transformers/models/modeling_auto.py, the family folder). Give file:line evidence and the auto class. Check docs/source/validate.md and tests/configs/*.json. Not mapped means not_supported.
3. Config features: load with AutoConfig and compare with the reference model the library validates for this family. Flag vocab changes, tie_word_embeddings, sliding window, rope settings, head counts, quantization_config, vision tower settings, a transformers_version newer than the pin, missing chat template.
4. Checkpoint keys versus the class, without weights: HfApi().get_safetensors_metadata for tensor names and shapes; build the class from config under torch.device("meta"); compare with model.state_dict(). Account for tied lm_head, buffers and legacy key renames (check the class's key mapping first). Report missing, unexpected and shape mismatches. .bin-only repos: not_checkable, say why.
5. Transform dry run on the meta model with the QEff auto class the tests use. Confirm attention modules became QEff classes and no "No transforms applied" warning. If wrapping fails only because of the meta device, mark not_run.
6. Sizing: params in billions, fp16 GB (params x 2), number of 32 GB devices with 20 percent headroom (one Ultra card exposes 4).

VERDICT: supported / supported_with_caveats / not_supported / blocked. Never "supported" on config evidence alone. List the exact commands run. Summary in two or three plain sentences.

OUTPUT FIELDS
model, verdict, confidence, hf_class, qeff_auto_class, qeff_wrapper_evidence, checkpoint_check (keys_match | keys_mismatch | not_checkable | not_run), checkpoint_detail, transform_dry_run (passed | failed | not_run), transform_detail, params_billion, fp16_weight_gb, min_devices_32gb, caveats[], blockers[], commands_run[], summary
