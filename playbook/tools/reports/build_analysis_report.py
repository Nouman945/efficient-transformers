"""Builds the MBZUAI onboarding analysis report (HTML, then PDF through headless Chrome)."""

import html
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
CATALOG = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "mbzuai_catalog.json"
RESULTS = HERE.parent / "results"
OUT_HTML = HERE / "report.html"
OUT_PDF = ROOT / "MBZUAI_AI100_Ultra_Onboarding_Analysis.pdf"

READY, WRAPPER, CONVERT, BLOCKED = (
    "Ready",
    "New wrapper",
    "Convert first",
    "Not onboardable",
)
READY_TYPES = {"gpt2", "qwen3", "qwen3_vl", "qwen3_vl_moe", "qwen2_5_vl", "gemma3"}
WRAPPER_TYPES = {"t5", "gpt_neo", "speecht5", "swiftformer", "qwen2_vl"}

FAMILY = {
    "gpt2": "GPT-2",
    "qwen3": "Qwen3",
    "qwen3_vl": "Qwen3-VL",
    "qwen3_vl_moe": "Qwen3-VL MoE",
    "qwen2_5_vl": "Qwen2.5-VL",
    "gemma3": "Gemma 3",
    "llama": "Llama",
    "llava": "LLaVA",
    "t5": "T5",
    "gpt_neo": "GPT-Neo",
    "speecht5": "SpeechT5",
    "swiftformer": "SwiftFormer",
    "qwen2_vl": "Qwen2-VL",
}


def classify(row):
    name, mtype = row["id"].split("/")[1], row.get("model_type")
    arch = (row.get("arch") or [None])[0]
    if row.get("adapter"):
        return BLOCKED, "LoRA adapter only, needs merging into its base model"
    if not row.get("has_cfg"):
        if "GGUF" in name:
            return BLOCKED, "GGUF build, the safetensors repo is the one to onboard"
        if name.endswith("onnx"):
            return BLOCKED, "Pre-exported ONNX, not a QEfficient export"
        return BLOCKED, "No transformers config, original research checkpoint"
    if mtype is None:
        return BLOCKED, "Custom pipeline, no transformers model class"
    if mtype in READY_TYPES or (mtype == "llama" and arch in ("LlamaForCausalLM", None)):
        return READY, FAMILY[mtype]
    if mtype == "llava" and arch == "LlavaForConditionalGeneration":
        return READY, "LLaVA"
    if mtype in WRAPPER_TYPES:
        return WRAPPER, FAMILY[mtype]
    if arch == "MobiLlamaForCausalLM":
        return WRAPPER, "MobiLlama"
    return CONVERT, arch or mtype


def size_of(row):
    if row.get("params"):
        return row["params"]
    match = re.search(r"(\d+(?:\.\d+)?)([BbM])(?![a-z0-9])", row["id"].split("/")[1].replace("_", "-"))
    if not match:
        return None
    digits, unit = match.groups()
    value = float(digits) / 10 if digits.startswith("0") and "." not in digits else float(digits)
    return value * (1e6 if unit == "M" else 1e9)


def fmt_size(params):
    if not params:
        return "-"
    return f"{params / 1e9:.1f}B" if params >= 1e9 else f"{params / 1e6:.0f}M"


def esc(text):
    return html.escape(str(text))


def table(headers, rows, widths=None):
    cols = "".join(f'<col style="width:{w}">' for w in widths) if widths else ""
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><colgroup>{cols}</colgroup><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def pill(text, kind):
    return f'<span class="pill {kind}">{esc(text)}</span>'


catalog = json.loads(CATALOG.read_text())
for row in catalog:
    row["tier"], row["note"] = classify(row)
    row["size"] = size_of(row)
    row["name"] = row["id"].split("/")[1]
counts = {t: sum(r["tier"] == t for r in catalog) for t in (READY, WRAPPER, CONVERT, BLOCKED)}
results = {json.loads(p.read_text())["model"]: json.loads(p.read_text()) for p in RESULTS.glob("*.json")}

# Tier 1 grouped by family
ready_groups = {}
for row in catalog:
    if row["tier"] == READY:
        ready_groups.setdefault(row["note"], []).append(row)
READY_NOTES = {
    "GPT-2": (
        "QEFFAutoModelForCausalLM",
        "All 7 pass the local token match at full size.",
    ),
    "Qwen3": (
        "QEFFAutoModelForCausalLM",
        "Passes the local token match on the first 2 of 36 layers.",
    ),
    "Llama": (
        "QEFFAutoModelForCausalLM",
        "Same class as the validated Llama 2/3 list. The 70B needs more than one card.",
    ),
    "Qwen3-VL": (
        "QEFFAutoModelForImageTextToText",
        "MediX-R1-2B exports to ONNX. ONNX Runtime check still open.",
    ),
    "Qwen3-VL MoE": (
        "QEFFAutoModelForImageTextToText",
        "Same class as the validated Qwen3-VL-30B-A3B.",
    ),
    "Qwen2.5-VL": (
        "QEFFAutoModelForImageTextToText",
        "Image path is validated in the library. Video input is not checked yet.",
    ),
    "LLaVA": (
        "QEFFAutoModelForImageTextToText",
        "HF-format LLaVA 1.5 with a Llama 3 text model.",
    ),
    "Gemma 3": (
        "QEFFAutoModelForImageTextToText",
        "Gated repo. Needs a Hugging Face login with access.",
    ),
}
ready_rows = []
for family in READY_NOTES:
    rows = sorted(ready_groups.get(family, []), key=lambda r: r["size"] or 0)
    names = ", ".join(r["name"] for r in rows)
    sizes = sorted(r["size"] for r in rows if r["size"])
    span = fmt_size(sizes[0]) if len(sizes) == 1 else f"{fmt_size(sizes[0])} to {fmt_size(sizes[-1])}" if sizes else "-"
    ready_rows.append(
        [
            f"<b>{esc(family)}</b>",
            len(rows),
            esc(span),
            f'<span class="names">{esc(names)}</span>',
            esc(READY_NOTES[family][1]),
        ]
    )

WRAPPER_NOTES = {
    "Qwen2-VL": (
        "Small",
        "Close to the existing Qwen2.5-VL wrapper. Best first target, AIN is a flagship Arabic model.",
    ),
    "GPT-Neo": (
        "Small",
        "Plain decoder with learned positions. Follows the GPT-2 wrapper pattern.",
    ),
    "MobiLlama": (
        "Small",
        "Llama with one FFN shared across layers, loaded through remote code.",
    ),
    "T5": (
        "Medium",
        "The library has T5 only as a text encoder for diffusion. Seq2seq generation needs an encoder plus a cached decoder flow.",
    ),
    "SpeechT5": (
        "Large",
        "Encoder-decoder speech model. TTS also needs the HiFi-GAN vocoder. Whisper is the nearest reference.",
    ),
    "SwiftFormer": (
        "Medium",
        "Small vision classifier. No image classification auto class in the library, plain ONNX plus qaic-compile may be simpler.",
    ),
}
wrapper_groups = {}
for row in catalog:
    if row["tier"] == WRAPPER:
        wrapper_groups.setdefault(row["note"], []).append(row)
wrapper_rows = [
    [
        f"<b>{esc(f)}</b>",
        len(wrapper_groups[f]),
        f"{sum(r['dl'] or 0 for r in wrapper_groups[f]):,}",
        esc(WRAPPER_NOTES[f][0]),
        esc(WRAPPER_NOTES[f][1]),
    ]
    for f in WRAPPER_NOTES
]

CONVERT_GROUPS = [
    (
        "LLaVA-style Llama 3 / Phi-3 (original LLaVA repo format)",
        {
            "LlavaLlamaForCausalLM",
            "LlavaPhiForCausalLM",
            "Phi3ForCausalLM",
            "LlamaForCausalLM",
        },
        "Convert the checkpoint to the HF LlavaForConditionalGeneration layout, then it runs on the existing LLaVA wrapper. BiMediX2-8B-hf shows this works.",
    ),
    (
        "GeoChat, PALO, ViMUL, BiMediX2-4B",
        {
            "GeoChatLlamaForCausalLM",
            "PaloForCausalLM",
            "LlavaQwenForCausalLM",
            "Phi3VForCausalLM",
        },
        "LLaVA variants with their own model code. Need conversion plus a check of the projector and vision tower.",
    ),
    (
        "GLaMM and GeoPixel (pixel grounding)",
        {"GLaMMForCausalLM", "LISAForCausalLM", "GeoPixelForCausalLM"},
        "Carry a SAM-style mask decoder on top of the LLM. Text side could run, segmentation output is out of scope for QEfficient today.",
    ),
    (
        "Omni-Embed",
        {"OmniEmbedForEmbedding"},
        "Gated, remote code, multimodal embedding. Needs access before it can be judged.",
    ),
]
convert_rows = []
for label, archs, note in CONVERT_GROUPS:
    rows = [r for r in catalog if r["tier"] == CONVERT and (r.get("arch") or [None])[0] in archs]
    convert_rows.append(
        [
            f"<b>{esc(label)}</b>",
            len(rows),
            f'<span class="names">{esc(", ".join(r["name"] for r in rows))}</span>',
            esc(note),
        ]
    )

blocked_groups = {}
for row in catalog:
    if row["tier"] == BLOCKED:
        blocked_groups.setdefault(row["note"], []).append(row)
blocked_rows = [
    [
        esc(note),
        len(rows),
        f'<span class="names">{esc(", ".join(r["name"] for r in rows))}</span>',
    ]
    for note, rows in sorted(blocked_groups.items(), key=lambda kv: -len(kv[1]))
]

result_rows = []
for model in sorted(results, key=lambda m: results[m]["params"]):
    res = results[model]
    scope = "Full model" if res.get("num_hidden_layers", -1) == -1 else f"First {res['num_hidden_layers']} layers"
    ok = res["hf_vs_kv_match"] and res.get("hf_vs_ort_match")
    result_rows.append(
        [
            f"<b>{esc(model.split('/')[1])}</b>",
            esc(res["architecture"]),
            esc(scope),
            f"{res['compared_tokens']} / {res['compared_tokens']}",
            pill("Match", "ok") if res["hf_vs_kv_match"] else pill("Mismatch", "bad"),
            pill("Match", "ok") if res.get("hf_vs_ort_match") else pill("Mismatch", "bad"),
            pill("Pass", "ok") if ok else pill("Fail", "bad"),
        ]
    )
result_rows.append(
    [
        "<b>MediX-R1-2B</b>",
        "Qwen3VLForConditionalGeneration",
        "Full model",
        "-",
        "-",
        pill("Open", "warn"),
        pill("Export only", "warn"),
    ]
)
passed = sum(1 for r in results.values() if r["hf_vs_kv_match"] and r.get("hf_vs_ort_match"))

TIER_KIND = {READY: "ok", WRAPPER: "info", CONVERT: "warn", BLOCKED: "bad"}
appendix_rows = [
    [
        esc(r["name"]),
        esc((r.get("arch") or ["-"])[0] or "-"),
        fmt_size(r["size"]),
        f"{r['dl'] or 0:,}",
        pill(r["tier"], TIER_KIND[r["tier"]]),
    ]
    for r in sorted(
        catalog,
        key=lambda r: (
            (READY, WRAPPER, CONVERT, BLOCKED).index(r["tier"]),
            -(r["dl"] or 0),
        ),
    )
]

total = len(catalog)
bar = "".join(
    f'<div class="seg {TIER_KIND[t]}" style="width:{counts[t] / total * 100:.2f}%">{counts[t]}</div>' for t in counts
)

CSS = """
@page { size: A4; margin: 16mm 14mm 16mm 14mm; }
:root { --blue:#3253DC; --navy:#0B1F4B; --ink:#1B2230; --grey:#5B6472; --line:#D9DEE8; --tint:#EEF2FD; --red:#E4002B;
        --ok:#0E7C4A; --okbg:#E3F4EB; --warn:#9A5B00; --warnbg:#FFF1D6; --badbg:#FDE3E7; }
* { box-sizing: border-box; }
body { font-family: "Inter","Helvetica Neue",Arial,sans-serif; color: var(--ink); font-size: 9.6pt; line-height: 1.42; margin: 0;
       -webkit-print-color-adjust: exact; print-color-adjust: exact; }
.cover { background: var(--navy); color: #fff; padding: 26px 28px 22px; border-radius: 6px; border-top: 6px solid var(--red); }
.cover .kicker { color: #AFC0FF; font-size: 8.6pt; letter-spacing: .14em; text-transform: uppercase; font-weight: 600; }
.cover h1 { font-size: 21pt; line-height: 1.15; margin: 8px 0 8px; font-weight: 700; }
.cover p { margin: 0; color: #D6DEF5; font-size: 9.6pt; }
.cover .meta { margin-top: 14px; font-size: 8.4pt; color: #AFC0FF; }
h2 { font-size: 13pt; color: var(--navy); margin: 22px 0 8px; padding-left: 9px; border-left: 4px solid var(--blue); break-after: avoid; }
h3 { font-size: 10.4pt; color: var(--blue); margin: 14px 0 5px; break-after: avoid; }
p { margin: 5px 0; } ul { margin: 5px 0 5px 0; padding-left: 16px; } li { margin: 2.5px 0; }
.tiles { display: grid; grid-template-columns: repeat(4, 1fr); gap: 9px; margin: 14px 0 10px; }
.tile { border: 1px solid var(--line); border-radius: 6px; padding: 10px 11px; border-top: 4px solid var(--blue); break-inside: avoid; }
.tile.ok { border-top-color: var(--ok); } .tile.info { border-top-color: var(--blue); }
.tile.warn { border-top-color: #E39B12; } .tile.bad { border-top-color: var(--red); }
.tile .n { font-size: 21pt; font-weight: 700; color: var(--navy); line-height: 1; }
.tile .t { font-weight: 700; margin-top: 5px; font-size: 9.2pt; } .tile .d { color: var(--grey); font-size: 8.2pt; margin-top: 2px; }
.bar { display: flex; height: 20px; border-radius: 4px; overflow: hidden; margin: 4px 0 4px; font-size: 8pt; font-weight: 700; color: #fff; }
.seg { display: flex; align-items: center; justify-content: center; }
.seg.ok { background: var(--ok); } .seg.info { background: var(--blue); } .seg.warn { background: #E39B12; } .seg.bad { background: var(--red); }
.callout { background: var(--tint); border-left: 4px solid var(--blue); padding: 9px 12px; border-radius: 0 5px 5px 0; margin: 10px 0; break-inside: avoid; }
.callout.risk { background: #FFF4F5; border-left-color: var(--red); }
.callout b.h { color: var(--navy); display: block; margin-bottom: 3px; }
table { width: 100%; border-collapse: collapse; margin: 7px 0 10px; font-size: 8.5pt; table-layout: fixed; }
th { background: var(--navy); color: #fff; text-align: left; padding: 5px 7px; font-weight: 600; font-size: 8.2pt; }
td { padding: 5px 7px; border-bottom: 1px solid var(--line); vertical-align: top; overflow-wrap: anywhere; }
tbody tr:nth-child(even) td { background: #F7F9FD; } tr { break-inside: avoid; } thead { display: table-header-group; }
.names { color: var(--grey); font-size: 8pt; }
.pill { display: inline-block; padding: 1px 7px; border-radius: 9px; font-size: 7.6pt; font-weight: 700; white-space: nowrap; }
.pill.ok { background: var(--okbg); color: var(--ok); } .pill.info { background: var(--tint); color: var(--blue); }
.pill.warn { background: var(--warnbg); color: var(--warn); } .pill.bad { background: var(--badbg); color: #B00020; }
.flow { display: grid; grid-template-columns: repeat(4, 1fr); gap: 7px; margin: 8px 0; }
.step { border: 1px solid var(--line); border-radius: 6px; padding: 8px 9px; break-inside: avoid; }
.step .k { color: var(--blue); font-weight: 700; font-size: 8pt; letter-spacing: .06em; text-transform: uppercase; }
.step .t { font-weight: 700; margin: 2px 0; } .step .d { color: var(--grey); font-size: 8.2pt; }
.step.local { border-top: 4px solid var(--ok); } .step.server { border-top: 4px solid var(--blue); }
pre { background: #0F1A36; color: #E6ECFF; padding: 9px 11px; border-radius: 5px; font-size: 7.8pt; line-height: 1.45;
      white-space: pre-wrap; overflow-wrap: anywhere; margin: 6px 0 10px; break-inside: avoid; font-family: "JetBrains Mono","DejaVu Sans Mono",monospace; }
code { font-family: "JetBrains Mono","DejaVu Sans Mono",monospace; font-size: 8.4pt; background: #F0F3FA; padding: 0 3px; border-radius: 3px; }
.foot { color: var(--grey); font-size: 7.8pt; margin-top: 16px; border-top: 1px solid var(--line); padding-top: 6px; }
.pb { break-before: page; }
"""

HTML = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>MBZUAI Models on Cloud AI 100 Ultra: Onboarding Analysis</title><style>{CSS}</style></head><body>

<div class="cover">
  <div class="kicker">Cloud AI 100 Ultra &middot; Model Onboarding Analysis</div>
  <h1>Mohamed Bin Zayed University of Artificial Intelligence (MBZUAI) Hugging Face models</h1>
  <p>Which of the {total} public MBZUAI repos can run on Qualcomm Cloud AI 100 Ultra through the efficient-transformers (QEfficient) library, what was checked so far, and what is left.</p>
  <div class="meta">5 October 2026 &nbsp;|&nbsp; Source: huggingface.co/MBZUAI &nbsp;|&nbsp; Library: quic/efficient-transformers 1.23.0.dev0, transformers 5.5.4</div>
</div>

<h2>Summary</h2>
<div class="tiles">
  <div class="tile ok"><div class="n">{counts[READY]}</div><div class="t">Ready</div><div class="d">Architecture already supported. No library code needed.</div></div>
  <div class="tile info"><div class="n">{counts[WRAPPER]}</div><div class="t">New wrapper</div><div class="d">In transformers, but QEfficient has no wrapper yet.</div></div>
  <div class="tile warn"><div class="n">{counts[CONVERT]}</div><div class="t">Convert first</div><div class="d">Custom LLaVA-style code. Needs checkpoint conversion or more.</div></div>
  <div class="tile bad"><div class="n">{counts[BLOCKED]}</div><div class="t">Not onboardable</div><div class="d">Adapters, GGUF, or raw research checkpoints.</div></div>
</div>
<div class="bar">{bar}</div>
<ul>
  <li><b>{counts[READY]} of {total} repos ({counts[READY] * 100 // total}%) map to an architecture the library already supports.</b> They need export, compile and validation only.</li>
  <li><b>{passed} models pass the local check so far.</b> Hugging Face PyTorch, QEfficient PyTorch and ONNX Runtime give the same tokens.</li>
  <li><b>MediX-R1-2B (Qwen3-VL) exports to ONNX.</b> Its ONNX Runtime check is not finished yet.</li>
  <li><b>{counts[WRAPPER]} repos need a new wrapper.</b> Qwen2-VL (AIN), GPT-Neo and MobiLlama are the small ones. T5 and SpeechT5 are bigger jobs.</li>
</ul>
<div class="callout risk"><b class="h">Open risks, read these first</b>
<ul>
  <li><b>Nothing has run on a card yet.</b> All results here are from a laptop with no AI 100 hardware or SDK. Compile and on-device token match are still to be done on the Qualcomm server.</li>
  <li><b>Large models were not checked at full size.</b> The laptop has 31 GB RAM, so anything above about 2B parameters is checked with reduced layers or not at all.</li>
  <li><b>4 repos are gated</b> (Llama-3-Nanda-10B-Chat, dialseg-ar-gemma3-4B, Omni-Embed-Mini 0.9B and 2.3B). They need a Hugging Face account with access.</li>
  <li><b>Video models are only image-validated.</b> Video-R2 and Video-CoM use the Qwen2.5-VL class, and the library examples cover image input.</li>
</ul></div>

<h2>How onboarding works in QEfficient</h2>
<p>The library takes a Hugging Face model, swaps its attention and cache modules for AI 100 friendly ones, exports to ONNX, and compiles a QPC (Qualcomm Program Container, the binary the card runs). A model counts as onboarded only when the tokens match at every stage.</p>
<div class="flow">
  <div class="step local"><div class="k">Stage 1 &middot; laptop</div><div class="t">HF PyTorch</div><div class="d">Reference tokens from the original model.</div></div>
  <div class="step local"><div class="k">Stage 2 &middot; laptop</div><div class="t">QEff PyTorch</div><div class="d">After the KV cache transforms. Tokens must match stage 1.</div></div>
  <div class="step local"><div class="k">Stage 3 &middot; laptop</div><div class="t">ONNX Runtime</div><div class="d">After ONNX export. Tokens must match stage 1.</div></div>
  <div class="step server"><div class="k">Stage 4 &middot; server</div><div class="t">Cloud AI 100 Ultra</div><div class="d">Compile to QPC and run on the card. Tokens must match stage 1.</div></div>
</div>
<ul>
  <li><b>Supported architecture:</b> load with the matching auto class (<code>QEFFAutoModelForCausalLM</code>, <code>QEFFAutoModelForImageTextToText</code>), then <code>export()</code>, <code>compile()</code>, <code>generate()</code>.</li>
  <li><b>New architecture:</b> add a wrapper under <code>QEfficient/transformers/models/&lt;family&gt;/</code>, register it in <code>pytorch_transforms.py</code>, wire the auto class in <code>modeling_auto.py</code>, add tests. The library pins transformers 5.5.4, so the model class must exist in that version.</li>
  <li><b>Vision-language models:</b> exported as two QPCs, one for the vision encoder and one for the language model (<code>kv_offload=True</code>).</li>
</ul>

<h2>Tier 1: Ready ({counts[READY]} repos)</h2>
<p>These use a model class that is already on the library's validated list. No code change is expected.</p>
{table(["Family", "Repos", "Size", "Models", "Status"], ready_rows, ["13%", "7%", "12%", "38%", "30%"])}

<h3>Local validation results</h3>
<p>Greedy decode, prompt "My name is", prefill 8, context 32, 24 generated tokens compared. The Qwen3 model was loaded with 2 layers because the full 4B model does not fit in RAM next to its ONNX export.</p>
{table(["Model", "Class", "Scope", "Tokens", "HF vs QEff", "HF vs ORT", "Result"], result_rows, ["19%", "25%", "13%", "9%", "11%", "11%", "12%"])}
<div class="callout"><b class="h">What "pass" means here</b>Stages 1 to 3 agree token for token. It does not cover the compiler, fp16 precision on the card, or performance. Those come from stage 4.</div>

<h2>Tier 2: New wrapper needed ({counts[WRAPPER]} repos)</h2>
<p>The model class exists in transformers, but QEfficient has no wrapper for it. Effort is a rough estimate.</p>
{table(["Family", "Repos", "Downloads", "Effort", "Notes"], wrapper_rows, ["13%", "7%", "11%", "9%", "60%"])}

<h2>Tier 3: Convert first ({counts[CONVERT]} repos)</h2>
<p>These were saved from research codebases (original LLaVA, GLaMM, GeoChat, PALO). The weights are fine, but the layout and model class are not what transformers loads natively.</p>
{table(["Group", "Repos", "Models", "Path"], convert_rows, ["22%", "7%", "36%", "35%"])}

<h2>Tier 4: Not onboardable as published ({counts[BLOCKED]} repos)</h2>
{table(["Reason", "Repos", "Models"], blocked_rows, ["38%", "8%", "54%"])}

<h2>Stage 4 on the Qualcomm server</h2>
<p>These commands follow the library docs and examples. They have not been run on a card for these models yet. Compile settings (context length, cores, devices) are starting points.</p>
<h3>Text models (GPT-2, Qwen3, Llama families)</h3>
<pre>source /opt/qti-aic/dev/python/qeff/bin/activate   # or a venv with efficient-transformers installed

python -m QEfficient.cloud.infer --model_name MBZUAI/LaMini-GPT-124M \\
  --batch_size 1 --prompt_len 32 --ctx_len 512 --mxfp6 --num_cores 16 \\
  --device_group [0] --prompt "My name is" --mos 1 --aic_enable_depth_first

python -m QEfficient.cloud.infer --model_name MBZUAI/OpenEarthAgent \\
  --batch_size 1 --prompt_len 128 --ctx_len 4096 --mxfp6 --num_cores 16 \\
  --device_group [0] --prompt "My name is" --mos 1 --aic_enable_depth_first</pre>
<h3>Vision-language models (Qwen3-VL family: MediX-R1, MedMO)</h3>
<p>Use <code>examples/image_text_to_text/models/qwen3vl/qwen3_vl.py</code> with <code>model_id</code> set to the MBZUAI repo, for example <code>MBZUAI/MediX-R1-2B</code>, and <code>num_devices</code> set to the cards available. For MediX-R1-30B use the <code>qwen3_vl_moe</code> example folder.</p>
<h3>Card memory, rough guide</h3>
<ul>
  <li>fp16 weights take about 2 GB per billion parameters. One AI 100 Ultra card has 128 GB, seen by the SDK as 4 devices of 32 GB each.</li>
  <li>Models up to about 8B parameters fit on one device. Larger ones are split across devices with <code>num_devices</code>.</li>
  <li>MediX-R1-30B (about 62 GB of weights) should fit on one card using 3 to 4 devices.</li>
  <li>BiMediX2-70B and BiMediX_3-1_70B_LLM (about 140 GB) need at least 2 cards.</li>
</ul>

<h2>Recommended next steps</h2>
<ul>
  <li><b>1. Run stage 4 for the 8 passed models</b> on the server and record token match plus tokens per second.</li>
  <li><b>2. Finish the Qwen3-VL ONNX Runtime check</b> on MediX-R1-2B, then repeat for MediX-R1-8B and MedMO on a larger host.</li>
  <li><b>3. Cover the rest of Tier 1</b> (Qwen2.5-VL, LLaVA, Llama 7B to 70B) on the server, since the laptop cannot hold them.</li>
  <li><b>4. Start Tier 2 with Qwen2-VL (AIN)</b>, then GPT-Neo and MobiLlama. Decide on T5 and SpeechT5 after that.</li>
  <li><b>5. Get Hugging Face access</b> for the 4 gated repos.</li>
</ul>

<h2 class="pb">Appendix: all {total} repos</h2>
<p>Sorted by tier, then by downloads. Size comes from the Hub when published, otherwise from the repo name.</p>
{table(["Repo (MBZUAI/...)", "Model class", "Size", "Downloads", "Tier"], appendix_rows, ["31%", "31%", "8%", "11%", "19%"])}

<div class="foot">Catalog pulled from the Hugging Face Hub API on 5 October 2026. Local checks ran on CPU with QEfficient 1.23.0.dev0, transformers 5.5.4, torch 2.7.0, onnxruntime 1.22. Colors are a Qualcomm-style theme only. This is a working analysis, not an official Qualcomm document.</div>
</body></html>"""

OUT_HTML.write_text(HTML)
subprocess.run(
    [
        "google-chrome",
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={OUT_PDF}",
        OUT_HTML.as_uri(),
    ],
    check=True,
    capture_output=True,
)
print(counts, "passed:", passed, "->", OUT_PDF)
