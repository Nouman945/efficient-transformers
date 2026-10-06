"""Builds the effort-ranked MBZUAI onboarding plan (HTML, then PDF through headless Chrome)."""

import html
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
OUT_HTML = HERE / "ranked_report.html"
OUT_PDF = ROOT / "MBZUAI_AI100_Ultra_Onboarding_Ranked_Plan.pdf"

catalog = {r["id"].split("/")[1]: r for r in json.loads((HERE / "mbzuai_catalog.json").read_text())}

EFFORT = {
    "XS": ("XS", "under 1 day", "ok"),
    "S": ("S", "1 to 3 days", "ok"),
    "M": ("M", "about 1 week", "info"),
    "L": ("L", "about 2 weeks", "warn"),
    "XL": ("XL", "3 weeks or more", "bad"),
}

# (work item, effort, models, why it sits at this rank, what is left to do)
FAST = [
    (
        "LaMini GPT-2 family",
        "XS",
        [
            "LaMini-Cerebras-111M",
            "LaMini-GPT-124M",
            "LaMini-Cerebras-256M",
            "LaMini-Cerebras-590M",
            "LaMini-GPT-774M",
            "LaMini-Cerebras-1.3B",
            "LaMini-GPT-1.5B",
        ],
        "All 7 pass the local token match at full size. Small, one device each.",
        "Compile and run on the card. One infer command per model.",
    ),
    (
        "OpenEarthAgent (Qwen3, 4B)",
        "XS",
        ["OpenEarthAgent"],
        "Passes the local token match on the first 2 of 36 layers. Plain Qwen3 text model.",
        "Compile and run on the card at full size.",
    ),
    (
        "MediX-R1-2B (Qwen3-VL)",
        "S",
        ["MediX-R1-2B"],
        "Passes at full size with a real image. ONNX Runtime gives the same 16 tokens as Hugging Face.",
        "Compile vision and language QPCs, run the image example.",
    ),
    (
        "Qwen3-VL 4B models",
        "S",
        ["MedMO-4B", "MedMO-4B-Next", "ParallelTubeDecoding-Qwen3-VL-4B"],
        "Same class as MediX-R1-2B. ParallelTubeDecoding passes a reduced-size local check.",
        "Full-size export, compile and run on the server.",
    ),
    (
        "Qwen3-VL 8B models",
        "S",
        ["MediX-R1-8B", "MedMO-8B", "MedMO-8B-Next"],
        "Same class again, only bigger. Not checked locally, too large for the laptop.",
        "Full-size export, compile and run on the server.",
    ),
    (
        "Bactrian-X Llama 7B and 13B",
        "S",
        ["bactrian-x-llama-7b-merged", "bactrian-x-llama-13b-merged"],
        "Plain LlamaForCausalLM, the most validated class in the library. Old 2023 checkpoints, not checked locally.",
        "Export, compile, run. 13B needs 2 devices.",
    ),
    (
        "BiMediX2-8B-hf (LLaVA)",
        "S",
        ["BiMediX2-8B-hf"],
        "HF-format LLaVA 1.5 with a Llama 3 text model. Class is validated, this checkpoint is not checked yet.",
        "Export both QPCs, compile, run the image example.",
    ),
    (
        "Gated repos",
        "S",
        ["Llama-3-Nanda-10B-Chat", "dialseg-ar-gemma3-4B"],
        "Llama and Gemma 3 classes are supported. Blocked only on Hugging Face access.",
        "Get access approved, then the normal flow.",
    ),
    (
        "Video-R2 and Video-CoM (Qwen2.5-VL, 7B)",
        "M",
        ["Video-R2", "Video-R2-SFT", "Video-CoM", "Video-CoM-SFT"],
        "Class is validated for images. These are video reasoning models, and video input is not covered by the library examples.",
        "Image path first. Then validate multi-frame video input, which is their real use.",
    ),
    (
        "MediX-R1-30B (Qwen3-VL MoE)",
        "M",
        ["MediX-R1-30B"],
        "Same class as the validated Qwen3-VL-30B-A3B. About 62 GB of weights, so 3 to 4 devices and a long compile.",
        "Multi-device compile, then run. Follow the qwen3_vl_moe examples.",
    ),
    (
        "BiMediX 3.1 70B LLM (Llama)",
        "M",
        ["BiMediX_3-1_70B_LLM"],
        "Plain Llama 3.1 70B layout, a validated size. About 140 GB of weights, so at least 2 cards.",
        "Multi-card compile and run. Mostly machine time.",
    ),
]

WRAPPER = [
    (
        "GPT-Neo (LaMini-Neo)",
        "S",
        ["LaMini-Neo-125M", "LaMini-Neo-1.3B"],
        "Plain decoder with learned positions, very close to GPT-2. Smallest wrapper of the set.",
        "Attention and block wrapper with KV cache, register transforms, add tests.",
    ),
    (
        "Qwen2-VL (AIN)",
        "M",
        ["AIN"],
        "The Qwen2.5-VL wrapper is the template. Vision tower differs slightly. AIN is the flagship Arabic model, so value is high.",
        "Encoder and decoder wrappers, auto class wiring, image tests.",
    ),
    (
        "MobiLlama",
        "M",
        ["MobiLlama-05B", "MobiLlama-08B", "MobiLlama-1B", "MobiLlama-05B-Chat", "MobiLlama-1B-Chat"],
        "Llama layers with one FFN shared across all blocks. Loaded through remote code, so the class mapping is manual.",
        "Map the remote-code modules to the QEff Llama attention, check the shared FFN exports once, add tests.",
    ),
    (
        "SwiftFormer",
        "M",
        ["swiftformer-xs", "swiftformer-s", "swiftformer-l1", "swiftformer-l3"],
        "Small image classifier, no KV cache. The library has no image classification auto class.",
        "Either a thin new auto class, or skip QEfficient and compile a plain ONNX export with qaic-compile.",
    ),
    (
        "T5 seq2seq (LaMini-T5, LaMini-Flan-T5)",
        "L",
        [
            "LaMini-T5-61M",
            "LaMini-T5-223M",
            "LaMini-T5-738M",
            "LaMini-Flan-T5-77M",
            "LaMini-Flan-T5-248M",
            "LaMini-Flan-T5-783M",
        ],
        "The library has T5 only as a text encoder for diffusion. Text generation needs an encoder QPC plus a decoder QPC with self and cross attention caches.",
        "New seq2seq flow: wrappers, relative position bias in the cached decoder, generate loop, tests.",
    ),
    (
        "SpeechT5 speech-to-text (ArTST ASR)",
        "L",
        ["artst_asr", "artst_asr_v2", "artst_asr_v2_qasr", "artst_asr_v3", "artst_asr_v3_qasr"],
        "Encoder-decoder with a convolutional speech front end. Whisper is the nearest reference in the library.",
        "Encoder and cached decoder wrappers, audio pre-processing, ASR generate loop, tests.",
    ),
    (
        "SpeechT5 text-to-speech (ClArTTS)",
        "XL",
        ["speecht5_tts_clartts_ar"],
        "Needs everything the ASR item needs, plus a spectrogram decoder loop with a stop threshold and the HiFi-GAN vocoder.",
        "Three graphs to compile (encoder, decoder, vocoder) and a new TTS runtime loop.",
    ),
]

PROPER = [
    (
        "DFJ judge models (Qwen2.5-VL)",
        "M",
        [
            "Qwen-2.5-VL-Instruct-3B-Pointwise-DFJ",
            "Qwen-2.5-VL-Instruct-3B-Pairwise-DFJ",
            "Qwen-2.5-VL-Instruct-7B-Pointwise-DFJ",
            "Qwen-2.5-VL-Instruct-7B-Pairwise-DFJ",
        ],
        "Config says Qwen2.5-VL, but the checkpoint has no lm_head weight and an older key layout. No model card. Loads with a randomly initialised output head.",
        "Find out how the judge is meant to be loaded and scored. Then it may drop into the fast group.",
    ),
    (
        "LLaVA-format Llama 3 and Phi-3",
        "M",
        [
            "LLaVA-Meta-Llama-3-8B-Instruct",
            "LLaVA-Meta-Llama-3-8B-Instruct-FT",
            "LLaVA-Meta-Llama-3-8B-Instruct-FT-S2",
            "LLaVA-Meta-Llama-3-8B-Instruct-pretrain",
            "BiMediX2-8B",
            "BiMediX2-8B-Bi",
            "BiMediX2-70B",
            "LLaVA-Phi-3-mini-4k-instruct",
            "LLaVA-Phi-3-mini-4k-instruct-FT",
            "LLaVA-Phi-3-mini-4k-instruct-pretrain",
        ],
        "Saved from the original LLaVA codebase. Weights are standard CLIP plus projector plus LLM. BiMediX2-8B-hf shows the converted form works.",
        "Write one converter to the HF LlavaForConditionalGeneration layout, verify outputs, then reuse the LLaVA wrapper. The S2 variant and the two pretrain repos need extra care.",
    ),
    (
        "GeoChat, PALO, ViMUL, BiMediX2-4B",
        "L",
        ["geochat-7B", "PALO-7B", "PALO-13B", "MobilePALO-1.7B", "ViMUL", "BiMediX2-4B"],
        "LLaVA variants with their own model code: different projectors, a SigLIP tower in ViMUL, a Phi-3-V layout in BiMediX2-4B.",
        "Per-model conversion and parity check against the original repo code. Some may need a small wrapper change.",
    ),
    (
        "Omni-Embed-Mini",
        "L",
        ["Omni-Embed-Mini-0.9B", "Omni-Embed-Mini-2.3B"],
        "Gated and remote code. Multimodal embedding model, architecture not visible without access.",
        "Get access, read the model code, then size the work properly.",
    ),
    (
        "GLaMM and GeoPixel (pixel grounding)",
        "XL",
        [
            "GLaMM-GranD-Pretrained",
            "GLaMM-FullScope",
            "GLaMM-FullScope_v0",
            "GLaMM-RefSeg",
            "GLaMM-GCG",
            "GLaMM-RegCap-VG",
            "GLaMM-RegCap-RefCOCOg",
            "GeoPixel-7B",
            "GeoPixel-7B-RES",
        ],
        "An LLM plus a SAM-style mask decoder driven by special tokens. GeoPixel is the most downloaded MBZUAI repo, so demand is real.",
        "Custom model code, a second vision encoder, mask decoder export, and a runtime that passes LLM hidden states to the decoder. New pipeline, not a wrapper.",
    ),
]

GROUPS = [
    (
        "A",
        "Fast onboarding",
        "Architecture already supported. No library code. Work is export, compile and validate.",
        FAST,
        "ok",
    ),
    (
        "B",
        "Needs a wrapper",
        "Model class exists in transformers 5.5.4. QEfficient needs new wrapper code and tests.",
        WRAPPER,
        "info",
    ),
    (
        "C",
        "Needs proper onboarding",
        "Custom checkpoints or code. Needs conversion, investigation or a new pipeline.",
        PROPER,
        "warn",
    ),
]

listed = [m for _, _, _, rows, _ in GROUPS for row in rows for m in row[2]]
missing = [m for m in listed if m not in catalog]
assert not missing, missing
assert len(listed) == len(set(listed)) == 81, len(listed)


def esc(text):
    return html.escape(str(text))


def pill(text, kind):
    return f'<span class="pill {kind}">{esc(text)}</span>'


def group_table(letter, rows):
    body = ""
    for rank, (item, effort, models, why, todo) in enumerate(rows, 1):
        label, days, kind = EFFORT[effort]
        downloads = sum(catalog[m].get("dl") or 0 for m in models)
        body += (
            f'<tr><td class="rank">{letter}{rank}</td>'
            f"<td><b>{esc(item)}</b><div class='names'>{len(models)} repo{'s' if len(models) > 1 else ''}: {esc(', '.join(models))}</div></td>"
            f"<td>{pill(label, kind)}<div class='days'>{esc(days)}</div></td>"
            f"<td>{downloads:,}</td><td>{esc(why)}</td><td>{esc(todo)}</td></tr>"
        )
    return (
        '<table><colgroup><col style="width:6%"><col style="width:27%"><col style="width:10%"><col style="width:9%">'
        '<col style="width:24%"><col style="width:24%"></colgroup><thead><tr><th>Rank</th><th>Work item</th><th>Effort</th>'
        f"<th>Downloads</th><th>Why this rank</th><th>What is left</th></tr></thead><tbody>{body}</tbody></table>"
    )


tiles = "".join(
    f'<div class="tile {kind}"><div class="n">{sum(len(r[2]) for r in rows)}</div><div class="t">{letter}. {esc(title)}</div>'
    f'<div class="d">{esc(desc)}</div></div>'
    for letter, title, desc, rows, kind in GROUPS
)
sections = "".join(
    f"<h2>Group {letter}: {esc(title)} ({sum(len(r[2]) for r in rows)} repos)</h2><p>{esc(desc)} Rank 1 is the least effort.</p>"
    + group_table(letter, rows)
    for letter, title, desc, rows, _ in GROUPS
)
legend = "".join(f"<span class='lg'>{pill(label, kind)} {esc(days)}</span>" for label, days, kind in EFFORT.values())

CSS = """
@page { size: A4 landscape; margin: 13mm 13mm 14mm 13mm; }
:root { --blue:#3253DC; --navy:#0B1F4B; --ink:#1B2230; --grey:#5B6472; --line:#D9DEE8; --tint:#EEF2FD; --red:#E4002B;
        --ok:#0E7C4A; --okbg:#E3F4EB; --warn:#9A5B00; --warnbg:#FFF1D6; --badbg:#FDE3E7; }
* { box-sizing: border-box; }
body { font-family: "Inter","Helvetica Neue",Arial,sans-serif; color: var(--ink); font-size: 9.4pt; line-height: 1.4; margin: 0;
       -webkit-print-color-adjust: exact; print-color-adjust: exact; }
.cover { background: var(--navy); color: #fff; padding: 20px 26px 18px; border-radius: 6px; border-top: 6px solid var(--red); }
.cover .kicker { color: #AFC0FF; font-size: 8.4pt; letter-spacing: .14em; text-transform: uppercase; font-weight: 600; }
.cover h1 { font-size: 19pt; line-height: 1.15; margin: 6px 0 6px; }
.cover p { margin: 0; color: #D6DEF5; } .cover .meta { margin-top: 10px; font-size: 8.2pt; color: #AFC0FF; }
h2 { font-size: 12.5pt; color: var(--navy); margin: 18px 0 5px; padding-left: 9px; border-left: 4px solid var(--blue); break-after: avoid; }
p { margin: 4px 0; } ul { margin: 4px 0; padding-left: 16px; } li { margin: 2px 0; }
.tiles { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 12px 0 8px; }
.tile { border: 1px solid var(--line); border-radius: 6px; padding: 9px 12px; border-top: 4px solid var(--blue); }
.tile.ok { border-top-color: var(--ok); } .tile.warn { border-top-color: #E39B12; }
.tile .n { font-size: 20pt; font-weight: 700; color: var(--navy); line-height: 1; }
.tile .t { font-weight: 700; margin-top: 4px; } .tile .d { color: var(--grey); font-size: 8.3pt; margin-top: 2px; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.callout { background: var(--tint); border-left: 4px solid var(--blue); padding: 8px 12px; border-radius: 0 5px 5px 0; margin: 8px 0; break-inside: avoid; }
.callout.risk { background: #FFF4F5; border-left-color: var(--red); }
.callout b.h { color: var(--navy); display: block; margin-bottom: 2px; }
table { width: 100%; border-collapse: collapse; margin: 6px 0 8px; font-size: 8.4pt; table-layout: fixed; }
th { background: var(--navy); color: #fff; text-align: left; padding: 5px 7px; font-weight: 600; font-size: 8.1pt; }
td { padding: 5px 7px; border-bottom: 1px solid var(--line); vertical-align: top; overflow-wrap: anywhere; }
tbody tr:nth-child(even) td { background: #F7F9FD; } tr { break-inside: avoid; } thead { display: table-header-group; }
td.rank { font-weight: 700; color: var(--blue); font-size: 10pt; }
.names { color: var(--grey); font-size: 7.7pt; margin-top: 2px; } .days { color: var(--grey); font-size: 7.6pt; margin-top: 2px; }
.pill { display: inline-block; padding: 1px 8px; border-radius: 9px; font-size: 7.8pt; font-weight: 700; white-space: nowrap; }
.pill.ok { background: var(--okbg); color: var(--ok); } .pill.info { background: var(--tint); color: var(--blue); }
.pill.warn { background: var(--warnbg); color: var(--warn); } .pill.bad { background: var(--badbg); color: #B00020; }
.legend { margin: 6px 0 2px; font-size: 8.3pt; color: var(--grey); } .lg { margin-right: 14px; white-space: nowrap; }
.foot { color: var(--grey); font-size: 7.6pt; margin-top: 12px; border-top: 1px solid var(--line); padding-top: 5px; }
"""

HTML = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>MBZUAI Models on Cloud AI 100 Ultra: Effort-Ranked Onboarding Plan</title><style>{CSS}</style></head><body>
<div class="cover">
  <div class="kicker">Cloud AI 100 Ultra &middot; Effort-Ranked Onboarding Plan</div>
  <h1>Mohamed Bin Zayed University of Artificial Intelligence (MBZUAI) Hugging Face models</h1>
  <p>The 81 onboardable MBZUAI repos, split into three groups and ranked by effort inside each group.</p>
  <div class="meta">5 October 2026 &nbsp;|&nbsp; Library: quic/efficient-transformers 1.23.0.dev0, transformers 5.5.4 &nbsp;|&nbsp; Companion to the Onboarding Analysis report</div>
</div>
<div class="tiles">{tiles}</div>
<div class="legend"><b>Effort, one engineer, per work item:</b> &nbsp; {legend}</div>
<div class="cols">
  <div class="callout"><b class="h">How to read the ranking</b>
    <ul><li>Effort is engineering time for the whole work item, not per repo. Fine-tunes of one architecture share the work.</li>
    <li>Downloads are Hugging Face Hub counts for the last month, summed over the item. Use them to weigh value against effort.</li>
    <li>28 more repos are left out: LoRA adapters, GGUF builds and raw research checkpoints. See the Onboarding Analysis report.</li></ul></div>
  <div class="callout risk"><b class="h">Read before planning</b>
    <ul><li><b>Effort numbers are estimates</b> from reading the library and the model configs. None of them is measured.</li>
    <li><b>Nothing has run on a card yet.</b> Local checks stop at ONNX Runtime. Group A effort assumes compile works first time.</li>
    <li><b>The 4 DFJ models moved from "ready" to Group C.</b> Their checkpoints have no output head weight, found after the first report.</li></ul></div>
</div>
{sections}
<h2>Suggested order</h2>
<ul>
  <li><b>Week 1:</b> A1 to A7 on the server, and request access for A8. That is 18 repos with no code change, 10 of them already checked locally.</li>
  <li><b>Then in parallel:</b> machine-heavy items A9 to A11, and the first wrappers B1 (GPT-Neo) and B2 (AIN).</li>
  <li><b>Next:</b> C2, the LLaVA converter. One converter unlocks 10 repos, including BiMediX2-8B and BiMediX2-70B.</li>
  <li><b>Decide later:</b> T5, SpeechT5, and GLaMM or GeoPixel. These are new flows, so they need a clear customer ask first.</li>
</ul>
<div class="foot">Catalog pulled from the Hugging Face Hub API on 5 October 2026. Local checks ran on CPU with QEfficient 1.23.0.dev0, transformers 5.5.4, torch 2.7.0. Colors are a Qualcomm-style theme only. This is a working analysis, not an official Qualcomm document.</div>
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
print({title: sum(len(r[2]) for r in rows) for _, title, _, rows, _ in GROUPS}, "->", OUT_PDF)
