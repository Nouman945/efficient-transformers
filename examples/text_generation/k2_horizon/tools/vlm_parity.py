"""
Local onboarding stages for one image-text-to-text Hub model (dual QPC, kv_offload=True):
HF PyTorch reference -> ONNX export of vision encoder + language decoder -> ONNX Runtime token match.
Compile and on-device runs are done on the Qualcomm server.

Usage: python vlm_parity.py MBZUAI/MediX-R1-2B [--reduced] [--prompt-len 128] [--ctx-len 4096]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.utils.run_utils import ApiRunnerVlm

RESULTS_DIR = Path(__file__).parent / "results"
IMG_URL = "https://picsum.photos/id/237/536/354"


def run_dual_onnx_on_ort(onnx_paths, inputs, grid_thw, prefill_seq_len, ctx_len, gen_len, pad_token_id):
    """Greedy generation on the exported encoder + decoder ONNX, mirroring the dual QPC generate() loop."""
    import onnxruntime

    encoder_path, decoder_path = [str(p) for p in onnx_paths]
    inputs = {k: np.array(v) for k, v in inputs.items()}

    encoder = onnxruntime.InferenceSession(encoder_path)
    t, h, w = grid_thw
    # The encoder reads the grid from the shape of image_grid_thw, not from its values.
    vision_feed = {
        "pixel_values": inputs["pixel_values"],
        "image_grid_thw": np.zeros((1, t, h, w), dtype=np.int64),
    }
    vision_out_names = [o.name for o in encoder.get_outputs()]
    state = dict(zip(vision_out_names, encoder.run(vision_out_names, vision_feed)))
    del encoder

    # The decoder marks unused cache slots with INT32_MAX, which the AI 100 ops skip but ORT rejects.
    # setup_ort_session rewrites those constants to 0; the initializers must stay alive with the session.
    added_initializers, decoder = ApiRunnerVlm.setup_ort_session(None, decoder_path)
    out_names = [o.name for o in decoder.get_outputs()]
    for inp in decoder.get_inputs():
        if inp.name.startswith("past_"):
            state[inp.name] = np.zeros((1, inp.shape[1], ctx_len, inp.shape[3]), dtype=np.float32)
    state["image_idx"] = np.array([[0]], dtype=np.int64)

    def step(input_ids, position_ids):
        outputs = dict(
            zip(
                out_names,
                decoder.run(
                    out_names,
                    {**state, "input_ids": input_ids, "position_ids": position_ids},
                ),
            )
        )
        for name, value in outputs.items():
            if name.endswith("_RetainedState"):
                state[name[: -len("_RetainedState")]] = value
        state["image_idx"] = outputs["image_idx_output"]
        return outputs["logits"]

    position_ids = inputs["position_ids"]
    padded_len = position_ids.shape[-1]
    input_ids = np.pad(
        inputs["input_ids"],
        ((0, 0), (0, padded_len - inputs["input_ids"].shape[1])),
        constant_values=pad_token_id,
    )
    for start in range(0, padded_len, prefill_seq_len):
        logits = step(
            input_ids[:, start : start + prefill_seq_len],
            position_ids[..., start : start + prefill_seq_len],
        )

    generated = []
    next_ids = logits.argmax(2)[:, -1:]
    next_pos = position_ids.max(axis=-1, keepdims=True) + 1
    generated.append(int(next_ids[0, 0]))
    for _ in range(1, gen_len):
        next_ids = step(next_ids, next_pos).argmax(2)
        next_pos = next_pos + 1
        generated.append(int(next_ids[0, 0]))
    del added_initializers
    return np.array(generated)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name")
    parser.add_argument("--query", default="Can you describe the image in detail.")
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--ctx-len", type=int, default=1024)
    parser.add_argument("--gen-len", type=int, default=16)
    parser.add_argument(
        "--reduced",
        action="store_true",
        help="2 text layers (and 9 vision blocks on Qwen3-VL) with real weights, for models too big for this host",
    )
    parser.add_argument("--skip-ort", action="store_true", help="Export only, skip the ONNX Runtime leg")
    args = parser.parse_args()

    torch.manual_seed(42)
    config = AutoConfig.from_pretrained(args.model_name)
    if args.reduced:
        config.text_config.num_hidden_layers = 2
        if hasattr(config.vision_config, "deepstack_visual_indexes"):
            config.vision_config.depth = 9
            config.vision_config.deepstack_visual_indexes = [8]

    processor = AutoProcessor.from_pretrained(args.model_name, padding=True)
    image = Image.open(requests.get(IMG_URL, stream=True, timeout=30).raw).convert("RGB")
    conversation = [
        {
            "role": "user",
            "content": [{"type": "text", "text": args.query}, {"type": "image"}],
        }
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    model_hf = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        config=config,
        low_cpu_mem_usage=False,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model_hf.eval()
    result = {
        "model": args.model_name,
        "architecture": model_hf.__class__.__name__,
        "params": sum(p.numel() for p in model_hf.parameters()),
        "reduced": args.reduced,
        "prompt_len": args.prompt_len,
        "ctx_len": args.ctx_len,
    }

    inputs = processor(images=image, text=prompt, return_tensors="pt")
    with torch.no_grad():
        # min_new_tokens keeps HF from stopping at EOS so all gen_len tokens are compared.
        output = model_hf.generate(
            **inputs,
            max_new_tokens=args.gen_len,
            min_new_tokens=args.gen_len,
            do_sample=False,
        )
    hf_tokens = output[0, inputs["input_ids"].shape[1] :].numpy()
    result["hf_tokens"] = hf_tokens.tolist()
    result["hf_text"] = processor.tokenizer.decode(hf_tokens)
    del model_hf

    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        args.model_name,
        kv_offload=True,
        config=config,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    start = time.time()
    onnx_paths = qeff_model.export()
    result["onnx_paths"] = [str(p) for p in onnx_paths]
    result["export_seconds"] = round(time.time() - start, 1)

    if not args.skip_ort:
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        # Some wrappers drop image_grid_thw in prepare_inputs_for_generation, so read it first.
        grid_thw = [int(x) for x in inputs["image_grid_thw"][0]]
        inputs = qeff_model.model.prepare_inputs_for_generation(
            inputs=inputs, prefill_seq_len=args.prompt_len, batch_size=1
        )
        ort_tokens = run_dual_onnx_on_ort(
            onnx_paths,
            inputs,
            grid_thw,
            args.prompt_len,
            args.ctx_len,
            args.gen_len,
            processor.tokenizer.pad_token_id,
        )
        n = min(len(hf_tokens), len(ort_tokens))
        result["ort_text"] = processor.tokenizer.decode(ort_tokens)
        result["compared_tokens"] = n
        result["hf_vs_ort_match"] = bool(n > 0 and (hf_tokens[:n] == ort_tokens[:n]).all())

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / (args.model_name.replace("/", "__") + ("__reduced" if args.reduced else "") + ".json")
    out.write_text(json.dumps(result, indent=2))
    print("RESULT", json.dumps({k: v for k, v in result.items() if k != "hf_tokens"}))
    sys.exit(0 if all(v for k, v in result.items() if k.endswith("_match")) else 1)


if __name__ == "__main__":
    main()
