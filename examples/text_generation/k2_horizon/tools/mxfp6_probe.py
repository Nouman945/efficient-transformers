"""
Simulate MXFP6 (E2M3 elements, 32-element blocks with a shared power-of-two scale, OCP MX spec)
on the matmul weights of a model and see which weights make greedy decoding go wrong.

    python mxfp6_probe.py IFM/K2-Horizon-0.9B                 # all linear weights quantized
    python mxfp6_probe.py IFM/K2-Horizon-0.9B --only mlp      # only gate/up/down
    python mxfp6_probe.py IFM/K2-Horizon-0.9B --skip-layers 0,1,2,3
    python mxfp6_probe.py IFM/K2-Horizon-0.9B --stats         # per-weight block dynamic range, no generation
"""

import argparse
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BLOCK = 32
E2M3_MAX = 7.5
E2M3_EMAX = 2  # floor(log2(7.5))


def e2m3_round(x):
    """Round values already divided by the block scale to the nearest E2M3 number."""
    ax = x.abs()
    exp = torch.floor(torch.log2(ax.clamp_min(1e-30))).clamp(min=0, max=E2M3_EMAX)
    step = torch.pow(2.0, exp - 3)  # 3 mantissa bits; subnormals share the step of exponent 0
    q = torch.round(ax / step) * step
    return torch.sign(x) * q.clamp(max=E2M3_MAX)


def mxfp6(weight):
    """Fake-quantize a [out, in] weight along the input dim in blocks of 32."""
    out_f, in_f = weight.shape
    pad = (-in_f) % BLOCK
    w = torch.nn.functional.pad(weight.float(), (0, pad)).view(out_f, -1, BLOCK)
    amax = w.abs().amax(dim=-1, keepdim=True)
    scale = torch.pow(2.0, torch.floor(torch.log2(amax.clamp_min(1e-30))) - E2M3_EMAX)
    q = e2m3_round(w / scale) * scale
    return q.view(out_f, -1)[:, :in_f].to(weight.dtype)


def linear_names(model):
    return [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--new-tokens", type=int, default=40)
    parser.add_argument("--only", choices=["mlp", "attn", "lm_head"], help="quantize only this group")
    parser.add_argument("--skip-layers", default="", help="comma list of layer indices to leave in full precision")
    parser.add_argument("--skip", default="", help="regex of module names to leave in full precision")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--ppl", action="store_true", help="also report perplexity on a fixed paragraph")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, dtype=torch.float32).eval()
    names = linear_names(model)

    if args.stats:
        rows = []
        for n in names:
            w = dict(model.named_modules())[n].weight.detach().float()
            out_f, in_f = w.shape
            pad = (-in_f) % BLOCK
            blocks = torch.nn.functional.pad(w, (0, pad)).view(out_f, -1, BLOCK)
            amax = blocks.abs().amax(-1)
            rms = blocks.pow(2).mean(-1).sqrt().clamp_min(1e-12)
            err = (mxfp6(w) - w).pow(2).mean().sqrt() / w.pow(2).mean().sqrt()
            rows.append((float(err), float((amax / rms).max()), float(w.abs().max()), n))
        for err, ratio, wmax, n in sorted(rows, reverse=True)[:25]:
            print(f"rel_err={err:.4f}  max(blockmax/blockrms)={ratio:7.1f}  |w|max={wmax:8.3f}  {n}")
        return

    inputs = tok(args.prompt, return_tensors="pt")
    ppl_text = (
        "The Eiffel Tower was completed in 1889 for the World's Fair held in Paris. It was designed by the "
        "engineering firm of Gustave Eiffel and stands about 330 metres tall. For four decades it was the tallest "
        "man-made structure in the world. Today it is one of the most visited monuments, with millions of visitors "
        "each year who climb its stairs or take the lifts to the observation decks."
    )
    ppl_ids = tok(ppl_text, return_tensors="pt")["input_ids"]

    def perplexity():
        with torch.no_grad():
            loss = model(input_ids=ppl_ids, labels=ppl_ids).loss
        return float(torch.exp(loss))

    with torch.no_grad():
        ref = model.generate(**inputs, max_new_tokens=args.new_tokens, do_sample=False)[
            0, inputs["input_ids"].shape[1] :
        ]
    print("fp32   :", repr(tok.decode(ref)))
    if args.ppl:
        print(f"fp32 perplexity: {perplexity():.3f}")

    skip_layers = {int(x) for x in args.skip_layers.split(",") if x}
    chosen = []
    for n in names:
        m = re.search(r"layers\.(\d+)\.", n)
        layer = int(m.group(1)) if m else None
        if args.only == "mlp" and ".mlp." not in n:
            continue
        if args.only == "attn" and ".self_attn." not in n:
            continue
        if args.only == "lm_head" and n != "lm_head":
            continue
        if layer in skip_layers or (args.skip and re.search(args.skip, n)):
            continue
        chosen.append(n)
    modules = dict(model.named_modules())
    with torch.no_grad():
        for n in chosen:
            modules[n].weight.copy_(mxfp6(modules[n].weight))
        out = model.generate(**inputs, max_new_tokens=args.new_tokens, do_sample=False)[
            0, inputs["input_ids"].shape[1] :
        ]
    n_match = int((out[: len(ref)] == ref[: len(out)]).cumprod(0).sum())
    print(f"mxfp6 on {len(chosen)} weights:", repr(tok.decode(out)))
    print(f"first {n_match} of {len(ref)} tokens match the fp32 run")
    if args.ppl:
        print(f"mxfp6 perplexity: {perplexity():.3f}")


if __name__ == "__main__":
    main()
