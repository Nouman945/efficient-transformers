"""
Fetch every public model of a Hugging Face org with the facts needed to classify it,
without downloading weights. Writes <org>_catalog.json and prints one line per repo.

    python catalog_fetch.py MBZUAI
"""

import concurrent.futures as cf
import json
import sys
import urllib.request


def fetch_json(url, timeout=30):
    return json.load(urllib.request.urlopen(url, timeout=timeout))


def describe(model):
    mid = model["id"]
    out = {"id": mid, "dl": model.get("downloads"), "tag": model.get("pipeline_tag")}
    try:
        info = fetch_json(f"https://huggingface.co/api/models/{mid}")
        out["gated"] = info.get("gated")
        files = [s["rfilename"] for s in info.get("siblings", [])]
        out["has_cfg"] = "config.json" in files
        out["adapter"] = "adapter_config.json" in files
        out["safetensors"] = any(f.endswith(".safetensors") for f in files)
        cfg = info.get("config") or {}
        out["model_type"] = cfg.get("model_type")
        out["arch"] = cfg.get("architectures")
        out["auto_map"] = bool(cfg.get("auto_map"))
        out["params"] = (info.get("safetensors") or {}).get("total")
        if out["has_cfg"]:
            cj = fetch_json(f"https://huggingface.co/{mid}/resolve/main/config.json")
            out["tv"] = cj.get("transformers_version")
            text = cj.get("text_config") or {}
            out["text_type"] = text.get("model_type")
            out["vision"] = cj.get("mm_vision_tower") or (cj.get("vision_config") or {}).get("model_type")
    except Exception as exc:  # gated repos return 401 here; record it and move on
        out["err"] = str(exc)[:80]
    return out


def main():
    org = sys.argv[1]
    models = fetch_json(f"https://huggingface.co/api/models?author={org}&limit=500&sort=downloads&direction=-1")
    with cf.ThreadPoolExecutor(16) as pool:
        rows = list(pool.map(describe, models))
    json.dump(rows, open(f"{org}_catalog.json", "w"), indent=1)
    for r in rows:
        size = f"{r['params'] / 1e9:.2f}B" if r.get("params") else "-"
        flags = [
            f
            for f, on in (
                ("gated", r.get("gated")),
                ("remote_code", r.get("auto_map")),
                ("adapter", r.get("adapter")),
                ("no_config", not r.get("has_cfg")),
            )
            if on
        ]
        print(r["id"], r.get("model_type"), r.get("arch"), size, " ".join(flags), r.get("err", ""), sep=" | ")
    print(f"\n{len(rows)} repos -> {org}_catalog.json")


if __name__ == "__main__":
    main()
