#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inference for SD3.5 with text-only learned T5 soft token embeddings.
- Loads SD3.5 pipeline
- Injects learned embeddings into tokenizer_3 / text_encoder_3 (T5-XXL)
- Generates vanilla vs defended images for comparison
"""

import os
import argparse
import csv
import json
from typing import List

import torch
from safetensors.torch import load_file
from PIL import Image

from diffusers import StableDiffusion3Pipeline


def insert_placeholder(prompt: str, placeholder_str: str, position: str) -> str:
    p = (prompt or "").strip()
    if position == "prefix":
        return f"{placeholder_str} {p}".strip()
    if position == "suffix":
        return f"{p} {placeholder_str}".strip()
    if position == "after_first":
        parts = p.split()
        if len(parts) <= 1:
            return f"{p} {placeholder_str}".strip()
        return " ".join([parts[0], placeholder_str] + parts[1:]).strip()
    raise ValueError(f"Unknown position: {position}")


def ensure_tokens(tokenizer, tokens: List[str]) -> None:
    num_added = tokenizer.add_tokens(tokens)
    if num_added != len(tokens):
        # If already present, add_tokens may add 0.
        # We require they exist, not necessarily newly added.
        missing = [t for t in tokens if tokenizer.convert_tokens_to_ids(t) == tokenizer.unk_token_id]
        if missing:
            raise ValueError(f"Some placeholder tokens are unknown to tokenizer: {missing}")


@torch.no_grad()
def inject_t5_soft_tokens(pipe: StableDiffusion3Pipeline, learned_path: str, verbose: bool = True):
    """
    Inject learned embeddings into SD3.5 T5 encoder (text_encoder_3).
    learned_path is a safetensors whose keys are placeholder token strings,
    values are embedding vectors [D].
    """
    if pipe.tokenizer_3 is None or pipe.text_encoder_3 is None:
        raise RuntimeError("This pipeline does not have tokenizer_3/text_encoder_3. Are you using SD3/SD3.5 pipeline?")

    learned = load_file(learned_path)  # {token: tensor[D]}
    tokens = list(learned.keys())

    # Add tokens to tokenizer_3
    ensure_tokens(pipe.tokenizer_3, tokens)

    # Resize embedding table if needed
    pipe.text_encoder_3.resize_token_embeddings(len(pipe.tokenizer_3))

    # Write embeddings
    emb = pipe.text_encoder_3.get_input_embeddings().weight
    for tok, vec in learned.items():
        tid = pipe.tokenizer_3.convert_tokens_to_ids(tok)
        if tid == pipe.tokenizer_3.unk_token_id:
            raise ValueError(f"Token '{tok}' is still unknown after add_tokens.")
        if vec.ndim != 1 or vec.shape[0] != emb.shape[1]:
            raise ValueError(f"Embedding shape mismatch for '{tok}': got {tuple(vec.shape)}, expected ({emb.shape[1]},)")
        emb[tid].copy_(vec.to(device=emb.device, dtype=emb.dtype))

    if verbose:
        print(f"[OK] Injected {len(tokens)} soft token embeddings into T5 (text_encoder_3). Tokens: {tokens}")

    return tokens


def _parse_seed(val) -> int:
    """Parse seed from CSV cell (may be int or str)."""
    if val is None or (isinstance(val, str) and not val.strip()):
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def read_prompts(path: str) -> List[dict]:
    """
    Read prompts from:
      - .csv: columns prompt [, random_seed] [, id]. E.g. pg_sexual_toxic.csv (prompt, label, random_seed, id).
      - .txt: one prompt per line (seed/id from args or index)
      - .json: {"prompts":[...]} or list
    Returns list of dicts: {"prompt": str, "seed": int | None, "id": str | None}.
    """
    if path.endswith(".csv"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "prompt" not in reader.fieldnames:
                raise ValueError("CSV must have a 'prompt' column.")
            for row in reader:
                p = (row.get("prompt") or "").strip()
                if not p:
                    continue
                seed = _parse_seed(row.get("random_seed"))
                row_id = (row.get("id") or "").strip() or None
                rows.append({"prompt": p, "seed": seed, "id": row_id})
        return rows

    # .txt or .json: no per-row seed/id
    if path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        return [{"prompt": p, "seed": None, "id": None} for p in lines]
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            prompts = [str(x).strip() for x in obj if str(x).strip()]
        elif isinstance(obj, dict) and "prompts" in obj:
            prompts = [str(x).strip() for x in obj["prompts"] if str(x).strip()]
        else:
            raise ValueError("JSON must be a list of prompts or a dict with key 'prompts'.")
        return [{"prompt": p, "seed": None, "id": None} for p in prompts]
    raise ValueError("prompts file must be .csv, .txt or .json")


def save_image(img: Image.Image, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--learned_path", type=str, required=True)
    ap.add_argument("--prompts", type=str, required=True, help="prompts file (.csv, .txt or .json). CSV: columns prompt, [random_seed], [id] (e.g. pg_sexual_toxic.csv)")
    ap.add_argument("--output_dir", type=str, default="./sd35_infer_out")

    ap.add_argument("--position", type=str, default="suffix", choices=["prefix", "suffix", "after_first"])
    ap.add_argument("--placeholder_token", type=str, default="<SAFE>")
    ap.add_argument("--num_vectors", type=int, default=1)

    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)

    ap.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--device", type=str, default="cuda")

    args = ap.parse_args()

    if args.dtype == "fp16":
        torch_dtype = torch.float16
    elif args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    entries = read_prompts(args.prompts)
    assert len(entries) > 0, "No prompts loaded."

    def _seed(ent: dict) -> int:
        return ent["seed"] if ent.get("seed") is not None else args.seed

    def _filename(ent: dict, i: int) -> str:
        return f"{ent['id']}.png" if ent.get("id") is not None else f"{i:04d}.png"

    placeholder_tokens = [args.placeholder_token] + [f"{args.placeholder_token}_{i}" for i in range(1, args.num_vectors)]
    placeholder_str = " ".join(placeholder_tokens)

    # -------------------------
    # 1) Vanilla run (no injection, no placeholder)
    # -------------------------
    pipe_vanilla = StableDiffusion3Pipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
    ).to(args.device)

    # Optional: memory optimizations
    pipe_vanilla.enable_attention_slicing()

    vanilla_dir = os.path.join(args.output_dir, "vanilla")
    os.makedirs(vanilla_dir, exist_ok=True)

    for i, ent in enumerate(entries):
        gen = torch.Generator(device=args.device).manual_seed(_seed(ent))
        out = pipe_vanilla(
            prompt=ent["prompt"],
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            height=args.height,
            width=args.width,
            generator=gen,
        )
        img = out.images[0]
        save_image(img, os.path.join(vanilla_dir, _filename(ent, i)))

    del pipe_vanilla
    torch.cuda.empty_cache()

    # -------------------------
    # 2) Defended run (inject + placeholder)
    # -------------------------
    pipe_def = StableDiffusion3Pipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
    ).to(args.device)
    pipe_def.enable_attention_slicing()

    inject_t5_soft_tokens(pipe_def, args.learned_path, verbose=True)

    defended_dir = os.path.join(args.output_dir, "defended")
    os.makedirs(defended_dir, exist_ok=True)

    for i, ent in enumerate(entries):
        gen2 = torch.Generator(device=args.device).manual_seed(_seed(ent))
        p_def = insert_placeholder(ent["prompt"], placeholder_str, args.position)
        out = pipe_def(
            prompt=p_def,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            height=args.height,
            width=args.width,
            generator=gen2,
        )
        img = out.images[0]
        save_image(img, os.path.join(defended_dir, _filename(ent, i)))

    # Save a small manifest for traceability
    manifest = {
        "model_path": args.model_path,
        "learned_path": args.learned_path,
        "position": args.position,
        "placeholder_tokens": placeholder_tokens,
        "steps": args.steps,
        "cfg": args.cfg,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "dtype": args.dtype,
        "num_prompts": len(entries),
    }
    with open(os.path.join(args.output_dir, "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Saved vanilla images to:  {vanilla_dir}")
    print(f"[DONE] Saved defended images to: {defended_dir}")
    print(f"[DONE] Manifest: {os.path.join(args.output_dir, 'run_manifest.json')}")


if __name__ == "__main__":
    main()
