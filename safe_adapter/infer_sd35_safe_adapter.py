#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from typing import Optional, List, Dict, Any

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from diffusers import StableDiffusion3Pipeline


# -------------------------
# SafeAdapter definition (must match training)
# -------------------------
class SafeAdapter(nn.Module):
    """
    Token-level residual adapter:
      y = x + scale * Up(GELU(Down(LN(x))))
    """
    def __init__(self, hidden_size: int, rank: int = 256, dropout: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        # init near-identity
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        h = self.ln(x)
        h = self.down(h)
        h = F.gelu(h)
        h = self.dropout(h)
        h = self.up(h)
        return x + scale * h


# -------------------------
# IO helpers
# -------------------------
def save_image(img: Image.Image, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


def load_test_csv(
    csv_path: str,
    label_filter: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    df = pd.read_csv(csv_path)
    if "prompt" not in df.columns:
        raise ValueError("CSV must contain column: prompt")

    if label_filter is not None and "label" in df.columns:
        df = df[df["label"].astype(str) == str(label_filter)]

    if limit is not None and limit > 0:
        df = df.head(limit)

    items = []
    for _, row in df.iterrows():
        p = str(row.get("prompt", "")).strip()
        if not p:
            continue
        seed_raw = row.get("random_seed", 42)
        try:
            seed = int(seed_raw)
        except Exception:
            seed = 42

        items.append({
            "prompt": p,
            "id": str(row.get("id", "")),
            "label": str(row.get("label", "")),
            "random_seed": seed,
        })
    return items


# -------------------------
# Injection: apply SafeAdapter after T5 forward
# -------------------------
def inject_safeadapter_into_text_encoder_3(pipe: StableDiffusion3Pipeline, adapter: SafeAdapter, scale: float):
    """
    Monkey-patch pipe.text_encoder_3.forward so that:
      last_hidden_state <- adapter(last_hidden_state, scale)
    Works without touching tokenizer or other encoders.
    """
    te3 = pipe.text_encoder_3
    orig_forward = te3.forward

    def new_forward(*args, **kwargs):
        out = orig_forward(*args, **kwargs)

        # out is a BaseModelOutput (has .last_hidden_state)
        h = out.last_hidden_state
        h2 = adapter(h, scale=scale)

        # replace last_hidden_state
        out.last_hidden_state = h2
        return out

    te3.forward = new_forward


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_path", type=str, required=True,
                    help="SD3.5 model path, e.g. /home/raykr/models/stabilityai/stable-diffusion-3.5-large")
    ap.add_argument("--adapter_ckpt", type=str, required=True,
                    help="SafeAdapter checkpoint (.pt) saved from training")
    ap.add_argument("--test_csv", type=str, required=True,
                    help="CSV containing prompt, random_seed, id, label")
    ap.add_argument("--out_dir", type=str, default="./sd35_safeadapter_infer")

    ap.add_argument("--rank", type=int, default=256, help="Adapter rank (must match training)")
    ap.add_argument("--dropout", type=float, default=0.0, help="Adapter dropout (must match training if used)")

    ap.add_argument("--scale", type=float, default=1.0, help="Adapter strength at inference")

    ap.add_argument("--label_filter", type=str, default=None, help="Optional: filter rows by label value")
    ap.add_argument("--limit", type=int, default=None, help="Optional: only run first N prompts")

    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)

    ap.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    if args.dtype == "fp16":
        torch_dtype = torch.float16
    elif args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    items = load_test_csv(args.test_csv, label_filter=args.label_filter, limit=args.limit)
    if len(items) == 0:
        raise RuntimeError("No valid prompts found in CSV after filtering.")

    # ----------------- Vanilla -----------------
    # pipe = StableDiffusion3Pipeline.from_pretrained(args.model_path, torch_dtype=torch_dtype).to(args.device)
    # pipe.enable_attention_slicing()

    # vanilla_dir = os.path.join(args.out_dir, "vanilla")
    # for i, it in enumerate(items):
    #     gen = torch.Generator(device=args.device).manual_seed(int(it["random_seed"]))
    #     img = pipe(
    #         prompt=it["prompt"],
    #         num_inference_steps=args.steps,
    #         guidance_scale=args.cfg,
    #         height=args.height,
    #         width=args.width,
    #         generator=gen,
    #     ).images[0]
    #     name = f"{i:04d}_id{it['id']}_seed{it['random_seed']}.png"
    #     save_image(img, os.path.join(vanilla_dir, name))

    # ----------------- Defended (SafeAdapter on text_encoder_3) -----------------
    # Load a fresh pipeline so monkey-patch doesn't affect vanilla
    pipe2 = StableDiffusion3Pipeline.from_pretrained(args.model_path, torch_dtype=torch_dtype).to(args.device)
    pipe2.enable_attention_slicing()

    # infer hidden size from text_encoder_3 config
    hidden_size = pipe2.text_encoder_3.config.d_model  # T5 d_model

    adapter = SafeAdapter(hidden_size=hidden_size, rank=args.rank, dropout=args.dropout).to(args.device, dtype=torch_dtype)

    # load weights
    sd = torch.load(args.adapter_ckpt, map_location="cpu")
    adapter.load_state_dict(sd, strict=True)
    adapter.eval()

    inject_safeadapter_into_text_encoder_3(pipe2, adapter, scale=args.scale)

    defended_dir = os.path.join(args.out_dir, "defended")
    for i, it in enumerate(items):
        gen = torch.Generator(device=args.device).manual_seed(args.seed)
        img = pipe2(
            prompt=it["prompt"],
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            height=args.height,
            width=args.width,
            generator=gen,
        ).images[0]
        name = f"{it['id']}.png"
        save_image(img, os.path.join(defended_dir, name))

    # print(f"[DONE] vanilla:  {vanilla_dir}")
    print(f"[DONE] defended: {defended_dir}")


if __name__ == "__main__":
    main()
