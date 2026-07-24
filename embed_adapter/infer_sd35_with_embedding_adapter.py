#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from typing import List, Dict, Any, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file

from diffusers import StableDiffusion3Pipeline


class LowRankEmbeddingAdapter(nn.Module):
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.A = nn.Linear(dim, rank, bias=False)
        self.B = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.A.weight)
        nn.init.zeros_(self.B.weight)

    def forward(self, emb: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        # ensure dtype matches weights
        emb = emb.to(dtype=self.A.weight.dtype)
        h = self.A(emb)
        h = F.gelu(h)
        delta = self.B(h)
        return emb + scale * delta


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
        raise ValueError("CSV must contain a 'prompt' column.")
    # Optional filters
    if label_filter is not None and "label" in df.columns:
        df = df[df["label"].astype(str) == str(label_filter)]
    if limit is not None and limit > 0:
        df = df.head(limit)

    rows = []
    for _, row in df.iterrows():
        prompt = str(row.get("prompt", "")).strip()
        if not prompt:
            continue
        item = {
            "prompt": prompt,
            "id": str(row.get("id", "")),
            "label": str(row.get("label", "")),
            "random_seed": int(row.get("random_seed", 42)) if str(row.get("random_seed", "")).strip() else 42,
        }
        rows.append(item)
    return rows


def wrap_text_encoder3_with_emb_adapter(pipe: StableDiffusion3Pipeline, adapter: LowRankEmbeddingAdapter, scale: float):
    """
    Monkey-patch pipe.text_encoder_3.forward to inject adapter at embedding stage by switching to inputs_embeds.
    """
    te = pipe.text_encoder_3
    orig_forward = te.forward

    def new_forward(*args, **kwargs):
        input_ids = kwargs.get("input_ids", None)
        inputs_embeds = kwargs.get("inputs_embeds", None)

        if inputs_embeds is None:
            if input_ids is None:
                return orig_forward(*args, **kwargs)
            emb_layer = te.get_input_embeddings()
            inputs_embeds = emb_layer(input_ids)

        inputs_embeds = adapter(inputs_embeds, scale=scale)

        kwargs["inputs_embeds"] = inputs_embeds
        kwargs["input_ids"] = None
        return orig_forward(*args, **kwargs)

    te.forward = new_forward


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--adapter_path", type=str, required=True)
    ap.add_argument("--test_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="./sd35_emb_adapter_infer")

    ap.add_argument("--label_filter", type=str, default=None, help="e.g., toxic (optional)")
    ap.add_argument("--limit", type=int, default=None, help="only run first N rows (optional)")

    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)

    ap.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--scale", type=float, default=1.0)
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
        raise RuntimeError("No valid prompts found after filtering.")

    # ---------- vanilla pipeline ----------
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

    # ---------- defended pipeline ----------
    tens = load_file(args.adapter_path)
    A = tens["A.weight"]
    B = tens["B.weight"]
    rank = A.shape[0]
    dim = A.shape[1]

    adapter = LowRankEmbeddingAdapter(dim=dim, rank=rank).to(args.device, dtype=torch_dtype)
    with torch.no_grad():
        adapter.A.weight.copy_(A.to(device=args.device, dtype=torch_dtype))
        adapter.B.weight.copy_(B.to(device=args.device, dtype=torch_dtype))

    pipe2 = StableDiffusion3Pipeline.from_pretrained(args.model_path, torch_dtype=torch_dtype).to(args.device)
    pipe2.enable_attention_slicing()
    wrap_text_encoder3_with_emb_adapter(pipe2, adapter, scale=args.scale)

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
