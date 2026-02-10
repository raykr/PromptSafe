#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flux CSV inference: 仿照 sd3/infer_sd35.py 的推理逻辑。
- 读取 CSV（prompt, rewritten_prompt, id, random_seed）
- 每行：先构建 baseline conditioning（无 adapter），再构建 adapter conditioning，
  同 seed 生成两张图，分别保存到 baseline/ 与 adapter_all/
- 可选：用 rewritten_prompt 生成参考图到 rewritten/
- 输出 infer_log.csv
"""

import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from tqdm import tqdm
from diffusers import FluxPipeline
from transformers import AutoTokenizer, CLIPTextModel, T5EncoderModel
from transformers.modeling_outputs import BaseModelOutputWithPooling

try:
    from transformers.masking_utils import create_causal_mask
except ImportError:
    try:
        from transformers.modeling_attn_mask_utils import create_causal_mask
    except ImportError:
        create_causal_mask = None


# -------------------------
# Adapter & encoder (与 infer_flux_adapter.py 一致)
# -------------------------
class LowRankEmbeddingAdapter(nn.Module):
    def __init__(self, dim, rank):
        super().__init__()
        self.A = nn.Linear(dim, rank, bias=False)
        self.B = nn.Linear(rank, dim, bias=False)

    def forward(self, emb, scale=1.0):
        in_dtype = emb.dtype
        w_dtype = self.A.weight.dtype
        x = emb.to(w_dtype)
        h = self.A(x)
        h = F.gelu(h)
        delta = self.B(h)
        return (x + scale * delta).to(in_dtype)


def _clip_pool_from_last(hidden, attention_mask):
    idx = attention_mask.long().sum(dim=1) - 1
    idx = idx.clamp(min=0)
    b = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[b, idx]


class EncoderWithAdapter(nn.Module):
    def __init__(self, encoder, adapter):
        super().__init__()
        self.encoder = encoder
        self.adapter = adapter

    def forward(self, input_ids, attention_mask, scale):
        emb_layer = self.encoder.get_input_embeddings()
        emb = emb_layer(input_ids)
        emb = self.adapter(emb, scale=scale)
        if hasattr(self.encoder, "text_model"):
            tm = self.encoder.text_model
            seq_length = emb.size(1)
            pos_ids = tm.embeddings.position_ids[:, :seq_length].to(emb.device)
            pos_emb = tm.embeddings.position_embedding(pos_ids)
            hidden_states = emb + pos_emb
            if create_causal_mask is not None:
                attn_mask = create_causal_mask(
                    config=tm.config,
                    input_embeds=hidden_states,
                    attention_mask=attention_mask,
                    cache_position=torch.arange(seq_length, device=hidden_states.device),
                    past_key_values=None,
                )
            else:
                dtype = hidden_states.dtype
                bsz = hidden_states.size(0)
                causal = torch.triu(
                    torch.full((seq_length, seq_length), float("-inf"), device=hidden_states.device, dtype=dtype),
                    diagonal=1,
                )
                pad = (1.0 - attention_mask.to(dtype=dtype)) * torch.finfo(dtype).min
                causal = causal.unsqueeze(0).unsqueeze(0)
                pad = pad.view(bsz, 1, 1, seq_length)
                attn_mask = causal + pad
            enc_out = tm.encoder(inputs_embeds=hidden_states, attention_mask=attn_mask)
            last_hidden_state = tm.final_layer_norm(enc_out.last_hidden_state)
            pooled = _clip_pool_from_last(last_hidden_state, attention_mask)
            return BaseModelOutputWithPooling(
                last_hidden_state=last_hidden_state,
                pooler_output=pooled,
            )
        return self.encoder(inputs_embeds=emb, attention_mask=attention_mask)


def safe_name(x):
    x = str(x)
    for b in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '\n', '\r', '\t']:
        x = x.replace(b, "_")
    return x[:180]


@torch.no_grad()
def build_flux_conditioning(
    tok_clip,
    tok_t5,
    enc_clip,
    enc_t5,
    enc_clip_ad,
    enc_t5_ad,
    prompt,
    device,
    max_len_t5,
    scale_clip,
    scale_t5,
    num_images_per_prompt=1,
):
    """对单条 prompt 构建 Flux 条件。scale=0 时用 base encoder，否则用 adapter。"""
    clip_in = tok_clip(
        [prompt],
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    t5_in = tok_t5(
        [prompt],
        padding="max_length",
        max_length=max_len_t5,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    if scale_clip == 0 and scale_t5 == 0:
        clip_out = enc_clip(clip_in.input_ids, attention_mask=clip_in.attention_mask)
        t5_out = enc_t5(t5_in.input_ids, attention_mask=t5_in.attention_mask)
    else:
        clip_out = enc_clip_ad(clip_in.input_ids, clip_in.attention_mask, scale=scale_clip)
        t5_out = enc_t5_ad(t5_in.input_ids, t5_in.attention_mask, scale=scale_t5)

    pooled = clip_out.pooler_output
    prompt_embeds = t5_out.last_hidden_state

    if num_images_per_prompt > 1:
        pooled = pooled.repeat_interleave(num_images_per_prompt, dim=0)
        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)

    return prompt_embeds, pooled


@torch.no_grad()
def build_negative_conditioning(tok_clip, tok_t5, enc_clip, enc_t5, negative_prompt, device, max_len_t5, num_images_per_prompt=1):
    """Negative 固定用 base encoder，不接 adapter。"""
    clip_in = tok_clip(
        [negative_prompt],
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    t5_in = tok_t5(
        [negative_prompt],
        padding="max_length",
        max_length=max_len_t5,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    clip_out = enc_clip(clip_in.input_ids, attention_mask=clip_in.attention_mask)
    t5_out = enc_t5(t5_in.input_ids, attention_mask=t5_in.attention_mask)
    pooled = clip_out.pooler_output
    prompt_embeds = t5_out.last_hidden_state
    if num_images_per_prompt > 1:
        pooled = pooled.repeat_interleave(num_images_per_prompt, dim=0)
        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
    return prompt_embeds, pooled


def main():
    parser = argparse.ArgumentParser("Flux CSV inference (baseline + adapter_all, 两文件夹)")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--scale_clip", type=float, default=1.0)
    parser.add_argument("--scale_t5", type=float, default=1.0)
    parser.add_argument("--max_len_t5", type=int, default=256)

    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--cfg", type=float, default=7.0)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--num_images_per_prompt", type=int, default=1)
    parser.add_argument("--save_rewritten_ref", action="store_true")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_base = os.path.join(args.out_dir, "baseline")
    out_adpt = os.path.join(args.out_dir, "adapter_all")
    out_rewr = os.path.join(args.out_dir, "rewritten")
    os.makedirs(out_base, exist_ok=True)
    os.makedirs(out_adpt, exist_ok=True)
    if args.save_rewritten_ref:
        os.makedirs(out_rewr, exist_ok=True)

    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    device = args.device

    print(f"Loading Flux: {args.model_path}")
    pipe = FluxPipeline.from_pretrained(args.model_path, torch_dtype=dtype).to(device)
    pipe.set_progress_bar_config(disable=True)

    tok_clip = AutoTokenizer.from_pretrained(args.model_path, subfolder="tokenizer", use_fast=False)
    tok_t5 = AutoTokenizer.from_pretrained(args.model_path, subfolder="tokenizer_2", use_fast=False)
    enc_clip = CLIPTextModel.from_pretrained(args.model_path, subfolder="text_encoder", torch_dtype=dtype).to(device).eval()
    enc_t5 = T5EncoderModel.from_pretrained(args.model_path, subfolder="text_encoder_2", torch_dtype=dtype).to(device).eval()

    ckpt = load_file(args.adapter_path)
    print(f"[Adapter] loaded: {args.adapter_path}")
    clip_a_key = "clip_l.A.weight" if "clip_l.A.weight" in ckpt else "clip.A.weight"
    clip_b_key = "clip_l.B.weight" if "clip_l.B.weight" in ckpt else "clip.B.weight"

    ad_clip = LowRankEmbeddingAdapter(enc_clip.config.hidden_size, ckpt[clip_a_key].shape[0]).to(device)
    ad_t5 = LowRankEmbeddingAdapter(enc_t5.config.d_model, ckpt["t5.A.weight"].shape[0]).to(device)
    with torch.no_grad():
        ad_clip.A.weight.copy_(ckpt[clip_a_key].to(device=device, dtype=ad_clip.A.weight.dtype))
        ad_clip.B.weight.copy_(ckpt[clip_b_key].to(device=device, dtype=ad_clip.B.weight.dtype))
        ad_t5.A.weight.copy_(ckpt["t5.A.weight"].to(device=device, dtype=ad_t5.A.weight.dtype))
        ad_t5.B.weight.copy_(ckpt["t5.B.weight"].to(device=device, dtype=ad_t5.B.weight.dtype))

    enc_clip_ad = EncoderWithAdapter(enc_clip, ad_clip)
    enc_t5_ad = EncoderWithAdapter(enc_t5, ad_t5)

    # 预计算 negative（固定 base encoder）
    neg_pe, neg_ppe = build_negative_conditioning(
        tok_clip, tok_t5, enc_clip, enc_t5,
        args.negative_prompt or "",
        device, args.max_len_t5, args.num_images_per_prompt,
    )

    df = pd.read_csv(args.csv_path)
    if args.limit > 0:
        df = df.iloc[:args.limit].copy()

    records = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Inference", unit="prompt"):
        prompt = str(row.get("prompt", ""))
        rewritten = str(row.get("rewritten_prompt", "")) if "rewritten_prompt" in row else ""
        rid = row.get("id", i)
        seed = row.get("random_seed", 0)
        try:
            seed = int(float(seed))
        except Exception:
            seed = 0
        name = safe_name(rid)

        # baseline（无 adapter）
        pe0, ppe0 = build_flux_conditioning(
            tok_clip, tok_t5, enc_clip, enc_t5, enc_clip_ad, enc_t5_ad,
            prompt, device, args.max_len_t5,
            scale_clip=0.0, scale_t5=0.0,
            num_images_per_prompt=args.num_images_per_prompt,
        )
        # adapter
        pe1, ppe1 = build_flux_conditioning(
            tok_clip, tok_t5, enc_clip, enc_t5, enc_clip_ad, enc_t5_ad,
            prompt, device, args.max_len_t5,
            scale_clip=args.scale_clip, scale_t5=args.scale_t5,
            num_images_per_prompt=args.num_images_per_prompt,
        )

        gen = torch.Generator(device=device).manual_seed(seed)
        imgs0 = pipe(
            prompt_embeds=pe0,
            pooled_prompt_embeds=ppe0,
            negative_prompt_embeds=neg_pe,
            negative_pooled_prompt_embeds=neg_ppe,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            generator=gen,
            height=args.height,
            width=args.width,
            output_type="pil",
        ).images

        gen = torch.Generator(device=device).manual_seed(seed)
        imgs1 = pipe(
            prompt_embeds=pe1,
            pooled_prompt_embeds=ppe1,
            negative_prompt_embeds=neg_pe,
            negative_pooled_prompt_embeds=neg_ppe,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            generator=gen,
            height=args.height,
            width=args.width,
            output_type="pil",
        ).images

        base_paths, adpt_paths = [], []
        for k, img in enumerate(imgs0):
            p = os.path.join(out_base, f"{name}.png")
            img.save(p)
            base_paths.append(p)
        for k, img in enumerate(imgs1):
            p = os.path.join(out_adpt, f"{name}.png")
            img.save(p)
            adpt_paths.append(p)

        rew_paths = []
        if args.save_rewritten_ref and rewritten.strip():
            peR, ppeR = build_flux_conditioning(
                tok_clip, tok_t5, enc_clip, enc_t5, enc_clip_ad, enc_t5_ad,
                rewritten, device, args.max_len_t5,
                scale_clip=0.0, scale_t5=0.0,
                num_images_per_prompt=args.num_images_per_prompt,
            )
            gen = torch.Generator(device=device).manual_seed(seed)
            imgsR = pipe(
                prompt_embeds=peR,
                pooled_prompt_embeds=ppeR,
                negative_prompt_embeds=neg_pe,
                negative_pooled_prompt_embeds=neg_ppe,
                num_inference_steps=args.steps,
                guidance_scale=args.cfg,
                generator=gen,
                height=args.height,
                width=args.width,
                output_type="pil",
            ).images
            for k, img in enumerate(imgsR):
                p = os.path.join(out_rewr, f"{name}.png")
                img.save(p)
                rew_paths.append(p)

        records.append({
            "id": rid,
            "seed": seed,
            "prompt": prompt,
            "rewritten_prompt": rewritten,
            "baseline_paths": "|".join(base_paths),
            "adapter_all_paths": "|".join(adpt_paths),
            "rewritten_ref_paths": "|".join(rew_paths) if rew_paths else "",
        })
        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(df)}] done")

    log_csv = os.path.join(args.out_dir, "infer_log.csv")
    pd.DataFrame(records).to_csv(log_csv, index=False)
    print(f"Done. Saved to: {args.out_dir}")
    print(f"  baseline/   adapter_all/   (optional rewritten/)")
    print(f"Log: {log_csv}")


if __name__ == "__main__":
    main()
