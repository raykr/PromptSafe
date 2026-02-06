import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from tqdm import tqdm
from diffusers import StableDiffusion3Pipeline


# -----------------------------
# Low-rank adapter: x + scale * B(A(x))
# A: [r, d], B: [d, r]
# -----------------------------
class LowRankAdapter(nn.Module):
    def __init__(self, d: int, r: int):
        super().__init__()
        self.A = nn.Linear(d, r, bias=False)
        self.B = nn.Linear(r, d, bias=False)

    def forward(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        x = x.to(self.A.weight.dtype)
        return x + scale * self.B(self.A(x))


def safe_name(x: str) -> str:
    x = str(x)
    bad = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '\n', '\r', '\t']
    for b in bad:
        x = x.replace(b, "_")
    return x[:180]


def pick_keys(ckpt: dict, prefix: str):
    ka = f"{prefix}.A.weight"
    kb = f"{prefix}.B.weight"
    if ka in ckpt and kb in ckpt:
        return ka, kb
    # fallback: some projects save as f"{prefix}_A.weight" etc.
    candA = [k for k in ckpt.keys() if k.endswith("A.weight") and prefix in k]
    candB = [k for k in ckpt.keys() if k.endswith("B.weight") and prefix in k]
    if len(candA) == 1 and len(candB) == 1:
        return candA[0], candB[0]
    return None, None


def load_adapter(ckpt: dict, keyA: str, keyB: str, expected_dim: int, device: str, dtype: torch.dtype):
    A = ckpt[keyA]
    B = ckpt[keyB]
    if A.dim() != 2 or B.dim() != 2:
        raise ValueError(f"Adapter weights must be 2D. Got {keyA}:{A.shape}, {keyB}:{B.shape}")

    r, d = A.shape
    if d != expected_dim:
        raise ValueError(f"Dim mismatch for {keyA}: got d={d}, expected {expected_dim}")
    if B.shape != (d, r):
        raise ValueError(f"Shape mismatch: A={A.shape}, B={B.shape}, expected B={(d, r)}")

    ad = LowRankAdapter(d, r).to(device=device, dtype=dtype).eval()
    with torch.no_grad():
        ad.A.weight.copy_(A.to(device=device, dtype=dtype))
        ad.B.weight.copy_(B.to(device=device, dtype=dtype))
    return ad


@torch.no_grad()
def get_clip_prompt_embeds(pipe, prompt: str, device: str, num_images_per_prompt: int, clip_model_index: int):
    # diffusers SD3 helper
    prompt_embeds, pooled = pipe._get_clip_prompt_embeds(
        prompt=prompt,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        clip_skip=None,
        clip_model_index=clip_model_index,
    )
    return prompt_embeds, pooled


@torch.no_grad()
def apply_token_adapter_to_hidden(
    hidden: torch.Tensor,
    adapter: nn.Module,
    scale: float,
):
    if adapter is None or scale == 0:
        return hidden
    return adapter(hidden, scale=scale)


@torch.no_grad()
def build_t5_hidden(pipe, text: str, device: str, num_images_per_prompt: int, max_sequence_length: int,
                    adapter_t5: nn.Module = None, scale_t5: float = 1.0):
    text = [text] if isinstance(text, str) else text
    batch_size = len(text)
    tok = pipe.tokenizer_3(
        text,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    input_ids = tok.input_ids.to(device)
    h = pipe.text_encoder_3(input_ids)[0]  # [B, L, D]
    h = h.to(device=device, dtype=pipe.text_encoder_3.dtype)
    h = apply_token_adapter_to_hidden(h, adapter_t5, scale_t5)

    _, L, D = h.shape
    h = h.repeat(1, num_images_per_prompt, 1).view(batch_size * num_images_per_prompt, L, D)
    return h


@torch.no_grad()
def build_sd3_conditioning_all(
    pipe,
    prompt: str,
    device: str,
    num_images_per_prompt: int = 1,
    max_len_t5: int = 256,
    adapter_clip_l: nn.Module = None,
    adapter_clip_g: nn.Module = None,
    adapter_t5: nn.Module = None,
    scale_clip_l: float = 1.0,
    scale_clip_g: float = 1.0,
    scale_t5: float = 1.0,
    negative_prompt: str = "",
    freeze_negative: bool = True,
    cached_negative=None,
):
    """
    Manual SD3 conditioning (all): apply adapters on:
      - CLIP-L prompt_embeds (token-level)
      - CLIP-G prompt_embeds (token-level)
      - T5 last_hidden_state (token-level)

    pooled_prompt_embeds are CLIP pooled; we DO NOT adapt pooled by default (consistent with token adapter training).
    """
    # ----- positive -----
    clip_l, pooled_l = get_clip_prompt_embeds(pipe, prompt, device, num_images_per_prompt, clip_model_index=0)
    clip_g, pooled_g = get_clip_prompt_embeds(pipe, prompt, device, num_images_per_prompt, clip_model_index=1)

    clip_l = apply_token_adapter_to_hidden(clip_l, adapter_clip_l, scale_clip_l)
    clip_g = apply_token_adapter_to_hidden(clip_g, adapter_clip_g, scale_clip_g)

    clip_cat = torch.cat([clip_l, clip_g], dim=-1)  # [B, Lc, dL+dG]

    t5 = build_t5_hidden(pipe, prompt, device, num_images_per_prompt, max_len_t5, adapter_t5, scale_t5)  # [B, Lt, dT]

    clip_cat = F.pad(clip_cat, (0, t5.shape[-1] - clip_cat.shape[-1]))
    prompt_embeds = torch.cat([clip_cat, t5], dim=-2)
    pooled_prompt_embeds = torch.cat([pooled_l, pooled_g], dim=-1)

    # ----- negative -----
    if cached_negative is not None:
        negative_prompt_embeds, negative_pooled_prompt_embeds = cached_negative
        return prompt_embeds, pooled_prompt_embeds, negative_prompt_embeds, negative_pooled_prompt_embeds

    n_clip_l, n_pooled_l = get_clip_prompt_embeds(pipe, negative_prompt, device, num_images_per_prompt, clip_model_index=0)
    n_clip_g, n_pooled_g = get_clip_prompt_embeds(pipe, negative_prompt, device, num_images_per_prompt, clip_model_index=1)

    if not freeze_negative:
        n_clip_l = apply_token_adapter_to_hidden(n_clip_l, adapter_clip_l, scale_clip_l)
        n_clip_g = apply_token_adapter_to_hidden(n_clip_g, adapter_clip_g, scale_clip_g)

    n_clip_cat = torch.cat([n_clip_l, n_clip_g], dim=-1)

    if freeze_negative:
        n_t5 = build_t5_hidden(pipe, negative_prompt, device, num_images_per_prompt, max_len_t5, None, 0.0)
    else:
        n_t5 = build_t5_hidden(pipe, negative_prompt, device, num_images_per_prompt, max_len_t5, adapter_t5, scale_t5)

    n_clip_cat = F.pad(n_clip_cat, (0, n_t5.shape[-1] - n_clip_cat.shape[-1]))
    negative_prompt_embeds = torch.cat([n_clip_cat, n_t5], dim=-2)
    negative_pooled_prompt_embeds = torch.cat([n_pooled_l, n_pooled_g], dim=-1)

    return prompt_embeds, pooled_prompt_embeds, negative_prompt_embeds, negative_pooled_prompt_embeds


def main():
    parser = argparse.ArgumentParser("SD3.5 Scheme-B ALL inference over CSV")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--scale_t5", type=float, default=10.0)
    parser.add_argument("--scale_clip_g", type=float, default=5.0)
    parser.add_argument("--scale_clip_l", type=float, default=5.0)
    parser.add_argument("--max_len_t5", type=int, default=256)

    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--cfg", type=float, default=7.0)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--freeze_negative", action="store_true", help="Keep uncond negative branch baseline (recommended).")
    parser.add_argument("--save_rewritten_ref", action="store_true")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--num_images_per_prompt", type=int, default=1)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_base = os.path.join(args.out_dir, "baseline")
    out_adpt = os.path.join(args.out_dir, "adapter_all")
    out_rewr = os.path.join(args.out_dir, "rewritten")
    os.makedirs(out_base, exist_ok=True)
    os.makedirs(out_adpt, exist_ok=True)
    if args.save_rewritten_ref:
        os.makedirs(out_rewr, exist_ok=True)

    # dtype
    if args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    device = args.device

    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_path, torch_dtype=torch_dtype).to(device)
    pipe.set_progress_bar_config(disable=True)

    ckpt_path = os.path.abspath(args.adapter_path)
    ckpt = load_file(ckpt_path)
    print(f"[Adapter] loaded: {ckpt_path}")
    print(f"[Adapter] keys: {sorted(list(ckpt.keys()))}")

    # Expect all three
    kA_t5, kB_t5 = pick_keys(ckpt, "t5")
    kA_g,  kB_g  = pick_keys(ckpt, "clip_g")
    kA_l,  kB_l  = pick_keys(ckpt, "clip_l")
    if None in [kA_t5, kB_t5, kA_g, kB_g, kA_l, kB_l]:
        raise KeyError(
            "ALL mode requires keys for t5/clip_g/clip_l. "
            f"Found: t5=({kA_t5},{kB_t5}), clip_g=({kA_g},{kB_g}), clip_l=({kA_l},{kB_l}). "
            "Please check adapter file naming."
        )

    # Load adapters with correct dims
    dim_t5 = pipe.text_encoder_3.config.d_model
    dim_l = pipe.text_encoder.config.hidden_size
    dim_g = pipe.text_encoder_2.config.hidden_size

    adapter_t5 = load_adapter(ckpt, kA_t5, kB_t5, dim_t5, device, pipe.text_encoder_3.dtype)
    adapter_l  = load_adapter(ckpt, kA_l,  kB_l,  dim_l,  device, pipe.text_encoder.dtype)
    adapter_g  = load_adapter(ckpt, kA_g,  kB_g,  dim_g,  device, pipe.text_encoder_2.dtype)

    # read csv
    df = pd.read_csv(args.csv_path)
    if args.limit > 0:
        df = df.iloc[:args.limit].copy()

    # cache negative baseline embeddings once (optional)
    cached_negative = None
    if args.freeze_negative:
        # Compute negative tensors once (independent of prompt)
        # Use a dummy prompt for positive part; we only keep negative outputs from the function.
        _, _, ne0, npe0 = build_sd3_conditioning_all(
            pipe,
            prompt="dummy",
            device=device,
            num_images_per_prompt=args.num_images_per_prompt,
            max_len_t5=args.max_len_t5,
            adapter_clip_l=None,
            adapter_clip_g=None,
            adapter_t5=None,
            scale_clip_l=0.0,
            scale_clip_g=0.0,
            scale_t5=0.0,
            negative_prompt=args.negative_prompt,
            freeze_negative=True,
            cached_negative=None,
        )
        cached_negative = (ne0, npe0)

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

        # baseline conditioning
        pe0, ppe0, ne0, npe0 = build_sd3_conditioning_all(
            pipe,
            prompt=prompt,
            device=device,
            num_images_per_prompt=args.num_images_per_prompt,
            max_len_t5=args.max_len_t5,
            adapter_clip_l=None,
            adapter_clip_g=None,
            adapter_t5=None,
            scale_clip_l=0.0,
            scale_clip_g=0.0,
            scale_t5=0.0,
            negative_prompt=args.negative_prompt,
            freeze_negative=args.freeze_negative,
            cached_negative=cached_negative if args.freeze_negative else None,
        )

        # adapted conditioning (ALL)
        pe1, ppe1, ne1, npe1 = build_sd3_conditioning_all(
            pipe,
            prompt=prompt,
            device=device,
            num_images_per_prompt=args.num_images_per_prompt,
            max_len_t5=args.max_len_t5,
            adapter_clip_l=adapter_l,
            adapter_clip_g=adapter_g,
            adapter_t5=adapter_t5,
            scale_clip_l=args.scale_clip_l,
            scale_clip_g=args.scale_clip_g,
            scale_t5=args.scale_t5,
            negative_prompt=args.negative_prompt,
            freeze_negative=args.freeze_negative,
            cached_negative=cached_negative if args.freeze_negative else None,
        )

        # generate baseline
        gen = torch.Generator(device=device).manual_seed(seed)
        imgs0 = pipe(
            prompt_embeds=pe0,
            pooled_prompt_embeds=ppe0,
            negative_prompt_embeds=ne0,
            negative_pooled_prompt_embeds=npe0,
            guidance_scale=args.cfg,
            num_inference_steps=args.steps,
            generator=gen,
            output_type="pil",
        ).images

        # generate adapted
        gen = torch.Generator(device=device).manual_seed(seed)
        imgs1 = pipe(
            prompt_embeds=pe1,
            pooled_prompt_embeds=ppe1,
            negative_prompt_embeds=ne1,
            negative_pooled_prompt_embeds=npe1,
            guidance_scale=args.cfg,
            num_inference_steps=args.steps,
            generator=gen,
            output_type="pil",
        ).images

        base_paths, adpt_paths = [], []
        for k, img in enumerate(imgs0):
            p = os.path.join(out_base, f"{name}_{k}.png")
            img.save(p)
            base_paths.append(p)
        for k, img in enumerate(imgs1):
            p = os.path.join(out_adpt, f"{name}_{k}.png")
            img.save(p)
            adpt_paths.append(p)

        rew_paths = []
        if args.save_rewritten_ref and rewritten.strip():
            peR, ppeR, neR, npeR = build_sd3_conditioning_all(
                pipe,
                prompt=rewritten,
                device=device,
                num_images_per_prompt=args.num_images_per_prompt,
                max_len_t5=args.max_len_t5,
                adapter_clip_l=None,
                adapter_clip_g=None,
                adapter_t5=None,
                scale_clip_l=0.0,
                scale_clip_g=0.0,
                scale_t5=0.0,
                negative_prompt=args.negative_prompt,
                freeze_negative=args.freeze_negative,
                cached_negative=cached_negative if args.freeze_negative else None,
            )
            gen = torch.Generator(device=device).manual_seed(seed)
            imgsR = pipe(
                prompt_embeds=peR,
                pooled_prompt_embeds=ppeR,
                negative_prompt_embeds=neR,
                negative_pooled_prompt_embeds=npeR,
                guidance_scale=args.cfg,
                num_inference_steps=args.steps,
                generator=gen,
                output_type="pil",
            ).images
            for k, img in enumerate(imgsR):
                p = os.path.join(out_rewr, f"{name}_{k}.png")
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
    print(f"Log CSV: {log_csv}")


if __name__ == "__main__":
    main()
