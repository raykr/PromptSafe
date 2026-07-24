#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a Low-Rank Embedding Adapter for SD3.5 T5-XXL (tokenizer_3/text_encoder_3).
Goal: mimic SafeAdapter's fast convergence but keep "Embedding Sanitization" narrative.

We freeze T5 encoder weights. We learn a low-rank residual edit on the *input embeddings*:
    e' = e + scale * (e @ A @ B)
where A: [D, r], B: [r, D], r << D.

We optimize using text pairs from CSV:
  - malicious: prompt
  - positive: rewritten_prompt
Optional benign constraint:
  - benign_mode=rewritten (default): benign = rewritten_prompt
  - benign_mode=label0_prompt: sample benign from prompt where label==0
  - benign_mode=none: no benign loss

Loss:
  L = lambda_tri * triplet + lambda_align * align + lambda_benign * benign_identity
where
  z_a = pool(E'(m))
  z_p = pool(E'(r))
  z_n0 = pool(E(m))     (base reference)
  triplet = relu(d(z_a,z_p) - d(z_a,z_n0) + margin)
  align   = mse(z_a, z_p)
  benign_identity: mse(pool(E'(b)), pool(E(b)))   (optional)

Outputs:
  - embedding_adapter.safetensors  (A,B + metadata)
"""

import os
import argparse
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, T5EncoderModel
from safetensors.torch import save_file, load_file


# -------------------------
# Utils
# -------------------------
def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


# -------------------------
# Low-Rank Embedding Adapter
# -------------------------
class LowRankEmbeddingAdapter(nn.Module):
    """
    e' = e + scale * (e @ A @ B), where A in R^{D x r}, B in R^{r x D}
    """
    def __init__(self, dim: int, rank: int = 64, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.A = nn.Linear(dim, rank, bias=False)   # D -> r
        self.B = nn.Linear(rank, dim, bias=False)   # r -> D
        self.dropout = nn.Dropout(dropout)

        # init near-identity: make B ~ 0 so residual starts small
        nn.init.normal_(self.A.weight, std=0.02)
        nn.init.zeros_(self.B.weight)

    def forward(self, emb: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        # emb: [B, L, D]
        h = self.A(emb)
        h = F.gelu(h)
        h = self.dropout(h)
        delta = self.B(h)
        return emb + scale * delta


class T5WithEmbeddingAdapter(nn.Module):
    """
    Wrap T5 encoder and inject embedding adapter at input embedding.
    Only adapter is trainable; T5 weights are frozen.
    """
    def __init__(self, t5: T5EncoderModel, emb_adapter: LowRankEmbeddingAdapter):
        super().__init__()
        self.t5 = t5
        self.emb_adapter = emb_adapter

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, scale: float = 1.0):
        # get input embeddings
        emb_layer = self.t5.get_input_embeddings()
        emb = emb_layer(input_ids)  # [B, L, D]
        emb = self.emb_adapter(emb, scale=scale)  # [B, L, D]
        # feed via inputs_embeds
        out = self.t5(inputs_embeds=emb, attention_mask=attention_mask)
        return out.last_hidden_state  # [B, L, D]


# -------------------------
# Dataset
# -------------------------
class PromptPairCSV(Dataset):
    """
    CSV fields:
      prompt, random_seed, label, id, category, origin_id, source, rewritten_prompt

    maps:
      malicious = prompt
      rewritten = rewritten_prompt
    optional filtering by category
    benign:
      - rewritten (default): benign = rewritten_prompt
      - label0_prompt: benign sampled from prompt with label==0
      - none: no benign
    """
    def __init__(
        self,
        csv_path: str,
        category: Optional[str] = None,
        benign_mode: str = "rewritten",
        label0_value: str = "0",
        seed: int = 42,
    ):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)

        df = pd.read_csv(csv_path)

        for c in ["prompt", "rewritten_prompt"]:
            if c not in df.columns:
                raise ValueError(f"CSV missing required column: {c}")

        if category is not None:
            if "category" not in df.columns:
                raise ValueError("You passed --category but CSV has no 'category' column.")
            df = df[df["category"].astype(str) == str(category)]

        self.m = df["prompt"].fillna("").astype(str).tolist()
        self.r = df["rewritten_prompt"].fillna("").astype(str).tolist()

        self.benign_mode = benign_mode
        self.rng = random.Random(seed)

        self.label0_pool: Optional[List[str]] = None
        if benign_mode == "label0_prompt":
            if "label" not in df.columns:
                raise ValueError("benign_mode=label0_prompt requires a 'label' column.")
            labels = df["label"].fillna("").astype(str).tolist()
            pool = [p for p, lab in zip(self.m, labels) if lab == str(label0_value) and p.strip()]
            if len(pool) == 0:
                raise ValueError(f"No samples with label=={label0_value} to use as benign.")
            self.label0_pool = pool

    def __len__(self):
        return len(self.m)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        mal = self.m[idx].strip()
        rew = self.r[idx].strip()

        if self.benign_mode == "rewritten":
            ben = rew
        elif self.benign_mode == "label0_prompt":
            ben = self.rng.choice(self.label0_pool)
        elif self.benign_mode == "none":
            ben = None
        else:
            raise ValueError(f"Unknown benign_mode: {self.benign_mode}")

        return {"malicious": mal, "rewritten": rew, "benign": ben}


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, List[Optional[str]]]:
    keys = batch[0].keys()
    return {k: [x[k] for x in batch] for k in keys}


# -------------------------
# Train
# -------------------------
@dataclass
class TrainCfg:
    model_path: str
    train_csv: str
    out_dir: str
    category: Optional[str]

    device: str
    dtype: str
    max_length: int
    batch_size: int
    num_epochs: int
    lr: float
    rank: int
    dropout: float
    scale: float

    margin: float
    lambda_tri: float
    lambda_align: float
    lambda_benign: float

    benign_mode: str
    label0_value: str

    log_dir: Optional[str]
    save_every: int


class Trainer:
    def __init__(self, cfg: TrainCfg):
        self.cfg = cfg
        os.makedirs(cfg.out_dir, exist_ok=True)

        # dtype
        if cfg.dtype == "fp16":
            self.torch_dtype = torch.float16
        elif cfg.dtype == "bf16":
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = torch.float32

        # SD3.5 T5-XXL
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, subfolder="tokenizer_3", use_fast=False)
        t5 = T5EncoderModel.from_pretrained(cfg.model_path, subfolder="text_encoder_3", torch_dtype=self.torch_dtype)

        # freeze T5
        t5.requires_grad_(False)
        t5.eval()

        dim = t5.config.d_model
        emb_adapter = LowRankEmbeddingAdapter(dim=dim, rank=cfg.rank, dropout=cfg.dropout)
        self.model = T5WithEmbeddingAdapter(t5, emb_adapter).to(cfg.device, dtype=self.torch_dtype)


        self.opt = torch.optim.AdamW(self.model.emb_adapter.parameters(), lr=cfg.lr, betas=(0.9, 0.999), weight_decay=0.0)

        self.step_count = 0

    @torch.no_grad()
    def encode_base(self, texts: List[str]) -> torch.Tensor:
        batch = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.cfg.max_length, return_tensors="pt"
        ).to(self.cfg.device)
        h = self.model.t5(**batch).last_hidden_state
        return mean_pool(h, batch.attention_mask)

    def encode_adapted(self, texts: List[str]) -> torch.Tensor:
        batch = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.cfg.max_length, return_tensors="pt"
        ).to(self.cfg.device)
        with torch.no_grad():
            # nothing; we keep T5 frozen but forward uses inputs_embeds so no need extra guard here
            pass
        h = self.model(batch.input_ids, batch.attention_mask, scale=self.cfg.scale)
        return mean_pool(h, batch.attention_mask)

    def step(self, malicious: List[str], rewritten: List[str], benign: List[Optional[str]]):
        z_a = self.encode_adapted(malicious)   # E'(m)
        z_p = self.encode_adapted(rewritten)   # E'(r)
        z_n0 = self.encode_base(malicious)     # E(m) base reference

        d_ap = F.pairwise_distance(z_a, z_p)
        d_an = F.pairwise_distance(z_a, z_n0)

        loss_tri = F.relu(d_ap - d_an + self.cfg.margin).mean()
        loss_align = F.mse_loss(z_a, z_p)

        benign_valid = [b for b in benign if isinstance(b, str) and b.strip()]
        if benign_valid and self.cfg.lambda_benign > 0 and self.cfg.benign_mode != "none":
            z_b_ad = self.encode_adapted(benign_valid)  # E'(b)
            z_b0 = self.encode_base(benign_valid)       # E(b)
            loss_ben = F.mse_loss(z_b_ad, z_b0)
        else:
            loss_ben = torch.tensor(0.0, device=self.cfg.device)

        loss = (
            self.cfg.lambda_tri * loss_tri
            + self.cfg.lambda_align * loss_align
            + self.cfg.lambda_benign * loss_ben
        )

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.emb_adapter.parameters(), max_norm=1.0)
        self.opt.step()

        self.step_count += 1

        return {
            "loss": float(loss.detach().cpu()),
            "tri": float(loss_tri.detach().cpu()),
            "align": float(loss_align.detach().cpu()),
            "benign": float(loss_ben.detach().cpu()),
            "d_ap": float(d_ap.mean().detach().cpu()),
            "d_an": float(d_an.mean().detach().cpu()),
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # save A/B in safetensors for clean injection
        state = {
            "A.weight": self.model.emb_adapter.A.weight.detach().cpu(),
            "B.weight": self.model.emb_adapter.B.weight.detach().cpu(),
        }
        meta = {
            "rank": str(self.cfg.rank),
            "dropout": str(self.cfg.dropout),
            "scale": str(self.cfg.scale),
            "subfolder_tokenizer": "tokenizer_3",
            "subfolder_text_encoder": "text_encoder_3",
        }
        save_file(state, path, metadata=meta)


def train(cfg: TrainCfg):
    log_dir = cfg.log_dir or os.path.join(cfg.out_dir, "runs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[TB] tensorboard --logdir {log_dir}")

    ds = PromptPairCSV(
        csv_path=cfg.train_csv,
        category=cfg.category,
        benign_mode=cfg.benign_mode,
        label0_value=cfg.label0_value,
        seed=42,
    )
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)

    trainer = Trainer(cfg)

    global_step = 0
    for epoch in range(cfg.num_epochs):
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{cfg.num_epochs}", dynamic_ncols=True)
        agg = {k: [] for k in ["loss", "tri", "align", "benign", "d_ap", "d_an"]}

        for batch in pbar:
            logs = trainer.step(batch["malicious"], batch["rewritten"], batch["benign"])
            for k in agg:
                agg[k].append(logs[k])

            writer.add_scalar("train/loss", logs["loss"], global_step)
            writer.add_scalar("train/tri", logs["tri"], global_step)
            writer.add_scalar("train/align", logs["align"], global_step)
            writer.add_scalar("train/benign", logs["benign"], global_step)
            writer.add_scalar("train/d_ap", logs["d_ap"], global_step)
            writer.add_scalar("train/d_an", logs["d_an"], global_step)

            pbar.set_postfix({
                "loss": f"{logs['loss']:.3f}",
                "tri": f"{logs['tri']:.3f}",
                "aln": f"{logs['align']:.3f}",
                "ben": f"{logs['benign']:.3f}",
            })
            global_step += 1

        avg = {k: sum(v)/max(len(v), 1) for k, v in agg.items()}
        for k, v in avg.items():
            writer.add_scalar(f"epoch/{k}", v, epoch)

        if cfg.save_every > 0 and (epoch + 1) % cfg.save_every == 0:
            ckpt = os.path.join(cfg.out_dir, f"embedding_adapter_epoch{epoch+1}.safetensors")
            trainer.save(ckpt)
            print(f"[SAVE] {ckpt}")

    final_path = os.path.join(cfg.out_dir, "embedding_adapter_final.safetensors")
    trainer.save(final_path)
    writer.close()
    print(f"[DONE] saved: {final_path}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default="/home/raykr/models/stabilityai/stable-diffusion-3.5-large")
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="./out_sd35_embedding_adapter")
    ap.add_argument("--category", type=str, default=None, help="optional: filter by CSV category")

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--scale", type=float, default=1.0)

    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--lambda_tri", type=float, default=1.0)
    ap.add_argument("--lambda_align", type=float, default=0.5)
    ap.add_argument("--lambda_benign", type=float, default=0.1)

    ap.add_argument("--benign_mode", type=str, default="rewritten", choices=["rewritten", "label0_prompt", "none"])
    ap.add_argument("--label0_value", type=str, default="0")

    ap.add_argument("--log_dir", type=str, default=None)
    ap.add_argument("--save_every", type=int, default=5)
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    cfg = TrainCfg(
        model_path=a.model_path,
        train_csv=a.train_csv,
        out_dir=a.out_dir,
        category=a.category,
        device=a.device,
        dtype=a.dtype,
        max_length=a.max_length,
        batch_size=a.batch_size,
        num_epochs=a.num_epochs,
        lr=a.lr,
        rank=a.rank,
        dropout=a.dropout,
        scale=a.scale,
        margin=a.margin,
        lambda_tri=a.lambda_tri,
        lambda_align=a.lambda_align,
        lambda_benign=a.lambda_benign,
        benign_mode=a.benign_mode,
        label0_value=a.label0_value,
        log_dir=a.log_dir,
        save_every=a.save_every,
    )
    train(cfg)
