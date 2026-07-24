#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, T5EncoderModel
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


# -------------------------
# Models
# -------------------------
class SafeAdapter(nn.Module):
    """
    Low-rank residual adapter:
      y = x + scale * Up(GELU(Down(LN(x))))
    """
    def __init__(self, hidden_size: int, rank: int = 256, dropout: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)  # start near-identity

    def forward(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        h = self.ln(x)
        h = self.down(h)
        h = F.gelu(h)
        h = self.dropout(h)
        h = self.up(h)
        return x + scale * h


class WrappedTextEncoder(nn.Module):
    def __init__(self, t5: T5EncoderModel, adapter: SafeAdapter):
        super().__init__()
        self.t5 = t5
        self.adapter = adapter


# -------------------------
# Dataset
# -------------------------
class YourCSVPairDataset(Dataset):
    """
    Your CSV columns:
      prompt, random_seed, label, id, category, origin_id, source, rewritten_prompt

    We map:
      malicious = prompt
      rewritten = rewritten_prompt

    benign is constructed by benign_source:
      - "rewritten": benign = rewritten_prompt (default, clean & stable)
      - "label0_prompt": benign sampled from prompts with label==0 (if exists)
      - "none": no benign constraint
    """
    def __init__(
        self,
        csv_path: str,
        benign_source: str = "rewritten",
        seed: int = 42,
        label0_value: str = "0",
    ):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)

        df = pd.read_csv(csv_path)

        required = ["prompt", "rewritten_prompt"]
        for c in required:
            if c not in df.columns:
                raise ValueError(f"CSV missing required column: {c}. Your CSV must have {required}.")

        self.prompt = df["prompt"].fillna("").astype(str).tolist()
        self.rewritten = df["rewritten_prompt"].fillna("").astype(str).tolist()

        self.benign_source = benign_source
        self.rng = random.Random(seed)

        # Build a pool for label==0 prompts if user chooses that benign source
        self.label0_pool: Optional[List[str]] = None
        if benign_source == "label0_prompt":
            if "label" not in df.columns:
                raise ValueError("benign_source=label0_prompt requires a 'label' column in CSV.")
            labels = df["label"].fillna("").astype(str).tolist()
            pool = [p for p, lab in zip(self.prompt, labels) if lab == str(label0_value) and p.strip()]
            if len(pool) == 0:
                raise ValueError(f"No prompts found with label=={label0_value}. Cannot use label0_prompt as benign.")
            self.label0_pool = pool

    def __len__(self):
        return len(self.prompt)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        m = self.prompt[idx].strip()
        r = self.rewritten[idx].strip()

        if self.benign_source == "rewritten":
            b = r  # treat rewritten as benign
        elif self.benign_source == "label0_prompt":
            b = self.rng.choice(self.label0_pool)  # random benign prompt
        elif self.benign_source == "none":
            b = None
        else:
            raise ValueError(f"Unknown benign_source: {self.benign_source}")

        return {"malicious": m, "rewritten": r, "benign": b}


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, List[Optional[str]]]:
    keys = batch[0].keys()
    return {k: [x[k] for x in batch] for k in keys}


# -------------------------
# Utils
# -------------------------
def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


# -------------------------
# Trainer
# -------------------------
@dataclass
class TrainArgs:
    model_path: str
    train_csv: str
    out_dir: str

    device: str
    dtype: str
    max_length: int
    batch_size: int
    num_epochs: int
    lr: float
    rank: int
    dropout: float

    margin: float
    lambda_tri: float
    lambda_align: float
    lambda_benign: float

    benign_source: str
    label0_value: str

    log_dir: Optional[str]
    save_every: int
    adapter_scale: float


class SafeAdapterTrainer:
    def __init__(self, args: TrainArgs):
        self.args = args
        os.makedirs(args.out_dir, exist_ok=True)

        # dtype
        if args.dtype == "fp16":
            self.torch_dtype = torch.float16
        elif args.dtype == "bf16":
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = torch.float32

        # SD3.5: tokenizer_3 / text_encoder_3 (T5-XXL)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, subfolder="tokenizer_3", use_fast=False)
        base = T5EncoderModel.from_pretrained(
            args.model_path, subfolder="text_encoder_3", torch_dtype=self.torch_dtype
        )

        # freeze T5
        base.requires_grad_(False)
        base.eval()

        hidden = base.config.d_model
        adapter = SafeAdapter(hidden_size=hidden, rank=args.rank, dropout=args.dropout)
        # keep adapter in the same dtype as base encoder to avoid BF16/FP32 mismatch
        adapter = adapter.to(dtype=self.torch_dtype)
        self.model = WrappedTextEncoder(base, adapter).to(args.device)

        self.opt = torch.optim.AdamW(
            self.model.adapter.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0
        )

        self._step = 0

    @torch.no_grad()
    def _encode_base(self, texts: List[str]) -> torch.Tensor:
        batch = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.args.max_length, return_tensors="pt"
        ).to(self.args.device)
        h = self.model.t5(**batch).last_hidden_state
        z = mean_pool(h, batch.attention_mask)
        return z

    def _encode_adapted(self, texts: List[str]) -> torch.Tensor:
        batch = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.args.max_length, return_tensors="pt"
        ).to(self.args.device)

        with torch.no_grad():
            h0 = self.model.t5(**batch).last_hidden_state

        h = self.model.adapter(h0, scale=self.args.adapter_scale)
        z = mean_pool(h, batch.attention_mask)
        return z

    def step(self, malicious: List[str], rewritten: List[str], benign: List[Optional[str]]):
        # anchor / positive (both adapted)
        z_a = self._encode_adapted(malicious)
        z_p = self._encode_adapted(rewritten)

        # negative reference: base(malicious)
        z_n0 = self._encode_base(malicious)

        d_ap = F.pairwise_distance(z_a, z_p)
        d_an = F.pairwise_distance(z_a, z_n0)

        loss_tri = F.relu(d_ap - d_an + self.args.margin).mean()
        loss_align = F.mse_loss(z_a, z_p)

        # benign identity constraint
        benign_valid = [b for b in benign if isinstance(b, str) and b.strip()]
        if benign_valid and self.args.lambda_benign > 0 and self.args.benign_source != "none":
            z_b_ad = self._encode_adapted(benign_valid)
            z_b0 = self._encode_base(benign_valid)
            loss_ben = F.mse_loss(z_b_ad, z_b0)
        else:
            loss_ben = torch.tensor(0.0, device=self.args.device)

        loss = (
            self.args.lambda_tri * loss_tri
            + self.args.lambda_align * loss_align
            + self.args.lambda_benign * loss_ben
        )

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.adapter.parameters(), max_norm=1.0)
        self.opt.step()

        self._step += 1

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
        torch.save(self.model.adapter.state_dict(), path)


def train(args: TrainArgs):
    log_dir = args.log_dir or os.path.join(args.out_dir, "runs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[TB] tensorboard --logdir {log_dir}")

    trainer = SafeAdapterTrainer(args)

    dataset = YourCSVPairDataset(
        csv_path=args.train_csv,
        benign_source=args.benign_source,
        seed=42,
        label0_value=args.label0_value,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)

    global_step = 0
    for epoch in range(args.num_epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.num_epochs}", dynamic_ncols=True)
        epoch_logs = {k: [] for k in ["loss", "tri", "align", "benign", "d_ap", "d_an"]}

        for batch in pbar:
            logs = trainer.step(batch["malicious"], batch["rewritten"], batch["benign"])

            for k in epoch_logs:
                epoch_logs[k].append(logs[k])

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

        avg = {k: sum(v) / max(len(v), 1) for k, v in epoch_logs.items()}
        for k, v in avg.items():
            writer.add_scalar(f"epoch/{k}", v, epoch)

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            ckpt = os.path.join(args.out_dir, f"safe_adapter_epoch{epoch+1}.pt")
            trainer.save(ckpt)
            print(f"[SAVE] {ckpt}")

    final_path = os.path.join(args.out_dir, "safe_adapter_final.pt")
    trainer.save(final_path)
    writer.close()
    print(f"[DONE] saved adapter: {final_path}")


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_path", type=str, default="/home/raykr/models/stabilityai/stable-diffusion-3.5-large")
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="./out_sd35_safe_adapter")

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--rank", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.0)

    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--lambda_tri", type=float, default=1.0)
    ap.add_argument("--lambda_align", type=float, default=0.5)
    ap.add_argument("--lambda_benign", type=float, default=0.1)

    ap.add_argument(
        "--benign_source",
        type=str,
        default="rewritten",
        choices=["rewritten", "label0_prompt", "none"],
        help="how to construct benign samples: rewritten (default) | label0_prompt | none",
    )
    ap.add_argument("--label0_value", type=str, default="0", help="which label value indicates benign when benign_source=label0_prompt")

    ap.add_argument("--adapter_scale", type=float, default=1.0)
    ap.add_argument("--log_dir", type=str, default=None)
    ap.add_argument("--save_every", type=int, default=5)

    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    args = TrainArgs(
        model_path=a.model_path,
        train_csv=a.train_csv,
        out_dir=a.out_dir,
        device=a.device,
        dtype=a.dtype,
        max_length=a.max_length,
        batch_size=a.batch_size,
        num_epochs=a.num_epochs,
        lr=a.lr,
        rank=a.rank,
        dropout=a.dropout,
        margin=a.margin,
        lambda_tri=a.lambda_tri,
        lambda_align=a.lambda_align,
        lambda_benign=a.lambda_benign,
        benign_source=a.benign_source,
        label0_value=a.label0_value,
        log_dir=a.log_dir,
        save_every=a.save_every,
        adapter_scale=a.adapter_scale,
    )
    train(args)
