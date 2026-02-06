#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
from safetensors.torch import save_file

from transformers import AutoTokenizer, T5EncoderModel, CLIPTextModel
from transformers.modeling_outputs import BaseModelOutputWithPooling

try:
    # Newer transformers
    from transformers.masking_utils import create_causal_mask
except ImportError:  # pragma: no cover
    try:
        # Older transformers
        from transformers.modeling_attn_mask_utils import create_causal_mask
    except ImportError:  # pragma: no cover
        create_causal_mask = None


# -------------------------
# Low-rank embedding adapter
# -------------------------
class LowRankEmbeddingAdapter(nn.Module):
    def __init__(self, dim: int, rank: int, dropout: float = 0.0):
        super().__init__()
        self.A = nn.Linear(dim, rank, bias=False)
        self.B = nn.Linear(rank, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.A.weight, std=0.02)
        nn.init.zeros_(self.B.weight)  # near-identity

    def forward(self, emb: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        emb = emb.to(dtype=self.A.weight.dtype)
        h = self.A(emb)
        h = F.gelu(h)
        h = self.dropout(h)
        delta = self.B(h)
        return emb + scale * delta


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def clip_pool(model_out, attention_mask: torch.Tensor) -> torch.Tensor:
    if hasattr(model_out, "pooler_output") and model_out.pooler_output is not None:
        return model_out.pooler_output
    h = model_out.last_hidden_state
    idx = attention_mask.long().sum(dim=1) - 1
    idx = idx.clamp(min=0)
    return h[torch.arange(h.size(0), device=h.device), idx]


def l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def inbatch_hard_neg(anchor: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
    """
    anchor: [B,D], cand: [B,D]  (candidates for negatives)
    Return hardest negative for each anchor from cand excluding itself:
    pick j that maximizes cosine similarity (hardest), i.e. minimal distance after normalize.
    """
    # cosine sim with normalized features
    a = l2_normalize(anchor)
    c = l2_normalize(cand)
    sim = a @ c.t()  # [B,B]
    # mask diagonal (avoid choosing itself)
    sim.fill_diagonal_(-1e9)
    idx = sim.argmax(dim=1)  # [B]
    neg = cand[idx]          # [B,D]
    return neg


class EncoderWithEmbAdapter(nn.Module):
    def __init__(self, encoder: nn.Module, adapter: LowRankEmbeddingAdapter):
        super().__init__()
        self.encoder = encoder
        self.adapter = adapter

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, scale: float):
        # Base token embeddings
        emb_layer = self.encoder.get_input_embeddings()
        emb = emb_layer(input_ids)
        emb = self.adapter(emb, scale=scale)

        # CLIPTextModel does NOT accept `inputs_embeds` in its public forward.
        # Detect CLIPTextModel via the presence of `text_model` and call its
        # internal transformer stack manually.
        if hasattr(self.encoder, "text_model"):
            tm = self.encoder.text_model  # CLIPTextTransformer

            # Add positional embeddings (normally done inside CLIPTextTransformer.forward)
            seq_length = emb.size(1)
            pos_ids = tm.embeddings.position_ids[:, :seq_length].to(emb.device)
            pos_emb = tm.embeddings.position_embedding(pos_ids)
            hidden_states = emb + pos_emb

            # Build causal + padding attention mask
            if create_causal_mask is not None:
                attn_mask = create_causal_mask(
                    config=tm.config,
                    input_embeds=hidden_states,
                    attention_mask=attention_mask,
                    cache_position=torch.arange(seq_length, device=hidden_states.device),
                    past_key_values=None,
                )
            else:
                # Fallback: simple 4D causal + padding mask
                dtype = hidden_states.dtype
                bsz = hidden_states.size(0)
                # [L, L] causal mask with -inf above diagonal
                causal = torch.triu(
                    torch.full(
                        (seq_length, seq_length),
                        float("-inf"),
                        device=hidden_states.device,
                        dtype=dtype,
                    ),
                    diagonal=1,
                )
                # padding mask: 0 for keep, -inf for pad
                pad = (1.0 - attention_mask.to(dtype=dtype)) * torch.finfo(dtype).min  # [B, L]
                causal = causal.unsqueeze(0).unsqueeze(0)  # [1,1,L,L]
                pad = pad.view(bsz, 1, 1, seq_length)      # [B,1,1,L]
                attn_mask = causal + pad                  # [B,1,L,L]

            enc_out = tm.encoder(inputs_embeds=hidden_states, attention_mask=attn_mask)
            last_hidden_state = tm.final_layer_norm(enc_out.last_hidden_state)

            # Reproduce CLIPTextTransformer pooling over the EOS token
            eos_token_id = getattr(tm.config, "eos_token_id", 2)
            input_ids_int = input_ids.to(dtype=torch.int, device=last_hidden_state.device)
            if eos_token_id == 2:
                # Old behavior: use argmax over sequence
                eos_pos = input_ids_int.argmax(dim=-1)
            else:
                # First position of eos_token_id (may share id with pad)
                eos_pos = (input_ids_int == eos_token_id).int().argmax(dim=-1)

            batch_indices = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
            pooled = last_hidden_state[batch_indices, eos_pos]

            out = BaseModelOutputWithPooling(
                last_hidden_state=last_hidden_state,
                pooler_output=pooled,
            )
            return out

        # T5EncoderModel (and similar) still accept `inputs_embeds`
        out = self.encoder(inputs_embeds=emb, attention_mask=attention_mask)
        return out


# -------------------------
# Dataset
# -------------------------
class PromptPairCSV(Dataset):
    def __init__(
        self,
        csv_path: str,
        category: Optional[str],
        benign_mode: str,
        label0_value: str,
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
                raise ValueError("benign_mode=label0_prompt requires 'label' column.")
            labels = df["label"].fillna("").astype(str).tolist()
            pool = [p for p, lab in zip(self.m, labels) if lab == str(label0_value) and p.strip()]
            if len(pool) == 0:
                raise ValueError(f"No samples with label=={label0_value} for benign pool.")
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
    return {k: [x[k] for x in batch] for k in batch[0].keys()}


# -------------------------
# Config
# -------------------------
@dataclass
class Cfg:
    model_path: str
    train_csv: str
    out_dir: str
    category: Optional[str]

    device: str
    dtype: str

    train_mode: str  # t5_only | clip_g_only | clip_l_only | all

    max_length_clip: int
    max_length_t5: int
    batch_size: int
    num_epochs: int
    lr: float

    rank_clip_l: int
    rank_clip_g: int
    rank_t5: int
    dropout: float
    scale: float

    margin: float
    lambda_tri: float
    lambda_align: float
    lambda_benign: float

    w_clip_l: float
    w_clip_g: float
    w_t5: float

    benign_mode: str
    label0_value: str

    log_dir: Optional[str]
    save_every: int


class MultiTrainer:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        os.makedirs(cfg.out_dir, exist_ok=True)

        if cfg.dtype == "fp16":
            self.torch_dtype = torch.float16
        elif cfg.dtype == "bf16":
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = torch.float32

        self.use_l = cfg.train_mode in ["clip_l_only", "all"]
        self.use_g = cfg.train_mode in ["clip_g_only", "all"]
        self.use_t5 = cfg.train_mode in ["t5_only", "all"]

        # tokenizers (only load what we need)
        if self.use_l:
            self.tok_l = AutoTokenizer.from_pretrained(cfg.model_path, subfolder="tokenizer", use_fast=False)
        if self.use_g:
            self.tok_g = AutoTokenizer.from_pretrained(cfg.model_path, subfolder="tokenizer_2", use_fast=False)
        if self.use_t5:
            self.tok_t5 = AutoTokenizer.from_pretrained(cfg.model_path, subfolder="tokenizer_3", use_fast=False)

        self.m_l = None
        self.m_g = None
        self.m_t5 = None

        params = []

        # CLIP-L
        if self.use_l:
            enc_l = CLIPTextModel.from_pretrained(cfg.model_path, subfolder="text_encoder", torch_dtype=self.torch_dtype)
            enc_l.requires_grad_(False).eval()
            dim_l = enc_l.config.hidden_size
            ad_l = LowRankEmbeddingAdapter(dim=dim_l, rank=cfg.rank_clip_l, dropout=cfg.dropout)
            self.m_l = EncoderWithEmbAdapter(enc_l, ad_l).to(cfg.device, dtype=self.torch_dtype)
            params += list(self.m_l.adapter.parameters())

        # CLIP-G
        if self.use_g:
            enc_g = CLIPTextModel.from_pretrained(cfg.model_path, subfolder="text_encoder_2", torch_dtype=self.torch_dtype)
            enc_g.requires_grad_(False).eval()
            dim_g = enc_g.config.hidden_size
            ad_g = LowRankEmbeddingAdapter(dim=dim_g, rank=cfg.rank_clip_g, dropout=cfg.dropout)
            self.m_g = EncoderWithEmbAdapter(enc_g, ad_g).to(cfg.device, dtype=self.torch_dtype)
            params += list(self.m_g.adapter.parameters())

        # T5
        if self.use_t5:
            enc_t5 = T5EncoderModel.from_pretrained(cfg.model_path, subfolder="text_encoder_3", torch_dtype=self.torch_dtype)
            enc_t5.requires_grad_(False).eval()
            dim_t5 = enc_t5.config.d_model
            ad_t5 = LowRankEmbeddingAdapter(dim=dim_t5, rank=cfg.rank_t5, dropout=cfg.dropout)
            self.m_t5 = EncoderWithEmbAdapter(enc_t5, ad_t5).to(cfg.device, dtype=self.torch_dtype)
            params += list(self.m_t5.adapter.parameters())

        self.opt = torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, 0.999), weight_decay=0.0)

    @torch.no_grad()
    def _encode_base(self, which: str, texts: List[str]) -> torch.Tensor:
        if which == "clip_l":
            tok, enc, maxlen = self.tok_l, self.m_l.encoder, self.cfg.max_length_clip
        elif which == "clip_g":
            tok, enc, maxlen = self.tok_g, self.m_g.encoder, self.cfg.max_length_clip
        elif which == "t5":
            tok, enc, maxlen = self.tok_t5, self.m_t5.encoder, self.cfg.max_length_t5
        else:
            raise ValueError(which)

        batch = tok(texts, padding=True, truncation=True, max_length=maxlen, return_tensors="pt").to(self.cfg.device)
        out = enc(**batch)
        if which.startswith("clip"):
            return clip_pool(out, batch.attention_mask)
        return mean_pool(out.last_hidden_state, batch.attention_mask)

    def _encode_adapted(self, which: str, texts: List[str]) -> torch.Tensor:
        if which == "clip_l":
            tok, mod, maxlen = self.tok_l, self.m_l, self.cfg.max_length_clip
        elif which == "clip_g":
            tok, mod, maxlen = self.tok_g, self.m_g, self.cfg.max_length_clip
        elif which == "t5":
            tok, mod, maxlen = self.tok_t5, self.m_t5, self.cfg.max_length_t5
        else:
            raise ValueError(which)

        batch = tok(texts, padding=True, truncation=True, max_length=maxlen, return_tensors="pt").to(self.cfg.device)
        out = mod(batch.input_ids, batch.attention_mask, scale=self.cfg.scale)
        if which.startswith("clip"):
            return clip_pool(out, batch.attention_mask)
        return mean_pool(out.last_hidden_state, batch.attention_mask)

    def _loss_one(self, which: str, mal: List[str], rew: List[str], benign: List[Optional[str]]):
        # Anchor: adapted(malicious)
        z_a = self._encode_adapted(which, mal)         # [B,D]

        # Positive: base(rewritten)  ✅ 关键：正样本不再过 adapter，避免“正样本也被一起移动”的投机解
        z_p = self._encode_base(which, rew)            # [B,D]

        # Negatives: in-batch negatives from rewritten (base)
        # Each anchor uses other samples' rewritten as negatives (semantic close but not identical)
        z_n_inbatch_all = z_p.detach()                 # [B,D], no grad

        # Option 1: hard negative (recommended)
        z_n = inbatch_hard_neg(z_a.detach(), z_n_inbatch_all)  # [B,D], choose hardest per-sample
        # If you want random in-batch neg instead, comment above and use below:
        # idx = torch.randperm(z_p.size(0), device=z_p.device)
        # z_n = z_p[idx].detach()

        # Normalize to fix scale mismatch across encoders
        z_a_n = l2_normalize(z_a)
        z_p_n = l2_normalize(z_p)
        z_n_n = l2_normalize(z_n)

        # Distances in normalized space (range ~[0,2])
        d_ap = F.pairwise_distance(z_a_n, z_p_n).mean()
        d_an = F.pairwise_distance(z_a_n, z_n_n).mean()

        # Triplet: keep active longer
        tri = F.relu(d_ap - d_an + self.cfg.margin)

        # Align in normalized space
        aln = F.mse_loss(z_a_n, z_p_n)

        # Benign: constrain adapted(benign) ~ base(benign) (also in normalized space)
        benign_valid = [b for b in benign if isinstance(b, str) and b.strip()]
        if benign_valid and self.cfg.lambda_benign > 0 and self.cfg.benign_mode != "none":
            z_b_ad = self._encode_adapted(which, benign_valid)
            z_b0 = self._encode_base(which, benign_valid)
            z_b_ad = l2_normalize(z_b_ad)
            z_b0 = l2_normalize(z_b0)
            ben = F.mse_loss(z_b_ad, z_b0)
        else:
            ben = torch.tensor(0.0, device=self.cfg.device)

        L = self.cfg.lambda_tri * tri + self.cfg.lambda_align * aln + self.cfg.lambda_benign * ben

        # extra diagnostics: active triplet ratio & gap
        gap = (d_an - d_ap).detach()
        active = ((d_ap - d_an + self.cfg.margin) > 0).float().mean().detach()

        return L, {
            "tri": float(tri.detach().cpu()),
            "align": float(aln.detach().cpu()),
            "benign": float(ben.detach().cpu()),
            "d_ap": float(d_ap.detach().cpu()),
            "d_an": float(d_an.detach().cpu()),
            "gap": float(gap.cpu()),
            "active": float(active.cpu()),
        }


    def step(self, mal: List[str], rew: List[str], benign: List[Optional[str]]):
        total = torch.tensor(0.0, device=self.cfg.device)
        logs: Dict[str, float] = {}

        if self.use_l:
            L_l, log_l = self._loss_one("clip_l", mal, rew, benign)
            w = 1.0 if self.cfg.train_mode == "clip_l_only" else self.cfg.w_clip_l
            total = total + w * L_l
            logs.update({f"clip_l/{k}": v for k, v in log_l.items()})
            logs["clip_l/loss"] = float(L_l.detach().cpu())

        if self.use_g:
            L_g, log_g = self._loss_one("clip_g", mal, rew, benign)
            w = 1.0 if self.cfg.train_mode == "clip_g_only" else self.cfg.w_clip_g
            total = total + w * L_g
            logs.update({f"clip_g/{k}": v for k, v in log_g.items()})
            logs["clip_g/loss"] = float(L_g.detach().cpu())

        if self.use_t5:
            L_t, log_t = self._loss_one("t5", mal, rew, benign)
            w = 1.0 if self.cfg.train_mode == "t5_only" else self.cfg.w_t5
            total = total + w * L_t
            logs.update({f"t5/{k}": v for k, v in log_t.items()})
            logs["t5/loss"] = float(L_t.detach().cpu())

        self.opt.zero_grad(set_to_none=True)
        total.backward()

        # clip grad
        params = []
        if self.use_l: params += list(self.m_l.adapter.parameters())
        if self.use_g: params += list(self.m_g.adapter.parameters())
        if self.use_t5: params += list(self.m_t5.adapter.parameters())
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

        self.opt.step()

        logs["loss"] = float(total.detach().cpu())
        return logs

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {}
        meta = {
            "train_mode": self.cfg.train_mode,
            "dropout": str(self.cfg.dropout),
            "scale": str(self.cfg.scale),
            "rank_clip_l": str(self.cfg.rank_clip_l),
            "rank_clip_g": str(self.cfg.rank_clip_g),
            "rank_t5": str(self.cfg.rank_t5),
        }
        if self.use_l:
            state["clip_l.A.weight"] = self.m_l.adapter.A.weight.detach().cpu()
            state["clip_l.B.weight"] = self.m_l.adapter.B.weight.detach().cpu()
        if self.use_g:
            state["clip_g.A.weight"] = self.m_g.adapter.A.weight.detach().cpu()
            state["clip_g.B.weight"] = self.m_g.adapter.B.weight.detach().cpu()
        if self.use_t5:
            state["t5.A.weight"] = self.m_t5.adapter.A.weight.detach().cpu()
            state["t5.B.weight"] = self.m_t5.adapter.B.weight.detach().cpu()

        save_file(state, path, metadata=meta)


def train(cfg: Cfg):
    log_dir = cfg.log_dir or os.path.join(cfg.out_dir, "runs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[TB] tensorboard --logdir {log_dir}")

    ds = PromptPairCSV(cfg.train_csv, cfg.category, cfg.benign_mode, cfg.label0_value)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)

    tr = MultiTrainer(cfg)

    global_step = 0
    for epoch in range(cfg.num_epochs):
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{cfg.num_epochs}", dynamic_ncols=True)
        for batch in pbar:
            logs = tr.step(batch["malicious"], batch["rewritten"], batch["benign"])
            for k, v in logs.items():
                writer.add_scalar(f"train/{k}", v, global_step)

            # compact postfix
            postfix = {"loss": f"{logs['loss']:.3f}"}
            if "t5/loss" in logs: postfix["t5"] = f"{logs['t5/loss']:.3f}"
            if "clip_g/loss" in logs: postfix["g"] = f"{logs['clip_g/loss']:.3f}"
            if "clip_l/loss" in logs: postfix["l"] = f"{logs['clip_l/loss']:.3f}"
            if "t5/active" in logs: postfix["t5_act"] = f"{logs['t5/active']:.2f}"
            if "clip_g/active" in logs: postfix["g_act"] = f"{logs['clip_g/active']:.2f}"
            if "clip_l/active" in logs: postfix["l_act"] = f"{logs['clip_l/active']:.2f}"
            pbar.set_postfix(postfix)

            global_step += 1

        if cfg.save_every > 0 and (epoch + 1) % cfg.save_every == 0:
            ckpt = os.path.join(cfg.out_dir, f"multi_emb_adapter_{cfg.train_mode}_epoch{epoch+1}.safetensors")
            tr.save(ckpt)
            print(f"[SAVE] {ckpt}")

    final_path = os.path.join(cfg.out_dir, f"multi_emb_adapter_{cfg.train_mode}_final.safetensors")
    tr.save(final_path)
    writer.close()
    print(f"[DONE] saved: {final_path}")


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_path", type=str, default="/home/raykr/models/stabilityai/stable-diffusion-3.5-large")
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="./out_sd35_multi_emb_adapter")
    ap.add_argument("--category", type=str, default=None)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])

    ap.add_argument("--train_mode", type=str, default="all",
                    choices=["t5_only", "clip_g_only", "clip_l_only", "all"])

    ap.add_argument("--max_length_clip", type=int, default=77)
    ap.add_argument("--max_length_t5", type=int, default=256)

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-4)

    ap.add_argument("--rank_clip_l", type=int, default=32)
    ap.add_argument("--rank_clip_g", type=int, default=32)
    ap.add_argument("--rank_t5", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--scale", type=float, default=1.0)

    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--lambda_tri", type=float, default=1.0)
    ap.add_argument("--lambda_align", type=float, default=0.5)
    ap.add_argument("--lambda_benign", type=float, default=0.1)

    ap.add_argument("--w_clip_l", type=float, default=0.2)
    ap.add_argument("--w_clip_g", type=float, default=0.3)
    ap.add_argument("--w_t5", type=float, default=0.5)

    ap.add_argument("--benign_mode", type=str, default="rewritten", choices=["rewritten", "label0_prompt", "none"])
    ap.add_argument("--label0_value", type=str, default="0")

    ap.add_argument("--log_dir", type=str, default=None)
    ap.add_argument("--save_every", type=int, default=5)

    a = ap.parse_args()
    return Cfg(
        model_path=a.model_path, train_csv=a.train_csv, out_dir=a.out_dir, category=a.category,
        device=a.device, dtype=a.dtype,
        train_mode=a.train_mode,
        max_length_clip=a.max_length_clip, max_length_t5=a.max_length_t5,
        batch_size=a.batch_size, num_epochs=a.num_epochs, lr=a.lr,
        rank_clip_l=a.rank_clip_l, rank_clip_g=a.rank_clip_g, rank_t5=a.rank_t5,
        dropout=a.dropout, scale=a.scale,
        margin=a.margin, lambda_tri=a.lambda_tri, lambda_align=a.lambda_align, lambda_benign=a.lambda_benign,
        w_clip_l=a.w_clip_l, w_clip_g=a.w_clip_g, w_t5=a.w_t5,
        benign_mode=a.benign_mode, label0_value=a.label0_value,
        log_dir=a.log_dir, save_every=a.save_every
    )


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
