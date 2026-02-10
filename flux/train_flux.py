#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

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
except ImportError:
    try:
        # Older transformers
        from transformers.modeling_attn_mask_utils import create_causal_mask
    except ImportError:
        create_causal_mask = None


# -------------------------
# Low-rank embedding adapter
# -------------------------
class LowRankEmbeddingAdapter(nn.Module):
    """
    Embedding-space low-rank residual:
        emb' = emb + scale * B(GELU(A(emb)))
    Init: A small, B zeros => near-identity at start.
    """
    def __init__(self, dim: int, rank: int, dropout: float = 0.0, init_std: float = 1e-3):
        super().__init__()
        self.A = nn.Linear(dim, rank, bias=False)
        self.B = nn.Linear(rank, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.A.weight, std=init_std)
        nn.init.zeros_(self.B.weight)

    def forward(self, emb: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        in_dtype = emb.dtype
        # Use adapter weight dtype so mat1/mat2 dtypes match (e.g. bf16 encoder)
        w_dtype = self.A.weight.dtype
        x = emb.to(w_dtype)
        h = self.A(x)
        h = F.gelu(h)
        h = self.dropout(h)
        delta = self.B(h)
        y = x + float(scale) * delta
        return y.to(in_dtype)


def l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def masked_mse(a: torch.Tensor, b: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    a,b: [B,L,D], attention_mask: [B,L] (1 for valid, 0 for pad)
    """
    m = attention_mask.unsqueeze(-1).to(a.dtype)  # [B,L,1]
    diff = (a - b) * m
    denom = m.sum().clamp(min=1.0)
    return (diff.pow(2).sum() / denom)


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def clip_pool_from_last(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Stable CLIP pooled vector: take last non-pad token by attention_mask.
    """
    idx = attention_mask.long().sum(dim=1) - 1
    idx = idx.clamp(min=0)
    b = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[b, idx]


# -------------------------
# Encoder wrapper that injects into input embeddings (match training/inference)
# -------------------------
class EncoderWithEmbAdapter(nn.Module):
    def __init__(self, encoder: nn.Module, adapter: LowRankEmbeddingAdapter):
        super().__init__()
        self.encoder = encoder
        self.adapter = adapter

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, scale: float):
        # 1) raw token embeddings
        emb_layer = self.encoder.get_input_embeddings()
        emb = emb_layer(input_ids)

        # 2) inject adapter
        emb = self.adapter(emb, scale=scale)

        # 3) forward through encoder stack
        if hasattr(self.encoder, "text_model"):
            # CLIPTextModel: we re-run text_model with inputs_embeds
            tm = self.encoder.text_model
            seq_length = emb.size(1)

            # position embeddings
            pos_ids = tm.embeddings.position_ids[:, :seq_length].to(emb.device)
            pos_emb = tm.embeddings.position_embedding(pos_ids)
            hidden_states = emb + pos_emb

            # attention mask (causal + padding) for CLIP text transformer
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

            pooled = clip_pool_from_last(last_hidden_state, attention_mask)

            return BaseModelOutputWithPooling(
                last_hidden_state=last_hidden_state,
                pooler_output=pooled,
            )

        # T5EncoderModel supports inputs_embeds directly
        return self.encoder(inputs_embeds=emb, attention_mask=attention_mask)


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

    train_mode: str  # t5_only | clip_l_only | all

    max_length_clip: int
    max_length_t5: int
    batch_size: int
    num_epochs: int
    lr: float

    rank_clip_l: int
    rank_t5: int
    dropout: float
    scale: float

    margin: float
    lambda_tri: float
    lambda_align: float
    lambda_benign: float

    w_clip_l: float
    w_t5: float

    benign_mode: str
    label0_value: str

    log_dir: Optional[str]
    save_every: int

    # --- new knobs (kept default; not exposed in CLI to keep your args unchanged) ---
    # token-level alignment
    use_token_align: bool = True
    lambda_token_align: float = 0.5

    # in-batch InfoNCE
    use_inbatch_nce: bool = False
    lambda_nce: float = 0.5
    nce_temp: float = 0.07


class FluxMultiTrainer:
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
        self.use_t5 = cfg.train_mode in ["t5_only", "all"]

        # Tokenizers
        if self.use_l:
            self.tok_l = AutoTokenizer.from_pretrained(cfg.model_path, subfolder="tokenizer", use_fast=False)
        if self.use_t5:
            self.tok_t5 = AutoTokenizer.from_pretrained(cfg.model_path, subfolder="tokenizer_2", use_fast=False)

        self.m_l = None
        self.m_t5 = None
        params = []

        # CLIP-L
        if self.use_l:
            print("Loading CLIP-L (text_encoder)...")
            enc_l = CLIPTextModel.from_pretrained(cfg.model_path, subfolder="text_encoder", torch_dtype=self.torch_dtype)
            enc_l.requires_grad_(False).eval()
            dim_l = enc_l.config.hidden_size
            ad_l = LowRankEmbeddingAdapter(dim=dim_l, rank=cfg.rank_clip_l, dropout=cfg.dropout, init_std=1e-3)
            self.m_l = EncoderWithEmbAdapter(enc_l, ad_l).to(cfg.device, dtype=self.torch_dtype)
            params += list(self.m_l.adapter.parameters())

        # T5
        if self.use_t5:
            print("Loading T5 (text_encoder_2)...")
            enc_t5 = T5EncoderModel.from_pretrained(cfg.model_path, subfolder="text_encoder_2", torch_dtype=self.torch_dtype)
            enc_t5.requires_grad_(False).eval()
            dim_t5 = enc_t5.config.d_model
            ad_t5 = LowRankEmbeddingAdapter(dim=dim_t5, rank=cfg.rank_t5, dropout=cfg.dropout, init_std=1e-3)
            self.m_t5 = EncoderWithEmbAdapter(enc_t5, ad_t5).to(cfg.device, dtype=self.torch_dtype)
            params += list(self.m_t5.adapter.parameters())

        self.opt = torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, 0.999), weight_decay=0.0)

    def _tokenize(self, which: str, texts: List[str]):
        if which == "clip_l":
            tok, maxlen = self.tok_l, self.cfg.max_length_clip
        elif which == "t5":
            tok, maxlen = self.tok_t5, self.cfg.max_length_t5
        else:
            raise ValueError(which)

        # Use max_length padding so all batches have same L (needed for token-level alignment)
        batch = tok(texts, padding="max_length", truncation=True, max_length=maxlen, return_tensors="pt")
        return batch.to(self.cfg.device)

    @torch.no_grad()
    def _encode_base(self, which: str, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          sent: [B,D] pooled sentence vector (for triplet/align)
          last: [B,L,D] last_hidden_state
          mask: [B,L] attention_mask
        """
        if which == "clip_l":
            enc = self.m_l.encoder
        elif which == "t5":
            enc = self.m_t5.encoder
        else:
            raise ValueError(which)

        batch = self._tokenize(which, texts)
        out = enc(**batch)

        if which == "clip_l":
            last = out.last_hidden_state
            sent = clip_pool_from_last(last, batch.attention_mask)
        else:
            last = out.last_hidden_state
            sent = mean_pool(last, batch.attention_mask)

        return sent, last, batch.attention_mask

    def _encode_adapted(self, which: str, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Same outputs as _encode_base but using injected adapter.
        """
        if which == "clip_l":
            mod = self.m_l
        elif which == "t5":
            mod = self.m_t5
        else:
            raise ValueError(which)

        batch = self._tokenize(which, texts)
        out = mod(batch.input_ids, batch.attention_mask, scale=self.cfg.scale)

        if which == "clip_l":
            last = out.last_hidden_state
            sent = clip_pool_from_last(last, batch.attention_mask)
        else:
            last = out.last_hidden_state
            sent = mean_pool(last, batch.attention_mask)

        return sent, last, batch.attention_mask

    def _inbatch_nce(self, z_a: torch.Tensor, z_p: torch.Tensor) -> torch.Tensor:
        """
        InfoNCE: positives are diagonal (a_i with p_i), negatives are p_j (j!=i).
        z_a, z_p: [B,D]
        """
        a = l2_normalize(z_a.float())
        p = l2_normalize(z_p.float())
        logits = (a @ p.t()) / float(self.cfg.nce_temp)  # [B,B]
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)

    def _loss_one(self, which: str, mal: List[str], rew: List[str], benign: List[Optional[str]]):
        # Anchor: adapted(malicious)
        z_a, last_a, mask_a = self._encode_adapted(which, mal)   # [B,D], [B,L,D], [B,L]

        # Positive: base(rewritten)
        z_p, last_p, mask_p = self._encode_base(which, rew)

        # Negative (FIXED): base(malicious)
        z_n, _, _ = self._encode_base(which, mal)
        z_n = z_n.detach()

        # Normalize
        z_a_n = l2_normalize(z_a)
        z_p_n = l2_normalize(z_p)
        z_n_n = l2_normalize(z_n)

        # Triplet (per-sample, then mean)
        d_ap = F.pairwise_distance(z_a_n, z_p_n)  # [B]
        d_an = F.pairwise_distance(z_a_n, z_n_n)  # [B]
        tri = F.relu(d_ap - d_an + self.cfg.margin).mean()

        # Align: cosine (more stable than MSE in semantic space)
        aln = (1.0 - (z_a_n * z_p_n).sum(dim=1)).mean()

        # Token-level alignment (optional, stronger for subject semantics)
        if self.cfg.use_token_align and self.cfg.lambda_token_align > 0:
            # Need same length masks; they are produced by the same tokenizer for each branch.
            tok_aln = masked_mse(last_a.float(), last_p.float(), mask_a)  # use mask of anchor
        else:
            tok_aln = torch.tensor(0.0, device=self.cfg.device)

        # In-batch NCE (optional)
        if self.cfg.use_inbatch_nce and self.cfg.lambda_nce > 0:
            nce = self._inbatch_nce(z_a, z_p)
        else:
            nce = torch.tensor(0.0, device=self.cfg.device)

        # Benign preservation
        benign_valid = [b for b in benign if isinstance(b, str) and b.strip()]
        if benign_valid and self.cfg.lambda_benign > 0 and self.cfg.benign_mode != "none":
            z_b_ad, last_b_ad, mask_b = self._encode_adapted(which, benign_valid)
            z_b0,   last_b0,   _      = self._encode_base(which, benign_valid)
            z_b_ad = l2_normalize(z_b_ad)
            z_b0   = l2_normalize(z_b0)
            ben = F.mse_loss(z_b_ad, z_b0)
        else:
            ben = torch.tensor(0.0, device=self.cfg.device)

        L = (
            self.cfg.lambda_tri * tri
            + self.cfg.lambda_align * aln
            + self.cfg.lambda_token_align * tok_aln
            + self.cfg.lambda_nce * nce
            + self.cfg.lambda_benign * ben
        )

        gap = (d_an.mean() - d_ap.mean()).detach()
        active = ((d_ap - d_an + self.cfg.margin) > 0).float().mean().detach()

        return L, {
            "tri": float(tri.detach().cpu()),
            "align": float(aln.detach().cpu()),
            "tok_align": float(tok_aln.detach().cpu()),
            "nce": float(nce.detach().cpu()),
            "benign": float(ben.detach().cpu()),
            "d_ap": float(d_ap.mean().detach().cpu()),
            "d_an": float(d_an.mean().detach().cpu()),
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

        if self.use_t5:
            L_t, log_t = self._loss_one("t5", mal, rew, benign)
            w = 1.0 if self.cfg.train_mode == "t5_only" else self.cfg.w_t5
            total = total + w * L_t
            logs.update({f"t5/{k}": v for k, v in log_t.items()})
            logs["t5/loss"] = float(L_t.detach().cpu())

        self.opt.zero_grad(set_to_none=True)
        total.backward()

        params = []
        if self.use_l: params += list(self.m_l.adapter.parameters())
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
            "rank_clip_l": str(self.cfg.rank_clip_l),
            "rank_t5": str(self.cfg.rank_t5),
            "use_token_align": str(self.cfg.use_token_align),
            "use_inbatch_nce": str(self.cfg.use_inbatch_nce),
        }
        if self.use_l:
            state["clip_l.A.weight"] = self.m_l.adapter.A.weight.detach().cpu()
            state["clip_l.B.weight"] = self.m_l.adapter.B.weight.detach().cpu()
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

    tr = FluxMultiTrainer(cfg)

    global_step = 0
    for epoch in range(cfg.num_epochs):
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{cfg.num_epochs}", dynamic_ncols=True)
        for batch in pbar:
            logs = tr.step(batch["malicious"], batch["rewritten"], batch["benign"])
            for k, v in logs.items():
                writer.add_scalar(f"train/{k}", v, global_step)

            postfix = {"loss": f"{logs['loss']:.3f}"}
            if "t5/loss" in logs: postfix["t5"] = f"{logs['t5/loss']:.3f}"
            if "clip_l/loss" in logs: postfix["l"] = f"{logs['clip_l/loss']:.3f}"
            if "t5/tok_align" in logs: postfix["t5_tok"] = f"{logs['t5/tok_align']:.3f}"
            if "clip_l/tok_align" in logs: postfix["l_tok"] = f"{logs['clip_l/tok_align']:.3f}"
            if "t5/active" in logs: postfix["t5_act"] = f"{logs['t5/active']:.2f}"
            pbar.set_postfix(postfix)

            global_step += 1

        if cfg.save_every > 0 and (epoch + 1) % cfg.save_every == 0:
            ckpt = os.path.join(cfg.out_dir, f"flux_emb_adapter_{cfg.train_mode}_epoch{epoch+1}.safetensors")
            tr.save(ckpt)
            print(f"[SAVE] {ckpt}")

    final_path = os.path.join(cfg.out_dir, f"flux_emb_adapter_{cfg.train_mode}_final.safetensors")
    tr.save(final_path)
    writer.close()
    print(f"[DONE] saved: {final_path}")


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_path", type=str, default="black-forest-labs/FLUX.1-schnell")
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="./out_flux_adapter")
    ap.add_argument("--category", type=str, default=None)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])

    ap.add_argument("--train_mode", type=str, default="all",
                    choices=["t5_only", "clip_l_only", "all"])

    ap.add_argument("--max_length_clip", type=int, default=77)
    ap.add_argument("--max_length_t5", type=int, default=256)

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-4)

    ap.add_argument("--rank_clip_l", type=int, default=32)
    ap.add_argument("--rank_t5", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--scale", type=float, default=1.0)

    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--lambda_tri", type=float, default=1.0)
    ap.add_argument("--lambda_align", type=float, default=0.5)
    ap.add_argument("--lambda_benign", type=float, default=0.1)

    ap.add_argument("--w_clip_l", type=float, default=0.3)
    ap.add_argument("--w_t5", type=float, default=0.7)

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
        rank_clip_l=a.rank_clip_l, rank_t5=a.rank_t5,
        dropout=a.dropout, scale=a.scale,
        margin=a.margin, lambda_tri=a.lambda_tri, lambda_align=a.lambda_align, lambda_benign=a.lambda_benign,
        w_clip_l=a.w_clip_l, w_t5=a.w_t5,
        benign_mode=a.benign_mode, label0_value=a.label0_value,
        log_dir=a.log_dir, save_every=a.save_every,
        # keep defaults of new knobs (no CLI change)
        use_token_align=True,
        lambda_token_align=0.5,
        use_inbatch_nce=False,
        lambda_nce=0.5,
        nce_temp=0.07,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
