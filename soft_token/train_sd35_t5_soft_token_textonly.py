#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Text-only soft token embedding training for SD3.5 (T5-XXL: text_encoder_3).
- No diffusion model, no denoiser, no VAE.
- Train only the placeholder token embedding in tokenizer_3 / text_encoder_3.
- Loss: align + preserve (+ optional triplet).
- Output: learned_embeds_t5_textonly.safetensors
"""

import os
import json
import math
import argparse
import logging
from dataclasses import dataclass
from itertools import cycle
from typing import Dict, Any, List, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.logging import get_logger

from transformers import AutoTokenizer, T5EncoderModel
from diffusers.optimization import get_scheduler
from safetensors.torch import save_file, load_file


logger = get_logger(__name__, log_level="INFO")


# -------------------------
# Utils
# -------------------------
def mean_pool(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    """
    last_hidden: [B, L, D]
    attn_mask:  [B, L] (1 for real tokens)
    """
    m = attn_mask.unsqueeze(-1).to(last_hidden.dtype)  # [B, L, 1]
    summed = (last_hidden * m).sum(dim=1)              # [B, D]
    denom = m.sum(dim=1).clamp(min=1e-6)               # [B, 1]
    return summed / denom


def pick_single_token_initializer(tokenizer, preferred: str) -> str:
    """
    T5 SentencePiece often tokenizes "safe" into multiple pieces.
    We try a list of candidates until we find one that maps to exactly one token id.
    """
    candidates = [preferred, "▁" + preferred, "good", "▁good", "nice", "▁nice", "a", "▁a", "the", "▁the", "."]
    for cand in candidates:
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if len(ids) == 1:
            return cand
    # fallback: take tokenizer's most common "▁" token? but still ensure single id
    # If nothing works, this environment is unusual.
    raise ValueError(
        "Cannot find a single-token initializer. "
        "Try passing --initializer_token with a token that maps to exactly one token in tokenizer_3."
    )


def insert_placeholder(prompt: str, placeholder_str: str, position: str) -> str:
    """
    Insert placeholder token string into prompt.
    placeholder_str may contain multiple tokens (e.g., "<SAFE> <SAFE_1>").
    """
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


# -------------------------
# Dataset
# -------------------------
class PromptPairCSV(Dataset):
    """
    CSV expected columns:
      - prompt
      - rewritten_prompt
    Optional:
      - pseudo_prompt
      - pseudo_rewritten
    """
    def __init__(self, csv_path: str, placeholder_str: str, position: str, repeats: int = 1):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)
        df = pd.read_csv(csv_path)

        required = ["prompt", "rewritten_prompt"]
        for c in required:
            if c not in df.columns:
                raise ValueError(f"CSV missing required column: {c}")

        self.prompts = df["prompt"].fillna("").astype(str).tolist()
        self.rewritten = df["rewritten_prompt"].fillna("").astype(str).tolist()

        self.pseudo_prompt = df["pseudo_prompt"].fillna("").astype(str).tolist() if "pseudo_prompt" in df.columns else None
        self.pseudo_rewritten = df["pseudo_rewritten"].fillna("").astype(str).tolist() if "pseudo_rewritten" in df.columns else None

        self.placeholder_str = placeholder_str
        self.position = position
        self.repeats = max(1, int(repeats))
        self.n = len(self.prompts)

    def __len__(self):
        return self.n * self.repeats

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        i = idx % self.n
        toxic = self.prompts[i]
        safe = self.rewritten[i]

        pt = self.pseudo_prompt[i] if self.pseudo_prompt is not None else toxic
        pb = self.pseudo_rewritten[i] if self.pseudo_rewritten is not None else safe

        # We will create BOTH versions:
        #   - with placeholder: used to "sanitize"
        #   - without placeholder: used as target / preserve reference
        toxic_with = insert_placeholder(toxic, self.placeholder_str, self.position)
        safe_with = insert_placeholder(safe, self.placeholder_str, self.position)
        pt_with = insert_placeholder(pt, self.placeholder_str, self.position)
        pb_with = insert_placeholder(pb, self.placeholder_str, self.position)

        return {
            "toxic": toxic,
            "safe": safe,
            "toxic_with": toxic_with,
            "safe_with": safe_with,
            "pseudo_toxic_with": pt_with,
            "pseudo_benign": pb,        # keep as plain target by default
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    keys = batch[0].keys()
    out = {}
    for k in keys:
        out[k] = [b[k] for b in batch]
    return out


# -------------------------
# Save learned embeds
# -------------------------
def save_learned_embeds(
    path: str,
    token_to_id: Dict[str, int],
    embed_weight: torch.Tensor,
    placeholder_tokens: List[str],
):
    """
    Save only placeholder token embeddings to safetensors.
    """
    tensors = {}
    for tok in placeholder_tokens:
        tid = token_to_id[tok]
        tensors[tok] = embed_weight[tid].detach().cpu()
    save_file(tensors, path)


# -------------------------
# Trainer
# -------------------------
@dataclass
class TrainConfig:
    model_path: str
    output_dir: str
    train_csv: str

    seed: int
    mixed_precision: str
    grad_accum: int
    batch_size: int
    num_workers: int

    max_length: int
    lr: float
    lr_scheduler: str
    lr_warmup_steps: int
    max_steps: int

    placeholder_token: str
    num_vectors: int
    initializer_token: str
    position: str
    repeats: int

    lambda_align: float
    lambda_preserve: float
    lambda_triplet: float
    margin_coef: float

    checkpoint_steps: int
    resume: bool


def _latest_checkpoint_dir(output_dir: str) -> Optional[str]:
    """Return path to checkpoint-{step} with largest step, or None."""
    import glob
    pattern = os.path.join(output_dir, "checkpoint-*")
    dirs = glob.glob(pattern)
    if not dirs:
        return None
    def step_from_path(p):
        try:
            return int(p.rsplit("-", 1)[-1])
        except ValueError:
            return -1
    return max(dirs, key=step_from_path)


class TextOnlySoftTokenTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg

        if cfg.seed is not None:
            set_seed(cfg.seed)

        logging_dir = os.path.join(cfg.output_dir, "logs")
        proj = ProjectConfiguration(project_dir=cfg.output_dir, logging_dir=logging_dir)
        self.acc = Accelerator(
            gradient_accumulation_steps=cfg.grad_accum,
            mixed_precision=cfg.mixed_precision,
            project_config=proj,
            log_with="tensorboard",
        )

        os.makedirs(cfg.output_dir, exist_ok=True)
        if self.acc.is_main_process:
            with open(os.path.join(cfg.output_dir, "config.json"), "w") as f:
                json.dump(cfg.__dict__, f, indent=2, ensure_ascii=False)

        # weight dtype
        self.weight_dtype = torch.float32
        if self.acc.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.acc.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        self._setup_model_and_tokenizer()
        self._setup_placeholder()
        self._setup_data()
        self._setup_optim()

        # prepare
        self.text_encoder, self.optimizer, self.train_loader, self.lr_sched = self.acc.prepare(
            self.text_encoder, self.optimizer, self.train_loader, self.lr_sched
        )

        self.global_step = 0
        if self.cfg.resume:
            self._resume_from_checkpoint()

        if self.acc.is_main_process:
            self.acc.init_trackers("sd35_t5_textonly", config=cfg.__dict__)

    def _setup_model_and_tokenizer(self):
        # SD3.5 T5 lives in subfolder text_encoder_3 / tokenizer_3
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_path, subfolder="tokenizer_3", use_fast=False
        )
        self.text_encoder = T5EncoderModel.from_pretrained(
            self.cfg.model_path, subfolder="text_encoder_3", torch_dtype=self.weight_dtype
        )

        # freeze all
        self.text_encoder.requires_grad_(False)
        # Triplet needs extra T5 forwards; use gradient checkpointing to trade compute for memory
        # if self.cfg.lambda_triplet > 0:
        #     self.text_encoder.gradient_checkpointing_enable()

    def _setup_placeholder(self):
        # build placeholder tokens list
        self.placeholder_tokens = [self.cfg.placeholder_token]
        for i in range(1, self.cfg.num_vectors):
            self.placeholder_tokens.append(f"{self.cfg.placeholder_token}_{i}")

        # add tokens
        num_added = self.tokenizer.add_tokens(self.placeholder_tokens)
        if num_added != self.cfg.num_vectors:
            raise ValueError(
                f"Tokenizer already contains some of these placeholder tokens. "
                f"Added {num_added}, expected {self.cfg.num_vectors}."
            )

        # pick initializer token that maps to a single token id
        init_tok = pick_single_token_initializer(self.tokenizer, self.cfg.initializer_token)
        if init_tok != self.cfg.initializer_token and self.acc.is_main_process:
            logger.warning(f"initializer_token '{self.cfg.initializer_token}' is not single-token; using '{init_tok}'")
        self.cfg.initializer_token = init_tok

        init_ids = self.tokenizer.encode(self.cfg.initializer_token, add_special_tokens=False)
        assert len(init_ids) == 1
        self.initializer_token_id = init_ids[0]

        # resize embedding
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        # init placeholder embeddings
        with torch.no_grad():
            emb = self.text_encoder.get_input_embeddings().weight
            for tok in self.placeholder_tokens:
                tid = self.tokenizer.convert_tokens_to_ids(tok)
                emb[tid] = emb[self.initializer_token_id].clone()

        # enable grad only for input embedding
        self.text_encoder.get_input_embeddings().requires_grad_(True)

        # backup full embedding to restore non-placeholder tokens every step
        self.orig_embeds = self.text_encoder.get_input_embeddings().weight.data.clone()

        # placeholder string for insertion
        self.placeholder_str = " ".join(self.placeholder_tokens)

    def _setup_data(self):
        ds = PromptPairCSV(
            csv_path=self.cfg.train_csv,
            placeholder_str=self.placeholder_str,
            position=self.cfg.position,
            repeats=self.cfg.repeats,
        )
        self.train_loader = DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def _setup_optim(self):
        self.optimizer = torch.optim.AdamW(
            [{"params": self.text_encoder.get_input_embeddings().parameters(), "lr": self.cfg.lr}],
            betas=(0.9, 0.999),
            weight_decay=0.0,
            eps=1e-8,
        )

        num_update_steps_per_epoch = math.ceil(len(self.train_loader) / self.cfg.grad_accum)
        # max_steps is authoritative
        self.lr_sched = get_scheduler(
            self.cfg.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.cfg.lr_warmup_steps * self.acc.num_processes,
            num_training_steps=self.cfg.max_steps * self.acc.num_processes,
        )

    def _encode(self, prompts: List[str]) -> torch.Tensor:
        toks = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.cfg.max_length,
            return_tensors="pt",
        )
        input_ids = toks.input_ids.to(self.acc.device)
        attn = toks.attention_mask.to(self.acc.device)
        out = self.text_encoder(input_ids=input_ids, attention_mask=attn)
        h = out.last_hidden_state.to(dtype=self.weight_dtype)  # [B,L,D]
        z = mean_pool(h, attn)                                 # [B,D]
        return z

    def _restore_non_placeholder(self):
        """
        Restore all non-placeholder embeddings to original values to avoid drifting the vocab.
        """
        with torch.no_grad():
            emb = self.acc.unwrap_model(self.text_encoder).get_input_embeddings().weight
            mask = torch.ones((emb.shape[0],), dtype=torch.bool, device=emb.device)
            # mark placeholder ids as False
            for tok in self.placeholder_tokens:
                tid = self.tokenizer.convert_tokens_to_ids(tok)
                mask[tid] = False
            emb[mask] = self.orig_embeds.to(emb.device)[mask]

    def _triplet_loss(self, z_anchor: torch.Tensor, z_pos: torch.Tensor, z_neg: torch.Tensor) -> torch.Tensor:
        """
        anchor closer to pos than neg by margin.
        """
        d_pos = F.mse_loss(z_anchor, z_pos, reduction="none").mean(dim=1)  # [B]
        d_neg = F.mse_loss(z_anchor, z_neg, reduction="none").mean(dim=1)
        # adaptive margin
        margin = self.cfg.margin_coef * F.mse_loss(z_pos, z_neg.detach())
        return F.relu(d_pos - d_neg + margin).mean()

    def train(self):
        self.text_encoder.train()
        # other layers are frozen; only embedding grads flow

        # Save orig_embeds once at start (for resume / restore non-placeholder)
        if self.global_step == 0:
            self._save_orig_embeds()

        # Step-based loop: run until global_step >= max_steps (progress bar = actual steps)
        if self.acc.is_local_main_process:
            from tqdm.auto import tqdm
            pbar = tqdm(total=self.cfg.max_steps, desc="Steps", initial=self.global_step)
        else:
            pbar = None

        train_iter = cycle(self.train_loader)
        while self.global_step < self.cfg.max_steps:
            batch = next(train_iter)
            with self.acc.accumulate(self.text_encoder):
                # encode representations
                z_t_with = self._encode(batch["toxic_with"])  # sanitized toxic
                z_s_plain = self._encode(batch["safe"])       # target safe (plain)

                # Align: toxic_with -> safe_plain
                loss_align = F.mse_loss(z_t_with, z_s_plain)

                # Preserve: safe_with should stay close to safe_plain
                z_s_with = self._encode(batch["safe_with"])
                loss_preserve = F.mse_loss(z_s_with, z_s_plain.detach())

                # Optional triplet: pseudo_toxic_with closer to safe_plain than toxic_plain
                # Use no_grad for toxic encode to avoid storing extra T5 activations (reduces OOM when lambda_triplet > 0)
                loss_triplet = torch.tensor(0.0, device=self.acc.device)
                if self.cfg.lambda_triplet > 0:
                    z_pt_with = self._encode(batch["pseudo_toxic_with"])
                    with torch.no_grad():
                        z_t_plain = self._encode(batch["toxic"])
                    loss_triplet = self._triplet_loss(z_pt_with, z_s_plain.detach(), z_t_plain)

                loss = (
                    self.cfg.lambda_align * loss_align
                    + self.cfg.lambda_preserve * loss_preserve
                    + self.cfg.lambda_triplet * loss_triplet
                )

                self.acc.backward(loss)
                self.optimizer.step()
                self.lr_sched.step()
                self.optimizer.zero_grad()
                self._restore_non_placeholder()

            if self.acc.sync_gradients:
                self.global_step += 1
                logs = {
                    "loss/total": float(loss.detach().cpu()),
                    "loss/align": float(loss_align.detach().cpu()),
                    "loss/preserve": float(loss_preserve.detach().cpu()),
                    "loss/triplet": float(loss_triplet.detach().cpu()) if torch.is_tensor(loss_triplet) else 0.0,
                    "lr": self.lr_sched.get_last_lr()[0],
                }
                self.acc.log(logs, step=self.global_step)
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(**logs)
                # Checkpoint every checkpoint_steps
                if self.global_step % self.cfg.checkpoint_steps == 0:
                    self._save_checkpoint(self.global_step)

        if pbar is not None:
            pbar.close()
        self._save()
        self.acc.end_training()

    def _save(self):
        self.acc.wait_for_everyone()
        if not self.acc.is_main_process:
            return

        save_path = os.path.join(self.cfg.output_dir, "learned_embeds_t5_textonly.safetensors")
        te = self.acc.unwrap_model(self.text_encoder)
        emb = te.get_input_embeddings().weight

        token_to_id = {t: self.tokenizer.convert_tokens_to_ids(t) for t in self.placeholder_tokens}
        save_learned_embeds(save_path, token_to_id, emb, self.placeholder_tokens)

        # also save a small json for easy loading later
        meta = {
            "model_path": self.cfg.model_path,
            "subfolder_text_encoder": "text_encoder_3",
            "subfolder_tokenizer": "tokenizer_3",
            "placeholder_tokens": self.placeholder_tokens,
            "position": self.cfg.position,
            "max_length": self.cfg.max_length,
        }
        with open(os.path.join(self.cfg.output_dir, "learned_embeds_meta.json"), "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved learned embeds to: {save_path}")

    def _save_orig_embeds(self):
        """Save full embedding table once for resume (restore non-placeholder)."""
        self.acc.wait_for_everyone()
        if not self.acc.is_main_process:
            return
        path = os.path.join(self.cfg.output_dir, "orig_embeds.safetensors")
        if os.path.exists(path):
            return
        te = self.acc.unwrap_model(self.text_encoder)
        save_file({"embedding": self.orig_embeds.cpu()}, path)
        logger.info(f"Saved orig_embeds to {path}")

    def _save_checkpoint(self, step: int):
        """Save checkpoint-{step}: learned_embeds, optimizer, lr_sched, global_step."""
        self.acc.wait_for_everyone()
        ckpt_dir = os.path.join(self.cfg.output_dir, f"checkpoint-{step}")
        if self.acc.is_main_process:
            os.makedirs(ckpt_dir, exist_ok=True)
            te = self.acc.unwrap_model(self.text_encoder)
            emb = te.get_input_embeddings().weight
            token_to_id = {t: self.tokenizer.convert_tokens_to_ids(t) for t in self.placeholder_tokens}
            save_learned_embeds(
                os.path.join(ckpt_dir, "learned_embeds.safetensors"),
                token_to_id, emb, self.placeholder_tokens,
            )
            state = {
                "global_step": step,
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_sched.state_dict(),
            }
            torch.save(state, os.path.join(ckpt_dir, "training_state.pt"))
            logger.info(f"Saved checkpoint to {ckpt_dir}")
        self.acc.wait_for_everyone()

    def _resume_from_checkpoint(self):
        """Load latest checkpoint and restore global_step, embeds, orig_embeds, optimizer, lr_sched."""
        ckpt_dir = _latest_checkpoint_dir(self.cfg.output_dir)
        if ckpt_dir is None:
            if self.acc.is_main_process:
                logger.warning("Resume requested but no checkpoint found; starting from step 0.")
            return
        self.acc.wait_for_everyone()
        state_path = os.path.join(ckpt_dir, "training_state.pt")
        emb_path = os.path.join(ckpt_dir, "learned_embeds.safetensors")
        orig_path = os.path.join(self.cfg.output_dir, "orig_embeds.safetensors")
        if not os.path.isfile(state_path) or not os.path.isfile(emb_path):
            if self.acc.is_main_process:
                logger.warning(f"Checkpoint incomplete at {ckpt_dir}; starting from step 0.")
            return
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        self.global_step = state["global_step"]
        if self.acc.is_main_process:
            logger.info(f"Resuming from {ckpt_dir} at step {self.global_step}")
        # Load learned placeholder embeds into model
        te = self.acc.unwrap_model(self.text_encoder)
        emb = te.get_input_embeddings().weight
        loaded = load_file(emb_path)
        with torch.no_grad():
            for tok, tensor in loaded.items():
                tid = self.tokenizer.convert_tokens_to_ids(tok)
                emb[tid] = tensor.to(emb.dtype).to(emb.device)
        # Restore orig_embeds for _restore_non_placeholder
        if os.path.isfile(orig_path):
            orig = load_file(orig_path)
            self.orig_embeds = orig["embedding"].to(emb.device, dtype=emb.dtype)
        else:
            if self.acc.is_main_process:
                logger.warning("orig_embeds.safetensors not found; non-placeholder restore may be wrong.")
        # Restore optimizer and lr_scheduler
        self.optimizer.load_state_dict(state["optimizer"])
        self.lr_sched.load_state_dict(state["lr_scheduler"])
        self.acc.wait_for_everyone()


# -------------------------
# Main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./out_sd35_t5_textonly")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--lr_scheduler", type=str, default="constant")
    p.add_argument("--lr_warmup_steps", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=2000)

    p.add_argument("--placeholder_token", type=str, default="<SAFE>")
    p.add_argument("--num_vectors", type=int, default=1)
    p.add_argument("--initializer_token", type=str, default="safe")  # will auto-adjust to single token if needed
    p.add_argument("--position", type=str, default="suffix", choices=["prefix", "suffix", "after_first"])
    p.add_argument("--repeats", type=int, default=1)

    p.add_argument("--lambda_align", type=float, default=1.0)
    p.add_argument("--lambda_preserve", type=float, default=0.2)
    p.add_argument("--lambda_triplet", type=float, default=0.0)
    p.add_argument("--margin_coef", type=float, default=0.2)

    p.add_argument("--checkpoint_steps", type=int, default=1000, help="Save checkpoint every N steps")
    p.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in output_dir")

    return p.parse_args()


def main():
    args = parse_args()

    cfg = TrainConfig(
        model_path=args.model_path,
        output_dir=args.output_dir,
        train_csv=args.train_csv,

        seed=args.seed,
        mixed_precision=args.mixed_precision,
        grad_accum=args.grad_accum,
        batch_size=args.batch_size,
        num_workers=args.num_workers,

        max_length=args.max_length,
        lr=args.lr,
        lr_scheduler=args.lr_scheduler,
        lr_warmup_steps=args.lr_warmup_steps,
        max_steps=args.max_steps,

        placeholder_token=args.placeholder_token,
        num_vectors=args.num_vectors,
        initializer_token=args.initializer_token,
        position=args.position,
        repeats=args.repeats,

        lambda_align=args.lambda_align,
        lambda_preserve=args.lambda_preserve,
        lambda_triplet=args.lambda_triplet,
        margin_coef=args.margin_coef,

        checkpoint_steps=args.checkpoint_steps,
        resume=args.resume,
    )

    # logging (use std logger before Accelerator is created in Trainer)
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=logging.INFO,
    )
    logging.getLogger(__name__).info(f"Config: {cfg}")

    trainer = TextOnlySoftTokenTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
