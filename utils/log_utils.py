from contextlib import nullcontext
import json
import logging
import os
import diffusers
import pandas as pd
import safetensors
import torch
import transformers
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import numpy as np
import wandb
from accelerate.logging import get_logger
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from packaging import version
from huggingface_hub import create_repo, upload_folder
from pathlib import Path

logger = get_logger(__name__, log_level="INFO")

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.30.0.dev0")


def save_model_card(repo_id: str, images: list = None, base_model: str = None, repo_folder: str = None):
    """保存模型卡片，不再使用 hub_utils 中的函数"""
    if repo_folder is None:
        repo_folder = repo_id

    # 创建模型卡片内容
    model_card = f"""
---
license: mit
base_model: {base_model}
tags:
- stable-diffusion
- text-to-image
- diffusers
- textual-inversion
inference: true
---

# Textual inversion - {repo_id}

This is a textual inversion model trained on the {base_model} model. You can find some example images in the following.
"""
    if images is not None:
        for i, image in enumerate(images):
            model_card += f"\n![img_{i}]({image})\n"

    # 保存模型卡片
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(model_card)


def log_validation(text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, epoch):
    if args.validation_file is not None:
        # 读取 csv 文件，取出prompt列
        validation_prompts = pd.read_csv(args.validation_file)['prompt'].tolist()
    else:
        validation_prompts = [args.validation_prompt]

    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {validation_prompts}."
    )
    # create pipeline (note: unet and vae are loaded again in float32)
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=unet,
        vae=vae,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    images = []
    for validation_prompt in validation_prompts:
        for _ in range(args.num_validation_images):
            if torch.backends.mps.is_available():
                autocast_ctx = nullcontext()
            else:
                autocast_ctx = torch.autocast(accelerator.device.type)

            with autocast_ctx:
                image = pipeline(validation_prompt, num_inference_steps=25, generator=generator).images[0]
            images.append(image)

        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                np_images = np.stack([np.asarray(img) for img in images])
                tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
            if tracker.name == "wandb":
                tracker.log(
                    {
                        "validation": [
                            wandb.Image(image, caption=f"{i}: {validation_prompt}") for i, image in enumerate(images)
                        ]
                    }
                )

    del pipeline
    torch.cuda.empty_cache()
    return images


def save_progress(text_encoder, placeholder_token_ids, accelerator, args, save_path, safe_serialization=True):
    logger.info("Saving embeddings")
    learned_embeds = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1]
    )
    learned_embeds_dict = {args.placeholder_token: learned_embeds.detach().cpu()}

    if safe_serialization:
        safetensors.torch.save_file(learned_embeds_dict, save_path, metadata={"format": "pt"})
    else:
        torch.save(learned_embeds_dict, save_path)


def log_setup(args, accelerator):
    if args.report_to == "wandb":
        if args.hub_token is not None:
            raise ValueError(
                "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            )
        
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # 记录加速器状态信息,所有进程都会记录
    logger.info(accelerator.state, main_process_only=False)

    # 如果是主进程:
    # - transformers库设置为warning级别日志
    # - diffusers库设置为info级别日志,可以看到更多训练细节
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    # 如果是其他进程:
    # - 两个库都设置为error级别,只显示错误信息
    # - 避免重复的日志输出
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    repo_id = None
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # save all the arguments in a config file in the output directory
    args_dict = vars(args)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(args_dict, f, indent=4)
    
    return repo_id



def t2i_gpu_setup(args, accelerator, vae, unet, text_encoder):
    # 对于 MPS (Metal Performance Shaders, 苹果 M1/M2 芯片的机器学习加速器) 禁用自动混合精度(AMP)训练
    # 因为 MPS 目前不支持 AMP,所以需要将 accelerator 的 native_amp 设置为 False
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
    
    if args.gradient_checkpointing:
        # Keep unet in train mode if we are using gradient checkpointing to save memory.
        # The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
        unet.train()
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()
        
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
        
        # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    return weight_dtype