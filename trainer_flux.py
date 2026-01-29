from abc import abstractmethod
import json
import logging
import math
import os
from pathlib import Path
import shutil
import warnings

import diffusers
import pandas as pd
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    FluxPipeline,
    Transformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
import transformers
from transformers import (
    T5EncoderModel,
    T5Tokenizer,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
)

from data import PromptPairDataset
from evaluate import OursPipeline
from utils.log_utils import save_progress, log_validation, save_model_card


logger = get_logger(__name__, log_level="INFO")

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.30.0.dev0")

# 抑制 CLIP tokenizer 的 77 tokens 截断警告
# Flux 使用 dual encoder (CLIP + T5)，CLIP 限制 77 tokens 是正常行为
# T5 encoder 支持 512 tokens，所以长文本会被 T5 正确处理
warnings.filterwarnings(
    "ignore",
    message=".*can only handle sequences up to 77 tokens.*",
    category=UserWarning,
)


class BaseFluxTrainer:
    """
    面向 Flux 的训练基类。

    设计目标：
    - 训练逻辑尽量复用现有 SD1.4 版的 BaseTrainer；
    - 默认优先尝试使用 T5 作为 text encoder；
    - 如果本地缺少 T5 相关文件，则自动回退到 CLIP text encoder，保证在只下了 CLIP 的情况下也能训练；
    - UNet 换成 Flux 的 Transformer2DModel（如果不存在，则退回 UNet2DConditionModel）。
    """

    def __init__(self, args):
        self.args = args
        self.setup_accelerator()
        self.setup_logger()
        self.setup_models()
        self.setup_memory()
        self.setup_placeholder_tokens()
        self.setup_optimizer()
        self.setup_dataset()
        self.setup_scheduler()
        self.setup_training_state()

    # ===== 基础环境 =====
    def setup_accelerator(self):
        if self.args.seed is not None:
            set_seed(self.args.seed)

        logging_dir = os.path.join(self.args.output_dir, self.args.logging_dir)
        accelerator_project_config = ProjectConfiguration(
            project_dir=self.args.output_dir, logging_dir=logging_dir
        )
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=self.args.mixed_precision,
            log_with=self.args.report_to,
            project_config=accelerator_project_config,
        )

        if torch.backends.mps.is_available():
            self.accelerator.native_amp = False

        # dtype 设置
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

    def setup_logger(self):
        if self.args.report_to == "wandb":
            if self.args.hub_token is not None:
                raise ValueError(
                    "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
                )

            if not is_wandb_available():
                raise ImportError(
                    "Make sure to install wandb if you want to use it for logging during training."
                )

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(self.accelerator.state, main_process_only=False)

        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        # Hub 仓库 & 输出目录
        if self.accelerator.is_main_process:
            if self.args.output_dir is not None:
                os.makedirs(self.args.output_dir, exist_ok=True)

            if self.args.push_to_hub:
                self.repo_id = create_repo(
                    repo_id=self.args.hub_model_id
                    or Path(self.args.output_dir).name,
                    exist_ok=True,
                    token=self.args.hub_token,
                ).repo_id

        # 保存一次 config
        args_dict = vars(self.args)
        with open(os.path.join(self.args.output_dir, "config.json"), "w") as f:
            json.dump(args_dict, f, indent=4)

    # ===== 模型相关 =====
    def setup_models(self):
        """
        加载 Flux 相关组件。

        优先尝试：
        - T5 tokenizer + T5 encoder (text_encoder_2, tokenizer_2)
        如果失败，自动回退到：
        - CLIP tokenizer + CLIP text encoder (tokenizer, text_encoder)
        """
        # 标记当前是否真的在使用 T5 encoder，后续冻结策略要区分处理
        self.use_t5_encoder = False

        # 1) tokenizer：优先 T5，失败回退 CLIP
        if getattr(self.args, "tokenizer_name", None):
            # 如果显式指定了 tokenizer_name，就认为用户想用 T5
            try:
                self.tokenizer = T5Tokenizer.from_pretrained(
                    self.args.tokenizer_name
                )
                self.use_t5_encoder = True
            except Exception as e:
                logger.warning(
                    f"Failed to load T5Tokenizer from '{self.args.tokenizer_name}' "
                    f"({e}), falling back to CLIPTokenizer from model tokenizer."
                )
                self.tokenizer = CLIPTokenizer.from_pretrained(
                    self.args.pretrained_model_name_or_path, subfolder="tokenizer"
                )
                self.use_t5_encoder = False
        else:
            # 尝试从 tokenizer_2 加载 T5Tokenizer
            try:
                self.tokenizer = T5Tokenizer.from_pretrained(
                    self.args.pretrained_model_name_or_path,
                    subfolder="tokenizer_2",
                )
                self.use_t5_encoder = True
            except Exception as e:
                logger.warning(
                    f"Failed to load T5 tokenizer from 'tokenizer_2' ({e}), "
                    "falling back to CLIPTokenizer from 'tokenizer'."
                )
                self.tokenizer = CLIPTokenizer.from_pretrained(
                    self.args.pretrained_model_name_or_path, subfolder="tokenizer"
                )
                self.use_t5_encoder = False

        # 2) 加载 Flux 的 transformer / vae / scheduler
        # 直接通过 FluxPipeline.from_pretrained 读取，避免不同 diffusers 版本下
        # Transformer2DModel / FluxTransformer2DModel 的 class remapping 兼容性问题。
        # 同时保留一个轻量级的 pipeline 实例（不重新加载权重），专门用来走
        # 官方 FluxPipeline 的 encode_prompt / latent_id 等逻辑。
        try:
            pipe = FluxPipeline.from_pretrained(
                self.args.pretrained_model_name_or_path,
                torch_dtype=self.weight_dtype,
            )
            # 只取我们需要的组件，避免改动原有训练逻辑太多
            self.transformer = pipe.transformer
            # 如果后面没有单独重新加载，则直接复用 pipeline 里的 scheduler / vae
            self.noise_scheduler = getattr(pipe, "scheduler", None)
            self.vae = getattr(pipe, "vae", self.vae if hasattr(self, "vae") else None)
            # 保留一个 encode 用的 pipeline，但不在训练中调用它的 __call__，
            # 只用其中的 encode_prompt / _prepare_latent_image_ids 等辅助函数。
            self.flux_pipeline = pipe
        except Exception as e:
            # 如果连 FluxPipeline 都加载失败，则训练无法继续，直接抛出明确错误
            logger.error(
                "Failed to load FluxPipeline. 请确认本地模型目录完整且 diffusers 版本支持 FLUX. "
                f"原始错误: {e}"
            )
            raise

        # 3) text encoder：优先 T5，失败则 CLIP
        if self.use_t5_encoder:
            try:
                self.text_encoder = T5EncoderModel.from_pretrained(
                    self.args.pretrained_model_name_or_path,
                    subfolder="text_encoder_2",
                    revision=self.args.revision,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load T5 encoder from 'text_encoder_2' ({e}), "
                    "falling back to CLIPTextModel from 'text_encoder'."
                )
                self.text_encoder = CLIPTextModel.from_pretrained(
                    self.args.pretrained_model_name_or_path,
                    subfolder="text_encoder",
                    revision=self.args.revision,
                )
                self.use_t5_encoder = False
        else:
            # 直接 CLIP
            self.text_encoder = CLIPTextModel.from_pretrained(
                self.args.pretrained_model_name_or_path,
                subfolder="text_encoder",
                revision=self.args.revision,
            )

        # 4) 可选 dual encoder（当前先不强依赖，仅作为扩展使用）
        self.text_encoder_1 = None
        self.tokenizer_1 = None
        if getattr(self.args, "use_dual_encoder", False):
            try:
                self.text_encoder_1 = CLIPTextModel.from_pretrained(
                    self.args.pretrained_model_name_or_path,
                    subfolder="text_encoder",
                    revision=self.args.revision,
                )
                self.tokenizer_1 = CLIPTokenizer.from_pretrained(
                    self.args.pretrained_model_name_or_path,
                    subfolder="tokenizer",
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load CLIP dual encoder ({e}), using single encoder only."
                )
                self.text_encoder_1 = None
                self.tokenizer_1 = None

        # 5) scheduler / VAE（如果上面 FluxPipeline 已经填充，则这里只在缺失时补齐）
        if self.noise_scheduler is None:
            self.noise_scheduler = DDPMScheduler.from_pretrained(
                self.args.pretrained_model_name_or_path, subfolder="scheduler"
            )
        if self.vae is None:
            self.vae = AutoencoderKL.from_pretrained(
                self.args.pretrained_model_name_or_path,
                subfolder="vae",
                revision=self.args.revision,
                variant=self.args.variant,
            )

        # 6) CLIP 图像 encoder（用于对比或扩展）
        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.args.clip_model_path
        )
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.args.clip_model_path
        )

        # 7) gradient checkpointing
        if self.args.gradient_checkpointing:
            self.transformer.train()
            # text encoder 的 gradient checkpointing 只在支持时开启
            if hasattr(self.text_encoder, "gradient_checkpointing_enable"):
                self.text_encoder.gradient_checkpointing_enable()
            if hasattr(self.transformer, "enable_gradient_checkpointing"):
                self.transformer.enable_gradient_checkpointing()
            if self.text_encoder_1 is not None and hasattr(
                self.text_encoder_1, "gradient_checkpointing_enable"
            ):
                self.text_encoder_1.gradient_checkpointing_enable()

    def setup_memory(self):
        # TF32
        if self.args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        # xformers
        if self.args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logger.warning(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. "
                        "If you observe problems during training, please update xFormers to at least 0.0.17."
                    )
                if hasattr(self.transformer, "enable_xformers_memory_efficient_attention"):
                    self.transformer.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError(
                    "xformers is not available. Make sure it is installed correctly"
                )

    # ===== textual inversion 部分 =====
    def setup_placeholder_tokens(self):
        """
        设置 placeholder tokens，逻辑与原 BaseTrainer 一致，
        但同时兼容 T5 / CLIP，只依赖 get_input_embeddings。
        """
        placeholder_tokens = [self.args.placeholder_token]
        for i in range(1, self.args.num_vectors):
            placeholder_tokens.append(f"{self.args.placeholder_token}_{i}")

        num_added_tokens = self.tokenizer.add_tokens(placeholder_tokens)

        if num_added_tokens != self.args.num_vectors:
            raise ValueError(
                f"The tokenizer already contains the token {self.args.placeholder_token}. "
                "Please pass a different `placeholder_token` that is not already in the tokenizer."
            )

        token_ids = self.tokenizer.encode(
            self.args.initializer_token, add_special_tokens=False
        )
        if len(token_ids) > 1:
            raise ValueError("The initializer token must be a single token.")

        self.initializer_token_id = token_ids[0]
        self.placeholder_token_ids = self.tokenizer.convert_tokens_to_ids(
            placeholder_tokens
        )

        # 调整 embedding 尺寸并初始化新 token 的向量
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        token_embeds = self.text_encoder.get_input_embeddings().weight.data
        with torch.no_grad():
            for token_id in self.placeholder_token_ids:
                token_embeds[token_id] = token_embeds[
                    self.initializer_token_id
                ].clone()

        # 冻结除 embedding 外的其他部分
        self.vae.requires_grad_(False)
        self.transformer.requires_grad_(False)

        # T5：冻结 encoder.block / final_layer_norm
        if self.use_t5_encoder and hasattr(self.text_encoder, "encoder"):
            enc = self.text_encoder.encoder
            if hasattr(enc, "block"):
                enc.block.requires_grad_(False)
            if hasattr(enc, "final_layer_norm"):
                enc.final_layer_norm.requires_grad_(False)
        else:
            # CLIP：复用原 BaseTrainer 的冻结策略
            if hasattr(self.text_encoder, "text_model"):
                tm = self.text_encoder.text_model
                if hasattr(tm, "encoder"):
                    tm.encoder.requires_grad_(False)
                if hasattr(tm, "final_layer_norm"):
                    tm.final_layer_norm.requires_grad_(False)
                if hasattr(tm, "embeddings") and hasattr(
                    tm.embeddings, "position_embedding"
                ):
                    tm.embeddings.position_embedding.requires_grad_(False)

        # dual encoder 的 CLIP 也全部冻结
        if self.text_encoder_1 is not None:
            self.text_encoder_1.requires_grad_(False)

    def setup_optimizer(self):
        # 只优化 embedding 层
        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": self.text_encoder.get_input_embeddings().parameters(),
                    "lr": self.args.learning_rate,
                }
            ],
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon,
        )

    # ===== 数据 & 训练状态 =====
    def setup_dataset(self):
        self.train_dataset = PromptPairDataset(
            csv_path=self.args.train_data_csv,
            tokenizer=self.tokenizer,
            position=self.args.position,
            placeholder_token=" ".join(
                self.tokenizer.convert_ids_to_tokens(self.placeholder_token_ids)
            ),
            repeats=self.args.repeats,
            set="train",
        )
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            num_workers=self.args.dataloader_num_workers,
        )

    def setup_scheduler(self):
        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.args.gradient_accumulation_steps
        )
        logger.info(f"  num_update_steps_per_epoch = {num_update_steps_per_epoch}")
        if self.args.max_train_steps is None:
            self.args.max_train_steps = (
                self.args.num_train_epochs * num_update_steps_per_epoch
            )
        self.args.num_train_epochs = math.ceil(
            self.args.max_train_steps
            / num_update_steps_per_epoch
            * self.accelerator.num_processes
        )

        self.lr_scheduler = get_scheduler(
            self.args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.lr_warmup_steps * self.accelerator.num_processes,
            num_training_steps=self.args.max_train_steps
            * self.accelerator.num_processes,
            num_cycles=self.args.lr_num_cycles,
        )

        if self.args.scale_lr:
            self.args.learning_rate = (
                self.args.learning_rate
                * self.args.gradient_accumulation_steps
                * self.args.train_batch_size
                * self.accelerator.num_processes
            )

    def setup_training_state(self):
        self.text_encoder.train()
        (
            self.text_encoder,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.text_encoder, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

        # 将冻结的模型移到正确的设备上（多 GPU 时会自动分布）
        self.transformer.to(self.accelerator.device, dtype=self.weight_dtype)
        self.transformer.train()  # 确保 transformer 处于训练模式
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        self.image_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        if self.text_encoder_1 is not None:
            self.text_encoder_1.to(self.accelerator.device, dtype=self.weight_dtype)
        
        # 确保 flux_pipeline 的 text encoder 也在正确的设备上
        # 在多 GPU 训练中，每个进程会在自己的设备上
        if hasattr(self, "flux_pipeline") and self.flux_pipeline is not None:
            if getattr(self.flux_pipeline, "text_encoder", None) is not None:
                self.flux_pipeline.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
            if getattr(self.flux_pipeline, "text_encoder_2", None) is not None:
                self.flux_pipeline.text_encoder_2.to(self.accelerator.device, dtype=self.weight_dtype)

        self.orig_embeds_params = (
            self.accelerator.unwrap_model(self.text_encoder)
            .get_input_embeddings()
            .weight.data.clone()
        )

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                "textual_inversion_flux", config=vars(self.args)
            )

    # ===== 训练主循环 =====
    def train(self):
        total_batch_size = (
            self.args.train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )

        logger.info("***** Running Flux training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        logger.info(
            f"  Instantaneous batch size per device = {self.args.train_batch_size}"
        )
        logger.info(f"  Device count = {self.accelerator.num_processes}")
        logger.info(
            f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}"
        )
        logger.info(
            "  Total train batch size "
            f"(w. parallel={self.args.train_batch_size}, "
            f"distributed={self.accelerator.num_processes}, "
            f"accumulation={self.args.gradient_accumulation_steps}) = {total_batch_size}"
        )
        logger.info(f"  Total optimization steps = {self.args.max_train_steps}")

        self.global_step = 0
        self.first_epoch = 0
        self.loss_list = []

        if self.args.resume_from_checkpoint:
            self.load_checkpoint()

        progress_bar = tqdm(
            range(0, self.args.max_train_steps),
            initial=self.global_step,
            desc="Steps",
            disable=not self.accelerator.is_local_main_process,
        )

        for epoch in range(self.first_epoch, self.args.num_train_epochs):
            self.text_encoder.train()
            self.transformer.train()  # 确保 transformer 处于训练模式，以便梯度能正确传播
            
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.text_encoder):
                    loss = self.training_step(batch)
                    self.accelerator.backward(loss)

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    self.update_embeddings()

                if self.accelerator.sync_gradients:
                    self.handle_sync_gradients(progress_bar)
                    
                    # 定期清理显存，避免碎片化
                    if self.global_step % 10 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()

                logs = {
                    "loss": loss.detach().cpu().item(),
                    "step": self.global_step,
                    "epoch": self.global_step // len(self.train_dataloader),
                    "progress": self.global_step / self.args.max_train_steps,
                }
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=self.global_step)

                if self.global_step >= self.args.max_train_steps:
                    break

        self.save_final_model()

    # ===== 抽象方法：由子类实现具体 loss =====
    @abstractmethod
    def training_step(self, batch):
        pass

    # ===== 噪声预测 & latent 初始化 =====
    def predict_noise(self, prompt, z_t, timesteps):
        """
        使用 Flux 官方实现的文本编码与 Transformer 调用逻辑进行噪声预测，
        不再兼容 / 适配 Stable Diffusion 的 UNet 接口。

        参考：
        - diffusers.pipelines.flux.FluxPipeline.encode_prompt
        - diffusers.pipelines.flux.FluxPipeline.__call__ 中对 transformer 的调用
        """
        if not hasattr(self, "flux_pipeline") or self.flux_pipeline is None:
            raise RuntimeError("Flux pipeline is not initialized; cannot run predict_noise with Flux logic.")

        device = self.accelerator.device
        # 确保 FluxPipeline 内部的 text_encoder / text_encoder_2 与我们当前的训练设备一致，
        # 否则 encode_prompt 会出现 CPU / CUDA 设备不匹配的问题。
        if getattr(self.flux_pipeline, "text_encoder", None) is not None:
            self.flux_pipeline.text_encoder.to(device=device, dtype=self.weight_dtype)
        if getattr(self.flux_pipeline, "text_encoder_2", None) is not None:
            self.flux_pipeline.text_encoder_2.to(device=device, dtype=self.weight_dtype)
        bsz = len(prompt)

        # 1) 文本编码：直接复用 FluxPipeline 的 encode_prompt
        # 注意：虽然 text encoder 的大部分参数被冻结，但 embedding 层需要梯度
        # 所以不能使用 no_grad()，否则梯度无法传播到 embedding
        prompt_embeds, pooled_prompt_embeds, text_ids = self.flux_pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )

        # 2) latent image ids：从 init_latent 中缓存的 self.latent_image_ids 读取
        latent_image_ids = getattr(self, "latent_image_ids", None)
        if latent_image_ids is None:
            raise RuntimeError(
                "latent_image_ids is not initialized; make sure init_latent is called before predict_noise."
            )
        latent_image_ids = latent_image_ids.to(device=device, dtype=prompt_embeds.dtype)

        # 3) 将离散 timestep 归一化到 [0, 1]，与 FluxPipeline 中 `timestep = timesteps / 1000` 的逻辑一致
        if isinstance(timesteps, torch.Tensor):
            t_continuous = timesteps.to(device).to(prompt_embeds.dtype) / 1000.0
        else:
            t_continuous = torch.tensor(
                timesteps,
                device=device,
                dtype=prompt_embeds.dtype,
            ) / 1000.0

        # 4) guidance：如果当前 FLUX 变体开启了 guidance_embeds，按官方训练脚本传入 guidance 标量
        guidance = None
        if getattr(self.transformer.config, "guidance_embeds", False):
            guidance_scale = getattr(self.args, "guidance_scale", 1.0)
            guidance = torch.full(
                (bsz,),
                float(guidance_scale),
                device=device,
                dtype=prompt_embeds.dtype,
            )

        # 5) 直接按 FluxPipeline.__call__ 的方式调用 FluxTransformer2DModel
        try:
            noise_pred = self.transformer(
                hidden_states=z_t,
                timestep=t_continuous,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs={},
                return_dict=False,
            )[0]
        except Exception as e:
            logger.error(f"Failed to call Flux transformer with official API: {e}")
            logger.error(f"Transformer type: {type(self.transformer)}")
            raise RuntimeError(
                f"Cannot call Flux transformer with official API. Error: {e}"
            )

        return noise_pred, prompt_embeds

    def init_latent(self, bsz):
        """
        初始化 latent，严格参考官方 Flux 训练脚本的形状约定：
        - 在 VAE latent 空间中采样「干净」latent pixel_latents ~ N(0, I)
        - 用 FlowMatchEulerDiscreteScheduler 的 sigmas 对其加噪得到 noisy_model_input
        - 使用 Flux 的 `_pack_latents` 将 [B, C, H, W] 打包为 [B, N_tokens, in_channels]
        - 同时返回对应的离散 timesteps 和 `latent_image_ids`（RoPE 所需）
        """
        if not hasattr(self, "flux_pipeline") or self.flux_pipeline is None:
            raise RuntimeError("Flux pipeline is not initialized; cannot init_latent.")

        device = self.accelerator.device

        # VAE latent 通道数与 Flux 中的打包规则：
        # pack 前的 latent 通道为 C_lat，pack 后最后一维为 C_lat * 4，应等于 transformer.config.in_channels。
        c_lat = max(1, getattr(self.transformer.config, "in_channels", 64) // 4)

        # 选择一个与 Flux 默认一致的 latent 分辨率。
        # 默认高度为 128 * vae_scale_factor（如 1024），latent 空间高度在 VAE 中再压缩 8 倍，
        # 这里直接用 64 作为 latent H=W 的经验值即可（与官方脚本的 encode_images 输出一致）。
        sample_size = getattr(self.transformer.config, "sample_size", 64)

        latent_shape = (bsz, c_lat, sample_size, sample_size)

        # 1) 采样「干净」 latent 和噪声
        pixel_latents = torch.randn(
            latent_shape,
            device=device,
            dtype=self.weight_dtype,
        )
        noise = torch.randn_like(pixel_latents)

        # 2) 从 FlowMatchEulerDiscreteScheduler 均匀采样离散 step，并取对应的 sigma
        if hasattr(self.noise_scheduler, "timesteps") and hasattr(self.noise_scheduler, "sigmas"):
            schedule_timesteps = self.noise_scheduler.timesteps.to(device=device)
            sigmas_all = self.noise_scheduler.sigmas.to(device=device, dtype=self.weight_dtype)

            indices = torch.randint(
                0,
                len(schedule_timesteps),
                (bsz,),
                device=device,
            )
            timesteps = schedule_timesteps[indices]

            sigma = sigmas_all[indices].view(bsz, 1, 1, 1)

            # Flow‑matching 噪声混合
            noisy_model_input = (1.0 - sigma) * pixel_latents + sigma * noise
        else:
            num_train_timesteps = getattr(
                getattr(self.noise_scheduler, "config", self.noise_scheduler),
                "num_train_timesteps",
                1000,
            )
            timesteps = torch.randint(
                0,
                num_train_timesteps,
                (bsz,),
                device=device,
            ).long()
            noisy_model_input = pixel_latents

        # 3) 将 [B, C_lat, H, W] 打包为 [B, N_tokens, in_channels]，与 FluxTransformer2DModel.forward 要求一致
        z_t = self.flux_pipeline._pack_latents(
            noisy_model_input,
            batch_size=bsz,
            num_channels_latents=noisy_model_input.shape[1],
            height=noisy_model_input.shape[2],
            width=noisy_model_input.shape[3],
        )

        # 4) 预计算 RoPE 所需的 latent_image_ids，缓存到 self 上供 predict_noise 使用
        self.latent_image_ids = self.flux_pipeline._prepare_latent_image_ids(
            bsz,
            noisy_model_input.shape[2] // 2,
            noisy_model_input.shape[3] // 2,
            device,
            self.weight_dtype,
        )

        return z_t, timesteps

    def update_embeddings(self):
        index_no_updates = torch.ones((len(self.tokenizer),), dtype=torch.bool)
        index_no_updates[
            min(self.placeholder_token_ids) : max(self.placeholder_token_ids) + 1
        ] = False

        with torch.no_grad():
            self.accelerator.unwrap_model(
                self.text_encoder
            ).get_input_embeddings().weight[index_no_updates] = self.orig_embeds_params[
                index_no_updates
            ]

    # ===== checkpoint & validate =====
    def handle_sync_gradients(self, progress_bar):
        progress_bar.update(1)
        self.global_step += 1

        if self.global_step % self.args.save_steps == 0:
            self.save_checkpoint()

        if self.accelerator.is_main_process:
            if self.global_step % self.args.checkpointing_steps == 0:
                self.save_state()

    def load_checkpoint(self):
        """
        从检查点恢复训练状态。
        支持：
        - "latest": 自动选择最新的检查点
        - 相对路径: 相对于 output_dir 的检查点目录名（如 "checkpoint-500"）
        - 绝对路径: 完整的检查点路径
        """
        resume_path = self.args.resume_from_checkpoint
        
        if resume_path == "latest":
            # 自动查找最新的检查点
            if os.path.exists(self.args.output_dir):
                dirs = os.listdir(self.args.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                if len(dirs) > 0:
                    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                    resume_path = os.path.join(self.args.output_dir, dirs[-1])
                else:
                    resume_path = None
            else:
                resume_path = None
        elif not os.path.isabs(resume_path):
            # 相对路径：相对于 output_dir
            resume_path = os.path.join(self.args.output_dir, os.path.basename(resume_path))
        # 如果是绝对路径，直接使用

        if resume_path is None or not os.path.exists(resume_path):
            self.accelerator.print(
                f"Checkpoint '{self.args.resume_from_checkpoint}' does not exist. "
                "Starting a new training run."
            )
            self.args.resume_from_checkpoint = None
            self.global_step = 0
            self.first_epoch = 0
        else:
            self.accelerator.print(f"Resuming training from checkpoint: {resume_path}")
            
            # 加载 accelerator 状态（包括 optimizer, lr_scheduler, model, dataloader 等）
            self.accelerator.load_state(resume_path)
            
            # 从检查点路径中提取 global_step
            checkpoint_name = os.path.basename(resume_path)
            if checkpoint_name.startswith("checkpoint-"):
                self.global_step = int(checkpoint_name.split("-")[1])
            else:
                # 如果无法从路径提取，尝试从 accelerator 状态中获取
                # 注意：accelerator 可能不会直接保存 global_step，所以我们需要从路径推断
                logger.warning(
                    f"Could not extract global_step from checkpoint path '{checkpoint_name}'. "
                    "Assuming step 0."
                )
                self.global_step = 0
            
            # 计算 first_epoch
            num_update_steps_per_epoch = math.ceil(
                len(self.train_dataloader) / self.args.gradient_accumulation_steps
            )
            self.first_epoch = self.global_step // num_update_steps_per_epoch
            
            self.accelerator.print(
                f"Resumed from step {self.global_step}, epoch {self.first_epoch}"
            )
            
            # 恢复 loss_list（如果之前有保存的话）
            loss_list_path = os.path.join(resume_path, "loss_list.json")
            if os.path.exists(loss_list_path):
                try:
                    with open(loss_list_path, "r") as f:
                        self.loss_list = json.load(f)
                    self.accelerator.print(f"Restored loss history with {len(self.loss_list)} entries")
                except Exception as e:
                    logger.warning(f"Failed to load loss_list from checkpoint: {e}")
                    self.loss_list = []

    def save_checkpoint(self):
        weight_name = (
            f"learned_embeds-steps-{self.global_step}.bin"
            if self.args.no_safe_serialization
            else f"learned_embeds-steps-{self.global_step}.safetensors"
        )
        save_path = os.path.join(self.args.output_dir, weight_name)
        save_progress(
            self.text_encoder,
            self.placeholder_token_ids,
            self.accelerator,
            self.args,
            save_path,
            safe_serialization=not self.args.no_safe_serialization,
        )

        if (
            self.args.validation_prompt is not None
            or self.args.validation_file is not None
        ) and self.global_step % self.args.validation_steps == 0:
            self.validate(save_path)

    def validate(self, save_path):
        if self.args.validation_file is not None:
            validation_prompts = pd.read_csv(self.args.validation_file)[
                "prompt"
            ].tolist()
        else:
            validation_prompts = [self.args.validation_prompt]

        logger.info(
            "Running validation... \n "
            f"Generating {self.args.num_validation_images} images with prompt: {validation_prompts}."
        )

        try:
            pipeline = OursPipeline(
                model_path=self.args.pretrained_model_name_or_path,
                safe_embedding_paths=[save_path],
                safe_tokens=[self.args.placeholder_token],
                device=self.accelerator.device,
            )
            pipeline.generate_images(
                validation_prompts,
                self.args.output_dir + "/validation/" + str(self.global_step),
                4,
                28,
                7.5,
                42,
                0,
                None,
            )
            del pipeline
        except Exception as e:
            logger.warning(
                f"Validation failed with existing pipeline ({e}). "
                "You may need a Flux-specific inference pipeline."
            )

        torch.cuda.empty_cache()

    def save_state(self):
        """
        保存完整的训练状态到检查点，包括：
        - optimizer 状态
        - lr_scheduler 状态
        - model 状态
        - random states
        - loss_list（额外保存）
        """
        # 清理旧的检查点（如果设置了限制）
        if self.args.checkpoints_total_limit is not None:
            if os.path.exists(self.args.output_dir):
                checkpoints = os.listdir(self.args.output_dir)
                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                if len(checkpoints) >= self.args.checkpoints_total_limit:
                    num_to_remove = len(checkpoints) - self.args.checkpoints_total_limit + 1
                    removing_checkpoints = checkpoints[0:num_to_remove]

                    logger.info(
                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                    )
                    logger.info(
                        f"removing checkpoints: {', '.join(removing_checkpoints)}"
                    )

                    for removing_checkpoint in removing_checkpoints:
                        removing_checkpoint = os.path.join(
                            self.args.output_dir, removing_checkpoint
                        )
                        if os.path.exists(removing_checkpoint):
                            shutil.rmtree(removing_checkpoint)

        save_path = os.path.join(
            self.args.output_dir, f"checkpoint-{self.global_step}"
        )
        
        # 保存 accelerator 状态（包括 optimizer, lr_scheduler, model 等）
        self.accelerator.save_state(save_path)
        
        # 额外保存 loss_list（方便后续分析）
        loss_list_path = os.path.join(save_path, "loss_list.json")
        try:
            with open(loss_list_path, "w") as f:
                json.dump(self.loss_list, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save loss_list to checkpoint: {e}")
        
        logger.info(f"Saved checkpoint to {save_path} (step {self.global_step})")

    def save_final_model(self):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            if self.args.push_to_hub and not self.args.save_as_full_pipeline:
                logger.warning(
                    "Enabling full model saving because --push_to_hub=True was specified."
                )
                save_full_model = True
            else:
                save_full_model = self.args.save_as_full_pipeline

            if save_full_model:
                # 尝试保存完整 Flux pipeline
                try:
                    pipeline = FluxPipeline.from_pretrained(
                        self.args.pretrained_model_name_or_path,
                        text_encoder=self.accelerator.unwrap_model(self.text_encoder),
                        vae=self.vae,
                        transformer=self.transformer,
                        tokenizer=self.tokenizer,
                    )
                    pipeline.save_pretrained(self.args.output_dir)
                except Exception as e:
                    logger.warning(
                        f"Failed to save full Flux pipeline ({e}), saving embeddings only."
                    )

            weight_name = (
                "learned_embeds.bin"
                if self.args.no_safe_serialization
                else "learned_embeds.safetensors"
            )
            save_path = os.path.join(self.args.output_dir, weight_name)
            save_progress(
                self.text_encoder,
                self.placeholder_token_ids,
                self.accelerator,
                self.args,
                save_path,
                safe_serialization=not self.args.no_safe_serialization,
            )

            if self.args.push_to_hub:
                save_model_card(
                    self.repo_id,
                    images=[],
                    base_model=self.args.pretrained_model_name_or_path,
                    repo_folder=self.args.output_dir,
                )
                upload_folder(
                    repo_id=self.repo_id,
                    folder_path=self.args.output_dir,
                    commit_message="End of training",
                    ignore_patterns=["step_*", "epoch_*"],
                )

        self.accelerator.end_training()

        # 保存 loss 轨迹
        loss_df = pd.DataFrame(self.loss_list)
        loss_df.to_csv(os.path.join(self.args.output_dir, "loss.csv"))


# ===== 具体 Trainer：ThreeLoss / TwoLoss，与原 trainer.py 对齐 =====
class ThreeLossFluxTrainer(BaseFluxTrainer):
    """Flux 版本的 ThreeLossTrainer"""

    def __init__(self, args):
        super().__init__(args)

    def training_step(self, batch):
        toxic_prompt = batch["prompt"]
        safe_prompt = batch["rewritten_prompt"]
        pseudo_toxic = batch["pseudo_prompt"]
        pseudo_benign = batch["pseudo_rewritten"]

        z_t, timesteps = self.init_latent(len(toxic_prompt))

        # 逐个编码 prompts 以减少显存峰值（虽然慢一些，但显存占用更小）
        device = self.accelerator.device
        
        # 确保 text encoders 在 GPU 上（只移动一次）
        if getattr(self.flux_pipeline, "text_encoder", None) is not None:
            self.flux_pipeline.text_encoder.to(device=device, dtype=self.weight_dtype)
        if getattr(self.flux_pipeline, "text_encoder_2", None) is not None:
            self.flux_pipeline.text_encoder_2.to(device=device, dtype=self.weight_dtype)

        bsz = len(toxic_prompt)
        
        # 逐个编码 prompts，避免批量编码时的显存峰值
        # 注意：
        # 1. 虽然 text encoder 的大部分参数被冻结，但 embedding 层需要梯度
        #    所以不能使用 no_grad()，否则梯度无法传播到 embedding
        # 2. Flux 使用 dual encoder (CLIP + T5)：
        #    - CLIP (text_encoder): 限制 77 tokens，超出的会被截断（这是正常的）
        #    - T5 (text_encoder_2): 支持 512 tokens，长文本由 T5 处理
        #    所以即使看到 CLIP 的截断警告，T5 仍会正确处理完整文本
        prompt_embeds_toxic, pooled_embeds_toxic, text_ids = self.flux_pipeline.encode_prompt(
            prompt=toxic_prompt,
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512,  # T5 的最大长度，CLIP 会自动截断到 77
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        prompt_embeds_safe, pooled_embeds_safe, _ = self.flux_pipeline.encode_prompt(
            prompt=safe_prompt,
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        prompt_embeds_pseudo, pooled_embeds_pseudo, _ = self.flux_pipeline.encode_prompt(
            prompt=pseudo_toxic,
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        prompt_embeds_benign, pooled_embeds_benign, _ = self.flux_pipeline.encode_prompt(
            prompt=pseudo_benign,
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 准备 timestep 和 guidance
        if isinstance(timesteps, torch.Tensor):
            t_continuous = timesteps.to(device).to(z_t.dtype) / 1000.0
        else:
            t_continuous = torch.tensor(timesteps, device=device, dtype=z_t.dtype) / 1000.0

        guidance = None
        if getattr(self.transformer.config, "guidance_embeds", False):
            guidance_scale = float(getattr(self.args, "guidance_scale", 1.0))
            guidance = torch.full((bsz,), guidance_scale, device=device, dtype=z_t.dtype)

        latent_image_ids = self.latent_image_ids.to(device=device, dtype=z_t.dtype)

        # 批量预测（复用相同的 z_t, timesteps, latent_image_ids）
        noise_pred_toxic = self.transformer(
            hidden_states=z_t,
            timestep=t_continuous,
            guidance=guidance,
            pooled_projections=pooled_embeds_toxic,
            encoder_hidden_states=prompt_embeds_toxic,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs={},
            return_dict=False,
        )[0]

        noise_pred_rewritten = self.transformer(
            hidden_states=z_t,
            timestep=t_continuous,
            guidance=guidance,
            pooled_projections=pooled_embeds_safe,
            encoder_hidden_states=prompt_embeds_safe,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs={},
            return_dict=False,
        )[0]

        noise_pred_pseudo = self.transformer(
            hidden_states=z_t,
            timestep=t_continuous,
            guidance=guidance,
            pooled_projections=pooled_embeds_pseudo,
            encoder_hidden_states=prompt_embeds_pseudo,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs={},
            return_dict=False,
        )[0]

        noise_pred_benign = self.transformer(
            hidden_states=z_t,
            timestep=t_continuous,
            guidance=guidance,
            pooled_projections=pooled_embeds_benign,
            encoder_hidden_states=prompt_embeds_benign,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs={},
            return_dict=False,
        )[0]

        # 清理中间变量，释放显存（在 loss 计算后）

        # noise_pred 是 pack 后的形状 [B, N_tokens, in_channels]，不是 [B, C, H, W]
        dist_ps_rw = F.mse_loss(
            noise_pred_pseudo, noise_pred_rewritten, reduction="none"
        ).reshape(bsz, -1).mean(dim=1).mean()
        dist_ps_pt = F.mse_loss(
            noise_pred_pseudo, noise_pred_toxic, reduction="none"
        ).reshape(bsz, -1).mean(dim=1).mean()
        dist_rw_or = F.mse_loss(
            noise_pred_rewritten, noise_pred_toxic, reduction="none"
        ).reshape(bsz, -1).mean(dim=1).mean()
        dist_bn_rw = F.mse_loss(
            noise_pred_benign, noise_pred_rewritten, reduction="none"
        ).reshape(bsz, -1).mean(dim=1).mean()
        margin = self.args.margin_coef * dist_rw_or.detach()

        triplet_loss = F.relu(dist_ps_rw - dist_ps_pt + margin).mean() * 10
        align_loss = dist_ps_rw
        benign_loss = dist_bn_rw

        loss = (
            self.args.lambda_align * align_loss
            + self.args.lambda_triplet * triplet_loss
            + self.args.lambda_benign * benign_loss
        )

        self.accelerator.log(
            {
                "loss/total": loss.detach().cpu().item(),
                "loss/align": align_loss.detach().cpu().item(),
                "loss/triplet": triplet_loss.detach().cpu().item(),
                "loss/benign": benign_loss.detach().cpu().item(),
                "dist/pseudo_rewritten": dist_ps_rw.detach().cpu().item(),
                "dist/pseudo_toxic": dist_ps_pt.detach().cpu().item(),
                "dist/rewritten_origin": dist_rw_or.detach().cpu().item(),
                "dist/benign_rewritten": dist_bn_rw.detach().cpu().item(),
                "train/margin": margin,
                "train/learning_rate": self.args.learning_rate,
            },
            step=self.global_step,
        )

        self.loss_list.append(
            {"loss": loss.detach().cpu().item(), "step": self.global_step}
        )

        return loss


class TwoLossFluxTrainer(BaseFluxTrainer):
    """Flux 版本的 TwoLossTrainer"""

    def __init__(self, args):
        super().__init__(args)

    def training_step(self, batch):
        toxic_prompt = batch["prompt"]
        safe_prompt = batch["rewritten_prompt"]
        pseudo_toxic = batch["pseudo_prompt"]
        pseudo_benign = batch["pseudo_rewritten"]

        z_t, timesteps = self.init_latent(len(toxic_prompt))

        # 逐个编码 prompts 以减少显存峰值（虽然慢一些，但显存占用更小）
        device = self.accelerator.device
        
        # 确保 text encoders 在 GPU 上（只移动一次）
        if getattr(self.flux_pipeline, "text_encoder", None) is not None:
            self.flux_pipeline.text_encoder.to(device=device, dtype=self.weight_dtype)
        if getattr(self.flux_pipeline, "text_encoder_2", None) is not None:
            self.flux_pipeline.text_encoder_2.to(device=device, dtype=self.weight_dtype)

        bsz = len(toxic_prompt)
        
        # 逐个编码 prompts，避免批量编码时的显存峰值
        # 注意：
        # 1. 虽然 text encoder 的大部分参数被冻结，但 embedding 层需要梯度
        #    所以不能使用 no_grad()，否则梯度无法传播到 embedding
        # 2. Flux 使用 dual encoder (CLIP + T5)：
        #    - CLIP (text_encoder): 限制 77 tokens，超出的会被截断（这是正常的）
        #    - T5 (text_encoder_2): 支持 512 tokens，长文本由 T5 处理
        #    所以即使看到 CLIP 的截断警告，T5 仍会正确处理完整文本
        prompt_embeds_toxic, pooled_embeds_toxic, text_ids = self.flux_pipeline.encode_prompt(
            prompt=toxic_prompt,
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512,  # T5 的最大长度，CLIP 会自动截断到 77
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        prompt_embeds_safe, pooled_embeds_safe, _ = self.flux_pipeline.encode_prompt(
            prompt=safe_prompt,
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        prompt_embeds_pseudo, pooled_embeds_pseudo, _ = self.flux_pipeline.encode_prompt(
            prompt=pseudo_toxic,
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        prompt_embeds_benign, pooled_embeds_benign, _ = self.flux_pipeline.encode_prompt(
            prompt=pseudo_benign,
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 准备 timestep 和 guidance
        if isinstance(timesteps, torch.Tensor):
            t_continuous = timesteps.to(device).to(z_t.dtype) / 1000.0
        else:
            t_continuous = torch.tensor(timesteps, device=device, dtype=z_t.dtype) / 1000.0

        guidance = None
        if getattr(self.transformer.config, "guidance_embeds", False):
            guidance_scale = float(getattr(self.args, "guidance_scale", 1.0))
            guidance = torch.full((bsz,), guidance_scale, device=device, dtype=z_t.dtype)

        latent_image_ids = self.latent_image_ids.to(device=device, dtype=z_t.dtype)

        # 批量预测（复用相同的 z_t, timesteps, latent_image_ids）
        noise_pred_toxic = self.transformer(
            hidden_states=z_t,
            timestep=t_continuous,
            guidance=guidance,
            pooled_projections=pooled_embeds_toxic,
            encoder_hidden_states=prompt_embeds_toxic,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs={},
            return_dict=False,
        )[0]

        noise_pred_rewritten = self.transformer(
            hidden_states=z_t,
            timestep=t_continuous,
            guidance=guidance,
            pooled_projections=pooled_embeds_safe,
            encoder_hidden_states=prompt_embeds_safe,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs={},
            return_dict=False,
        )[0]

        noise_pred_pseudo = self.transformer(
            hidden_states=z_t,
            timestep=t_continuous,
            guidance=guidance,
            pooled_projections=pooled_embeds_pseudo,
            encoder_hidden_states=prompt_embeds_pseudo,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs={},
            return_dict=False,
        )[0]

        noise_pred_benign = self.transformer(
            hidden_states=z_t,
            timestep=t_continuous,
            guidance=guidance,
            pooled_projections=pooled_embeds_benign,
            encoder_hidden_states=prompt_embeds_benign,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs={},
            return_dict=False,
        )[0]

        # 清理中间变量，释放显存（在 loss 计算后）

        # noise_pred 是 pack 后的形状 [B, N_tokens, in_channels]，不是 [B, C, H, W]
        dist_ps_rw = F.mse_loss(
            noise_pred_pseudo, noise_pred_rewritten, reduction="none"
        ).reshape(bsz, -1).mean(dim=1).mean()
        dist_ps_pt = F.mse_loss(
            noise_pred_pseudo, noise_pred_toxic, reduction="none"
        ).reshape(bsz, -1).mean(dim=1).mean()
        dist_rw_or = F.mse_loss(
            noise_pred_rewritten, noise_pred_toxic, reduction="none"
        ).reshape(bsz, -1).mean(dim=1).mean()
        dist_bn_rw = F.mse_loss(
            noise_pred_benign, noise_pred_rewritten, reduction="none"
        ).reshape(bsz, -1).mean(dim=1).mean()
        margin = self.args.margin_coef * dist_rw_or.detach()

        triplet_loss = F.relu(dist_ps_rw - dist_ps_pt + margin).mean() * 10
        benign_loss = dist_bn_rw

        loss = (
            self.args.lambda_triplet * triplet_loss
            + (1 - self.args.lambda_triplet) * benign_loss
        )

        # 可选：对抗训练，与原 TwoLossTrainer 保持一致
        adv_loss = 0
        if getattr(self.args, "adv_train", False):
            from attack_utils import pgd_attack

            pseudo_adv = noise_pred_pseudo.detach().clone().requires_grad_(True)
            if self.args.adv_type == "pgd":
                pseudo_adv = pgd_attack(
                    pseudo_adv,
                    noise_pred_rewritten.detach(),
                    F.mse_loss,
                    lambda x: x,
                    epsilon=getattr(self.args, "adv_eps", 0.3),
                    alpha=getattr(self.args, "adv_alpha", 0.1),
                    iters=getattr(self.args, "adv_iters", 20),
                )
            else:
                raise ValueError(
                    f"Unsupported adversarial attack type: {self.args.adv_type}"
                )
            if pseudo_adv is not None:
                dist_ps_rw_adv = F.mse_loss(
                    pseudo_adv, noise_pred_rewritten.detach(), reduction="none"
                ).mean()
                adv_loss = self.args.lambda_triplet * dist_ps_rw_adv
                loss = loss + adv_loss

        self.accelerator.log(
            {
                "loss/total": loss.detach().cpu().item(),
                "loss/triplet": triplet_loss.detach().cpu().item(),
                "loss/benign": benign_loss.detach().cpu().item(),
                "dist/pseudo_rewritten": dist_ps_rw.detach().cpu().item(),
                "dist/pseudo_toxic": dist_ps_pt.detach().cpu().item(),
                "dist/rewritten_origin": dist_rw_or.detach().cpu().item(),
                "dist/benign_rewritten": dist_bn_rw.detach().cpu().item(),
                "train/margin": margin,
                "train/learning_rate": self.args.learning_rate,
                "loss/adv": adv_loss
                if isinstance(adv_loss, float)
                else (
                    adv_loss.detach().cpu().item()
                    if torch.is_tensor(adv_loss)
                    else 0
                ),
            },
            step=self.global_step,
        )

        self.loss_list.append(
            {"loss": loss.detach().cpu().item(), "step": self.global_step}
        )

        return loss

