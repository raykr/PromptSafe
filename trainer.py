from abc import abstractmethod
import json
import logging
import math
import os
from pathlib import Path
import shutil
import diffusers
import pandas as pd
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import upload_folder, create_repo
from diffusers.utils.import_utils import is_xformers_available
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor, CLIPTextModelWithProjection
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
import transformers
from evaluate import OursPipeline
from utils.log_utils import save_progress, log_validation, save_model_card
from data import PromptPairDataset
from diffusers.utils import check_min_version, is_wandb_available
from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="INFO")

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.30.0.dev0")

class BaseTrainer:
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

    def setup_accelerator(self):
        if self.args.seed is not None:
            set_seed(self.args.seed)

        logging_dir = os.path.join(self.args.output_dir, self.args.logging_dir)
        accelerator_project_config = ProjectConfiguration(project_dir=self.args.output_dir, logging_dir=logging_dir)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=self.args.mixed_precision,
            log_with=self.args.report_to,
            project_config=accelerator_project_config,
        )

        if torch.backends.mps.is_available():
            self.accelerator.native_amp = False

        # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
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
                raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

        # Make one log on every process with the configuration for debugging.
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

        # Handle the repository creation
        if self.accelerator.is_main_process:
            if self.args.output_dir is not None:
                os.makedirs(self.args.output_dir, exist_ok=True)

            if self.args.push_to_hub:    
                self.repo_id = create_repo(
                    repo_id=self.args.hub_model_id or Path(self.args.output_dir).name, exist_ok=True, token=self.args.hub_token
                ).repo_id

        # save all the arguments in a config file in the output directory
        args_dict = vars(self.args)
        with open(os.path.join(self.args.output_dir, "config.json"), "w") as f:
            json.dump(args_dict, f, indent=4)

    def setup_models(self):
        # Load tokenizer
        if self.args.tokenizer_name:
            self.tokenizer = CLIPTokenizer.from_pretrained(self.args.tokenizer_name)
        else:
            self.tokenizer = CLIPTokenizer.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="tokenizer")

        self.unet = UNet2DConditionModel.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder="unet", revision=self.args.revision, variant=self.args.variant
        )

        self.text_encoder = CLIPTextModel.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder="text_encoder", revision=self.args.revision
        )

        # Load scheduler and models
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder="vae", revision=self.args.revision, variant=self.args.variant
        )
        # Load CLIP model
        self.image_processor = CLIPImageProcessor.from_pretrained(self.args.clip_model_path)
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.args.clip_model_path)

        if self.args.gradient_checkpointing:
            # Keep unet in train mode if we are using gradient checkpointing to save memory.
            # The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
            self.unet.train()
            self.text_encoder.gradient_checkpointing_enable()
            self.unet.enable_gradient_checkpointing()

    def setup_memory(self):
        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if self.args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            
        if self.args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logger.warning(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                self.unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")
        
    def setup_placeholder_tokens(self):
        placeholder_tokens = [self.args.placeholder_token]
        for i in range(1, self.args.num_vectors):
            placeholder_tokens.append(f"{self.args.placeholder_token}_{i}")

        num_added_tokens = self.tokenizer.add_tokens(placeholder_tokens)
        if num_added_tokens != self.args.num_vectors:
            raise ValueError(
                f"The tokenizer already contains the token {self.args.placeholder_token}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )

        token_ids = self.tokenizer.encode(self.args.initializer_token, add_special_tokens=False)
        if len(token_ids) > 1:
            raise ValueError("The initializer token must be a single token.")

        self.initializer_token_id = token_ids[0]
        self.placeholder_token_ids = self.tokenizer.convert_tokens_to_ids(placeholder_tokens)

        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        token_embeds = self.text_encoder.get_input_embeddings().weight.data
        with torch.no_grad():
            for token_id in self.placeholder_token_ids:
                token_embeds[token_id] = token_embeds[self.initializer_token_id].clone()

        # Freeze models
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.text_model.encoder.requires_grad_(False)
        self.text_encoder.text_model.final_layer_norm.requires_grad_(False)
        self.text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

    def setup_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            [{
                "params": self.text_encoder.get_input_embeddings().parameters(),
                "lr": self.args.learning_rate,
            }],
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon,
        )

    def setup_dataset(self):
        self.train_dataset = PromptPairDataset(
            csv_path=self.args.train_data_csv,
            tokenizer=self.tokenizer,
            position=self.args.position,
            placeholder_token=(" ".join(self.tokenizer.convert_ids_to_tokens(self.placeholder_token_ids))),
            repeats=self.args.repeats,
            set="train",
        )
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            num_workers=self.args.dataloader_num_workers
        )

    def setup_scheduler(self):
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        logger.info(f"  num_update_steps_per_epoch = {num_update_steps_per_epoch}")
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch
        self.args.num_train_epochs = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch * self.accelerator.num_processes)

        self.lr_scheduler = get_scheduler(
            self.args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.lr_warmup_steps * self.accelerator.num_processes,
            num_training_steps=self.args.max_train_steps * self.accelerator.num_processes,
            num_cycles=self.args.lr_num_cycles,
        )

        if self.args.scale_lr:
            self.args.learning_rate = (
                self.args.learning_rate * self.args.gradient_accumulation_steps * self.args.train_batch_size * self.accelerator.num_processes
            )

    def setup_training_state(self):
        self.text_encoder.train()
        self.text_encoder, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.text_encoder, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

        self.unet.to(self.accelerator.device, dtype=self.weight_dtype)
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        self.image_encoder.to(self.accelerator.device, dtype=self.weight_dtype)

        self.orig_embeds_params = self.accelerator.unwrap_model(self.text_encoder).get_input_embeddings().weight.data.clone()

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers("textual_inversion", config=vars(self.args))

    def train(self):
        total_batch_size = self.args.train_batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.args.train_batch_size}")
        logger.info(f"  Device count = {self.accelerator.num_processes}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total train batch size (w. parallel={self.args.train_batch_size}, distributed={self.accelerator.num_processes}, accumulation={self.args.gradient_accumulation_steps}) = {total_batch_size}")
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
                
                # 记录进度信息
                logs = {
                    "loss": loss.detach().cpu().item(),
                    "step": self.global_step,
                    "epoch": self.global_step // len(self.train_dataloader),
                    "progress": self.global_step / self.args.max_train_steps
                }
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=self.global_step)

                if self.global_step >= self.args.max_train_steps:
                    break

        self.save_final_model()

    @abstractmethod
    def training_step(self, batch):
        pass

    def predict_noise(self, prompt, z_t, timesteps):
        input_ids = self.tokenizer(
            prompt, padding="max_length", truncation=True,
            max_length=self.tokenizer.model_max_length, return_tensors="pt"
        ).input_ids.to(self.accelerator.device)
        
        text_encoder_out = self.text_encoder(input_ids)
        encoder_hidden_states = text_encoder_out.last_hidden_state.to(dtype=self.weight_dtype)

        noise_pred = self.unet(
            z_t, timesteps,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs={}
        ).sample
        return noise_pred, encoder_hidden_states
    
    def init_latent(self, bsz):
        latent_shape = (bsz, self.unet.config.in_channels, 64, 64)  # (B, 4, 64, 64)
        z = torch.randn(latent_shape, device=self.accelerator.device, dtype=self.weight_dtype)
        noise = torch.randn_like(z)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=z.device).long()
        z_t = self.noise_scheduler.add_noise(z, noise, timesteps)
        return z_t, timesteps
    
    def update_embeddings(self):
        index_no_updates = torch.ones((len(self.tokenizer),), dtype=torch.bool)
        index_no_updates[min(self.placeholder_token_ids) : max(self.placeholder_token_ids) + 1] = False

        with torch.no_grad():
            self.accelerator.unwrap_model(self.text_encoder).get_input_embeddings().weight[
                index_no_updates
            ] = self.orig_embeds_params[index_no_updates]

    def handle_sync_gradients(self, progress_bar):
        progress_bar.update(1)
        self.global_step += 1

        if self.global_step % self.args.save_steps == 0:
            self.save_checkpoint()

        if self.accelerator.is_main_process:
            if self.global_step % self.args.checkpointing_steps == 0:
                self.save_state()
            
    def load_checkpoint(self):
        if self.args.resume_from_checkpoint != "latest":
            path = os.path.basename(self.args.resume_from_checkpoint)
        else:
            dirs = os.listdir(self.args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            self.accelerator.print(
                f"Checkpoint '{self.args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            self.args.resume_from_checkpoint = None
            self.global_step = 0
        else:
            self.accelerator.print(f"Resuming from checkpoint {path}")
            self.accelerator.load_state(os.path.join(self.args.output_dir, path))
            self.global_step = int(path.split("-")[1])
            self.first_epoch = self.global_step // math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)

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

        if (self.args.validation_prompt is not None or self.args.validation_file is not None) and self.global_step % self.args.validation_steps == 0:
            self.validate(save_path)
    
    def validate(self, save_path):
        if self.args.validation_file is not None:
            # 读取 csv 文件，取出prompt列
            validation_prompts = pd.read_csv(self.args.validation_file)['prompt'].tolist()
        else:
            validation_prompts = [self.args.validation_prompt]

        logger.info(
            f"Running validation... \n Generating {self.args.num_validation_images} images with prompt:"
            f" {validation_prompts}."
        )

        pipeline = OursPipeline(
            model_path=self.args.pretrained_model_name_or_path,
            safe_embedding_paths=[save_path],  # Pass save_path as a list
            safe_tokens=[self.args.placeholder_token],  # Pass the placeholder_token as a list.
            device=self.accelerator.device
        )
        pipeline.generate_images(validation_prompts, self.args.output_dir + "/validation/" + str(self.global_step), 4, 28, 7.5, 42, 0, None)
        del pipeline
        torch.cuda.empty_cache()
        

    def save_state(self):
        if self.args.checkpoints_total_limit is not None:
            checkpoints = os.listdir(self.args.output_dir)
            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

            if len(checkpoints) >= self.args.checkpoints_total_limit:
                num_to_remove = len(checkpoints) - self.args.checkpoints_total_limit + 1
                removing_checkpoints = checkpoints[0:num_to_remove]

                logger.info(
                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                )
                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                for removing_checkpoint in removing_checkpoints:
                    removing_checkpoint = os.path.join(self.args.output_dir, removing_checkpoint)
                    shutil.rmtree(removing_checkpoint)

        save_path = os.path.join(self.args.output_dir, f"checkpoint-{self.global_step}")
        self.accelerator.save_state(save_path)
        logger.info(f"Saved state to {save_path}")

    def run_validation(self):
        images = log_validation(
            self.text_encoder,
            self.tokenizer,
            self.unet,
            self.vae,
            self.args,
            self.accelerator,
            self.weight_dtype,
            self.global_step // math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        )
        # Save validation set images
        os.makedirs(os.path.join(self.args.output_dir, 'validation'), exist_ok=True)
        for i, image in enumerate(images):
            image.save(os.path.join(self.args.output_dir, 'validation', f"{self.global_step}_{i}.png"))

    def save_final_model(self):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            if self.args.push_to_hub and not self.args.save_as_full_pipeline:
                logger.warning("Enabling full model saving because --push_to_hub=True was specified.")
                save_full_model = True
            else:
                save_full_model = self.args.save_as_full_pipeline

            if save_full_model:
                pipeline = StableDiffusionPipeline.from_pretrained(
                    self.args.pretrained_model_name_or_path,
                    text_encoder=self.accelerator.unwrap_model(self.text_encoder),
                    vae=self.vae,
                    unet=self.unet,
                    tokenizer=self.tokenizer,
                )
                pipeline.save_pretrained(self.args.output_dir)

            weight_name = "learned_embeds.bin" if self.args.no_safe_serialization else "learned_embeds.safetensors"
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

        # Save the loss information
        loss_df = pd.DataFrame(self.loss_list)
        loss_df.to_csv(os.path.join(self.args.output_dir, "loss.csv")) 

    
class ThreeLossTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)

    def training_step(self, batch):
        # 提取 prompt pair
        toxic_prompt = batch["prompt"]
        safe_prompt = batch["rewritten_prompt"]
        pseudo_toxic = batch["pseudo_prompt"]
        pseudo_benign = batch["pseudo_rewritten"]

        z_t, timesteps = self.init_latent(len(toxic_prompt))

        # 预测噪声
        noise_pred_toxic, _ = self.predict_noise(toxic_prompt, z_t, timesteps)
        noise_pred_rewritten, _ = self.predict_noise(safe_prompt, z_t, timesteps)
        noise_pred_pseudo, _ = self.predict_noise(pseudo_toxic, z_t, timesteps)
        noise_pred_benign, _ = self.predict_noise(pseudo_benign, z_t, timesteps)

        dist_ps_rw = F.mse_loss(noise_pred_pseudo, noise_pred_rewritten, reduction='none').mean(dim=(1,2,3)).mean()
        dist_ps_pt = F.mse_loss(noise_pred_pseudo, noise_pred_toxic, reduction='none').mean(dim=(1,2,3)).mean()
        dist_rw_or = F.mse_loss(noise_pred_rewritten, noise_pred_toxic, reduction='none').mean(dim=(1,2,3)).mean()
        dist_bn_rw = F.mse_loss(noise_pred_benign, noise_pred_rewritten, reduction='none').mean(dim=(1,2,3)).mean()
        margin = self.args.margin_coef * dist_rw_or.detach()  # detach 是为了不反传梯度

        # 计算 triplet loss
        triplet_loss = F.relu(dist_ps_rw - dist_ps_pt + margin).mean() * 10 # 拉平数量级
        align_loss = dist_ps_rw
        benign_loss = dist_bn_rw

        loss = self.args.lambda_align * align_loss + self.args.lambda_triplet * triplet_loss + self.args.lambda_benign * benign_loss

        # 记录日志
        self.accelerator.log({
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
        }, step=self.global_step)

        # Record loss info
        self.loss_list.append({
            "loss": loss.detach().cpu().item(),
            "step": self.global_step
        })

        return loss

class TwoLossTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)

    def training_step(self, batch):
        # 提取 prompt pair
        toxic_prompt = batch["prompt"]
        safe_prompt = batch["rewritten_prompt"]
        pseudo_toxic = batch["pseudo_prompt"]
        pseudo_benign = batch["pseudo_rewritten"]

        z_t, timesteps = self.init_latent(len(toxic_prompt))

        # 预测噪声
        noise_pred_toxic, _ = self.predict_noise(toxic_prompt, z_t, timesteps)
        noise_pred_rewritten, _ = self.predict_noise(safe_prompt, z_t, timesteps)
        noise_pred_pseudo, _ = self.predict_noise(pseudo_toxic, z_t, timesteps)
        noise_pred_benign, _ = self.predict_noise(pseudo_benign, z_t, timesteps)

        dist_ps_rw = F.mse_loss(noise_pred_pseudo, noise_pred_rewritten, reduction='none').mean(dim=(1,2,3)).mean()
        dist_ps_pt = F.mse_loss(noise_pred_pseudo, noise_pred_toxic, reduction='none').mean(dim=(1,2,3)).mean()
        dist_rw_or = F.mse_loss(noise_pred_rewritten, noise_pred_toxic, reduction='none').mean(dim=(1,2,3)).mean()
        dist_bn_rw = F.mse_loss(noise_pred_benign, noise_pred_rewritten, reduction='none').mean(dim=(1,2,3)).mean()
        margin = self.args.margin_coef * dist_rw_or.detach()  # detach 是为了不反传梯度

        # 计算 triplet loss
        triplet_loss = F.relu(dist_ps_rw - dist_ps_pt + margin).mean() * 10 # 拉平数量级
        benign_loss = dist_bn_rw

        loss = self.args.lambda_triplet * triplet_loss + (1 - self.args.lambda_triplet) * benign_loss

        # 对抗训练部分（直接对noise_pred_pseudo做扰动）
        adv_loss = 0
        if hasattr(self.args, 'adv_train') and self.args.adv_train:
            from attack_utils import pgd_attack
            pseudo_adv = noise_pred_pseudo.detach().clone().requires_grad_(True)
            # 选择攻击类型
            if self.args.adv_type == 'pgd':
                pseudo_adv = pgd_attack(pseudo_adv, noise_pred_rewritten.detach(), F.mse_loss, lambda x: x, epsilon=getattr(self.args, 'adv_eps', 0.3), alpha=getattr(self.args, 'adv_alpha', 0.1), iters=getattr(self.args, 'adv_iters', 20))
            else:
                raise ValueError(f"Unsupported adversarial attack type: {self.args.adv_type}")
            if pseudo_adv is not None:
                dist_ps_rw_adv = F.mse_loss(pseudo_adv, noise_pred_rewritten.detach(), reduction='none').mean()
                adv_loss = self.args.lambda_triplet * dist_ps_rw_adv
                loss = loss + adv_loss

        # 记录日志
        self.accelerator.log({
            "loss/total": loss.detach().cpu().item(),
            "loss/triplet": triplet_loss.detach().cpu().item(),
            "loss/benign": benign_loss.detach().cpu().item(),
            "dist/pseudo_rewritten": dist_ps_rw.detach().cpu().item(),
            "dist/pseudo_toxic": dist_ps_pt.detach().cpu().item(),
            "dist/rewritten_origin": dist_rw_or.detach().cpu().item(),
            "dist/benign_rewritten": dist_bn_rw.detach().cpu().item(),
            "train/margin": margin,
            "train/learning_rate": self.args.learning_rate,
            "loss/adv": adv_loss if isinstance(adv_loss, float) else (adv_loss.detach().cpu().item() if torch.is_tensor(adv_loss) else 0),
        }, step=self.global_step)

        # Record loss info
        self.loss_list.append({
            "loss": loss.detach().cpu().item(),
            "step": self.global_step
        })

        return loss