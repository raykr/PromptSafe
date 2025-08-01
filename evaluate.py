import os
import argparse
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionPipelineSafe, StableDiffusion3Pipeline, StableDiffusionXLPipeline
from tqdm import tqdm
import pandas as pd
from typing import List, Optional
from transformers import CLIPTokenizer
import numpy as np
import random
from abc import ABC
from diffusers.pipelines.stable_diffusion_safe import SafetyConfig
import torch
from lightning import Fabric
from model import Prompt_Classifier, Contra_Classifier
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from sld import SLDPipeline
from safetensors.torch import load_file
import yaml
from argparse import Namespace


class Predictor:
    def __init__(self, ckpt_path="experiments/attack/runs/prompt_classifier_singleclass_newfinaldata_v0"):
        # 读取ckpt_path路径下的config.yaml
        config_path = os.path.join(ckpt_path, 'config.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
                print(f"Loaded config from {config_path}: {config_dict}")
            opt = Namespace(**config_dict)
            opt.model_dir = ckpt_path
        
        # 初始化tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(opt.tokenizer, subfolder="tokenizer", revision=opt.revision)
        
        # 初始化Fabric
        torch.set_float32_matmul_precision("medium")
        fabric = Fabric(accelerator="cuda", precision="bf16-mixed", devices=1)
        fabric.launch()
        
        # 初始化模型参数
        if opt.single_contra:
            self.model = Contra_Classifier(opt, fabric)
        else:
            self.model = Prompt_Classifier(opt, fabric)

        # 加载模型权重
        checkpoint_path = os.path.join(opt.model_dir, 'model_classifier_best.pth')
        state_dict = torch.load(checkpoint_path, map_location=fabric.device)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict, strict=True)
        self.model = fabric.setup(self.model)
        self.model.eval()

        # 设置设备
        self.device = next(self.model.parameters()).device

    
    def _get_results(self, prompts):
        """
        对单条或多条prompt进行预测，返回预测的类别和置信度，以及所有类别的置信度
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        encoded_batch = self.tokenizer.batch_encode_plus(
            prompts,
            return_tensors="pt",
            max_length=77,
            padding='max_length',
            truncation=True,
        )
        device = next(self.model.parameters()).device
        encoded_batch = {k: v.to(device) for k, v in encoded_batch.items()}
        batch_size = len(prompts)
        indices = torch.zeros(batch_size, dtype=torch.long, device=device)
        label = torch.zeros(batch_size, dtype=torch.long, device=device)
        id = torch.arange(batch_size, device=device)
        self.model.eval()
        with torch.no_grad():
            _, outputs, _, _ = self.model(encoded_batch, indices, label, id)
            probabilities = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            confidences = torch.max(probabilities, dim=1)[0]

            # 统计一下pred=0的数量
            toxic_count = (preds == 0).sum().item()
            benign_count = (preds == 1).sum().item()
            print(f"Predictions: {toxic_count} toxic, {benign_count} benign")

        return zip(preds.cpu().numpy(), confidences.cpu().numpy())
    
    def predict_realvalue(self, prompts, scale=2.0):
        """
        直接按真实值预测
        """
        # 首先scores最后输出的是有毒的置信度
        # 但是预测是按类别的，预测lable=0是有害分类，label=1是无害分类
        # 将所有prompts的预测结果转换为置信度
        results = self._get_results(prompts)
        scores = [conf * scale if pred == 0 else (1 - conf) for pred, conf in results]
        return scores
    
    def predict_realvalue_with_threshold_and_scale(self, prompts, threshold=0.8, scale=2.0):
        """
        直接按真实值预测
        """
        # 首先scores最后输出的是有毒的置信度
        # 但是预测是按类别的，预测lable=0是有害分类，label=1是无害分类
        # 将所有prompts的预测结果转换为置信度
        results = self._get_results(prompts)
        scores = []
        for pred, conf in results:
            if conf <= threshold:
                # 如果置信度低于阈值，则应该按照反标签预测
                # 例如pref=0, conf=0.5，则认为是无害的，直接赋值最小分0
                # 例如pref=1, conf=0.5，则认为是有害的，直接赋值value * scale
                if pred == 0:
                    scores.append(0.0)
                else:
                    scores.append(conf * scale)
            else:
                # 如果置信度高于阈值，则按真实值预测
                # 例如pref=0, conf=0.9，则认为是有害的，直接赋值 value * scale
                # 例如pref=1, conf=0.9，则认为是无害的，直接赋值最小分0
                # 这里的scale是为了放大无害的置信度
                if pred == 0:
                    scores.append(conf * scale)
                else:
                    scores.append(0)
        return scores
    
    def predict_polarization(self, prompts, threshold=0.1):
        """
        max-min 极化预测
        输出的score是预测为有害的得分
        """
        scores = self.predict_realvalue(prompts)
        # scores 大于 threshold 的认为是有害，直接设置为1，否则为0
        scores = [1.0 if score >= threshold else 0.0 for score in scores]
        return scores
    
    def predict_emphasis_defense(self, prompts, threshold=0.7):
        # 强调防御预测
        results = self._get_results(prompts)
        # 只有在无害置信度大于阈值才被认为是无害
        scores = [conf if pred == 1 and conf >= threshold else 1.0 for pred, conf in results]
        return scores
    
    def predict_emphasis_benign(self, prompts, threshold=0.7):
        # 强调良性预测
        results = self._get_results(prompts)
        # 只有在有害置信度大于阈值才被认为是有害
        scores = [conf if pred == 0 and conf >= threshold else 0.0 for pred, conf in results]
        return scores
    
    def predict_threshold(self, prompts, threshold=0.5):
        """
        基于阈值的预测
        输出的score是预测为有害的得分
        """
        results = self._get_results(prompts)
        scores = []
        for pred, conf in results:
            if pred == 0 and conf >= threshold:
                scores.append(1.0)
            elif pred == 1 and conf >= threshold:
                scores.append(0.0)
            else:
                # 如果置信度低于阈值，按真实值预测
                scores.append(conf if pred == 0 else 1 - conf)
        return scores

    def multi_head_predict(self, prompts, as_json=False):
        """
        输入: prompts(str或list)
        输出: 
          - as_json=False: 每个prompt的所有类别置信度（二维list，每行为一个prompt的softmax概率）
          - as_json=True: 每个prompt输出为{类别名: 置信度}的dict，返回list of dict
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        encoded_batch = self.tokenizer.batch_encode_plus(
            prompts,
            return_tensors="pt",
            max_length=77,
            padding='max_length',
            truncation=True,
        )
        encoded_batch = {k: v.to(self.device) for k, v in encoded_batch.items()}
        batch_size = len(prompts)
        indices = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        label = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        id = torch.arange(batch_size, device=self.device)
        with torch.no_grad():
            _, outputs, _, _ = self.model(encoded_batch, indices, label, id)
            probabilities = torch.softmax(outputs, dim=1)
        prob_np = probabilities.cpu().numpy()
        if not as_json:
            return prob_np.tolist()
        # 获取类别名
        from yx_trival.utils.dataset_utils import prompt_class
        idx2name = [k for k, v in sorted(prompt_class.items(), key=lambda x: x[1])]
        result = []
        for row in prob_np:
            result.append({name: float(conf) for name, conf in zip(idx2name, row)})
        return result


class BaseInferencePipeline(ABC):
    """Basic Reasoning Pipeline Class"""
    
    def __init__(self, args, device: str = "cuda"):
        self.args = args
        self.model_path = args.model_path
        self.device = device
        self.tokenizer = CLIPTokenizer.from_pretrained("/home/raykr/models/openai/clip-vit-large-patch14")
        self.pipe = None
        self.safety_config = None
        self.setup_pipeline()
        self.load_defense_embeddings()
    
    def setup_pipeline(self):
        """设置基础pipeline"""
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        self.pipe = self.pipe.to(self.device)
        
        # 移除安全检查器
        def dummy_checker(images, **kwargs):
            return images, [False] * len(images)
        self.pipe.safety_checker = dummy_checker

    def load_defense_embeddings(self):
        pass
    
    def process_prompt(self, prompt: str) -> str:
        """处理提示词，截断到最大长度"""
        # CLIP tokenizer 的最大长度是 77，我们不需要预留空间
        tokenized = self.tokenizer(prompt, truncation=True, max_length=getattr(self.tokenizer, "model_max_length", 77), return_tensors="pt")
        return self.tokenizer.decode(tokenized["input_ids"][0], skip_special_tokens=True)
    
    def generate_images(
        self,
        prompts: List[str],
        output_dir: str,
        batch_size: int = 1,
        num_inference_steps: int = 28,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        start_index: int = 0,
        ids: Optional[List[int]] = None,
    ):
        """生成图像的主方法"""
        if start_index >= len(prompts):
            print(f"Warning: start_index {start_index} is out of range. Total prompts: {len(prompts)}")
            return
        
        # 设置随机种子
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        generator = None if seed is None else torch.Generator(device=self.device).manual_seed(seed)
        os.makedirs(output_dir, exist_ok=True)
        
        total = len(prompts) - start_index
        batches = (total + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(batches), desc=f"Generating images on {self.device}"):
            batch_start = start_index + batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]
            
            # 处理提示词
            processed_prompts = [self.process_prompt(p) for p in batch_prompts]
            
            # 生成图像
            if self.safety_config:
                images = self.pipe(
                    processed_prompts,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    **self.safety_config
                ).images
            else: 
                images = self.pipe(
                    processed_prompts,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                ).images
            
            # 保存图像
            for i, img in enumerate(images):
                img_idx = batch_start + i
                if ids is not None:
                    img_id = ids[img_idx]
                else:
                    img_id = img_idx
                img.save(os.path.join(output_dir, f"{img_id}.png"))


class SDv14Pipeline(BaseInferencePipeline):
    """Stable Diffusion v1.4"""
    
    def __init__(self, args, device: str = "cuda"):
        super().__init__(args, device)


class SDv3Pipeline(BaseInferencePipeline):
    """Stable Diffusion v3 基础实现，支持多卡分布式"""
    def __init__(self, args, device: str = "cuda", enable_multi_gpu: bool = True):
        self.enable_multi_gpu = enable_multi_gpu
        super().__init__(args, device)

    def setup_pipeline(self):
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model_nf4 = SD3Transformer2DModel.from_pretrained(
            self.model_path,
            subfolder="transformer",
            quantization_config=nf4_config,
            torch_dtype=torch.bfloat16
        )

        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            self.model_path, 
            transformer=model_nf4,
            torch_dtype=torch.bfloat16
        )
        self.pipe.enable_model_cpu_offload()


class SDXLPipeline(BaseInferencePipeline):
    """Stable Diffusion XL"""
    
    def __init__(self, args, device: str = "cuda"):
        super().__init__(args, device)

    def setup_pipeline(self):
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model_path, 
            torch_dtype=torch.float16, 
            variant="fp16", 
            use_safetensors=True
        )
        self.pipe = self.pipe.to(self.device)
        
        def dummy_checker(images, **kwargs):
            return images, [False] * len(images)
        self.pipe.safety_checker = dummy_checker


class SafeGenPipeline(BaseInferencePipeline):
    def __init__(self, args, device: str = "cuda", safety_config="MAX"):
        super().__init__(args, device)

        if safety_config == "MAX":
            self.safety_config = SafetyConfig.MAX
        elif safety_config == "WEAK":
            self.safety_config = SafetyConfig.WEAK
        elif safety_config == "STRONG":
            self.safety_config = SafetyConfig.STRONG
        elif safety_config == "MEDIUM":
            self.safety_config = SafetyConfig.MEDIUM
        
    def setup_pipeline(self):
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        self.pipe = StableDiffusionPipelineSafe.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
        )

        def dummy_checker(images, **kwargs):
            return images, [False] * len(images)
        self.pipe.safety_checker = dummy_checker

        self.pipe = self.pipe.to(self.device)
        

    def process_prompt(self, prompt: str) -> str:
        print(f"Processing prompt: {prompt} (length: {len(prompt)})")
        tokenized = self.tokenizer(prompt, truncation=True, max_length=77, return_tensors="pt")
        return self.tokenizer.decode(tokenized["input_ids"][0], skip_special_tokens=True)


class MACEPipeline(BaseInferencePipeline):
    def __init__(self, args, device: str = "cuda"):
        super().__init__(args, device)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)


class SLD_Pipeline(BaseInferencePipeline):
    def __init__(self, args, device: str = "cuda", safety_config="MAX"):
        super().__init__(args, device)

        if safety_config == "MAX":
            self.safety_config = SafetyConfig.MAX
        elif safety_config == "WEAK":
            self.safety_config = SafetyConfig.WEAK
        elif safety_config == "STRONG":
            self.safety_config = SafetyConfig.STRONG
        elif safety_config == "MEDIUM":
            self.safety_config = SafetyConfig.MEDIUM

    def setup_pipeline(self):
        self.pipe = SLDPipeline.from_pretrained(self.model_path).to(self.device)
        # 移除安全检查器
        def dummy_checker(images, **kwargs):
            return images, [False] * len(images)
        self.pipe.safety_checker = dummy_checker


class UCEPipeline(BaseInferencePipeline):
    def __init__(self, args, device: str = "cuda"):
        self.uce_model_path = args.uce_model_path
        super().__init__(args, device)

    def setup_pipeline(self):
        self.pipe = StableDiffusionPipeline.from_pretrained(self.model_path, 
                                         torch_dtype=torch.float16,
                                         safety_checker=None).to(self.device)
        
        if self.uce_model_path is not None:
            uce_weights = load_file(self.uce_model_path)
            self.pipe.unet.load_state_dict(uce_weights, strict=False)


class OursPipeline(BaseInferencePipeline):
    def __init__(self, args, device: str = "cuda"):
        self.safe_embedding_paths = args.safe_embedding_paths
        self.safe_tokens = args.safe_tokens
        self.position = args.position

        if len(self.safe_embedding_paths) != len(self.safe_tokens):
            raise ValueError("Number of safe embedding paths must match number of safe tokens")
        
        super().__init__(args, device)
        
    def load_defense_embeddings(self):
        self.pipe.load_textual_inversion(self.safe_embedding_paths, self.safe_tokens)
    
    def process_prompt(self, prompt: str) -> str:
        """Process the prompt, add a security token."""
        # Calculate the number of tokens that need to be reserved
        safe_tokens_length = len(self.safe_tokens)
        max_length = 77 - safe_tokens_length
        
        tokenized = self.tokenizer(prompt, truncation=True, max_length=max_length, return_tensors="pt")
        base_prompt = self.tokenizer.decode(tokenized["input_ids"][0], skip_special_tokens=True)
        if self.position == "start":
            return f"{' '.join(self.safe_tokens)} {base_prompt}"
        else:
            return f"{base_prompt} {' '.join(self.safe_tokens)}"


class PromptGuardPipeline(OursPipeline):
    def __init__(self, args, device: str = "cuda"):
        super().__init__(args, device)


class OursDynamicPipeline(OursPipeline):
    """Inference pipeline based on dynamic interpolation of soft prompts using textual inversion"""

    def __init__(self, args, device: str = "cuda"):
        self.predictor = Predictor(args.predictor_path)

        super().__init__(args, device)

        # get token ids
        self.token_ids = self.pipe.tokenizer.convert_tokens_to_ids(args.safe_tokens)
        self.embedding_layer = self.pipe.text_encoder.get_input_embeddings()

        # Save the embedding loaded by textual inversion (strong defense)
        self.soft_embedding = self.embedding_layer.weight[self.token_ids].detach().clone().to(device)

        # Constructing a 'weak defense' version: can use zero vectors or random initialization, more versatile
        self.original_embedding = torch.zeros_like(self.soft_embedding)  # shape: [N, 768]

    def process_prompt(self, prompt: str, add_soft_token: bool = True) -> str:
        safe_tokens_length = len(self.safe_tokens)
        max_length = 77 - safe_tokens_length if add_soft_token else 77
        tokenized = self.tokenizer(prompt, truncation=True, max_length=max_length, return_tensors="pt")
        base_prompt = self.tokenizer.decode(tokenized["input_ids"][0], skip_special_tokens=True)
        if add_soft_token:
            if self.position == "start":
                return f"{' '.join(self.safe_tokens)} {base_prompt}"
            else:
                return f"{base_prompt} {' '.join(self.safe_tokens)}"
        else:
            return base_prompt

    def generate_images(
        self,
        prompts: List[str],
        output_dir: str,
        batch_size: int = 1,
        num_inference_steps: int = 28,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        start_index: int = 0,
        ids: Optional[List[int]] = None,
    ):
        if start_index >= len(prompts):
            print(f"Warning: start_index {start_index} is out of range. Total prompts: {len(prompts)}")
            return

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        generator = None if seed is None else torch.Generator(device=self.device).manual_seed(seed)
        os.makedirs(output_dir, exist_ok=True)

        total = len(prompts) - start_index
        batches = (total + batch_size - 1) // batch_size

        if self.args.predict_type == "polarization":
            scores = self.predictor.predict_polarization(prompts)
        elif self.args.predict_type == "realvalue":
            scores = self.predictor.predict_realvalue(prompts)
        elif self.args.predict_type == "emphasis_defense":
            scores = self.predictor.predict_emphasis_defense(prompts)
        elif self.args.predict_type == "emphasis_benign":
            scores = self.predictor.predict_emphasis_benign(prompts)
        else:
            scores = [1.0] * len(prompts)

        # print(scores)

        for batch_idx in tqdm(range(batches), desc=f"Generating images on {self.device}"):
            batch_start = start_index + batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]

            processed_prompts = []

            for i, prompt in enumerate(batch_prompts):
                score = scores[batch_start + i] if scores is not None else 1.0

                # If the uninstall mode is enabled, then when the score is less than 0.1, the soft token will be uninstalled directly.
                if self.args.enbale_detactive and score < 0.1:
                    processed_prompt = self.process_prompt(prompt, add_soft_token=False)
                else:
                    interpolated = (1 - score) * self.original_embedding + score * self.soft_embedding
                    with torch.no_grad():
                        for j, token_id in enumerate(self.token_ids):
                            self.embedding_layer.weight[token_id] = interpolated[j].to(self.device)
                    processed_prompt = self.process_prompt(prompt, add_soft_token=True)

                processed_prompts.append(processed_prompt)

            # Perform inference
            if self.safety_config:
                images = self.pipe(
                    processed_prompts,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    **self.safety_config
                ).images
            else:
                images = self.pipe(
                    processed_prompts,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                ).images

            # save image
            for i, img in enumerate(images):
                img_idx = batch_start + i
                img_id = ids[img_idx] if ids is not None else img_idx
                img.save(os.path.join(output_dir, f"{img_id}.png"))


def main():
    parser = argparse.ArgumentParser(description="Generate images from prompts using Stable Diffusion")
    parser.add_argument("--input_files", nargs="+", required=True, help="Input CSV file paths")
    parser.add_argument("--output_dir", required=True, help="Output directory for generated images")
    parser.add_argument("--model_path", default="CompVis/stable-diffusion-v1-4", help="Path to the model")
    parser.add_argument("--predictor_path", default="", help="Path to the Lambda predictor model")
    parser.add_argument("--predict_type", default="disable", choices=["realvalue", "polarization", "emphasis_defense", "emphasis_benign", "disable"], help="Type of predictor to use")
    parser.add_argument("--enbale_detactive", action="store_true", help="whether to enable detactive")
    parser.add_argument("--safe_embedding_paths", nargs="+", help="Paths to the safe embeddings")
    parser.add_argument("--soft_prompt_module_path", help="Path to the soft prompt module")
    parser.add_argument("--safe_tokens", nargs="+", help="Safe tokens for defense")
    parser.add_argument("--position", default="end", choices=["start", "end"], help="Position of the safe tokens")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation")
    parser.add_argument("--start_index", type=int, default=0, help="Start index for generation")
    parser.add_argument("--prompt_field", default="prompt", help="Prompt field in the CSV file")
    parser.add_argument("--id_field", default="id", help="ID field in the CSV file")
    parser.add_argument("--pipeline_type", type=str, required=True, help="Type of pipeline to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--safety_config", default="MAX", help="Safety configuration for the pipeline: MAX, WEAK, STRONG, MEDIUM")
    parser.add_argument("--uce_model_path", default=None, help="Safety configuration for the pipeline: MAX, WEAK, STRONG, MEDIUM")
    args = parser.parse_args()

    # Set up device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create the corresponding instance based on the pipeline type.
    if args.pipeline_type == "sdv1x":
        pipeline = SDv14Pipeline(args, device)
    elif args.pipeline_type == "sdv3x":
        pipeline = SDv3Pipeline(args, device)
    elif args.pipeline_type == "sdxl":
        pipeline = SDXLPipeline(args, device)
    elif args.pipeline_type == "safegen":
        pipeline = SafeGenPipeline(args, device, args.safety_config)
    elif args.pipeline_type == "sld":
        pipeline = SLD_Pipeline(args, device, args.safety_config)
    elif args.pipeline_type == "mace":
        pipeline = MACEPipeline(args, device)
    elif args.pipeline_type == "uce":
        pipeline = UCEPipeline(args, device)
    elif args.pipeline_type == "promptguard":
        pipeline = PromptGuardPipeline(args, device)
    elif args.pipeline_type == "ours":
        pipeline = OursPipeline(args, device)
    elif args.pipeline_type == "ours_dynamic":
        pipeline = OursDynamicPipeline(args, device)
    else:
        raise ValueError(f"Unknown pipeline type: {args.pipeline_type}")
    
    # Process each file
    for input_file in args.input_files:
        print(f"\nProcessing {input_file}...")
        
        df = pd.read_csv(input_file)
        prompts = df[args.prompt_field].tolist()
        # If there is an ID field, retrieve the ID list; if not, generate sequentially
        ids = None
        if args.id_field in df.columns:
            ids = df[args.id_field].tolist()
        else:
            ids = list(range(len(prompts)))
        
        # Generate image
        pipeline.generate_images(
            prompts,
            args.output_dir,
            args.batch_size,
            args.num_inference_steps,
            args.guidance_scale,
            args.seed,
            args.start_index,
            ids
        )

if __name__ == "__main__":
    main()