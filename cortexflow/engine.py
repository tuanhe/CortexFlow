"""
Main CortexFlow inference engine
"""

from typing import Dict, Optional, Tuple, Any
import torch
import torch.nn as nn
import numpy as np

from .configs import ModelConfig, InferenceConfig, BackendType, VLAModelType
from .backends import ComputeBackend, TritonBackend, PyTorchBackend
from .models import VLAModel, Pi05Model, OpenVLAModel


class CortexFlowEngine:
    """CortexFlow 统一推理引擎
    
    支持多种模型和后端的可插拔架构
    """
    
    def __init__(
        self,
        checkpoint: Dict[str, Any],
        model_config: ModelConfig,
        inference_config: InferenceConfig,
    ):
        """初始化推理引擎"""
        self.model_config = model_config
        self.inference_config = inference_config
        
        # 创建模型和后端
        self.model = self._create_model(model_config.model_type)
        self.backend = self._create_backend(inference_config.backend)
        
        # 初始化权重和缓冲区
        self.weights = self._initialize_weights(checkpoint)
        self.buffers = self._initialize_buffers()
        
        # 设置 tokenizer
        self.tokenizer = None
        self.prompt_embedding = None
        self._prompt_embed_scale = None
        if inference_config.discrete_state_input:
            self._setup_tokenizer(checkpoint)
        
        # 设置 RoPE embeddings
        self._setup_rope_embeddings()
        
        # 设置 CUDA graphs
        self.infer_graph = None
        if inference_config.use_cuda_graphs:
            self._setup_cuda_graphs()
        
        print(f"✓ CortexFlow Engine initialized")
        print(f"  - Model: {model_config.model_type.value}")
        print(f"  - Backend: {inference_config.backend.value}")
        print(f"  - Device: {inference_config.device}")
    
    def _create_model(self, model_type: VLAModelType) -> VLAModel:
        """创建模型实例"""
        model_map = {
            VLAModelType.PI05: Pi05Model,
            VLAModelType.PI0: Pi05Model,
            VLAModelType.OPENVLA: OpenVLAModel,
        }
        
        if model_type not in model_map:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return model_map[model_type](self.model_config)
    
    def _create_backend(self, backend_type: BackendType) -> ComputeBackend:
        """创建计算后端实例"""
        backend_map = {
            BackendType.TRITON: TritonBackend,
            BackendType.PYTORCH: PyTorchBackend,
        }
        
        if backend_type not in backend_map:
            raise ValueError(f"Unsupported backend type: {backend_type}")
        
        return backend_map[backend_type](self.model_config)
    
    def _initialize_weights(self, checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """初始化权重张量"""
        weights = {}
        weight_shapes = self.model.get_weight_shapes()
        
        # 分配权重张量
        for name, shape in weight_shapes.items():
            weights[name] = torch.empty(
                shape,
                dtype=self.inference_config.dtype,
                device=self.inference_config.device
            )
        
        # 从 checkpoint 加载
        processed_checkpoint = self.model.preprocess_checkpoint(checkpoint)
        for k, v in processed_checkpoint.items():
            if k in weights:
                if v.dtype != self.inference_config.dtype:
                    v = v.to(dtype=self.inference_config.dtype)
                if v.device != torch.device(self.inference_config.device):
                    v = v.to(device=self.inference_config.device)
                weights[k].copy_(v)
        
        # 处理 language embeddings
        if 'language_embeds' in processed_checkpoint:
            weights['language_embeds'] = processed_checkpoint['language_embeds'].to(
                dtype=self.inference_config.dtype,
                device=self.inference_config.device
            )
        
        return weights
    
    def _initialize_buffers(self) -> Dict[str, torch.Tensor]:
        """初始化缓冲区张量"""
        buffers = {}
        buffer_shapes = self.model.get_buffer_shapes(
            self.inference_config.num_views,
            self.inference_config.chunk_size
        )
        
        for name, shape in buffer_shapes.items():
            if 'logits' in name or 'split_k' in name:
                dtype = torch.float32
            elif name == 'valid_encoder_len':
                dtype = torch.int32
            else:
                dtype = self.inference_config.dtype
            
            buffers[name] = torch.empty(
                shape,
                dtype=dtype,
                device=self.inference_config.device
            )
        
        return buffers
    
    def _setup_tokenizer(self, checkpoint: Dict[str, Any]):
        """设置 tokenizer"""
        from transformers import AutoTokenizer
        
        if self.inference_config.tokenizer_path is None:
            raise ValueError("tokenizer_path required when discrete_state_input=True")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.inference_config.tokenizer_path
        )
        
        if "embedding_weight" not in checkpoint:
            raise KeyError("checkpoint must contain 'embedding_weight'")
        
        emb_w = checkpoint["embedding_weight"]
        if isinstance(emb_w, np.ndarray):
            emb_w = torch.from_numpy(emb_w)
        
        emb_w = emb_w.to(
            device=self.inference_config.device,
            dtype=self.inference_config.dtype
        )
        
        self.prompt_embedding = nn.Embedding(
            num_embeddings=emb_w.shape[0],
            embedding_dim=emb_w.shape[1],
            device=self.inference_config.device,
            dtype=self.inference_config.dtype,
        )
        
        with torch.no_grad():
            self.prompt_embedding.weight.copy_(emb_w)
        
        self._prompt_embed_scale = float(emb_w.shape[1] ** 0.5)
    
    def _setup_rope_embeddings(self):
        """设置 RoPE embeddings"""
        num_views = self.inference_config.num_views
        chunk_size = self.inference_config.chunk_size
        max_prompt_len = self.inference_config.max_tokenize_len
        
        max_pos = (num_views * 256 + max_prompt_len - 1) + chunk_size
        position_ids = torch.arange(max_pos + 1, device=self.inference_config.device)
        
        head_dim = self.model_config.head_dim
        inv_freq = 1.0 / (10000 ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32, 
                        device=self.inference_config.device) / head_dim
        ))
        
        k_phase = inv_freq[None, :] * position_ids[:, None]
        k_cos = torch.cos(k_phase).to(self.inference_config.dtype)
        k_sin = torch.sin(k_phase).to(self.inference_config.dtype)
        
        self._rope_table = torch.cat(
            [k_cos[:, :, None], k_sin[:, :, None]], 2
        ).view(-1, head_dim)
        
        # 初始化编码器 rope weights
        prefix_alloc = num_views * 256 + max_prompt_len
        if 'encoder_rope_weights' in self.buffers:
            self.buffers['encoder_rope_weights'][:prefix_alloc].copy_(
                self._rope_table[:prefix_alloc]
            )
        
        if 'valid_encoder_len' in self.buffers:
            self.buffers['valid_encoder_len'].fill_(num_views * 256 + 1)
    
    def _setup_cuda_graphs(self):
        """设置 CUDA graph"""
        self.infer_graph = torch.cuda.CUDAGraph()
        
        for _ in range(self.inference_config.num_warmup_iters):
            self._run_inference()
        
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            self.infer_graph.capture_begin()
            self._run_inference()
            self.infer_graph.capture_end()
        
        print("✓ CUDA graph captured")
    
    def _run_inference(self):
        """执行完整推理"""
        # 1. 视觉编码
        self.backend.vision_encoder(self.buffers, self.weights)
        
        # 2. Token 剪枝（如果启用）
        if self.inference_config.vision_token_pruner is not None:
            self.buffers['vision_x'] = self.inference_config.vision_token_pruner.prune(
                self.buffers['vision_x']
            )
        
        # 3. Transformer 编码
        self.backend.transformer_encoder(self.buffers, self.weights)
        
        # 4. Transformer 解码（带扩散）
        self.backend.transformer_decoder(
            self.buffers,
            self.weights,
            self.model_config.num_diffusion_steps
        )
    
    def build_prompt_embeds(
        self,
        task_prompt: Optional[str],
        state_tokens: Optional[np.ndarray]
    ) -> Tuple[torch.Tensor, int]:
        """构建 prompt embeddings"""
        if not self.inference_config.discrete_state_input:
            return self.weights['language_embeds'], self.weights['language_embeds'].shape[0]
        
        if task_prompt is None or state_tokens is None:
            raise ValueError("task_prompt and state_tokens required")
        
        task_prompt = task_prompt.strip().replace("_", " ")
        state_str = " ".join(map(str, state_tokens.tolist()))
        full_prompt = f"Task: {task_prompt}, State: {state_str};\nAction: "
        
        token_ids = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.inference_config.max_tokenize_len,
            padding=False,
        )["input_ids"][0].to(device=self.inference_config.device)
        
        embeds = self.prompt_embedding(token_ids) * self._prompt_embed_scale
        return embeds, int(embeds.shape[0])
    
    def forward(
        self,
        observation_images: torch.Tensor,
        diffusion_noise: torch.Tensor,
        task_prompt: Optional[str] = None,
        state_tokens: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """执行推理"""
        prompt_embeds, prompt_len = self.build_prompt_embeds(task_prompt, state_tokens)
        
        start = self.inference_config.num_views * 256
        self.buffers['encoder_x'][start:start + prompt_len].copy_(prompt_embeds)
        self.buffers['valid_encoder_len'].fill_(start + prompt_len)
        
        rope_start = start + prompt_len - 1
        rope_end = rope_start + self.inference_config.chunk_size
        if 'decoder_rope_weights' in self.buffers:
            self.buffers['decoder_rope_weights'].copy_(self._rope_table[rope_start:rope_end])
        
        self.buffers['observation_images_normalized'].copy_(observation_images)
        self.buffers['diffusion_noise'].copy_(diffusion_noise)
        
        if self.infer_graph is not None:
            self.infer_graph.replay()
        else:
            self._run_inference()
        
        return self.buffers['diffusion_noise']
    
    @torch.no_grad()
    def predict(
        self,
        observation_images: torch.Tensor,
        task_prompt: Optional[str] = None,
        state_tokens: Optional[np.ndarray] = None,
        noise_scale: float = 1.0,
    ) -> np.ndarray:
        """高层预测接口"""
        diffusion_noise = torch.randn(
            self.inference_config.chunk_size,
            self.model_config.action_dim,
            dtype=self.inference_config.dtype,
            device=self.inference_config.device
        ) * noise_scale
        
        actions = self.forward(
            observation_images,
            diffusion_noise,
            task_prompt,
            state_tokens
        )
        
        actions = self.model.postprocess_output(actions)
        return actions.cpu().numpy()
