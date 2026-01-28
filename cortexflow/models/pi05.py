"""
Pi0.5 VLA model implementation
"""

from typing import Dict, Tuple, Any
import torch
import numpy as np

from .base import VLAModel
from ..configs.model_config import ModelConfig


class Pi05Model(VLAModel):
    """Pi0.5 VLA 模型实现"""
    
    def get_weight_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Pi0.5 权重张量形状"""
        return {
            # Vision Encoder
            "vision_patch_embedding_w": (14, 14, 3, 1152),
            "vision_patch_embedding_b": (1152,),
            "vision_position_embedding": (256, 1152),
            "vision_attn_qkv_w": (27, 1152, 3 * 1152),
            "vision_attn_qkv_b": (27, 3 * 1152),
            "vision_attn_o_w": (27, 1152, 1152),
            "vision_attn_o_b": (27, 1152),
            "vision_ffn_up_w": (27, 1152, 4304),
            "vision_ffn_up_b": (27, 4304),
            "vision_ffn_down_w": (27, 4304, 1152),
            "vision_ffn_down_b": (27, 1152),
            "vision_pre_attn_norm_w": (27, 1152),
            "vision_pre_attn_norm_b": (27, 1152),
            "vision_pre_ffn_norm_w": (27, 1152),
            "vision_pre_ffn_norm_b": (27, 1152),
            "vision_final_norm_w": (1152,),
            "vision_final_norm_b": (1152,),
            
            # Transformer Encoder
            "encoder_multi_modal_projector_w": (1152, 2048),
            "encoder_multi_modal_projector_b": (2048,),
            "encoder_attn_qkv_w": (18, 2048, 2560),
            "encoder_attn_o_w": (18, 2048, 2048),
            "encoder_ffn_gate_w": (18, 2048, 16384),
            "encoder_ffn_up_w": (18, 2048, 16384),
            "encoder_ffn_down_w": (18, 16384, 2048),
            
            # Transformer Decoder
            "decoder_time_embeds": (10, 1024),
            "decoder_time_mlp_in_w": (1024, 1024),
            "decoder_time_mlp_in_b": (1024,),
            "decoder_time_mlp_out_w": (1024, 1024),
            "decoder_time_mlp_out_b": (1024,),
            "decoder_action_in_proj_w": (32, 1024),
            "decoder_action_in_proj_b": (1024,),
            "decoder_pre_attn_norm_mod_w": (18, 1024, 3 * 1024),
            "decoder_pre_attn_norm_mod_b": (18, 3 * 1024),
            "decoder_pre_ffn_norm_mod_w": (18, 1024, 3 * 1024),
            "decoder_pre_ffn_norm_mod_b": (18, 3 * 1024),
            "decoder_attn_qkv_w": (18, 1024, 2560),
            "decoder_attn_o_w": (18, 2048, 1024),
            "decoder_ffn_gate_w": (18, 1024, 4096),
            "decoder_ffn_up_w": (18, 1024, 4096),
            "decoder_ffn_down_w": (18, 4096, 1024),
            "decoder_action_out_proj_w": (1024, 32),
            "decoder_action_out_proj_b": (32,),
            "decoder_final_norm_mod_w": (1024, 3 * 1024),
            "decoder_final_norm_mod_b": (3 * 1024),
        }
    
    def get_buffer_shapes(self, num_views: int, chunk_size: int) -> Dict[str, Tuple[int, ...]]:
        """Pi0.5 缓冲区张量形状"""
        max_prompt_len = 200
        encoder_seq_len = num_views * 256 + max_prompt_len
        decoder_seq_len = chunk_size
        
        return {
            'observation_images_normalized': (num_views, 224, 224, 3),
            'diffusion_noise': (chunk_size, 32),
            'vision_x': (num_views, 256, 1152),
            'vision_x_norm': (num_views, 256, 1152),
            'vision_QKV': (num_views, 256, 3 * 1152),
            'vision_hidden': (num_views, 256, 4304),
            'vision_x_split_k_buf': (num_views * 256 * 1152 * 4,),
            'encoder_rope_weights': (encoder_seq_len, 256),
            'encoder_x': (encoder_seq_len, 2048),
            'encoder_x_norm': (encoder_seq_len, 2048),
            'encoder_K': (18, encoder_seq_len + decoder_seq_len, 256),
            'encoder_V': (18, encoder_seq_len + decoder_seq_len, 256),
            'encoder_Q': (encoder_seq_len * 8, 256),
            'encoder_hidden': (encoder_seq_len, 16384),
            'valid_encoder_len': (1,),
            'encoder_logits_buf': (encoder_seq_len * 8, encoder_seq_len),
            'encoder_attn_buf': (encoder_seq_len * 8, encoder_seq_len),
            'encoder_ctx_buf': (encoder_seq_len * 8, 256),
            'decoder_rope_weights': (decoder_seq_len, 256),
            'decoder_x': (decoder_seq_len, 1024),
            'decoder_x_buf': (decoder_seq_len, 1024),
            'decoder_action_buf': (decoder_seq_len, 32),
            'decoder_time_emb': (10, decoder_seq_len, 1024),
            'decoder_style_attn': (10, 18, decoder_seq_len, 1024 * 3),
            'decoder_style_ffn': (10, 18, decoder_seq_len, 1024 * 3),
            'decoder_style_final': (10, decoder_seq_len, 1024 * 3),
            'decoder_norm_factor_buf': (decoder_seq_len,),
            'decoder_q_buf': (decoder_seq_len * 8, 256),
            'decoder_logits_buf': (decoder_seq_len * 8, encoder_seq_len + decoder_seq_len),
            'decoder_attn_buf': (decoder_seq_len * 8, encoder_seq_len + decoder_seq_len),
            'decoder_hidden': (decoder_seq_len, 4096),
            'decode_split_k_buf': (2, decoder_seq_len, 1024),
            'x_normed_buf': (decoder_seq_len, 1024),
            'gate_buf': (decoder_seq_len, 1024),
        }
    
    def preprocess_checkpoint(self, checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """预处理 Pi0.5 checkpoint"""
        processed = {}
        num_steps = self.config.num_diffusion_steps
        
        for k, v in checkpoint.items():
            if k != "embedding_weight":
                if isinstance(v, np.ndarray):
                    v = torch.from_numpy(v)
                processed[k] = v
        
        # 缩放输出投影层
        if 'decoder_action_out_proj_w' in processed:
            processed['decoder_action_out_proj_w'] *= (-1.0 / num_steps)
        if 'decoder_action_out_proj_b' in processed:
            processed['decoder_action_out_proj_b'] *= (-1.0 / num_steps)
        
        # 处理 language embeddings
        if 'language_embeds' in checkpoint:
            if isinstance(checkpoint['language_embeds'], np.ndarray):
                processed['language_embeds'] = torch.from_numpy(checkpoint['language_embeds'])
            else:
                processed['language_embeds'] = checkpoint['language_embeds']
        
        return processed
