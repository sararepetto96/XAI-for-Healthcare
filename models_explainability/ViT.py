from timm.models.vision_transformer import VisionTransformer, Block
import math

from typing import Type

from timm.models.vision_transformer import _cfg
import torch
import torch.nn as nn
from torch.jit import Final
from timm.layers import use_fused_attn
from timm.models.registry import register_model
from functools import partial


class CustomAttention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        if self.fused_attn :
            dropout_p=attn_drop if self.training else 0
            self.attn_drop = nn.Dropout(p=dropout_p)
        else:
            self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def detach_qk(self, q, k):
        
        return q, k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        
        #Adding an dummy identity function to allow monkey patching for LRP
        q, k = self.detach_qk(q, k)

        if self.fused_attn:

            #x = F.scaled_dot_product_attention(
                #q, k, v,
                #dropout_p=self.attn_drop.p if self.training else 0.,
            #)
            dropout_p=self.attn_drop.p if self.training else 0
            L, S = q.size(-2), k.size(-2)
            scale_factor = 1 / math.sqrt(q.size(-1))
            attn_bias = torch.zeros(L, S, dtype=q.dtype, device=q.device)
            attn_weight = q @ k.transpose(-2, -1) * scale_factor
            attn_weight += attn_bias
            attn_weight = torch.softmax(attn_weight, dim=-1)
            attn_weight = self.attn_drop(attn_weight)
            x = attn_weight @ v
   
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


from timm.models import create_model

@register_model

def custom_vit_base_patch16_224(pretrained=False, **kwargs):
    # Eliminate parameters that do not serve your custom model
    kwargs.pop('pretrained_cfg', None)
    kwargs.pop('pretrained_cfg_overlay', None)
    kwargs.pop('cache_dir', None)
    # Upload the basic ViT model with weights (without modifying it)
    model = create_model('vit_base_patch16_224', pretrained=pretrained, **kwargs)

    # Replace all attention blocks
    for block in model.blocks:
        old_attn = block.attn
        block.attn = CustomAttention(
            dim=old_attn.qkv.in_features,
            num_heads=old_attn.num_heads,
            qkv_bias=hasattr(old_attn.qkv, 'bias') and old_attn.qkv.bias is not None,
            proj_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )

    model.default_cfg = _cfg()
    return model

@register_model

def custom_vit_base_patch16_224_mae(pretrained=False, **kwargs):
    # Eliminate parameters that do not serve your custom model
    kwargs.pop('pretrained_cfg', None)
    kwargs.pop('pretrained_cfg_overlay', None)
    kwargs.pop('cache_dir', None)
    # Upload the basic MAE model with weights (without modifying it)
    model = create_model('vit_base_patch16_224.mae', pretrained=pretrained, **kwargs)

    # Replace all attention blocks
    for block in model.blocks:
        old_attn = block.attn
        block.attn = CustomAttention(
            dim=old_attn.qkv.in_features,
            num_heads=old_attn.num_heads,
            qkv_bias=hasattr(old_attn.qkv, 'bias') and old_attn.qkv.bias is not None,
            proj_bias=True,  
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )

    model.default_cfg = _cfg()
    return model
@register_model

def custom_deit_base_patch16_224(pretrained=False, **kwargs):
    # Eliminate parameters that do not serve your custom model
    kwargs.pop('pretrained_cfg', None)
    kwargs.pop('pretrained_cfg_overlay', None)
    kwargs.pop('cache_dir', None)
    # Upload the deit model with weights (without modifying it)
    model = create_model('deit_base_patch16_224', pretrained=pretrained, **kwargs)

    # Replace all attention blocks
    for block in model.blocks:
        old_attn = block.attn
        block.attn = CustomAttention(
            dim=old_attn.qkv.in_features,
            num_heads=old_attn.num_heads,
            qkv_bias=hasattr(old_attn.qkv, 'bias') and old_attn.qkv.bias is not None,
            proj_bias=True,  
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )

    model.default_cfg = _cfg()
    return model

@register_model

def custom_pit_b_224(pretrained=False, **kwargs):
    # Eliminate parameters that do not serve your custom model
    kwargs.pop('pretrained_cfg', None)
    kwargs.pop('pretrained_cfg_overlay', None)
    kwargs.pop('cache_dir', None)
    # Upload the deit model with weights (without modifying it)
    model = create_model('pit_b_224', pretrained=pretrained, **kwargs)

    # Replace all attention blocks
    for transformer in model.transformers:
        for block in transformer.blocks:
            old_attn = block.attn
            block.attn = CustomAttention(
                dim=old_attn.qkv.in_features,
                num_heads=old_attn.num_heads,
                qkv_bias=hasattr(old_attn.qkv, 'bias') and old_attn.qkv.bias is not None,
                proj_bias=True,  
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
            )

    model.default_cfg = _cfg()
    return model

