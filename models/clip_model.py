import torch.nn as nn
import torch
import numpy as np
from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder

class CLIPModel(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))
        
    def forward(self, images, text_inputs):
        image_embeds = self.image_encoder(images)
        text_embeds = self.text_encoder(**text_inputs)
        
        # 归一化
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        # 计算相似度
        logit_scale = self.logit_scale.exp()
        logits = torch.matmul(image_embeds, text_embeds.t()) * logit_scale
        return logits
