import numpy as np
import torch
from torch_utils import persistence


#----------------------------------------------------------------------------
# Guidance pipeline main class for guided consistency training

@persistence.persistent_class
class GuidancePipe:
    """
    See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

    Args:
        timesteps (`torch.Tensor`):
            generate embedding vectors at these timesteps
        embedding_dim (`int`, *optional*, defaults to 512):
            dimension of the embeddings to generate
        dtype:
            data type of the generated embeddings

    Returns:
        `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
    
    Notes: 
        in repo: https://github.com/luosiallen/latent-consistency-model/blob/main/LCM_Training_Script/consistency_distillation/train_lcm_distill_sd_wds.py
        & repo: from https://github.com/yandex-research/invertible-cd/blob/main/training/train_icd_sd15_lora.py,
        embedding_dim is set to 512.
    """
    def __init__(self, w_embed_dim):
        super().__init__()
        self.w_embed_dim = w_embed_dim   
        self.dtype = torch.float32              

    def __call__(self, w):
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = self.w_embed_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=self.dtype) * -emb)
        emb = w.to(self.dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.w_embed_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], self.w_embed_dim)
        return emb
