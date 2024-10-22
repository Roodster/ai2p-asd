import torch.nn as nn
import torch as th
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.similarity = nn.CosineSimilarity(dim=-1, eps=1e-7)
    
    def forward(self, outputs):

        a, b = outputs
        
        B, C, T = a.shape
        
        a_flat = a.view(B, -1)
        b_flat = a.view(B, -1) 
        features = th.cat([a_flat, b_flat], dim=0)
                
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(features[:,None,:], features[None,:,:], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = th.eye(cos_sim.shape[0], dtype=th.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)

        # InfoNCE loss
        cos_sim = cos_sim / self.temperature
        nll = -cos_sim[pos_mask] + th.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        return nll