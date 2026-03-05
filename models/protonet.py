# ProtoNet: prototype-based few-shot classifier (Snell et al. 2017)
# forward() returns logits [N_query, n_way],  pass directly to nn.CrossEntropyLoss
# support_labels must be contiguous integers [0, n_way), remapping needed
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import ProtoNetEncoder


def euclidean_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # pairwise squared Euclidean: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b^T
    # a: [N, D], b: [M, D] -> [N, M]
    a2 = (a ** 2).sum(dim=1, keepdim=True)   # [N, 1]
    b2 = (b ** 2).sum(dim=1, keepdim=True)   # [M, 1]
    ab = a @ b.T                              # [N, M]
    return a2 + b2.T - 2 * ab


def cosine_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # cosine_similarity; included for ablation (Fig 3)
    # a: [N, D], b: [M, D] -> [N, M]
    a_norm = F.normalize(a, dim=1)
    b_norm = F.normalize(b, dim=1)
    return 1 - a_norm @ b_norm.T


DISTANCE_FNS = {
    "euclidean": euclidean_dist,
    "cosine": cosine_dist,
}

class ProtoNet(nn.Module):
    # distance: "euclidean" (default, per paper) or "cosine" (ablation)
    def __init__(
        self,
        encoder: ProtoNetEncoder,
        distance: str = "euclidean",
    ):
        super().__init__()

        if distance not in DISTANCE_FNS:
            raise ValueError(f"distance must be one of {list(DISTANCE_FNS)}, got '{distance}'")

        self.encoder = encoder
        self.distance_fn = DISTANCE_FNS[distance]

    def _compute_prototypes(self, support_emb: torch.Tensor, support_labels: torch.Tensor) -> torch.Tensor:
        # class prototype = mean of support embeddings per class (Eq. 1)
        # support_emb: [N_support, D], support_labels: [N_support] -> [n_way, D]
        n_way = int(support_labels.max().item() + 1)
        D = support_emb.shape[1]
        prototypes = torch.zeros(n_way, D, device=support_emb.device, dtype=support_emb.dtype)
        for k in range(n_way):
            mask = support_labels == k
            prototypes[k] = support_emb[mask].mean(dim=0)
        return prototypes

    def forward(
        self,
        support_imgs: torch.Tensor, # [N_support, C, H, W]
        support_labels: torch.Tensor, # [N_support], contiguous in [0, n_way)
        query_imgs: torch.Tensor, # [N_query, C, H, W]
    ) -> torch.Tensor: # [N_query, n_way] logits
        # single encoder pass over support + query
        all_emb = self.encoder(torch.cat([support_imgs, query_imgs], dim=0))
        support_emb = all_emb[:support_imgs.shape[0]]
        query_emb   = all_emb[support_imgs.shape[0]:]

        prototypes = self._compute_prototypes(support_emb, support_labels)

        dists = self.distance_fn(query_emb, prototypes)   # [N_query, n_way]
        return -dists                                      # negate: higher = closer = more likely
