"""ProtoNet: prototype-based few-shot classifier (Snell et al. 2017)."""

from enum import Enum

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor, nn

from models.encoder import ProtoNetEncoder


def _euclidean_dist(
    a: Float[Tensor, "n dim"],
    b: Float[Tensor, "m dim"],
) -> Float[Tensor, "n m"]:
    """
    The reason for this construction is to avoid materialization of the full [n m] matrix.

    ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b^T"""
    a2 = (a**2).sum(dim=1, keepdim=True)
    b2 = (b**2).sum(dim=1, keepdim=True)
    ab = a @ b.T
    return a2 + b2.T - 2 * ab


def _cosine_dist(
    a: Float[Tensor, "n dim"],
    b: Float[Tensor, "m dim"],
) -> Float[Tensor, "n m"]:
    a_norm = F.normalize(a, dim=1)
    b_norm = F.normalize(b, dim=1)
    return 1 - a_norm @ b_norm.T


class DistanceMetric(Enum):
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"

    def __call__(
        self,
        a: Float[Tensor, "n dim"],
        b: Float[Tensor, "m dim"],
    ) -> Float[Tensor, "n m"]:
        match self:
            case DistanceMetric.EUCLIDEAN:
                return _euclidean_dist(a, b)
            case DistanceMetric.COSINE:
                return _cosine_dist(a, b)


class ProtoNet(nn.Module):
    def __init__(
        self,
        encoder: ProtoNetEncoder,
        distance: DistanceMetric = DistanceMetric.EUCLIDEAN,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.distance = distance

    def _compute_prototypes(
        self,
        support_emb: Float[Tensor, "n_support dim"],
        support_labels: Int[Tensor, " n_support"],
    ) -> Float[Tensor, "n_way dim"]:
        n_way = int(support_labels.max().item() + 1)
        dim = support_emb.shape[1]
        prototypes = torch.zeros(n_way, dim, device=support_emb.device, dtype=support_emb.dtype)
        for class_idx in range(n_way):
            mask = support_labels == class_idx
            prototypes[class_idx] = support_emb[mask].mean(dim=0)
        return prototypes

    def forward(
        self,
        support_imgs: Float[Tensor, "n_support channels height width"],
        support_labels: Int[Tensor, " n_support"],
        query_imgs: Float[Tensor, "n_query channels height width"],
    ) -> Float[Tensor, "n_query n_way"]:
        all_emb = self.encoder(torch.cat([support_imgs, query_imgs], dim=0))
        support_emb = all_emb[: support_imgs.shape[0]]
        query_emb = all_emb[support_imgs.shape[0] :]

        prototypes = self._compute_prototypes(support_emb, support_labels)
        dists = self.distance(query_emb, prototypes)
        # negate: smaller distance = higher logit = higher predicted probability
        return -dists
