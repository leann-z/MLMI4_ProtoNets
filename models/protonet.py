"""ProtoNet: prototype-based few-shot classifier (Snell et al. 2017).

Transductive soft k-means refinement from Bateni et al. (2022).
"""

from collections.abc import Callable
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
    """||a - b||^2"""
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


def _l1_dist(
    a: Float[Tensor, "n dim"],
    b: Float[Tensor, "m dim"],
) -> Float[Tensor, "n m"]:
    return (a.unsqueeze(1) - b.unsqueeze(0)).abs().sum(dim=2)


def _linf_dist(
    a: Float[Tensor, "n dim"],
    b: Float[Tensor, "m dim"],
) -> Float[Tensor, "n m"]:
    return (a.unsqueeze(1) - b.unsqueeze(0)).abs().amax(dim=2)


class DistanceMetric(Enum):
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
    L1 = "l1"
    LINF = "linf"

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
            case DistanceMetric.L1:
                return _l1_dist(a, b)
            case DistanceMetric.LINF:
                return _linf_dist(a, b)


def _huber_weights(
    scaled_norms: Float[Tensor, " n"],
    delta: float = 1.345,
) -> Float[Tensor, " n"]:
    return torch.where(
        scaled_norms <= delta,
        torch.ones_like(scaled_norms),
        delta / scaled_norms.clamp(min=1e-8),
    )


def _tukey_weights(
    scaled_norms: Float[Tensor, " n"],
    c: float = 4.685,
) -> Float[Tensor, " n"]:
    u = scaled_norms / c
    return torch.where(
        scaled_norms <= c,
        (1 - u**2) ** 2,
        torch.zeros_like(scaled_norms),
    )


def _irls_prototype(
    embeddings: Float[Tensor, "n dim"],
    weight_fn: Callable[[Float[Tensor, "n dim"]], Float[Tensor, "n dim"]],
    n_iter: int = 5,
) -> Float[Tensor, " dim"]:
    """Coordinate-wise M-estimator of location via IRLS."""
    if embeddings.shape[0] <= 1:
        return embeddings.mean(dim=0)
    estimate = embeddings.median(dim=0).values
    for _ in range(n_iter):
        residuals = embeddings - estimate.unsqueeze(0)
        med = residuals.median(dim=0).values
        scale = 1.4826 * (residuals - med.unsqueeze(0)).abs().median(dim=0).values
        scale = scale.clamp(min=1e-8)
        scaled = residuals.abs() / scale.unsqueeze(0)
        weights = weight_fn(scaled)
        weight_sum = weights.sum(dim=0).clamp(min=1e-8)
        estimate = (weights * embeddings).sum(dim=0) / weight_sum
    return estimate


class PrototypeAggregation(Enum):
    MEAN = "mean"
    MEDIAN = "median"
    HUBER = "huber"
    TUKEY = "tukey"

    def __call__(
        self,
        embeddings: Float[Tensor, "n dim"],
    ) -> Float[Tensor, " dim"]:
        match self:
            case PrototypeAggregation.MEAN:
                return embeddings.mean(dim=0)
            case PrototypeAggregation.MEDIAN:
                return embeddings.median(dim=0).values
            case PrototypeAggregation.HUBER:
                return _irls_prototype(embeddings, _huber_weights)
            case PrototypeAggregation.TUKEY:
                return _irls_prototype(embeddings, _tukey_weights)


class DistanceScaling(Enum):
    NONE = "none"
    SQRT_DIM = "sqrt_dim"
    DIM = "dim"


class ProtoNet(nn.Module):
    def __init__(
        self,
        encoder: ProtoNetEncoder,
        distance: DistanceMetric = DistanceMetric.EUCLIDEAN,
        aggregation: PrototypeAggregation = PrototypeAggregation.MEAN,
        distance_scaling: DistanceScaling = DistanceScaling.NONE,
        transductive: bool = False,
        n_refinement_steps: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.distance = distance
        self.aggregation = aggregation
        self.distance_scaling = distance_scaling
        self.transductive = transductive
        self.n_refinement_steps = n_refinement_steps

    def _scaled_dist(
        self,
        a: Float[Tensor, "n dim"],
        b: Float[Tensor, "m dim"],
    ) -> Float[Tensor, "n m"]:
        dists = self.distance(a, b)
        dim = a.shape[1]
        match self.distance_scaling:
            case DistanceScaling.NONE:
                return dists
            case DistanceScaling.SQRT_DIM:
                return dists / dim**0.5
            case DistanceScaling.DIM:
                return dists / dim

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
            prototypes[class_idx] = self.aggregation(support_emb[mask])
        return prototypes

    def _refine_prototypes(
        self,
        support_emb: Float[Tensor, "n_support dim"],
        support_labels: Int[Tensor, " n_support"],
        query_emb: Float[Tensor, "n_query dim"],
        prototypes: Float[Tensor, "n_way dim"],
    ) -> Float[Tensor, "n_way dim"]:
        """Soft k-means transductive refinement

        Alternates between:
          E-step: soft-label query examples via softmax over distances to
                  current prototypes
          M-step: recompute prototypes as weighted mean of support (hard
                  labels) and query (soft labels) embeddings

        Converges when hard assignments stop changing, or after
        n_refinement_steps iterations
        """
        n_way = prototypes.shape[0]
        n_support = support_emb.shape[0]

        # hard responsibility matrix for support: w_jk = 1 iff label_j == k
        support_weights = torch.zeros(
            n_support,
            n_way,
            device=support_emb.device,
            dtype=support_emb.dtype,
        )
        support_weights.scatter_(1, support_labels.unsqueeze(1), 1.0)

        all_emb: Float[Tensor, "n_all dim"] = torch.cat([support_emb, query_emb], dim=0)

        prev_hard_assignments: Int[Tensor, " n_query"] | None = None

        for _ in range(self.n_refinement_steps):
            # E-step: soft assignments for query [n_query, n_way]
            query_weights = F.softmax(-self._scaled_dist(query_emb, prototypes), dim=1)

            # early stopping: check whether hard assignments have changed
            hard_assignments = query_weights.argmax(dim=1)
            if prev_hard_assignments is not None and torch.equal(hard_assignments, prev_hard_assignments):
                break
            prev_hard_assignments = hard_assignments

            # M-step: weighted mean over support + query [n_way, dim]
            all_weights: Float[Tensor, "n_all n_way"] = torch.cat(
                [support_weights, query_weights],
                dim=0,
            )
            weight_sum = all_weights.sum(dim=0).clamp(min=1e-8)  # [n_way]
            prototypes = (all_weights.T @ all_emb) / weight_sum.unsqueeze(1)

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

        # transductive refinement is eval-only
        if self.transductive and not self.training:
            prototypes = self._refine_prototypes(
                support_emb,
                support_labels,
                query_emb,
                prototypes,
            )

        dists = self._scaled_dist(query_emb, prototypes)
        return -dists
