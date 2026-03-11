"""ProtoNet: prototype-based few-shot classifier (Snell et al. 2017).

Transductive soft k-means refinement from Bateni et al. (2022).
"""
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
    """||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b^T"""
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
        transductive: bool = False,
        n_refinement_steps: int = 4,
    ) -> None:
        """
        Args:
            encoder:            feature extractor, returns [B, D] embeddings
            distance:           distance metric for prototype assignment
            transductive:       if true, refine prototypes at eval time using
                                soft-labelled query examples
            n_refinement_steps: max soft k-means iterations. terminates early if
                                hard assignments stabilise. Bateni use max=4 on
                                Meta-Dataset, max=10 on mini/tiered-ImageNet.
        """
        super().__init__()
        self.encoder = encoder
        self.distance = distance
        self.transductive = transductive
        self.n_refinement_steps = n_refinement_steps

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
            n_support, n_way, device=support_emb.device, dtype=support_emb.dtype
        )
        support_weights.scatter_(1, support_labels.unsqueeze(1), 1.0)

        all_emb: Float[Tensor, "n_all dim"] = torch.cat([support_emb, query_emb], dim=0)

        prev_hard_assignments: Int[Tensor, " n_query"] | None = None

        for _ in range(self.n_refinement_steps):
            # E-step: soft assignments for query [n_query, n_way]
            query_weights = F.softmax(-self.distance(query_emb, prototypes), dim=1)

            # early stopping: check whether hard assignments have changed
            hard_assignments = query_weights.argmax(dim=1)
            if prev_hard_assignments is not None and torch.equal(hard_assignments, prev_hard_assignments):
                break
            prev_hard_assignments = hard_assignments

            # M-step: weighted mean over support + query [n_way, dim]
            all_weights: Float[Tensor, "n_all n_way"] = torch.cat(
                [support_weights, query_weights], dim=0
            )
            weight_sum = all_weights.sum(dim=0).clamp(min=1e-8)   # [n_way]
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
                support_emb, support_labels, query_emb, prototypes
            )

        dists = self.distance(query_emb, prototypes)
        return -dists