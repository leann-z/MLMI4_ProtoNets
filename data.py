"""Episodic sampler for few-shot learning (Prototypical Networks).

All datasets return one episode as:
    support_x  float32 tensor [N_way*N_shot,  ...]   image (C,H,W) or feature (1024,)
    support_y  int64   tensor [N_way*N_shot]          labels 0..N_way-1
    query_x    float32 tensor [N_way*N_query, ...]   image (C,H,W) or feature (1024,)
    query_y    int64   tensor [N_way*N_query]         labels 0..N_way-1

Usage — miniImageNet / Omniglot:
    sampler = get_sampler("miniimagenet", "train")   # or "omniglot"
    support_x, support_y, query_x, query_y = sampler(n_way=5, n_shot=1, n_query=15)
    # x tensors are images: (N, C, H, W)

Usage — CUB (zero-shot, paper config):
    sampler = get_sampler("cub", "train")   # or "val" / "test"
    support_x, support_y, query_x, query_y, class_ids = sampler(n_way=50, n_shot=0, n_query=10)
    # x tensors are 1,024-dim GoogLeNet features, NOT images
    # class_ids: int64 tensor (n_way,) — original CUB class IDs (1-indexed)
    #            episode label i corresponds to class_ids[i]
    #
    # To get the 312-dim attribute vectors for this episode's classes:
    episode_attrs = sampler.class_attrs[class_ids - 1]   # (n_way, 312)
    #
    # Typical model use:
    #   prototypes = F.normalize(g(episode_attrs), dim=-1)   # (n_way, 1024)
    #   logits     = -torch.cdist(f(query_x), prototypes)**2
"""

import random
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float, Int
from PIL import Image
from torchvision import transforms


# ---------------------------------------------------------------------------
# Per-dataset transforms
# ---------------------------------------------------------------------------

TRANSFORMS = {
    "miniimagenet": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std= [0.229, 0.224, 0.225]),
    ]),
    "omniglot": transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1.0 - x),
        transforms.Normalize(mean=[0.0782], std=[0.2685]),
    ]),
}

# PIL mode to use when opening images
IMAGE_MODE = {
    "miniimagenet": "RGB",
    "omniglot":     "L",
}


# ---------------------------------------------------------------------------
# Dataset index: maps class_name → list of image paths
# ---------------------------------------------------------------------------

def build_index(root: Path) -> dict[str, list[Path]]:
    """Scan root/<class>/*.{jpg,png} and return class → [paths] mapping."""
    index = {}
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        imgs = sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.png"))
        if imgs:
            index[class_dir.name] = imgs
    return index


# ---------------------------------------------------------------------------
# Episode sampler
# ---------------------------------------------------------------------------

def sample_episode(
    index: dict[str, list[Path]],
    transform,
    n_way: int,
    n_shot: int,
    n_query: int,
    image_mode: str = "RGB",
) -> tuple[
    Float[torch.Tensor, "nway_nshot C H W"],
    Int[torch.Tensor, "nway_nshot"],
    Float[torch.Tensor, "nway_nquery C H W"],
    Int[torch.Tensor, "nway_nquery"],
]:
    """Sample one N-way K-shot episode from a class index."""
    classes = random.sample(list(index.keys()), n_way)

    support_imgs, support_labels = [], []
    query_imgs,   query_labels   = [], []

    for label, cls in enumerate(classes):
        paths = random.sample(index[cls], n_shot + n_query)
        for p in paths[:n_shot]:
            support_imgs.append(transform(Image.open(p).convert(image_mode)))
            support_labels.append(label)
        for p in paths[n_shot:]:
            query_imgs.append(transform(Image.open(p).convert(image_mode)))
            query_labels.append(label)

    support_x = torch.stack(support_imgs)           # [N_way*N_shot,  C, H, W]
    support_y = torch.tensor(support_labels)        # [N_way*N_shot]
    query_x   = torch.stack(query_imgs)             # [N_way*N_query, C, H, W]
    query_y   = torch.tensor(query_labels)          # [N_way*N_query]

    return support_x, support_y, query_x, query_y


# ---------------------------------------------------------------------------
# Convenience: build sampler for a given dataset + split
# ---------------------------------------------------------------------------

DATASET_ROOTS = {
    "miniimagenet": Path("data/miniImageNet"),
    "omniglot":     Path("data/Omniglot"),
}


# ---------------------------------------------------------------------------
# CUB zero-shot episode sampler
# ---------------------------------------------------------------------------

def build_cub_index(features, labels):
    """Return {class_id: array of row indices} from the feature matrix."""
    index = {}
    for i, lbl in enumerate(labels):
        index.setdefault(int(lbl), []).append(i)
    return index


def sample_cub_episode(
    features,   # (N, 1024) float32 numpy array
    labels,     # (N,)      int32  numpy array
    index,      # {class_id: [row indices]}
    n_way: int,
    n_shot: int,
    n_query: int,
) -> tuple[
    Float[torch.Tensor, "nway_nshot 1024"],
    Int[torch.Tensor, "nway_nshot"],
    Float[torch.Tensor, "nway_nquery 1024"],
    Int[torch.Tensor, "nway_nquery"],
    Int[torch.Tensor, "nway"],
]:
    """Sample one N-way episode for CUB; returns feature vectors instead of images.

    Also returns class_ids (shape: n_way) — the original CUB class IDs (1-indexed)
    for the selected classes, in episode-label order (episode label 0 → class_ids[0]).
    The model uses these to look up sampler.class_attrs[class_ids - 1] for prototypes.
    """
    class_ids = random.sample(list(index.keys()), n_way)

    support_feats, support_labels = [], []
    query_feats,   query_labels   = [], []

    for episode_label, cid in enumerate(class_ids):
        rows = random.sample(index[cid], n_shot + n_query)
        support_feats.append(features[rows[:n_shot]])
        support_labels.extend([episode_label] * n_shot)
        query_feats.append(features[rows[n_shot:]])
        query_labels.extend([episode_label] * n_query)

    support_x = torch.tensor(np.concatenate(support_feats), dtype=torch.float32) if n_shot > 0 else torch.zeros(0, features.shape[1])
    support_y = torch.tensor(support_labels, dtype=torch.int64)
    query_x   = torch.tensor(np.concatenate(query_feats),   dtype=torch.float32)
    query_y   = torch.tensor(query_labels,                  dtype=torch.int64)
    class_ids = torch.tensor(class_ids,                     dtype=torch.int64)  # (n_way,)

    return support_x, support_y, query_x, query_y, class_ids


# ---------------------------------------------------------------------------
# Convenience: build sampler for a given dataset + split
# ---------------------------------------------------------------------------

def get_sampler(dataset: str, split: str):
    """Return a callable that produces one episode when called.

    Args:
        dataset: 'miniimagenet', 'omniglot', or 'cub'
        split:   'train', 'val', or 'test'

    Returns:
        For image datasets:
            sampler(n_way, n_shot, n_query) → (support_x, support_y, query_x, query_y)
        For 'cub' (zero-shot):
            sampler(n_way, n_query) → (support_attrs, query_x, query_y)
    """
    if dataset == "cub":
        data     = np.load(Path("data/CUB_200_2011/features") / f"{split}_features.npz")
        features = data["features"]   # (N, 1024)
        labels   = data["labels"]     # (N,)
        index    = build_cub_index(features, labels)

        def sampler(n_way: int, n_shot: int, n_query: int):
            return sample_cub_episode(features, labels, index, n_way, n_shot, n_query)

        sampler.index       = index
        # class_attrs[i] is the 312-dim attribute vector for CUB class id (i+1).
        # The model uses this to build prototypes via g: R^312 → R^1024.
        sampler.class_attrs = torch.tensor(data["class_attrs"], dtype=torch.float32)
        return sampler

    root       = DATASET_ROOTS[dataset] / split
    transform  = TRANSFORMS[dataset]
    index      = build_index(root)
    image_mode = IMAGE_MODE[dataset]

    def sampler(n_way: int, n_shot: int, n_query: int):
        return sample_episode(index, transform, n_way, n_shot, n_query, image_mode)

    sampler.index = index   # expose for inspection
    return sampler