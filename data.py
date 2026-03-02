"""Episodic sampler for few-shot learning (Prototypical Networks).

Returns one episode as:
    support_x  float32 tensor [N_way*N_shot,  C, H, W]
    support_y  int64   tensor [N_way*N_shot]   labels 0..N_way-1
    query_x    float32 tensor [N_way*N_query, C, H, W]
    query_y    int64   tensor [N_way*N_query]  labels 0..N_way-1

Usage:
    from data import get_sampler
    sampler = get_sampler("miniimagenet"/"omniglot", "train"/"val"/"test")
    support_x, support_y, query_x, query_y = sampler(n_way=n_way, n_shot=n_shot, n_query=n_query)
"""

import random
from pathlib import Path

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


def get_sampler(dataset: str, split: str):
    """Return a callable that produces one episode when called.

    Args:
        dataset: 'miniimagenet' or 'omniglot'
        split:   'train', 'val', or 'test'

    Returns:
        sampler(n_way, n_shot, n_query) → (support_x, support_y, query_x, query_y)
    """
    root      = DATASET_ROOTS[dataset] / split
    transform = TRANSFORMS[dataset]
    index      = build_index(root)
    image_mode = IMAGE_MODE[dataset]

    def sampler(n_way: int, n_shot: int, n_query: int):
        return sample_episode(index, transform, n_way, n_shot, n_query, image_mode)

    sampler.index = index   # expose for inspection
    return sampler