"""Episodic sampler for few-shot learning.

CUB class IDs are 1-indexed. To get the 312-dim attribute vectors for an episode:
    episode_attrs = sampler.class_attrs[episode.class_ids - 1]  # (n_way, 312)
Typical zero-shot use: prototypes = F.normalize(g(episode_attrs), dim=-1)  # (n_way, 1024)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from jaxtyping import Float, Int
from PIL import Image
from torch import Tensor
from torchvision import transforms


class Dataset(Enum):
    MINI_IMAGENET = "miniimagenet"
    OMNIGLOT = "omniglot"
    CUB = "cub"


class Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class ImageMode(Enum):
    RGB = "RGB"
    GRAYSCALE = "L"


@dataclass(frozen=True)
class ImageDatasetConfig:
    root: Path
    transform: transforms.Compose
    image_mode: ImageMode


_IMAGE_DATASET_CONFIGS: dict[Dataset, ImageDatasetConfig] = {
    Dataset.MINI_IMAGENET: ImageDatasetConfig(
        root=Path("data/miniImageNet"),
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ],
        ),
        image_mode=ImageMode.RGB,
    ),
    Dataset.OMNIGLOT: ImageDatasetConfig(
        root=Path("data/Omniglot"),
        transform=transforms.Compose(
            [
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: 1.0 - x),
                transforms.Normalize(mean=[0.0782], std=[0.2685]),
            ],
        ),
        image_mode=ImageMode.GRAYSCALE,
    ),
}


class ImageEpisode(NamedTuple):
    support_x: Float[Tensor, "n_support channels height width"]
    support_y: Int[Tensor, " n_support"]
    query_x: Float[Tensor, "n_query channels height width"]
    query_y: Int[Tensor, " n_query"]


class CubEpisode(NamedTuple):
    support_x: Float[Tensor, "n_support feature_dim"]
    support_y: Int[Tensor, " n_support"]
    query_x: Float[Tensor, "n_query feature_dim"]
    query_y: Int[Tensor, " n_query"]
    class_ids: Int[Tensor, " n_way"]


def build_index(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        imgs = sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.png"))
        if imgs:
            index[class_dir.name] = imgs
    return index


def sample_episode(
    index: dict[str, list[Path]],
    transform: transforms.Compose,
    n_way: int,
    n_shot: int,
    n_query: int,
    *,
    image_mode: ImageMode = ImageMode.RGB,
) -> ImageEpisode:
    classes = random.sample(list(index.keys()), n_way)

    support_imgs: list[Tensor] = []
    support_labels: list[int] = []
    query_imgs: list[Tensor] = []
    query_labels: list[int] = []

    for label, class_name in enumerate(classes):
        paths = random.sample(index[class_name], n_shot + n_query)
        for path in paths[:n_shot]:
            support_imgs.append(transform(Image.open(path).convert(image_mode.value)))
            support_labels.append(label)
        for path in paths[n_shot:]:
            query_imgs.append(transform(Image.open(path).convert(image_mode.value)))
            query_labels.append(label)

    support_x = torch.stack(support_imgs)
    support_y = torch.tensor(support_labels)
    query_x = torch.stack(query_imgs)
    query_y = torch.tensor(query_labels)

    assert isinstance(support_x, Float[Tensor, " n_support channels height width"])  # type: ignore
    assert isinstance(support_y, Int[Tensor, "n_support"])  # type: ignore
    assert isinstance(query_x, Float[Tensor, "n_query channels height width"])  # type: ignore
    assert isinstance(query_y, Int[Tensor, " n_query"])  # type: ignore

    return ImageEpisode(support_x, support_y, query_x, query_y)


def build_cub_index(labels: np.ndarray) -> dict[int, list[int]]:
    index: dict[int, list[int]] = {}
    for row, label in enumerate(labels):
        index.setdefault(int(label), []).append(row)
    return index


def sample_cub_episode(
    features: np.ndarray,
    index: dict[int, list[int]],
    n_way: int,
    n_shot: int,
    n_query: int,
) -> CubEpisode:
    class_ids_list = random.sample(list(index.keys()), n_way)

    support_feats: list[np.ndarray] = []
    support_labels: list[int] = []
    query_feats: list[np.ndarray] = []
    query_labels: list[int] = []

    for episode_label, class_id in enumerate(class_ids_list):
        rows = random.sample(index[class_id], n_shot + n_query)
        support_feats.append(features[rows[:n_shot]])
        support_labels.extend([episode_label] * n_shot)
        query_feats.append(features[rows[n_shot:]])
        query_labels.extend([episode_label] * n_query)

    support_x = (
        torch.tensor(np.concatenate(support_feats), dtype=torch.float32)
        if n_shot > 0
        else torch.zeros(0, features.shape[1])
    )
    support_y = torch.tensor(support_labels, dtype=torch.int64)
    query_x = torch.tensor(np.concatenate(query_feats), dtype=torch.float32)
    query_y = torch.tensor(query_labels, dtype=torch.int64)
    class_ids = torch.tensor(class_ids_list, dtype=torch.int64)

    assert isinstance(support_x, Float[Tensor, "n_support feature_dim"])  # type: ignore
    assert isinstance(support_y, Int[Tensor, " n_support"])  # type: ignore
    assert isinstance(query_x, Float[Tensor, "n_query feature_dim"])  # type: ignore
    assert isinstance(query_y, Int[Tensor, " n_query"])  # type: ignore
    assert isinstance(class_ids, Int[Tensor, " n_way"])  # type: ignore

    return CubEpisode(support_x, support_y, query_x, query_y, class_ids)


@dataclass(frozen=True)
class ImageEpisodeSampler:
    index: dict[str, list[Path]]
    config: ImageDatasetConfig

    def __call__(self, n_way: int, n_shot: int, n_query: int) -> ImageEpisode:
        return sample_episode(
            self.index,
            self.config.transform,
            n_way,
            n_shot,
            n_query,
            image_mode=self.config.image_mode,
        )


@dataclass(frozen=True)
class CubEpisodeSampler:
    features: np.ndarray
    index: dict[int, list[int]]
    class_attrs: Float[Tensor, " n_classes attr_dim"]

    def __call__(self, n_way: int, n_shot: int, n_query: int) -> CubEpisode:
        return sample_cub_episode(self.features, self.index, n_way, n_shot, n_query)


def get_sampler(dataset: Dataset, split: Split) -> ImageEpisodeSampler | CubEpisodeSampler:
    if dataset == Dataset.CUB:
        data = np.load(Path("data/CUB_200_2011/features") / f"{split.value}_features.npz")
        return CubEpisodeSampler(
            features=data["features"],
            index=build_cub_index(data["labels"]),
            class_attrs=torch.tensor(data["class_attrs"], dtype=torch.float32),
        )

    config = _IMAGE_DATASET_CONFIGS[dataset]
    index = build_index(config.root / split.value)
    return ImageEpisodeSampler(index=index, config=config)
