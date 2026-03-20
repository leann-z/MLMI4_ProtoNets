"""Episodic training for ProtoNets (Snell et al. 2017).

Usage:
    python train.py --experiment miniimagenet_euclidean_20way_5shot
    python train.py --experiment cub_zeroshot
    python train.py --list
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from tqdm import trange

from data import CubEpisodeSampler, Dataset, ImageEpisodeSampler, Split, get_sampler
from models.encoder import ProtoNetEncoder
from models.protonet import DistanceMetric, ProtoNet


@dataclass(frozen=True)
class TrainConfig:
    dataset: Dataset
    distance: DistanceMetric
    train_way: int
    train_shot: int
    n_query: int
    lr: float
    lr_step_every: int
    max_episodes: int
    val_every: int
    patience: int
    weight_decay: float = 0.0


def _experiment_name(dataset: Dataset, distance: DistanceMetric, way: int, shot: int) -> str:
    if dataset == Dataset.CUB:
        return "cub_zeroshot"
    return f"{dataset.value}_{distance.value}_{way}way_{shot}shot"


def _image_config(dataset: Dataset, distance: DistanceMetric, way: int, shot: int) -> TrainConfig:
    n_query = 5 if dataset == Dataset.OMNIGLOT else 15
    lr_step = 2000  # paper: "same learning rate schedule" for both datasets
    return TrainConfig(
        dataset=dataset,
        distance=distance,
        train_way=way,
        train_shot=shot,
        n_query=n_query,
        lr=1e-3,
        lr_step_every=lr_step,
        max_episodes=20_000,
        val_every=500,
        patience=20,
    )


def _build_experiments() -> dict[str, TrainConfig]:
    experiments: dict[str, TrainConfig] = {}

    for shot in [1, 5]:
        cfg = _image_config(Dataset.OMNIGLOT, DistanceMetric.EUCLIDEAN, 60, shot)
        experiments[_experiment_name(cfg.dataset, cfg.distance, cfg.train_way, cfg.train_shot)] = cfg

    for distance in [DistanceMetric.EUCLIDEAN, DistanceMetric.COSINE]:
        ways = [5, 10, 15, 20, 25, 30] if distance == DistanceMetric.EUCLIDEAN else [5, 20]
        for way in ways:
            for shot in [1, 5]:
                cfg = _image_config(Dataset.MINI_IMAGENET, distance, way, shot)
                experiments[_experiment_name(cfg.dataset, cfg.distance, cfg.train_way, cfg.train_shot)] = cfg

    experiments["cub_zeroshot"] = TrainConfig(
        dataset=Dataset.CUB,
        distance=DistanceMetric.EUCLIDEAN,
        train_way=50,
        train_shot=0,
        n_query=10,
        lr=1e-4,
        lr_step_every=999_999,
        max_episodes=20_000,
        val_every=200,
        patience=10,
        weight_decay=1e-5,
    )

    return experiments


EXPERIMENTS = _build_experiments()


def train_image(config: TrainConfig, output_dir: Path, device: torch.device) -> None:
    in_channels = 1 if config.dataset == Dataset.OMNIGLOT else 3
    encoder = ProtoNetEncoder(in_channels=in_channels)
    model = ProtoNet(encoder, distance=config.distance).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = StepLR(optimizer, step_size=config.lr_step_every, gamma=0.5)

    train_sampler = get_sampler(config.dataset, Split.TRAIN)
    val_sampler = get_sampler(config.dataset, Split.VAL)
    assert isinstance(train_sampler, ImageEpisodeSampler)
    assert isinstance(val_sampler, ImageEpisodeSampler)

    best_val_acc = 0.0
    patience_counter = 0

    for ep_idx in trange(config.max_episodes, desc="Training"):
        model.train()
        ep = train_sampler(config.train_way, config.train_shot, config.n_query)
        logits = model(ep.support_x.to(device), ep.support_y.to(device), ep.query_x.to(device))
        loss = F.cross_entropy(logits, ep.query_y.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (ep_idx + 1) % config.val_every == 0:
            model.eval()
            with torch.no_grad():
                total = 0.0
                for _ in range(100):
                    ep = val_sampler(5, config.train_shot, config.n_query)
                    logits = model(ep.support_x.to(device), ep.support_y.to(device), ep.query_x.to(device))
                    total += (logits.argmax(1) == ep.query_y.to(device)).float().mean().item()
                val_acc = total / 100
            print(f"\n  Episode {ep_idx + 1}: val_acc={val_acc:.4f} (best={best_val_acc:.4f})")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                output_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "train_config": {
                            "in_channels": in_channels,
                            "hidden_channels": 64,
                            "distance": config.distance.value,
                        },
                        "model_state_dict": model.state_dict(),
                    },
                    output_dir / "main.pt",
                )
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    print(f"  Early stopping at episode {ep_idx + 1}")
                    break

    print(f"Best val accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint: {output_dir / 'main.pt'}")


def train_cub(config: TrainConfig, output_dir: Path, device: torch.device) -> None:
    train_sampler = get_sampler(Dataset.CUB, Split.TRAIN)
    val_sampler = get_sampler(Dataset.CUB, Split.VAL)
    assert isinstance(train_sampler, CubEpisodeSampler)
    assert isinstance(val_sampler, CubEpisodeSampler)

    train_attrs = train_sampler.class_attrs.to(device)
    feature_dim = train_sampler.features.shape[1]
    attr_dim = int(train_attrs.shape[1])
    embed_dim = 1024

    image_head = torch.nn.Linear(feature_dim, embed_dim).to(device)
    attr_head = torch.nn.Linear(attr_dim, embed_dim).to(device)

    optimizer = torch.optim.Adam(
        list(image_head.parameters()) + list(attr_head.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    best_val_acc = 0.0
    patience_counter = 0

    for ep_idx in trange(config.max_episodes, desc="Training CUB"):
        image_head.train()
        attr_head.train()

        ep = train_sampler(config.train_way, 0, config.n_query)
        episode_attrs = train_attrs[ep.class_ids - 1]

        query_emb = image_head(ep.query_x.to(device))
        prototypes = F.normalize(attr_head(episode_attrs), dim=1)
        dists = DistanceMetric.EUCLIDEAN(query_emb, prototypes)
        loss = F.cross_entropy(-dists, ep.query_y.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (ep_idx + 1) % config.val_every == 0:
            image_head.eval()
            attr_head.eval()
            with torch.no_grad():
                all_attrs = val_sampler.class_attrs.to(device)
                total = 0.0
                for _ in range(100):
                    ep = val_sampler(50, 0, 10)
                    episode_attrs = all_attrs[ep.class_ids - 1]
                    query_emb = image_head(ep.query_x.to(device))
                    prototypes = F.normalize(attr_head(episode_attrs), dim=1)
                    dists = DistanceMetric.EUCLIDEAN(query_emb, prototypes)
                    total += ((-dists).argmax(1) == ep.query_y.to(device)).float().mean().item()
                val_acc = total / 100
            print(f"\n  Episode {ep_idx + 1}: val_acc={val_acc:.4f} (best={best_val_acc:.4f})")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                output_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "train_config": {
                            "feature_dim": image_head.in_features,
                            "attr_dim": attr_head.in_features,
                            "embed_dim": image_head.out_features,
                        },
                        "image_head_state_dict": image_head.state_dict(),
                        "attr_head_state_dict": attr_head.state_dict(),
                    },
                    output_dir / "main.pt",
                )
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    print(f"  Early stopping at episode {ep_idx + 1}")
                    break

    print(f"Best val accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint: {output_dir / 'main.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ProtoNet models")
    parser.add_argument("--experiment", type=str, help="Experiment name (use --list to see options)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--list", action="store_true", help="List available experiments")
    args = parser.parse_args()

    if args.list:
        for name in sorted(EXPERIMENTS):
            print(f"  {name}")
        return

    if not args.experiment:
        parser.error("--experiment is required (use --list to see options)")

    if args.experiment not in EXPERIMENTS:
        parser.error(f"Unknown experiment: {args.experiment}. Use --list to see options.")

    config = EXPERIMENTS[args.experiment]
    device = torch.device(args.device)
    output_dir = Path("checkpoints") / args.experiment

    print(f"Training: {args.experiment}")
    print(f"  Output: {output_dir / 'main.pt'}")

    if config.dataset == Dataset.CUB:
        train_cub(config, output_dir, device)
    else:
        train_image(config, output_dir, device)


if __name__ == "__main__":
    main()
