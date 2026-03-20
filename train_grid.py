from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from tqdm import trange

from data import Dataset, Split, get_sampler
from models.encoder import ProtoNetEncoder
from models.protonet import DistanceMetric, DistanceScaling, ProtoNet, PrototypeAggregation

TRAIN_WAY = 20
TRAIN_SHOT = 5
N_QUERY = 15
LR = 1e-3
LR_STEP_EVERY = 2000
MAX_EPISODES = 5000
VAL_EVERY = 500
PATIENCE = 20


def train_single(
    distance: DistanceMetric,
    aggregation: PrototypeAggregation,
    scaling: DistanceScaling,
    output_dir: Path,
    device: torch.device,
) -> None:
    encoder = ProtoNetEncoder(in_channels=3)
    model = ProtoNet(
        encoder,
        distance=distance,
        aggregation=aggregation,
        distance_scaling=scaling,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=LR_STEP_EVERY, gamma=0.5)

    train_sampler = get_sampler(Dataset.MINI_IMAGENET, Split.TRAIN)
    val_sampler = get_sampler(Dataset.MINI_IMAGENET, Split.VAL)

    best_val_acc = 0.0
    patience_counter = 0

    tag = f"{distance.value}+{scaling.value}"
    for ep_idx in trange(MAX_EPISODES, desc=tag):
        model.train()
        ep = train_sampler(TRAIN_WAY, TRAIN_SHOT, N_QUERY)
        logits = model(ep.support_x.to(device), ep.support_y.to(device), ep.query_x.to(device))
        loss = F.cross_entropy(logits, ep.query_y.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (ep_idx + 1) % VAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                total = 0.0
                for _ in range(100):
                    ep = val_sampler(5, TRAIN_SHOT, N_QUERY)
                    logits = model(ep.support_x.to(device), ep.support_y.to(device), ep.query_x.to(device))
                    total += (logits.argmax(1) == ep.query_y.to(device)).float().mean().item()
                val_acc = total / 100
            print(f"\n  {tag} ep {ep_idx + 1}: val_acc={val_acc:.4f} (best={best_val_acc:.4f})")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                output_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "train_config": {
                            "in_channels": 3,
                            "hidden_channels": 64,
                            "distance": distance.value,
                            "aggregation": aggregation.value,
                            "distance_scaling": scaling.value,
                        },
                        "model_state_dict": model.state_dict(),
                    },
                    output_dir / "main.pt",
                )
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"  Early stopping at episode {ep_idx + 1}")
                    break

    print(f"{tag}: best val accuracy = {best_val_acc:.4f}")
    print(f"Checkpoint: {output_dir / 'main.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train distance/scaling ablation grid")
    parser.add_argument("--distance", type=str, help="Distance metric")
    parser.add_argument("--scaling", type=str, default="none", help="Scaling: none, sqrt_dim, dim")
    parser.add_argument("--aggregation", type=str, default="mean", help="Aggregation method")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--list", action="store_true", help="List all scaling ablation experiments")
    args = parser.parse_args()

    if args.list:
        for d in DistanceMetric:
            for s in DistanceScaling:
                print(f"  {d.value} {s.value}")
        return

    if not args.distance:
        parser.error("--distance is required")

    distance = DistanceMetric(args.distance)
    scaling = DistanceScaling(args.scaling)
    aggregation = PrototypeAggregation(args.aggregation)
    device = torch.device(args.device)
    output_dir = Path("checkpoints") / "grid" / f"{distance.value}_{scaling.value}"

    print(f"Training: {distance.value} ({scaling.value})")
    print(f"  Output: {output_dir / 'main.pt'}")
    train_single(distance, aggregation, scaling, output_dir, device)


if __name__ == "__main__":
    main()
