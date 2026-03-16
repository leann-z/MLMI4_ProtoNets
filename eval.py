"""Episodic evaluation for few-shot and zero-shot benchmarks.

Benchmarks from:
  - Snell et al. 2017 (Prototypical Networks): Tables 1, 2, 3, 5, 6; Figures 2, 4
  - Bateni et al. 2022 (Transductive CNAPS): Tables 2, 4; Figure 6

Usage:
    python eval.py --benchmark omniglot_fewshot --checkpoint checkpoints/omni.pt
    python eval.py --benchmark cub_zeroshot --checkpoint checkpoints/cub.pt
    python eval.py --benchmark refinement_steps --checkpoint checkpoints/mini.pt
    python eval.py --list-benchmarks
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from tqdm import trange

from data import Dataset, Split, get_sampler
from models.encoder import ProtoNetEncoder
from models.protonet import DistanceMetric, ProtoNet


class Benchmark(Enum):
    OMNIGLOT_FEWSHOT = "omniglot_fewshot"
    MINIIMAGENET_FEWSHOT = "miniimagenet_fewshot"
    MINIIMAGENET_DISTANCE_WAY = "miniimagenet_distance_way"
    MINIIMAGENET_WAY_ABLATION = "miniimagenet_way_ablation"
    MINIIMAGENET_TRANSDUCTIVE = "miniimagenet_transductive"
    MINIIMAGENET_10WAY = "miniimagenet_10way"
    REFINEMENT_STEPS = "refinement_steps"
    CUB_ZEROSHOT = "cub_zeroshot"


@dataclass(frozen=True)
class EvalConfig:
    dataset: Dataset
    distance: DistanceMetric
    n_way_test: int
    n_shot_test: int
    n_query: int
    n_episodes: int
    transductive: bool
    train_way: int
    train_shot: int
    n_refinement_steps: int = 4


@dataclass(frozen=True)
class EvalResult:
    benchmark: str
    config: EvalConfig
    model_checkpoint: str
    accuracy_mean: float
    accuracy_std: float
    accuracy_ci95: float
    episode_accuracies: list[float]


# ---------- Benchmark presets (matching paper configurations) ----------

# Snell Table 1: Omniglot 5/20-way × 1/5-shot
_OMNIGLOT_FEWSHOT_CONFIGS = [
    EvalConfig(Dataset.OMNIGLOT, DistanceMetric.EUCLIDEAN, n_way_test=5, n_shot_test=1, n_query=5, n_episodes=1000, transductive=False, train_way=60, train_shot=1),
    EvalConfig(Dataset.OMNIGLOT, DistanceMetric.EUCLIDEAN, n_way_test=5, n_shot_test=5, n_query=5, n_episodes=1000, transductive=False, train_way=60, train_shot=5),
    EvalConfig(Dataset.OMNIGLOT, DistanceMetric.EUCLIDEAN, n_way_test=20, n_shot_test=1, n_query=5, n_episodes=1000, transductive=False, train_way=60, train_shot=1),
    EvalConfig(Dataset.OMNIGLOT, DistanceMetric.EUCLIDEAN, n_way_test=20, n_shot_test=5, n_query=5, n_episodes=1000, transductive=False, train_way=60, train_shot=5),
]

# Snell Table 2: miniImageNet 5-way 1/5-shot
_MINIIMAGENET_FEWSHOT_CONFIGS = [
    EvalConfig(Dataset.MINI_IMAGENET, DistanceMetric.EUCLIDEAN, n_way_test=5, n_shot_test=1, n_query=15, n_episodes=600, transductive=False, train_way=30, train_shot=1),
    EvalConfig(Dataset.MINI_IMAGENET, DistanceMetric.EUCLIDEAN, n_way_test=5, n_shot_test=5, n_query=15, n_episodes=600, transductive=False, train_way=20, train_shot=5),
]

# Snell Figure 2 / Table 5: cosine vs euclidean × 5/20-way training
_MINIIMAGENET_DISTANCE_WAY_CONFIGS = [
    EvalConfig(Dataset.MINI_IMAGENET, dist, n_way_test=5, n_shot_test=shot, n_query=15, n_episodes=600, transductive=False, train_way=way, train_shot=shot)
    for dist in [DistanceMetric.COSINE, DistanceMetric.EUCLIDEAN]
    for way in [5, 20]
    for shot in [1, 5]
]

# Snell Figure 4 / Table 6: training way ablation
_MINIIMAGENET_WAY_ABLATION_CONFIGS = [
    EvalConfig(Dataset.MINI_IMAGENET, DistanceMetric.EUCLIDEAN, n_way_test=5, n_shot_test=shot, n_query=15, n_episodes=600, transductive=False, train_way=way, train_shot=shot)
    for way in [5, 10, 15, 20, 25, 30]
    for shot in [1, 5]
]

# Snell + Bateni: 5-way transductive vs inductive
_MINIIMAGENET_TRANSDUCTIVE_CONFIGS = [
    EvalConfig(Dataset.MINI_IMAGENET, DistanceMetric.EUCLIDEAN, n_way_test=5, n_shot_test=shot, n_query=15, n_episodes=600, transductive=trans, train_way=20, train_shot=shot)
    for shot in [1, 5]
    for trans in [False, True]
]

# Bateni Table 2: miniImageNet 10-way 1/5-shot (inductive + transductive)
_MINIIMAGENET_10WAY_CONFIGS = [
    EvalConfig(Dataset.MINI_IMAGENET, DistanceMetric.EUCLIDEAN, n_way_test=10, n_shot_test=shot, n_query=10, n_episodes=600, transductive=trans, train_way=20, train_shot=shot)
    for shot in [1, 5]
    for trans in [False, True]
]

# Bateni Figure 6 / Table 4: sweep max refinement steps 0..10
_REFINEMENT_STEPS_CONFIGS = [
    EvalConfig(Dataset.MINI_IMAGENET, DistanceMetric.EUCLIDEAN, n_way_test=5, n_shot_test=shot, n_query=15, n_episodes=600, transductive=True, train_way=20, train_shot=shot, n_refinement_steps=steps)
    for shot in [1, 5]
    for steps in range(11)
]

# Snell Table 3: CUB 50-way 0-shot
_CUB_ZEROSHOT_CONFIGS = [
    EvalConfig(Dataset.CUB, DistanceMetric.EUCLIDEAN, n_way_test=50, n_shot_test=0, n_query=10, n_episodes=600, transductive=False, train_way=50, train_shot=0),
]

BENCHMARK_CONFIGS: dict[Benchmark, list[EvalConfig]] = {
    Benchmark.OMNIGLOT_FEWSHOT: _OMNIGLOT_FEWSHOT_CONFIGS,
    Benchmark.MINIIMAGENET_FEWSHOT: _MINIIMAGENET_FEWSHOT_CONFIGS,
    Benchmark.MINIIMAGENET_DISTANCE_WAY: _MINIIMAGENET_DISTANCE_WAY_CONFIGS,
    Benchmark.MINIIMAGENET_WAY_ABLATION: _MINIIMAGENET_WAY_ABLATION_CONFIGS,
    Benchmark.MINIIMAGENET_TRANSDUCTIVE: _MINIIMAGENET_TRANSDUCTIVE_CONFIGS,
    Benchmark.MINIIMAGENET_10WAY: _MINIIMAGENET_10WAY_CONFIGS,
    Benchmark.REFINEMENT_STEPS: _REFINEMENT_STEPS_CONFIGS,
    Benchmark.CUB_ZEROSHOT: _CUB_ZEROSHOT_CONFIGS,
}


# ---------- Model loading ----------

def _build_image_protonet(checkpoint: dict, device: torch.device) -> ProtoNet:
    cfg = checkpoint["train_config"]
    encoder = ProtoNetEncoder(
        in_channels=cfg["in_channels"],
        hidden_channels=cfg.get("hidden_channels", 64),
    )
    distance = DistanceMetric(cfg["distance"])
    model = ProtoNet(encoder, distance=distance, transductive=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


def load_model(
    checkpoint_path: str | Path,
    config: EvalConfig,
    device: torch.device,
) -> ProtoNet:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = _build_image_protonet(checkpoint, device)
    model.transductive = config.transductive
    model.n_refinement_steps = config.n_refinement_steps
    model.distance = config.distance
    model.eval()
    return model


def load_zeroshot_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[torch.nn.Linear, torch.nn.Linear]:
    """Load zero-shot CUB model: image_head and attr_head.

    Paper (Snell 2017, Section 3.3): learned a simple linear mapping on top
    of both 1024-dim image features and 312-dim attribute vectors to produce
    a 1024-dim output space. Prototypes (embedded attributes) are normalized
    to unit length.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint["train_config"]

    image_head = torch.nn.Linear(cfg["feature_dim"], cfg["embed_dim"])
    attr_head = torch.nn.Linear(cfg["attr_dim"], cfg["embed_dim"])

    image_head.load_state_dict(checkpoint["image_head_state_dict"])
    attr_head.load_state_dict(checkpoint["attr_head_state_dict"])

    return image_head.to(device).eval(), attr_head.to(device).eval()


# ---------- Episode accuracy ----------

@torch.no_grad()
def episode_accuracy_fewshot(
    model: ProtoNet,
    support_x: Tensor,
    support_y: Tensor,
    query_x: Tensor,
    query_y: Tensor,
) -> float:
    logits = model(support_x, support_y, query_x)
    preds = logits.argmax(dim=1)
    return (preds == query_y).float().mean().item()


@torch.no_grad()
def episode_accuracy_zeroshot(
    image_head: torch.nn.Module,
    attr_head: torch.nn.Module,
    query_features: Float[Tensor, "n_query feature_dim"],
    query_labels: Tensor,
    class_attrs: Float[Tensor, "n_way attr_dim"],
    distance: DistanceMetric,
) -> float:
    query_emb = image_head(query_features)
    prototypes = F.normalize(attr_head(class_attrs), dim=1)
    dists = distance(query_emb, prototypes)
    preds = (-dists).argmax(dim=1)
    return (preds == query_labels).float().mean().item()


# ---------- Main evaluation loops ----------

def evaluate_fewshot(
    model: ProtoNet,
    config: EvalConfig,
    device: torch.device,
) -> list[float]:
    sampler = get_sampler(config.dataset, Split.TEST)
    accuracies: list[float] = []

    for _ in trange(config.n_episodes, desc=f"{config.n_way_test}-way {config.n_shot_test}-shot"):
        episode = sampler(config.n_way_test, config.n_shot_test, config.n_query)
        acc = episode_accuracy_fewshot(
            model,
            episode.support_x.to(device),
            episode.support_y.to(device),
            episode.query_x.to(device),
            episode.query_y.to(device),
        )
        accuracies.append(acc)

    return accuracies


def evaluate_zeroshot(
    image_head: torch.nn.Module,
    attr_head: torch.nn.Module,
    config: EvalConfig,
    device: torch.device,
) -> list[float]:
    sampler = get_sampler(Dataset.CUB, Split.TEST)
    assert hasattr(sampler, "class_attrs"), "CUB sampler must expose class_attrs"
    all_class_attrs = sampler.class_attrs.to(device)

    accuracies: list[float] = []

    for _ in trange(config.n_episodes, desc="50-way 0-shot"):
        episode = sampler(config.n_way_test, n_shot=0, n_query=config.n_query)
        episode_attrs = all_class_attrs[episode.class_ids - 1]
        acc = episode_accuracy_zeroshot(
            image_head,
            attr_head,
            episode.query_x.to(device),
            episode.query_y.to(device),
            episode_attrs,
            config.distance,
        )
        accuracies.append(acc)

    return accuracies


# ---------- Stats ----------

def compute_stats(accuracies: list[float]) -> tuple[float, float, float]:
    n = len(accuracies)
    mean = sum(accuracies) / n
    std = math.sqrt(sum((a - mean) ** 2 for a in accuracies) / (n - 1))
    ci95 = 1.96 * std / math.sqrt(n)
    return mean, std, ci95


# ---------- Serialization ----------

def _serialize_config(config: EvalConfig) -> dict:
    d = asdict(config)
    d["dataset"] = config.dataset.value
    d["distance"] = config.distance.value
    return d


def _save_result(result: EvalResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "benchmark": result.benchmark,
        "config": _serialize_config(result.config),
        "model_checkpoint": result.model_checkpoint,
        "accuracy_mean": result.accuracy_mean,
        "accuracy_std": result.accuracy_std,
        "accuracy_ci95": result.accuracy_ci95,
        "episode_accuracies": result.episode_accuracies,
    }
    path.write_text(json.dumps(data, indent=2))
    print(f"  Saved: {path}")


# ---------- Run a full benchmark ----------

def _config_tag(benchmark: Benchmark, config: EvalConfig) -> str:
    tag = f"{config.n_way_test}way_{config.n_shot_test}shot"
    if benchmark in {Benchmark.MINIIMAGENET_DISTANCE_WAY, Benchmark.MINIIMAGENET_WAY_ABLATION}:
        tag += f"_train{config.train_way}way_{config.distance.value}"
    if benchmark in {Benchmark.MINIIMAGENET_TRANSDUCTIVE, Benchmark.MINIIMAGENET_10WAY}:
        tag += f"_trans{config.transductive}"
    if benchmark == Benchmark.REFINEMENT_STEPS:
        tag += f"_steps{config.n_refinement_steps}"
    return tag


def run_benchmark(
    benchmark: Benchmark,
    checkpoint_path: str | Path,
    output_dir: Path,
    device: torch.device,
) -> list[EvalResult]:
    configs = BENCHMARK_CONFIGS[benchmark]
    results: list[EvalResult] = []

    for config in configs:
        tag = _config_tag(benchmark, config)
        print(f"\n=== Evaluating {benchmark.value}: {tag} ===")

        if config.dataset == Dataset.CUB and config.n_shot_test == 0:
            image_head, attr_head = load_zeroshot_model(checkpoint_path, device)
            accuracies = evaluate_zeroshot(image_head, attr_head, config, device)
        else:
            model = load_model(checkpoint_path, config, device)
            accuracies = evaluate_fewshot(model, config, device)

        mean, std, ci95 = compute_stats(accuracies)
        print(f"  Accuracy: {mean:.4f} ± {ci95:.4f} (95% CI)")

        result = EvalResult(
            benchmark=benchmark.value,
            config=config,
            model_checkpoint=str(checkpoint_path),
            accuracy_mean=mean,
            accuracy_std=std,
            accuracy_ci95=ci95,
            episode_accuracies=accuracies,
        )
        results.append(result)

        result_path = output_dir / f"{benchmark.value}_{tag}.json"
        _save_result(result, result_path)

    return results


# ---------- CLI ----------

def main() -> None:
    parser = argparse.ArgumentParser(description="Few-shot / zero-shot evaluation")
    parser.add_argument("--benchmark", type=str, help="Benchmark name (use --list-benchmarks to see options)")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for JSON results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--list-benchmarks", action="store_true", help="List available benchmarks and exit")
    args = parser.parse_args()

    if args.list_benchmarks:
        for b in Benchmark:
            configs = BENCHMARK_CONFIGS[b]
            print(f"\n{b.value}:")
            for c in configs:
                extras = []
                if c.transductive:
                    extras.append("transductive")
                if c.n_refinement_steps != 4:
                    extras.append(f"steps={c.n_refinement_steps}")
                suffix = f", {', '.join(extras)}" if extras else ""
                print(f"  {c.n_way_test}-way {c.n_shot_test}-shot, {c.distance.value}, train_way={c.train_way}{suffix}")
        sys.exit(0)

    if not args.benchmark or not args.checkpoint:
        parser.error("--benchmark and --checkpoint are required")

    benchmark = Benchmark(args.benchmark)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)

    run_benchmark(benchmark, args.checkpoint, output_dir, device)


if __name__ == "__main__":
    main()
