from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import trange

if TYPE_CHECKING:
    from jaxtyping import Float

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
    OMNIGLOT_REFINEMENT_STEPS = "omniglot_refinement_steps"
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


_OMNIGLOT_FEWSHOT_CONFIGS = [
    EvalConfig(
        Dataset.OMNIGLOT,
        DistanceMetric.EUCLIDEAN,
        n_way_test=way,
        n_shot_test=shot,
        n_query=5,
        n_episodes=1000,
        transductive=False,
        train_way=60,
        train_shot=shot,
    )
    for way in [5, 20]
    for shot in [1, 5]
]

_MINIIMAGENET_FEWSHOT_CONFIGS = [
    EvalConfig(
        Dataset.MINI_IMAGENET,
        DistanceMetric.EUCLIDEAN,
        n_way_test=5,
        n_shot_test=1,
        n_query=15,
        n_episodes=600,
        transductive=False,
        train_way=30,
        train_shot=1,
    ),
    EvalConfig(
        Dataset.MINI_IMAGENET,
        DistanceMetric.EUCLIDEAN,
        n_way_test=5,
        n_shot_test=5,
        n_query=15,
        n_episodes=600,
        transductive=False,
        train_way=20,
        train_shot=5,
    ),
]

_MINIIMAGENET_DISTANCE_WAY_CONFIGS = [
    EvalConfig(
        Dataset.MINI_IMAGENET,
        dist,
        n_way_test=5,
        n_shot_test=shot,
        n_query=15,
        n_episodes=600,
        transductive=False,
        train_way=way,
        train_shot=shot,
    )
    for dist in [DistanceMetric.COSINE, DistanceMetric.EUCLIDEAN]
    for way in [5, 20]
    for shot in [1, 5]
]

_MINIIMAGENET_WAY_ABLATION_CONFIGS = [
    EvalConfig(
        Dataset.MINI_IMAGENET,
        DistanceMetric.EUCLIDEAN,
        n_way_test=5,
        n_shot_test=shot,
        n_query=15,
        n_episodes=600,
        transductive=False,
        train_way=way,
        train_shot=shot,
    )
    for way in [5, 10, 15, 20, 25, 30]
    for shot in [1, 5]
]

_MINIIMAGENET_TRANSDUCTIVE_CONFIGS = [
    EvalConfig(
        Dataset.MINI_IMAGENET,
        DistanceMetric.EUCLIDEAN,
        n_way_test=5,
        n_shot_test=shot,
        n_query=15,
        n_episodes=600,
        transductive=trans,
        train_way=20,
        train_shot=shot,
    )
    for shot in [1, 5]
    for trans in [False, True]
]

_MINIIMAGENET_10WAY_CONFIGS = [
    EvalConfig(
        Dataset.MINI_IMAGENET,
        DistanceMetric.EUCLIDEAN,
        n_way_test=10,
        n_shot_test=shot,
        n_query=10,
        n_episodes=600,
        transductive=trans,
        train_way=20,
        train_shot=shot,
    )
    for shot in [1, 5]
    for trans in [False, True]
]

_REFINEMENT_STEPS_CONFIGS = [
    EvalConfig(
        Dataset.MINI_IMAGENET,
        DistanceMetric.EUCLIDEAN,
        n_way_test=5,
        n_shot_test=shot,
        n_query=15,
        n_episodes=600,
        transductive=True,
        train_way=20,
        train_shot=shot,
        n_refinement_steps=steps,
    )
    for shot in [1, 5]
    for steps in range(11)
]

_OMNIGLOT_REFINEMENT_STEPS_CONFIGS = [
    EvalConfig(
        Dataset.OMNIGLOT,
        DistanceMetric.EUCLIDEAN,
        n_way_test=5,
        n_shot_test=shot,
        n_query=5,
        n_episodes=1000,
        transductive=True,
        train_way=60,
        train_shot=shot,
        n_refinement_steps=steps,
    )
    for shot in [1, 5]
    for steps in range(11)
]

_CUB_ZEROSHOT_CONFIGS = [
    EvalConfig(
        Dataset.CUB,
        DistanceMetric.EUCLIDEAN,
        n_way_test=50,
        n_shot_test=0,
        n_query=10,
        n_episodes=600,
        transductive=False,
        train_way=50,
        train_shot=0,
    ),
]

BENCHMARK_CONFIGS: dict[Benchmark, list[EvalConfig]] = {
    Benchmark.OMNIGLOT_FEWSHOT: _OMNIGLOT_FEWSHOT_CONFIGS,
    Benchmark.MINIIMAGENET_FEWSHOT: _MINIIMAGENET_FEWSHOT_CONFIGS,
    Benchmark.MINIIMAGENET_DISTANCE_WAY: _MINIIMAGENET_DISTANCE_WAY_CONFIGS,
    Benchmark.MINIIMAGENET_WAY_ABLATION: _MINIIMAGENET_WAY_ABLATION_CONFIGS,
    Benchmark.MINIIMAGENET_TRANSDUCTIVE: _MINIIMAGENET_TRANSDUCTIVE_CONFIGS,
    Benchmark.MINIIMAGENET_10WAY: _MINIIMAGENET_10WAY_CONFIGS,
    Benchmark.REFINEMENT_STEPS: _REFINEMENT_STEPS_CONFIGS,
    Benchmark.OMNIGLOT_REFINEMENT_STEPS: _OMNIGLOT_REFINEMENT_STEPS_CONFIGS,
    Benchmark.CUB_ZEROSHOT: _CUB_ZEROSHOT_CONFIGS,
}


def load_model(
    checkpoint_path: str | Path,
    config: EvalConfig,
    device: torch.device,
) -> ProtoNet:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint["train_config"]
    encoder = ProtoNetEncoder(
        in_channels=cfg["in_channels"],
        hidden_channels=cfg.get("hidden_channels", 64),
    )
    model = ProtoNet(encoder, distance=DistanceMetric(cfg["distance"]), transductive=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
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


def _config_tag(benchmark: Benchmark, config: EvalConfig) -> str:
    tag = f"{config.n_way_test}way_{config.n_shot_test}shot"
    if benchmark in {Benchmark.MINIIMAGENET_DISTANCE_WAY, Benchmark.MINIIMAGENET_WAY_ABLATION}:
        tag += f"_train{config.train_way}way_{config.distance.value}"
    if benchmark in {Benchmark.MINIIMAGENET_TRANSDUCTIVE, Benchmark.MINIIMAGENET_10WAY}:
        tag += f"_trans{config.transductive}"
    if benchmark in {Benchmark.REFINEMENT_STEPS, Benchmark.OMNIGLOT_REFINEMENT_STEPS}:
        tag += f"_steps{config.n_refinement_steps}"
    return tag


EVAL_CONFIGS: dict[str, tuple[Benchmark, EvalConfig]] = {
    f"{benchmark.value}_{_config_tag(benchmark, cfg)}": (benchmark, cfg)
    for benchmark, cfgs in BENCHMARK_CONFIGS.items()
    for cfg in cfgs
}


def run_single_eval(
    benchmark: Benchmark,
    config: EvalConfig,
    checkpoint_path: str | Path,
    output_dir: Path,
    device: torch.device,
) -> EvalResult:
    tag = _config_tag(benchmark, config)
    print(f"\n=== Evaluating {benchmark.value}: {tag} ===")

    if config.dataset == Dataset.CUB and config.n_shot_test == 0:
        image_head, attr_head = load_zeroshot_model(checkpoint_path, device)
        accuracies = evaluate_zeroshot(image_head, attr_head, config, device)
    else:
        model = load_model(checkpoint_path, config, device)
        accuracies = evaluate_fewshot(model, config, device)

    n = len(accuracies)
    mean = sum(accuracies) / n
    std = math.sqrt(sum((a - mean) ** 2 for a in accuracies) / (n - 1))
    ci95 = 1.96 * std / math.sqrt(n)
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

    serialized_config = asdict(config)
    serialized_config["dataset"] = config.dataset.value
    serialized_config["distance"] = config.distance.value
    result_path = output_dir / f"{benchmark.value}_{tag}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(
            {
                "benchmark": result.benchmark,
                "config": serialized_config,
                "model_checkpoint": result.model_checkpoint,
                "accuracy_mean": result.accuracy_mean,
                "accuracy_std": result.accuracy_std,
                "accuracy_ci95": result.accuracy_ci95,
                "episode_accuracies": result.episode_accuracies,
            },
            indent=2,
        ),
    )
    print(f"  Saved: {result_path}")
    return result


def run_benchmark(
    benchmark: Benchmark,
    checkpoint_path: str | Path,
    output_dir: Path,
    device: torch.device,
) -> list[EvalResult]:
    return [
        run_single_eval(benchmark, config, checkpoint_path, output_dir, device)
        for config in BENCHMARK_CONFIGS[benchmark]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Few-shot / zero-shot evaluation")
    parser.add_argument("--eval", dest="eval_name", type=str, help="Single eval config name")
    parser.add_argument("--benchmark", type=str, help="Run all configs in a benchmark")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for JSON results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--list-benchmarks", action="store_true", help="List available eval configs and exit")
    args = parser.parse_args()

    if args.list_benchmarks:
        for b in Benchmark:
            print(f"\n{b.value}:")
            for cfg in BENCHMARK_CONFIGS[b]:
                tag = _config_tag(b, cfg)
                print(f"  {b.value}_{tag}")
        sys.exit(0)

    if not args.checkpoint:
        parser.error("--checkpoint is required")

    if not args.eval_name and not args.benchmark:
        parser.error("--eval or --benchmark is required")

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)

    if args.eval_name:
        if args.eval_name not in EVAL_CONFIGS:
            parser.error(f"Unknown eval config: {args.eval_name}. Use --list-benchmarks to see options.")
        benchmark, config = EVAL_CONFIGS[args.eval_name]
        run_single_eval(benchmark, config, args.checkpoint, output_dir, device)
    else:
        benchmark = Benchmark(args.benchmark)
        run_benchmark(benchmark, args.checkpoint, output_dir, device)


if __name__ == "__main__":
    main()
