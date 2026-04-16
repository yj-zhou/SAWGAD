import argparse

import numpy as np
import torch
from tqdm.auto import tqdm

from .data import build_weakly_supervised_split, load_amazon_graph
from .seed import set_global_seed
from .train import TrainConfig, train_one_run


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SAWGAD: weakly-supervised graph anomaly detection on Amazon.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)

    parser.add_argument("--num-labeled-anomalies", type=int, default=30)
    parser.add_argument("--contamination-rate", type=float, default=0.01)

    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--drop-rate", type=float, default=0.25)
    parser.add_argument("--weight-decay", type=float, default=0.002)

    parser.add_argument("--encoder-num-layers", type=int, default=1)
    parser.add_argument("--encoder-num-heads", type=int, default=4)
    parser.add_argument("--encoder-hidden-multiplier", type=float, default=1.25)

    parser.add_argument("--synthetic-anomalies-per-type", type=int, default=16)
    parser.add_argument("--regularization-weight", type=float, default=4.44)

    parser.add_argument("--warmup-epochs", type=int, default=100)
    parser.add_argument("--warmup-batch-size", type=int, default=256)
    parser.add_argument("--warmup-batches-per-epoch", type=int, default=10)
    parser.add_argument("--warmup-lr", type=float, default=0.0003)

    parser.add_argument("--full-epochs", type=int, default=350)
    parser.add_argument("--full-batch-size", type=int, default=256)
    parser.add_argument("--full-lr", type=float, default=0.003)
    return parser


def _config_from_args(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        hidden_dim=args.hidden_dim,
        drop_rate=args.drop_rate,
        weight_decay=args.weight_decay,
        encoder_num_layers=args.encoder_num_layers,
        encoder_num_heads=args.encoder_num_heads,
        encoder_hidden_multiplier=args.encoder_hidden_multiplier,
        synthetic_anomalies_per_type=args.synthetic_anomalies_per_type,
        regularization_weight=args.regularization_weight,
        warmup_epochs=args.warmup_epochs,
        warmup_batch_size=args.warmup_batch_size,
        warmup_batches_per_epoch=args.warmup_batches_per_epoch,
        warmup_lr=args.warmup_lr,
        full_epochs=args.full_epochs,
        full_batch_size=args.full_batch_size,
        full_lr=args.full_lr,
    )


def main() -> None:
    args = _build_parser().parse_args()
    config = _config_from_args(args)

    auc_scores: list[float] = []
    ap_scores: list[float] = []

    run_bar = tqdm(range(args.num_runs), desc="runs")
    for run_idx in run_bar:
        run_seed = args.base_seed + run_idx
        set_global_seed(run_seed)

        graph = load_amazon_graph(device=args.device)
        split = build_weakly_supervised_split(
            graph,
            seed=args.split_seed,
            num_labeled_anomalies=args.num_labeled_anomalies,
            contamination_rate=args.contamination_rate,
        )

        auc, ap = train_one_run(
            split=split,
            config=config,
            device=args.device,
            run_desc=f"run-{run_idx}",
        )
        auc_scores.append(auc)
        ap_scores.append(ap)
        run_bar.set_postfix(
            auc=f"{auc:.4f}",
            ap=f"{ap:.4f}",
            mean_auc=f"{np.mean(auc_scores):.4f}",
            mean_ap=f"{np.mean(ap_scores):.4f}",
        )

        del graph, split
        torch.cuda.empty_cache()

    auc_arr = np.asarray(auc_scores)
    ap_arr = np.asarray(ap_scores)
    print(
        f"AUROC: {auc_arr.mean():.4f} +/- {auc_arr.std():.4f} "
        f"| AUPRC: {ap_arr.mean():.4f} +/- {ap_arr.std():.4f} "
        f"over {args.num_runs} runs"
    )


if __name__ == "__main__":
    main()
