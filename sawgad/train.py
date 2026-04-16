import copy
from dataclasses import dataclass
from typing import Iterator

import dgl
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .anomaly import NUM_ANOMALY_TYPES, inject_synthetic_anomalies
from .data import (
    BalancedBatchSampler,
    DataSplit,
    FullBatchSampler,
    IndexLabelDataset,
)
from .model import SAWGADModel


@dataclass
class TrainConfig:
    hidden_dim: int = 64
    drop_rate: float = 0.0
    weight_decay: float = 1e-4

    encoder_num_layers: int = 2
    encoder_num_heads: int = 4
    encoder_hidden_multiplier: float = 1.0

    synthetic_anomalies_per_type: int = 32
    regularization_weight: float = 4.0

    warmup_epochs: int = 100
    warmup_batch_size: int = 128
    warmup_batches_per_epoch: int = 10
    warmup_lr: float = 1e-3

    full_epochs: int = 256
    full_batch_size: int = 512
    full_lr: float = 1e-2


def _compute_metrics(
    model: SAWGADModel,
    graph: dgl.DGLGraph,
    nodes: torch.Tensor,
) -> tuple[float, float]:
    with torch.no_grad():
        scores = model.score_real(graph, nodes)
    labels_np = graph.ndata["label"][nodes].cpu().numpy()
    scores_np = scores.cpu().numpy()
    return (
        float(roc_auc_score(labels_np, scores_np)),
        float(average_precision_score(labels_np, scores_np)),
    )


def _make_synthetic_loaders(
    nodes: torch.Tensor,
    per_type_labels: list[torch.Tensor],
    batch_size_per_head: int,
    num_batches: int,
) -> list[Iterator]:
    loaders = []
    for label in per_type_labels:
        loader = DataLoader(
            IndexLabelDataset(nodes, label),
            batch_sampler=BalancedBatchSampler(label, batch_size_per_head, num_batches),
        )
        loaders.append(loader)
    return [iter(loader) for loader in loaders]


def _next_synthetic_batch(
    iters: list[Iterator], device: torch.device
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    indices_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    for it in iters:
        idx, y = next(it)
        indices_list.append(idx.to(device).int())
        labels_list.append(y.to(device).unsqueeze(1).float())
    return indices_list, labels_list


def _prepare_synthetic_epoch(
    graph: dgl.DGLGraph,
    train_nodes: torch.Tensor,
    per_head_batch_size: int,
    num_batches: int,
    anomalies_per_type: int,
) -> tuple[dgl.DGLGraph, list[Iterator]]:
    label = graph.ndata["label"]
    negative_train_nodes = train_nodes[label[train_nodes] == 0]
    perturbed_graph, per_type_labels = inject_synthetic_anomalies(
        graph,
        negative_train_nodes,
        anomalies_per_type=anomalies_per_type,
    )
    loader_iters = _make_synthetic_loaders(
        negative_train_nodes,
        per_type_labels,
        per_head_batch_size,
        num_batches,
    )
    return perturbed_graph, loader_iters


def _synthetic_loss(
    model: SAWGADModel,
    perturbed_graph: dgl.DGLGraph,
    loader_iters: list[Iterator],
) -> torch.Tensor:
    device = perturbed_graph.device
    indices_list, labels_list = _next_synthetic_batch(loader_iters, device)
    score_list = model.score_synthetic(perturbed_graph, indices_list)
    per_head_losses = [
        F.binary_cross_entropy_with_logits(score, y)
        for score, y in zip(score_list, labels_list)
    ]
    return torch.stack(per_head_losses).mean()


def _synthetic_val_auc(
    model: SAWGADModel,
    perturbed_val_graph: dgl.DGLGraph,
    val_negative_nodes: torch.Tensor,
    per_type_val_labels: list[torch.Tensor],
) -> float:
    with torch.no_grad():
        scores = model.score_synthetic(
            perturbed_val_graph, [val_negative_nodes] * NUM_ANOMALY_TYPES
        )
    aucs = [
        roc_auc_score(label.cpu().numpy(), score.cpu().numpy())
        for label, score in zip(per_type_val_labels, scores)
    ]
    return float(sum(aucs) / NUM_ANOMALY_TYPES)


def _warmup_phase(
    model: SAWGADModel,
    split: DataSplit,
    config: TrainConfig,
    progress_desc: str,
) -> None:
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.warmup_lr, weight_decay=config.weight_decay
    )
    per_head_batch_size = config.warmup_batch_size // NUM_ANOMALY_TYPES
    graph = split.graph

    label = graph.ndata["label"]
    val_negative_nodes = split.val_nodes[label[split.val_nodes] == 0]
    perturbed_val_graph, per_type_val_labels = inject_synthetic_anomalies(
        graph,
        val_negative_nodes,
        anomalies_per_type=config.synthetic_anomalies_per_type,
    )

    best_auc = -1.0
    best_state = copy.deepcopy(model.state_dict())

    epoch_bar = tqdm(
        range(config.warmup_epochs),
        desc=progress_desc,
        leave=False,
    )
    for _ in epoch_bar:
        model.train()
        perturbed_graph, loader_iters = _prepare_synthetic_epoch(
            graph,
            split.train_nodes,
            per_head_batch_size,
            config.warmup_batches_per_epoch,
            config.synthetic_anomalies_per_type,
        )
        batch_bar = tqdm(
            range(config.warmup_batches_per_epoch),
            desc="warmup-batches",
            leave=False,
        )
        for _ in batch_bar:
            optimizer.zero_grad()
            loss = _synthetic_loss(model, perturbed_graph, loader_iters)
            loss.backward()
            optimizer.step()
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        synth_auc = _synthetic_val_auc(
            model, perturbed_val_graph, val_negative_nodes, per_type_val_labels
        )
        epoch_bar.set_postfix(synth_val_auc=f"{synth_auc:.4f}")
        if synth_auc > best_auc:
            best_auc = synth_auc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)


def _full_training_phase(
    model: SAWGADModel,
    split: DataSplit,
    config: TrainConfig,
    progress_desc: str,
) -> None:
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.full_lr, weight_decay=config.weight_decay
    )
    graph = split.graph
    label = graph.ndata["label"]
    train_labels = label[split.train_nodes]
    sampler = FullBatchSampler(train_labels)
    loader = DataLoader(
        IndexLabelDataset(split.train_nodes, labels=train_labels),
        batch_sampler=sampler,
    )

    per_head_batch_size = (config.full_batch_size // 2) // NUM_ANOMALY_TYPES
    device = graph.device

    best_auc = -1.0
    best_state = copy.deepcopy(model.state_dict())

    epoch_bar = tqdm(
        range(config.full_epochs),
        desc=progress_desc,
        leave=False,
    )
    for _ in epoch_bar:
        model.train()
        perturbed_graph, loader_iters = _prepare_synthetic_epoch(
            graph,
            split.train_nodes,
            per_head_batch_size,
            num_batches=1,
            anomalies_per_type=config.synthetic_anomalies_per_type,
        )
        for batch_indices, batch_labels in loader:
            optimizer.zero_grad()
            indices = batch_indices.to(device).int()
            y = batch_labels.to(device).unsqueeze(1).float()
            logits = model.score_real(graph, indices)

            pos_weight = (1 - label[indices]).sum() / label[indices].sum().clamp_min(1)
            real_loss = F.binary_cross_entropy_with_logits(
                logits, y, pos_weight=pos_weight
            )
            synth_loss = _synthetic_loss(model, perturbed_graph, loader_iters)
            loss = real_loss + config.regularization_weight * synth_loss

            loss.backward()
            optimizer.step()

        model.eval()
        val_auc, _ = _compute_metrics(model, graph, split.val_nodes)
        epoch_bar.set_postfix(val_auc=f"{val_auc:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)


def train_one_run(
    split: DataSplit,
    config: TrainConfig,
    device: str,
    run_desc: str,
) -> tuple[float, float]:
    graph = split.graph
    in_features = int(graph.ndata["feature"].shape[1])
    model = SAWGADModel(
        in_features=in_features,
        hidden_dim=config.hidden_dim,
        num_synthetic_heads=NUM_ANOMALY_TYPES,
        drop_rate=config.drop_rate,
        encoder_num_layers=config.encoder_num_layers,
        encoder_num_heads=config.encoder_num_heads,
        encoder_hidden_multiplier=config.encoder_hidden_multiplier,
    ).to(device)

    _warmup_phase(model, split, config, progress_desc=f"{run_desc} warmup")
    _full_training_phase(model, split, config, progress_desc=f"{run_desc} full   ")

    model.eval()
    auc, ap = _compute_metrics(model, graph, split.test_nodes)
    return auc, ap
