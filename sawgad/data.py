from dataclasses import dataclass
from typing import cast

import dgl
import numpy as np
import torch
from dgl.data import FraudAmazonDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Sampler


@dataclass
class DataSplit:
    graph: dgl.DGLGraph
    train_nodes: torch.Tensor
    val_nodes: torch.Tensor
    test_nodes: torch.Tensor


class IndexLabelDataset(Dataset):
    def __init__(self, indices: torch.Tensor, labels: torch.Tensor):
        self.indices = indices
        self.labels = labels

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.indices[idx], self.labels[idx]


class BalancedBatchSampler(Sampler):
    def __init__(self, labels: torch.Tensor, batch_size: int, num_batches: int):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.pos_indices = torch.where(labels == 1)[0]
        self.neg_indices = torch.where(labels == 0)[0]

    def __iter__(self):
        half = self.batch_size // 2
        for _ in range(self.num_batches):
            pos = self.pos_indices[
                torch.randint(
                    len(self.pos_indices), (half,), device=self.pos_indices.device
                )
            ]
            neg = self.neg_indices[
                torch.randint(
                    len(self.neg_indices), (half,), device=self.neg_indices.device
                )
            ]
            batch = torch.cat((pos, neg))
            batch = batch[torch.randperm(batch.size(0), device=batch.device)]
            yield batch.tolist()

    def __len__(self) -> int:
        return self.num_batches


class FullBatchSampler(Sampler):
    def __init__(self, labels: torch.Tensor):
        self.size = len(labels)

    def __iter__(self):
        yield torch.arange(self.size).tolist()

    def __len__(self) -> int:
        return 1


def _to_homogeneous(graph: dgl.DGLGraph) -> dgl.DGLGraph:
    if not graph.is_homogeneous:
        graph = dgl.to_homogeneous(graph, ndata=["feature", "label"])
    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)
    graph = dgl.to_simple(graph)
    return cast(dgl.DGLGraph, graph)


def _normalize_features(graph: dgl.DGLGraph) -> None:
    feature = cast(torch.Tensor, graph.ndata["feature"]).float()
    feature_min = feature.min(dim=0).values
    feature_max = feature.max(dim=0).values
    denom = (feature_max - feature_min).clamp_min(1e-12)
    graph.ndata["feature"] = (feature - feature_min) / denom


def load_amazon_graph(device: str) -> dgl.DGLGraph:
    dataset = FraudAmazonDataset()
    graph = dataset[0]
    graph = _to_homogeneous(graph)
    _normalize_features(graph)
    return graph.to(device)


def build_weakly_supervised_split(
    graph: dgl.DGLGraph,
    seed: int,
    num_labeled_anomalies: int,
    contamination_rate: float,
) -> DataSplit:
    label = cast(torch.Tensor, graph.ndata["label"]).clone()
    all_nodes = np.arange(graph.number_of_nodes())

    rng = np.random.RandomState(seed)

    train_nodes, val_test_nodes = train_test_split(
        all_nodes,
        test_size=0.2,
        stratify=label.cpu().numpy(),
        random_state=rng,
    )
    val_nodes, test_nodes = train_test_split(
        val_test_nodes,
        test_size=0.5,
        stratify=label[val_test_nodes].cpu().numpy(),
        random_state=rng,
    )

    train_label = label[train_nodes]
    negative_nodes = train_nodes[train_label.cpu().numpy() == 0]
    positive_nodes = train_nodes[train_label.cpu().numpy() == 1]

    num_labeled = min(num_labeled_anomalies, len(positive_nodes))
    chosen = positive_nodes[
        rng.choice(len(positive_nodes), num_labeled, replace=False)
    ]

    train_nodes = np.concatenate((chosen, negative_nodes))
    remaining_positive = np.setdiff1d(positive_nodes, chosen)

    if contamination_rate > 0 and len(remaining_positive) > 0:
        contamination_count = int(len(negative_nodes) * contamination_rate)
        contamination_count = min(contamination_count, len(remaining_positive))
        contamination = remaining_positive[
            rng.choice(len(remaining_positive), contamination_count, replace=False)
        ]
        train_nodes = np.concatenate((train_nodes, contamination))
        label[contamination] = 0
        remaining_positive = np.setdiff1d(remaining_positive, contamination)

    split_halves = np.array_split(remaining_positive, 2)
    val_nodes = np.concatenate((val_nodes, split_halves[0]))
    test_nodes = np.concatenate((test_nodes, split_halves[1]))

    graph.ndata["label"] = label

    device = graph.device
    return DataSplit(
        graph=graph,
        train_nodes=torch.as_tensor(train_nodes, dtype=torch.long, device=device),
        val_nodes=torch.as_tensor(val_nodes, dtype=torch.long, device=device),
        test_nodes=torch.as_tensor(test_nodes, dtype=torch.long, device=device),
    )
