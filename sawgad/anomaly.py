from copy import deepcopy
from typing import Callable, Optional, cast

import dgl
import torch


AnomalyFn = Callable[[dgl.DGLGraph, torch.Tensor, Optional[torch.Generator]], None]


def _degree_anomaly(
    graph: dgl.DGLGraph,
    target_indices: torch.Tensor,
    generator: Optional[torch.Generator],
    min_factor: int = 3,
    max_factor: int = 5,
) -> None:
    idtype = graph.idtype
    target_indices = target_indices.to(idtype)

    degree_std = graph.in_degrees().float().std().item()
    degrees_to_add = (
        torch.randint(
            min_factor,
            max_factor + 1,
            (len(target_indices),),
            device=graph.device,
            generator=generator,
        ).float()
        * degree_std
    ).int()

    num_nodes = graph.num_nodes()
    source_nodes = torch.cat(
        [
            torch.randperm(
                num_nodes, dtype=idtype, device=graph.device, generator=generator
            )[: int(degrees_to_add[i].item())]
            for i in range(len(target_indices))
        ]
    )

    repeated_targets = torch.repeat_interleave(target_indices, degrees_to_add)
    graph.add_edges(source_nodes, repeated_targets)


def _dissimilar_connection_anomaly(
    graph: dgl.DGLGraph,
    target_indices: torch.Tensor,
    generator: Optional[torch.Generator],
    candidate_size: int = 4096,
) -> None:
    idtype = graph.idtype
    target_indices = target_indices.to(idtype)

    num_nodes = graph.num_nodes()
    candidate_nodes = torch.randperm(
        num_nodes, dtype=idtype, device=graph.device, generator=generator
    )[:candidate_size]

    target_features = graph.ndata["feature"][target_indices]
    candidate_features = graph.ndata["feature"][candidate_nodes]

    distances = torch.cdist(
        target_features.unsqueeze(1), candidate_features.unsqueeze(0), p=2
    )
    most_dissimilar_local = torch.argmax(distances, dim=2).squeeze(1)
    graph.add_edges(candidate_nodes[most_dissimilar_local], target_indices)


def _structural_reorganization_anomaly(
    graph: dgl.DGLGraph,
    target_indices: torch.Tensor,
    generator: Optional[torch.Generator],
) -> None:
    idtype = graph.idtype
    target_indices = target_indices.to(idtype)

    for target_idx in target_indices:
        degree = int(graph.in_degrees(target_idx).item()) - 1
        if degree <= 0:
            continue

        in_eid = cast(torch.Tensor, graph.in_edges(target_idx, form="eid"))
        graph.remove_edges(in_eid)

        num_nodes = graph.num_nodes()
        new_sources = torch.randperm(
            num_nodes, dtype=idtype, device=graph.device, generator=generator
        )[:degree]

        new_targets = torch.full(
            (degree,), target_idx, dtype=idtype, device=graph.device
        )
        self_loop_src = target_idx.reshape(1)
        self_loop_dst = target_idx.reshape(1)

        graph.add_edges(
            torch.cat([new_sources, self_loop_src]),
            torch.cat([new_targets, self_loop_dst]),
        )


def _feature_replacement_anomaly(
    graph: dgl.DGLGraph,
    target_indices: torch.Tensor,
    generator: Optional[torch.Generator],
    candidate_size: int = 4096,
) -> None:
    num_nodes = graph.num_nodes()
    candidate_nodes = torch.randperm(
        num_nodes, device=graph.device, generator=generator
    )[:candidate_size]

    target_features = graph.ndata["feature"][target_indices]
    candidate_features = graph.ndata["feature"][candidate_nodes]

    distances = torch.cdist(
        target_features.unsqueeze(1), candidate_features.unsqueeze(0), p=2
    )
    most_dissimilar_local = torch.argmax(distances, dim=2).squeeze(1)

    graph.ndata["feature"][target_indices] = candidate_features[most_dissimilar_local]


def _feature_perturbation_anomaly(
    graph: dgl.DGLGraph,
    target_indices: torch.Tensor,
    generator: Optional[torch.Generator],
    min_dims: int = 2,
    max_dims: int = 5,
    max_dim_percentage: float = 0.1,
) -> None:
    all_features = cast(torch.Tensor, graph.ndata["feature"])
    feature_std = all_features.std(dim=0)
    feature_max = all_features.max(dim=0).values

    perturbed = all_features[target_indices].clone()
    num_dims = all_features.size(1)
    upper_bound = max(max_dims, int(num_dims * max_dim_percentage)) + 1

    for idx in range(len(target_indices)):
        num_dims_to_change = int(
            torch.randint(
                min_dims,
                upper_bound,
                (1,),
                device=graph.device,
                generator=generator,
            ).item()
        )
        dims_to_change = torch.randperm(
            num_dims, device=graph.device, generator=generator
        )[:num_dims_to_change]

        perturbation = (
            feature_max[dims_to_change]
            + torch.rand(
                num_dims_to_change, device=graph.device, generator=generator
            )
            * feature_std[dims_to_change]
        )
        perturbed[idx, dims_to_change] = perturbation

    all_features[target_indices] = perturbed
    graph.ndata["feature"] = all_features


ANOMALY_FUNCTIONS: tuple[AnomalyFn, ...] = (
    _degree_anomaly,
    _dissimilar_connection_anomaly,
    _structural_reorganization_anomaly,
    _feature_replacement_anomaly,
    _feature_perturbation_anomaly,
)

NUM_ANOMALY_TYPES = len(ANOMALY_FUNCTIONS)


def inject_synthetic_anomalies(
    graph: dgl.DGLGraph,
    candidate_nodes: torch.Tensor,
    anomalies_per_type: int,
    generator: Optional[torch.Generator] = None,
) -> tuple[dgl.DGLGraph, list[torch.Tensor]]:
    new_graph = deepcopy(graph)

    total_needed = NUM_ANOMALY_TYPES * anomalies_per_type
    shuffled_indices = torch.randperm(
        len(candidate_nodes), device=graph.device, generator=generator
    )[:total_needed]
    per_type_local_indices = shuffled_indices.split(anomalies_per_type)

    for i, anomaly_fn in enumerate(ANOMALY_FUNCTIONS):
        type_targets = candidate_nodes[per_type_local_indices[i]]
        anomaly_fn(new_graph, type_targets, generator)

    per_type_labels: list[torch.Tensor] = []
    for i in range(NUM_ANOMALY_TYPES):
        label = torch.zeros(candidate_nodes.size(0), dtype=torch.long)
        label[per_type_local_indices[i].cpu()] = 1
        per_type_labels.append(label)

    return new_graph, per_type_labels
