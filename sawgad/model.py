import dgl
import torch
import torch.nn.functional as F
from dgl import ops
from dgl.nn.functional import edge_softmax


class _FeedForward(torch.nn.Module):
    def __init__(self, dim: int, hidden_multiplier: float, drop_rate: float, input_multiplier: int = 1):
        super().__init__()
        input_dim = int(dim * input_multiplier)
        hidden_dim = int(dim * hidden_multiplier)
        self.linear_1 = torch.nn.Linear(input_dim, hidden_dim)
        self.dropout_1 = torch.nn.Dropout(drop_rate)
        self.act = torch.nn.GELU()
        self.linear_2 = torch.nn.Linear(hidden_dim, dim)
        self.dropout_2 = torch.nn.Dropout(drop_rate)

    def forward(self, _graph: dgl.DGLGraph, h: torch.Tensor) -> torch.Tensor:
        h = self.linear_1(h)
        h = self.dropout_1(h)
        h = self.act(h)
        h = self.linear_2(h)
        h = self.dropout_2(h)
        return h


class _GATSepModule(torch.nn.Module):
    def __init__(self, dim: int, hidden_multiplier: float, num_heads: int, drop_rate: float):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.input_linear = torch.nn.Linear(dim, dim)
        self.attn_linear_u = torch.nn.Linear(dim, num_heads)
        self.attn_linear_v = torch.nn.Linear(dim, num_heads, bias=False)
        self.attn_act = torch.nn.LeakyReLU(negative_slope=0.2)
        self.feed_forward = _FeedForward(
            dim=dim,
            hidden_multiplier=hidden_multiplier,
            drop_rate=drop_rate,
            input_multiplier=2,
        )

    def forward(self, graph: dgl.DGLGraph, h: torch.Tensor) -> torch.Tensor:
        h = self.input_linear(h)

        attn_u = self.attn_linear_u(h)
        attn_v = self.attn_linear_v(h)
        attn_scores = self.attn_act(ops.u_add_v(graph, attn_u, attn_v))
        attn_probs = edge_softmax(graph, attn_scores)

        h_heads = h.reshape(-1, self.head_dim, self.num_heads)
        message = ops.u_mul_e_sum(graph, h_heads, attn_probs).reshape(-1, self.dim)

        combined = torch.cat([h, message], dim=1)
        return self.feed_forward(graph, combined)


class _ResidualBlock(torch.nn.Module):
    def __init__(self, inner: torch.nn.Module, dim: int):
        super().__init__()
        self.norm = torch.nn.Identity()
        self.inner = inner

    def forward(self, graph: dgl.DGLGraph, h: torch.Tensor) -> torch.Tensor:
        return h + self.inner(graph, self.norm(h))


class GATSepEncoder(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        hidden_multiplier: float = 1.0,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.input_linear = torch.nn.Linear(in_features, hidden_dim)
        self.act = torch.nn.GELU()
        self.drop_rate = drop_rate
        self.dropout = torch.nn.Dropout(drop_rate) if drop_rate > 0 else None

        self.layers = torch.nn.ModuleList(
            [
                _ResidualBlock(
                    _GATSepModule(
                        dim=hidden_dim,
                        hidden_multiplier=hidden_multiplier,
                        num_heads=num_heads,
                        drop_rate=drop_rate,
                    ),
                    dim=hidden_dim,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, graph: dgl.DGLGraph) -> torch.Tensor:
        h = graph.ndata["feature"]
        h = self.input_linear(h)
        h = self.act(h)
        for layer in self.layers:
            h = layer(graph, h)
        if self.dropout is not None:
            h = self.dropout(h)
        return h


class _TwoLayerMLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, drop_rate: float = 0.0):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.Dropout(drop_rate),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SAWGADModel(torch.nn.Module):
    """Feature encoder + specialized synthetic heads + real-anomaly scoring head."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_synthetic_heads: int,
        drop_rate: float = 0.0,
        encoder_num_layers: int = 2,
        encoder_num_heads: int = 4,
        encoder_hidden_multiplier: float = 1.0,
    ):
        super().__init__()
        self.encoder = GATSepEncoder(
            in_features=in_features,
            hidden_dim=hidden_dim,
            num_layers=encoder_num_layers,
            num_heads=encoder_num_heads,
            hidden_multiplier=encoder_hidden_multiplier,
            drop_rate=drop_rate,
        )
        self.synthetic_heads = torch.nn.ModuleList(
            [
                _TwoLayerMLP(hidden_dim, hidden_dim, 1, drop_rate=drop_rate)
                for _ in range(num_synthetic_heads)
            ]
        )
        self.real_head = _TwoLayerMLP(hidden_dim, hidden_dim, 1, drop_rate=drop_rate)

    def encode(self, graph: dgl.DGLGraph) -> torch.Tensor:
        return self.encoder(graph)

    def score_synthetic(
        self, graph: dgl.DGLGraph, per_head_indices: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        h = self.encoder(graph)
        return [head(h[idx]) for head, idx in zip(self.synthetic_heads, per_head_indices)]

    def score_real(self, graph: dgl.DGLGraph, indices: torch.Tensor) -> torch.Tensor:
        h = self.encoder(graph)[indices]
        return self.real_head(h)
