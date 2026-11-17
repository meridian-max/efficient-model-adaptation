import math
import torch
from torch import nn


class LoRALinear(nn.Module):
    """A frozen linear map plus a trainable, scaled low-rank update."""
    def __init__(self, base: nn.Linear, rank=2, alpha=2.0):
        super().__init__()
        if type(rank) is not int or rank <= 0 or rank > min(base.in_features, base.out_features):
            raise ValueError("rank must be positive and no greater than the smaller feature dimension")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        self.base = nn.Linear(base.in_features, base.out_features, bias=base.bias is not None,
                              device=base.weight.device, dtype=base.weight.dtype)
        self.base.load_state_dict(base.state_dict())
        self.base.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + (x @ self.a.T @ self.b.T) * self.scale

    def merged(self):
        """Return an independent ordinary Linear layer; do not mutate the adapter."""
        layer = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None,
                          device=self.base.weight.device, dtype=self.base.weight.dtype)
        with torch.no_grad():
            layer.weight.copy_(self.base.weight + self.b @ self.a * self.scale)
            if layer.bias is not None:
                layer.bias.copy_(self.base.bias)
        return layer


def parameter_counts(model):
    return {"total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
