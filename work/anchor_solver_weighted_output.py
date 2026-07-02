from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

import anchor_solver_ml_distance_completion as dc


class WeightedDistanceCompletionNet(nn.Module):
    """Distance completer with an additional per-pair spring-weight multiplier head.

    The distance path is architecture-compatible with DistanceCompletionNet. The
    spring weight is a multiplier on the existing heuristic optimizer spring
    weight, so weight_multiplier=1.0 preserves previous behavior.
    """

    def __init__(
        self,
        node_feature_count: int,
        edge_feature_count: int,
        *,
        hidden: int,
        layers: int,
        dropout: float,
        max_abs_log_weight: float = 2.0,
    ) -> None:
        super().__init__()
        self.max_abs_log_weight = float(max_abs_log_weight)
        self.node_projection = nn.Sequential(
            nn.Linear(node_feature_count, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.message_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden * 2 + edge_feature_count, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, hidden),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, hidden),
                )
                for _ in range(layers)
            ]
        )
        self.update_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden * 2, hidden),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, hidden),
                )
                for _ in range(layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.graph_projection = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.distance_head = nn.Sequential(
            nn.Linear(hidden * 4 + edge_feature_count, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.weight_head = nn.Sequential(
            nn.Linear(hidden * 4 + edge_feature_count, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.initialize_neutral_weight_head()

    def initialize_neutral_weight_head(self) -> None:
        for module in self.weight_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        final = self.weight_head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def load_distance_teacher_state_dict(self, teacher_state: dict[str, torch.Tensor]) -> None:
        own = self.state_dict()
        copied: dict[str, torch.Tensor] = {}
        for key in own:
            if key.startswith("distance_head."):
                teacher_key = "pair_head." + key[len("distance_head.") :]
            elif key.startswith("weight_head."):
                continue
            else:
                teacher_key = key
            if teacher_key in teacher_state and own[key].shape == teacher_state[teacher_key].shape:
                copied[key] = teacher_state[teacher_key]
        missing, unexpected = self.load_state_dict(copied, strict=False)
        unexpected = [item for item in unexpected if not item.startswith("weight_head.")]
        missing_allowed = [item for item in missing if item.startswith("weight_head.")]
        missing_bad = [item for item in missing if not item.startswith("weight_head.")]
        if unexpected or missing_bad:
            raise RuntimeError(f"Unexpected teacher load mismatch: missing={missing_bad[:8]} unexpected={unexpected[:8]}")
        if not missing_allowed:
            raise RuntimeError("Expected neutral weight_head parameters to be absent from teacher load.")
        self.initialize_neutral_weight_head()

    def _pair_input(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        mask: torch.Tensor,
        measured_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.node_projection(node_features) * mask.unsqueeze(-1)
        hop_norm = edge_features[..., 7]
        measured_weight = measured_mask.float()
        short_path_weight = torch.exp(-3.0 * hop_norm) * (
            (hop_norm > 0.0) & (hop_norm <= 0.50) & pair_mask & ~measured_mask
        ).float()
        message_weight = (measured_weight + 0.35 * short_path_weight).unsqueeze(-1)
        normalizer = message_weight.sum(dim=2).clamp_min(1.0)
        for message_layer, update_layer, norm in zip(self.message_layers, self.update_layers, self.norms):
            source = h.unsqueeze(1).expand(-1, h.shape[1], -1, -1)
            target = h.unsqueeze(2).expand(-1, -1, h.shape[1], -1)
            message_input = torch.cat([target, source, edge_features], dim=-1)
            messages = message_layer(message_input) * message_weight
            aggregate = messages.sum(dim=2) / normalizer
            update = update_layer(torch.cat([h, aggregate], dim=-1))
            h = norm(h + update) * mask.unsqueeze(-1)

        weights = mask.float().unsqueeze(-1)
        graph_context = (h * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        graph_context = self.graph_projection(graph_context)
        graph_pair = graph_context.unsqueeze(1).unsqueeze(2).expand(-1, h.shape[1], h.shape[1], -1)
        hi = h.unsqueeze(2).expand(-1, -1, h.shape[1], -1)
        hj = h.unsqueeze(1).expand(-1, h.shape[1], -1, -1)
        return torch.cat([hi + hj, torch.abs(hi - hj), hi * hj, graph_pair, edge_features], dim=-1)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        mask: torch.Tensor,
        measured_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pair_input = self._pair_input(node_features, edge_features, mask, measured_mask, pair_mask)
        raw_distance = self.distance_head(pair_input).squeeze(-1)
        positive = F.softplus(raw_distance) + 1e-4
        shortest_norm = edge_features[..., 5]
        radio_lower_norm = edge_features[..., 9]
        missing = edge_features[..., 2] > 0.5
        upper = torch.maximum(shortest_norm, radio_lower_norm + 0.05)
        bounded_missing = radio_lower_norm + torch.sigmoid(raw_distance) * (upper - radio_lower_norm)
        pred = torch.where(missing, bounded_missing, positive)
        pred = 0.5 * (pred + pred.transpose(1, 2))

        raw_weight = self.weight_head(pair_input).squeeze(-1)
        log_weight = self.max_abs_log_weight * torch.tanh(raw_weight / self.max_abs_log_weight)
        weight_multiplier = torch.exp(log_weight)
        weight_multiplier = 0.5 * (weight_multiplier + weight_multiplier.transpose(1, 2))

        eye = torch.eye(pred.shape[1], dtype=torch.bool, device=pred.device).unsqueeze(0)
        valid = pair_mask & ~eye
        pred = pred.masked_fill(~valid, 0.0)
        weight_multiplier = weight_multiplier.masked_fill(~valid, 0.0)
        return pred, weight_multiplier


def load_expanded_weighted_state_dict(
    model: WeightedDistanceCompletionNet,
    source_state: dict[str, torch.Tensor],
) -> dict[str, object]:
    """Load a weighted checkpoint into a model with extra input features.

    Linear layers whose input dimension grew get their old columns copied and the
    new feature columns zero-filled, preserving the old policy at initialization.
    """

    own = model.state_dict()
    copied: dict[str, torch.Tensor] = {}
    expanded: list[str] = []
    skipped: list[str] = []
    for key, source_value in source_state.items():
        if key not in own:
            skipped.append(key)
            continue
        target_value = own[key]
        if target_value.shape == source_value.shape:
            copied[key] = source_value
            continue
        if (
            target_value.ndim == 2
            and source_value.ndim == 2
            and target_value.shape[0] == source_value.shape[0]
            and target_value.shape[1] >= source_value.shape[1]
        ):
            expanded_value = torch.zeros_like(target_value)
            expanded_value[:, : source_value.shape[1]] = source_value.to(device=target_value.device, dtype=target_value.dtype)
            copied[key] = expanded_value
            expanded.append(key)
            continue
        skipped.append(key)
    missing, unexpected = model.load_state_dict(copied, strict=False)
    return {
        "copied": len(copied),
        "expanded": expanded,
        "missing": list(missing),
        "unexpected": list(unexpected),
        "skipped": skipped,
    }


def weighted_model_from_teacher_checkpoint(
    checkpoint: dict,
    *,
    node_feature_count: int,
    edge_feature_count: int,
    device: torch.device,
    max_abs_log_weight: float = 2.0,
) -> WeightedDistanceCompletionNet:
    saved_args = checkpoint.get("args", {})
    model = WeightedDistanceCompletionNet(
        node_feature_count,
        edge_feature_count,
        hidden=int(saved_args.get("hidden", 144)),
        layers=int(saved_args.get("layers", 5)),
        dropout=float(saved_args.get("dropout", 0.0)),
        max_abs_log_weight=max_abs_log_weight,
    ).to(device)
    model.load_distance_teacher_state_dict(checkpoint["model_state_dict"])
    return model
