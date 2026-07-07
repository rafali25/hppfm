from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import PulsePPGResNet1D


class MixtureOfExpertsRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        n_experts: int = 3,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or max(1, input_dim // 2)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(n_experts)
            ]
        )
        self.gate = nn.Sequential(nn.Linear(input_dim, n_experts), nn.Softmax(dim=1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        gate_weights = self.gate(x)
        output = torch.sum(gate_weights.unsqueeze(-1) * expert_outputs, dim=1)
        return output, gate_weights


class MorphologyAwarePulsePPG(nn.Module):
    """Pulse-PPG encoder trained with PaPaGei-S morphology supervision."""

    def __init__(
        self,
        encoder: PulsePPGResNet1D | None = None,
        *,
        embedding_dim: int = 512,
        n_experts: int = 3,
    ):
        super().__init__()
        self.encoder = encoder or PulsePPGResNet1D()
        feature_dim = self.encoder.output_dim
        self.embedding = nn.Linear(feature_dim, embedding_dim)
        self.ipa_head = MixtureOfExpertsRegressor(feature_dim, n_experts=n_experts)
        self.sqi_head = MixtureOfExpertsRegressor(feature_dim, n_experts=n_experts, dropout=0.3)

    def forward(self, signal: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(signal)
        embeddings = self.embedding(features)
        ipa_pred, ipa_gate = self.ipa_head(features)
        sqi_pred, sqi_gate = self.sqi_head(features)
        return {
            "embedding": embeddings,
            "features": features,
            "ipa": ipa_pred,
            "sqi": sqi_pred,
            "ipa_gate": ipa_gate,
            "sqi_gate": sqi_gate,
        }


def supervised_nt_xent_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Label-aware NT-Xent used for morphology bins."""
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got {embeddings.shape}")

    labels = labels.view(-1)
    batch_size = embeddings.shape[0]
    if batch_size <= 1:
        return embeddings.sum() * 0.0

    z = F.normalize(embeddings, dim=1)
    logits = torch.matmul(z, z.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    eye = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
    nonself_mask = ~eye
    positive_mask = labels[:, None].eq(labels[None, :]) & nonself_mask

    exp_logits = torch.exp(logits) * nonself_mask.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))

    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if not bool(valid.any()):
        return embeddings.sum() * 0.0

    mean_log_prob = (positive_mask.float() * log_prob).sum(dim=1) / positive_count.clamp_min(1)
    return -mean_log_prob[valid].mean()


@dataclass
class MorphologyLossOutput:
    loss: torch.Tensor
    contrastive: torch.Tensor
    ipa: torch.Tensor
    sqi: torch.Tensor


def morphology_ssl_loss(
    outputs: dict[str, torch.Tensor],
    svri_bin: torch.Tensor,
    ipa: torch.Tensor,
    sqi: torch.Tensor,
    *,
    alpha: float = 0.6,
    temperature: float = 0.1,
    use_sqi_loss: bool = True,
) -> MorphologyLossOutput:
    contrastive = supervised_nt_xent_loss(
        outputs["embedding"],
        svri_bin,
        temperature=temperature,
    )
    ipa_loss = F.l1_loss(outputs["ipa"].squeeze(-1), ipa.float())
    sqi_loss = F.l1_loss(outputs["sqi"].squeeze(-1), sqi.float())
    if use_sqi_loss:
        loss = alpha * contrastive + ((1.0 - alpha) / 2.0) * (ipa_loss + sqi_loss)
    else:
        loss = alpha * contrastive + (1.0 - alpha) * ipa_loss
    return MorphologyLossOutput(
        loss=loss,
        contrastive=contrastive.detach(),
        ipa=ipa_loss.detach(),
        sqi=sqi_loss.detach(),
    )
